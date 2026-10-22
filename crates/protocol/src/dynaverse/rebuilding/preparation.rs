//! Fixed public data only. This table never certifies a witness or opening.
use super::*;
use crate::incremental_rebuild::CachePreparationStats;

type PreparedG2=<Bn254 as Pairing>::G2Prepared;
const ENTRY_ENVELOPE:u64=65_536;

struct RootPreparation {
    root:Fr,
    // Retained as the public affine divisor for provenance and future device
    // upload. CPU Miller loops consume only the matching prepared value.
    divisor:G2Affine,
    prepared:PreparedG2,
}

pub(super) struct PublicPreparation {
    owner:usize,
    n:usize,
    powers:[G2Affine;2],
    generator:PreparedG2,
    roots:Vec<RootPreparation>,
    stats:CachePreparationStats,
}

fn coefficient_bytes(prepared:&PreparedG2)->u64 {
    // The pinned BN254 preparation owns a Vec of three-Fq2 line coefficients.
    // Capacity measures requested heap storage, not just initialized length.
    (prepared.ell_coeffs.capacity()*std::mem::size_of::<(ark_bn254::Fq2,ark_bn254::Fq2,ark_bn254::Fq2)>()) as u64
}

impl PublicPreparation {
    pub(super) fn host_bound(n:usize,cap:Option<usize>,mode:VerificationMode)->Result<u64> {
        let Some(cap)=cap else {return Ok(0)};
        ensure!((1..=4096).contains(&cap),"prepared root cap must be in 1..=4096");
        ensure!(mode==VerificationMode::Product,"prepared root cache requires Product mode");
        // Seven temporary preparations cover the fallback checker, consumed
        // pair clones and construction, plus the retained generator. Also
        // reserve normalization/construction arrays and table/vector overhead.
        let count=n.min(cap) as u64;
        Ok((count+8)*ENTRY_ENVELOPE+count*512+4096)
    }

    pub(super) fn new(params:&DynaverseParams,cap:usize,host_upper_bytes:u64)->Result<Self> {
        let started=Instant::now();let count=params.circuit.n().min(cap);
        ensure!(host_upper_bytes==Self::host_bound(params.circuit.n(),Some(cap),VerificationMode::Product)?,
            "prepared table allocation lacks the checked resource envelope");
        let public=params.srs.kzg_x.rebuild_g2_powers();let powers=[public[0],public[1]];
        let generator:PreparedG2=powers[0].into();
        let mut projective=Vec::with_capacity(count);
        for &root in &params.srs.sparse.small_points[..count] {
            projective.push(G2Projective::from(powers[1])-G2Projective::from(powers[0])*root);
        }
        let affine=G2Projective::normalize_batch(&projective);
        let mut roots=Vec::with_capacity(count);
        for (i,&divisor) in affine.iter().enumerate() {
            let prepared:PreparedG2=divisor.into();
            ensure!(std::mem::size_of::<RootPreparation>() as u64+coefficient_bytes(&prepared)<=ENTRY_ENVELOPE,
                "prepared root allocation exceeds per-entry envelope");
            roots.push(RootPreparation {root:params.srs.sparse.small_points[i],divisor,prepared});
        }
        ensure!(std::mem::size_of::<PreparedG2>() as u64+coefficient_bytes(&generator)<=ENTRY_ENVELOPE,
            "prepared generator allocation exceeds per-entry envelope");
        let retained_bytes=std::mem::size_of::<Self>() as u64
            +(roots.capacity()*std::mem::size_of::<RootPreparation>()) as u64
            +coefficient_bytes(&generator)+roots.iter().map(|r|coefficient_bytes(&r.prepared)).sum::<u64>();
        ensure!(retained_bytes<=host_upper_bytes,"prepared retained memory exceeds envelope");
        drop(affine);drop(projective);
        let stats=CachePreparationStats {roots:count,prepared_entries:count+1,retained_bytes,
            host_upper_bytes,precompute_ns:started.elapsed().as_nanos()};
        Ok(Self {owner:std::ptr::from_ref(params) as usize,n:params.circuit.n(),powers,generator,roots,stats})
    }

    pub(super) fn validate_owner(&self,params:&DynaverseParams)->Result<()> {
        let powers=params.srs.kzg_x.rebuild_g2_powers();
        ensure!(self.owner==std::ptr::from_ref(params) as usize && self.n==params.circuit.n()
            && self.powers==[powers[0],powers[1]],"prepared cache owner/SRS/domain differs");
        Ok(())
    }

    pub(super) fn contains(&self,index:usize)->bool {index<self.roots.len()}

    pub(super) fn pair(&self,index:usize,root:Fr)->Result<(G2Affine,&PreparedG2,&PreparedG2)> {
        let entry=self.roots.get(index).ok_or_else(||anyhow::anyhow!("prepared root outside resident prefix"))?;
        ensure!(entry.root==root,"prepared cache root order differs");
        Ok((entry.divisor,&self.generator,&entry.prepared))
    }

    pub(super) fn stats(&self)->CachePreparationStats {self.stats}
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng,SeedableRng};

    #[test]
    fn public_preparation_rejects_foreign_owner_srs_and_root_order() {
        let c=ToyCircuit::new(8).unwrap();
        let mut p=DynaverseParams::setup(c.clone(),&mut StdRng::seed_from_u64(25010)).unwrap();
        let other=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(25011)).unwrap();
        let bound=PublicPreparation::host_bound(8,Some(3),VerificationMode::Product).unwrap();
        let table=PublicPreparation::new(&p,3,bound).unwrap();
        table.validate_owner(&p).unwrap();assert!(table.validate_owner(&other).is_err());
        let cloned=p.clone();assert!(table.validate_owner(&cloned).is_err());
        let root=p.srs.sparse.small_points[1];
        let (divisor,_,_)=table.pair(1,root).unwrap();
        let powers=p.srs.kzg_x.rebuild_g2_powers();
        assert_eq!(divisor,(G2Projective::from(powers[1])-G2Projective::from(powers[0])*root).into_affine());
        assert!(table.pair(1,p.srs.sparse.small_points[2]).is_err());
        assert!(!table.contains(3));assert!(table.pair(3,root).is_err());
        // Even an internal test mutation cannot reuse preparations with a
        // different public SRS under the original parameter address.
        p.srs=other.srs.clone();assert!(table.validate_owner(&p).is_err());
        let stats=table.stats();assert_eq!(stats.prepared_entries,4);
        assert!(stats.retained_bytes>3*128 && stats.retained_bytes<bound);
    }

    #[test]
    fn preparation_resource_preflight_precedes_backend_allocation() {
        use crate::incremental_rebuild::operators::CpuRebuildBackend;
        let p=DynaverseParams::setup(ToyCircuit::new(8).unwrap(),&mut StdRng::seed_from_u64(25012)).unwrap();
        let mut backend=CpuRebuildBackend::default();let before=backend.stats();
        for cap in [0,4097,usize::MAX] {
            let cfg=IncrementalConfig {rebuild_prepared_cache_roots:Some(cap),rebuild_cache_mode:VerificationMode::Product,..Default::default()};
            assert!(Program::new(&p,&cfg,&mut backend).is_err());assert_eq!(before,backend.stats());
        }
        let cfg=IncrementalConfig {rebuild_prepared_cache_roots:Some(8),..Default::default()};
        assert!(Program::new(&p,&cfg,&mut backend).is_err());assert_eq!(before,backend.stats());
        let base=8*16_384+(256u64<<20);
        let cfg=IncrementalConfig {rebuild_prepared_cache_roots:Some(8),rebuild_cache_mode:VerificationMode::Product,
            memory_limit_bytes:base,..Default::default()};
        let error=Program::new(&p,&cfg,&mut backend).err().unwrap();
        assert!(error.to_string().contains("resource estimate"));assert_eq!(before,backend.stats());
    }
}
