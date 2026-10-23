//! Fixed public affine divisors only; never a witness acceptance cache.
use super::*;
use crate::incremental_rebuild::CacheDivisorStats;

struct RootDivisor { root:Fr, divisor:G2Affine }

pub(super) struct PublicDivisors {
    owner:usize,
    n:usize,
    powers:[G2Affine;2],
    roots:Vec<RootDivisor>,
    stats:CacheDivisorStats,
}
impl PublicDivisors {
    pub(super) fn host_bound(n:usize,enabled:bool,mode:VerificationMode)->Result<u64> {
        if !enabled {return Ok(0);}
        ensure!(n.is_power_of_two() && n<=65536,"divisor cache supports power-of-two n<=65536");
        ensure!(mode==VerificationMode::Product,"divisor cache requires Product mode");
        // 2048 bytes per temporary root conservatively covers projective and
        // affine arrays plus normalization scratch; at most64 roots coexist.
        let entries=(n as u64).checked_mul(std::mem::size_of::<RootDivisor>() as u64);
        entries.and_then(|v|v.checked_add(std::mem::size_of::<Self>() as u64))
            .and_then(|v|v.checked_add(n.min(64) as u64*2048+4096))
            .ok_or_else(||anyhow::anyhow!("divisor cache host envelope overflow"))
    }
    pub(super) fn new(params:&DynaverseParams,bound:u64)->Result<Self> {
        let n=params.circuit.n();
        ensure!(bound==Self::host_bound(n,true,VerificationMode::Product)?,
            "divisor cache lacks its checked host envelope");
        ensure!(params.srs.sparse.small_points.len()>=n,"divisor domain is incomplete");
        let started=Instant::now();
        let public=params.srs.kzg_x.rebuild_g2_powers();
        let powers=[public[0],public[1]];
        let mut roots=Vec::new();
        roots.try_reserve_exact(n)?;
        for start in (0..n).step_by(64) {
            let end=(start+64).min(n);
            let mut projective=Vec::with_capacity(end-start);
            for &root in &params.srs.sparse.small_points[start..end] {
                projective.push(G2Projective::from(powers[1])-G2Projective::from(powers[0])*root);
            }
            let affine=G2Projective::normalize_batch(&projective);
            for (j,&divisor) in affine.iter().enumerate() {
                roots.push(RootDivisor {root:params.srs.sparse.small_points[start+j],divisor});
            }
        }
        let retained_bytes=std::mem::size_of::<Self>() as u64
            +roots.capacity() as u64*std::mem::size_of::<RootDivisor>() as u64;
        ensure!(retained_bytes<=bound,"divisor retained allocation exceeds envelope");
        let stats=CacheDivisorStats {roots:n,construction_divisors:n,retained_bytes,
            host_upper_bytes:bound,precompute_ns:started.elapsed().as_nanos()};
        Ok(Self {owner:std::ptr::from_ref(params) as usize,n,powers,roots,stats})
    }
    pub(super) fn validate_owner(&self,params:&DynaverseParams)->Result<()> {
        let public=params.srs.kzg_x.rebuild_g2_powers();
        ensure!(self.owner==std::ptr::from_ref(params) as usize && self.n==params.circuit.n()
            && self.powers==[public[0],public[1]],"divisor cache owner/SRS/domain differs");
        Ok(())
    }
    pub(super) fn get(&self,index:usize,root:Fr)->Result<G2Affine> {
        let entry=self.roots.get(index).ok_or_else(||anyhow::anyhow!("divisor root index out of range"))?;
        ensure!(entry.root==root,"divisor root order differs");
        Ok(entry.divisor)
    }
    pub(super) fn stats(&self)->CacheDivisorStats {self.stats}
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng,SeedableRng};

    #[test]
    fn public_divisors_are_exact_and_bound_to_owner_srs_root() {
        let c=ToyCircuit::new(128).unwrap();
        let mut params=DynaverseParams::setup(c.clone(),&mut StdRng::seed_from_u64(27100)).unwrap();
        let foreign=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(27101)).unwrap();
        let bound=PublicDivisors::host_bound(128,true,VerificationMode::Product).unwrap();
        let table=PublicDivisors::new(&params,bound).unwrap();
        table.validate_owner(&params).unwrap();
        assert!(table.validate_owner(&foreign).is_err());
        assert!(table.validate_owner(&params.clone()).is_err());
        let powers=params.srs.kzg_x.rebuild_g2_powers();
        for i in 0..128 {
            let root=params.srs.sparse.small_points[i];
            let expected=(G2Projective::from(powers[1])-G2Projective::from(powers[0])*root).into_affine();
            assert_eq!(table.get(i,root).unwrap(),expected);
        }
        assert!(table.get(1,params.srs.sparse.small_points[2]).is_err());
        assert!(table.get(128,Fr::zero()).is_err());
        assert!(PublicDivisors::new(&params,bound-1).is_err());
        let stats=table.stats();
        assert_eq!(stats.roots,128);assert_eq!(stats.construction_divisors,128);
        assert!(stats.retained_bytes<=stats.host_upper_bytes);
        params.srs=foreign.srs.clone();
        assert!(table.validate_owner(&params).is_err());
    }

    #[test]
    fn divisor_host_bound_rejects_invalid_mode_and_domain() {
        assert_eq!(PublicDivisors::host_bound(131072,false,VerificationMode::Reference).unwrap(),0);
        for n in [0,3,65537,131072,usize::MAX] {
            assert!(PublicDivisors::host_bound(n,true,VerificationMode::Product).is_err());
        }
        assert!(PublicDivisors::host_bound(128,true,VerificationMode::Reference).is_err());
        assert!(PublicDivisors::host_bound(65536,true,VerificationMode::Product).unwrap()<32<<20);
    }

    #[test]
    fn divisor_table_admission_precedes_allocations() {
        use crate::incremental_rebuild::operators::CpuRebuildBackend;
        let params=DynaverseParams::setup(ToyCircuit::new(8).unwrap(),&mut StdRng::seed_from_u64(27102)).unwrap();
        let mut backend=CpuRebuildBackend::default();
        let before=backend.stats();
        let config=IncrementalConfig {rebuild_divisor_cache:true,..Default::default()};
        assert!(Program::new(&params,&config,&mut backend).is_err());
        assert_eq!(backend.stats(),before);
        let config=IncrementalConfig {rebuild_divisor_cache:true,rebuild_cache_mode:VerificationMode::Product,
            memory_limit_bytes:8*16384+(256u64<<20),..Default::default()};
        assert!(Program::new(&params,&config,&mut backend).is_err());
        assert_eq!(backend.stats(),before);
        let config=IncrementalConfig {rebuild_divisor_cache:true,rebuild_cache_mode:VerificationMode::Product,
            ..Default::default()};
        let program=Program::new(&params,&config,&mut backend).unwrap();
        let table=program.cache_divisors.as_ref().unwrap();
        assert_eq!(table.stats().roots,8);
        assert_eq!(program.resource.cache_divisor_host_bytes,table.stats().host_upper_bytes);
    }
}
