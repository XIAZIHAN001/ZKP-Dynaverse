//! Fixed-SRS linear fields only: no local polynomial or affected-set outputs.
use super::*;
use ark_ec::AffineRepr;
use crate::kzg::gather_msm_batch;
use super::sparse::CountedBackend;

pub(super) struct LinearFields {
    pub wires:WireCommitments,
    pub dynamo:crate::DynamoProof,
    pub consistency:ConsistencyProof,
    pub public_zero_quotient:G1Affine,
}

/// Constant-sized absolute T(W). Projective points avoid 56 inversions at
/// every addition; one batch normalization materializes public proof fields.
#[derive(Clone)]
pub(super) struct FixedLinear {points:[G1Projective;56]}

impl FixedLinear {
    pub fn points(&self)->&[G1Projective;56] {&self.points}
    pub fn from_anchor(anchor:&AnchorProof)->Result<Self> {
        ensure!(anchor.dynamo.q.is_zero(),"live linear state requires unmasked mode");
        Ok(Self::from_fields(&anchor.wires,&anchor.dynamo,&anchor.consistency,anchor.public_zero_quotient))
    }

    fn from_fields(w:&WireCommitments,d:&crate::DynamoProof,c:&ConsistencyProof,q:G1Affine)->Self {
        let mut p=[G1Projective::zero();56];
        for (i,v) in [w.s1,w.s2,w.s3,w.s4,w.s5,w.s6].into_iter().enumerate() {p[i]=v.into();}
        let dc=&d.commitments;
        for (i,v) in [dc.alpha,dc.beta,dc.z,dc.h,dc.b,d.instance_z,d.instance_h].into_iter().enumerate() {p[6+i]=v.into();}
        for (wire,s) in c.slices.iter().enumerate() {
            for (i,v) in [s.qac,s.qcd,s.qbd,s.d,s.a_degree,s.b_degree,s.d_degree].into_iter().enumerate() {p[13+7*wire+i]=v.into();}
        }
        p[55]=q.into();Self {points:p}
    }

    pub fn add(&self,other:&Self)->Self {Self {points:std::array::from_fn(|i|self.points[i]+other.points[i])}}
    pub fn relative_to(&self,anchor:&AnchorProof)->Result<Self> {
        let old=Self::from_anchor(anchor)?;
        Ok(Self {points:std::array::from_fn(|i|self.points[i]-old.points[i])})
    }

    pub fn fields(&self)->LinearFields {
        let p=G1Projective::normalize_batch(&self.points);
        let wires=WireCommitments {s1:p[0],s2:p[1],s3:p[2],s4:p[3],s5:p[4],s6:p[5]};
        let dynamo=crate::DynamoProof {commitments:crate::DynamoCommitments {
            alpha:p[6],beta:p[7],z:p[8],h:p[9],b:p[10]},q:G1Affine::identity(),instance_z:p[11],instance_h:p[12]};
        let slices=std::array::from_fn(|wire| {
            let i=13+7*wire;crate::SliceConsistencyProof {qac:p[i],qcd:p[i+1],qbd:p[i+2],d:p[i+3],
                a_degree:p[i+4],b_degree:p[i+5],d_degree:p[i+6]}
        });
        LinearFields {wires,dynamo,consistency:ConsistencyProof {slices},public_zero_quotient:p[55]}
    }

    pub fn compute(params:&DynaverseParams,entries:&[(usize,Fr)],backend:&dyn OperatorBackend)->Result<Self> {
        if entries.is_empty() {return Ok(Self {points:[G1Projective::zero();56]});}
        let n=params.circuit.n();
        ensure!(entries.iter().all(|&(i,_)|i<6*n),"live linear index outside witness");
        let changed=SparseVector::try_new(params.circuit.m(),entries.to_vec())?;
        let h=derive_h_sparse(&changed,params.circuit.permutation_ref())?;
        let dynamo=dynamo_prove_unmasked(&params.srs.dynamo,&changed,&h,&CountedBackend(backend))?;
        let mut slices:[Vec<(usize,Fr)>;6]=std::array::from_fn(|_|Vec::new());
        crate::workcount::record_scan(entries.len());
        for &(i,v) in entries {slices[i/n].push((i%n,v));}
        let requests:Vec<(&[G1Affine],&[(usize,Fr)])>=slices.iter()
            .map(|slice|(params.srs.sparse.small_lagrange_g1.as_slice(),slice.as_slice())).collect();
        let w=gather_msm_batch(&requests,backend)?;
        let wires=WireCommitments {s1:w[0],s2:w[1],s3:w[2],s4:w[3],s5:w[4],s6:w[5]};
        let mut consistency=Vec::with_capacity(6);
        for (i,slice) in slices.iter().enumerate() {
            let small=SparseVector::try_new(n,slice.clone())?;
            consistency.push(params.srs.consistency[i].prove_sparse(&changed,&small,backend)?);
        }
        let consistency=ConsistencyProof {slices:consistency.try_into().map_err(|_|anyhow::anyhow!("six linear consistency slices required"))?};
        let public=gather_msm_batch(&[(&params.srs.sparse.public_quotient_g1,entries)],backend)?[0];
        Ok(Self::from_fields(&wires,&dynamo,&consistency,public))
    }
}
