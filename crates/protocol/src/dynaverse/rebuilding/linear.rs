//! Opt-in transition: correct frozen linear fields with sparse backend MSMs.
//! The live absolute-state/device-fusion design is deliberately separate.
use super::*;
use ark_ec::AffineRepr;
use super::program::{POINTS,PUBLIC_QUOTIENT};
use crate::kzg::gather_msm_batch;
use crate::dynaverse::sparse::CountedBackend;

pub(super) struct LinearState {
    pub points:[G1Projective;POINTS],
    pub s5_g2:G2Projective,
}

fn from_anchor(anchor:&AnchorProof)->LinearState {
    let mut p=[G1Projective::zero();POINTS];
    let w=&anchor.wires;
    for (i,v) in [w.s1,w.s2,w.s3,w.s4,w.s5,w.s6].into_iter().enumerate() {p[i]=v.into();}
    let d=&anchor.dynamo;let c=&d.commitments;
    for (i,v) in [c.alpha,c.beta,c.z,c.h,c.b,d.instance_z,d.instance_h].into_iter().enumerate() {
        p[6+i]=v.into();
    }
    for (t,s) in anchor.consistency.slices.iter().enumerate() {
        for (i,v) in [s.qac,s.qcd,s.qbd,s.d,s.a_degree,s.b_degree,s.d_degree].into_iter().enumerate() {
            p[13+7*t+i]=v.into();
        }
    }
    p[PUBLIC_QUOTIENT]=anchor.public_zero_quotient.into();
    LinearState {points:p,s5_g2:anchor.s5_g2.into()}
}

#[cfg(test)]
pub(super) fn correct(params:&DynaverseParams,anchor:&AnchorProof,
    frozen:&BTreeMap<usize,Fr>,backend:&dyn OperatorBackend)->Result<LinearState> {
    correct_traced(params,anchor,frozen,backend,&mut Default::default())
}

pub(super) fn correct_traced(params:&DynaverseParams,anchor:&AnchorProof,
    frozen:&BTreeMap<usize,Fr>,backend:&dyn OperatorBackend,
    trace:&mut crate::incremental_rebuild::LinearTimings)->Result<LinearState> {
    let started=trace.start();
    ensure!(anchor.dynamo.q.is_zero(),"linear correction requires the unmasked anchor mode");
    let mut result=from_anchor(anchor);
    if frozen.is_empty() {trace.finish(0,started);return Ok(result);}
    let n=params.circuit.n();
    let entries:Vec<(usize,Fr)>=frozen.iter().map(|(&i,&v)|(i,v)).collect();
    ensure!(entries.iter().all(|&(i,_)|i<6*n),"frozen linear index outside witness");
    let changed=SparseVector::try_new(params.circuit.m(),entries.clone())?;
    let h=derive_h_sparse(&changed,params.circuit.permutation_ref())?;
    let mut slices:[Vec<(usize,Fr)>;6]=std::array::from_fn(|_|Vec::new());
    crate::workcount::record_scan(entries.len());
    for &(i,v) in &entries {slices[i/n].push((i%n,v));}
    let bases=&params.srs.sparse;
    let requests:Vec<(&[G1Affine],&[(usize,Fr)])>=slices.iter()
        .map(|s|(bases.small_lagrange_g1.as_slice(),s.as_slice())).collect();
    let wires=gather_msm_batch(&requests,backend)?;
    for (i,p) in wires.into_iter().enumerate() {result.points[i]+=p;}
    let d=dynamo_prove_unmasked(&params.srs.dynamo,&changed,&h,&CountedBackend(backend))?;
    let c=d.commitments;
    for (i,p) in [c.alpha,c.beta,c.z,c.h,c.b,d.instance_z,d.instance_h].into_iter().enumerate() {
        result.points[6+i]+=p;
    }
    for (t,s) in slices.iter().enumerate() {
        let small=SparseVector::try_new(n,s.clone())?;
        let s=params.srs.consistency[t].prove_sparse(&changed,&small,backend)?;
        for (i,p) in [s.qac,s.qcd,s.qbd,s.d,s.a_degree,s.b_degree,s.d_degree].into_iter().enumerate() {
            result.points[13+7*t+i]+=p;
        }
    }
    result.points[PUBLIC_QUOTIENT]+=gather_msm_batch(&[
        (bases.public_quotient_g1.as_slice(),entries.as_slice())],backend)?[0];
    trace.finish(0,started);
    let started=trace.start();
    let mut points=Vec::with_capacity(slices[4].len());
    let mut scalars=Vec::with_capacity(slices[4].len());
    for &(i,v) in &slices[4] {points.push(bases.small_lagrange_g2[i]);scalars.push(v);}
    if !points.is_empty() {
        crate::workcount::record_g2_msm(points.len());
        result.s5_g2+=backend.msm_g2_host(&points,&scalars)?;
    }
    trace.finish(1,started);
    Ok(result)
}

/// Conservative protocol envelope, not calibrated nanoseconds. Sparse h has
/// at most 2b entries; the 56 G1 fields gather at most 27b point contributions.
/// Charge padding/per-call envelopes and support bookkeeping conservatively.
pub(super) fn work_bound(limit:usize)->Result<WorkBound> {
    let b=u64::try_from(limit).map_err(|_|anyhow::anyhow!("linear support bound overflow"))?;
    let g1=b.checked_mul(27).and_then(|v|v.checked_add(64*56))
        .ok_or_else(||anyhow::anyhow!("linear work overflow"))?;
    let g2=b.checked_add(64).ok_or_else(||anyhow::anyhow!("linear work overflow"))?;
    let log=u64::from(usize::BITS-limit.leading_zeros())+1;
    Ok(WorkBound {
        field_ops:b.checked_mul(4096*log).ok_or_else(||anyhow::anyhow!("linear field bound overflow"))?,
        g1_muls:g1.checked_mul(4).ok_or_else(||anyhow::anyhow!("linear G1 bound overflow"))?,
        g1_adds:g1.checked_mul(1024).ok_or_else(||anyhow::anyhow!("linear G1 bound overflow"))?,
        g2_muls:g2.checked_mul(4).ok_or_else(||anyhow::anyhow!("linear G2 bound overflow"))?,
        g2_adds:g2.checked_mul(1024).ok_or_else(||anyhow::anyhow!("linear G2 bound overflow"))?,
        copy_bytes:b.checked_mul(8192).and_then(|v|v.checked_add(32768))
            .ok_or_else(||anyhow::anyhow!("linear copy bound overflow"))?,
        host_elements:b.checked_mul(256*log).and_then(|v|v.checked_add(1024))
            .ok_or_else(||anyhow::anyhow!("linear host bound overflow"))?,
        launches:256,
    })
}
