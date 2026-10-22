//! Independent global s5 commitment; never the local affected-set G2 field.
use super::*;

/// None means the bounded probe found more than limit s5 entries. The caller
/// then executes its original first dense tile. No partial result is published.
pub(super) fn correct(params:&DynaverseParams,source:&Frozen,limit:usize,
    backend:&dyn OperatorBackend)->Result<Option<(G2Projective,usize)>> {
    let anchor=G2Projective::from(source.anchor.proof.s5_g2);
    if source.wire_mask&(1<<4)==0 {return Ok(Some((anchor,0)));}
    let n=params.circuit.n();
    // Fixed capacity avoids geometric over-allocation beyond the host bound.
    let capacity=limit.min(n);
    let mut points=Vec::with_capacity(capacity);let mut scalars=Vec::with_capacity(capacity);
    // The BTree range avoids visiting unrelated wire deltas. The probe stops
    // after limit+1 visits and retains at most limit points/scalars.
    for (&index,&value) in source.delta.range(4*n..5*n) {
        crate::workcount::record_scan(1);
        if points.len()==limit {return Ok(None);}
        points.push(params.srs.sparse.small_lagrange_g2[index-4*n]);
        scalars.push(value);
    }
    let count=points.len();
    let correction=if count==0 {G2Projective::zero()} else {
        crate::workcount::record_g2_msm(count);
        G2Projective::from(backend.msm_g2_host(&points,&scalars)?)
    };
    Ok(Some((anchor+correction,count)))
}

/// Cover bounded probing (also on dense fallback), encoding/validation,
/// normalization, copy and native padded MSM; conservative, not calibrated time.
pub(super) fn work_bound(limit:usize)->Result<WorkBound> {
    let b=u64::try_from(limit).map_err(|_|anyhow::anyhow!("global G2 bound overflow"))?;
    let points=b.checked_mul(2).and_then(|v|v.checked_add(64))
        .ok_or_else(||anyhow::anyhow!("global G2 bound overflow"))?;
    Ok(WorkBound {field_ops:(b+1)*4096,g2_muls:points*4,g2_adds:points*1024,
        copy_bytes:(b+1)*4096+4096,host_elements:b*16+4096,launches:512,
        ..Default::default()})
}
