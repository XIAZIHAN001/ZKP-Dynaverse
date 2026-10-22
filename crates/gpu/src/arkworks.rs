//! CPU adapter for the same arkworks MSM used by the original KZG path.
//! The deliberately simple CpuBackend remains the independent test oracle.
use crate::CpuBackend;
use anyhow::{ensure, Result};
use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::{CurveConfig, CurveGroup, VariableBaseMSM};
use dynaverse_protocol::{BatchMsmRequest, OperatorBackend};

// The on-curve check is sufficient only for this concrete cofactor-one G1.
// Keep the premise explicit even in builds without the ICICLE feature.
const _: () = {
    assert!(ark_bn254::g1::Config::COFACTOR.len() == 1);
    assert!(ark_bn254::g1::Config::COFACTOR[0] == 1);
};

pub struct ArkworksBackend;

impl OperatorBackend for ArkworksBackend {
    fn msm(&self, points: &[G1Affine], scalars: &[Fr]) -> Result<G1Affine> {
        ensure!(points.len() == scalars.len(), "MSM length mismatch");
        for point in points { ensure!(point.is_on_curve(), "invalid G1 point"); }
        // Equal lengths were checked above; this is the original KZG library
        // call, with boundary validation rather than a new CPU algorithm.
        Ok(G1Projective::msm_unchecked(points, scalars).into_affine())
    }

    fn msm_batch(&self, request: &BatchMsmRequest<'_>) -> Result<Vec<G1Affine>> {
        dispatch_batch(self, request)
    }

    fn ntt(&self, values: &mut [Fr], inverse: bool) -> Result<()> {
        CpuBackend.ntt(values, inverse)
    }
}

// Both new adapters use the same checked batch slices. The trait object is
// borrowed only for these synchronous calls and is never stored in a result.
pub(crate) fn dispatch_batch(backend: &dyn OperatorBackend, request: &BatchMsmRequest<'_>) -> Result<Vec<G1Affine>> {
    let count = request.point_sets.len();
    ensure!(request.offsets.len() == count && request.lengths.len() == count, "batch shape mismatch");
    for i in 0..count {
        let end = request.offsets[i].checked_add(request.lengths[i])
            .ok_or_else(|| anyhow::anyhow!("scalar range overflow"))?;
        ensure!(end <= request.shared_scalars.len(), "scalar range out of bounds");
        ensure!(request.lengths[i] <= request.point_sets[i].len(), "point range out of bounds");
    }
    let mut results = Vec::with_capacity(count);
    for i in 0..count {
        let offset = request.offsets[i];
        let length = request.lengths[i];
        results.push(backend.msm(&request.point_sets[i][..length], &request.shared_scalars[offset..offset + length])?);
    }
    Ok(results)
}
