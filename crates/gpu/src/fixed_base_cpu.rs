//! Algorithm-matched CPU setup baseline. Tables contain public generator
//! multiples only; trapdoor scalars are borrowed for one call, never retained.
use crate::{ArkworksBackend, arkworks::dispatch_batch};
use anyhow::Result;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{scalar_mul::fixed_base::FixedBase, CurveGroup, Group};
use ark_ff::PrimeField;
use dynaverse_protocol::{BatchMsmRequest, OperatorBackend};
use std::cell::{Cell, RefCell};

/// Four-bit tables match the first CUDA implementation, independently using
/// arkworks 0.4 arithmetic. Construction and normalization count in caller wall.
#[derive(Default)]
pub struct FixedBaseCpuBackend {
    g1: RefCell<Option<Vec<Vec<G1Affine>>>>,
    g2: RefCell<Option<Vec<Vec<G2Affine>>>>,
    builds: Cell<(u64, u64)>,
}

impl FixedBaseCpuBackend {
    pub fn table_build_counts(&self) -> (u64, u64) { self.builds.get() }
}

impl OperatorBackend for FixedBaseCpuBackend {
    fn fixed_base_g1(&self, scalars: &[Fr]) -> Result<Vec<G1Affine>> {
        if scalars.is_empty() { return Ok(Vec::new()); }
        let mut table = self.g1.borrow_mut();
        if table.is_none() {
            *table = Some(FixedBase::get_window_table::<G1Projective>(
                Fr::MODULUS_BIT_SIZE as usize, 4, G1Projective::generator()));
            let (a, b) = self.builds.get(); self.builds.set((a+1, b));
        }
        let mut output = Vec::with_capacity(scalars.len());
        for chunk in scalars.chunks(4096) {
            let points = FixedBase::msm::<G1Projective>(Fr::MODULUS_BIT_SIZE as usize,
                4, table.as_ref().unwrap(), chunk);
            output.extend(G1Projective::normalize_batch(&points));
        }
        Ok(output)
    }

    fn fixed_base_g2(&self, scalars: &[Fr]) -> Result<Vec<G2Affine>> {
        if scalars.is_empty() { return Ok(Vec::new()); }
        let mut table = self.g2.borrow_mut();
        if table.is_none() {
            *table = Some(FixedBase::get_window_table::<G2Projective>(
                Fr::MODULUS_BIT_SIZE as usize, 4, G2Projective::generator()));
            let (a, b) = self.builds.get(); self.builds.set((a, b+1));
        }
        let mut output = Vec::with_capacity(scalars.len());
        for chunk in scalars.chunks(4096) {
            let points = FixedBase::msm::<G2Projective>(Fr::MODULUS_BIT_SIZE as usize,
                4, table.as_ref().unwrap(), chunk);
            output.extend(G2Projective::normalize_batch(&points));
        }
        Ok(output)
    }

    fn msm(&self, points: &[G1Affine], scalars: &[Fr]) -> Result<G1Affine> {
        ArkworksBackend.msm(points, scalars)
    }
    fn msm_batch(&self, request: &BatchMsmRequest<'_>) -> Result<Vec<G1Affine>> {
        dispatch_batch(self, request)
    }
    fn ntt(&self, values: &mut [Fr], inverse: bool) -> Result<()> {
        ArkworksBackend.ntt(values, inverse)
    }
}
