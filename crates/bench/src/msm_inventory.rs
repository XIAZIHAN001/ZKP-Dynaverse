//! Bench-only observer, used in a separate diagnostic CPU sequence.
use anyhow::{ensure, Result};
use ark_bn254::{Fr, G1Affine};
use ark_ff::{BigInteger, PrimeField, Zero};
use dynaverse_protocol::{BatchMsmRequest, OperatorBackend};
use std::cell::{Cell, RefCell};
use std::path::Path;

pub fn reject_output_alias(output: &Path, inventory: &Path) -> Result<()> {
    ensure!(output != inventory, "diagnostic output paths must refer to different files");
    if let (Ok(original), Ok(other)) = (std::fs::canonicalize(output), std::fs::canonicalize(inventory)) {
        ensure!(original != other, "diagnostic output paths must refer to different files");
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if let (Ok(original), Ok(other)) = (std::fs::metadata(output), std::fs::metadata(inventory)) {
            ensure!(original.dev() != other.dev() || original.ino() != other.ino(),
                "diagnostic output paths must refer to different files");
        }
    }
    Ok(())
}

pub struct MsmCall {
    pub operation: &'static str,
    pub step: usize,
    pub call: usize,
    pub length: usize,
    pub nonzero_scalars: usize,
    pub max_scalar_bits: u32,
}

/// Bench-only notification for a nested diagnostic observer. Both observers
/// begin the same operation before any MSM, so NVTX and CSV identities agree.
pub trait MsmContext {
    fn begin(&self, operation: &'static str, step: usize);
}

// The observer borrows its backend for one sequence. Cell/RefCell permit
// recording through OperatorBackend's shared &self; it stays on one thread.
pub struct InventoryBackend<'a> {
    inner: &'a dyn OperatorBackend,
    operation: Cell<&'static str>,
    step: Cell<usize>,
    next_call: Cell<usize>,
    calls: RefCell<Vec<MsmCall>>,
    context: Option<&'a dyn MsmContext>,
}

impl<'a> InventoryBackend<'a> {
    pub fn new(inner: &'a dyn OperatorBackend) -> Self {
        Self { inner, operation: Cell::new("prove"), step: Cell::new(0),
            next_call: Cell::new(0), calls: RefCell::new(Vec::new()), context: None }
    }

    #[cfg(feature = "icicle")]
    pub fn with_context(inner: &'a dyn OperatorBackend, context: &'a dyn MsmContext) -> Self {
        let mut observer = Self::new(inner);
        observer.context = Some(context);
        observer
    }

    pub fn begin(&self, operation: &'static str, step: usize) {
        self.operation.set(operation);
        self.step.set(step);
        self.next_call.set(0);
        if let Some(context) = self.context { context.begin(operation, step); }
    }

    pub fn take_calls(&self) -> Vec<MsmCall> { self.calls.take() }

    fn record(&self, scalars: &[Fr]) {
        let mut nonzero_scalars = 0;
        let mut max_scalar_bits = 0;
        for scalar in scalars {
            if !scalar.is_zero() { nonzero_scalars += 1; }
            max_scalar_bits = max_scalar_bits.max(scalar.into_bigint().num_bits());
        }
        let call = self.next_call.get();
        self.calls.borrow_mut().push(MsmCall { operation: self.operation.get(), step: self.step.get(),
            call, length: scalars.len(), nonzero_scalars, max_scalar_bits });
        self.next_call.set(call + 1);
    }
}

impl OperatorBackend for InventoryBackend<'_> {
    fn msm(&self, points: &[G1Affine], scalars: &[Fr]) -> Result<G1Affine> {
        let result = self.inner.msm(points, scalars)?;
        self.record(scalars);
        Ok(result)
    }

    fn msm_batch(&self, request: &BatchMsmRequest<'_>) -> Result<Vec<G1Affine>> {
        let results = self.inner.msm_batch(request)?;
        // The successful inner adapter validated the complete request before
        // dispatch. Record only the exact scalar slices consumed by each set.
        for i in 0..request.point_sets.len() {
            let offset = request.offsets[i];
            self.record(&request.shared_scalars[offset..offset + request.lengths[i]]);
        }
        Ok(results)
    }

    fn ntt(&self, values: &mut [Fr], inverse: bool) -> Result<()> { self.inner.ntt(values, inverse) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::{Fr, G1Affine, G1Projective};
    use ark_ec::{CurveGroup, Group};
    use dynaverse_gpu::ArkworksBackend;
    use dynaverse_protocol::{BatchMsmRequest, OperatorBackend};

    #[test]
    fn inventory_records_exact_scalar_statistics_and_operation_indices() {
        let backend = InventoryBackend::new(&ArkworksBackend);
        let generator = G1Projective::generator();
        let points = [generator.into_affine(), generator.into_affine(), G1Affine::identity()];
        let scalars = [Fr::from(0u64), Fr::from(2u64), Fr::from(3u64)];
        backend.begin("prove", 0);
        assert_eq!(backend.msm(&points, &scalars).unwrap(), (generator * Fr::from(2u64)).into_affine());
        assert!(backend.msm(&points, &scalars[..2]).is_err());
        backend.begin("update", 1);
        let sets: [&[G1Affine]; 2] = [&points, &points];
        let request = BatchMsmRequest { point_sets: &sets, shared_scalars: &scalars, offsets: &[1, 0], lengths: &[2, 1] };
        assert_eq!(backend.msm_batch(&request).unwrap(), vec![(generator * Fr::from(5u64)).into_affine(), G1Affine::identity()]);
        let calls = backend.take_calls();
        assert_eq!(calls.len(), 3);
        assert_eq!((calls[0].operation, calls[0].step, calls[0].call, calls[0].length, calls[0].nonzero_scalars, calls[0].max_scalar_bits),
            ("prove", 0, 0, 3, 2, 2));
        assert_eq!((calls[1].operation, calls[1].step, calls[1].call, calls[1].length, calls[1].nonzero_scalars, calls[1].max_scalar_bits),
            ("update", 1, 0, 2, 2, 2));
        assert_eq!((calls[2].operation, calls[2].step, calls[2].call, calls[2].length, calls[2].nonzero_scalars, calls[2].max_scalar_bits),
            ("update", 1, 1, 1, 0, 0));
        assert!(backend.take_calls().is_empty());
    }
}
