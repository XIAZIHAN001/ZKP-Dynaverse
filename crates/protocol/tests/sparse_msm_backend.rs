use anyhow::{ensure, Result};
use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::{CurveGroup, Group, VariableBaseMSM};
use dynaverse_protocol::{BatchMsmRequest, OperatorBackend, SparseMsmRequest};
use std::cell::{Cell, RefCell};

// This boundary records the actual batch layout and computes real group sums.
// It models unavailable/short native output without replacing the arithmetic.
#[derive(Default)]
struct RecordingBackend {
    calls: Cell<usize>,
    fail: Cell<bool>,
    short: Cell<bool>,
    scalar_count: Cell<usize>,
    offsets: RefCell<Vec<usize>>,
}

impl OperatorBackend for RecordingBackend {
    fn msm(&self, points: &[G1Affine], scalars: &[Fr]) -> Result<G1Affine> {
        ensure!(points.len() == scalars.len(), "MSM length mismatch");
        Ok(G1Projective::msm_unchecked(points, scalars).into_affine())
    }

    fn msm_batch(&self, request: &BatchMsmRequest<'_>) -> Result<Vec<G1Affine>> {
        self.calls.set(self.calls.get() + 1);
        ensure!(!self.fail.get(), "injected batch failure");
        self.scalar_count.set(request.shared_scalars.len());
        *self.offsets.borrow_mut() = request.offsets.to_vec();
        let mut result = Vec::new();
        for i in 0..request.point_sets.len() {
            let offset = request.offsets[i];
            let len = request.lengths[i];
            result.push(self.msm(&request.point_sets[i][..len],
                &request.shared_scalars[offset..offset + len])?);
        }
        if self.short.get() { result.pop(); }
        Ok(result)
    }

    fn ntt(&self, _: &mut [Fr], _: bool) -> Result<()> {
        anyhow::bail!("unexpected NTT")
    }
}

fn bases() -> Vec<G1Affine> {
    let g = G1Projective::generator();
    (1..=4).map(|i| (g * Fr::from(i as u64)).into_affine()).collect()
}

#[test]
fn indexed_fallback_preserves_order_duplicates_zero_and_full_width_scalars() {
    let backend = RecordingBackend::default();
    let points = bases();
    let opposite: Vec<_> = points.iter().map(|point| -*point).collect();
    let shared = [(3, Fr::from(7u64)), (1, Fr::from(0u64)),
        (0, -Fr::from(5u64)), (3, Fr::from(2u64))];
    let independent = [(3, Fr::from(7u64)), (1, Fr::from(0u64)),
        (0, -Fr::from(6u64)), (3, Fr::from(2u64))];
    let requests = [
        SparseMsmRequest { bases: &points, entries: &shared },
        SparseMsmRequest { bases: &opposite, entries: &shared },
        SparseMsmRequest { bases: &points, entries: &independent },
        SparseMsmRequest { bases: &[], entries: &[] },
    ];
    let g = G1Projective::generator();
    assert_eq!(backend.msm_sparse_batch(&requests).unwrap(), vec![
        (g * Fr::from(31u64)).into_affine(),
        (g * -Fr::from(31u64)).into_affine(),
        (g * Fr::from(30u64)).into_affine(), G1Affine::identity(),
    ]);
    // Sharing is tied to the same borrowed entry slice, never equal lengths.
    assert_eq!(backend.scalar_count.get(), 8);
    assert_eq!(&backend.offsets.borrow()[..3], &[0, 0, 4]);
}

#[test]
fn indexed_fallback_checks_late_invalid_index_even_for_zero_before_dispatch() {
    let backend = RecordingBackend::default();
    let points = bases();
    let good = [(0, Fr::from(3u64))];
    for bad_index in [points.len(), usize::MAX] {
        let bad = [(bad_index, Fr::from(0u64))];
        let requests = [SparseMsmRequest { bases: &points, entries: &good },
            SparseMsmRequest { bases: &points, entries: &bad }];
        assert!(backend.msm_sparse_batch(&requests).is_err());
        assert_eq!(backend.calls.get(), 0);
    }
}

#[test]
fn indexed_fallback_propagates_error_and_rejects_short_results() {
    let backend = RecordingBackend::default();
    let points = bases();
    let entries = [(2, Fr::from(9u64))];
    let requests = [SparseMsmRequest { bases: &points, entries: &entries }];
    backend.fail.set(true);
    assert!(backend.msm_sparse_batch(&requests).unwrap_err().to_string()
        .contains("injected batch failure"));
    assert_eq!(backend.calls.get(), 1);
    backend.fail.set(false);
    backend.short.set(true);
    assert!(backend.msm_sparse_batch(&requests).is_err());
    assert_eq!(backend.calls.get(), 2);
    backend.short.set(false);
    assert_eq!(backend.msm_sparse_batch(&requests).unwrap(),
        vec![(G1Projective::generator() * Fr::from(27u64)).into_affine()]);
}

#[test]
fn indexed_empty_batch_avoids_dispatch_and_empty_rows_return_identity() {
    let backend = RecordingBackend::default();
    assert!(backend.msm_sparse_batch(&[]).unwrap().is_empty());
    assert_eq!(backend.calls.get(), 0);
    let requests = [SparseMsmRequest { bases: &[], entries: &[] }; 2];
    assert_eq!(backend.msm_sparse_batch(&requests).unwrap(), vec![G1Affine::identity(); 2]);
}
