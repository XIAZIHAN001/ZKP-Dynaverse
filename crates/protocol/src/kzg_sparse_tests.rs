use super::*;
use crate::SparseMsmRequest;
use ark_ec::Group;
use std::cell::Cell;

struct IndexedOnly<'a> {
    bases: &'a [G1Affine], entries: &'a [(usize, Fr)],
    calls: Cell<usize>, fail: Cell<bool>, short: Cell<bool>,
}

impl OperatorBackend for IndexedOnly<'_> {
    fn msm(&self, _: &[G1Affine], _: &[Fr]) -> Result<G1Affine> {
        anyhow::bail!("indexed path must not dispatch a gathered single MSM")
    }
    fn msm_batch(&self, _: &BatchMsmRequest<'_>) -> Result<Vec<G1Affine>> {
        anyhow::bail!("indexed path must not dispatch a gathered dense batch")
    }
    fn msm_sparse_batch(&self, requests: &[SparseMsmRequest<'_>]) -> Result<Vec<G1Affine>> {
        self.calls.set(self.calls.get() + 1);
        ensure!(!self.fail.get(), "indexed dispatch failure");
        let mut result = Vec::new();
        for request in requests {
            assert!(std::ptr::eq(request.bases, self.bases));
            assert!(std::ptr::eq(request.entries, self.entries));
            let mut sum = G1Projective::zero();
            for &(i, scalar) in request.entries { sum += request.bases[i] * scalar; }
            result.push(sum.into_affine());
        }
        if self.short.get() { result.pop(); }
        Ok(result)
    }
    fn ntt(&self, _: &mut [Fr], _: bool) -> Result<()> { anyhow::bail!("unexpected NTT") }
}

#[test]
fn delta_gather_dispatches_original_borrows_and_preserves_work_counts() {
    let bases = vec![G1Projective::generator().into_affine(); 8];
    let entries = [(7, Fr::from(3u64)), (0, -Fr::from(1u64)), (7, Fr::from(2u64))];
    let backend = IndexedOnly { bases: &bases, entries: &entries,
        calls: Cell::new(0), fail: Cell::new(false), short: Cell::new(false) };
    let (result, work) = crate::workcount::measure(||
        gather_msm_batch(&[(&bases, &entries), (&bases, &entries)], &backend));
    assert_eq!(result.unwrap(), vec![(G1Projective::generator() * Fr::from(4u64)).into_affine(); 2]);
    assert_eq!(backend.calls.get(), 1);
    assert_eq!(work.msm_points, 6);
    assert_eq!(work.scan_elements, 6);
    backend.short.set(true);
    assert!(gather_msm_batch(&[(&bases, &entries)], &backend).is_err());
    backend.short.set(false);
    backend.fail.set(true);
    assert!(gather_msm_batch(&[(&bases, &entries)], &backend).unwrap_err().to_string()
        .contains("indexed dispatch failure"));
    let before = backend.calls.get();
    let bad = [(8, Fr::zero())];
    assert!(gather_msm_batch(&[(&bases, &entries), (&bases, &bad)], &backend).is_err());
    assert_eq!(backend.calls.get(), before);
}
