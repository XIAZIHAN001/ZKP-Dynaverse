#![cfg(feature = "icicle")]

use ark_bn254::{Fq, Fr, G1Affine, G1Projective};
use ark_ec::{CurveGroup, Group};
use ark_ff::{UniformRand, Zero};
use dynaverse_gpu::{HybridBackend, IcicleBackend, MAX_MSM_SIZE};
use dynaverse_protocol::{OperatorBackend, SparseMsmRequest};
use rand::{rngs::StdRng, SeedableRng};

// Independent projective multiply-and-add, without batch packing or GPU code.
fn reference(bases: &[G1Affine], entries: &[(usize, Fr)]) -> G1Affine {
    let mut sum = G1Projective::zero();
    for &(index, value) in entries { sum += bases[index] * value; }
    sum.into_affine()
}

#[test]
fn indexed_gpu_matches_full_width_reference_and_reuses_dense_batch_buffers() {
    let owner = IcicleBackend::new().unwrap();
    let mut rng = StdRng::seed_from_u64(2026102201);
    let bases: Vec<_> = (0..97).map(|_| G1Projective::rand(&mut rng).into_affine()).collect();
    let opposite: Vec<_> = bases.iter().map(|point| -*point).collect();
    let entries: Vec<_> = (0..53).map(|i| ((37 * i + 11) % 97, Fr::rand(&mut rng))).collect();
    let mut different = entries.clone();
    different[0].1 += Fr::from(7u64);
    let repeated = [(96, Fr::from(9u64)), (0, Fr::zero()), (96, -Fr::from(2u64))];
    let requests = [
        SparseMsmRequest { bases: &bases, entries: &entries },
        SparseMsmRequest { bases: &opposite, entries: &entries },
        SparseMsmRequest { bases: &bases, entries: &different },
        SparseMsmRequest { bases: &[], entries: &[] },
        SparseMsmRequest { bases: &bases, entries: &repeated },
    ];
    let expected: Vec<_> = requests.iter().map(|r| reference(r.bases, r.entries)).collect();
    assert_eq!(owner.msm_sparse_batch(&requests).unwrap(), expected);
    let first = owner.batch_stats().unwrap();
    assert_eq!(first.max_batch_size, 3);
    assert_eq!(first.icicle_calls, 2);
    for _ in 0..3 { assert_eq!(owner.msm_sparse_batch(&requests).unwrap(), expected); }
    assert_eq!(owner.batch_stats().unwrap().buffer_grows, first.buffer_grows);
    let zeros = vec![(1, Fr::zero()); 53];
    let zero_requests = [SparseMsmRequest { bases: &bases, entries: &zeros }; 3];
    assert_eq!(owner.msm_sparse_batch(&zero_requests).unwrap(), vec![G1Affine::identity(); 3]);
    assert_eq!(owner.batch_stats().unwrap().buffer_grows, first.buffer_grows);
}

#[test]
fn indexed_gpu_rejects_late_bad_inputs_before_any_native_batch() {
    let owner = IcicleBackend::new().unwrap();
    let g = G1Projective::generator().into_affine();
    let invalid = G1Affine::new_unchecked(Fq::from(1u64), Fq::from(1u64));
    let bases = [g, invalid];
    let good = [(0, Fr::from(3u64))];
    let before = owner.batch_stats().unwrap();
    for bad_index in [1, 2, usize::MAX] {
        // Even a zero scalar cannot hide an invalid selected point or index.
        let bad = [(bad_index, Fr::zero())];
        let requests = [SparseMsmRequest { bases: &bases, entries: &good },
            SparseMsmRequest { bases: &bases, entries: &bad }];
        assert!(owner.msm_sparse_batch(&requests).is_err());
        assert_eq!(owner.batch_stats().unwrap(), before);
    }
    // Unselected entries are outside the input to the MSM, as in old gather.
    assert_eq!(owner.msm_sparse_batch(&[SparseMsmRequest { bases: &bases, entries: &good }]).unwrap(),
        vec![(G1Projective::generator() * Fr::from(3u64)).into_affine()]);
}

#[test]
fn indexed_gpu_splits_large_request_counts_and_checks_length_before_dispatch() {
    let owner = IcicleBackend::new().unwrap();
    let bases = [G1Projective::generator().into_affine()];
    let entries = [(0, Fr::from(5u64)); 17];
    let requests = vec![SparseMsmRequest { bases: &bases, entries: &entries }; 70];
    let expected = (G1Projective::generator() * Fr::from(85u64)).into_affine();
    assert_eq!(owner.msm_sparse_batch(&requests).unwrap(), vec![expected; 70]);
    let before = owner.batch_stats().unwrap();
    assert_eq!(before.max_batch_size, 64);
    assert_eq!(before.icicle_calls, 2);
    let oversized = vec![(0, Fr::zero()); MAX_MSM_SIZE + 1];
    assert!(owner.msm_sparse_batch(&[SparseMsmRequest { bases: &bases, entries: &oversized }]).is_err());
    assert_eq!(owner.batch_stats().unwrap(), before);
    assert!(owner.msm_sparse_batch(&[]).unwrap().is_empty());
    assert_eq!(owner.msm_sparse_batch(&[SparseMsmRequest { bases: &[], entries: &[] }]).unwrap(),
        vec![G1Affine::identity()]);
    assert_eq!(owner.batch_stats().unwrap(), before);
}

#[test]
fn indexed_hybrid_preserves_cpu_selection_and_batches_selected_gpu_requests() {
    let owner = IcicleBackend::new().unwrap();
    let hybrid = HybridBackend::new(&owner, 8).unwrap();
    let bases = [G1Projective::generator().into_affine(); 3];
    let cpu = [(2, Fr::from(1u64)), (0, Fr::zero())];
    let gpu = [(1, Fr::from(3u64)); 17];
    let other = [(2, Fr::from(4u64)); 17];
    let requests = [SparseMsmRequest { bases: &bases, entries: &gpu },
        SparseMsmRequest { bases: &bases, entries: &cpu },
        SparseMsmRequest { bases: &bases, entries: &other }];
    let expected: Vec<_> = requests.iter().map(|r| reference(r.bases, r.entries)).collect();
    assert_eq!(hybrid.msm_sparse_batch(&requests).unwrap(), expected);
    let stats = owner.batch_stats().unwrap();
    assert_eq!(stats.icicle_calls, 1);
    assert_eq!(stats.max_batch_size, 2);
}

#[test]
#[ignore = "requires the existing FAIL_ALLOC_AT=1 native allocation failure fixture"]
fn indexed_hybrid_propagates_gpu_failure_without_cpu_retry() {
    let owner = IcicleBackend::new().unwrap();
    let hybrid = HybridBackend::new(&owner, 2).unwrap();
    let bases = [G1Projective::generator().into_affine()];
    let entries = [(0, Fr::from(3u64)); 2];
    let request = [SparseMsmRequest { bases: &bases, entries: &entries }];
    assert!(hybrid.msm_sparse_batch(&request).is_err());
    assert!(hybrid.msm_sparse_batch(&request).unwrap_err().to_string().contains("retired"));
    let cpu = [(0, Fr::from(1u64))];
    assert_eq!(hybrid.msm_sparse_batch(&[SparseMsmRequest { bases: &bases, entries: &cpu }]).unwrap(),
        vec![bases[0]]);
}
