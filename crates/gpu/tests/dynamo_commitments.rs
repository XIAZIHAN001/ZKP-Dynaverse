use ark_bn254::Fr;
use dynaverse_gpu::CpuBackend;
use dynaverse_protocol::dynamo::{DynamoParams, DynamoPermutation, SparseVector};
use rand::{rngs::StdRng, SeedableRng};

#[test]
fn five_sparse_commitments_match_dense_msm_at_small_and_warp_boundary_sizes() {
    for seed in [7u64, 17, 2026] {
        let mut rng = StdRng::seed_from_u64(seed);
        for m in [8usize, 16, 256] {
            let sigma = DynamoPermutation::try_new((0..m).rev().collect()).unwrap();
            let params = DynamoParams::setup(m, sigma, &mut rng).unwrap();
            for k in [0usize, 1, 3, m] {
                let mut dense = vec![Fr::from(0u64); m];
                for (offset, value) in dense.iter_mut().enumerate().take(k) {
                    *value = Fr::from((offset as u64) + 1);
                }
                let entries = dense
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|(_, value)| *value != Fr::from(0u64))
                    .collect();
                let sparse = SparseVector::try_new(m, entries).unwrap();
                let got = params.commit_five_sparse(&sparse, &CpuBackend).unwrap();
                let want = params.commit_five_dense_reference(&dense).unwrap();
                assert_eq!(got.alpha, want.alpha, "alpha seed={seed}, m={m}, k={k}");
                assert_eq!(got.beta, want.beta, "beta seed={seed}, m={m}, k={k}");
                assert_eq!(got.z, want.z, "Z seed={seed}, m={m}, k={k}");
                assert_eq!(got.h, want.h, "H seed={seed}, m={m}, k={k}");
                assert_eq!(got.b, want.b, "B seed={seed}, m={m}, k={k}");
            }
        }
    }
}

#[test]
fn setup_rejects_mismatched_and_non_radix_two_sizes() {
    let mut rng = StdRng::seed_from_u64(7);
    let sigma = DynamoPermutation::try_new(vec![0, 1, 2, 3]).unwrap();
    assert!(DynamoParams::setup(3, sigma.clone(), &mut rng).is_err());
    assert!(DynamoParams::setup(8, sigma, &mut rng).is_err());
}
