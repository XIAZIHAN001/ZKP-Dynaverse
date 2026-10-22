use ark_bn254::Fr;
use dynaverse_protocol::dynamo::{derive_h_sparse, DynamoPermutation, SparseVector};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};

#[test]
fn sparse_relation_matches_hand_calculated_permutation() {
    let sigma = DynamoPermutation::try_new(vec![2, 0, 3, 1]).unwrap();
    let z = SparseVector::try_new(
        4,
        vec![(0, Fr::from(3u64)), (2, Fr::from(5u64))],
    )
    .unwrap();

    let h = derive_h_sparse(&z, &sigma).unwrap();
    assert_eq!(
        h.entries(),
        &[
            (0, -Fr::from(2u64)),
            (1, -Fr::from(3u64)),
            (2, Fr::from(5u64)),
        ]
    );
}

#[test]
fn sparse_relation_matches_dense_oracle_for_seeded_permutations() {
    for seed in [7u64, 17, 2026] {
        let mut rng = StdRng::seed_from_u64(seed);
        for m in [8usize, 16, 256] {
            let mut sigma: Vec<usize> = (0..m).collect();
            sigma.shuffle(&mut rng);
            let mut dense_z = vec![Fr::from(0u64); m];
            for _ in 0..(m / 4) {
                let index = rng.gen_range(0..m);
                dense_z[index] = Fr::from(rng.gen_range(1u64..100));
            }
            let entries = dense_z
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, value)| *value != Fr::from(0u64))
                .collect();
            let z = SparseVector::try_new(m, entries).unwrap();
            let permutation = DynamoPermutation::try_new(sigma.clone()).unwrap();
            let h = derive_h_sparse(&z, &permutation).unwrap();
            let mut actual = vec![Fr::from(0u64); m];
            for &(index, value) in h.entries() {
                actual[index] = value;
            }
            let expected: Vec<Fr> = (0..m)
                .map(|i| dense_z[i] - dense_z[sigma[i]])
                .collect();
            assert_eq!(actual, expected, "seed={seed}, m={m}");
            assert!(h.entries().len() <= 2 * z.entries().len());
        }
    }
}

#[test]
fn identity_and_zero_vector_have_zero_relation() {
    let sigma = DynamoPermutation::try_new(vec![0, 1, 2, 3]).unwrap();
    let zero = SparseVector::try_new(4, vec![]).unwrap();
    assert!(derive_h_sparse(&zero, &sigma).unwrap().entries().is_empty());
    let one = SparseVector::try_new(4, vec![(2, Fr::from(9u64))]).unwrap();
    assert!(derive_h_sparse(&one, &sigma).unwrap().entries().is_empty());
}

#[test]
fn malformed_permutation_and_sparse_indices_are_rejected() {
    assert!(DynamoPermutation::try_new(vec![]).is_err());
    assert!(DynamoPermutation::try_new(vec![0, 1, 2]).is_err());
    assert!(DynamoPermutation::try_new(vec![0, 1, 1, 3]).is_err());
    assert!(DynamoPermutation::try_new(vec![0, 1, 2, 4]).is_err());
    assert!(SparseVector::try_new(4, vec![(1, Fr::from(1u64)), (1, Fr::from(2u64))]).is_err());
    assert!(SparseVector::try_new(4, vec![(2, Fr::from(1u64)), (1, Fr::from(2u64))]).is_err());
    assert!(SparseVector::try_new(4, vec![(4, Fr::from(1u64))]).is_err());
    assert!(SparseVector::try_new(4, vec![(1, Fr::from(0u64))]).is_err());
    let z = SparseVector::try_new(8, vec![]).unwrap();
    let sigma = DynamoPermutation::try_new(vec![0, 1, 2, 3]).unwrap();
    assert!(derive_h_sparse(&z, &sigma).is_err());
}
