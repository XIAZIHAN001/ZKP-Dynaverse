use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::{AffineRepr, CurveGroup};
use dynaverse_gpu::CpuBackend;
use dynaverse_protocol::dynamo::{
    derive_h_sparse, dynamo_prove_sparse, dynamo_verify, DynamoParams, DynamoPermutation,
    SparseVector,
};
use dynaverse_protocol::DynamoProof;
use rand::{rngs::StdRng, SeedableRng};

fn fixture(seed: u64, m: usize) -> (DynamoParams, SparseVector, SparseVector, StdRng) {
    let mut rng = StdRng::seed_from_u64(seed);
    let sigma = DynamoPermutation::try_new((0..m).rev().collect()).unwrap();
    let params = DynamoParams::setup(m, sigma.clone(), &mut rng).unwrap();
    let z = SparseVector::try_new(
        m,
        vec![(1, Fr::from(3u64)), (m / 2, Fr::from(5u64))],
    )
    .unwrap();
    let h = derive_h_sparse(&z, &sigma).unwrap();
    (params, z, h, rng)
}

fn bumped(point: G1Affine) -> G1Affine {
    (G1Projective::from(point) + G1Projective::from(G1Affine::generator())).into_affine()
}

fn rejects_bump(params: &DynamoParams, valid: &DynamoProof, field: &str) {
    let mut bad = valid.clone();
    match field {
        "alpha" => bad.commitments.alpha = bumped(bad.commitments.alpha),
        "beta" => bad.commitments.beta = bumped(bad.commitments.beta),
        "Z" => bad.commitments.z = bumped(bad.commitments.z),
        "H" => bad.commitments.h = bumped(bad.commitments.h),
        "B" => bad.commitments.b = bumped(bad.commitments.b),
        "q" => bad.q = bumped(bad.q),
        "instance_z" => bad.instance_z = bumped(bad.instance_z),
        "instance_h" => bad.instance_h = bumped(bad.instance_h),
        _ => panic!("unknown proof field"),
    }
    assert!(!dynamo_verify(params, &bad).unwrap(), "tampered {field} accepted");
}

#[test]
fn valid_sparse_proofs_verify_at_two_sizes() {
    for (seed, m) in [(7u64, 8usize), (17, 16)] {
        let (params, z, h, mut rng) = fixture(seed, m);
        let proof = dynamo_prove_sparse(&params, &z, &h, &CpuBackend, &mut rng).unwrap();
        assert!(dynamo_verify(&params, &proof).unwrap(), "seed={seed}, m={m}");
    }
}

#[test]
fn modifying_any_named_commitment_or_instance_is_rejected() {
    let (params, z, h, mut rng) = fixture(2026, 8);
    let proof = dynamo_prove_sparse(&params, &z, &h, &CpuBackend, &mut rng).unwrap();
    assert!(dynamo_verify(&params, &proof).unwrap());
    for field in ["alpha", "beta", "Z", "H", "B", "q", "instance_z", "instance_h"] {
        rejects_bump(&params, &proof, field);
    }
}

#[test]
fn broken_h_relation_fails_verification() {
    let (params, z, _, mut rng) = fixture(7, 8);
    let wrong_h = SparseVector::try_new(8, vec![(0, Fr::from(1u64))]).unwrap();
    let proof = dynamo_prove_sparse(&params, &z, &wrong_h, &CpuBackend, &mut rng).unwrap();
    assert!(!dynamo_verify(&params, &proof).unwrap());
}

#[test]
fn empty_and_identity_permutation_proofs_verify() {
    let mut rng = StdRng::seed_from_u64(17);
    let identity = DynamoPermutation::try_new(vec![0, 1, 2, 3, 4, 5, 6, 7]).unwrap();
    let params = DynamoParams::setup(8, identity, &mut rng).unwrap();
    let empty = SparseVector::try_new(8, vec![]).unwrap();
    let empty_proof = dynamo_prove_sparse(&params, &empty, &empty, &CpuBackend, &mut rng).unwrap();
    assert!(dynamo_verify(&params, &empty_proof).unwrap());
    let z = SparseVector::try_new(8, vec![(3, Fr::from(12u64))]).unwrap();
    let proof = dynamo_prove_sparse(&params, &z, &empty, &CpuBackend, &mut rng).unwrap();
    assert!(dynamo_verify(&params, &proof).unwrap());
}
