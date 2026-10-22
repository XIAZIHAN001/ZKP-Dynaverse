use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{CurveGroup, Group};
use dynaverse_protocol::{
    dynaverse::{DynaverseParams, DynaverseVerifyingKey},
    toy_circuit::ToyCircuit,
    DynaverseProof, Instance,
};
use rand::{rngs::StdRng, SeedableRng};

#[test]
fn initial_proof_verifies_using_only_public_key_and_instance() {
    for n in [1, 8, 16, 32] {
        let circuit = ToyCircuit::new(n).unwrap();
        let (instance, witness) = circuit.valid_instance();
        let mut rng = StdRng::seed_from_u64(20260925 + n as u64);
        let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
        let verifier = params.verifying_key();
        let proof = params.prove_anchor(&instance, &witness).unwrap();
        drop(params);
        drop(witness);
        assert!(verifier.verify_anchor(&instance, &proof).unwrap(), "n={n}");
    }
}

fn bump_g1(point: G1Affine) -> G1Affine {
    (G1Projective::from(point) + G1Projective::generator()).into_affine()
}

fn bump_g2(point: G2Affine) -> G2Affine {
    (G2Projective::from(point) + G2Projective::generator()).into_affine()
}

fn assert_rejected(
    vk: &DynaverseVerifyingKey,
    instance: &Instance,
    proof: &DynaverseProof,
    mutate: impl FnOnce(&mut DynaverseProof),
) {
    let mut bad = proof.clone();
    mutate(&mut bad);
    assert!(!vk.verify_anchor(instance, &bad).unwrap());
}

#[test]
fn public_inputs_copy_and_addition_are_checked_by_public_verifier() {
    let n = 8;
    let circuit = ToyCircuit::new(n).unwrap();
    let (instance, witness) = circuit.valid_instance();
    let mut rng = StdRng::seed_from_u64(99);
    let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
    let vk = params.verifying_key();
    let proof = params.prove_anchor(&instance, &witness).unwrap();

    let mut wrong_public = instance.clone();
    wrong_public.public_inputs[0] += Fr::from(1u64);
    assert!(!vk.verify_anchor(&wrong_public, &proof).unwrap());
    wrong_public.public_inputs.pop();
    assert!(!vk.verify_anchor(&wrong_public, &proof).unwrap());

    let mut bad_add = witness.clone();
    bad_add.wires[n + 1] += Fr::from(1u64);
    let bad_add_proof = params.prove_anchor(&instance, &bad_add).unwrap();
    assert!(!vk.verify_anchor(&instance, &bad_add_proof).unwrap());

    let mut bad_copy = witness.clone();
    bad_copy.wires[3 * n] = Fr::from(1u64);
    bad_copy.wires[5 * n] = bad_copy.wires[4 * n];
    let bad_copy_proof = params.prove_anchor(&instance, &bad_copy).unwrap();
    assert!(!vk.verify_anchor(&instance, &bad_copy_proof).unwrap());

    let mut bad_mul = witness.clone();
    bad_mul.wires[5 * n + 1] += Fr::from(1u64);
    assert!(params.prove_anchor(&instance, &bad_mul).is_err());
}

#[test]
fn every_anchor_component_is_bound_into_verification() {
    let circuit = ToyCircuit::new(8).unwrap();
    let (instance, witness) = circuit.valid_instance();
    let mut rng = StdRng::seed_from_u64(101);
    let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
    let vk = params.verifying_key();
    let proof = params.prove_anchor(&instance, &witness).unwrap();
    assert!(vk.verify_anchor(&instance, &proof).unwrap());
    let mut other_rng = StdRng::seed_from_u64(102);
    let other = DynaverseParams::setup(ToyCircuit::new(8).unwrap(), &mut other_rng).unwrap();
    assert!(!other.verifying_key().verify_anchor(&instance, &proof).unwrap());

    assert_rejected(&vk, &instance, &proof, |p| p.anchor.wires.s1 = bump_g1(p.anchor.wires.s1));
    assert_rejected(&vk, &instance, &proof, |p| p.anchor.wires.s2 = bump_g1(p.anchor.wires.s2));
    assert_rejected(&vk, &instance, &proof, |p| p.anchor.wires.s3 = bump_g1(p.anchor.wires.s3));
    assert_rejected(&vk, &instance, &proof, |p| p.anchor.wires.s4 = bump_g1(p.anchor.wires.s4));
    assert_rejected(&vk, &instance, &proof, |p| p.anchor.wires.s5 = bump_g1(p.anchor.wires.s5));
    assert_rejected(&vk, &instance, &proof, |p| p.anchor.wires.s6 = bump_g1(p.anchor.wires.s6));
    assert_rejected(&vk, &instance, &proof, |p| p.anchor.s5_g2 = bump_g2(p.anchor.s5_g2));
    assert_rejected(&vk, &instance, &proof, |p| p.anchor.w_commitment = bump_g1(p.anchor.w_commitment));
    assert_rejected(&vk, &instance, &proof, |p| p.anchor.h_commitment = bump_g1(p.anchor.h_commitment));
    assert_rejected(&vk, &instance, &proof, |p| p.anchor.public_zero_quotient = bump_g1(p.anchor.public_zero_quotient));
    assert_rejected(&vk, &instance, &proof, |p| p.anchor.multiplication_quotient = bump_g1(p.anchor.multiplication_quotient));
    assert_rejected(&vk, &instance, &proof, |p| p.anchor.dynamo.commitments.alpha = bump_g1(p.anchor.dynamo.commitments.alpha));
    assert_rejected(&vk, &instance, &proof, |p| p.anchor.dynamo.q = bump_g1(p.anchor.dynamo.q));
    assert_rejected(&vk, &instance, &proof, |p| p.anchor.consistency.slices[4].qbd = bump_g1(p.anchor.consistency.slices[4].qbd));
}
