use dynaverse_protocol::{
    dynaverse::{DynaverseParams, DynaverseVerifyingKey},
    toy_circuit::ToyCircuit,
    DeltaProof, DynaverseApi, DynaverseProof, Instance, Update,
};
use dynaverse_protocol::domain::{affected_set_factorization, evaluate_polynomial};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{CurveGroup, Group};
use ark_ff::{Field, Zero};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use rand::{rngs::StdRng, SeedableRng};

#[test]
fn initial_proof_has_no_delta_envelope() {
    let circuit = ToyCircuit::new(8).unwrap();
    let (instance, witness) = circuit.valid_instance();
    let mut rng = StdRng::seed_from_u64(22);
    let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
    let proof = params.prove_anchor(&instance, &witness).unwrap();
    assert!(proof.delta.is_none());
}

#[test]
fn affected_set_factors_small_domain_for_empty_single_and_full_sets() {
    let domain = Radix2EvaluationDomain::<Fr>::new(8).unwrap();
    for affected in [vec![], vec![3], (0..8).collect()] {
        let (ai, bi) = affected_set_factorization(&domain, &affected).unwrap();
        for x in [Fr::from(2u64), Fr::from(7u64), Fr::from(19u64)] {
            assert_eq!(
                evaluate_polynomial(&ai, x) * evaluate_polynomial(&bi, x),
                x.pow([8]) - Fr::from(1u64)
            );
        }
    }
    assert!(affected_set_factorization(&domain, &[2, 2]).is_err());
    assert!(affected_set_factorization(&domain, &[8]).is_err());
}

#[test]
fn legal_one_wire_change_produces_a_delta_proof() {
    let n = 8;
    let circuit = ToyCircuit::new(n).unwrap();
    let (instance, witness) = circuit.valid_instance();
    let mut rng = StdRng::seed_from_u64(23);
    let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
    let (proof, mut aux) = params.prove_with_aux(&instance, &witness).unwrap();
    let update = Update {
        indices: vec![4 * n],
        values: vec![witness.wires[4 * n] + Fr::from(1u64)],
    };
    let changed = params
        .update_proof(&instance, &instance, &witness, &update, &proof, &mut aux)
        .unwrap();
    assert!(changed.delta.is_some());
    assert_eq!(changed.generation, 1);
    let vk = params.verifying_key();
    drop(params);
    drop(aux);
    drop(witness);
    assert!(vk.verify(&instance, &changed).unwrap());
    let mut missing_delta = changed.clone();
    missing_delta.delta = None;
    assert!(!vk.verify(&instance, &missing_delta).unwrap());
}

#[test]
fn zero_change_preserves_proof_and_stale_state_is_rejected() {
    let circuit = ToyCircuit::new(8).unwrap();
    let (instance, witness) = circuit.valid_instance();
    let mut rng = StdRng::seed_from_u64(24);
    let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
    let (proof, mut aux) = params.prove_with_aux(&instance, &witness).unwrap();
    let no_change = Update { indices: vec![], values: vec![] };
    let same = params.update_proof(
        &instance, &instance, &witness, &no_change, &proof, &mut aux
    ).unwrap();
    assert_eq!(same, proof);
    assert_eq!(aux.scheduler().generation, 0);
    let mut stale = witness.clone();
    stale.wires[0] += Fr::from(1u64);
    assert!(params.update_proof(
        &instance, &instance, &stale, &no_change, &proof, &mut aux
    ).is_err());
}

#[test]
fn two_addition_changes_use_the_empty_affected_set() {
    let n = 8;
    let circuit = ToyCircuit::new(n).unwrap();
    let (instance, witness) = circuit.valid_instance();
    let mut rng = StdRng::seed_from_u64(25);
    let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
    let (proof, mut aux) = params.prove_with_aux(&instance, &witness).unwrap();
    let mut changed_instance = instance.clone();
    changed_instance.public_inputs[1] += Fr::from(1u64);
    let update = Update {
        indices: vec![1, n + 1],
        values: vec![
            witness.wires[1] + Fr::from(1u64),
            witness.wires[n + 1] - Fr::from(1u64),
        ],
    };
    let changed = params.update_proof(
        &instance, &changed_instance, &witness, &update, &proof, &mut aux
    ).unwrap();
    let delta = changed.delta.as_ref().unwrap();
    assert_eq!(delta.affected_divisor_g1, G1Projective::generator().into_affine());
    assert_eq!(delta.changed_local.s4, G1Projective::zero().into_affine());
    assert!(params.verifying_key().verify(&changed_instance, &changed).unwrap());
}

#[test]
fn threshold_rebuilds_and_reanchors_after_four_true_changes() {
    let n = 8;
    let circuit = ToyCircuit::new(n).unwrap();
    let (instance, witness) = circuit.valid_instance();
    let mut rng = StdRng::seed_from_u64(26);
    let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
    let (proof, mut aux) = params.prove_with_aux(&instance, &witness).unwrap();
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for i in [1, 2] {
        indices.push(4 * n + i);
        values.push(witness.wires[4 * n + i] + Fr::from(1u64));
        indices.push(5 * n + i);
        values.push(witness.wires[5 * n + i] + witness.wires[3 * n + i]);
    }
    let update = Update { indices, values };
    let changed = params.update_proof(
        &instance, &instance, &witness, &update, &proof, &mut aux
    ).unwrap();
    assert!(changed.delta.is_none());
    assert_eq!(changed.generation, 1);
    assert_eq!(aux.anchor_witness(), aux.current_witness());
    assert_eq!(aux.scheduler().updates_since_rebuild, 0);
    assert!(params.verifying_key().verify(&instance, &changed).unwrap());
}

fn bump_g1(point: G1Affine) -> G1Affine {
    (G1Projective::from(point) + G1Projective::generator()).into_affine()
}

fn bump_g2(point: G2Affine) -> G2Affine {
    (G2Projective::from(point) + G2Projective::generator()).into_affine()
}

fn reject_delta_change(
    vk: &DynaverseVerifyingKey,
    instance: &Instance,
    proof: &DynaverseProof,
    mutate: impl FnOnce(&mut DeltaProof),
) {
    let mut wrong = proof.clone();
    mutate(wrong.delta.as_mut().unwrap());
    assert!(!vk.verify(instance, &wrong).unwrap());
}

#[test]
fn public_delta_verifier_rejects_each_field_class_and_wrong_instance() {
    let n = 8;
    let circuit = ToyCircuit::new(n).unwrap();
    let (instance, witness) = circuit.valid_instance();
    let mut rng = StdRng::seed_from_u64(27);
    let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
    let vk = params.verifying_key();
    let (proof, mut aux) = params.prove_with_aux(&instance, &witness).unwrap();
    let update = Update {
        indices: vec![4 * n],
        values: vec![witness.wires[4 * n] + Fr::from(1u64)],
    };
    let changed = params.update_proof(
        &instance, &instance, &witness, &update, &proof, &mut aux
    ).unwrap();
    assert!(vk.verify(&instance, &changed).unwrap());
    let mut wrong_instance = instance.clone();
    wrong_instance.public_inputs[0] += Fr::from(1u64);
    assert!(!vk.verify(&wrong_instance, &changed).unwrap());

    reject_delta_change(&vk, &instance, &changed, |d| d.changed_wires.s1 = bump_g1(d.changed_wires.s1));
    reject_delta_change(&vk, &instance, &changed, |d| d.changed_wires.s2 = bump_g1(d.changed_wires.s2));
    reject_delta_change(&vk, &instance, &changed, |d| d.changed_wires.s3 = bump_g1(d.changed_wires.s3));
    reject_delta_change(&vk, &instance, &changed, |d| d.changed_wires.s4 = bump_g1(d.changed_wires.s4));
    reject_delta_change(&vk, &instance, &changed, |d| d.changed_wires.s5 = bump_g1(d.changed_wires.s5));
    reject_delta_change(&vk, &instance, &changed, |d| d.changed_wires.s6 = bump_g1(d.changed_wires.s6));
    reject_delta_change(&vk, &instance, &changed, |d| d.changed_w_commitment = bump_g1(d.changed_w_commitment));
    reject_delta_change(&vk, &instance, &changed, |d| d.changed_h_commitment = bump_g1(d.changed_h_commitment));
    reject_delta_change(&vk, &instance, &changed, |d| d.public_zero_quotient = bump_g1(d.public_zero_quotient));
    reject_delta_change(&vk, &instance, &changed, |d| d.affected_divisor_g1 = bump_g1(d.affected_divisor_g1));
    reject_delta_change(&vk, &instance, &changed, |d| d.affected_divisor_g2 = bump_g2(d.affected_divisor_g2));
    reject_delta_change(&vk, &instance, &changed, |d| d.unaffected_divisor_g1 = bump_g1(d.unaffected_divisor_g1));
    reject_delta_change(&vk, &instance, &changed, |d| d.unaffected_divisor_g2 = bump_g2(d.unaffected_divisor_g2));
    reject_delta_change(&vk, &instance, &changed, |d| d.anchor_local.s4 = bump_g1(d.anchor_local.s4));
    reject_delta_change(&vk, &instance, &changed, |d| d.changed_local.s5 = bump_g1(d.changed_local.s5));
    reject_delta_change(&vk, &instance, &changed, |d| d.updated_s5_g2 = bump_g2(d.updated_s5_g2));
    reject_delta_change(&vk, &instance, &changed, |d| d.anchor_openings.s6 = bump_g1(d.anchor_openings.s6));
    reject_delta_change(&vk, &instance, &changed, |d| d.changed_openings.s4 = bump_g1(d.changed_openings.s4));
    reject_delta_change(&vk, &instance, &changed, |d| d.unaffected_zero_quotients.s5 = bump_g1(d.unaffected_zero_quotients.s5));
    reject_delta_change(&vk, &instance, &changed, |d| d.local_multiplication_quotient = bump_g1(d.local_multiplication_quotient));
    reject_delta_change(&vk, &instance, &changed, |d| d.dynamo.commitments.alpha = bump_g1(d.dynamo.commitments.alpha));
    reject_delta_change(&vk, &instance, &changed, |d| d.consistency.slices[0].qac = bump_g1(d.consistency.slices[0].qac));
}

#[test]
fn stable_api_proves_updates_and_verifies() {
    let n = 8;
    let circuit = ToyCircuit::new(n).unwrap();
    let (instance, witness) = circuit.valid_instance();
    let mut rng = StdRng::seed_from_u64(28);
    let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
    let (proof, mut aux) = DynaverseApi::prove(&params, &instance, &witness).unwrap();
    assert!(DynaverseApi::verify(&params, &instance, &proof).unwrap());
    let update = Update {
        indices: vec![4 * n],
        values: vec![witness.wires[4 * n] + Fr::from(1u64)],
    };
    let changed = DynaverseApi::update(
        &params, &instance, &instance, &witness, &update, &proof, &mut aux
    ).unwrap();
    assert!(DynaverseApi::verify(&params, &instance, &changed).unwrap());
}

#[test]
fn update_matrix_accepts_k_zero_one_two_and_four_at_three_sizes() {
    for n in [8usize, 16, 32] {
        let circuit = ToyCircuit::new(n).unwrap();
        let (instance, witness) = circuit.valid_instance();
        let mut rng = StdRng::seed_from_u64(1000 + n as u64);
        let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
        let vk = params.verifying_key();
        for k in [0, 1, 2, 4] {
            let (proof, mut aux) = params.prove_with_aux(&instance, &witness).unwrap();
            let update = match k {
                0 => Update { indices: vec![], values: vec![] },
                1 => Update {
                    indices: vec![4 * n],
                    values: vec![witness.wires[4 * n] + Fr::from(1u64)],
                },
                2 => Update {
                    indices: vec![4 * n + 1, 5 * n + 1],
                    values: vec![
                        witness.wires[4 * n + 1] + Fr::from(1u64),
                        witness.wires[5 * n + 1] + witness.wires[3 * n + 1],
                    ],
                },
                4 => {
                    let mut indices = Vec::new();
                    let mut values = Vec::new();
                    for i in [1, 2] {
                        indices.extend([4 * n + i, 5 * n + i]);
                        values.extend([
                            witness.wires[4 * n + i] + Fr::from(1u64),
                            witness.wires[5 * n + i] + witness.wires[3 * n + i],
                        ]);
                    }
                    Update { indices, values }
                }
                _ => unreachable!(),
            };
            let changed = params.update_proof(
                &instance, &instance, &witness, &update, &proof, &mut aux
            ).unwrap();
            assert!(vk.verify(&instance, &changed).unwrap(), "n={n}, k={k}");
            assert_eq!(changed.delta.is_some(), k > 0 && k < super_threshold(n));
        }
    }
}

fn super_threshold(n: usize) -> usize {
    (1..=n).find(|root| root * root >= n).unwrap()
}

#[test]
fn consecutive_updates_and_duplicate_index_normalization() {
    let n = 32;
    let circuit = ToyCircuit::new(n).unwrap();
    let (instance, witness) = circuit.valid_instance();
    let mut rng = StdRng::seed_from_u64(1002);
    let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
    let vk = params.verifying_key();
    let (proof0, mut aux) = params.prove_with_aux(&instance, &witness).unwrap();
    let first = Update {
        indices: vec![4 * n, 4 * n],
        values: vec![witness.wires[4 * n], witness.wires[4 * n] + Fr::from(1u64)],
    };
    let proof1 = params.update_proof(
        &instance, &instance, &witness, &first, &proof0, &mut aux
    ).unwrap();
    assert!(vk.verify(&instance, &proof1).unwrap());
    let witness1 = aux.current_witness().clone();
    let second = Update {
        indices: vec![4 * n + 1, 5 * n + 1],
        values: vec![
            witness.wires[4 * n + 1] + Fr::from(1u64),
            witness.wires[5 * n + 1] + witness.wires[3 * n + 1],
        ],
    };
    let proof2 = params.update_proof(
        &instance, &instance, &witness1, &second, &proof1, &mut aux
    ).unwrap();
    assert!(vk.verify(&instance, &proof2).unwrap());
    assert_eq!(proof2.generation, 2);
    assert!(proof2.delta.is_some());
}

#[test]
fn invalid_updates_are_rejected_before_state_changes() {
    let n = 8;
    let circuit = ToyCircuit::new(n).unwrap();
    let (instance, witness) = circuit.valid_instance();
    let mut rng = StdRng::seed_from_u64(1003);
    let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
    let (proof, mut aux) = params.prove_with_aux(&instance, &witness).unwrap();
    let original_aux = aux.clone();
    let cases = [
        Update { indices: vec![n + 1], values: vec![witness.wires[n + 1] + Fr::from(1u64)] },
        Update { indices: vec![3 * n], values: vec![Fr::from(1u64)] },
        Update { indices: vec![5 * n + 1], values: vec![witness.wires[5 * n + 1] + Fr::from(1u64)] },
        Update { indices: vec![6 * n], values: vec![Fr::from(1u64)] },
        Update { indices: vec![1], values: vec![] },
    ];
    for update in cases {
        assert!(params.update_proof(
            &instance, &instance, &witness, &update, &proof, &mut aux
        ).is_err());
        assert_eq!(aux, original_aux);
    }
    let mut wrong_public = instance.clone();
    wrong_public.public_inputs[0] += Fr::from(1u64);
    assert!(params.update_proof(
        &instance, &wrong_public, &witness,
        &Update { indices: vec![], values: vec![] }, &proof, &mut aux
    ).is_err());
    assert_eq!(aux, original_aux);
}

#[test]
fn mismatched_prover_state_cannot_be_committed() {
    let n = 8;
    let circuit = ToyCircuit::new(n).unwrap();
    let (instance, witness) = circuit.valid_instance();
    let mut rng = StdRng::seed_from_u64(1004);
    let params = DynaverseParams::setup(circuit.clone(), &mut rng).unwrap();
    let (proof, mut aux) = params.prove_with_aux(&instance, &witness).unwrap();
    let update = Update {
        indices: vec![4 * n],
        values: vec![witness.wires[4 * n] + Fr::from(1u64)],
    };
    let mut other_rng = StdRng::seed_from_u64(1005);
    let other = DynaverseParams::setup(circuit, &mut other_rng).unwrap();
    let snapshot = aux.clone();
    assert!(other.update_proof(
        &instance, &instance, &witness, &update, &proof, &mut aux
    ).is_err());
    assert_eq!(aux, snapshot);

}
