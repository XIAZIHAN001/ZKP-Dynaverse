use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{CurveGroup, Group};
use dynaverse_protocol::dynaverse::{DynaverseParams, DynaverseVerifyingKey};
use dynaverse_protocol::toy_circuit::ToyCircuit;
use dynaverse_protocol::trace::{CpuTrace, VerificationTrace};
use dynaverse_protocol::{AnchorOpeningPolicy, DynaverseProof, Instance, Update, VerificationMode};
use rand::{rngs::StdRng, SeedableRng};

fn bump_g1(point: G1Affine) -> G1Affine {
    (G1Projective::from(point) + G1Projective::generator()).into_affine()
}

fn bump_g2(point: G2Affine) -> G2Affine {
    (G2Projective::from(point) + G2Projective::generator()).into_affine()
}

fn assert_rejected(vk: &DynaverseVerifyingKey, instance: &Instance, proof: &DynaverseProof) {
    assert!(!vk.verify_reference(instance, proof).unwrap(), "reference accepted tampering");
    assert!(!vk.verify(instance, proof).unwrap(), "product verifier accepted tampering");
}

#[test]
fn product_verifier_preserves_equations_and_reduces_final_exponentiations() {
    for n in [1, 8, 32] {
        let circuit = ToyCircuit::new(n).unwrap();
        let (instance, witness) = circuit.valid_instance();
        let mut rng = StdRng::seed_from_u64(2910 + n as u64);
        let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
        let vk = params.verifying_key();
        let (anchor, mut aux) = params.prove_with_aux_policy(&instance, &witness, AnchorOpeningPolicy::Dense).unwrap();
        let mut trace = VerificationTrace::new();
        assert!(vk.verify_traced(&instance, &anchor, VerificationMode::Product, &mut trace).unwrap());
        assert!(vk.verify_reference(&instance, &anchor).unwrap());
        assert_eq!(trace.counts().pairing_terms, 84);
        assert_eq!(trace.counts().equations, 37);
        assert_eq!(trace.counts().miller_loops, 37);
        assert_eq!(trace.counts().final_exponentiations, 37);
        if n == 1 { continue; }
        let update = Update { indices: vec![4 * n], values: vec![witness.wires[4 * n] + Fr::from(1u64)] };
        let delta = params.update_proof(&instance, &instance, &witness, &update, &anchor, &mut aux).unwrap();
        trace = VerificationTrace::new();
        assert!(vk.verify_traced(&instance, &delta, VerificationMode::Product, &mut trace).unwrap());
        assert!(vk.verify_reference(&instance, &delta).unwrap());
        assert_eq!(trace.counts().pairing_terms, 192);
        assert_eq!(trace.counts().equations, 86);
        assert_eq!(trace.counts().miller_loops, 86);
        assert_eq!(trace.counts().final_exponentiations, 86);
    }
}

#[test]
fn product_verifier_rejects_each_consistency_field_and_canceling_errors() {
    let circuit = ToyCircuit::new(8).unwrap();
    let (instance, witness) = circuit.valid_instance();
    let mut rng = StdRng::seed_from_u64(2919);
    let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
    let vk = params.verifying_key();
    let (anchor, mut aux) = params.prove_with_aux_policy(&instance, &witness, AnchorOpeningPolicy::Dense).unwrap();
    let update = Update { indices: vec![32], values: vec![witness.wires[32] + Fr::from(1u64)] };
    let delta = params.update_proof(&instance, &instance, &witness, &update, &anchor, &mut aux).unwrap();
    for use_delta in [false, true] {
        let proof = if use_delta { &delta } else { &anchor };
        for index in 0..6 {
            for field in 0..7 {
                let mut wrong = proof.clone();
                // Borrow one slice exclusively for the mutation. The borrow
                // ends before the complete proof is passed to both verifiers.
                let slice = if use_delta { &mut wrong.delta.as_mut().unwrap().consistency.slices[index] }
                    else { &mut wrong.anchor.consistency.slices[index] };
                match field {
                    0 => slice.qac = bump_g1(slice.qac),
                    1 => slice.qcd = bump_g1(slice.qcd),
                    2 => slice.qbd = bump_g1(slice.qbd),
                    3 => slice.d = bump_g1(slice.d),
                    4 => slice.a_degree = bump_g1(slice.a_degree),
                    5 => slice.b_degree = bump_g1(slice.b_degree),
                    _ => slice.d_degree = bump_g1(slice.d_degree),
                }
                assert_rejected(&vk, &instance, &wrong);
            }
        }
        let mut wrong = proof.clone();
        let slices = if use_delta { &mut wrong.delta.as_mut().unwrap().consistency.slices }
            else { &mut wrong.anchor.consistency.slices };
        slices[0].a_degree = bump_g1(slices[0].a_degree);
        slices[1].a_degree = (G1Projective::from(slices[1].a_degree) - G1Projective::generator()).into_affine();
        // Opposite errors cancel in an unweighted aggregate across equations.
        // Both verifiers must still reject each independently checked relation.
        assert_rejected(&vk, &instance, &wrong);
    }
}

#[test]
fn product_verifier_matches_reference_for_other_proof_fields_and_public_inputs() {
    let circuit = ToyCircuit::new(8).unwrap();
    let (instance, witness) = circuit.valid_instance();
    let mut rng = StdRng::seed_from_u64(2920);
    let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
    let vk = params.verifying_key();
    let (anchor, mut aux) = params.prove_with_aux_policy(&instance, &witness, AnchorOpeningPolicy::Dense).unwrap();
    let update = Update { indices: vec![32], values: vec![witness.wires[32] + Fr::from(1u64)] };
    let delta = params.update_proof(&instance, &instance, &witness, &update, &anchor, &mut aux).unwrap();
    for field in 0..18 {
        let mut wrong = anchor.clone();
        let a = &mut wrong.anchor;
        match field {
            0 => a.wires.s1 = bump_g1(a.wires.s1),
            1 => a.wires.s2 = bump_g1(a.wires.s2),
            2 => a.wires.s3 = bump_g1(a.wires.s3),
            3 => a.wires.s4 = bump_g1(a.wires.s4),
            4 => a.wires.s5 = bump_g1(a.wires.s5),
            5 => a.wires.s6 = bump_g1(a.wires.s6),
            6 => a.w_commitment = bump_g1(a.w_commitment),
            7 => a.h_commitment = bump_g1(a.h_commitment),
            8 => a.s5_g2 = bump_g2(a.s5_g2),
            9 => a.public_zero_quotient = bump_g1(a.public_zero_quotient),
            10 => a.multiplication_quotient = bump_g1(a.multiplication_quotient),
            11 => a.dynamo.commitments.alpha = bump_g1(a.dynamo.commitments.alpha),
            12 => a.dynamo.commitments.beta = bump_g1(a.dynamo.commitments.beta),
            13 => a.dynamo.commitments.z = bump_g1(a.dynamo.commitments.z),
            14 => a.dynamo.commitments.h = bump_g1(a.dynamo.commitments.h),
            15 => a.dynamo.commitments.b = bump_g1(a.dynamo.commitments.b),
            16 => a.dynamo.q = bump_g1(a.dynamo.q),
            _ => a.dynamo.instance_z = bump_g1(a.dynamo.instance_z),
        }
        assert_rejected(&vk, &instance, &wrong);
    }
    for field in 0..27 {
        let mut wrong = delta.clone();
        let d = wrong.delta.as_mut().unwrap();
        match field {
            0 => d.changed_wires.s1 = bump_g1(d.changed_wires.s1),
            1 => d.changed_wires.s2 = bump_g1(d.changed_wires.s2),
            2 => d.changed_wires.s3 = bump_g1(d.changed_wires.s3),
            3 => d.changed_wires.s4 = bump_g1(d.changed_wires.s4),
            4 => d.changed_wires.s5 = bump_g1(d.changed_wires.s5),
            5 => d.changed_wires.s6 = bump_g1(d.changed_wires.s6),
            6 => d.public_zero_quotient = bump_g1(d.public_zero_quotient),
            7 => d.affected_divisor_g1 = bump_g1(d.affected_divisor_g1),
            8 => d.affected_divisor_g2 = bump_g2(d.affected_divisor_g2),
            9 => d.unaffected_divisor_g1 = bump_g1(d.unaffected_divisor_g1),
            10 => d.unaffected_divisor_g2 = bump_g2(d.unaffected_divisor_g2),
            11 => d.anchor_local.s4 = bump_g1(d.anchor_local.s4),
            12 => d.changed_local.s5 = bump_g1(d.changed_local.s5),
            13 => d.updated_s5_g2 = bump_g2(d.updated_s5_g2),
            14 => d.anchor_openings.s6 = bump_g1(d.anchor_openings.s6),
            15 => d.changed_openings.s4 = bump_g1(d.changed_openings.s4),
            16 => d.unaffected_zero_quotients.s5 = bump_g1(d.unaffected_zero_quotients.s5),
            17 => d.local_multiplication_quotient = bump_g1(d.local_multiplication_quotient),
            18 => d.dynamo.commitments.alpha = bump_g1(d.dynamo.commitments.alpha),
            19 => d.dynamo.commitments.beta = bump_g1(d.dynamo.commitments.beta),
            20 => d.dynamo.commitments.z = bump_g1(d.dynamo.commitments.z),
            21 => d.dynamo.commitments.h = bump_g1(d.dynamo.commitments.h),
            22 => d.dynamo.commitments.b = bump_g1(d.dynamo.commitments.b),
            23 => d.dynamo.q = bump_g1(d.dynamo.q),
            24 => d.dynamo.instance_h = bump_g1(d.dynamo.instance_h),
            25 => d.changed_w_commitment = bump_g1(d.changed_w_commitment),
            _ => d.changed_h_commitment = bump_g1(d.changed_h_commitment),
        }
        assert_rejected(&vk, &instance, &wrong);
    }
    let mut wrong_instance = instance.clone();
    wrong_instance.public_inputs[0] += Fr::from(1u64);
    assert_rejected(&vk, &wrong_instance, &delta);
    let mut wrong_generation = delta.clone();
    wrong_generation.anchor_generation = wrong_generation.generation;
    assert_rejected(&vk, &instance, &wrong_generation);
    let mut other_rng = StdRng::seed_from_u64(2921);
    let other = DynaverseParams::setup(ToyCircuit::new(8).unwrap(), &mut other_rng).unwrap();
    assert_rejected(&other.verifying_key(), &instance, &delta);
}

#[test]
fn candidate_verifier_modes_preserve_proofs_and_state_through_rebuild_and_noop() {
    let n = 16;
    let circuit = ToyCircuit::new(n).unwrap();
    let (instance, witness) = circuit.valid_instance();
    let mut rng = StdRng::seed_from_u64(2922);
    let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
    for policy in [AnchorOpeningPolicy::Dense, AnchorOpeningPolicy::Precompute] {
        let (mut reference, mut reference_aux) = params.prove_with_aux_policy(&instance, &witness, policy).unwrap();
        let (mut product, mut product_aux) = params.prove_with_aux_policy(&instance, &witness, policy).unwrap();
        for step in 0..5 {
            // Owned snapshots release the immutable aux borrows before update
            // takes exclusive mutable borrows of the two auxiliary states.
            let old_reference = reference_aux.current_witness().clone();
            let old_product = product_aux.current_witness().clone();
            let update = if step == 3 { Update { indices: vec![], values: vec![] } }
                else if step == 0 || step == 4 {
                    Update { indices: vec![4 * n], values: vec![old_reference.wires[4 * n] + Fr::from(1u64)] }
                } else {
                    let gate = step;
                    let multiplier = old_reference.wires[4 * n + gate] + Fr::from(1u64);
                    Update { indices: vec![4 * n + gate, 5 * n + gate],
                        values: vec![multiplier, old_reference.wires[3 * n + gate] * multiplier] }
                };
            let mut reference_trace = CpuTrace::new();
            let mut product_trace = CpuTrace::new();
            let before_noop = reference_aux.clone();
            reference = params.update_proof_traced_with_verifier(&instance, &instance, &old_reference,
                &update, &reference, &mut reference_aux, VerificationMode::Reference, &mut reference_trace).unwrap();
            product = params.update_proof_traced_with_verifier(&instance, &instance, &old_product,
                &update, &product, &mut product_aux, VerificationMode::Product, &mut product_trace).unwrap();
            assert_eq!(product, reference);
            assert_eq!(product_aux, reference_aux);
            if step == 2 { assert!(product.delta.is_none()); }
            if step == 3 {
                assert_eq!(reference_aux, before_noop);
                assert_eq!(product_trace.verification().counts().equations, 0);
            } else {
                assert_eq!(product_trace.verification().counts().pairing_terms,
                    reference_trace.verification().counts().pairing_terms);
                assert!(product_trace.verification().counts().final_exponentiations
                    < reference_trace.verification().counts().final_exponentiations);
            }
        }
        let old = product_aux.current_witness().clone();
        let invalid = Update { indices: vec![5 * n + 1], values: vec![old.wires[5 * n + 1] + Fr::from(1u64)] };
        let snapshot = product_aux.clone();
        assert!(params.update_proof_traced_with_verifier(&instance, &instance, &old, &invalid,
            &product, &mut product_aux, VerificationMode::Product, &mut CpuTrace::new()).is_err());
        assert_eq!(product_aux, snapshot);
    }
}
