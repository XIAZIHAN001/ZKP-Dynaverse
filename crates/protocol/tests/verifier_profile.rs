use ark_bn254::Fr;
use dynaverse_protocol::dynaverse::DynaverseParams;
use dynaverse_protocol::toy_circuit::ToyCircuit;
use dynaverse_protocol::trace::{VerificationPhase, VerificationTrace};
use dynaverse_protocol::{AnchorOpeningPolicy, Update};
use rand::{rngs::StdRng, SeedableRng};
use std::time::Instant;

#[test]
fn reference_profile_counts_anchor_and_delta_equations_without_overlap() {
    let n = 16;
    let circuit = ToyCircuit::new(n).unwrap();
    let (instance, witness) = circuit.valid_instance();
    let mut rng = StdRng::seed_from_u64(2901);
    let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
    let vk = params.verifying_key();
    let (proof, mut aux) = params.prove_with_aux_policy(
        &instance, &witness, AnchorOpeningPolicy::Dense,
    ).unwrap();
    let mut trace = VerificationTrace::new();
    let started = Instant::now();
    assert!(vk.verify_reference_traced(&instance, &proof, &mut trace).unwrap());
    assert!(trace.recorded_total() <= started.elapsed());
    let totals = trace.counts();
    assert_eq!(totals.equations, 37);
    assert_eq!(totals.pairing_terms, 84);
    assert_eq!(totals.miller_loops, 84);
    assert_eq!(totals.final_exponentiations, 84);
    assert_eq!(trace.phase(VerificationPhase::AnchorConsistency).counts.equations, 30);
    assert_eq!(trace.phase(VerificationPhase::DeltaConsistency).counts.equations, 0);

    let update = Update { indices: vec![4 * n], values: vec![witness.wires[4 * n] + Fr::from(1u64)] };
    let changed = params.update_proof(&instance, &instance, &witness,
        &update, &proof, &mut aux).unwrap();
    trace = VerificationTrace::new();
    let started = Instant::now();
    assert!(vk.verify_reference_traced(&instance, &changed, &mut trace).unwrap());
    assert!(trace.recorded_total() <= started.elapsed());
    let totals = trace.counts();
    assert_eq!(totals.equations, 86);
    assert_eq!(totals.pairing_terms, 192);
    assert_eq!(totals.miller_loops, 192);
    assert_eq!(totals.final_exponentiations, 192);
    for (phase, equations, terms) in [
        (VerificationPhase::AnchorDynamo, 4, 11),
        (VerificationPhase::AnchorConsistency, 30, 66),
        (VerificationPhase::AnchorRelations, 3, 7),
        (VerificationPhase::DeltaDynamo, 4, 11),
        (VerificationPhase::DeltaConsistency, 30, 66),
        (VerificationPhase::DeltaDivisors, 4, 8),
        (VerificationPhase::DeltaMultiplication, 2, 5),
        (VerificationPhase::DeltaOpenings, 9, 18),
    ] {
        let measured = trace.phase(phase);
        assert_eq!(measured.counts.equations, equations);
        assert_eq!(measured.counts.pairing_terms, terms);
        assert!(measured.miller_time + measured.final_exponentiation_time <= measured.elapsed);
    }
}

#[test]
fn reference_profile_rejects_invalid_shapes_before_pairing() {
    let circuit = ToyCircuit::new(8).unwrap();
    let (mut instance, witness) = circuit.valid_instance();
    let mut rng = StdRng::seed_from_u64(2902);
    let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
    let proof = params.prove_anchor(&instance, &witness).unwrap();
    instance.public_inputs.pop();
    let mut trace = VerificationTrace::new();
    assert!(!params.verifying_key().verify_reference_traced(&instance, &proof, &mut trace).unwrap());
    assert_eq!(trace.counts().pairing_terms, 0);
}
