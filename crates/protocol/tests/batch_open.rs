use ark_bn254::Fr;
use dynaverse_protocol::{AnchorOpeningPolicy, BatchOpenMode, Update, VerificationMode};
use dynaverse_protocol::{dynaverse::DynaverseParams, toy_circuit::ToyCircuit};
use dynaverse_protocol::trace::{CpuPhase, CpuTrace};
use rand::{rngs::StdRng, SeedableRng};

#[test]
fn prepared_public_division_preserves_proofs_and_aux_through_rebuild() {
    let n = 16;
    let circuit = ToyCircuit::new(n).unwrap();
    let (instance, witness) = circuit.valid_instance();
    let mut rng = StdRng::seed_from_u64(2902);
    let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
    let vk = params.verifying_key();
    for policy in [AnchorOpeningPolicy::Dense, AnchorOpeningPolicy::Precompute] {
        let (mut reference, mut reference_aux) = params.prove_with_aux_policy(&instance, &witness, policy).unwrap();
        let mut prepared = reference.clone();
        let mut prepared_aux = reference_aux.clone();
        for gate in [0usize, 0, 1, 2, 3, 4] {
            let old = reference_aux.current_witness().clone();
            let multiplier = 4 * n + gate;
            let product = 5 * n + gate;
            let value = old.wires[multiplier] + Fr::from(1u64);
            let update = Update { indices: vec![multiplier, product], values: vec![value, old.wires[3 * n + gate] * value] };
            reference = params.update_proof_traced_with_modes(&instance, &instance, &old, &update,
                &reference, &mut reference_aux, VerificationMode::Reference, BatchOpenMode::Reference, &mut CpuTrace::new()).unwrap();
            let mut trace = CpuTrace::new();
            prepared = params.update_proof_traced_with_modes(&instance, &instance, &old, &update,
                &prepared, &mut prepared_aux, VerificationMode::Product, BatchOpenMode::Prepared, &mut trace).unwrap();
            assert_eq!(reference, prepared);
            assert_eq!(reference_aux, prepared_aux);
            assert!(vk.verify_reference(&instance, &prepared).unwrap());
            assert!(trace.batch_open().recorded_total() <= trace.elapsed(CpuPhase::BatchOpen));
        }
        for mode in [BatchOpenMode::Reference, BatchOpenMode::Prepared] {
            let old = prepared_aux.current_witness().clone();
            let snapshot = prepared_aux.clone();
            let unchanged = params.update_proof_traced_with_modes(&instance, &instance, &old,
                &Update { indices: vec![], values: vec![] }, &prepared, &mut prepared_aux,
                VerificationMode::Product, mode, &mut CpuTrace::new()).unwrap();
            assert_eq!(unchanged, prepared);
            assert_eq!(prepared_aux, snapshot);
            let invalid = Update { indices: vec![5 * n + 1], values: vec![Fr::from(99u64)] };
            assert!(params.update_proof_traced_with_modes(&instance, &instance, &old, &invalid,
                &prepared, &mut prepared_aux, VerificationMode::Product, mode, &mut CpuTrace::new()).is_err());
            assert_eq!(prepared_aux, snapshot);
        }
    }
}
