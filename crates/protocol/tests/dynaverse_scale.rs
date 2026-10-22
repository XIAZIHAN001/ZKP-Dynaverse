use ark_bn254::Fr;
use dynaverse_protocol::{
    dynaverse::DynaverseParams, toy_circuit::ToyCircuit, BatchOpenMode, Update, VerificationMode,
};
use dynaverse_protocol::trace::CpuTrace;
use rand::{rngs::StdRng, SeedableRng};

/// Explicit scale gate: setup uses substantial memory and is kept out of the
/// fast workspace regression suite.
#[test]
#[ignore = "run explicitly for the CPU n=256/512/1024 scale gate"]
fn cpu_anchor_and_one_wire_update_at_three_scale_sizes() {
    for n in [256usize, 512, 1024] {
        eprintln!("SCALE_SETUP n={n}");
        let circuit = ToyCircuit::new(n).unwrap();
        let (instance, witness) = circuit.valid_instance();
        let mut rng = StdRng::seed_from_u64(2000 + n as u64);
        let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
        let vk = params.verifying_key();
        let (proof, mut aux) = params.prove_with_aux(&instance, &witness).unwrap();
        assert!(vk.verify(&instance, &proof).unwrap(), "anchor n={n}");
        assert!(vk.verify_reference(&instance, &proof).unwrap(), "reference anchor n={n}");
        let update = Update {
            indices: vec![4 * n],
            values: vec![witness.wires[4 * n] + Fr::from(1u64)],
        };
        let mut reference_aux = aux.clone();
        let original = params.update_proof_traced_with_modes(
            &instance, &instance, &witness, &update, &proof, &mut reference_aux,
            VerificationMode::Product, BatchOpenMode::Reference, &mut CpuTrace::new(),
        ).unwrap();
        let changed = params.update_proof(
            &instance, &instance, &witness, &update, &proof, &mut aux
        ).unwrap();
        assert_eq!(changed, original, "prepared/reference proof n={n}");
        assert_eq!(aux, reference_aux, "prepared/reference aux n={n}");
        assert!(vk.verify(&instance, &changed).unwrap(), "delta n={n}");
        assert!(vk.verify_reference(&instance, &changed).unwrap(), "reference delta n={n}");
        eprintln!("SCALE_OK n={n}");
    }
}
