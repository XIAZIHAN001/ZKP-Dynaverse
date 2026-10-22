use ark_bn254::Fr;
use dynaverse_protocol::dynaverse::DynaverseParams;
use dynaverse_protocol::toy_circuit::ToyCircuit;
use dynaverse_protocol::trace::{BatchOpenPhase, CpuPhase, CpuTrace};
use dynaverse_protocol::{AnchorOpeningPolicy, Update};
use rand::{rngs::StdRng, SeedableRng};
use std::time::{Duration, Instant};

#[test]
fn dense_and_cached_paths_produce_identical_proofs_through_rebuild() {
    let n = 16;
    let circuit = ToyCircuit::new(n).unwrap();
    let (instance, witness) = circuit.valid_instance();
    let mut rng = StdRng::seed_from_u64(2810);
    let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
    let mut cache_trace = CpuTrace::new();
    let started = Instant::now();
    let (mut cached, mut cached_aux) = params.prove_with_aux_traced(
        &instance, &witness, AnchorOpeningPolicy::Precompute, &mut cache_trace,
    ).unwrap();
    assert!(cache_trace.recorded_total() <= started.elapsed());
    assert!(cache_trace.elapsed(CpuPhase::CachePrecompute) > Duration::ZERO);
    let mut dense_trace = CpuTrace::new();
    let (mut dense, mut dense_aux) = params.prove_with_aux_traced(
        &instance, &witness, AnchorOpeningPolicy::Dense, &mut dense_trace,
    ).unwrap();
    assert_eq!(cached, dense);
    assert_eq!(dense_trace.elapsed(CpuPhase::CachePrecompute), Duration::ZERO);
    assert_eq!(cached_aux.cached_opening_points(), 3 * n);
    assert_eq!(dense_aux.cached_opening_points(), 0);

    let updates = [
        Update { indices: vec![4 * n], values: vec![witness.wires[4 * n] + Fr::from(1u64)] },
        Update { indices: vec![4 * n + 1, 5 * n + 1], values: vec![
            witness.wires[4 * n + 1] + Fr::from(1u64),
            witness.wires[5 * n + 1] + witness.wires[3 * n + 1],
        ] },
        Update { indices: vec![4 * n + 2, 5 * n + 2], values: vec![
            witness.wires[4 * n + 2] + Fr::from(1u64),
            witness.wires[5 * n + 2] + witness.wires[3 * n + 2],
        ] },
        Update { indices: vec![4 * n], values: vec![witness.wires[4 * n] + Fr::from(2u64)] },
    ];
    for (step, update) in updates.iter().enumerate() {
        // Own the old witness snapshot before borrowing aux mutably for update.
        let old_cached = cached_aux.current_witness().clone();
        let old_dense = dense_aux.current_witness().clone();
        cache_trace = CpuTrace::new();
        dense_trace = CpuTrace::new();
        let started = Instant::now();
        cached = params.update_proof_traced(&instance, &instance, &old_cached, update,
            &cached, &mut cached_aux, &mut cache_trace).unwrap();
        assert!(cache_trace.recorded_total() <= started.elapsed());
        dense = params.update_proof_traced(&instance, &instance, &old_dense, update,
            &dense, &mut dense_aux, &mut dense_trace).unwrap();
        assert_eq!(cached, dense, "policy mismatch at step {step}");
        assert!(params.verifying_key().verify(&instance, &cached).unwrap());
        assert_eq!(dense_aux.cached_opening_points(), 0);
        assert_eq!(dense_trace.elapsed(CpuPhase::CachePrecompute), Duration::ZERO);
        assert!(cache_trace.elapsed(CpuPhase::PairingVerify) > Duration::ZERO);
        if step == 2 {
            assert!(cached.delta.is_none());
            assert!(cache_trace.elapsed(CpuPhase::AnchorRebuild) > Duration::ZERO);
            assert!(cache_trace.elapsed(CpuPhase::CachePrecompute) > Duration::ZERO);
        } else {
            assert!(cached.delta.is_some());
            // Nested details must stay inside their parent, not be added to
            // the exclusive operation total a second time.
            assert!(cache_trace.batch_open().recorded_total() <= cache_trace.elapsed(CpuPhase::BatchOpen));
            assert!(dense_trace.batch_open().recorded_total() <= dense_trace.elapsed(CpuPhase::BatchOpen));
            for phase in BatchOpenPhase::ALL {
                assert!(dense_trace.batch_open().elapsed(phase) > Duration::ZERO,
                    "missing batch detail {}", phase.name());
            }
            assert!(cache_trace.elapsed(CpuPhase::AnchorOpenings) > Duration::ZERO);
            assert_eq!(cache_trace.elapsed(CpuPhase::CachePrecompute), Duration::ZERO);
        }
    }
}

#[test]
fn traced_dense_updates_preserve_noop_and_reject_invalid_state() {
    let n = 8;
    let circuit = ToyCircuit::new(n).unwrap();
    let (instance, witness) = circuit.valid_instance();
    let mut rng = StdRng::seed_from_u64(2811);
    let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
    let (proof, mut aux) = params.prove_with_aux_policy(
        &instance, &witness, AnchorOpeningPolicy::Dense,
    ).unwrap();
    let snapshot = aux.clone();
    let mut trace = CpuTrace::new();
    let same = params.update_proof_traced(&instance, &instance, &witness,
        &Update { indices: vec![], values: vec![] }, &proof, &mut aux, &mut trace).unwrap();
    assert_eq!(same, proof);
    assert_eq!(aux, snapshot);
    assert_eq!(trace.elapsed(CpuPhase::AnchorOpenings), Duration::ZERO);
    assert_eq!(trace.batch_open().recorded_total(), Duration::ZERO);
    let wrong = Update { indices: vec![5 * n + 1], values: vec![Fr::from(99u64)] };
    assert!(params.update_proof_traced(&instance, &instance, &witness,
        &wrong, &proof, &mut aux, &mut CpuTrace::new()).is_err());
    assert_eq!(aux, snapshot);
}
