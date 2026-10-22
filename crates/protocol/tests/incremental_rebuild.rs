use ark_bn254::Fr;
use ark_ff::One;
use dynaverse_protocol::{dynaverse::DynaverseParams, toy_circuit::ToyCircuit, Update};
use dynaverse_protocol::incremental_rebuild::{IncrementalConfig, SparseDispatchPolicy, operators::CpuRebuildBackend};
use rand::{SeedableRng, rngs::StdRng};

#[test]
fn stepped_session_rebuilds_and_matches_public_verifiers() {
    let n = 16;
    let mut rng = StdRng::seed_from_u64(5001);
    let circuit = ToyCircuit::new(n).unwrap();
    let (instance, witness) = circuit.valid_instance();
    let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
    let vk = params.verifying_key();
    let mut session = params.start_incremental_session(
        &instance, &witness, IncrementalConfig::default(), CpuRebuildBackend::default(),
    ).unwrap();
    let mut switches = 0;
    for step in 0..12 {
        let i = step % n;
        let next = session.current_witness().wires[4*n+i] + Fr::one();
        let output = session.current_witness().wires[3*n+i] * next;
        let proof = session.update(&Update {
            indices: vec![4*n+i, 5*n+i], values: vec![next, output],
        }).unwrap();
        assert!(vk.verify(session.current_instance(), &proof).unwrap());
        assert!(vk.verify_reference(session.current_instance(), &proof).unwrap());
        let receipt = session.last_receipt();
        assert!(receipt.rebuild_units <= receipt.allowance);
        if receipt.swapped { switches += 1; }
        assert!(session.anchor_distance() <= 2*session.window());
    }
    assert!(switches > 0);
}

#[test]
fn force_gpu_and_resource_limits_fail_during_initialization() {
    let circuit=ToyCircuit::new(2).unwrap();
    let (instance,witness)=circuit.valid_instance();
    let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(5002)).unwrap();
    let force=IncrementalConfig {sparse:SparseDispatchPolicy::ForceGpu,..Default::default()};
    let result=params.start_incremental_session(&instance,&witness,force,CpuRebuildBackend::default());
    assert!(result.err().unwrap().to_string().contains("ForceGpu requires"));
    let too_small=IncrementalConfig {memory_limit_bytes:1024,..Default::default()};
    let result=params.start_incremental_session(&instance,&witness,too_small,CpuRebuildBackend::default());
    assert!(result.err().unwrap().to_string().contains("resource estimate"));
}
