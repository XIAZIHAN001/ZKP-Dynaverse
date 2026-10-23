use ark_bn254::Fr;
use dynaverse_protocol::{Update,Witness,VerificationMode,dynaverse::DynaverseParams,toy_circuit::ToyCircuit,
    incremental_rebuild::{IncrementalConfig,operators::{CpuRebuildBackend,RebuildBackend}}};
use rand::{rngs::StdRng,SeedableRng};
#[allow(dead_code)]
#[path="support/linear_fault.rs"]
mod fault;

fn edits(w:&Witness,n:usize)->Update {
    let mut u=Update {indices:Vec::new(),values:Vec::new()};
    for i in 1..=3 {
        let v=w.wires[4*n+i]+Fr::from(3u64);
        u.indices.extend([4*n+i,5*n+i]);u.values.extend([v,w.wires[3*n+i]*v]);
    }
    u
}

#[test]
fn full_divisor_late_cache_corruption_restores_entire_session_and_retries_exactly() {
    let n=16;let circuit=ToyCircuit::new(n).unwrap();let(instance,witness)=circuit.valid_instance();
    let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(27210)).unwrap();
    let config=IncrementalConfig {window:Some(4),trace_updates:true,rebuild_divisor_cache:true,
        rebuild_cache_mode:VerificationMode::Product,rebuild_prepared_cache_roots:Some(3),
        rebuild_cache_readback_roots:Some(8),unchanged_cache_reuse:true,..Default::default()};
    let (backend,control)=fault::FaultBackend::new();
    let mut s=params.start_incremental_session(&instance,&witness,config.clone(),backend).unwrap();
    let mut clean=params.start_incremental_session(&instance,&witness,config,CpuRebuildBackend::default()).unwrap();
    let warm=edits(&witness,n);s.update(&warm).unwrap();clean.update(&warm).unwrap();
    let update=edits(s.current_witness(),n);
    let proof=s.proof().clone();let witness=s.current_witness().clone();
    let receipt=*s.last_receipt();let timing=*s.update_timings();let blocks=*s.block_timings();
    let progress=s.rebuild_progress();let allocation=s.backend().stats().bytes_allocated;
    let table=s.cache_divisor_stats();let initial=s.initial_cache_stats();
    control.arm_corrupt_cache_after(1);
    let error=s.update(&update).unwrap_err();
    assert!(error.to_string().contains("opening cache check failed"));
    assert_eq!(control.cache_reads(),2,"one valid cache block must precede the corrupted block");
    assert_eq!(s.proof(),&proof);assert_eq!(s.current_witness(),&witness);
    assert_eq!(*s.last_receipt(),receipt);assert_eq!(*s.update_timings(),timing);
    assert_eq!(*s.block_timings(),blocks);assert_eq!(s.rebuild_progress(),progress);
    assert_eq!(s.backend().stats().bytes_allocated,allocation);assert!(!s.is_faulted());
    assert_eq!(s.cache_divisor_stats(),table);assert_eq!(s.initial_cache_stats(),initial);
    s.update(&update).unwrap();clean.update(&update).unwrap();
    assert_eq!(s.proof(),clean.proof());assert_eq!(s.current_witness(),clean.current_witness());
    assert_eq!(s.last_receipt(),clean.last_receipt());
    assert_eq!(s.update_timings().cache_detail.divisors_formed,0);
    assert_eq!(s.update_timings().cache_detail.equations,32);
    assert!(params.verifying_key().verify_reference(&instance,s.proof()).unwrap());
    assert_eq!(s.proof().anchor,params.prove_anchor(&instance,s.current_witness()).unwrap().anchor);
}
