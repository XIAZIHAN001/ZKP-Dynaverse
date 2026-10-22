use ark_bn254::Fr;
use dynaverse_protocol::{dynaverse::DynaverseParams,toy_circuit::ToyCircuit,Update,
    incremental_rebuild::{IncrementalConfig,operators::CpuRebuildBackend}};
use rand::{rngs::StdRng,SeedableRng};
#[path="support/linear_fault.rs"]
mod linear_fault;

fn multiplier(w:&dynaverse_protocol::Witness,n:usize,i:usize)->Update {
    let next=w.wires[4*n+i]+Fr::from(1u64);
    Update {indices:vec![4*n+i,5*n+i],values:vec![next,w.wires[3*n+i]*next]}
}

#[test]
fn live_fixed_fields_match_every_reference_proof_with_fewer_cumulative_msm_inputs() {
    let n=256;let circuit=ToyCircuit::new(n).unwrap();let (instance,witness)=circuit.valid_instance();
    let p=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(18700)).unwrap();
    let base=IncrementalConfig {linear_rebuild_reuse:true,unchanged_cache_reuse:true,..Default::default()};
    let mut old=p.start_incremental_session(&instance,&witness,base.clone(),CpuRebuildBackend::default()).unwrap();
    let mut live=p.start_incremental_session(&instance,&witness,
        IncrementalConfig {foreground_linear_reuse:true,..base},CpuRebuildBackend::default()).unwrap();
    for step in 0..10 {
        let i=1+step%7;let update=multiplier(live.current_witness(),n,i);
        assert_eq!(live.update(&update).unwrap(),old.update(&update).unwrap(),"step={step}");
        assert_eq!(live.rebuild_progress(),old.rebuild_progress());
        assert!(live.last_receipt().foreground_linear_reused);
        assert_eq!(live.last_receipt().foreground_linear_inputs,2);
        assert!(p.verifying_key().verify_reference(&instance,live.proof()).unwrap());
    }
    let a=old.sparse_dispatch_stats();let b=live.sparse_dispatch_stats();
    assert!(b.cpu.g1_points<a.cpu.g1_points,"live state still recomputed all cumulative fixed fields");
}

#[test]
fn live_state_handles_public_sum_all_wires_revert_noop_and_paid_large_rebuild() {
    let n=16;let circuit=ToyCircuit::new(n).unwrap();let (instance,witness)=circuit.valid_instance();
    let p=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(18701)).unwrap();
    let base=IncrementalConfig {linear_rebuild_reuse:true,unchanged_cache_reuse:true,trace_updates:true,..Default::default()};
    let mut old=p.start_incremental_session(&instance,&witness,base.clone(),CpuRebuildBackend::default()).unwrap();
    let mut live=p.start_incremental_session(&instance,&witness,
        IncrementalConfig {foreground_linear_reuse:true,..base},CpuRebuildBackend::default()).unwrap();
    for step in 0..8 {
        let i=1;let w=live.current_witness();
        let (u,public)=match step%4 {
            0=>{let left=w.wires[i]+Fr::from(1u64);let right=w.wires[n+i]-Fr::from(1u64);
                (Update {indices:vec![i,n+i],values:vec![left,right]},Update {indices:vec![i],values:vec![left]})},
            1=>(multiplier(w,n,i),Update {indices:vec![],values:vec![]}),
            2=>{let right=w.wires[n+i]+Fr::from(1u64);let sum=w.wires[i]+right;
                (Update {indices:vec![n+i,2*n+i,3*n+i,5*n+i],values:vec![right,sum,sum,sum*w.wires[4*n+i]]},Update {indices:vec![],values:vec![]})},
            _=>{let value=w.wires[4*n];
                (Update {indices:vec![4*n,4*n],values:vec![value+Fr::from(1u64),value]},Update {indices:vec![],values:vec![]})}
        };
        assert_eq!(live.update_with_public_inputs(&u,&public).unwrap(),old.update_with_public_inputs(&u,&public).unwrap());
        assert!(p.verifying_key().verify_reference(live.current_instance(),live.proof()).unwrap());
        if step%4==3 {assert_eq!(live.last_receipt().foreground_linear_inputs,0);}
        if live.last_receipt().large_batch {assert_eq!(live.last_receipt().foreground_linear_inputs,0);}
    }
    let receipt=*live.last_receipt();let timing=*live.update_timings();let proof=live.proof().clone();
    assert!(live.update(&Update {indices:vec![0],values:vec![Fr::from(999u64)]}).is_err());
    assert_eq!(live.proof(),&proof);assert_eq!(*live.last_receipt(),receipt);assert_eq!(*live.update_timings(),timing);
}

#[test]
fn failure_after_live_work_keeps_absolute_state_for_exact_retry() {
    let n=64;let circuit=ToyCircuit::new(n).unwrap();let (instance,witness)=circuit.valid_instance();
    let p=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(18702)).unwrap();
    let (backend,control)=linear_fault::FaultBackend::new();
    let config=IncrementalConfig {foreground_linear_reuse:true,trace_updates:true,..Default::default()};
    let mut live=p.start_incremental_session(&instance,&witness,config.clone(),backend).unwrap();
    let mut reference=p.start_incremental_session(&instance,&witness,config,CpuRebuildBackend::default()).unwrap();
    let before=live.proof().clone();let receipt=*live.last_receipt();let timing=*live.update_timings();
    let u=multiplier(&witness,n,1);control.arm_g2();
    assert!(live.update(&u).is_err());
    assert_eq!(live.proof(),&before);assert_eq!(*live.last_receipt(),receipt);assert_eq!(*live.update_timings(),timing);
    assert_eq!(live.current_witness(),&witness);assert!(live.rebuild_progress().is_none());assert!(!live.is_faulted());
    assert_eq!(live.update(&u).unwrap(),reference.update(&u).unwrap());
    for i in 2..6 {let u=multiplier(live.current_witness(),n,i);
        assert_eq!(live.update(&u).unwrap(),reference.update(&u).unwrap());}
}

#[test]
fn same_affected_support_reuses_geometry_but_recomputes_changed_values() {
    let n=256;let circuit=ToyCircuit::new(n).unwrap();let (instance,witness)=circuit.valid_instance();
    let p=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(18703)).unwrap();
    let base=IncrementalConfig {foreground_linear_reuse:true,..Default::default()};
    let mut old=p.start_incremental_session(&instance,&witness,base.clone(),CpuRebuildBackend::default()).unwrap();
    let mut reuse=p.start_incremental_session(&instance,&witness,
        IncrementalConfig {support_geometry_reuse:true,..base},CpuRebuildBackend::default()).unwrap();
    let mut initial=Update {indices:vec![],values:vec![]};
    for i in 1..=3 {let u=multiplier(&witness,n,i);initial.indices.extend(u.indices);initial.values.extend(u.values);}
    assert_eq!(reuse.update(&initial).unwrap(),old.update(&initial).unwrap());
    assert_eq!(reuse.last_receipt().support_geometry_builds,1);
    let mut hits=0;
    for step in 0..6 {
        let u=multiplier(reuse.current_witness(),n,1);
        assert_eq!(reuse.update(&u).unwrap(),old.update(&u).unwrap());
        assert!(reuse.last_receipt().support_geometry_reused,"step={step}");hits+=1;
        assert!(p.verifying_key().verify_reference(&instance,reuse.proof()).unwrap());
    }
    assert!(hits>0);
    assert!(reuse.sparse_dispatch_stats().cpu.ntt_elements<old.sparse_dispatch_stats().cpu.ntt_elements);
    let u=multiplier(reuse.current_witness(),n,4);
    assert_eq!(reuse.update(&u).unwrap(),old.update(&u).unwrap());
    assert_eq!(reuse.last_receipt().support_geometry_builds,1);
    assert!(!reuse.last_receipt().support_geometry_reused);
}

#[test]
fn geometry_rolls_back_on_failure_and_changes_key_after_anchor_switch() {
    let n=64;let circuit=ToyCircuit::new(n).unwrap();let (instance,witness)=circuit.valid_instance();
    let p=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(18704)).unwrap();
    let config=IncrementalConfig {foreground_linear_reuse:true,support_geometry_reuse:true,..Default::default()};
    let (backend,control)=linear_fault::FaultBackend::new();
    let mut reuse=p.start_incremental_session(&instance,&witness,config.clone(),backend).unwrap();
    let mut old=p.start_incremental_session(&instance,&witness,config,CpuRebuildBackend::default()).unwrap();
    let first=multiplier(&witness,n,1);
    assert_eq!(reuse.update(&first).unwrap(),old.update(&first).unwrap());
    let before=reuse.proof().clone();let receipt=*reuse.last_receipt();
    let u=multiplier(reuse.current_witness(),n,2);control.arm_g2();
    assert!(reuse.update(&u).is_err());assert_eq!(reuse.proof(),&before);assert_eq!(*reuse.last_receipt(),receipt);
    assert_eq!(reuse.update(&u).unwrap(),old.update(&u).unwrap());
    for i in 3..8 {let u=multiplier(reuse.current_witness(),n,i);
        assert_eq!(reuse.update(&u).unwrap(),old.update(&u).unwrap());
        assert!(p.verifying_key().verify_reference(&instance,reuse.proof()).unwrap());}
}

#[test]
fn row_batch_four_preserves_changed_cache_proof_and_element_work_with_fewer_calls() {
    use dynaverse_protocol::incremental_rebuild::operators::RebuildBackend;
    let n=64;let circuit=ToyCircuit::new(n).unwrap();let (instance,witness)=circuit.valid_instance();
    let p=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(18705)).unwrap();
    let cfg=IncrementalConfig {linear_rebuild_reuse:true,unchanged_cache_reuse:true,
        foreground_linear_reuse:true,support_geometry_reuse:true,..Default::default()};
    let mut one=p.start_incremental_session(&instance,&witness,cfg.clone(),CpuRebuildBackend::default()).unwrap();
    let mut four=p.start_incremental_session(&instance,&witness,
        IncrementalConfig {transform_row_batch:4,..cfg},CpuRebuildBackend::default()).unwrap();
    let a=one.backend().stats();let b=four.backend().stats();
    let mut u=Update {indices:vec![],values:vec![]};
    for i in 1..=3 {let part=multiplier(&witness,n,i);u.indices.extend(part.indices);u.values.extend(part.values);}
    assert_eq!(four.update(&u).unwrap(),one.update(&u).unwrap());
    assert_eq!(four.proof().anchor,p.prove_anchor(&instance,four.current_witness()).unwrap().anchor);
    let aa=one.backend().stats();let bb=four.backend().stats();
    assert_eq!(aa.g1_transform_elements-a.g1_transform_elements,bb.g1_transform_elements-b.g1_transform_elements);
    assert_eq!(aa.fr_transform_elements-a.fr_transform_elements,bb.fr_transform_elements-b.fr_transform_elements);
    assert!(bb.calls-b.calls<aa.calls-a.calls,"row batching did not reduce resident calls");
    assert!(four.last_receipt().rebuild_units<=four.last_receipt().allowance);
    for i in 4..8 {let u=multiplier(four.current_witness(),n,i);four.update(&u).unwrap();
        assert!(p.verifying_key().verify_reference(&instance,four.proof()).unwrap());}
}

#[test]
fn invalid_row_batch_is_rejected_during_initialization() {
    let circuit=ToyCircuit::new(16).unwrap();let (i,w)=circuit.valid_instance();
    let p=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(18706)).unwrap();
    assert_eq!(IncrementalConfig::default().transform_row_batch,1);
    for batch in [0,17] {
        assert!(p.start_incremental_session(&i,&w,IncrementalConfig {transform_row_batch:batch,..Default::default()},CpuRebuildBackend::default()).is_err());
    }
}

#[test]
fn empty_affected_geometry_is_safe_for_repeated_public_only_wire_edits() {
    let n=64;let circuit=ToyCircuit::new(n).unwrap();let (instance,witness)=circuit.valid_instance();
    let p=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(18707)).unwrap();
    for linear in [false,true] {
        let cfg=IncrementalConfig {foreground_linear_reuse:linear,..Default::default()};
        let mut old=p.start_incremental_session(&instance,&witness,cfg.clone(),CpuRebuildBackend::default()).unwrap();
        let mut reuse=p.start_incremental_session(&instance,&witness,
            IncrementalConfig {support_geometry_reuse:true,..cfg},CpuRebuildBackend::default()).unwrap();
        for step in 0..4 {
            let left=reuse.current_witness().wires[1]+Fr::from(1u64);
            let right=reuse.current_witness().wires[n+1]-Fr::from(1u64);
            let u=Update {indices:vec![1,n+1],values:vec![left,right]};let public=Update {indices:vec![1],values:vec![left]};
            assert_eq!(reuse.update_with_public_inputs(&u,&public).unwrap(),old.update_with_public_inputs(&u,&public).unwrap());
            if step>0 {assert!(reuse.last_receipt().support_geometry_reused);}
            assert!(p.verifying_key().verify_reference(reuse.current_instance(),reuse.proof()).unwrap());
        }
    }
}
