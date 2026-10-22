use ark_bn254::Fr;
use dynaverse_protocol::{dynaverse::DynaverseParams, toy_circuit::ToyCircuit, Update,
    incremental_rebuild::{IncrementalConfig, operators::CpuRebuildBackend}};
use rand::{rngs::StdRng, SeedableRng};
use dynaverse_protocol::incremental_rebuild::operators::RebuildBackend;
#[path="support/linear_fault.rs"]
mod linear_fault;

#[test]
fn telemetry_distinguishes_submitted_changes_and_preserves_last_success_on_error() {
    let n=16;
    let circuit=ToyCircuit::new(n).unwrap();
    let (instance,witness)=circuit.valid_instance();
    let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(18001)).unwrap();
    let mut session=params.start_incremental_session(&instance,&witness,
        IncrementalConfig {trace_updates:true,..Default::default()},CpuRebuildBackend::default()).unwrap();
    let original=witness.wires[4*n];
    let proof=session.proof().clone();
    session.update(&Update {indices:vec![4*n;3],
        values:vec![original+Fr::from(2u64),original+Fr::from(1u64),original]}).unwrap();
    assert_eq!(session.last_receipt().raw_edits,3);
    assert_eq!(session.last_receipt().changed,0);
    assert_eq!(session.proof(),&proof);
    assert!(session.update_timings().total_ns>=session.update_timings().normalization_ns);
    session.update(&Update {indices:vec![4*n],values:vec![original+Fr::from(1u64)]}).unwrap();
    assert_eq!(session.last_receipt().changed,1);
    assert_eq!(session.last_receipt().affected_gates,1);
    assert_eq!(session.last_receipt().delta_entries,1);
    assert_eq!(session.last_receipt().changed_wire_mask,1<<4);
    let receipt=*session.last_receipt();
    let timings=*session.update_timings();
    assert!(session.update(&Update {indices:vec![0],values:vec![Fr::from(77u64)]}).is_err());
    assert_eq!(*session.last_receipt(),receipt);
    assert_eq!(*session.update_timings(),timings);
}

#[test]
fn enabled_update_timing_separates_outer_wall_from_nested_rebuild_work() {
    let n=16;
    let circuit=ToyCircuit::new(n).unwrap();
    let (instance,witness)=circuit.valid_instance();
    let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(18002)).unwrap();
    let mut session=params.start_incremental_session(&instance,&witness,
        IncrementalConfig {trace_updates:true,..Default::default()},CpuRebuildBackend::default()).unwrap();
    let mut blocks=0;
    for step in 0..4 {
        let i=step%n;
        let next=session.current_witness().wires[4*n+i]+Fr::from(1u64);
        let output=session.current_witness().wires[3*n+i]*next;
        session.update(&Update {indices:vec![4*n+i,5*n+i],values:vec![next,output]}).unwrap();
        let t=session.update_timings();
        assert!(t.total_ns>=t.normalization_ns+t.constraints_ns+t.preparation_ns+t.publication_ns);
        assert!(t.preparation_ns>=t.delta_proof_ns+t.rebase_ns+t.rebuild_ns.iter().sum::<u128>());
        blocks+=t.rebuild_calls.iter().sum::<u64>();
        assert_eq!(session.last_receipt().raw_edits,2);
        assert!(params.verifying_key().verify_reference(session.current_instance(),session.proof()).unwrap());
    }
    assert!(blocks>0);
}

#[test]
fn frozen_sparse_correction_matches_dense_anchors_and_reduces_resident_msm_inputs() {
    let n=16;
    let circuit=ToyCircuit::new(n).unwrap();
    let (instance,witness)=circuit.valid_instance();
    let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(18003)).unwrap();
    let mut dense=params.start_incremental_session(&instance,&witness,
        IncrementalConfig::default(),CpuRebuildBackend::default()).unwrap();
    let mut reuse=params.start_incremental_session(&instance,&witness,
        IncrementalConfig {linear_rebuild_reuse:true,trace_updates:true,..Default::default()},
        CpuRebuildBackend::default()).unwrap();
    let mut history=vec![witness];
    let mut corrected=0;
    for step in 0..8 {
        let i=step%n;
        let next=reuse.current_witness().wires[4*n+i]+Fr::from(1u64);
        let output=reuse.current_witness().wires[3*n+i]*next;
        let update=Update {indices:vec![4*n+i,5*n+i],values:vec![next,output]};
        dense.update(&update).unwrap();
        reuse.update(&update).unwrap();
        history.push(reuse.current_witness().clone());
        assert_eq!(reuse.current_witness(),dense.current_witness());
        let epoch=reuse.proof().anchor_generation as usize;
        let expected=params.prove_anchor(&instance,&history[epoch]).unwrap();
        assert_eq!(reuse.proof().anchor,expected.anchor);
        assert!(params.verifying_key().verify_reference(&instance,reuse.proof()).unwrap());
        assert!(reuse.last_receipt().rebuild_units<=reuse.last_receipt().allowance);
        if reuse.last_receipt().linear_reused {
            corrected+=1;
            assert!(reuse.last_receipt().linear_skipped_g1_points>0);
        }
    }
    assert!(corrected>0,"frozen sparse linear replacement was never used");
    assert!(reuse.backend().stats().msm_g1_points<dense.backend().stats().msm_g1_points);
    assert!(reuse.backend().stats().msm_g2_points<dense.backend().stats().msm_g2_points);
}

#[test]
fn over_window_frozen_delta_keeps_dense_fallback_and_exact_proof() {
    let n=16;
    let circuit=ToyCircuit::new(n).unwrap();
    let (instance,witness)=circuit.valid_instance();
    let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(18004)).unwrap();
    let mut session=params.start_incremental_session(&instance,&witness,
        IncrementalConfig {linear_rebuild_reuse:true,..Default::default()},CpuRebuildBackend::default()).unwrap();
    let mut indices=Vec::new();let mut values=Vec::new();
    for i in 0..8 {
        let next=witness.wires[4*n+i]+Fr::from(1u64);
        indices.extend([4*n+i,5*n+i]);values.extend([next,witness.wires[3*n+i]*next]);
    }
    session.update(&Update {indices,values}).unwrap();
    assert!(session.last_receipt().large_batch && session.last_receipt().swapped);
    assert!(!session.last_receipt().linear_reused);
    let expected=params.prove_anchor(&instance,session.current_witness()).unwrap();
    assert_eq!(session.proof().anchor,expected.anchor);
    assert!(params.verifying_key().verify_reference(&instance,session.proof()).unwrap());
}

#[test]
fn explicit_reference_policy_accounts_for_the_large_eligible_freeze_correction() {
    use dynaverse_protocol::incremental_rebuild::SparseDispatchPolicy;
    let n=16;let circuit=ToyCircuit::new(n).unwrap();
    let (instance,witness)=circuit.valid_instance();
    let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(18005)).unwrap();
    let mut session=params.start_incremental_session(&instance,&witness,
        IncrementalConfig {linear_rebuild_reuse:true,sparse:SparseDispatchPolicy::CpuReference,
            ..Default::default()},CpuRebuildBackend::default()).unwrap();
    let a=witness.wires[4*n]+Fr::from(1u64);
    let b=witness.wires[4*n+1]+Fr::from(1u64);
    session.update(&Update {indices:vec![4*n,4*n+1,5*n+1],
        values:vec![a,b,witness.wires[3*n+1]*b]}).unwrap();
    assert!(session.last_receipt().large_batch && session.last_receipt().linear_reused);
    let stats=session.sparse_dispatch_stats();
    assert!(stats.cpu.g1_points>0,"correction dispatch was omitted from session diagnostics");
    assert_eq!(stats.cpu.g2_points,2);
    assert_eq!(stats.gpu,Default::default());
}

#[test]
fn correction_first_middle_last_failure_rolls_back_fields_cursor_and_host_writes() {
    let n=16;let circuit=ToyCircuit::new(n).unwrap();
    let (instance,witness)=circuit.valid_instance();
    let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(18006)).unwrap();
    let a=witness.wires[4*n]+Fr::from(1u64);let b=witness.wires[4*n+1]+Fr::from(1u64);
    let update=Update {indices:vec![4*n,4*n+1,5*n+1],
        values:vec![a,b,witness.wires[3*n+1]*b]};
    // Nine G1 batches precede the optional final G2 correction.
    for at in [Some(0),Some(4),Some(8),None] {
        let (backend,control)=linear_fault::FaultBackend::new();
        let mut session=params.start_incremental_session(&instance,&witness,
            IncrementalConfig {linear_rebuild_reuse:true,trace_updates:true,..Default::default()},backend).unwrap();
        let before=session.proof().clone();let receipt=*session.last_receipt();
        let timings=*session.update_timings();
        let allocated=session.backend().stats().bytes_allocated;
        if let Some(at)=at {control.arm(at);} else {control.arm_g2();}
        assert!(session.update(&update).unwrap_err().to_string().contains("injected linear"));
        assert_eq!(session.proof(),&before);
        assert_eq!(session.current_witness(),&witness);
        assert_eq!(*session.last_receipt(),receipt);
        assert_eq!(*session.update_timings(),timings);
        assert!(session.rebuild_progress().is_none());
        assert!(!session.is_faulted());
        assert_eq!(session.backend().stats().bytes_allocated,allocated);
        session.update(&update).unwrap();
        assert!(session.last_receipt().linear_reused && session.last_receipt().swapped);
        // Receipt retains the processed job, although publication clears it.
        assert_eq!(session.last_receipt().frozen_entries,3);
        assert_eq!(session.last_receipt().delta_entries,0);
        assert_eq!(session.last_receipt().overlay_entries,0);
        assert!(session.rebuild_progress().is_none());
        let expected=params.prove_anchor(&instance,session.current_witness()).unwrap();
        assert_eq!(session.proof().anchor,expected.anchor);
        assert!(params.verifying_key().verify_reference(&instance,session.proof()).unwrap());
    }
}

#[test]
fn nonzero_sum_and_copy_slices_match_an_independent_dense_anchor() {
    let n=16;let circuit=ToyCircuit::new(n).unwrap();
    let (instance,witness)=circuit.valid_instance();
    let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(18008)).unwrap();
    let mut session=params.start_incremental_session(&instance,&witness,
        IncrementalConfig {linear_rebuild_reuse:true,..Default::default()},CpuRebuildBackend::default()).unwrap();
    let i=1;let right=witness.wires[n+i]+Fr::from(1u64);
    let sum=witness.wires[i]+right;
    session.update(&Update {indices:vec![n+i,2*n+i,3*n+i,5*n+i],
        values:vec![right,sum,sum,sum*witness.wires[4*n+i]]}).unwrap();
    assert!(session.last_receipt().linear_reused && session.last_receipt().swapped);
    assert_eq!(session.last_receipt().frozen_entries,4);
    assert_eq!(session.last_receipt().frozen_wire_mask,0b10_1110);
    let expected=params.prove_anchor(&instance,session.current_witness()).unwrap();
    assert_eq!(session.proof().anchor,expected.anchor);
    assert!(params.verifying_key().verify_reference(&instance,session.proof()).unwrap());
}

#[test]
fn failed_update_preserves_an_already_pending_epoch_and_its_timings() {
    let n=64;let circuit=ToyCircuit::new(n).unwrap();
    let (instance,witness)=circuit.valid_instance();
    let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(18009)).unwrap();
    let (backend,control)=linear_fault::FaultBackend::new();
    let mut session=params.start_incremental_session(&instance,&witness,
        // Use the dense reference schedule to exercise an existing epoch;
        // its smaller Cmax leaves this n=64 rebuild pending after two edits.
        IncrementalConfig {trace_updates:true,..Default::default()},backend).unwrap();
    // Each call pays two changes; the second call reaches H=4 and freezes.
    for i in 1..=2 {
        let next=witness.wires[4*n+i]+Fr::from(1u64);
        session.update(&Update {indices:vec![4*n+i,5*n+i],values:vec![next,witness.wires[3*n+i]*next]}).unwrap();
    }
    let progress=session.rebuild_progress();assert!(progress.is_some());
    let before=session.proof().clone();let old_witness=session.current_witness().clone();
    let receipt=*session.last_receipt();let timings=*session.update_timings();
    let allocated=session.backend().stats().bytes_allocated;
    let mut update=Update {indices:Vec::new(),values:Vec::new()};
    for i in [3,4] {
        let next=witness.wires[4*n+i]+Fr::from(1u64);
        update.indices.extend([4*n+i,5*n+i]);
        update.values.extend([next,witness.wires[3*n+i]*next]);
    }
    // This fault is in the next foreground sparse proof while an epoch exists.
    control.arm_g2();
    assert!(session.update(&update).unwrap_err().to_string().contains("injected linear G2"));
    assert_eq!(session.proof(),&before);assert_eq!(session.current_witness(),&old_witness);
    assert_eq!(session.rebuild_progress(),progress);assert_eq!(*session.last_receipt(),receipt);
    assert_eq!(*session.update_timings(),timings);
    assert_eq!(session.backend().stats().bytes_allocated,allocated);assert!(!session.is_faulted());
    session.update(&update).unwrap();
    assert!(session.last_receipt().swapped && session.last_receipt().restarted);
    assert_eq!(session.last_receipt().frozen_entries,4);
    assert_eq!(session.last_receipt().overlay_entries,4);
    assert_eq!(session.last_receipt().delta_entries,4);
    let restarted=session.rebuild_progress().unwrap();
    assert_eq!(restarted.source_generation,session.proof().generation);
    assert_eq!(restarted.paid_changes,0);assert_eq!(restarted.command,0);
    let expected=params.prove_anchor(&instance,&old_witness).unwrap();
    assert_eq!(session.proof().anchor,expected.anchor);
    assert!(params.verifying_key().verify_reference(&instance,session.proof()).unwrap());
}

#[test]
fn reuse_preserves_frozen_epochs_across_public_edits_hot_reverts_and_noops() {
    let n=64;let circuit=ToyCircuit::new(n).unwrap();
    let (instance,witness)=circuit.valid_instance();
    let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(18007)).unwrap();
    let mut session=params.start_incremental_session(&instance,&witness,
        IncrementalConfig {linear_rebuild_reuse:true,trace_updates:true,..Default::default()},
        CpuRebuildBackend::default()).unwrap();
    let mut history=std::collections::BTreeMap::new();
    history.insert(0,(instance.clone(),witness.clone()));
    let mut reused=false;
    for step in 0..12 {
        let (update,public)=if step%4==2 {
            let left=session.current_witness().wires[0]+Fr::from(1u64);
            let right=session.current_witness().wires[n]-Fr::from(1u64);
            (Update {indices:vec![0,n],values:vec![left,right]},Update {indices:vec![0],values:vec![left]})
        } else if step%4==3 {
            let v=session.current_witness().wires[4*n+1];
            (Update {indices:vec![4*n+1;2],values:vec![v+Fr::from(77u64),v]},Update {indices:vec![],values:vec![]})
        } else {
            let i=1+step%3;
            let next=if step==4 {witness.wires[4*n+i]} else {session.current_witness().wires[4*n+i]+Fr::from(1u64)};
            (Update {indices:vec![4*n+i,5*n+i],values:vec![next,session.current_witness().wires[3*n+i]*next]},
                Update {indices:vec![],values:vec![]})
        };
        session.update_with_public_inputs(&update,&public).unwrap();
        let generation=session.proof().generation;
        history.insert(generation,(session.current_instance().clone(),session.current_witness().clone()));
        let (old_instance,old_witness)=&history[&session.proof().anchor_generation];
        let expected=params.prove_anchor(old_instance,old_witness).unwrap();
        assert_eq!(session.proof().anchor,expected.anchor);
        assert!(params.verifying_key().verify_reference(session.current_instance(),session.proof()).unwrap());
        assert!(session.last_receipt().rebuild_units<=session.last_receipt().allowance);
        reused|=session.last_receipt().linear_reused;
    }
    assert!(reused);
}
