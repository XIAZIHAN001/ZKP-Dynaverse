use ark_bn254::Fr;
use dynaverse_protocol::{dynaverse::{DynaverseParams,IncrementalSession},toy_circuit::ToyCircuit,
    Instance,Witness,Update,incremental_rebuild::{IncrementalConfig,operators::{CpuRebuildBackend,RebuildBackend}}};
use rand::{rngs::StdRng,SeedableRng};
use std::collections::BTreeMap;
#[path="support/linear_fault.rs"]
mod linear_fault;

fn empty()->Update {Update {indices:vec![],values:vec![]}}
fn multiplier(w:&Witness,n:usize,i:usize)->Update {
    let next=w.wires[4*n+i]+Fr::from(1u64);
    Update {indices:vec![4*n+i,5*n+i],values:vec![next,w.wires[3*n+i]*next]}
}
type History=BTreeMap<u64,(Instance,Witness)>;
fn check<B:RebuildBackend>(p:&DynaverseParams,s:&IncrementalSession<'_,B>,h:&History) {
    let (i,w)=&h[&s.proof().anchor_generation];
    assert_eq!(s.proof().anchor,p.prove_anchor(i,w).unwrap().anchor);
    assert!(p.verifying_key().verify_reference(s.current_instance(),s.proof()).unwrap());
    assert!(s.last_receipt().rebuild_units<=s.last_receipt().allowance);
    assert_eq!(s.last_receipt().work.units().unwrap(),s.last_receipt().rebuild_units);
}

#[test]
fn unchanged_wire_reuse_reduces_group_transforms_and_keeps_exact_anchor() {
    let n=16;let circuit=ToyCircuit::new(n).unwrap();let (instance,witness)=circuit.valid_instance();
    let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(18400)).unwrap();
    for linear in [false,true] {
        let base=IncrementalConfig {linear_rebuild_reuse:linear,..Default::default()};
        let mut dense=params.start_incremental_session(&instance,&witness,base.clone(),CpuRebuildBackend::default()).unwrap();
        let mut reuse=params.start_incremental_session(&instance,&witness,
            IncrementalConfig {unchanged_cache_reuse:true,..base},CpuRebuildBackend::default()).unwrap();
        let d0=dense.backend().stats();let r0=reuse.backend().stats();
        let update=multiplier(&witness,n,1);
        dense.update(&update).unwrap();reuse.update(&update).unwrap();
        assert!(reuse.last_receipt().swapped);
        assert_eq!(reuse.last_receipt().cache_reused_wire_mask,1);
        assert_eq!(reuse.last_receipt().cache_reused_points,n);
        assert_eq!(reuse.proof(),dense.proof());
        assert_eq!(reuse.proof().anchor,params.prove_anchor(&instance,reuse.current_witness()).unwrap().anchor);
        assert!(params.verifying_key().verify_reference(&instance,reuse.proof()).unwrap());
        let dr=dense.backend().stats();let rr=reuse.backend().stats();
        assert!((rr.g1_transform_elements-r0.g1_transform_elements)<(dr.g1_transform_elements-d0.g1_transform_elements));
        // Coefficient IFFTs remain, but the skipped cache convolution also
        // contained a scalar Fr transform, so both counters should decrease.
        assert!((rr.fr_transform_elements-r0.fr_transform_elements)<(dr.fr_transform_elements-d0.fr_transform_elements));
        assert_eq!(rr.bytes_allocated,r0.bytes_allocated);
    }
}

#[test]
fn cache_masks_cover_public_preserved_nonzero_sum_and_all_changed_wires() {
    let n=16;let circuit=ToyCircuit::new(n).unwrap();let (instance,witness)=circuit.valid_instance();
    let p=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(18401)).unwrap();
    for (case,expected_mask) in [(0,7u8),(1,2),(2,0)] {
        let mut s=p.start_incremental_session(&instance,&witness,
            IncrementalConfig {linear_rebuild_reuse:true,unchanged_cache_reuse:true,..Default::default()},CpuRebuildBackend::default()).unwrap();
        let i=1;
        let (u,public)=if case==0 {
            let left=witness.wires[i]+Fr::from(1u64);let right=witness.wires[n+i]-Fr::from(1u64);
            (Update {indices:vec![i,n+i],values:vec![left,right]},Update {indices:vec![i],values:vec![left]})
        } else {
            let right=witness.wires[n+i]+Fr::from(1u64);let sum=witness.wires[i]+right;
            let next=witness.wires[4*n+i]+Fr::from(u64::from(case==2));
            (Update {indices:vec![n+i,2*n+i,3*n+i,4*n+i,5*n+i],values:vec![right,sum,sum,next,sum*next]},empty())
        };
        s.update_with_public_inputs(&u,&public).unwrap();
        assert!(s.last_receipt().swapped);
        assert_eq!(s.last_receipt().cache_reused_wire_mask,expected_mask);
        assert_eq!(s.last_receipt().cache_reused_points,n*expected_mask.count_ones() as usize);
        assert_eq!(s.proof().anchor,p.prove_anchor(s.current_instance(),s.current_witness()).unwrap().anchor);
        assert!(p.verifying_key().verify_reference(s.current_instance(),s.proof()).unwrap());
        // Later sparse proofs consume the installed cache, so copied cache
        // corruption cannot hide behind equality of anchor commitments alone.
        for step in 0..4 {
            s.update(&multiplier(s.current_witness(),n,step+2)).unwrap();
            assert!(p.verifying_key().verify_reference(s.current_instance(),s.proof()).unwrap());
        }
    }
}

#[test]
fn frozen_clean_cache_ignores_live_dirty_overlay_and_survives_restarts() {
    let n=64;let circuit=ToyCircuit::new(n).unwrap();let (instance,witness)=circuit.valid_instance();
    let p=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(18402)).unwrap();
    let mut s=p.start_incremental_session(&instance,&witness,
        IncrementalConfig {unchanged_cache_reuse:true,trace_updates:true,..Default::default()},CpuRebuildBackend::default()).unwrap();
    let mut history=History::from([(0,(instance.clone(),witness.clone()))]);
    let mut copied_mask=0;
    for i in [1,2] {
        s.update(&multiplier(s.current_witness(),n,i)).unwrap();
        history.insert(s.proof().generation,(s.current_instance().clone(),s.current_witness().clone()));
        check(&p,&s,&history);
        copied_mask|=s.last_receipt().cache_reused_wire_mask;
    }
    let frozen=s.rebuild_progress().expect("fixture must preserve an unfinished frozen epoch");
    let i=3;let right=s.current_witness().wires[n+i]+Fr::from(1u64);let sum=s.current_witness().wires[i]+right;
    s.update(&Update {indices:vec![n+i,2*n+i,3*n+i,5*n+i],values:vec![right,sum,sum,sum*s.current_witness().wires[4*n+i]]}).unwrap();
    history.insert(s.proof().generation,(s.current_instance().clone(),s.current_witness().clone()));
    assert!(s.last_receipt().swapped && s.last_receipt().restarted);
    assert_eq!(s.proof().anchor_generation,frozen.source_generation);
    assert_eq!(copied_mask|s.last_receipt().cache_reused_wire_mask,1);
    assert_eq!(s.last_receipt().frozen_wire_mask&8,0);
    assert_ne!(s.last_receipt().changed_wire_mask&8,0);
    check(&p,&s,&history);
    let progress=s.rebuild_progress();let proof=s.proof().clone();
    s.update(&empty()).unwrap();assert_eq!(s.proof(),&proof);assert_eq!(s.rebuild_progress(),progress);
    assert_eq!(s.last_receipt().cache_reused_points,0);
    for step in 0..8 {
        s.update(&multiplier(s.current_witness(),n,4+step)).unwrap();
        history.insert(s.proof().generation,(s.current_instance().clone(),s.current_witness().clone()));
        check(&p,&s,&history);
    }
}

#[test]
fn backend_failure_after_old_cache_copy_rolls_back_and_retry_consumes_exact_cache() {
    let n=16;let circuit=ToyCircuit::new(n).unwrap();let (instance,witness)=circuit.valid_instance();
    let p=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(18403)).unwrap();
    let (backend,control)=linear_fault::FaultBackend::new();
    let mut s=p.start_incremental_session(&instance,&witness,
        IncrementalConfig {linear_rebuild_reuse:true,unchanged_cache_reuse:true,trace_updates:true,..Default::default()},backend).unwrap();
    let proof=s.proof().clone();let receipt=*s.last_receipt();let timings=*s.update_timings();
    let allocated=s.backend().stats().bytes_allocated;
    // q4 is copied first; the first backend cache read is for changed q5.
    control.arm_cache_read();
    let update=multiplier(&witness,n,1);
    assert!(s.update(&update).unwrap_err().to_string().contains("injected cache read"));
    assert_eq!(s.proof(),&proof);assert_eq!(s.current_witness(),&witness);
    assert_eq!(*s.last_receipt(),receipt);assert_eq!(*s.update_timings(),timings);
    assert!(s.rebuild_progress().is_none());assert!(!s.is_faulted());
    assert_eq!(s.backend().stats().bytes_allocated,allocated);
    s.update(&update).unwrap();assert_eq!(s.last_receipt().cache_reused_points,n);
    assert_eq!(s.proof().anchor,p.prove_anchor(&instance,s.current_witness()).unwrap().anchor);
    for i in 2..6 {
        s.update(&multiplier(s.current_witness(),n,i)).unwrap();
        assert!(p.verifying_key().verify_reference(&instance,s.proof()).unwrap());
    }
}

#[test]
fn reverting_before_freeze_keeps_all_multiplication_wire_caches_clean() {
    let n=64;let circuit=ToyCircuit::new(n).unwrap();let (instance,witness)=circuit.valid_instance();
    let p=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(18404)).unwrap();
    let mut s=p.start_incremental_session(&instance,&witness,
        IncrementalConfig {linear_rebuild_reuse:true,unchanged_cache_reuse:true,..Default::default()},CpuRebuildBackend::default()).unwrap();
    let original=witness.wires[4*n];
    s.update(&Update {indices:vec![4*n],values:vec![original+Fr::from(1u64)]}).unwrap();
    s.update(&Update {indices:vec![4*n],values:vec![original]}).unwrap();
    assert_eq!(s.anchor_distance(),0);assert!(s.rebuild_progress().is_none());
    for i in [1,2] {
        let left=s.current_witness().wires[i]+Fr::from(1u64);
        let right=s.current_witness().wires[n+i]-Fr::from(1u64);
        s.update_with_public_inputs(&Update {indices:vec![i,n+i],values:vec![left,right]},
            &Update {indices:vec![i],values:vec![left]}).unwrap();
    }
    assert!(s.last_receipt().swapped);assert_eq!(s.last_receipt().frozen_wire_mask,3);
    assert_eq!(s.last_receipt().cache_reused_wire_mask,7);assert_eq!(s.last_receipt().cache_reused_points,3*n);
    assert_eq!(s.proof().anchor,p.prove_anchor(s.current_instance(),s.current_witness()).unwrap().anchor);
    assert!(p.verifying_key().verify_reference(s.current_instance(),s.proof()).unwrap());
}

#[test]
fn singleton_domain_reuses_identity_openings_for_unchanged_wires() {
    let circuit=ToyCircuit::new(1).unwrap();let (instance,witness)=circuit.valid_instance();
    let p=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(18405)).unwrap();
    let mut s=p.start_incremental_session(&instance,&witness,
        IncrementalConfig {linear_rebuild_reuse:true,unchanged_cache_reuse:true,..Default::default()},CpuRebuildBackend::default()).unwrap();
    s.update(&Update {indices:vec![4],values:vec![witness.wires[4]+Fr::from(1u64)]}).unwrap();
    assert_eq!(s.last_receipt().cache_reused_wire_mask,5);assert_eq!(s.last_receipt().cache_reused_points,2);
    assert_eq!(s.proof().anchor,p.prove_anchor(&instance,s.current_witness()).unwrap().anchor);
    assert!(p.verifying_key().verify_reference(&instance,s.proof()).unwrap());
}
