use ark_bn254::Fr;
use dynaverse_protocol::{dynaverse::DynaverseParams,toy_circuit::ToyCircuit,Update,
    incremental_rebuild::{IncrementalConfig,UpdateTimings,operators::CpuRebuildBackend}};
use rand::{rngs::StdRng,SeedableRng};
use ark_ff::UniformRand;
use dynaverse_protocol::incremental_rebuild::operators::RebuildBackend;
#[path="support/linear_fault.rs"]
mod linear_fault;
#[path="support/frozen_profile.rs"]
mod frozen_profile;

fn full_width_additions(w:&dynaverse_protocol::Witness,n:usize,roots:usize,seed:u64)->Update {
    let mut rng=StdRng::seed_from_u64(seed);let mut u=Update {indices:vec![],values:vec![]};
    for root in 1..=roots {
        let right=Fr::rand(&mut rng);let sum=w.wires[root]+right;
        u.indices.extend([n+root,2*n+root,3*n+root,5*n+root]);
        u.values.extend([right,sum,sum,sum*w.wires[4*n+root]]);
    }
    u
}

#[test]
fn frozen_live_capture_skips_dense_g1_but_retains_global_g2() {
    let n=64;let c=ToyCircuit::new(n).unwrap();let (i,w)=c.valid_instance();
    let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(19401)).unwrap();
    let cfg=IncrementalConfig {foreground_linear_reuse:true,trace_updates:true,..Default::default()};
    let mut old=p.start_incremental_session(&i,&w,cfg.clone(),CpuRebuildBackend::default()).unwrap();
    let mut captured=p.start_incremental_session(&i,&w,IncrementalConfig {freeze_linear_capture_limit:Some(32),..cfg},CpuRebuildBackend::default()).unwrap();
    for seed in [19402,19403] {
        let u=full_width_additions(old.current_witness(),n,3,seed);
        assert_eq!(captured.update(&u).unwrap(),old.update(&u).unwrap());
        let r=captured.last_receipt();let t=captured.update_timings();
        assert!(r.frozen_linear_captured,"frozen target still recomputed dense fixed G1");
        assert!(r.large_batch && r.swapped && r.foreground_linear_reused);
        assert_eq!(r.foreground_linear_inputs,u.indices.len());
        assert!(r.frozen_linear_skipped_g1_points>n*56);
        assert!(!r.linear_reused);assert_eq!(t.linear_detail.ns[0],0);
        assert!(t.linear_detail.ns[1]>0 && t.linear_detail.ns[2]>0);
        assert!(t.linear_detail.recorded_ns()<=t.rebuild_ns[1]);
        assert!(r.rebuild_units<=r.allowance);
        assert_eq!(captured.proof().anchor,p.prove_anchor(&i,captured.current_witness()).unwrap().anchor);
        assert!(p.verifying_key().verify_reference(&i,captured.proof()).unwrap());
    }
}

#[test]
fn paid_linear_timing_separates_g1_g2_and_trace_off_is_zero() {
    let n=16;let c=ToyCircuit::new(n).unwrap();let (i,w)=c.valid_instance();
    let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(19400)).unwrap();
    for enabled in [true,false] {
        let mut s=p.start_incremental_session(&i,&w,IncrementalConfig {trace_updates:enabled,..Default::default()},CpuRebuildBackend::default()).unwrap();
        let mut u=Update {indices:vec![],values:vec![]};
        for root in 1..=3 {
            let next=w.wires[4*n+root]+Fr::from(1u64);
            u.indices.extend([4*n+root,5*n+root]);u.values.extend([next,w.wires[3*n+root]*next]);
        }
        s.update(&u).unwrap();assert!(s.last_receipt().swapped);
        let t=s.update_timings();
        if enabled {
            assert!(t.linear_detail.ns[0]>0,"missing fixed G1 timing");
            assert!(t.linear_detail.ns[1]>0,"missing global G2 timing");
            assert_eq!(t.linear_detail.ns[2],0);
            assert!(t.linear_detail.recorded_ns()<=t.rebuild_ns[1]);
        } else {assert_eq!(*t,UpdateTimings::default());}
        assert!(p.verifying_key().verify_reference(&i,s.proof()).unwrap());
    }
}

#[test]
fn invalid_capture_configuration_is_rejected_before_allocation() {
    let c=ToyCircuit::new(16).unwrap();let (i,w)=c.valid_instance();
    let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(19410)).unwrap();
    assert_eq!(IncrementalConfig::default().freeze_linear_capture_limit,None);
    for (live,limit) in [(false,8),(true,0),(true,4097),(true,usize::MAX)] {
        let cfg=IncrementalConfig {foreground_linear_reuse:live,freeze_linear_capture_limit:Some(limit),..Default::default()};
        assert!(p.start_incremental_session(&i,&w,cfg,CpuRebuildBackend::default()).is_err());
    }
}

#[test]
fn over_cap_dense_reset_allows_next_capture_and_trace_off_stays_zero() {
    let n=16;let c=ToyCircuit::new(n).unwrap();let (i,w)=c.valid_instance();
    let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(19411)).unwrap();
    let cfg=IncrementalConfig {foreground_linear_reuse:true,freeze_linear_capture_limit:Some(4),..Default::default()};
    let mut s=p.start_incremental_session(&i,&w,cfg,CpuRebuildBackend::default()).unwrap();
    assert_eq!(s.resource_plan().frozen_linear_bytes,3*56*96);
    for (roots,expected) in [(2,false),(1,true),(2,false),(1,true)] {
        let u=full_width_additions(s.current_witness(),n,roots,19412+100*s.proof().generation);
        s.update(&u).unwrap();let r=s.last_receipt();
        assert_eq!(r.frozen_linear_captured,expected);assert_eq!(r.foreground_linear_reused,expected);
        assert_eq!(r.foreground_linear_inputs,if expected {4} else {0});
        assert!(r.swapped && r.rebuild_units<=r.allowance);
        assert_eq!(*s.update_timings(),UpdateTimings::default());
        assert_eq!(s.proof().anchor,p.prove_anchor(&i,s.current_witness()).unwrap().anchor);
        assert!(p.verifying_key().verify_reference(&i,s.proof()).unwrap());
    }
}

#[test]
fn global_g2_failure_after_capture_rolls_back_and_retry_is_exact() {
    let n=64;let c=ToyCircuit::new(n).unwrap();let (i,w)=c.valid_instance();
    let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(19420)).unwrap();
    let cfg=IncrementalConfig {foreground_linear_reuse:true,freeze_linear_capture_limit:Some(32),trace_updates:true,..Default::default()};
    let (backend,control)=linear_fault::FaultBackend::new();
    let mut s=p.start_incremental_session(&i,&w,cfg.clone(),backend).unwrap();
    let mut clean=p.start_incremental_session(&i,&w,cfg,CpuRebuildBackend::default()).unwrap();
    for seed in [19421,19422] {
        let u=full_width_additions(s.current_witness(),n,3,seed);
        let proof=s.proof().clone();let witness=s.current_witness().clone();let r=*s.last_receipt();
        let t=*s.update_timings();let progress=s.rebuild_progress();let bytes=s.backend().stats().bytes_allocated;
        control.arm_resident_g2();
        assert!(s.update(&u).unwrap_err().to_string().contains("injected frozen global G2"));
        assert_eq!(s.proof(),&proof);assert_eq!(s.current_witness(),&witness);
        assert_eq!(*s.last_receipt(),r);assert_eq!(*s.update_timings(),t);
        assert_eq!(s.rebuild_progress(),progress);assert!(!s.is_faulted());
        assert_eq!(s.backend().stats().bytes_allocated,bytes);
        assert_eq!(s.update(&u).unwrap(),clean.update(&u).unwrap());
        assert_eq!(s.last_receipt(),clean.last_receipt());
        assert!(s.last_receipt().frozen_linear_captured);
        assert!(p.verifying_key().verify_reference(&i,s.proof()).unwrap());
    }
}

#[test]
fn public_edits_reverts_and_noops_keep_capture_bound_to_actual_wire_eta() {
    let n=16;let c=ToyCircuit::new(n).unwrap();let (i,w)=c.valid_instance();
    let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(19430)).unwrap();
    let cfg=IncrementalConfig {foreground_linear_reuse:true,freeze_linear_capture_limit:Some(32),trace_updates:true,..Default::default()};
    let mut s=p.start_incremental_session(&i,&w,cfg,CpuRebuildBackend::default()).unwrap();
    for revert in [false,true] {
        let left=if revert {w.wires[1]} else {Fr::rand(&mut StdRng::seed_from_u64(19431))};
        let right=w.wires[1]+w.wires[n+1]-left;
        let u=Update {indices:vec![1,n+1],values:vec![left,right]};
        let public=Update {indices:vec![1],values:vec![left]};
        s.update_with_public_inputs(&u,&public).unwrap();
        assert!(s.last_receipt().frozen_linear_captured);assert_eq!(s.last_receipt().foreground_linear_inputs,2);
        assert!(p.verifying_key().verify_reference(s.current_instance(),s.proof()).unwrap());
        assert_eq!(s.proof().anchor,p.prove_anchor(s.current_instance(),s.current_witness()).unwrap().anchor);
    }
    let proof=s.proof().clone();let value=s.current_witness().wires[1];
    s.update(&Update {indices:vec![1,1],values:vec![value+Fr::from(1u64),value]}).unwrap();
    assert_eq!(s.proof(),&proof);assert_eq!(s.last_receipt().changed,0);
    assert_eq!(s.last_receipt().foreground_linear_work,Default::default());
}

#[test]
#[ignore="explicit serial CPU full-width paid ablation; no overlapping GPU measurements"]
fn local_frozen_cpu_profile() {
    frozen_profile::profile("cpu",&[64,256],1,
        dynaverse_protocol::incremental_rebuild::SparseDispatchPolicy::CpuReference,CpuRebuildBackend::default);
}

#[test]
fn pending_capture_switch_and_restart_use_committed_live_target() {
    let n=1024;let c=ToyCircuit::new(n).unwrap();let (i,w)=c.valid_instance();
    let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(19450)).unwrap();
    let cfg=IncrementalConfig {foreground_linear_reuse:true,freeze_linear_capture_limit:Some(128),
        trace_updates:true,rebuild_cache_mode:dynaverse_protocol::VerificationMode::Product,..Default::default()};
    let mut s=p.start_incremental_session(&i,&w,cfg,CpuRebuildBackend::default()).unwrap();
    let mut history=std::collections::BTreeMap::from([(0,(i.clone(),w.clone()))]);
    for root in 1..=s.horizon()/2 {
        let next=s.current_witness().wires[4*n+root]+Fr::from(1u64);
        let out=s.current_witness().wires[3*n+root]*next;
        s.update(&Update {indices:vec![4*n+root,5*n+root],values:vec![next,out]}).unwrap();
        history.insert(s.proof().generation,(s.current_instance().clone(),s.current_witness().clone()));
    }
    assert!(s.rebuild_progress().is_some(),"fixture must leave captured source pending");
    let roots=s.horizon()/4;
    let u=full_width_additions(s.current_witness(),n,roots,19451);
    s.update(&u).unwrap();history.insert(s.proof().generation,(s.current_instance().clone(),s.current_witness().clone()));
    assert!(s.last_receipt().swapped && s.last_receipt().restarted);
    assert!(s.rebuild_progress().is_some());
    let mut later_capture=false;
    for step in 0..s.horizon() {
        let root=20+step;let next=s.current_witness().wires[4*n+root]+Fr::from(1u64);
        let out=s.current_witness().wires[3*n+root]*next;
        s.update(&Update {indices:vec![4*n+root,5*n+root],values:vec![next,out]}).unwrap();
        history.insert(s.proof().generation,(s.current_instance().clone(),s.current_witness().clone()));
        later_capture|=s.last_receipt().frozen_linear_captured;
        assert!(p.verifying_key().verify_reference(s.current_instance(),s.proof()).unwrap());
        if s.last_receipt().swapped {break;}
    }
    assert!(later_capture && s.last_receipt().swapped);
    let (anchor_i,anchor_w)=&history[&s.proof().anchor_generation];
    assert_eq!(s.proof().anchor,p.prove_anchor(anchor_i,anchor_w).unwrap().anchor);
}
