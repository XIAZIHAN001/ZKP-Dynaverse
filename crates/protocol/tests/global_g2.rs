use ark_bn254::Fr;
use ark_ff::UniformRand;
use dynaverse_protocol::{dynaverse::DynaverseParams,toy_circuit::ToyCircuit,Update,Witness,VerificationMode,
    incremental_rebuild::{IncrementalConfig,operators::{CpuRebuildBackend,RebuildBackend}}};
use rand::{rngs::StdRng,SeedableRng};
#[path="support/linear_fault.rs"]
mod linear_fault;
#[path="support/global_profile.rs"]
mod global_profile;

fn edits(w:&Witness,n:usize,roots:usize,addition:bool,seed:u64)->Update {
    let mut rng=StdRng::seed_from_u64(seed);let mut u=Update {indices:vec![],values:vec![]};
    for i in 1..=roots {
        let value=Fr::rand(&mut rng);
        if addition {
            let sum=w.wires[i]+value;u.indices.extend([n+i,2*n+i,3*n+i,5*n+i]);
            u.values.extend([value,sum,sum,sum*w.wires[4*n+i]]);
        } else {u.indices.extend([4*n+i,5*n+i]);u.values.extend([value,w.wires[3*n+i]*value]);}
    }
    u
}
fn config(limit:Option<usize>)->IncrementalConfig {
    IncrementalConfig {foreground_linear_reuse:true,freeze_linear_capture_limit:Some(128),
        frozen_g2_reuse_limit:limit,trace_updates:true,transform_row_batch:4,
        rebuild_cache_mode:VerificationMode::Product,..Default::default()}
}

#[test]
fn unchanged_frozen_s5_copies_global_commitment_and_skips_dense_g2() {
    let n=16;let c=ToyCircuit::new(n).unwrap();let (i,w)=c.valid_instance();
    let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(20000)).unwrap();
    let mut old=p.start_incremental_session(&i,&w,config(None),CpuRebuildBackend::default()).unwrap();
    let mut new=p.start_incremental_session(&i,&w,config(Some(8)),CpuRebuildBackend::default()).unwrap();
    let before=new.backend().stats().msm_g2_points;
    let u=edits(&w,n,2,true,20001);
    assert_eq!(new.update(&u).unwrap(),old.update(&u).unwrap());
    assert!(new.last_receipt().global_g2_reused,"unchanged frozen global G2 still recomputed densely");
    assert_eq!(new.last_receipt().global_g2_inputs,0);
    assert_eq!(new.last_receipt().global_g2_skipped_points,n);
    assert_eq!(new.backend().stats().msm_g2_points,before);
    assert_eq!(new.proof().anchor,p.prove_anchor(&i,new.current_witness()).unwrap().anchor);
    assert!(p.verifying_key().verify_reference(&i,new.proof()).unwrap());
}

#[test]
fn sparse_cap_boundary_fallback_and_next_epoch_match_dense_global_g2() {
    let n=16;let c=ToyCircuit::new(n).unwrap();let (i,w)=c.valid_instance();
    let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(20010)).unwrap();
    let mut old=p.start_incremental_session(&i,&w,config(None),CpuRebuildBackend::default()).unwrap();
    let mut new=p.start_incremental_session(&i,&w,config(Some(3)),CpuRebuildBackend::default()).unwrap();
    assert_eq!(new.resource_plan().global_g2_host_bytes,3*512+4096);
    assert_eq!(new.resource_plan().global_g2_workspace_bytes,(64<<20)+3*1024);
    for (step,roots) in [3,4,2,3].into_iter().enumerate() {
        let u=edits(new.current_witness(),n,roots,false,20011+step as u64);
        let before=new.backend().stats().msm_g2_points;
        let sparse=new.sparse_dispatch_stats().cpu.g2_points;
        assert_eq!(new.update(&u).unwrap(),old.update(&u).unwrap());
        let r=new.last_receipt();assert!(r.large_batch);assert_eq!(r.global_g2_reused,roots<=3);
        assert_eq!(r.global_g2_inputs,if roots<=3 {roots} else {0});
        assert_eq!(new.backend().stats().msm_g2_points-before,if roots<=3 {0} else {n as u64});
        assert_eq!(new.sparse_dispatch_stats().cpu.g2_points-sparse,if roots<=3 {roots as u64} else {0});
        assert!(r.rebuild_units<=r.allowance && r.swapped);
        assert!(new.update_timings().linear_detail.recorded_ns()<=new.update_timings().rebuild_ns[1]);
        assert_eq!(new.proof().anchor,p.prove_anchor(&i,new.current_witness()).unwrap().anchor);
        assert!(p.verifying_key().verify_reference(&i,new.proof()).unwrap());
    }
}

#[test]
fn g2_cap_counts_only_s5_and_works_without_g1_capture() {
    let n=64;let c=ToyCircuit::new(n).unwrap();let (i,w)=c.valid_instance();
    let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(20020)).unwrap();
    for capture in [false,true] {
        let cfg=IncrementalConfig {foreground_linear_reuse:capture,
            freeze_linear_capture_limit:if capture {Some(128)} else {None},
            frozen_g2_reuse_limit:Some(1),..Default::default()};
        let mut s=p.start_incremental_session(&i,&w,cfg,CpuRebuildBackend::default()).unwrap();
        let mut u=edits(&w,n,3,true,20021);let value=Fr::rand(&mut StdRng::seed_from_u64(20022));
        u.indices.extend([4*n+8,5*n+8]);u.values.extend([value,w.wires[3*n+8]*value]);
        s.update(&u).unwrap();let r=s.last_receipt();
        assert_eq!(r.frozen_entries,14);assert!(r.global_g2_reused);assert_eq!(r.global_g2_inputs,1);
        assert_eq!(r.frozen_linear_captured,capture);
        assert_eq!(s.proof().anchor,p.prove_anchor(&i,s.current_witness()).unwrap().anchor);
        assert!(p.verifying_key().verify_reference(&i,s.proof()).unwrap());
    }
}

#[test]
fn g2_or_later_cache_failure_restores_published_state_and_exact_retry() {
    let n=16;let c=ToyCircuit::new(n).unwrap();let (i,w)=c.valid_instance();
    let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(20030)).unwrap();
    for later_cache in [false,true] {
        let (backend,control)=linear_fault::FaultBackend::new();
        let mut s=p.start_incremental_session(&i,&w,config(Some(3)),backend).unwrap();
        let mut clean=p.start_incremental_session(&i,&w,config(Some(3)),CpuRebuildBackend::default()).unwrap();
        let seed=edits(&w,n,2,false,20031);
        assert_eq!(s.update(&seed).unwrap(),clean.update(&seed).unwrap());
        let before=s.proof().clone();let witness=s.current_witness().clone();
        let receipt=*s.last_receipt();let timing=*s.update_timings();let progress=s.rebuild_progress();
        let bytes=s.backend().stats().bytes_allocated;
        let u=edits(s.current_witness(),n,3,false,20032);
        if later_cache {control.arm_cache_read();} else {control.arm_g2();}
        let error=s.update(&u).unwrap_err().to_string();assert!(error.contains("injected"),"{error}");
        assert_eq!(s.proof(),&before);assert_eq!(s.current_witness(),&witness);
        assert_eq!(*s.last_receipt(),receipt);assert_eq!(*s.update_timings(),timing);
        assert_eq!(s.rebuild_progress(),progress);assert!(!s.is_faulted());
        assert_eq!(s.backend().stats().bytes_allocated,bytes);
        assert_eq!(s.update(&u).unwrap(),clean.update(&u).unwrap());
        assert_eq!(s.last_receipt(),clean.last_receipt());
        assert!(p.verifying_key().verify_reference(&i,s.proof()).unwrap());
    }
}

#[test]
fn invalid_global_g2_limits_and_insufficient_host_envelope_are_rejected() {
    let c=ToyCircuit::new(16).unwrap();let (i,w)=c.valid_instance();
    let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(20040)).unwrap();
    assert_eq!(IncrementalConfig::default().frozen_g2_reuse_limit,None);
    for limit in [0,4097,usize::MAX] {
        assert!(p.start_incremental_session(&i,&w,config(Some(limit)),CpuRebuildBackend::default()).is_err());
    }
    let old=p.start_incremental_session(&i,&w,config(None),CpuRebuildBackend::default()).unwrap();
    assert!(p.start_incremental_session(&i,&w,IncrementalConfig {memory_limit_bytes:old.resource_plan().upper_bytes,
        ..config(Some(3))},CpuRebuildBackend::default()).is_err());
}

#[test]
fn public_reverts_noops_and_trace_off_preserve_global_binding() {
    use dynaverse_protocol::incremental_rebuild::UpdateTimings;
    let n=16;let c=ToyCircuit::new(n).unwrap();let (i,w)=c.valid_instance();
    let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(20050)).unwrap();
    let mut s=p.start_incremental_session(&i,&w,IncrementalConfig {trace_updates:false,..config(Some(3))},CpuRebuildBackend::default()).unwrap();
    for revert in [false,true] {
        let left=if revert {w.wires[1]} else {Fr::rand(&mut StdRng::seed_from_u64(20051))};
        let right=w.wires[1]+w.wires[n+1]-left;
        s.update_with_public_inputs(&Update {indices:vec![1,n+1],values:vec![left,right]},&Update {indices:vec![1],values:vec![left]}).unwrap();
        assert!(s.last_receipt().global_g2_reused);assert_eq!(s.last_receipt().global_g2_inputs,0);
        assert_eq!(*s.update_timings(),UpdateTimings::default());
        assert_eq!(s.proof().anchor,p.prove_anchor(s.current_instance(),s.current_witness()).unwrap().anchor);
        assert!(p.verifying_key().verify_reference(s.current_instance(),s.proof()).unwrap());
    }
    let proof=s.proof().clone();let value=s.current_witness().wires[4*n+1];
    s.update(&Update {indices:vec![4*n+1,4*n+1],values:vec![value+Fr::from(1u64),value]}).unwrap();
    assert_eq!(s.proof(),&proof);assert!(!s.last_receipt().global_g2_reused);
}

#[test]
fn pending_restart_copies_frozen_s5_despite_later_dirty_s5_overlay() {
    let n=1024;let c=ToyCircuit::new(n).unwrap();let (i,w)=c.valid_instance();
    let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(20060)).unwrap();
    let mut s=p.start_incremental_session(&i,&w,config(Some(128)),CpuRebuildBackend::default()).unwrap();
    let mut history=std::collections::BTreeMap::from([(0,w.clone())]);
    for root in 1..=s.horizon()/2 {
        let value=s.current_witness().wires[4*n+root]+Fr::from(1u64);
        let out=s.current_witness().wires[3*n+root]*value;
        s.update(&Update {indices:vec![4*n+root,5*n+root],values:vec![value,out]}).unwrap();
        history.insert(s.proof().generation,s.current_witness().clone());
    }
    assert!(s.rebuild_progress().is_some());
    let u=edits(s.current_witness(),n,s.horizon()/4,true,20061);
    s.update(&u).unwrap();history.insert(s.proof().generation,s.current_witness().clone());
    assert!(s.last_receipt().swapped && s.last_receipt().restarted);
    let mut copied=false;
    for step in 0..s.horizon() {
        let root=20+step;let value=s.current_witness().wires[4*n+root]+Fr::from(1u64);
        let out=s.current_witness().wires[3*n+root]*value;
        s.update(&Update {indices:vec![4*n+root,5*n+root],values:vec![value,out]}).unwrap();
        history.insert(s.proof().generation,s.current_witness().clone());
        copied|=s.last_receipt().global_g2_reused && s.last_receipt().global_g2_inputs==0;
        assert!(p.verifying_key().verify_reference(&i,s.proof()).unwrap());
        if s.last_receipt().swapped {break;}
    }
    assert!(copied && s.last_receipt().swapped);
    assert_eq!(s.proof().anchor,p.prove_anchor(&i,&history[&s.proof().anchor_generation]).unwrap().anchor);
}

#[test]
#[ignore="explicit serial fixed-k CPU row4/16 controls, three repeats at each n"]
fn local_global_g2_cpu_scaling() {global_profile::profile("cpu",false,CpuRebuildBackend::default);}
