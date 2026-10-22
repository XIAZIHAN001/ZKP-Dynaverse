use ark_bn254::Fr;
use ark_ff::UniformRand;
use dynaverse_protocol::{dynaverse::DynaverseParams,toy_circuit::ToyCircuit,Update,Witness,VerificationMode,
    incremental_rebuild::{IncrementalConfig,operators::CpuRebuildBackend}};
use rand::{rngs::StdRng,SeedableRng};
#[path="support/linear_fault.rs"]
mod linear_fault;

fn config(cap:Option<usize>)->IncrementalConfig {
    IncrementalConfig {window:Some(4),linear_rebuild_reuse:true,unchanged_cache_reuse:true,
        foreground_linear_reuse:true,freeze_linear_capture_limit:Some(128),frozen_g2_reuse_limit:Some(128),
        support_geometry_reuse:true,trace_updates:true,transform_row_batch:16,transform_packet_max_len:cap,
        rebuild_cache_mode:VerificationMode::Product,..Default::default()}
}
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

#[test]
fn bounded_packet_schedule_matches_complete_anchor_and_reports_execution() {
    let n=16;let c=ToyCircuit::new(n).unwrap();let(i,w)=c.valid_instance();
    let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(23000)).unwrap();
    let mut old=p.start_incremental_session(&i,&w,config(None),CpuRebuildBackend::default()).unwrap();
    let mut new=p.start_incremental_session(&i,&w,config(Some(64)),CpuRebuildBackend::default()).unwrap();
    let u=edits(&w,n,4,true,23001);
    assert_eq!(new.update(&u).unwrap(),old.update(&u).unwrap());
    assert_eq!(new.proof().anchor,p.prove_anchor(&i,new.current_witness()).unwrap().anchor);
    assert!(p.verifying_key().verify_reference(&i,new.proof()).unwrap());
    assert!(new.last_receipt().transform_packet_calls>0,"complete transform still uses only tiled steps");
    assert!(new.last_receipt().transform_packet_elements>0);
    assert!(new.last_receipt().rebuild_units<=new.last_receipt().allowance);
}

#[test]
fn packet_cap_rejects_invalid_limits_and_preserves_default_or_over_cap_schedule() {
    assert_eq!(IncrementalConfig::default().transform_packet_max_len,None);
    let n=16;let c=ToyCircuit::new(n).unwrap();let(i,w)=c.valid_instance();
    let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(23010)).unwrap();
    for cap in [0,4097,usize::MAX] {
        assert!(p.start_incremental_session(&i,&w,config(Some(cap)),CpuRebuildBackend::default()).is_err());
    }
    let mut old=p.start_incremental_session(&i,&w,config(None),CpuRebuildBackend::default()).unwrap();
    let mut small=p.start_incremental_session(&i,&w,config(Some(1)),CpuRebuildBackend::default()).unwrap();
    let u=edits(&w,n,4,true,23011);assert_eq!(small.update(&u).unwrap(),old.update(&u).unwrap());
    assert_eq!(small.last_receipt().transform_packet_calls,0);
    assert_eq!(small.budget(),old.budget());
}

#[test]
fn repeated_epochs_and_mixed_cap_fallback_preserve_complete_proofs() {
    let n=16;let c=ToyCircuit::new(n).unwrap();let(i,w)=c.valid_instance();
    let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(23020)).unwrap();
    let mut sessions=Vec::new();
    for cap in [None,Some(16),Some(64)] {
        sessions.push(p.start_incremental_session(&i,&w,config(cap),CpuRebuildBackend::default()).unwrap());
    }
    for step in 0..4 {
        let u=edits(sessions[0].current_witness(),n,if step%2==0 {8} else {4},step%2==1,23021+step);
        for s in &mut sessions {
            s.update(&u).unwrap();assert!(s.last_receipt().swapped);
            assert!(s.last_receipt().rebuild_units<=s.last_receipt().allowance);
            assert_eq!(s.proof().anchor,p.prove_anchor(&i,s.current_witness()).unwrap().anchor);
            assert!(p.verifying_key().verify_reference(&i,s.proof()).unwrap());
        }
        assert_eq!(sessions[0].proof(),sessions[1].proof());assert_eq!(sessions[0].proof(),sessions[2].proof());
        assert!(sessions[1].last_receipt().transform_packet_calls>0);
        assert!(sessions[2].last_receipt().transform_packet_elements>sessions[1].last_receipt().transform_packet_elements);
    }
}

#[test]
fn later_cache_failure_after_packet_restores_public_state_and_exact_retry() {
    use dynaverse_protocol::incremental_rebuild::operators::RebuildBackend;
    let n=16;let c=ToyCircuit::new(n).unwrap();let(i,w)=c.valid_instance();
    let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(23030)).unwrap();
    let (backend,control)=linear_fault::FaultBackend::new();
    let mut s=p.start_incremental_session(&i,&w,config(Some(64)),backend).unwrap();
    let mut clean=p.start_incremental_session(&i,&w,config(Some(64)),CpuRebuildBackend::default()).unwrap();
    let u=edits(&w,n,2,false,23031);assert_eq!(s.update(&u).unwrap(),clean.update(&u).unwrap());
    let proof=s.proof().clone();let witness=s.current_witness().clone();let receipt=*s.last_receipt();
    let timing=*s.update_timings();let progress=s.rebuild_progress();let before=s.backend().stats();
    let u=edits(s.current_witness(),n,3,false,23032);control.arm_cache_read();
    let error=s.update(&u).unwrap_err().to_string();assert!(error.contains("injected"),"{error}");
    assert!(s.backend().stats().g1_transform_elements>before.g1_transform_elements);
    assert_eq!(s.proof(),&proof);assert_eq!(s.current_witness(),&witness);assert_eq!(*s.last_receipt(),receipt);
    assert_eq!(*s.update_timings(),timing);assert_eq!(s.rebuild_progress(),progress);assert!(!s.is_faulted());
    assert_eq!(s.backend().stats().bytes_allocated,before.bytes_allocated);
    assert_eq!(s.update(&u).unwrap(),clean.update(&u).unwrap());assert_eq!(s.last_receipt(),clean.last_receipt());
    assert_eq!(s.proof().anchor,p.prove_anchor(&i,s.current_witness()).unwrap().anchor);
    assert!(p.verifying_key().verify_reference(&i,s.proof()).unwrap());
}

#[path="support/packet_profile.rs"]
mod packet_profile;
#[test]
#[ignore="explicit serial three-repeat row16 versus packet CPU control"]
fn local_transform_packet_cpu_scaling() {packet_profile::profile("cpu",false,CpuRebuildBackend::default);}
#[test]
#[ignore="explicit serial legal k2 pending rebuild CPU diagnostic"]
fn local_transform_packet_cpu_pending() {packet_profile::pending_profile("cpu",CpuRebuildBackend::default);}
