use ark_bn254::Fr;
use dynaverse_protocol::{dynaverse::DynaverseParams,toy_circuit::ToyCircuit,Update,VerificationMode,
    incremental_rebuild::{IncrementalConfig,operators::CpuRebuildBackend}};
use rand::{rngs::StdRng,SeedableRng};
#[allow(dead_code)]
#[path="support/linear_fault.rs"]
mod linear_fault;

fn edits(w:&dynaverse_protocol::Witness,n:usize,step:u64)->Update {
    let mut update=Update {indices:vec![],values:vec![]};
    for root in 1..=3 {
        let next=w.wires[4*n+root]+Fr::from(step+1);
        update.indices.extend([4*n+root,5*n+root]);
        update.values.extend([next,w.wires[3*n+root]*next]);
    }
    update
}

#[test]
fn prepared_roots_survive_multiple_epochs_without_skipping_new_equations() {
    let n=8;let c=ToyCircuit::new(n).unwrap();let (instance,witness)=c.valid_instance();
    let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(25000)).unwrap();
    let cfg=IncrementalConfig {window:Some(2),trace_updates:true,
        rebuild_cache_mode:VerificationMode::Product,..Default::default()};
    let mut old=p.start_incremental_session(&instance,&witness,cfg.clone(),CpuRebuildBackend::default()).unwrap();
    let mut new=p.start_incremental_session(&instance,&witness,IncrementalConfig {
        rebuild_prepared_cache_roots:Some(n),..cfg},CpuRebuildBackend::default()).unwrap();
    let preparation=new.cache_preparation_stats().unwrap();
    assert_eq!(old.cache_preparation_stats(),None);
    assert_eq!(preparation.roots,n);assert_eq!(preparation.prepared_entries,n+1);
    assert!(preparation.precompute_ns>0 && preparation.retained_bytes>0);
    assert_eq!(new.resource_plan().cache_preparation_host_bytes,preparation.host_upper_bytes);
    assert_eq!(new.resource_plan().upper_bytes,old.resource_plan().upper_bytes+preparation.host_upper_bytes);
    assert!(new.budget().total>old.budget().total);
    for step in 0..3 {
        let u=edits(new.current_witness(),n,step);
        assert_eq!(new.update(&u).unwrap(),old.update(&u).unwrap());
        assert_eq!(new.rebuild_progress(),None);
        let a=old.update_timings().cache_detail;let b=new.update_timings().cache_detail;
        assert_eq!(b.equations,3*n as u64);
        assert_eq!(b.final_exponentiations,b.equations);
        assert_eq!(b.pairing_terms,2*b.equations);
        assert_eq!(b.divisors_formed,0,"fixed public divisors rebuilt during update");
        assert_eq!(b.g2_prepare_builds,0,"fixed prepared G2 rebuilt during update");
        assert_eq!(b.g2_prepare_hits,2*b.equations);
        assert!(a.divisors_formed>0 && a.g2_prepare_builds>0);
        assert!(b.recorded_ns()<=new.update_timings().rebuild_ns[6]);
        assert!(p.verifying_key().verify_reference(&instance,new.proof()).unwrap());
        assert_eq!(new.proof().anchor,p.prove_anchor(&instance,new.current_witness()).unwrap().anchor);
        assert!(new.last_receipt().rebuild_units<=new.last_receipt().allowance);
        assert_eq!(new.cache_preparation_stats(),Some(preparation));
    }
}

#[test]
fn partial_prefix_falls_back_without_cache_thrashing_or_equation_loss() {
    assert_eq!(IncrementalConfig::default().rebuild_prepared_cache_roots,None);
    let n=8;let c=ToyCircuit::new(n).unwrap();let(i,w)=c.valid_instance();
    let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(25020)).unwrap();
    for cap in [1,3,8,4096] {
        let cfg=IncrementalConfig {window:Some(2),trace_updates:true,rebuild_cache_mode:VerificationMode::Product,
            rebuild_prepared_cache_roots:Some(cap),..Default::default()};
        let mut s=p.start_incremental_session(&i,&w,cfg,CpuRebuildBackend::default()).unwrap();
        let stats=s.cache_preparation_stats().unwrap();assert_eq!(stats.roots,n.min(cap));
        for step in 0..2 {
            s.update(&edits(s.current_witness(),n,step)).unwrap();
            let t=s.update_timings().cache_detail;
            assert_eq!(t.equations,3*n as u64);assert_eq!(t.final_exponentiations,t.equations);
            assert_eq!(t.divisors_formed,(3*(n-n.min(cap))) as u64);
            assert!(t.prepared_peak_entries<=stats.prepared_entries+5);
            assert_eq!(s.cache_preparation_stats(),Some(stats));
            assert_eq!(s.proof().anchor,p.prove_anchor(&i,s.current_witness()).unwrap().anchor);
            assert!(p.verifying_key().verify_reference(&i,s.proof()).unwrap());
        }
    }
}

#[test]
fn failed_cached_equation_restores_public_state_and_reuses_constants_on_retry() {
    let n=16;let c=ToyCircuit::new(n).unwrap();let(i,w)=c.valid_instance();
    let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(25030)).unwrap();
    let cfg=IncrementalConfig {window:Some(2),trace_updates:true,rebuild_cache_mode:VerificationMode::Product,
        rebuild_prepared_cache_roots:Some(16),..Default::default()};
    let (backend,control)=linear_fault::FaultBackend::new();
    let mut bad=p.start_incremental_session(&i,&w,cfg.clone(),backend).unwrap();
    let mut good=p.start_incremental_session(&i,&w,cfg,CpuRebuildBackend::default()).unwrap();
    for step in 0..2 {
        let proof=bad.proof().clone();let witness=bad.current_witness().clone();
        let timing=*bad.update_timings();let receipt=*bad.last_receipt();let progress=bad.rebuild_progress();
        let constants=bad.cache_preparation_stats();let u=edits(bad.current_witness(),n,step);
        control.arm_corrupt_cache();
        assert!(bad.update(&u).unwrap_err().to_string().contains("opening cache check failed"));
        assert_eq!(bad.proof(),&proof);assert_eq!(bad.current_witness(),&witness);
        assert_eq!(*bad.update_timings(),timing);assert_eq!(*bad.last_receipt(),receipt);
        assert_eq!(bad.rebuild_progress(),progress);assert_eq!(bad.cache_preparation_stats(),constants);
        assert!(!bad.is_faulted());assert_eq!(bad.update(&u).unwrap(),good.update(&u).unwrap());
        assert_eq!(bad.last_receipt(),good.last_receipt());
    }
}

#[test]
fn delayed_cached_equation_failure_restores_state_after_an_accepted_chunk() {
    let n=16;let c=ToyCircuit::new(n).unwrap();let(i,w)=c.valid_instance();
    let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(26001)).unwrap();
    let cfg=IncrementalConfig {window:Some(2),trace_updates:true,rebuild_cache_mode:VerificationMode::Product,
        rebuild_prepared_cache_roots:Some(16),..Default::default()};
    let (backend,control)=linear_fault::FaultBackend::new();
    let mut failed=p.start_incremental_session(&i,&w,cfg.clone(),backend).unwrap();
    let mut reference=p.start_incremental_session(&i,&w,cfg,CpuRebuildBackend::default()).unwrap();
    for step in 0..2 {
        let proof=failed.proof().clone();let witness=failed.current_witness().clone();
        let timing=*failed.update_timings();let receipt=*failed.last_receipt();let progress=failed.rebuild_progress();
        let constants=failed.cache_preparation_stats();let update=edits(failed.current_witness(),n,step);
        control.arm_corrupt_cache_after(1);
        let error=failed.update(&update).unwrap_err().to_string();
        assert!(error.contains("opening cache check failed"),"{error}");
        assert_eq!(control.cache_reads(),2,"failure must follow one valid, accepted cache chunk");
        assert_eq!(failed.proof(),&proof);assert_eq!(failed.current_witness(),&witness);
        assert_eq!(*failed.update_timings(),timing);assert_eq!(*failed.last_receipt(),receipt);
        assert_eq!(failed.rebuild_progress(),progress);assert_eq!(failed.cache_preparation_stats(),constants);
        assert!(!failed.is_faulted());
        assert_eq!(failed.update(&update).unwrap(),reference.update(&update).unwrap());
        assert_eq!(failed.last_receipt(),reference.last_receipt());
        assert!(p.verifying_key().verify_reference(&i,failed.proof()).unwrap());
        assert_eq!(failed.proof().anchor,p.prove_anchor(&i,failed.current_witness()).unwrap().anchor);
    }
}

#[allow(dead_code)]
#[path="support/cache_preparation_profile.rs"]
mod profile;
#[test]
#[ignore="explicit serial same-source prepared cache CPU control"]
fn local_cache_preparation_cpu_scaling() {profile::profile("cpu",false,CpuRebuildBackend::default);}
#[test]
#[ignore="explicit serial k2 pending-work CPU diagnostic"]
fn local_cache_preparation_cpu_pending() {profile::pending_profile("cpu",CpuRebuildBackend::default);}
