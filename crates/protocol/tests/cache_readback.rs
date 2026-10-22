use dynaverse_protocol::{dynaverse::DynaverseParams,toy_circuit::ToyCircuit,VerificationMode,
    incremental_rebuild::{IncrementalConfig,SparseDispatchPolicy,operators::CpuRebuildBackend}};
use rand::{rngs::StdRng,SeedableRng};
#[allow(dead_code)]
#[path="support/linear_fault.rs"]
mod linear_fault;
#[allow(dead_code)]
#[path="support/cache_preparation_profile.rs"]
mod fixture;
#[allow(dead_code)]
#[path="support/cache_readback_profile.rs"]
mod profile;

#[test]
fn bounded_readback_reduces_calls_without_skipping_equations() {
    let n=32;let c=ToyCircuit::new(n).unwrap();let(i,w)=c.valid_instance();
    let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(26100)).unwrap();
    let cfg=fixture::config("prepared",SparseDispatchPolicy::CpuReference);
    let (old_backend,old_control)=linear_fault::FaultBackend::new();
    let (new_backend,new_control)=linear_fault::FaultBackend::new();
    let mut old=p.start_incremental_session(&i,&w,cfg.clone(),old_backend).unwrap();
    let mut new=p.start_incremental_session(&i,&w,dynaverse_protocol::incremental_rebuild::IncrementalConfig {
        rebuild_cache_readback_roots:Some(16),..cfg},new_backend).unwrap();
    let u=fixture::edits(&w,n,8,false,26101);
    let old_before=old_control.cache_reads();let new_before=new_control.cache_reads();
    assert_eq!(new.update(&u).unwrap(),old.update(&u).unwrap());
    assert!(new.last_receipt().swapped && new.rebuild_progress().is_none());
    let old_reads=old_control.cache_reads()-old_before;
    let new_reads=new_control.cache_reads()-new_before;
    assert!(new_reads<old_reads,"bounded readback must reduce backend reads: new={new_reads}, old={old_reads}");
    assert_eq!(new_reads,2*n.div_ceil(16));assert_eq!(old_reads,2*n.div_ceil(4));
    assert_eq!(new.update_timings().cache_detail.read_batches,new_reads as u64);
    assert_eq!(new.update_timings().cache_detail.read_points,2*n as u64);
    assert_eq!(new.update_timings().cache_detail.max_read_roots,16);
    assert_eq!(new.update_timings().cache_detail.equations,2*n as u64);
    assert_eq!(new.update_timings().cache_detail.equations,old.update_timings().cache_detail.equations);
    assert!(p.verifying_key().verify_reference(&i,new.proof()).unwrap());
    assert_eq!(new.proof().anchor,p.prove_anchor(&i,new.current_witness()).unwrap().anchor);
}

#[test]
fn readback_caps_tails_and_mixed_prepared_prefix_preserve_all_equations() {
    assert_eq!(IncrementalConfig::default().rebuild_cache_readback_roots,None);
    let n=16;let c=ToyCircuit::new(n).unwrap();let(i,w)=c.valid_instance();
    let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(26110)).unwrap();
    for mode in [VerificationMode::Reference,VerificationMode::Product] {
        for cap in [4,7,16,64] {
            let cfg=IncrementalConfig {window:Some(2),trace_updates:true,rebuild_cache_mode:mode,
                rebuild_prepared_cache_roots:if mode==VerificationMode::Product {Some(3)} else {None},
                ..Default::default()};
            let mut old=p.start_incremental_session(&i,&w,cfg.clone(),CpuRebuildBackend::default()).unwrap();
            let mut new=p.start_incremental_session(&i,&w,IncrementalConfig {
                rebuild_cache_readback_roots:Some(cap),..cfg},CpuRebuildBackend::default()).unwrap();
            for step in 0..2 {
                let u=fixture::edits(new.current_witness(),n,3,false,26111+step);
                assert_eq!(new.update(&u).unwrap(),old.update(&u).unwrap());
                assert!(new.last_receipt().swapped && new.rebuild_progress().is_none());
                let t=new.update_timings().cache_detail;
                assert_eq!(t.read_batches,(3*n.div_ceil(cap)) as u64);
                assert_eq!(t.read_points,3*n as u64);assert!(t.max_read_roots<=cap);
                assert_eq!(t.equations,3*n as u64);
                let expected=if mode==VerificationMode::Product {3*n} else {6*n};
                assert_eq!(t.final_exponentiations,expected as u64);
                assert!(t.prepared_peak_entries<=9,"read merging must not retain a larger checker");
                assert!(t.recorded_ns()<=new.update_timings().rebuild_ns[6]);
                assert!(new.last_receipt().rebuild_units<=new.last_receipt().allowance);
                assert!(new.budget().total>=old.budget().total);
                assert!(new.budget().max_chunk>=old.budget().max_chunk);
                assert_eq!(new.resource_plan().upper_bytes,old.resource_plan().upper_bytes
                    +new.resource_plan().cache_readback_host_bytes);
                assert!(p.verifying_key().verify_reference(&i,new.proof()).unwrap());
            }
        }
    }
}

#[test]
fn delayed_failure_after_merged_cache_write_preserves_exact_retry() {
    let n=32;let c=ToyCircuit::new(n).unwrap();let(i,w)=c.valid_instance();
    let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(26120)).unwrap();
    let cfg=IncrementalConfig {rebuild_cache_readback_roots:Some(16),
        ..fixture::config("prepared",SparseDispatchPolicy::CpuReference)};
    let (backend,control)=linear_fault::FaultBackend::new();
    let mut bad=p.start_incremental_session(&i,&w,cfg.clone(),backend).unwrap();
    let mut good=p.start_incremental_session(&i,&w,cfg,CpuRebuildBackend::default()).unwrap();
    for step in 0..2 {
        let u=fixture::edits(bad.current_witness(),n,8,false,26121+step);
        let proof=bad.proof().clone();let witness=bad.current_witness().clone();
        let receipt=*bad.last_receipt();let timings=*bad.update_timings();let progress=bad.rebuild_progress();
        control.arm_corrupt_cache_after(1);
        assert!(bad.update(&u).unwrap_err().to_string().contains("opening cache check failed"));
        assert_eq!(control.cache_reads(),2);assert_eq!(bad.proof(),&proof);
        assert_eq!(bad.current_witness(),&witness);assert_eq!(*bad.last_receipt(),receipt);
        assert_eq!(*bad.update_timings(),timings);assert_eq!(bad.rebuild_progress(),progress);
        assert!(!bad.is_faulted());assert_eq!(bad.update(&u).unwrap(),good.update(&u).unwrap());
        assert_eq!(bad.last_receipt(),good.last_receipt());
        assert_eq!(bad.proof().anchor,p.prove_anchor(&i,bad.current_witness()).unwrap().anchor);
    }
}

#[test]
fn readback_trace_off_has_no_diagnostics_and_identical_proof() {
    let n=16;let c=ToyCircuit::new(n).unwrap();let(i,w)=c.valid_instance();
    let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(26130)).unwrap();
    let cfg=IncrementalConfig {window:Some(2),rebuild_cache_readback_roots:Some(7),
        rebuild_cache_mode:VerificationMode::Product,rebuild_prepared_cache_roots:Some(3),..Default::default()};
    let mut disabled=p.start_incremental_session(&i,&w,cfg.clone(),CpuRebuildBackend::default()).unwrap();
    let mut enabled=p.start_incremental_session(&i,&w,IncrementalConfig {trace_updates:true,..cfg},CpuRebuildBackend::default()).unwrap();
    let u=fixture::edits(&w,n,3,false,26131);
    assert_eq!(disabled.update(&u).unwrap(),enabled.update(&u).unwrap());
    assert_eq!(*disabled.update_timings(),Default::default());
    assert!(enabled.update_timings().cache_detail.read_batches>0);
}

#[test]
#[ignore="explicit serial three-repeat CPU readback-cap control"]
fn local_cache_readback_cpu_scaling() {profile::profile("cpu",CpuRebuildBackend::default);}

#[test]
#[ignore="explicit serial small-update CPU readback diagnostic"]
fn local_cache_readback_cpu_pending() {profile::pending_profile("cpu",CpuRebuildBackend::default);}
