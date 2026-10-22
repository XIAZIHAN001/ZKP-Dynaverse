use ark_bn254::Fr;
use dynaverse_protocol::{dynaverse::DynaverseParams,toy_circuit::ToyCircuit,Update,
    incremental_rebuild::{IncrementalConfig,operators::CpuRebuildBackend}};
use rand::{rngs::StdRng,SeedableRng};
#[allow(dead_code)]
#[path="support/linear_fault.rs"]
mod linear_fault;

fn update(w:&dynaverse_protocol::Witness,n:usize,i:usize)->Update {
    let next=w.wires[4*n+i]+Fr::from(1u64);
    Update {indices:vec![4*n+i,5*n+i],values:vec![next,w.wires[3*n+i]*next]}
}

#[test]
fn nested_sparse_timings_are_disjoint_optional_and_transactional() {
    let n=16;let c=ToyCircuit::new(n).unwrap();let (i,w)=c.valid_instance();
    let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(19001)).unwrap();
    let cfg=IncrementalConfig {foreground_linear_reuse:true,support_geometry_reuse:true,..Default::default()};
    let mut off=p.start_incremental_session(&i,&w,cfg.clone(),CpuRebuildBackend::default()).unwrap();
    let mut on=p.start_incremental_session(&i,&w,IncrementalConfig {trace_updates:true,..cfg},CpuRebuildBackend::default()).unwrap();
    for step in 0..4 {
        let u=update(on.current_witness(),n,1);
        assert_eq!(on.update(&u).unwrap(),off.update(&u).unwrap());
        let t=on.update_timings();
        assert!(t.delta_detail.recorded_ns()>0,"sparse detail not recorded at step {step}");
        assert!(t.delta_detail.ns[6]>0,"mandatory CPU proof validation hidden in residual");
        assert!(t.delta_detail.recorded_ns()<=t.delta_proof_ns);
        assert!(t.rebase_detail.recorded_ns()<=t.rebase_ns);
        assert_eq!(off.update_timings().delta_detail.recorded_ns(),0);
        assert!(p.verifying_key().verify_reference(&i,on.proof()).unwrap());
    }
    let before=*on.update_timings();let proof=on.proof().clone();
    assert!(on.update(&Update {indices:vec![0],values:vec![Fr::from(999u64)]}).is_err());
    assert_eq!(*on.update_timings(),before);assert_eq!(on.proof(),&proof);
}

#[test]
fn product_rebuild_completes_identical_work_with_one_fe_per_cache_equation() {
    use dynaverse_protocol::VerificationMode;
    let n=16;let c=ToyCircuit::new(n).unwrap();let (i,w)=c.valid_instance();
    let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(19002)).unwrap();
    assert_eq!(IncrementalConfig::default().rebuild_cache_mode,VerificationMode::Reference);
    let cfg=IncrementalConfig {trace_updates:true,linear_rebuild_reuse:true,transform_row_batch:4,..Default::default()};
    let mut reference=p.start_incremental_session(&i,&w,cfg.clone(),CpuRebuildBackend::default()).unwrap();
    let mut product=p.start_incremental_session(&i,&w,IncrementalConfig {
        rebuild_cache_mode:VerificationMode::Product,..cfg},CpuRebuildBackend::default()).unwrap();
    let mut u=Update {indices:vec![],values:vec![]};
    for root in 1..=3 {let part=update(&w,n,root);u.indices.extend(part.indices);u.values.extend(part.values);}
    assert_eq!(product.update(&u).unwrap(),reference.update(&u).unwrap());
    assert_eq!(product.rebuild_progress(),None);assert_eq!(reference.rebuild_progress(),None);
    assert_eq!(product.last_receipt(),reference.last_receipt());
    let a=reference.update_timings();let b=product.update_timings();
    assert_eq!(b.cache_detail.equations,3*n as u64,"production cache path untraced");
    assert_eq!(a.cache_detail.equations,b.cache_detail.equations);
    assert_eq!(a.cache_detail.final_exponentiations,6*n as u64);
    assert_eq!(b.cache_detail.final_exponentiations,3*n as u64);
    assert_eq!(b.cache_detail.pairing_terms,6*n as u64);
    assert!(b.cache_detail.prepared_peak_entries<=5);
    assert!(b.cache_detail.g2_prepare_hits>0);
    assert!(b.cache_detail.recorded_ns()<=b.rebuild_ns[6]);
    assert!(a.cache_detail.recorded_ns()<=a.rebuild_ns[6]);
    assert_eq!(product.proof().anchor,p.prove_anchor(&i,product.current_witness()).unwrap().anchor);
    assert!(p.verifying_key().verify_reference(&i,product.proof()).unwrap());
}

#[test]
fn successful_update_then_real_revert_preserves_live_state_for_later_updates() {
    let n=64;let c=ToyCircuit::new(n).unwrap();let (i,w)=c.valid_instance();
    let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(19003)).unwrap();
    let mut reference=p.start_incremental_session(&i,&w,IncrementalConfig::default(),CpuRebuildBackend::default()).unwrap();
    let mut live=p.start_incremental_session(&i,&w,IncrementalConfig {
        foreground_linear_reuse:true,support_geometry_reuse:true,..Default::default()},CpuRebuildBackend::default()).unwrap();
    let first=update(&w,n,1);
    assert_eq!(live.update(&first).unwrap(),reference.update(&first).unwrap());
    let revert=Update {indices:first.indices.clone(),values:first.indices.iter().map(|&j|w.wires[j]).collect()};
    assert_eq!(live.update(&revert).unwrap(),reference.update(&revert).unwrap());
    assert_eq!(live.current_witness(),&w);assert_eq!(live.last_receipt().delta_entries,0);
    assert_eq!(live.last_receipt().foreground_linear_inputs,2);
    assert!(p.verifying_key().verify_reference(&i,live.proof()).unwrap());
    for root in [2,2,3] {
        let u=update(live.current_witness(),n,root);
        assert_eq!(live.update(&u).unwrap(),reference.update(&u).unwrap());
        assert!(p.verifying_key().verify_reference(&i,live.proof()).unwrap());
    }
}

#[test]
fn complete_geometry_candidate_is_not_published_after_later_backend_failure() {
    let n=64;let c=ToyCircuit::new(n).unwrap();let (i,w)=c.valid_instance();
    let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(19004)).unwrap();
    let cfg=IncrementalConfig {foreground_linear_reuse:true,support_geometry_reuse:true,
        trace_updates:true,rebuild_cache_mode:dynaverse_protocol::VerificationMode::Product,..Default::default()};
    let (backend,control)=linear_fault::FaultBackend::new();
    let mut live=p.start_incremental_session(&i,&w,cfg.clone(),backend).unwrap();
    let mut reference=p.start_incremental_session(&i,&w,cfg,CpuRebuildBackend::default()).unwrap();
    let first=update(&w,n,1);assert_eq!(live.update(&first).unwrap(),reference.update(&first).unwrap());
    assert_eq!(live.last_receipt().support_geometry_builds,1);
    let proof=live.proof().clone();let witness=live.current_witness().clone();
    let receipt=*live.last_receipt();let timing=*live.update_timings();let progress=live.rebuild_progress();
    let second=update(live.current_witness(),n,2);control.arm_materialize();
    let error=live.update(&second).unwrap_err();
    assert!(error.to_string().contains("materialize after complete sparse proof"),"{error}");
    assert_eq!(live.proof(),&proof);assert_eq!(live.current_witness(),&witness);
    assert_eq!(*live.last_receipt(),receipt);assert_eq!(*live.update_timings(),timing);
    assert_eq!(live.rebuild_progress(),progress);assert!(!live.is_faulted());
    assert_eq!(live.update(&second).unwrap(),reference.update(&second).unwrap());
    assert_eq!(live.last_receipt().support_geometry_builds,1,"failed candidate escaped transaction");
    assert!(!live.last_receipt().support_geometry_reused);
    let third=update(live.current_witness(),n,2);
    assert_eq!(live.update(&third).unwrap(),reference.update(&third).unwrap());
    assert!(live.last_receipt().support_geometry_reused,"successful candidate not reusable");
    assert!(p.verifying_key().verify_reference(&i,live.proof()).unwrap());
}

#[test]
fn product_and_reference_reject_corrupt_cache_and_retry_without_publishing() {
    use dynaverse_protocol::VerificationMode;
    for mode in [VerificationMode::Reference,VerificationMode::Product] {
        let n=16;let c=ToyCircuit::new(n).unwrap();let (i,w)=c.valid_instance();
        let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(19005)).unwrap();
        let cfg=IncrementalConfig {rebuild_cache_mode:mode,trace_updates:true,..Default::default()};
        let (backend,control)=linear_fault::FaultBackend::new();
        let mut live=p.start_incremental_session(&i,&w,cfg.clone(),backend).unwrap();
        let mut reference=p.start_incremental_session(&i,&w,cfg,CpuRebuildBackend::default()).unwrap();
        let mut u=Update {indices:vec![],values:vec![]};
        for root in 1..=3 {let v=update(&w,n,root);u.indices.extend(v.indices);u.values.extend(v.values);}
        let proof=live.proof().clone();let receipt=*live.last_receipt();let timing=*live.update_timings();
        control.arm_corrupt_cache();
        assert!(live.update(&u).unwrap_err().to_string().contains("opening cache check failed"));
        assert_eq!(live.proof(),&proof);assert_eq!(live.current_witness(),&w);
        assert_eq!(*live.last_receipt(),receipt);assert_eq!(*live.update_timings(),timing);
        assert_eq!(live.rebuild_progress(),None);assert!(!live.is_faulted());
        assert_eq!(live.update(&u).unwrap(),reference.update(&u).unwrap());
        assert_eq!(live.rebuild_progress(),None);
        assert_eq!(live.last_receipt(),reference.last_receipt());
    }
}
