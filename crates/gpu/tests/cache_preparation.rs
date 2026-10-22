#![cfg(feature="icicle")]
use dynaverse_gpu::IcicleBackend;
use dynaverse_protocol::{dynaverse::DynaverseParams,toy_circuit::ToyCircuit,
    incremental_rebuild::{SparseDispatchPolicy,operators::{CpuRebuildBackend,RebuildBackend}}};
use rand::{rngs::StdRng,SeedableRng};
#[allow(dead_code)]
#[path="../../protocol/tests/support/cache_preparation_profile.rs"]
mod profile;

#[test]
fn prepared_gpu_cache_matches_cpu_and_uncached_gpu_in_multiple_epochs() {
    for n in [16,64] {for cap in [3,64] {
        let c=ToyCircuit::new(n).unwrap();let(i,w)=c.valid_instance();
        let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(25100+n as u64)).unwrap();
        let make=||IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap();
        let mut cfg=profile::config("prepared",SparseDispatchPolicy::ForceGpu);
        cfg.rebuild_prepared_cache_roots=Some(cap);
        let mut new=p.start_incremental_session(&i,&w,cfg.clone(),make()).unwrap();
        cfg.rebuild_prepared_cache_roots=None;
        let mut old=p.start_incremental_session(&i,&w,cfg.clone(),make()).unwrap();
        cfg.sparse=SparseDispatchPolicy::CpuReference;
        let mut cpu=p.start_incremental_session(&i,&w,cfg,CpuRebuildBackend::default()).unwrap();
        let stats=new.cache_preparation_stats().unwrap();let bytes=new.backend().stats().bytes_allocated;
        for step in 0..4 {
            let addition=step%2==1;
            let u=profile::edits(new.current_witness(),n,if addition {4} else {8},addition,25101+step);
            let proof=new.update(&u).unwrap();assert_eq!(proof,old.update(&u).unwrap());
            assert_eq!(proof,cpu.update(&u).unwrap());assert!(new.rebuild_progress().is_none());
            let t=new.update_timings().cache_detail;
            // Addition changes four total wires, but s5 remains unchanged;
            // both legal families change exactly two opening-cache wires.
            let dirty=((new.last_receipt().frozen_wire_mask>>3)&7).count_ones() as u64;
            assert_eq!(dirty,2);
            assert_eq!(t.equations,dirty*n as u64);
            assert_eq!(t.final_exponentiations,t.equations);
            assert_eq!(t.divisors_formed,dirty*(n-n.min(cap)) as u64);
            assert_eq!(new.cache_preparation_stats(),Some(stats));
            assert_eq!(new.backend().stats().bytes_allocated,bytes);
            assert!(new.last_receipt().rebuild_units<=new.last_receipt().allowance);
            assert!(p.verifying_key().verify_reference(&i,&proof).unwrap());
            assert_eq!(proof.anchor,p.prove_anchor(&i,new.current_witness()).unwrap().anchor);
        }
        eprintln!("CP_GPU_EQUAL n={n} cap={cap} roots={} retained={} host_upper={} precompute_ns={}",stats.roots,stats.retained_bytes,stats.host_upper_bytes,stats.precompute_ns);
    }}
}

#[test]
#[ignore="explicit serial three-repeat prepared cache GPU scaling"]
fn local_cache_preparation_gpu_scaling() {
    profile::profile("gpu",false,||IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap());
}
#[test]
#[ignore="explicit serial k2 pending-work GPU diagnostic"]
fn local_cache_preparation_gpu_pending() {
    profile::pending_profile("gpu",||IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap());
}
