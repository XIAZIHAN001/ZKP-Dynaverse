#![cfg(feature="icicle")]
use dynaverse_gpu::IcicleBackend;
use dynaverse_protocol::{dynaverse::DynaverseParams,toy_circuit::ToyCircuit,
    incremental_rebuild::{IncrementalConfig,SparseDispatchPolicy,operators::{CpuRebuildBackend,RebuildBackend}}};
use rand::{rngs::StdRng,SeedableRng};
#[allow(dead_code)]
#[path="../../protocol/tests/support/cache_preparation_profile.rs"]
mod fixture;
#[allow(dead_code)]
#[path="../../protocol/tests/support/cache_readback_profile.rs"]
mod profile;

#[test]
fn coalesced_gpu_reads_match_cpu_and_old_gpu_across_prefixes_and_epochs() {
    for n in [16,64] {for cap in [7,16,64] {
        let c=ToyCircuit::new(n).unwrap();let(i,w)=c.valid_instance();
        let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(26200+n as u64)).unwrap();
        let make=||IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap();
        let cfg=IncrementalConfig {rebuild_prepared_cache_roots:Some(3),
            ..fixture::config("prepared",SparseDispatchPolicy::ForceGpu)};
        let mut new=p.start_incremental_session(&i,&w,IncrementalConfig {
            rebuild_cache_readback_roots:Some(cap),..cfg.clone()},make()).unwrap();
        let mut old=p.start_incremental_session(&i,&w,cfg.clone(),make()).unwrap();
        let mut cpu=p.start_incremental_session(&i,&w,IncrementalConfig {
            sparse:SparseDispatchPolicy::CpuReference,..cfg},CpuRebuildBackend::default()).unwrap();
        let bytes=new.backend().stats().bytes_allocated;
        for step in 0..2 {
            let addition=step%2==1;
            let u=fixture::edits(new.current_witness(),n,if addition {4} else {8},addition,26201+step);
            let proof=new.update(&u).unwrap();assert_eq!(proof,old.update(&u).unwrap());
            assert_eq!(proof,cpu.update(&u).unwrap());assert!(new.rebuild_progress().is_none());
            let t=new.update_timings().cache_detail;
            assert_eq!(t.equations,2*n as u64);assert_eq!(t.final_exponentiations,t.equations);
            assert_eq!(t.read_batches,(2*n.div_ceil(cap)) as u64);
            assert_eq!(t.read_points,2*n as u64);assert!(t.max_read_roots<=cap);
            assert!(t.prepared_peak_entries<=9);assert_eq!(t.divisors_formed,2*(n-3) as u64);
            assert_eq!(new.backend().stats().bytes_allocated,bytes);
            assert!(new.last_receipt().rebuild_units<=new.last_receipt().allowance);
            assert!(p.verifying_key().verify_reference(&i,&proof).unwrap());
            assert_eq!(proof.anchor,p.prove_anchor(&i,new.current_witness()).unwrap().anchor);
        }
        eprintln!("READBACK_GPU_EQUAL n={n} cap={cap} resident={bytes}");
    }}
}

#[test]
#[ignore="explicit serial three-repeat GPU readback-cap scan"]
fn local_cache_readback_gpu_scaling() {
    profile::profile("gpu",||IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap());
}

#[test]
#[ignore="explicit serial small-update GPU readback diagnostic"]
fn local_cache_readback_gpu_pending() {
    profile::pending_profile("gpu",||IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap());
}
