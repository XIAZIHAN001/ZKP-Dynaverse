#![cfg(feature="icicle")]
use dynaverse_gpu::IcicleBackend;
use dynaverse_protocol::{dynaverse::DynaverseParams,toy_circuit::ToyCircuit,
    incremental_rebuild::{SparseDispatchPolicy,operators::{CpuRebuildBackend,RebuildBackend}}};
use rand::{rngs::StdRng,SeedableRng};
#[path="../../protocol/tests/support/frozen_profile.rs"]
mod frozen_profile;

#[test]
fn frozen_gpu_full_width_proofs_match_cpu_and_dense_anchors() {
    for n in [16,64] {
        let c=ToyCircuit::new(n).unwrap();let (i,w)=c.valid_instance();
        let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(19700+n as u64)).unwrap();
        let mut cpu=p.start_incremental_session(&i,&w,frozen_profile::config(true,SparseDispatchPolicy::CpuReference),CpuRebuildBackend::default()).unwrap();
        let mut gpu=p.start_incremental_session(&i,&w,frozen_profile::config(true,SparseDispatchPolicy::ForceGpu),IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap()).unwrap();
        let bytes=gpu.backend().stats().bytes_allocated;
        for step in 0..4 {
            let u=frozen_profile::edits(cpu.current_witness(),n,cpu.horizon()/2+1,step%2==1,19710+n as u64*100+step as u64);
            assert_eq!(gpu.update(&u).unwrap(),cpu.update(&u).unwrap());
            assert_eq!(gpu.last_receipt(),cpu.last_receipt());
            assert!(gpu.last_receipt().frozen_linear_captured && gpu.rebuild_progress().is_none());
            assert!(gpu.update_timings().linear_detail.ns[1]>0);
            assert_eq!(gpu.update_timings().linear_detail.ns[0],0);
            assert_eq!(gpu.backend().stats().bytes_allocated,bytes);
            assert!(p.verifying_key().verify_reference(&i,gpu.proof()).unwrap());
            assert_eq!(gpu.proof().anchor,p.prove_anchor(&i,gpu.current_witness()).unwrap().anchor);
        }
        eprintln!("FP_EQUAL n={n} gates={} updates=4 resident={bytes}",2*n);
    }
}

#[test]
#[ignore="serial GPU full-width paid comparison; three counterbalanced repeats"]
fn local_frozen_gpu_profile() {
    frozen_profile::profile("gpu",&[64,256],3,SparseDispatchPolicy::ForceGpu,
        ||IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap());
}
#[test]
#[ignore="only after small admission; one exploratory full-width n1024 trial"]
fn local_frozen_gpu_profile_1024() {
    frozen_profile::profile("gpu",&[1024],1,SparseDispatchPolicy::ForceGpu,
        ||IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap());
}
