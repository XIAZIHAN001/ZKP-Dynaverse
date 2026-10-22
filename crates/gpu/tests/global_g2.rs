#![cfg(feature="icicle")]
use dynaverse_gpu::IcicleBackend;
use dynaverse_protocol::{dynaverse::DynaverseParams,toy_circuit::ToyCircuit,
    incremental_rebuild::{SparseDispatchPolicy,operators::{CpuRebuildBackend,RebuildBackend}}};
use rand::{rngs::StdRng,SeedableRng};
#[path="../../protocol/tests/support/global_profile.rs"]
mod global_profile;

#[test]
fn global_g2_gpu_cpu_and_row_batches_match_dense_complete_anchors() {
    for n in [16,64] {
        let c=ToyCircuit::new(n).unwrap();let (i,w)=c.valid_instance();
        let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(21200+n as u64)).unwrap();
        let mut previous_work=None;
        for route in ["g2","row16"] {
            let mut cpu_cfg=global_profile::config(route,SparseDispatchPolicy::CpuReference);cpu_cfg.frozen_g2_reuse_limit=Some(2);
            let mut gpu_cfg=global_profile::config(route,SparseDispatchPolicy::ForceGpu);gpu_cfg.frozen_g2_reuse_limit=Some(2);
            let mut cpu=p.start_incremental_session(&i,&w,cpu_cfg,CpuRebuildBackend::default()).unwrap();
            let mut gpu=p.start_incremental_session(&i,&w,gpu_cfg,IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap()).unwrap();
            let before=gpu.backend().stats();let bytes=before.bytes_allocated;
            for step in 0..4 {
                let roots=if step==1 || step==2 {3} else {2};
                let u=global_profile::edits(cpu.current_witness(),n,roots,step==1,21300+n as u64*100+step as u64);
                assert_eq!(gpu.update(&u).unwrap(),cpu.update(&u).unwrap());
                assert_eq!(gpu.last_receipt(),cpu.last_receipt());
                assert_eq!(gpu.last_receipt().global_g2_reused,step!=2);
                assert_eq!(gpu.last_receipt().global_g2_inputs,if step==0 || step==3 {2} else {0});
                assert!(gpu.rebuild_progress().is_none());assert_eq!(gpu.backend().stats().bytes_allocated,bytes);
                assert!(p.verifying_key().verify_reference(&i,gpu.proof()).unwrap());
                assert_eq!(gpu.proof().anchor,p.prove_anchor(&i,gpu.current_witness()).unwrap().anchor);
            }
            let after=gpu.backend().stats();let work=(after.fr_transform_elements-before.fr_transform_elements,after.g1_transform_elements-before.g1_transform_elements);
            if let Some(expected)=previous_work {assert_eq!(work,expected);}previous_work=Some(work);
            eprintln!("GGS_EQUAL n={n} route={route} updates=4 resident={bytes}");
        }
    }
}

#[test]
#[ignore="fixed k16/window4/dirty-wire profile; three counterbalanced repetitions"]
fn local_global_g2_gpu_scaling() {
    global_profile::profile("gpu",false,||IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap());
}
#[test]
#[ignore="fixed k4 eligible-R1 diagnostic; separate from k16 scaling, three repeats"]
fn local_global_g2_r1_diagnostic() {
    global_profile::profile("gpu",true,||IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap());
}
