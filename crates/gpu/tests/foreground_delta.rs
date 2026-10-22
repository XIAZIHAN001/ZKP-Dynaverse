#![cfg(feature="icicle")]
use ark_bn254::Fr;
use dynaverse_gpu::IcicleBackend;
use dynaverse_protocol::{dynaverse::DynaverseParams,toy_circuit::ToyCircuit,Update,
    incremental_rebuild::{IncrementalConfig,SparseDispatchPolicy,operators::{CpuRebuildBackend,RebuildBackend}}};
use rand::{rngs::StdRng,SeedableRng};
use std::collections::BTreeMap;

#[test]
fn foreground_live_geometry_gpu_matches_cpu_all_proof_fields_and_epochs() {
    for n in [16,64] {
        let circuit=ToyCircuit::new(n).unwrap();let (instance,witness)=circuit.valid_instance();
        let p=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(18800+n as u64)).unwrap();
        let cfg=IncrementalConfig {linear_rebuild_reuse:true,unchanged_cache_reuse:true,
            foreground_linear_reuse:true,support_geometry_reuse:true,transform_row_batch:4,..Default::default()};
        let mut cpu=p.start_incremental_session(&instance,&witness,cfg.clone(),CpuRebuildBackend::default()).unwrap();
        let backend=IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap();
        let mut gpu=p.start_incremental_session(&instance,&witness,
            IncrementalConfig {sparse:SparseDispatchPolicy::ForceGpu,..cfg},backend).unwrap();
        let bytes=gpu.backend().stats().bytes_allocated;
        let mut history=BTreeMap::from([(0,(instance.clone(),witness.clone()))]);
        let mut hits=0;
        for step in 0..14 {
            let i=if step<6 {1} else {1+step%4};let w=cpu.current_witness();
            let (u,public)=if step<6 || step%4==1 {
                let next=w.wires[4*n+i]+Fr::from(1u64);
                (Update {indices:vec![4*n+i,5*n+i],values:vec![next,w.wires[3*n+i]*next]},Update {indices:vec![],values:vec![]})
            } else if step%4==0 {
                let left=w.wires[i]+Fr::from(1u64);let right=w.wires[n+i]-Fr::from(1u64);
                (Update {indices:vec![i,n+i],values:vec![left,right]},Update {indices:vec![i],values:vec![left]})
            } else {
                let right=w.wires[n+i]+Fr::from(1u64);let sum=w.wires[i]+right;
                let next=w.wires[4*n+i]+Fr::from(u64::from(step%4==3));
                (Update {indices:vec![n+i,2*n+i,3*n+i,4*n+i,5*n+i],values:vec![right,sum,sum,next,sum*next]},Update {indices:vec![],values:vec![]})
            };
            let expected=cpu.update_with_public_inputs(&u,&public).unwrap();
            let actual=gpu.update_with_public_inputs(&u,&public).unwrap();
            assert_eq!(actual,expected,"n={n} step={step}");assert_eq!(gpu.last_receipt(),cpu.last_receipt());
            assert_eq!(gpu.rebuild_progress(),cpu.rebuild_progress());
            history.insert(cpu.proof().generation,(cpu.current_instance().clone(),cpu.current_witness().clone()));
            let (old_i,old_w)=&history[&cpu.proof().anchor_generation];
            assert_eq!(actual.anchor,p.prove_anchor(old_i,old_w).unwrap().anchor);
            assert!(p.verifying_key().verify_reference(gpu.current_instance(),&actual).unwrap());
            assert_eq!(gpu.backend().stats().bytes_allocated,bytes);
            hits+=usize::from(gpu.last_receipt().support_geometry_reused);
        }
        if n==64 {assert!(hits>0);}
        eprintln!("FOREGROUND_GPU_EQUAL n={n} geometry_hits={hits} resident={bytes}");
    }
}
