#![cfg(feature="icicle")]
use ark_bn254::Fr;
use dynaverse_gpu::IcicleBackend;
use dynaverse_protocol::{dynaverse::DynaverseParams,toy_circuit::ToyCircuit,Update,
    incremental_rebuild::{IncrementalConfig,SparseDispatchPolicy,operators::{CpuRebuildBackend,RebuildBackend}}};
use rand::{rngs::StdRng,SeedableRng};
use std::collections::BTreeMap;

#[test]
fn frozen_cache_reuse_matches_cpu_proofs_receipts_and_independent_anchors() {
    for n in [16,64] {
        let circuit=ToyCircuit::new(n).unwrap();let (instance,witness)=circuit.valid_instance();
        let p=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(18500+n as u64)).unwrap();
        let config=IncrementalConfig {linear_rebuild_reuse:true,unchanged_cache_reuse:true,..Default::default()};
        let mut cpu=p.start_incremental_session(&instance,&witness,config.clone(),CpuRebuildBackend::default()).unwrap();
        let gpu=IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap();
        let mut gpu=p.start_incremental_session(&instance,&witness,
            IncrementalConfig {sparse:SparseDispatchPolicy::ForceGpu,..config},gpu).unwrap();
        let allocated=gpu.backend().stats().bytes_allocated;
        let mut history=BTreeMap::from([(0,(instance.clone(),witness.clone()))]);
        let mut copied=0;
        for step in 0..12 {
            let i=1+step%4;let w=cpu.current_witness();
            let (u,public)=match step%4 {
                0=>{
                    let left=w.wires[i]+Fr::from(1u64);let right=w.wires[n+i]-Fr::from(1u64);
                    (Update {indices:vec![i,n+i],values:vec![left,right]},Update {indices:vec![i],values:vec![left]})
                },
                1=>{
                    let next=w.wires[4*n+i]+Fr::from(1u64);
                    (Update {indices:vec![4*n+i,5*n+i],values:vec![next,w.wires[3*n+i]*next]},Update {indices:vec![],values:vec![]})
                },
                case=>{
                    let right=w.wires[n+i]+Fr::from(1u64);let sum=w.wires[i]+right;
                    let next=w.wires[4*n+i]+Fr::from(u64::from(case==3));
                    (Update {indices:vec![n+i,2*n+i,3*n+i,4*n+i,5*n+i],values:vec![right,sum,sum,next,sum*next]},Update {indices:vec![],values:vec![]})
                }
            };
            let expected=cpu.update_with_public_inputs(&u,&public).unwrap();
            let actual=gpu.update_with_public_inputs(&u,&public).unwrap();
            assert_eq!(actual,expected,"n={n} step={step}");
            assert_eq!(gpu.last_receipt(),cpu.last_receipt());assert_eq!(gpu.rebuild_progress(),cpu.rebuild_progress());
            history.insert(cpu.proof().generation,(cpu.current_instance().clone(),cpu.current_witness().clone()));
            let (old_i,old_w)=&history[&cpu.proof().anchor_generation];
            assert_eq!(actual.anchor,p.prove_anchor(old_i,old_w).unwrap().anchor);
            assert!(p.verifying_key().verify_reference(gpu.current_instance(),&actual).unwrap());
            assert_eq!(gpu.backend().stats().bytes_allocated,allocated);
            copied+=gpu.last_receipt().cache_reused_points;
        }
        assert!(copied>0);eprintln!("CACHE_GPU_EQUAL n={n} copied={copied} resident={allocated}");
    }
}
