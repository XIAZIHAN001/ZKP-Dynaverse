#![cfg(feature="icicle")]
use ark_bn254::Fr;
use dynaverse_gpu::IcicleBackend;
use dynaverse_protocol::{dynaverse::DynaverseParams,toy_circuit::ToyCircuit,Update,
    incremental_rebuild::{IncrementalConfig,SparseDispatchPolicy,UpdateTimings,
        operators::{CpuRebuildBackend,RebuildBackend}}};
use rand::{rngs::StdRng,SeedableRng};
use std::time::Instant;

#[test]
fn frozen_linear_gpu_correction_matches_cpu_each_generation_and_switch() {
    for n in [16,64] {
        let circuit=ToyCircuit::new(n).unwrap();let (instance,witness)=circuit.valid_instance();
        let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(18100+n as u64)).unwrap();
        let config=IncrementalConfig {linear_rebuild_reuse:true,..Default::default()};
        let mut cpu=params.start_incremental_session(&instance,&witness,config.clone(),CpuRebuildBackend::default()).unwrap();
        let gpu=IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap();
        let mut gpu=params.start_incremental_session(&instance,&witness,
            IncrementalConfig {sparse:SparseDispatchPolicy::ForceGpu,..config},gpu).unwrap();
        let allocated=gpu.backend().stats().bytes_allocated;
        let mut reused=0;
        for step in 0..8 {
            let i=step%n;let next=cpu.current_witness().wires[4*n+i]+Fr::from(1u64);
            let update=Update {indices:vec![4*n+i,5*n+i],values:vec![next,cpu.current_witness().wires[3*n+i]*next]};
            let expected=cpu.update(&update).unwrap();let actual=gpu.update(&update).unwrap();
            assert_eq!(actual,expected,"n={n} step={step}");
            assert_eq!(gpu.last_receipt(),cpu.last_receipt());
            assert_eq!(gpu.rebuild_progress(),cpu.rebuild_progress());
            assert!(params.verifying_key().verify_reference(&instance,&actual).unwrap());
            assert_eq!(gpu.backend().stats().bytes_allocated,allocated);
            reused+=usize::from(gpu.last_receipt().linear_reused);
        }
        assert!(reused>0);
        eprintln!("DELTA_GPU_EQUAL n={n} reused={reused} resident={allocated}");
    }
}

#[test]
fn nonzero_sum_copy_gpu_correction_matches_independent_dense_anchor() {
    let n=16;let circuit=ToyCircuit::new(n).unwrap();let (instance,witness)=circuit.valid_instance();
    let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(18199)).unwrap();
    let gpu=IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap();
    let mut session=params.start_incremental_session(&instance,&witness,
        IncrementalConfig {linear_rebuild_reuse:true,sparse:SparseDispatchPolicy::ForceGpu,..Default::default()},gpu).unwrap();
    let i=1;let right=witness.wires[n+i]+Fr::from(1u64);let sum=witness.wires[i]+right;
    session.update(&Update {indices:vec![n+i,2*n+i,3*n+i,5*n+i],
        values:vec![right,sum,sum,sum*witness.wires[4*n+i]]}).unwrap();
    assert!(session.last_receipt().linear_reused && session.last_receipt().swapped);
    assert_eq!(session.last_receipt().frozen_entries,4);
    let expected=params.prove_anchor(&instance,session.current_witness()).unwrap();
    assert_eq!(session.proof().anchor,expected.anchor);
    assert!(params.verifying_key().verify_reference(&instance,session.proof()).unwrap());
}

#[test]
#[ignore="explicit local paired diagnostic sequence; no H200 claim"]
fn local_dynamic_rebuild_comparison_probe() {
    let n=16;let circuit=ToyCircuit::new(n).unwrap();let (instance,witness)=circuit.valid_instance();
    let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(18200)).unwrap();
    let mut sessions=Vec::new();
    for reuse in [false,true] {
        let gpu=IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap();
        sessions.push(params.start_incremental_session(&instance,&witness,
            IncrementalConfig {linear_rebuild_reuse:reuse,trace_updates:true,
                sparse:SparseDispatchPolicy::ForceGpu,..Default::default()},gpu).unwrap());
    }
    for step in 0..8 {
        let i=step%n;let next=sessions[0].current_witness().wires[4*n+i]+Fr::from(1u64);
        let update=Update {indices:vec![4*n+i,5*n+i],values:vec![next,sessions[0].current_witness().wires[3*n+i]*next]};
        for index in if step%2==0 {[0,1]} else {[1,0]} {
            let session=&mut sessions[index];let before=session.backend().stats();
            let sparse_before=session.sparse_dispatch_stats();
            let started=Instant::now();let proof=session.update(&update).unwrap();let wall=started.elapsed().as_nanos();
            let r=session.last_receipt();let t=session.update_timings();let after=session.backend().stats();
            let sb=session.sparse_dispatch_stats();
            eprintln!("DELTA_PROFILE n={n} reuse={index} step={step} r={} k={} a={} d={} ds={} o={} wires={} frozen_wires={} start={} swap={} linear={} wall_ns={wall} total_ns={} norm_ns={} constraints_ns={} prepare_ns={} publish_ns={} delta_ns={} rebase_ns={} resident_g1_points={} resident_g2_points={} allowance={} work={} skipped={} sparse_g1_points={} sparse_g2_points={} sparse_ntt_elements={} resident_fr_elements={} resident_g1_transform_elements={}",
                r.raw_edits,r.changed,r.affected_gates,r.delta_entries,r.frozen_entries,r.overlay_entries,
                r.changed_wire_mask,r.frozen_wire_mask,r.started,r.swapped,
                r.linear_reused,t.total_ns,t.normalization_ns,t.constraints_ns,t.preparation_ns,t.publication_ns,
                t.delta_proof_ns,t.rebase_ns,after.msm_g1_points-before.msm_g1_points,
                after.msm_g2_points-before.msm_g2_points,r.allowance,r.rebuild_units,r.linear_skipped_g1_points,
                sb.cpu.g1_points+sb.gpu.g1_points-sparse_before.cpu.g1_points-sparse_before.gpu.g1_points,
                sb.cpu.g2_points+sb.gpu.g2_points-sparse_before.cpu.g2_points-sparse_before.gpu.g2_points,
                sb.cpu.ntt_elements+sb.gpu.ntt_elements-sparse_before.cpu.ntt_elements-sparse_before.gpu.ntt_elements,
                after.fr_transform_elements-before.fr_transform_elements,
                after.g1_transform_elements-before.g1_transform_elements);
            for (phase,name) in UpdateTimings::REBUILD_NAMES.iter().enumerate() {
                eprintln!("DELTA_PHASE reuse={index} step={step} phase={name} ns={} calls={}",t.rebuild_ns[phase],t.rebuild_calls[phase]);
            }
            let started=Instant::now();assert!(params.verifying_key().verify_reference(&instance,&proof).unwrap());
            eprintln!("DELTA_EXTERNAL_VERIFY reuse={index} step={step} ns={}",started.elapsed().as_nanos());
        }
        assert_eq!(sessions[0].current_witness(),sessions[1].current_witness());
    }
}
