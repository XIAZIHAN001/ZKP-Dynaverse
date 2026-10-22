#![cfg(feature = "icicle")]
use ark_bn254::Fr;
use ark_ff::One;
use dynaverse_gpu::IcicleBackend;
use dynaverse_protocol::{
    dynaverse::DynaverseParams, toy_circuit::ToyCircuit, Update,
    incremental_rebuild::{IncrementalConfig, SparseDispatchPolicy, operators::{CpuRebuildBackend,RebuildBackend}},
};
use rand::{rngs::StdRng,SeedableRng};
use std::time::Instant;

#[test]
fn gpu_incremental_updates_match_cpu_at_every_generation_and_rebuild() {
    for n in [2,16,64] {
        let circuit=ToyCircuit::new(n).unwrap();
        let (instance,witness)=circuit.valid_instance();
        let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(5100+n as u64)).unwrap();
        let vk=params.verifying_key();
        let config=IncrementalConfig::default();
        let mut cpu=params.start_incremental_session(&instance,&witness,config.clone(),
            CpuRebuildBackend::default()).unwrap();
        let gpu=IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap();
        let gpu_config=IncrementalConfig {sparse:SparseDispatchPolicy::ForceGpu,..config};
        let mut gpu=params.start_incremental_session(&instance,&witness,gpu_config,gpu).unwrap();
        let allocated=gpu.backend().stats().bytes_allocated;
        let mut swaps=0;
        for step in 0..12 {
            let i=step%n;
            let next=cpu.current_witness().wires[4*n+i]+Fr::one();
            let output=cpu.current_witness().wires[3*n+i]*next;
            let update=Update {indices:vec![4*n+i,5*n+i],values:vec![next,output]};
            let expected=cpu.update(&update).unwrap();
            let actual=gpu.update(&update).unwrap();
            assert_eq!(actual,expected,"n={n}, step={step}");
            assert!(vk.verify(gpu.current_instance(),&actual).unwrap());
            assert!(vk.verify_reference(gpu.current_instance(),&actual).unwrap());
            assert_eq!(gpu.last_receipt(),cpu.last_receipt());
            assert_eq!(gpu.rebuild_progress(),cpu.rebuild_progress());
            assert!(gpu.last_receipt().rebuild_units<=gpu.last_receipt().allowance);
            assert_eq!(gpu.backend().stats().bytes_allocated,allocated);
            if gpu.last_receipt().swapped {swaps+=1;}
        }
        let stats=gpu.backend().stats();
        assert!(swaps>0);
        assert!(stats.fr_transform_elements>0 && stats.g1_transform_elements>0);
        assert!(stats.msm_g1_points>0 && stats.msm_g2_points>0);
        let sparse=gpu.sparse_dispatch_stats();
        assert_eq!(sparse.cpu,Default::default());
        assert!(sparse.gpu.g1_calls>0 && sparse.gpu.g2_calls>0 && sparse.gpu.ntt_calls>0);
        eprintln!("GPU_SESSION n={n} swaps={swaps} resident={} h2d={} d2h={} fr={} ec={} g1={} g2={}",
            stats.bytes_allocated,stats.h2d_bytes,stats.d2h_bytes,stats.fr_transform_elements,
            stats.g1_transform_elements,stats.msm_g1_points,stats.msm_g2_points);
    }
}

#[test]
fn gpu_session_rejects_insufficient_journal_at_initialization() {
    let circuit=ToyCircuit::new(2).unwrap();
    let (instance,witness)=circuit.valid_instance();
    let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(5300)).unwrap();
    let backend=IcicleBackend::with_resident_limits(256<<20,1024).unwrap();
    let result=params.start_incremental_session(&instance,&witness,IncrementalConfig::default(),backend);
    assert!(result.err().unwrap().to_string().contains("journal"));
}

#[test]
#[ignore = "explicit local GPU latency/work probe, includes CPU public/prover checks"]
fn gpu_incremental_latency_probe() {
    for n in [16,64,256] {
        let circuit=ToyCircuit::new(n).unwrap();
        let (instance,witness)=circuit.valid_instance();
        let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(5200+n as u64)).unwrap();
        let backend=IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap();
        let mut session=params.start_incremental_session(&instance,&witness,
            IncrementalConfig {sparse:SparseDispatchPolicy::ForceGpu,..Default::default()},backend).unwrap();
        let before=session.backend().stats();
        let mut samples=Vec::new();
        for step in 0..24 {
            let i=step%n;
            let next=session.current_witness().wires[4*n+i]+Fr::one();
            let output=session.current_witness().wires[3*n+i]*next;
            let started=Instant::now();
            session.update(&Update {indices:vec![4*n+i,5*n+i],values:vec![next,output]}).unwrap();
            let elapsed=started.elapsed().as_nanos();
            let receipt=session.last_receipt();
            samples.push(elapsed);
            eprintln!("UPDATE n={n} step={step} k={} ns={elapsed} start={} swap={} restart={} units={} cap={} blocks={}",
                receipt.changed,receipt.started,receipt.swapped,receipt.restarted,
                receipt.rebuild_units,receipt.allowance,receipt.blocks);
        }
        samples.sort_unstable();
        let after=session.backend().stats();
        let blocks=session.block_timings();
        eprintln!("SPARSE n={n} stats={:?}",session.sparse_dispatch_stats());
        eprintln!("RESOURCE n={n} plan={:?} diagnostics={:?}",session.resource_plan(),session.backend().resident_diagnostics().unwrap());
        eprintln!("LATENCY n={n} samples={} p50_ns={} p99_ns={} max_ns={} block_count={} block_p50_upper_ns={} block_p99_upper_ns={} block_max_ns={} h2d={} d2h={} fr={} ec={} g1={} g2={}",
            samples.len(),samples[samples.len()/2],samples[(samples.len()*99).div_ceil(100)-1],
            samples[samples.len()-1],blocks.count,blocks.percentile_upper_ns(50).unwrap_or(0),
            blocks.percentile_upper_ns(99).unwrap_or(0),blocks.max_ns,
            after.h2d_bytes-before.h2d_bytes,after.d2h_bytes-before.d2h_bytes,
            after.fr_transform_elements-before.fr_transform_elements,
            after.g1_transform_elements-before.g1_transform_elements,
            after.msm_g1_points-before.msm_g1_points,after.msm_g2_points-before.msm_g2_points);
    }
}
