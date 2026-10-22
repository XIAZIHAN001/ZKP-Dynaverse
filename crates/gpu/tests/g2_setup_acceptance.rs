#![cfg(feature="icicle")]
//! G2-6 whole-setup and GPU-produced-SRS acceptance; explicit probes are ignored
//! by the ordinary suite and do not silently skip when a GPU is unavailable.
use dynaverse_gpu::{FixedBaseCpuBackend,IcicleBackend,SetupMode};
use dynaverse_gpu::g2_validation::{G2KernelConfig,G2ValidationMode};
use dynaverse_protocol::{dynaverse::DynaverseParams,toy_circuit::ToyCircuit};
use rand::{rngs::StdRng,SeedableRng};
use ark_bn254::Fr;
use ark_ff::One;
use anyhow::{ensure,Result};
use dynaverse_protocol::{DynaverseProof,Update};
use dynaverse_protocol::incremental_rebuild::{IncrementalConfig,SparseDispatchPolicy,operators::{CpuRebuildBackend,RebuildBackend}};
use std::{io::Write,time::Instant};
#[path="support/g2_artifacts.rs"]
mod g2_artifacts;

#[test]
fn artifact_output_override_requires_an_absolute_path() {
    assert!(g2_artifacts::resolve(Some("")).is_err());
    assert!(g2_artifacts::resolve(Some("relative-output")).is_err());
    let absolute=std::env::temp_dir().join("g2-artifact-path-check");
    assert_eq!(g2_artifacts::resolve(absolute.to_str()).unwrap(),absolute);
    assert!(g2_artifacts::resolve(None).unwrap().is_absolute());
}

const MODES:[G2ValidationMode;6]=[
    G2ValidationMode::CpuReference,G2ValidationMode::CpuFrobenius127,G2ValidationMode::CpuFrobenius63,
    G2ValidationMode::GpuFullOrder,G2ValidationMode::GpuFrobenius127,G2ValidationMode::GpuFrobenius63,
];

#[test]
fn setup_profile_sizes_are_bounded_before_gpu_allocation() {
    assert_eq!(setup_sizes(None).unwrap(),vec![64,256,1024]);
    assert_eq!(setup_sizes(Some("64,256,1024,4096")).unwrap(),vec![64,256,1024,4096]);
    for bad in ["", "0", "128", "65536", "64,64", "64,", "4096,1024"] {
        assert!(setup_sizes(Some(bad)).is_err(),"accepted {bad}");
    }
}

#[test]
fn gpu_validated_srs_survives_initial_and_paid_rebuild_with_two_cpu_oracles() {
    let n=16;let circuit=ToyCircuit::new(n).unwrap();let seed=1760000+n as u64;
    let oracle=DynaverseParams::setup(circuit.clone(),&mut StdRng::seed_from_u64(seed)).unwrap();
    for mode in MODES {
        let gpu=IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap();
        gpu.set_setup_mode(SetupMode::TableBatchNormalize).unwrap();
        gpu.set_g2_validation_mode(mode).unwrap();
        gpu.set_g2_kernel_config(G2KernelConfig{threads:64,schedule:1,layout:0,fused:0}).unwrap();
        let params=DynaverseParams::setup_with_backend(circuit.clone(),&mut StdRng::seed_from_u64(seed),&gpu).unwrap();
        let result=exercise_chain(&params,&oracle,gpu,n);
        assert!(result.initial_ns>0 && result.rebuild_ns>0 && result.oracle_ns>0);
        assert!(result.resident_bytes>0 && result.journal_peak_bytes<=result.transaction_bytes);
    }
}

fn setup_sizes(requested:Option<&str>)->Result<Vec<usize>> {
    let Some(text)=requested else {return Ok(vec![64,256,1024]);};
    let mut sizes=Vec::new();
    for part in text.split(',') {
        let value:usize=part.parse()?;
        ensure!([64,256,1024,4096].contains(&value),"unsupported G2 setup size");
        ensure!(sizes.last().is_none_or(|last|*last<value),"sizes must be increasing and unique");
        sizes.push(value);
    }
    Ok(sizes)
}

struct ChainResult {
    initial_ns:u128,rebuild_ns:u128,oracle_ns:u128,
    resident_bytes:u64,transaction_bytes:u64,journal_peak_bytes:u64,
}

fn exercise_chain(params:&DynaverseParams,oracle:&DynaverseParams,gpu:IcicleBackend,n:usize)->ChainResult {
    let circuit=ToyCircuit::new(n).unwrap();let (instance,witness)=circuit.valid_instance();
    let started=Instant::now();
    let dense=oracle.prove_with_aux(&instance,&witness).unwrap().0;
    let mut cpu=oracle.start_incremental_session(&instance,&witness,IncrementalConfig::default(),CpuRebuildBackend::default()).unwrap();
    let mut oracle_ns=started.elapsed().as_nanos();
    assert_eq!(*cpu.proof(),dense);
    let started=Instant::now();
    let mut session=params.start_incremental_session(&instance,&witness,
        IncrementalConfig{sparse:SparseDispatchPolicy::ForceGpu,..Default::default()},gpu).unwrap();
    let initial_ns=started.elapsed().as_nanos();
    let started=Instant::now();
    assert_eq!(*session.proof(),dense);
    assert!(oracle.verifying_key().verify(&instance,session.proof()).unwrap());
    assert!(oracle.verifying_key().verify_reference(&instance,session.proof()).unwrap());
    oracle_ns+=started.elapsed().as_nanos();
    let resident_bytes=session.backend().stats().bytes_allocated;
    let transaction_bytes=session.resource_plan().transaction_bytes;
    let mut update=Update{indices:Vec::with_capacity(2*n),values:Vec::with_capacity(2*n)};
    for i in 0..n {
        let next=witness.wires[4*n+i]+Fr::one();
        update.indices.push(4*n+i);update.values.push(next);
        update.indices.push(5*n+i);update.values.push(witness.wires[3*n+i]*next);
    }
    let started=Instant::now();let expected=cpu.update(&update).unwrap();oracle_ns+=started.elapsed().as_nanos();
    let started=Instant::now();let actual=session.update(&update).unwrap();let rebuild_ns=started.elapsed().as_nanos();
    assert_eq!(actual,expected);assert_eq!(session.last_receipt(),cpu.last_receipt());
    assert!(session.last_receipt().large_batch && session.last_receipt().swapped);
    assert!(session.last_receipt().rebuild_units<=session.last_receipt().allowance);
    assert_eq!(session.backend().stats().bytes_allocated,resident_bytes);
    let started=Instant::now();
    assert_eq!(actual.anchor,oracle.prove_anchor(&instance,session.current_witness()).unwrap().anchor);
    assert!(oracle.verifying_key().verify(&instance,&actual).unwrap());
    assert!(oracle.verifying_key().verify_reference(&instance,&actual).unwrap());
    oracle_ns+=started.elapsed().as_nanos();
    let journal_peak_bytes=session.backend().resident_diagnostics().unwrap().journal_peak_bytes;
    assert!(journal_peak_bytes<=transaction_bytes);
    let stats=session.backend().stats();
    assert!(stats.fr_transform_elements>0 && stats.g1_transform_elements>0 && stats.msm_g1_points>0 && stats.msm_g2_points>0);
    ChainResult{initial_ns,rebuild_ns,oracle_ns,resident_bytes,transaction_bytes,journal_peak_bytes}
}

fn configured_gpu(mode:G2ValidationMode)->IcicleBackend {
    let gpu=IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap();
    gpu.set_setup_mode(SetupMode::TableBatchNormalize).unwrap();
    gpu.set_g2_validation_mode(mode).unwrap();
    gpu.set_g2_kernel_config(G2KernelConfig::default()).unwrap();gpu
}

fn setup_sample(n:usize,seed:u64,mode:Option<G2ValidationMode>,sample:i32,cpu:&FixedBaseCpuBackend,gpu:&IcicleBackend)->(DynaverseParams,String) {
    let circuit=ToyCircuit::new(n).unwrap();let mut rng=StdRng::seed_from_u64(seed);
    let mut row=vec![mode.map_or_else(||"natural_cpu".into(),|value|format!("{value:?}")),n.to_string(),sample.to_string()];
    if let Some(mode)=mode {
        gpu.set_g2_validation_mode(mode).unwrap();
        let s=gpu.setup_stats().unwrap();let h=gpu.setup_host_stats();let a=gpu.g2_validation_stats().unwrap();
        let host=gpu.g2_validation_host_stats();let traffic=gpu.g2_validation_traffic().unwrap();let points=gpu.setup_point_counts();
        let started=Instant::now();let params=DynaverseParams::setup_with_backend(circuit,&mut rng,gpu).unwrap();let wall=started.elapsed().as_nanos();
        row.push(wall.to_string());let end=gpu.setup_stats().unwrap();let hh=gpu.setup_host_stats();let b=gpu.g2_validation_stats().unwrap();
        let host_end=gpu.g2_validation_host_stats();let traffic_end=gpu.g2_validation_traffic().unwrap();let point_end=gpu.setup_point_counts();
        for value in [point_end.0-points.0,point_end.1-points.1,end.native_wall_ns-s.native_wall_ns,
            hh.g1_validate_ns-h.g1_validate_ns,hh.g2_validate_ns-h.g2_validate_ns] {row.push(value.to_string());}
        for index in 0..5 {row.push((end.device_ns[index]-s.device_ns[index]).to_string());}
        for value in [b.parse_curve_device_ns-a.parse_curve_device_ns,b.subgroup_device_ns-a.subgroup_device_ns,b.reduce_device_ns-a.reduce_device_ns,
            host_end.host_parse_ns-host.host_parse_ns,host_end.host_curve_ns-host.host_curve_ns,
            host_end.host_subgroup_ns-host.host_subgroup_ns,host_end.host_publish_decode_ns-host.host_publish_decode_ns,
            traffic_end.scalar_h2d_bytes-traffic.scalar_h2d_bytes,traffic_end.point_h2d_bytes-traffic.point_h2d_bytes,
            traffic_end.point_d2h_bytes-traffic.point_d2h_bytes,traffic_end.status_d2h_bytes-traffic.status_d2h_bytes] {row.push(value.to_string());}
        assert_eq!(traffic_end.point_h2d_bytes-traffic.point_h2d_bytes,0);
        return (params,row.join(",")+"\n");
    }
    let started=Instant::now();let params=DynaverseParams::setup_with_backend(circuit,&mut rng,cpu).unwrap();
    row.push(started.elapsed().as_nanos().to_string());row.extend(vec![String::new();21]);
    (params,row.join(",")+"\n")
}

#[test]
#[ignore="explicit G2-6 whole setup samples and initial/paid rebuild acceptance"]
fn full_setup_and_proof_acceptance_profile() {
    let requested=std::env::var("DVGPU_G2_SETUP_SIZES").ok();let sizes=setup_sizes(requested.as_deref()).unwrap();
    let directory=g2_artifacts::directory().unwrap();
    let mut output=std::io::BufWriter::new(std::fs::File::create(directory.join("G2-6-setup.csv")).unwrap());
    let mut proofs=std::io::BufWriter::new(std::fs::File::create(directory.join("G2-6-proof.csv")).unwrap());
    writeln!(output,"route,n,sample,wall_ns,g1_points,g2_points,native_wall_ns,g1_host_validate_ns,g2_host_validate_ns,device_table_ns,device_h2d_ns,device_generate_ns,device_normalize_ns,device_d2h_ns,g2_curve_ns,g2_subgroup_ns,g2_reduce_ns,g2_host_parse_ns,g2_host_curve_ns,g2_host_subgroup_ns,g2_host_publish_ns,scalar_h2d_bytes,point_h2d_bytes,point_d2h_bytes,status_d2h_bytes,comparison_ns").unwrap();
    writeln!(proofs,"route,n,initial_ns,rebuild_ns,oracle_ns,resident_bytes,transaction_bytes,journal_peak_bytes").unwrap();
    let routes=[None,Some(MODES[0]),Some(MODES[1]),Some(MODES[2]),Some(MODES[3]),Some(MODES[4]),Some(MODES[5])];
    for n in sizes {
        let seed=1760000+n as u64;let circuit=ToyCircuit::new(n).unwrap();let (instance,witness)=circuit.valid_instance();
        let oracle=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(seed)).unwrap();
        let expected:DynaverseProof=oracle.prove_with_aux(&instance,&witness).unwrap().0;
        let cpu=FixedBaseCpuBackend::default();let gpu=configured_gpu(MODES[0]);
        // First call has a fresh table/owner. Context creation is outside wall.
        for mode in routes {
            let fresh_cpu=FixedBaseCpuBackend::default();let fresh_gpu=configured_gpu(mode.unwrap_or(MODES[0]));
            let (params,row)=setup_sample(n,seed,mode,-1,&fresh_cpu,&fresh_gpu);
            let checked=Instant::now();
            assert_eq!(params.prove_with_aux(&instance,&witness).unwrap().0,expected);
            writeln!(output,"{},{}",row.trim_end(),checked.elapsed().as_nanos()).unwrap();output.flush().unwrap();
        }
        let mut last=Vec::new();for _ in routes {last.push(None);}
        for round in -2i32..10 {
            for k in 0..routes.len() {
                let index=(k+(round+2) as usize*3)%routes.len();let mode=routes[index];
                let (params,row)=setup_sample(n,seed,mode,round,&cpu,&gpu);
                // Every completed setup is checked outside its wall timer.
                let checked=Instant::now();
                assert_eq!(params.prove_with_aux(&instance,&witness).unwrap().0,expected);
                if round>=0 {writeln!(output,"{},{}",row.trim_end(),checked.elapsed().as_nanos()).unwrap();output.flush().unwrap();}
                last[index]=Some(params);
            }
            eprintln!("G2_SETUP_PROGRESS n={n} sample={round} routes=7");
        }
        drop(gpu);
        for index in 4..7 {
            let mode=routes[index].unwrap();let gpu=configured_gpu(mode);
            let result=exercise_chain(last[index].as_ref().unwrap(),&oracle,gpu,n);
            writeln!(proofs,"{mode:?},{n},{},{},{},{},{},{}",result.initial_ns,result.rebuild_ns,result.oracle_ns,
                result.resident_bytes,result.transaction_bytes,result.journal_peak_bytes).unwrap();proofs.flush().unwrap();
            eprintln!("G2_CHAIN_PASS n={n} mode={mode:?} initial_ns={} rebuild_ns={} journal={}/{}",result.initial_ns,result.rebuild_ns,result.journal_peak_bytes,result.transaction_bytes);
        }
    }
}
