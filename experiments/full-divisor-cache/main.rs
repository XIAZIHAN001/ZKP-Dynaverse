//! Paired full-flow control: only the public affine-divisor table changes.
use anyhow::{ensure,Result};
use ark_bn254::Fr;
use ark_ff::{BigInteger,PrimeField,UniformRand};
#[cfg(feature="gpu")]
use ark_bn254::{G1Affine,G1Projective};
#[cfg(feature="gpu")]
use ark_ec::CurveGroup;
#[cfg(feature="gpu")]
use ark_ff::Zero;
#[cfg(feature="gpu")]
use dynaverse_gpu::{G2ValidationMode,IcicleBackend,SetupMode};
use dynaverse_protocol::{Update,Witness,VerificationMode,dynaverse::DynaverseParams,
    toy_circuit::ToyCircuit,trace::CpuTrace,
    incremental_rebuild::{IncrementalConfig,SparseDispatchPolicy,UpdateTimings,
        CacheCheckTimings,LinearTimings,SparseProofTimings,
        operators::{RebuildBackend,CpuRebuildBackend}}};
use rand::{rngs::StdRng,SeedableRng};
use serde_json::{json,Value};
use std::time::Instant;
#[path="../adaptive-fullopt/strategy.rs"]
mod strategy;
#[path="../adaptive-fullopt/readback.rs"]
#[cfg(feature="gpu")]
mod readback;

fn emit(value:Value) {println!("ROW {value}");}
#[cfg(feature="gpu")]
fn fullopt_backend(n:usize,optimized:bool,affine:bool)->Result<IcicleBackend> {
    let p=strategy::Strategy::for_size(n).map_err(anyhow::Error::msg)?;
    let mut b=IcicleBackend::with_resident_limits(p.limit_bytes,p.undo_bytes)?;
    b.set_bounded_g1_decode(optimized);b.set_device_affine_readback(optimized && affine);
    b.set_setup_mode(SetupMode::TableBatchNormalize)?;
    b.set_g2_validation_mode(G2ValidationMode::GpuFrobenius63)?;
    Ok(b)
}
fn config(n:usize,enabled:bool,gpu:bool)->Result<IncrementalConfig> {
    let p=strategy::Strategy::for_size(n).map_err(anyhow::Error::msg)?;
    Ok(IncrementalConfig {window:Some(p.window),trace_updates:true,memory_limit_bytes:p.limit_bytes,
        sparse:if gpu {SparseDispatchPolicy::ForceGpu} else {SparseDispatchPolicy::CpuReference},
        linear_rebuild_reuse:true,unchanged_cache_reuse:true,foreground_linear_reuse:true,
        freeze_linear_capture_limit:Some(128),frozen_g2_reuse_limit:Some(128),support_geometry_reuse:true,
        transform_row_batch:16,transform_packet_max_len:p.packet_cap,
        rebuild_cache_mode:VerificationMode::Product,rebuild_prepared_cache_roots:Some(p.cache_roots),
        rebuild_cache_readback_roots:Some(p.cache_roots),rebuild_divisor_cache:enabled,..Default::default()})
}
fn edits(w:&Witness,n:usize,step:usize)->Result<Update> {
    let mut rng=StdRng::seed_from_u64(22100+n as u64*100+step as u64);
    let addition=step%2==1;let roots=if addition {4} else {8};
    let mut u=Update {indices:Vec::new(),values:Vec::new()};
    for i in 1..=roots {
        let value=Fr::rand(&mut rng);
        let old=if addition {w.wires[n+i]} else {w.wires[4*n+i]};
        ensure!((value-old).into_bigint().num_bits()>128,"eta is not full width");
        if addition {
            let sum=w.wires[i]+value;u.indices.extend([n+i,2*n+i,3*n+i,5*n+i]);
            u.values.extend([value,sum,sum,sum*w.wires[4*n+i]]);
        } else {u.indices.extend([4*n+i,5*n+i]);u.values.extend([value,w.wires[3*n+i]*value]);}
    }
    Ok(u)
}
fn named(names:&[&str],ns:&[u128])->Value {
    let mut out=serde_json::Map::new();
    for (key,value) in names.iter().zip(ns) {out.insert((*key).into(),json!(value));}
    Value::Object(out)
}
fn timings(t:&UpdateTimings)->Value {
    json!({"enabled":t.enabled,"total_ns":t.total_ns,"normalization_ns":t.normalization_ns,
        "constraints_ns":t.constraints_ns,"preparation_ns":t.preparation_ns,"publication_ns":t.publication_ns,
        "live_linear_ns":t.live_linear_ns,"delta_proof_ns":t.delta_proof_ns,"rebase_ns":t.rebase_ns,
        "rebuild_ns":named(&UpdateTimings::REBUILD_NAMES,&t.rebuild_ns),
        "cache_ns":named(&CacheCheckTimings::NAMES,&t.cache_detail.ns),
        "linear_ns":named(&LinearTimings::NAMES,&t.linear_detail.ns),
        "delta_detail_ns":named(&SparseProofTimings::NAMES,&t.delta_detail.ns),
        "rebase_detail_ns":named(&SparseProofTimings::NAMES,&t.rebase_detail.ns)})
}
fn run<B:RebuildBackend>(n:usize,gpu:bool,make:impl Fn(usize,bool,bool)->Result<B>,
    calibrate:impl Fn(usize,strategy::Strategy)->Result<bool>)->Result<()> {
    ensure!([256,1024,65536].contains(&n),"only admitted n256/1024/65536");
    let policy=strategy::Strategy::for_size(n).map_err(anyhow::Error::msg)?;
    let tag=if gpu {"gpu"} else {"cpu"};
    emit(json!({"kind":"divisor_start","schema":1,"backend":tag,"n":n,
        "window":policy.window,"prepared_roots":64,"readback_roots":64,"row_batch":16,
        "packet_cap":policy.packet_cap,"native_owner_limit_bytes":policy.limit_bytes,
        "undo_bytes":policy.undo_bytes,"steps":4,"measured_steps":2,"construction_chunk_roots":64}));
    let circuit=ToyCircuit::new(n)?;let(instance,witness)=circuit.valid_instance();
    let setup=make(n,false,false)?;let mut trace=CpuTrace::new();let started=Instant::now();
    let params=DynaverseParams::setup_with_backend_traced(circuit,
        &mut StdRng::seed_from_u64(2026102220+n as u64),&setup,&mut trace)?;
    emit(json!({"kind":"divisor_setup","n":n,"ns":started.elapsed().as_nanos()}));
    drop(setup);
    let affine=calibrate(n,policy)?;
    ensure!(!affine || policy.calibrate_affine,"calibration enabled an unapproved readback mode");
    let vk=params.verifying_key();let mut sessions=Vec::new();let mut allocations=Vec::new();
    for route in 0..2 {
        let started=Instant::now();
        let s=params.start_incremental_session(&instance,&witness,config(n,route==1,gpu)?,make(n,true,affine)?)?;
        let ns=started.elapsed().as_nanos();let p=s.resource_plan();let initial=s.initial_cache_stats();
        let cache=s.cache_divisor_stats();let prepared=s.cache_preparation_stats().unwrap();let budget=s.budget();
        ensure!(p.upper_bytes<=policy.limit_bytes && p.transaction_bytes<=policy.undo_bytes,"resource admission failed");
        ensure!(initial.equations==3*n as u64 && initial.divisors_formed==if route==1 {0} else {n as u64}
            && initial.divisor_cache_hits==if route==1 {n as u64} else {0},"initial work differs");
        if let Some(c)=cache {
            ensure!(route==1 && c.roots==n && c.construction_divisors==n
                && c.retained_bytes<=c.host_upper_bytes && c.host_upper_bytes==p.cache_divisor_host_bytes,
                "one-time divisor construction not accounted");
        } else {ensure!(route==0 && p.cache_divisor_host_bytes==0,"divisor table missing");}
        let started=Instant::now();ensure!(vk.verify_reference(&instance,s.proof())?,"initial Reference failed");
        let verify_ns=started.elapsed().as_nanos();allocations.push(s.backend().stats().bytes_allocated);
        let table=cache.map(|c|json!({"roots":c.roots,"construction_divisors":c.construction_divisors,
            "retained_bytes":c.retained_bytes,"host_upper_bytes":c.host_upper_bytes,"precompute_ns":c.precompute_ns}));
        let resource=json!({"upper_bytes":p.upper_bytes,"resident_bytes":p.resident_bytes,
            "workspace_reserve":p.workspace_reserve,"transaction_bytes":p.transaction_bytes,"tile_len":p.tile_len,
            "frozen_linear_bytes":p.frozen_linear_bytes,"global_g2_host_bytes":p.global_g2_host_bytes,
            "global_g2_workspace_bytes":p.global_g2_workspace_bytes,
            "cache_preparation_host_bytes":p.cache_preparation_host_bytes,
            "cache_readback_host_bytes":p.cache_readback_host_bytes,"cache_divisor_host_bytes":p.cache_divisor_host_bytes});
        emit(json!({"kind":"divisor_initial","n":n,"route":route,"ns":ns,"verify_ns":verify_ns,
            "correct":true,"affine":affine,"table":table,"resource":resource,
            "prepared_roots":prepared.roots,"prepared_precompute_ns":prepared.precompute_ns,
            "initial":{"equations":initial.equations,"divisors_formed":initial.divisors_formed,"hits":initial.divisor_cache_hits},
            "budget":{"q":budget.q.to_string(),"total":budget.total.to_string(),
                "max_chunk":budget.max_chunk.to_string(),"large":budget.large.to_string()}}));
        sessions.push(s);
    }
    ensure!(sessions[0].proof()==sessions[1].proof() && sessions[0].current_witness()==sessions[1].current_witness(),"initial pair differs");
    emit(json!({"kind":"divisor_pair","n":n,"step":-1,"proof_equal":true,"witness_equal":true}));
    for step in 0..4 {
        let u=edits(sessions[0].current_witness(),n,step)?;let mut arithmetic=Vec::new();
        for offset in 0..2 {
            let route=if (step/2+step)%2==0 {offset} else {1-offset};let s=&mut sessions[route];
            let before=s.backend().stats();let sparse_before=s.sparse_dispatch_stats();let started=Instant::now();
            let proof=s.update(&u)?;let ns=started.elapsed().as_nanos();let r=*s.last_receipt();
            let t=*s.update_timings();let c=t.cache_detail;let after=s.backend().stats();let sparse=s.sparse_dispatch_stats();
            ensure!(r.changed==16 && r.large_batch && r.swapped && !r.restarted && s.rebuild_progress().is_none(),"incomplete paid rebuild");
            ensure!(r.rebuild_units<=r.allowance && r.allowance==16*s.budget().q,"work quota differs");
            ensure!(after.bytes_allocated==allocations[route],"resident buffers grew");
            let fallback=(2*(n-64)) as u64;
            ensure!(c.divisors_formed==if route==1 {0} else {fallback}
                && c.divisor_cache_hits==if route==1 {fallback} else {0},"divisor cache did not execute");
            ensure!(c.equations==2*n as u64 && c.miller_calls==c.equations && c.final_exponentiations==c.equations
                && c.pairing_terms==2*c.equations,"independent equations changed");
            ensure!(c.g2_prepare_builds==fallback*5/4 && c.g2_prepare_hits+c.g2_prepare_builds==4*n as u64,
                "prepared work changed");
            ensure!(r.frozen_linear_captured && r.foreground_linear_reused && r.global_g2_reused && r.cache_reused_points==n,
                "full optimization mechanism missed");
            ensure!(t.enabled && t.total_ns<=ns && c.recorded_ns()<=t.rebuild_ns[6]
                && t.linear_detail.recorded_ns()<=t.rebuild_ns[1],"nested timing exceeds parent");
            if gpu {ensure!(sparse.cpu.g1_points==sparse_before.cpu.g1_points,"ForceGpu fell back");}
            let fr=after.fr_transform_elements-before.fr_transform_elements;
            let g1=after.g1_transform_elements-before.g1_transform_elements;
            arithmetic.push((fr,g1,c.equations,c.miller_calls,c.final_exponentiations,c.g2_prepare_builds));
            let started=Instant::now();ensure!(vk.verify_reference(&instance,&proof)?,"Reference failed");
            let verify_ns=started.elapsed().as_nanos();
            let counts=json!({"equations":c.equations,"miller_calls":c.miller_calls,"final_exponentiations":c.final_exponentiations,
                "pairing_terms":c.pairing_terms,"divisors_formed":c.divisors_formed,"divisor_cache_hits":c.divisor_cache_hits,
                "g2_prepare_builds":c.g2_prepare_builds,"g2_prepare_hits":c.g2_prepare_hits,
                "read_batches":c.read_batches,"read_points":c.read_points,"max_read_roots":c.max_read_roots});
            emit(json!({"kind":"divisor_update","n":n,"step":step,"route":route,"order":offset,"measured":step>=2,
                "ns":ns,"verify_ns":verify_ns,"changed":r.changed,"correct":true,"large_batch":r.large_batch,
                "swapped":r.swapped,"pending":s.rebuild_progress().is_some(),"restarted":r.restarted,
                "generation":proof.generation,"anchor_generation":proof.anchor_generation,
                "rebuild_units":r.rebuild_units.to_string(),"allowance":r.allowance.to_string(),
                "resident_bytes":after.bytes_allocated,"fr_elements":fr,"g1_elements":g1,
                "sparse_cpu_g1_points":sparse.cpu.g1_points-sparse_before.cpu.g1_points,
                "sparse_gpu_g1_points":sparse.gpu.g1_points-sparse_before.gpu.g1_points,
                "frozen_linear_captured":r.frozen_linear_captured,"foreground_linear_reused":r.foreground_linear_reused,
                "global_g2_reused":r.global_g2_reused,"global_g2_inputs":r.global_g2_inputs,
                "cache_reused_points":r.cache_reused_points,"transform_packet_calls":r.transform_packet_calls,
                "transform_packet_elements":r.transform_packet_elements,"cache":counts,"timings":timings(&t)}));
        }
        ensure!(sessions[0].proof()==sessions[1].proof() && sessions[0].current_witness()==sessions[1].current_witness(),"pair differs");
        ensure!(arithmetic[0]==arithmetic[1],"paired arithmetic/equations differ");
        emit(json!({"kind":"divisor_pair","n":n,"step":step,"proof_equal":true,"witness_equal":true,"arithmetic_equal":true}));
    }
    let started=Instant::now();let dense=params.prove_anchor(&instance,sessions[0].current_witness())?;
    ensure!(sessions.iter().all(|s|s.proof().anchor==dense.anchor),"CPU dense anchor differs");
    emit(json!({"kind":"divisor_dense","n":n,"ns":started.elapsed().as_nanos(),"correct":true}));
    println!("FULL_DIVISOR_OK backend={tag} n={n} measured_steps=2");Ok(())
}
fn main()->Result<()> {
    let args:Vec<_>=std::env::args().collect();ensure!(args.len()==3,"usage: full_divisor_probe <cpu|gpu> <n>");
    let n=args[2].parse()?;
    match args[1].as_str() {
        "cpu"=>run(n,false,|_,_,_|Ok(CpuRebuildBackend::default()),|_,_|Ok(false)),
        #[cfg(feature="gpu")]
        "gpu"=>run(n,true,fullopt_backend,readback::calibrate),
        _=>anyhow::bail!("expected cpu, or gpu built with --features gpu"),
    }
}
