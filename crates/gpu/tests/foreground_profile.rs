#![cfg(feature="icicle")]
use ark_bn254::Fr;
use dynaverse_gpu::IcicleBackend;
use dynaverse_protocol::{dynaverse::DynaverseParams,toy_circuit::ToyCircuit,Update,
    incremental_rebuild::{IncrementalConfig,SparseDispatchPolicy,UpdateTimings,operators::RebuildBackend}};
use rand::{rngs::StdRng,SeedableRng};
use std::time::Instant;

fn profile(sizes:&[usize]) {
    for &n in sizes {
        let circuit=ToyCircuit::new(n).unwrap();let (instance,witness)=circuit.valid_instance();
        let started=Instant::now();
        let p=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(18900+n as u64)).unwrap();
        eprintln!("FG_CASE n={n} gates={} setup_ns={}",2*n,started.elapsed().as_nanos());
        for case in ["growing","hotspot"] {
            let mut sessions=Vec::new();let mut allocations=Vec::new();
            for route in 0..4 {
                let backend=IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap();
                let cfg=IncrementalConfig {linear_rebuild_reuse:true,unchanged_cache_reuse:true,
                    foreground_linear_reuse:route>=1,support_geometry_reuse:route>=2,
                    transform_row_batch:if route==3 {4} else {1},trace_updates:true,
                    sparse:SparseDispatchPolicy::ForceGpu,..Default::default()};
                let started=Instant::now();let mut s=p.start_incremental_session(&instance,&witness,cfg,backend).unwrap();
                let bytes=s.backend().stats().bytes_allocated;
                eprintln!("FG_INITIAL n={n} case={case} route={route} ns={} resident={bytes}",started.elapsed().as_nanos());
                if case=="hotspot" {
                    let gates=if n==64 {1} else if n==256 {3} else {7};
                    let mut seed=Update {indices:vec![],values:vec![]};
                    for i in 1..=gates {
                        let next=witness.wires[4*n+i]+Fr::from(1u64);
                        seed.indices.extend([4*n+i,5*n+i]);seed.values.extend([next,witness.wires[3*n+i]*next]);
                    }
                    let started=Instant::now();s.update(&seed).unwrap();
                    eprintln!("FG_SEED n={n} case={case} route={route} edits={} ns={}",seed.indices.len(),started.elapsed().as_nanos());
                    assert!(p.verifying_key().verify_reference(&instance,s.proof()).unwrap());
                }
                sessions.push(s);allocations.push(bytes);
            }
            let steps=if n==1024 {32} else {8};
            for step in 0..steps {
                let i=if case=="hotspot" {1} else {step+1};
                let next=sessions[0].current_witness().wires[4*n+i]+Fr::from(1u64);
                let u=Update {indices:vec![4*n+i,5*n+i],values:vec![next,sessions[0].current_witness().wires[3*n+i]*next]};
                for offset in 0..4 {
                    let route=(step+offset)%4;let s=&mut sessions[route];
                    let before=s.backend().stats();let sb=s.sparse_dispatch_stats();
                    let started=Instant::now();let proof=s.update(&u).unwrap();let wall=started.elapsed().as_nanos();
                    let r=s.last_receipt();let t=s.update_timings();let after=s.backend().stats();let sa=s.sparse_dispatch_stats();
                    let progress=s.rebuild_progress();
                    eprintln!("FG_UPDATE n={n} case={case} route={route} step={step} generation={} anchor_generation={} pending={} source_generation={} cursor={} commands={} paid={} r={} k={} a={} d={} ds={} o={} start={} swap={} restart={} live={} live_inputs={} foreground_units={} geometry_hit={} geometry_builds={} cache_points={} wall_ns={wall} total_ns={} norm_ns={} constraints_ns={} prepare_ns={} publish_ns={} live_ns={} delta_ns={} rebase_ns={} work={} allowance={} resident_g1={} resident_g2={} sparse_g1={} sparse_g2={} sparse_ntt={} fr_elements={} g1_elements={} resident_calls={}",
                        proof.generation,proof.anchor_generation,progress.is_some(),progress.map_or(-1,|p|p.source_generation as i64),
                        progress.map_or(0,|p|p.command),progress.map_or(0,|p|p.commands),progress.map_or(0,|p|p.paid_changes),
                        r.raw_edits,r.changed,r.affected_gates,r.delta_entries,r.frozen_entries,r.overlay_entries,r.started,r.swapped,r.restarted,
                        r.foreground_linear_reused,r.foreground_linear_inputs,r.foreground_linear_work.units().unwrap(),
                        r.support_geometry_reused,r.support_geometry_builds,r.cache_reused_points,
                        t.total_ns,t.normalization_ns,t.constraints_ns,t.preparation_ns,t.publication_ns,t.live_linear_ns,t.delta_proof_ns,t.rebase_ns,
                        r.rebuild_units,r.allowance,after.msm_g1_points-before.msm_g1_points,after.msm_g2_points-before.msm_g2_points,
                        sa.cpu.g1_points+sa.gpu.g1_points-sb.cpu.g1_points-sb.gpu.g1_points,
                        sa.cpu.g2_points+sa.gpu.g2_points-sb.cpu.g2_points-sb.gpu.g2_points,
                        sa.cpu.ntt_elements+sa.gpu.ntt_elements-sb.cpu.ntt_elements-sb.gpu.ntt_elements,
                        after.fr_transform_elements-before.fr_transform_elements,after.g1_transform_elements-before.g1_transform_elements,
                        after.calls-before.calls);
                    for (phase,name) in UpdateTimings::REBUILD_NAMES.iter().enumerate() {
                        eprintln!("FG_PHASE n={n} case={case} route={route} step={step} phase={name} ns={} calls={}",t.rebuild_ns[phase],t.rebuild_calls[phase]);
                    }
                    let started=Instant::now();assert!(p.verifying_key().verify_reference(&instance,&proof).unwrap());
                    eprintln!("FG_VERIFY n={n} case={case} route={route} step={step} ns={}",started.elapsed().as_nanos());
                    assert_eq!(s.backend().stats().bytes_allocated,allocations[route]);
                }
                for route in 1..4 {assert_eq!(sessions[route].current_witness(),sessions[0].current_witness());}
            }
        }
    }
}

#[test]
#[ignore="explicit four-route local ablation, n64/256 admission first"]
fn local_foreground_ablation() {profile(&[64,256]);}
#[test]
#[ignore="explicit n1024 enlargement after admission; no H200 claim"]
fn local_foreground_ablation_1024() {profile(&[1024]);}
