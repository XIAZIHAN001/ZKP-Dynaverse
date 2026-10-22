#![cfg(feature="icicle")]
use ark_bn254::Fr;
use dynaverse_gpu::IcicleBackend;
use dynaverse_protocol::{dynaverse::DynaverseParams,toy_circuit::ToyCircuit,Update,Witness,VerificationMode,
    incremental_rebuild::{IncrementalConfig,SparseDispatchPolicy,UpdateTimings,SparseProofTimings,CacheCheckTimings,
        operators::{CpuRebuildBackend,RebuildBackend}}};
use rand::{rngs::StdRng,SeedableRng};
use std::time::Instant;

fn edits(w:&Witness,n:usize,roots:&[usize],addition:bool)->Update {
    let mut u=Update {indices:vec![],values:vec![]};
    for &i in roots {
        if addition {
            let right=w.wires[n+i]+Fr::from(1u64);let sum=w.wires[i]+right;
            u.indices.extend([n+i,2*n+i,3*n+i,5*n+i]);
            u.values.extend([right,sum,sum,sum*w.wires[4*n+i]]);
        } else {
            let next=w.wires[4*n+i]+Fr::from(1u64);
            u.indices.extend([4*n+i,5*n+i]);u.values.extend([next,w.wires[3*n+i]*next]);
        }
    }
    u
}
fn config(mode:VerificationMode,sparse:SparseDispatchPolicy)->IncrementalConfig {
    IncrementalConfig {linear_rebuild_reuse:true,unchanged_cache_reuse:true,
        foreground_linear_reuse:true,support_geometry_reuse:true,transform_row_batch:4,
        trace_updates:true,rebuild_cache_mode:mode,sparse,..Default::default()}
}

#[test]
fn product_gpu_matches_cpu_reference_proofs_and_completed_rebuilds() {
    for n in [16,64] {
        let c=ToyCircuit::new(n).unwrap();let (i,w)=c.valid_instance();
        let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(19100+n as u64)).unwrap();
        let mut cpu=p.start_incremental_session(&i,&w,config(VerificationMode::Reference,SparseDispatchPolicy::CpuReference),CpuRebuildBackend::default()).unwrap();
        let backend=IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap();
        let mut gpu=p.start_incremental_session(&i,&w,config(VerificationMode::Product,SparseDispatchPolicy::ForceGpu),backend).unwrap();
        let bytes=gpu.backend().stats().bytes_allocated;let mut equations=0;
        for step in 0..10 {
            let roots=if step==0 {vec![1,2,3]} else {vec![1+step%3]};
            let u=edits(cpu.current_witness(),n,&roots,step>0 && step%3==0);
            let actual=gpu.update(&u).unwrap();let expected=cpu.update(&u).unwrap();
            assert_eq!(actual,expected,"n={n} step={step}");
            assert_eq!(gpu.last_receipt(),cpu.last_receipt());
            assert_eq!(gpu.rebuild_progress(),cpu.rebuild_progress());
            let a=cpu.update_timings().cache_detail;let b=gpu.update_timings().cache_detail;
            assert_eq!(a.equations,b.equations);assert_eq!(a.final_exponentiations,2*b.final_exponentiations);
            assert_eq!(b.final_exponentiations,b.equations);assert!(b.prepared_peak_entries<=5);
            assert!(b.recorded_ns()<=gpu.update_timings().rebuild_ns[6]);equations+=b.equations;
            assert!(p.verifying_key().verify_reference(&i,&actual).unwrap());
            if step==0 {
                assert!(gpu.rebuild_progress().is_none());
                assert_eq!(actual.anchor,p.prove_anchor(&i,gpu.current_witness()).unwrap().anchor);
            }
            assert_eq!(gpu.backend().stats().bytes_allocated,bytes);
        }
        assert!(equations>0);eprintln!("RP_EQUAL n={n} gates={} equations={equations} resident={bytes}",2*n);
    }
}

/// All paths use identical budgets/row batching/reuse. Reference/Product is
/// the sole change unless dispatch_trial=true, which is labelled separately.
fn profile(sizes:&[usize],repeats:usize,dispatch_trial:bool) {
    for &n in sizes {
        for repeat in 0..repeats {
            let c=ToyCircuit::new(n).unwrap();let (instance,witness)=c.valid_instance();
            let started=Instant::now();
            let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(19200+n as u64+repeat as u64)).unwrap();
            eprintln!("RP_SETUP n={n} gates={} repeat={repeat} dispatch={dispatch_trial} ns={}",2*n,started.elapsed().as_nanos());
            let cases=if dispatch_trial {vec!["hotspot"]} else {vec!["growing","hotspot","paid"]};
            for case in cases {
                let routes=if dispatch_trial {3} else {2};
                let mut sessions=Vec::new();let mut allocations=Vec::new();
                for route in 0..routes {
                    let mode=if !dispatch_trial && route==0 {VerificationMode::Reference} else {VerificationMode::Product};
                    let sparse=if !dispatch_trial || route==0 {SparseDispatchPolicy::ForceGpu}
                        else if route==1 {SparseDispatchPolicy::CpuReference}
                        else {SparseDispatchPolicy::Auto {cpu_below:32}};
                    let started=Instant::now();let backend=IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap();
                    eprintln!("RP_BACKEND n={n} repeat={repeat} case={case} dispatch={dispatch_trial} route={route} ns={}",started.elapsed().as_nanos());
                    let started=Instant::now();let mut s=p.start_incremental_session(&instance,&witness,config(mode,sparse),backend).unwrap();
                    let bytes=s.backend().stats().bytes_allocated;
                    eprintln!("RP_INITIAL n={n} repeat={repeat} case={case} dispatch={dispatch_trial} route={route} ns={} resident={bytes}",started.elapsed().as_nanos());
                    if case=="hotspot" {
                        let roots:Vec<usize>=(1..=s.horizon().saturating_sub(1)/2).collect();
                        let seed=edits(&witness,n,&roots,false);let started=Instant::now();s.update(&seed).unwrap();
                        eprintln!("RP_SEED n={n} repeat={repeat} case={case} dispatch={dispatch_trial} route={route} edits={} ns={}",seed.indices.len(),started.elapsed().as_nanos());
                        let started=Instant::now();assert!(p.verifying_key().verify_reference(&instance,s.proof()).unwrap());
                        eprintln!("RP_SEED_VERIFY n={n} repeat={repeat} case={case} dispatch={dispatch_trial} route={route} ns={}",started.elapsed().as_nanos());
                    }
                    sessions.push(s);allocations.push(bytes);
                }
                let steps=if case=="paid" {2} else {8};
                for step in 0..steps {
                    let roots:Vec<usize>=if case=="paid" {(1..=sessions[0].horizon()/2+1).collect()}
                        else {vec![if case=="hotspot" {1} else {1+step}]};
                    let u=edits(sessions[0].current_witness(),n,&roots,case=="paid" && step%2==1);
                    for offset in 0..routes {
                        let route=(step+offset+repeat)%routes;let s=&mut sessions[route];
                        let before=s.backend().stats();let sb=s.sparse_dispatch_stats();
                        let started=Instant::now();let proof=s.update(&u).unwrap();let wall=started.elapsed().as_nanos();
                        let r=s.last_receipt();let t=s.update_timings();let after=s.backend().stats();let sa=s.sparse_dispatch_stats();
                        let progress=s.rebuild_progress();let cache=t.cache_detail;
                        let id=format!("n={n} repeat={repeat} case={case} dispatch={dispatch_trial} route={route} step={step}");
                        eprintln!("RP_SAMPLE {id} gates={} k={} a={} d={} ds={} overlay={} pending={} cursor={} commands={} source={} paid={} generation={} anchor={} wall_ns={wall} total_ns={} delta_ns={} rebase_ns={} live_ns={} prepare_ns={} publish_ns={} norm_ns={} constraints_ns={} work={} allowance={} cache_eq={} cache_fe={} cache_miller={} cache_terms={} g2_builds={} g2_hits={} prepared_peak={} divisors={} resident_calls={} fr_elements={} g1_elements={} sparse_cpu_g1={} sparse_gpu_g1={} sparse_cpu_g2={} sparse_gpu_g2={} sparse_cpu_ntt={} sparse_gpu_ntt={}",
                            2*n,r.changed,r.affected_gates,r.delta_entries,r.frozen_entries,r.overlay_entries,progress.is_some(),
                            progress.map_or(0,|v|v.command),progress.map_or(0,|v|v.commands),progress.map_or(0,|v|v.source_generation),progress.map_or(0,|v|v.paid_changes),
                            proof.generation,proof.anchor_generation,t.total_ns,t.delta_proof_ns,t.rebase_ns,t.live_linear_ns,t.preparation_ns,t.publication_ns,t.normalization_ns,t.constraints_ns,
                            r.rebuild_units,r.allowance,cache.equations,cache.final_exponentiations,cache.miller_calls,cache.pairing_terms,cache.g2_prepare_builds,cache.g2_prepare_hits,cache.prepared_peak_entries,cache.divisors_formed,
                            after.calls-before.calls,after.fr_transform_elements-before.fr_transform_elements,after.g1_transform_elements-before.g1_transform_elements,
                            sa.cpu.g1_points-sb.cpu.g1_points,sa.gpu.g1_points-sb.gpu.g1_points,sa.cpu.g2_points-sb.cpu.g2_points,sa.gpu.g2_points-sb.gpu.g2_points,
                            sa.cpu.ntt_elements-sb.cpu.ntt_elements,sa.gpu.ntt_elements-sb.gpu.ntt_elements);
                        for (scope,names,ns) in [("delta",SparseProofTimings::NAMES.as_slice(),t.delta_detail.ns.as_slice()),
                            ("rebase",SparseProofTimings::NAMES.as_slice(),t.rebase_detail.ns.as_slice()),
                            ("cache",CacheCheckTimings::NAMES.as_slice(),t.cache_detail.ns.as_slice()),
                            ("rebuild",UpdateTimings::REBUILD_NAMES.as_slice(),t.rebuild_ns.as_slice())] {
                            for (j,name) in names.iter().enumerate() {eprintln!("RP_DETAIL {id} scope={scope} phase={name} ns={}",ns[j]);}
                        }
                        assert!(t.delta_detail.recorded_ns()<=t.delta_proof_ns);
                        assert!(t.rebase_detail.recorded_ns()<=t.rebase_ns);
                        assert!(cache.recorded_ns()<=t.rebuild_ns[6]);
                        assert_eq!(s.backend().stats().bytes_allocated,allocations[route]);
                        if case=="paid" {assert!(progress.is_none(),"paid case did not complete rebuild");}
                        let started=Instant::now();assert!(p.verifying_key().verify_reference(&instance,&proof).unwrap());
                        eprintln!("RP_VERIFY {id} ns={}",started.elapsed().as_nanos());
                    }
                    for route in 1..routes {
                        assert_eq!(sessions[route].proof(),sessions[0].proof());
                        assert_eq!(sessions[route].last_receipt(),sessions[0].last_receipt());
                        assert_eq!(sessions[route].rebuild_progress(),sessions[0].rebuild_progress());
                    }
                }
            }
        }
    }
}

#[test]
#[ignore="explicit serial local n64/256 same-work ablation, three repeats"]
fn local_product_profile() {profile(&[64,256],3,false);}
#[test]
#[ignore="after small-scale admission; includes fully completed n1024 paid rebuilds"]
fn local_product_profile_1024() {profile(&[1024],1,false);}
#[test]
#[ignore="explicit Auto32 threshold trial, not a production calibration or fusion result"]
fn local_product_dispatch_trial() {profile(&[256],1,true);}
