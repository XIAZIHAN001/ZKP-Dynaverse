use ark_bn254::Fr;
use ark_ff::{BigInteger,PrimeField,UniformRand};
use dynaverse_protocol::{dynaverse::DynaverseParams,toy_circuit::ToyCircuit,Update,Witness,VerificationMode,
    incremental_rebuild::{IncrementalConfig,SparseDispatchPolicy,UpdateTimings,LinearTimings,CacheCheckTimings,SparseProofTimings,
        operators::RebuildBackend}};
use rand::{rngs::StdRng,SeedableRng};
use std::time::Instant;

pub fn config(route:&str,policy:SparseDispatchPolicy)->IncrementalConfig {
    IncrementalConfig {window:Some(4),linear_rebuild_reuse:true,unchanged_cache_reuse:true,
        foreground_linear_reuse:true,freeze_linear_capture_limit:if route=="r1" {None} else {Some(128)},
        frozen_g2_reuse_limit:if route=="g2" || route=="row16" {Some(128)} else {None},
        support_geometry_reuse:true,trace_updates:true,transform_row_batch:if route=="row16" {16} else {4},
        rebuild_cache_mode:VerificationMode::Product,sparse:policy,..Default::default()}
}

pub fn edits(w:&Witness,n:usize,roots:usize,addition:bool,seed:u64)->Update {
    let mut rng=StdRng::seed_from_u64(seed);let mut u=Update {indices:vec![],values:vec![]};
    for i in 1..=roots {
        let value=Fr::rand(&mut rng);let old=if addition {w.wires[n+i]} else {w.wires[4*n+i]};
        assert!((value-old).into_bigint().num_bits()>128,"primary eta must have full width");
        if addition {
            let sum=w.wires[i]+value;u.indices.extend([n+i,2*n+i,3*n+i,5*n+i]);
            u.values.extend([value,sum,sum,sum*w.wires[4*n+i]]);
        } else {u.indices.extend([4*n+i,5*n+i]);u.values.extend([value,w.wires[3*n+i]*value]);}
    }
    u
}

pub fn profile<B:RebuildBackend>(tag:&str,diagnostic:bool,mut make:impl FnMut()->B) {
    let routes=if diagnostic {vec!["r1","g1","g2"]} else if tag=="cpu" {vec!["g2","row16"]} else {vec!["g1","g2","row16"]};
    let policy=if tag=="cpu" {SparseDispatchPolicy::CpuReference} else {SparseDispatchPolicy::ForceGpu};
    for n in [64,256,1024] {for repeat in 0..3 {
        let c=ToyCircuit::new(n).unwrap();let (instance,witness)=c.valid_instance();let start=Instant::now();
        let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(21000+n as u64+repeat as u64+u64::from(diagnostic)*1_000_000)).unwrap();
        eprintln!("GGS_SETUP backend={tag} diagnostic={} n={n} repeat={repeat} ns={}",usize::from(diagnostic),start.elapsed().as_nanos());
        let mut sessions=Vec::new();let mut allocations=Vec::new();
        for &route in &routes {
            let started=Instant::now();let b=make();
            eprintln!("GGS_BACKEND backend={tag} diagnostic={} n={n} repeat={repeat} route={route} ns={}",usize::from(diagnostic),started.elapsed().as_nanos());
            let started=Instant::now();let s=p.start_incremental_session(&instance,&witness,config(route,policy),b).unwrap();
            let r=s.resource_plan();let budget=s.budget();allocations.push(s.backend().stats().bytes_allocated);
            eprintln!("GGS_INITIAL backend={tag} diagnostic={} n={n} repeat={repeat} route={route} ns={} window={} horizon={} resident={} upper={} host={} workspace={} q={} w={} cmax={}",
                usize::from(diagnostic),started.elapsed().as_nanos(),s.window(),s.horizon(),s.backend().stats().bytes_allocated,r.upper_bytes,r.global_g2_host_bytes,r.global_g2_workspace_bytes,budget.q,budget.total,budget.max_chunk);
            sessions.push(s);
        }
        let steps=if diagnostic {2} else {4};
        for step in 0..steps {
            let addition=!diagnostic && step%2==1;
            let roots=if diagnostic {2} else if addition {4} else {8};
            let k=if diagnostic {4} else {16};let warm=if diagnostic {step==0} else {step<2};
            let u=edits(sessions[0].current_witness(),n,roots,addition,21100+n as u64*100+repeat as u64*10+step as u64+u64::from(diagnostic)*1_000_000);
            let mut element_work=Vec::new();let mut equations=Vec::new();
            for offset in 0..routes.len() {
                let index=(repeat+step+offset)%routes.len();let route=routes[index];let s=&mut sessions[index];
                let before=s.backend().stats();let sparse_before=s.sparse_dispatch_stats();
                let started=Instant::now();let proof=s.update(&u).unwrap();let wall=started.elapsed().as_nanos();
                let r=s.last_receipt();let t=s.update_timings();let after=s.backend().stats();let sparse=s.sparse_dispatch_stats();
                assert_eq!(r.changed,k);assert!(r.large_batch && r.swapped);assert!(s.rebuild_progress().is_none());
                assert!(r.rebuild_units<=r.allowance);assert_eq!(after.bytes_allocated,allocations[index]);
                assert_eq!(r.global_g2_reused,route=="g2" || route=="row16");
                if r.global_g2_reused {assert_eq!(r.global_g2_inputs,if addition {0} else {roots});}
                if route=="r1" {assert!(r.linear_reused);}
                assert!(t.linear_detail.recorded_ns()<=t.rebuild_ns[1]);assert!(t.cache_detail.recorded_ns()<=t.rebuild_ns[6]);
                assert!(t.delta_detail.recorded_ns()<=t.delta_proof_ns && t.rebase_detail.recorded_ns()<=t.rebase_ns);
                let id=format!("backend={tag} diagnostic={} n={n} repeat={repeat} route={route} step={step}",usize::from(diagnostic));
                let family=if addition {"addition"} else {"multiplier"};
                eprintln!("GGS_SAMPLE {id} family={family} warm={} gates={} k={k} roots={roots} dirty={} pending=0 captured={} global_reused={} global_inputs={} global_skipped={} wall_ns={wall} total_ns={} normalization_ns={} constraints_ns={} preparation_ns={} publication_ns={} live_ns={} delta_ns={} rebase_ns={} work={} allowance={} foreground_work={} resident={} resident_calls={} fr_elements={} g1_elements={} dense_g2={} sparse_cpu_g1={} sparse_gpu_g1={} sparse_cpu_g2={} sparse_gpu_g2={} cache_eq={} cache_fe={} cache_miller={} g2_builds={} g2_hits={} prepared_peak={}",
                    usize::from(warm),2*n,if addition {4} else {2},usize::from(r.frozen_linear_captured),usize::from(r.global_g2_reused),r.global_g2_inputs,r.global_g2_skipped_points,
                    t.total_ns,t.normalization_ns,t.constraints_ns,t.preparation_ns,t.publication_ns,t.live_linear_ns,t.delta_proof_ns,t.rebase_ns,r.rebuild_units,r.allowance,r.foreground_linear_work.units().unwrap(),after.bytes_allocated,after.calls-before.calls,
                    after.fr_transform_elements-before.fr_transform_elements,after.g1_transform_elements-before.g1_transform_elements,after.msm_g2_points-before.msm_g2_points,
                    sparse.cpu.g1_points-sparse_before.cpu.g1_points,sparse.gpu.g1_points-sparse_before.gpu.g1_points,sparse.cpu.g2_points-sparse_before.cpu.g2_points,sparse.gpu.g2_points-sparse_before.gpu.g2_points,
                    t.cache_detail.equations,t.cache_detail.final_exponentiations,t.cache_detail.miller_calls,t.cache_detail.g2_prepare_builds,t.cache_detail.g2_prepare_hits,t.cache_detail.prepared_peak_entries);
                for (scope,names,ns,calls) in [
                    ("rebuild",UpdateTimings::REBUILD_NAMES.as_slice(),t.rebuild_ns.as_slice(),Some(t.rebuild_calls.as_slice())),
                    ("linear",LinearTimings::NAMES.as_slice(),t.linear_detail.ns.as_slice(),Some(t.linear_detail.calls.as_slice())),
                    ("cache",CacheCheckTimings::NAMES.as_slice(),t.cache_detail.ns.as_slice(),None),
                    ("delta",SparseProofTimings::NAMES.as_slice(),t.delta_detail.ns.as_slice(),Some(t.delta_detail.calls.as_slice())),
                    ("rebase",SparseProofTimings::NAMES.as_slice(),t.rebase_detail.ns.as_slice(),Some(t.rebase_detail.calls.as_slice()))] {
                    for (j,name) in names.iter().enumerate() {
                        eprintln!("GGS_DETAIL {id} scope={scope} phase={name} ns={} calls={}",ns[j],calls.map_or(0,|c|c[j]));
                    }
                }
                element_work.push((after.fr_transform_elements-before.fr_transform_elements,after.g1_transform_elements-before.g1_transform_elements));
                equations.push(t.cache_detail.equations);
                let started=Instant::now();assert!(p.verifying_key().verify_reference(&instance,&proof).unwrap());
                eprintln!("GGS_VERIFY {id} ns={}",started.elapsed().as_nanos());
            }
            for index in 1..routes.len() {assert_eq!(sessions[index].proof(),sessions[0].proof());}
            assert!(element_work.iter().all(|v|*v==element_work[0]));
            assert!(equations.iter().all(|v|*v==equations[0]));
        }
        let started=Instant::now();let last=sessions.last().unwrap();
        assert_eq!(last.proof().anchor,p.prove_anchor(&instance,last.current_witness()).unwrap().anchor);
        eprintln!("GGS_DENSE backend={tag} diagnostic={} n={n} repeat={repeat} ns={}",usize::from(diagnostic),started.elapsed().as_nanos());
    }}
}
