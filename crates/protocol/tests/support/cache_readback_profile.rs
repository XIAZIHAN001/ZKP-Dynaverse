use ark_bn254::Fr;
use ark_ff::{BigInteger,PrimeField,UniformRand};
use dynaverse_protocol::{dynaverse::DynaverseParams,toy_circuit::ToyCircuit,Update,
    incremental_rebuild::{IncrementalConfig,SparseDispatchPolicy,UpdateTimings,LinearTimings,
        CacheCheckTimings,SparseProofTimings,operators::RebuildBackend}};
use rand::{rngs::StdRng,SeedableRng};
use std::{time::Instant,collections::BTreeMap};
#[allow(dead_code)]
#[path="cache_preparation_profile.rs"]
mod parent;

pub fn config(cap:Option<usize>,policy:SparseDispatchPolicy)->IncrementalConfig {
    IncrementalConfig {rebuild_cache_readback_roots:cap,..parent::config("prepared",policy)}
}

fn details(prefix:&str,id:&str,t:&UpdateTimings) {
    for (scope,names,ns,calls) in [
        ("rebuild",UpdateTimings::REBUILD_NAMES.as_slice(),t.rebuild_ns.as_slice(),Some(t.rebuild_calls.as_slice())),
        ("linear",LinearTimings::NAMES.as_slice(),t.linear_detail.ns.as_slice(),Some(t.linear_detail.calls.as_slice())),
        ("cache",CacheCheckTimings::NAMES.as_slice(),t.cache_detail.ns.as_slice(),None),
        ("delta",SparseProofTimings::NAMES.as_slice(),t.delta_detail.ns.as_slice(),Some(t.delta_detail.calls.as_slice())),
        ("rebase",SparseProofTimings::NAMES.as_slice(),t.rebase_detail.ns.as_slice(),Some(t.rebase_detail.calls.as_slice()))] {
        for (j,phase) in names.iter().enumerate() {
            eprintln!("{prefix}_DETAIL {id} scope={scope} phase={phase} ns={} calls={}",ns[j],calls.map_or(0,|c|c[j]));
        }
    }
}

pub fn profile<B:RebuildBackend>(tag:&str,mut make:impl FnMut()->B) {
    let caps=[None,Some(16),Some(32),Some(64)];
    let policy=if tag=="cpu" {SparseDispatchPolicy::CpuReference} else {SparseDispatchPolicy::ForceGpu};
    for n in [64,256,1024] {for repeat in 0..3 {
        let c=ToyCircuit::new(n).unwrap();let(i,w)=c.valid_instance();let start=Instant::now();
        let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(27000+n as u64+repeat as u64)).unwrap();
        eprintln!("CR_SETUP backend={tag} n={n} repeat={repeat} ns={}",start.elapsed().as_nanos());
        let mut sessions=Vec::new();let mut allocations=Vec::new();
        for cap in caps {
            let started=Instant::now();let backend=make();let backend_ns=started.elapsed().as_nanos();
            let started=Instant::now();let s=p.start_incremental_session(&i,&w,config(cap,policy),backend).unwrap();
            let initial_ns=started.elapsed().as_nanos();let r=s.resource_plan();let b=s.budget();
            let preparation=s.cache_preparation_stats().unwrap();
            eprintln!("CR_INITIAL backend={tag} n={n} repeat={repeat} cap={} ns={initial_ns} backend_ns={backend_ns} window={} horizon={} upper={} resident={} readback_host={} prepared_host={} prepared_retained={} precompute_ns={} w={} cmax={} q={}",
                cap.unwrap_or(0),s.window(),s.horizon(),r.upper_bytes,s.backend().stats().bytes_allocated,
                r.cache_readback_host_bytes,r.cache_preparation_host_bytes,preparation.retained_bytes,preparation.precompute_ns,b.total,b.max_chunk,b.q);
            allocations.push(s.backend().stats().bytes_allocated);sessions.push(s);
        }
        for step in 0..4 {
            let addition=step%2==1;let warm=step<2;let roots=if addition {4} else {8};
            let u=parent::edits(sessions[0].current_witness(),n,roots,addition,
                27100+n as u64*100+repeat as u64*10+step as u64);
            let mut element_work=Vec::new();
            for offset in 0..caps.len() {
                let index=(repeat+step+offset)%caps.len();let cap=caps[index];let s=&mut sessions[index];
                let before=s.backend().stats();let started=Instant::now();let proof=s.update(&u).unwrap();
                let wall=started.elapsed().as_nanos();let r=s.last_receipt();let t=s.update_timings();let after=s.backend().stats();
                assert!(r.large_batch && r.swapped && s.rebuild_progress().is_none());assert_eq!(r.changed,16);
                assert!(r.rebuild_units<=r.allowance);assert_eq!(after.bytes_allocated,allocations[index]);
                assert_eq!(t.cache_detail.equations,2*n as u64);
                assert_eq!(t.cache_detail.read_points,2*n as u64);
                assert_eq!(t.cache_detail.read_batches,(2*n.div_ceil(cap.unwrap_or(4))) as u64);
                assert_eq!(t.cache_detail.final_exponentiations,t.cache_detail.equations);
                assert_eq!(t.cache_detail.divisors_formed,0);assert_eq!(t.cache_detail.g2_prepare_builds,0);
                assert!(t.cache_detail.max_read_roots<=cap.unwrap_or(4));
                assert!(t.cache_detail.recorded_ns()<=t.rebuild_ns[6]);
                let id=format!("backend={tag} n={n} repeat={repeat} cap={} step={step}",cap.unwrap_or(0));
                eprintln!("CR_SAMPLE {id} family={} warm={} gates={} k=16 roots={roots} dirty=2 wall_ns={wall} total_ns={} normalization_ns={} constraints_ns={} preparation_ns={} publication_ns={} live_ns={} delta_ns={} rebase_ns={} work={} allowance={} foreground_work={} pending=0 generation={} anchor_generation={} swapped=1 resident={} backend_calls={} h2d={} d2h={} fr_elements={} g1_elements={} cache_eq={} cache_fe={} read_batches={} read_points={} max_read={} prepared_retained_peak={}",
                    if addition {"addition"} else {"multiplier"},usize::from(warm),2*n,t.total_ns,t.normalization_ns,t.constraints_ns,
                    t.preparation_ns,t.publication_ns,t.live_linear_ns,t.delta_proof_ns,t.rebase_ns,r.rebuild_units,r.allowance,
                    r.foreground_linear_work.units().unwrap(),proof.generation,proof.anchor_generation,after.bytes_allocated,
                    after.calls-before.calls,after.h2d_bytes-before.h2d_bytes,after.d2h_bytes-before.d2h_bytes,
                    after.fr_transform_elements-before.fr_transform_elements,after.g1_transform_elements-before.g1_transform_elements,
                    t.cache_detail.equations,t.cache_detail.final_exponentiations,t.cache_detail.read_batches,
                    t.cache_detail.read_points,t.cache_detail.max_read_roots,t.cache_detail.prepared_peak_entries);
                details("CR",&id,t);
                element_work.push((after.fr_transform_elements-before.fr_transform_elements,
                    after.g1_transform_elements-before.g1_transform_elements));
                let started=Instant::now();assert!(p.verifying_key().verify_reference(&i,&proof).unwrap());
                eprintln!("CR_VERIFY {id} ns={}",started.elapsed().as_nanos());
            }
            for s in &sessions {assert_eq!(s.proof(),sessions[0].proof());}
            assert!(element_work.iter().all(|work|*work==element_work[0]));
        }
        let started=Instant::now();let last=sessions.last().unwrap();
        assert_eq!(last.proof().anchor,p.prove_anchor(&i,last.current_witness()).unwrap().anchor);
        eprintln!("CR_DENSE backend={tag} n={n} repeat={repeat} ns={}",started.elapsed().as_nanos());
    }}
}

pub fn pending_profile<B:RebuildBackend>(tag:&str,mut make:impl FnMut()->B) {
    let caps=[None,Some(64)];
    let policy=if tag=="cpu" {SparseDispatchPolicy::CpuReference} else {SparseDispatchPolicy::ForceGpu};
    for n in [256,1024] {
        let c=ToyCircuit::new(n).unwrap();let(i,w)=c.valid_instance();let started=Instant::now();
        let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(27200+n as u64)).unwrap();
        eprintln!("CRP_SETUP backend={tag} n={n} ns={}",started.elapsed().as_nanos());
        let mut sessions=Vec::new();let mut histories=Vec::new();let mut allocations=Vec::new();
        for cap in caps {
            let mut cfg=config(cap,policy);cfg.window=Some(16);
            let started=Instant::now();let s=p.start_incremental_session(&i,&w,cfg,make()).unwrap();
            let b=s.budget();let r=s.resource_plan();
            eprintln!("CRP_INITIAL backend={tag} n={n} cap={} ns={} w={} cmax={} q={} upper={} readback_host={}",
                cap.unwrap_or(0),started.elapsed().as_nanos(),b.total,b.max_chunk,b.q,r.upper_bytes,r.cache_readback_host_bytes);
            allocations.push(s.backend().stats().bytes_allocated);
            sessions.push(s);histories.push(BTreeMap::from([(0,w.clone())]));
        }
        for step in 0..12 {
            let root=step+1;let mut rng=StdRng::seed_from_u64(27300+n as u64+step as u64);
            let value=Fr::rand(&mut rng);let current=sessions[0].current_witness();
            assert!((value-current.wires[4*n+root]).into_bigint().num_bits()>128);
            let u=Update {indices:vec![4*n+root,5*n+root],values:vec![value,current.wires[3*n+root]*value]};
            for offset in 0..2 {
                let index=(step+offset)%2;let cap=caps[index];let s=&mut sessions[index];
                let before=s.backend().stats();let started=Instant::now();s.update(&u).unwrap();
                let wall=started.elapsed().as_nanos();let r=s.last_receipt();let t=s.update_timings();let b=s.budget();
                assert_eq!(r.changed,2);assert!(!r.large_batch);assert!(r.rebuild_units<=r.allowance);
                assert_eq!(s.backend().stats().bytes_allocated,allocations[index]);
                assert!(t.cache_detail.recorded_ns()<=t.rebuild_ns[6]);
                histories[index].insert(s.proof().generation,s.current_witness().clone());
                let progress=s.rebuild_progress();
                let id=format!("backend={tag} n={n} cap={} step={step}",cap.unwrap_or(0));
                eprintln!("CRP_SAMPLE {id} k=2 warm={} wall_ns={wall} total_ns={} work={} allowance={} w={} cmax={} q={} pending={} command={} commands={} paid_changes={} frozen_generation={} generation={} anchor_generation={} swapped={} restarted={} backend_calls={} read_batches={} read_points={} max_read={} cache_eq={} resident={}",
                    usize::from(step<2),t.total_ns,r.rebuild_units,r.allowance,b.total,b.max_chunk,b.q,
                    usize::from(progress.is_some()),progress.map_or(0,|p|p.command),progress.map_or(0,|p|p.commands),
                    progress.map_or(0,|p|p.paid_changes),progress.map_or(0,|p|p.source_generation),
                    s.proof().generation,s.proof().anchor_generation,usize::from(r.swapped),usize::from(r.restarted),
                    s.backend().stats().calls-before.calls,t.cache_detail.read_batches,t.cache_detail.read_points,
                    t.cache_detail.max_read_roots,t.cache_detail.equations,s.backend().stats().bytes_allocated);
                details("CRP",&id,t);
                let started=Instant::now();assert!(p.verifying_key().verify_reference(&i,s.proof()).unwrap());
                eprintln!("CRP_VERIFY {id} ns={}",started.elapsed().as_nanos());
            }
            assert_eq!(sessions[0].current_witness(),sessions[1].current_witness());
            if sessions[0].proof().anchor_generation==sessions[1].proof().anchor_generation {
                assert_eq!(sessions[0].proof(),sessions[1].proof());
            }
        }
        for (index,s) in sessions.iter().enumerate() {
            let generation=s.proof().anchor_generation;let started=Instant::now();
            assert_eq!(s.proof().anchor,p.prove_anchor(&i,&histories[index][&generation]).unwrap().anchor);
            eprintln!("CRP_END backend={tag} n={n} cap={} pending={} generation={} anchor_generation={generation} dense_ns={}",
                caps[index].unwrap_or(0),usize::from(s.rebuild_progress().is_some()),s.proof().generation,started.elapsed().as_nanos());
        }
    }
}
