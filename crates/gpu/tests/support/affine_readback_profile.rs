use ark_bn254::Fr;
use ark_ff::{BigInteger,PrimeField,UniformRand};
use dynaverse_gpu::IcicleBackend;
use dynaverse_protocol::{dynaverse::DynaverseParams,toy_circuit::ToyCircuit,Update,
    incremental_rebuild::{IncrementalConfig,SparseDispatchPolicy,operators::RebuildBackend}};
use rand::{rngs::StdRng,SeedableRng};
use std::{time::Instant,collections::BTreeMap};
#[allow(dead_code)]
#[path="../../../protocol/tests/support/cache_preparation_profile.rs"]
mod parent;

fn make(enabled:bool)->IcicleBackend {
    let mut b=IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap();
    b.set_device_affine_readback(enabled);b
}
fn config(cap:Option<usize>,window:usize)->IncrementalConfig {
    IncrementalConfig {rebuild_cache_readback_roots:cap,window:Some(window),
        ..parent::config("prepared",SparseDispatchPolicy::ForceGpu)}
}

pub fn scaling() {
    let routes=[("legacy4",None,false),("legacy64",Some(64),false),("affine64",Some(64),true)];
    for n in [64,256,1024] {for repeat in 0..3 {
        let c=ToyCircuit::new(n).unwrap();let(i,w)=c.valid_instance();let started=Instant::now();
        let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(28200+n as u64+repeat as u64)).unwrap();
        eprintln!("AR_SETUP n={n} repeat={repeat} ns={}",started.elapsed().as_nanos());
        let mut sessions=BTreeMap::new();let mut allocations=BTreeMap::new();
        // Counterbalance initialization as well as rotating update order.
        for offset in 0..3 {
            let index=(repeat+offset)%3;let(route,cap,enabled)=routes[index];
            let started=Instant::now();let backend=make(enabled);let backend_ns=started.elapsed().as_nanos();
            let started=Instant::now();let s=p.start_incremental_session(&i,&w,config(cap,4),backend).unwrap();
            let b=s.budget();let r=s.resource_plan();
            eprintln!("AR_INITIAL n={n} repeat={repeat} route={route} order={offset} ns={} backend_ns={backend_ns} resident={} upper={} readback_host={} w={} cmax={} q={}",
                started.elapsed().as_nanos(),s.backend().stats().bytes_allocated,r.upper_bytes,r.cache_readback_host_bytes,b.total,b.max_chunk,b.q);
            allocations.insert(route,s.backend().stats().bytes_allocated);sessions.insert(route,s);
        }
        for step in 0..4 {
            let addition=step%2==1;let roots=if addition {4} else {8};
            let u=parent::edits(sessions["legacy4"].current_witness(),n,roots,addition,28300+n as u64*100+repeat as u64*10+step as u64);
            for offset in 0..3 {
                let index=(repeat+step+offset)%3;let(route,cap,_)=routes[index];let s=sessions.get_mut(route).unwrap();
                let before=s.backend().stats();let d=s.backend().resident_diagnostics().unwrap();
                let started=Instant::now();let proof=s.update(&u).unwrap();let wall=started.elapsed().as_nanos();
                let r=s.last_receipt();let t=s.update_timings();let after=s.backend().stats();let diag=s.backend().resident_diagnostics().unwrap();
                assert!(r.large_batch && r.swapped && s.rebuild_progress().is_none());assert_eq!(r.changed,16);
                assert!(r.rebuild_units<=r.allowance);assert_eq!(after.bytes_allocated,allocations[route]);
                assert_eq!(diag.resident_allocations,d.resident_allocations);
                assert_eq!(t.cache_detail.equations,2*n as u64);
                assert_eq!(t.cache_detail.final_exponentiations,t.cache_detail.equations);
                assert_eq!(t.cache_detail.read_points,2*n as u64);
                assert_eq!(t.cache_detail.read_batches,(2*n.div_ceil(cap.unwrap_or(4))) as u64);
                assert_eq!(t.cache_detail.divisors_formed,0);assert_eq!(t.cache_detail.g2_prepare_builds,0);
                eprintln!("AR_SAMPLE n={n} repeat={repeat} route={route} step={step} order={offset} warm={} family={} wall_ns={wall} cache_ns={} read_ns={} work={} allowance={} pending=0 generation={} anchor_generation={} resident={} physical_d2h={} logical_d2h={} read_batches={} read_points={} cache_eq={} cache_fe={} fr_elements={} g1_elements={}",
                    usize::from(step<2),if addition {"addition"} else {"multiplier"},t.rebuild_ns[6],t.cache_detail.ns[0],r.rebuild_units,r.allowance,
                    proof.generation,proof.anchor_generation,after.bytes_allocated,diag.physical_d2h_bytes-d.physical_d2h_bytes,
                    after.d2h_bytes-before.d2h_bytes,t.cache_detail.read_batches,t.cache_detail.read_points,t.cache_detail.equations,t.cache_detail.final_exponentiations,
                    after.fr_transform_elements-before.fr_transform_elements,after.g1_transform_elements-before.g1_transform_elements);
                assert!(p.verifying_key().verify_reference(&i,&proof).unwrap());
                eprintln!("AR_VERIFY n={n} repeat={repeat} route={route} step={step}");
            }
            assert_eq!(sessions["legacy4"].proof(),sessions["legacy64"].proof());
            assert_eq!(sessions["legacy4"].proof(),sessions["affine64"].proof());
        }
        let last=&sessions["affine64"];
        assert_eq!(last.proof().anchor,p.prove_anchor(&i,last.current_witness()).unwrap().anchor);
        eprintln!("AR_DENSE n={n} repeat={repeat}");
    }}
}

pub fn pending() {
    for n in [256,1024] {
        let c=ToyCircuit::new(n).unwrap();let(i,w)=c.valid_instance();
        let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(28400+n as u64)).unwrap();
        let order=if n==256 {[false,true]} else {[true,false]};
        let mut sessions=BTreeMap::new();let mut history=BTreeMap::from([(0,w.clone())]);
        for enabled in order {
            let started=Instant::now();let s=p.start_incremental_session(&i,&w,config(Some(64),16),make(enabled)).unwrap();
            eprintln!("ARP_INITIAL n={n} enabled={} ns={} resident={}",usize::from(enabled),started.elapsed().as_nanos(),s.backend().stats().bytes_allocated);
            sessions.insert(enabled,s);
        }
        for step in 0..12 {
            let root=step+1;let mut rng=StdRng::seed_from_u64(28500+n as u64+step as u64);
            let value=Fr::rand(&mut rng);let current=sessions[&false].current_witness();
            assert!((value-current.wires[4*n+root]).into_bigint().num_bits()>128);
            let u=Update {indices:vec![4*n+root,5*n+root],values:vec![value,current.wires[3*n+root]*value]};
            for offset in 0..2 {
                let enabled=(step+offset)%2==1;let s=sessions.get_mut(&enabled).unwrap();
                let allocated=s.backend().stats().bytes_allocated;let started=Instant::now();let proof=s.update(&u).unwrap();let wall=started.elapsed().as_nanos();
                let r=s.last_receipt();let t=s.update_timings();let progress=s.rebuild_progress();
                assert!(!r.large_batch);assert_eq!(r.changed,2);assert!(r.rebuild_units<=r.allowance);
                assert_eq!(s.backend().stats().bytes_allocated,allocated);
                eprintln!("ARP_SAMPLE n={n} enabled={} step={step} warm={} wall_ns={wall} read_ns={} work={} allowance={} pending={} command={} commands={} generation={} anchor_generation={} swapped={} resident={allocated}",
                    usize::from(enabled),usize::from(step<2),t.cache_detail.ns[0],r.rebuild_units,r.allowance,usize::from(progress.is_some()),
                    progress.map_or(0,|p|p.command),progress.map_or(0,|p|p.commands),proof.generation,proof.anchor_generation,usize::from(r.swapped));
                assert!(p.verifying_key().verify_reference(&i,&proof).unwrap());
                eprintln!("ARP_VERIFY n={n} enabled={} step={step}",usize::from(enabled));
            }
            assert_eq!(sessions[&false].proof(),sessions[&true].proof());
            assert_eq!(sessions[&false].rebuild_progress(),sessions[&true].rebuild_progress());
            history.insert(sessions[&false].proof().generation,sessions[&false].current_witness().clone());
        }
        for enabled in [false,true] {
            let s=&sessions[&enabled];let generation=s.proof().anchor_generation;
            assert_eq!(s.proof().anchor,p.prove_anchor(&i,&history[&generation]).unwrap().anchor);
            eprintln!("ARP_END n={n} enabled={} pending={} generation={} anchor_generation={generation}",usize::from(enabled),usize::from(s.rebuild_progress().is_some()),s.proof().generation);
        }
    }
}
