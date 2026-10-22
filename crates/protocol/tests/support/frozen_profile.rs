use ark_bn254::Fr;
use ark_ff::{UniformRand,PrimeField,BigInteger};
use dynaverse_protocol::{dynaverse::DynaverseParams,toy_circuit::ToyCircuit,Update,Witness,VerificationMode,
    incremental_rebuild::{IncrementalConfig,SparseDispatchPolicy,operators::RebuildBackend}};
use rand::{rngs::StdRng,SeedableRng};
use std::time::Instant;

pub fn edits(w:&Witness,n:usize,roots:usize,addition:bool,seed:u64)->Update {
    let mut rng=StdRng::seed_from_u64(seed);let mut u=Update {indices:vec![],values:vec![]};
    for i in 1..=roots {
        let next=Fr::rand(&mut rng);
        let old=if addition {w.wires[n+i]} else {w.wires[4*n+i]};
        assert!((next-old).into_bigint().num_bits()>128,"profile eta must be full width");
        if addition {
            let sum=w.wires[i]+next;
            u.indices.extend([n+i,2*n+i,3*n+i,5*n+i]);
            u.values.extend([next,sum,sum,sum*w.wires[4*n+i]]);
        } else {
            u.indices.extend([4*n+i,5*n+i]);u.values.extend([next,w.wires[3*n+i]*next]);
        }
    }
    u
}
pub fn config(capture:bool,sparse:SparseDispatchPolicy)->IncrementalConfig {
    IncrementalConfig {linear_rebuild_reuse:true,unchanged_cache_reuse:true,
        foreground_linear_reuse:true,freeze_linear_capture_limit:if capture {Some(128)} else {None},
        support_geometry_reuse:true,transform_row_batch:4,trace_updates:true,
        rebuild_cache_mode:VerificationMode::Product,sparse,..Default::default()}
}

/// Two cold warm-up calls (one per wire family) then one measured call per
/// family; counterbalance route order. Every call finishes a paid rebuild.
pub fn profile<B:RebuildBackend>(tag:&str,sizes:&[usize],repeats:usize,
    policy:SparseDispatchPolicy,mut make:impl FnMut()->B) {
    for &n in sizes {for repeat in 0..repeats {
        let c=ToyCircuit::new(n).unwrap();let (instance,witness)=c.valid_instance();
        let started=Instant::now();
        let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(19500+n as u64+repeat as u64)).unwrap();
        eprintln!("FP_SETUP backend={tag} n={n} repeat={repeat} ns={}",started.elapsed().as_nanos());
        let mut sessions=Vec::new();let mut allocations=Vec::new();
        for route in 0..2 {
            let started=Instant::now();let b=make();
            eprintln!("FP_BACKEND backend={tag} n={n} repeat={repeat} route={route} ns={}",started.elapsed().as_nanos());
            let started=Instant::now();let s=p.start_incremental_session(&instance,&witness,config(route==1,policy),b).unwrap();
            let resource=s.resource_plan();allocations.push(s.backend().stats().bytes_allocated);
            eprintln!("FP_INITIAL backend={tag} n={n} repeat={repeat} route={route} ns={} resident={} upper={} shadow={} q={} w={} cmax={}",
                started.elapsed().as_nanos(),allocations[route],resource.upper_bytes,resource.frozen_linear_bytes,s.budget().q,s.budget().total,s.budget().max_chunk);
            sessions.push(s);
        }
        for step in 0..4 {
            let addition=step%2==1;let roots=sessions[0].horizon()/2+1;
            let u=edits(sessions[0].current_witness(),n,roots,addition,19600+n as u64*100+repeat as u64*10+step as u64);
            for offset in 0..2 {
                let route=(step+repeat+offset)%2;let s=&mut sessions[route];
                let started=Instant::now();let proof=s.update(&u).unwrap();let wall=started.elapsed().as_nanos();
                let r=s.last_receipt();let t=s.update_timings();
                let pending=usize::from(s.rebuild_progress().is_some());
                assert_eq!(pending,0);assert!(r.swapped && r.large_batch);
                assert_eq!(r.frozen_linear_captured,route==1);assert!(r.rebuild_units<=r.allowance);
                assert!(t.linear_detail.recorded_ns()<=t.rebuild_ns[1]);
                assert_eq!(s.backend().stats().bytes_allocated,allocations[route]);
                let family=if addition {"addition"} else {"multiplier"};
                eprintln!("FP_SAMPLE backend={tag} n={n} repeat={repeat} route={route} step={step} family={family} warm={} gates={} k={} eta={} captured={} skipped={} pending={pending} wall_ns={wall} total_ns={} live_ns={} delta_ns={} rebase_ns={} fixed_g1_ns={} g2_ns={} copy_ns={} quotient_g1_ns={} linear_ns={} fr_ns={} transform_g1_ns={} cache_ns={} finish_ns={} validation_ns={} work={} foreground_work={} allowance={} resident={}",
                    usize::from(step<2),2*n,r.changed,r.foreground_linear_inputs,usize::from(r.frozen_linear_captured),r.frozen_linear_skipped_g1_points,
                    t.total_ns,t.live_linear_ns,t.delta_proof_ns,t.rebase_ns,t.linear_detail.ns[0],t.linear_detail.ns[1],t.linear_detail.ns[2],t.linear_detail.ns[3],t.rebuild_ns[1],t.rebuild_ns[2],t.rebuild_ns[3],t.rebuild_ns[6],t.rebuild_ns[7],
                    t.delta_detail.ns[6]+t.rebase_detail.ns[6],r.rebuild_units,r.foreground_linear_work.units().unwrap(),r.allowance,allocations[route]);
                let started=Instant::now();assert!(p.verifying_key().verify_reference(&instance,&proof).unwrap());
                eprintln!("FP_VERIFY backend={tag} n={n} repeat={repeat} route={route} step={step} ns={}",started.elapsed().as_nanos());
            }
            assert_eq!(sessions[0].proof(),sessions[1].proof());
            assert_eq!(sessions[0].current_witness(),sessions[1].current_witness());
        }
        let started=Instant::now();
        assert_eq!(sessions[1].proof().anchor,p.prove_anchor(&instance,sessions[1].current_witness()).unwrap().anchor);
        eprintln!("FP_DENSE backend={tag} n={n} repeat={repeat} ns={}",started.elapsed().as_nanos());
    }}
}
