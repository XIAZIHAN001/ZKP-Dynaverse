use ark_bn254::{Fr,G1Affine};
use ark_ec::{AffineRepr,CurveGroup};
use ark_ff::{BigInteger,PrimeField,UniformRand};
use dynaverse_gpu::IcicleBackend;
use dynaverse_protocol::{dynaverse::DynaverseParams,toy_circuit::ToyCircuit,Update,
    incremental_rebuild::{IncrementalConfig,SparseDispatchPolicy,operators::{BufferId,BufferKind,RebuildBackend}}};
use rand::{rngs::StdRng,SeedableRng};
use std::{time::Instant,collections::BTreeMap};
#[allow(dead_code)]
#[path="../../../protocol/tests/support/cache_preparation_profile.rs"]
mod parent;

const ROUTES:[(&str,bool,bool);4]=[
    ("host_full",false,false),("host_bounded",false,true),
    ("device_full",true,false),("device_bounded",true,true)];
fn make(device:bool,bounded:bool)->IcicleBackend {
    let mut b=IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap();
    b.set_device_affine_readback(device);b.set_bounded_g1_decode(bounded);b
}
fn config(window:usize)->IncrementalConfig {
    IncrementalConfig {rebuild_cache_readback_roots:Some(64),window:Some(window),
        ..parent::config("prepared",SparseDispatchPolicy::ForceGpu)}
}

fn points(b:&mut IcicleBackend,n:usize)->(BufferId,Vec<G1Affine>) {
    let mut rng=StdRng::seed_from_u64(29400);let mut input=Vec::new();let mut scalars=Vec::new();let mut expected=Vec::new();
    for i in 0..n {
        let point=if i%11==0 {G1Affine::identity()} else {(G1Affine::generator()*Fr::from(i as u64+1)).into_affine()};
        let scalar=if i%13==0 {Fr::from(0u64)} else {Fr::rand(&mut rng)};
        input.push(point);scalars.push(scalar);expected.push((point*scalar).into_affine());
    }
    let a=b.alloc(BufferKind::G1,n,false).unwrap();let s=b.alloc(BufferKind::Fr,n,false).unwrap();let out=b.alloc(BufferKind::G1,n,true).unwrap();
    b.write_points(a,0,&input).unwrap();b.write_fields(s,0,&scalars).unwrap();
    b.scale_points(a,0,s,0,out,0,n).unwrap();(out,expected)
}

pub fn micro() {
    let mut b=make(false,false);let(id,expected)=points(&mut b,1032);b.seal();
    let allocated=b.stats().bytes_allocated;
    for n in [0,1,16,64,128,256,257,1024] {for repeat in 0..8 {for offset in 0..4 {
        let(route,device,bounded)=ROUTES[(repeat+offset)%4];
        b.set_device_affine_readback(device);b.set_bounded_g1_decode(bounded);
        let mut output=vec![G1Affine::identity();n];let before=b.resident_diagnostics().unwrap();
        let p=b.profile_read_points(id,3,&mut output).unwrap();let after=b.resident_diagnostics().unwrap();
        assert_eq!(output,expected[3..3+n]);assert_eq!(b.stats().bytes_allocated,allocated);
        assert_eq!(p.g1_cofactor_decode_points,if bounded && n<=256 {n as u64} else {0});
        assert_eq!(p.g1_full_decode_points,if bounded && n<=256 {0} else {n as u64});
        let width=if device && n<=256 {64} else {96};
        assert_eq!(after.physical_d2h_bytes-before.physical_d2h_bytes,n as u64*width);
        eprintln!("GD_MICRO n={n} repeat={repeat} route={route} warm={} total_ns={} ffi_ns={} decode_ns={} native_ns={} conversion_ns={} event_operation_ns={} full_points={} cofactor_points={} physical_d2h={}",
            usize::from(repeat<2),p.total_ns,p.ffi_ns,p.decoding_ns,p.native.native_wall_ns,p.native.conversion_host_ns,
            p.native.event_operation_ns,p.g1_full_decode_points,p.g1_cofactor_decode_points,after.physical_d2h_bytes-before.physical_d2h_bytes);
    }}}
}

pub fn scaling() {
    for n in [64,256,1024] {for repeat in 0..4 {
        let c=ToyCircuit::new(n).unwrap();let(i,w)=c.valid_instance();let started=Instant::now();
        let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(29500+n as u64+repeat as u64)).unwrap();
        eprintln!("GD_SETUP n={n} repeat={repeat} ns={}",started.elapsed().as_nanos());
        let mut sessions=BTreeMap::new();let mut allocations=BTreeMap::new();
        // Four repeats give each route every initialization position once.
        for offset in 0..4 {
            let(route,device,bounded)=ROUTES[(repeat+offset)%4];
            let started=Instant::now();let backend=make(device,bounded);let backend_ns=started.elapsed().as_nanos();
            let started=Instant::now();let s=p.start_incremental_session(&i,&w,config(4),backend).unwrap();
            let budget=s.budget();let resource=s.resource_plan();
            eprintln!("GD_INITIAL n={n} repeat={repeat} route={route} order={offset} ns={} backend_ns={backend_ns} resident={} upper={} w={} cmax={} q={}",
                started.elapsed().as_nanos(),s.backend().stats().bytes_allocated,resource.upper_bytes,budget.total,budget.max_chunk,budget.q);
            allocations.insert(route,s.backend().stats().bytes_allocated);sessions.insert(route,s);
        }
        for (route,_,_) in ROUTES {
            assert_eq!(sessions[route].resource_plan(),sessions["host_full"].resource_plan());
            assert_eq!(sessions[route].budget(),sessions["host_full"].budget());
        }
        for step in 0..4 {
            let addition=step%2==1;let roots=if addition {4} else {8};
            let u=parent::edits(sessions["host_full"].current_witness(),n,roots,addition,29600+n as u64*100+repeat as u64*10+step as u64);
            for offset in 0..4 {
                let(route,_,_)=ROUTES[(repeat+step+offset)%4];let s=sessions.get_mut(route).unwrap();
                let before=s.backend().stats();let diag=s.backend().resident_diagnostics().unwrap();let started=Instant::now();
                let proof=s.update(&u).unwrap();let wall=started.elapsed().as_nanos();let t=s.update_timings();let r=s.last_receipt();
                let after=s.backend().stats();let d=s.backend().resident_diagnostics().unwrap();
                assert!(r.large_batch && r.swapped && s.rebuild_progress().is_none());assert_eq!(r.changed,16);assert!(r.rebuild_units<=r.allowance);
                assert_eq!(after.bytes_allocated,allocations[route]);assert_eq!(d.resident_allocations,diag.resident_allocations);
                assert_eq!(t.cache_detail.equations,2*n as u64);assert_eq!(t.cache_detail.final_exponentiations,2*n as u64);
                assert_eq!(t.cache_detail.read_points,2*n as u64);assert_eq!(t.cache_detail.read_batches,2*n.div_ceil(64) as u64);
                assert_eq!(t.cache_detail.divisors_formed,0);assert_eq!(t.cache_detail.g2_prepare_builds,0);
                eprintln!("GD_SAMPLE n={n} repeat={repeat} route={route} step={step} order={offset} warm={} family={} wall_ns={wall} total_ns={} cache_ns={} read_ns={} miller_ns={} fe_ns={} work={} allowance={} pending=0 generation={} anchor_generation={} resident={} physical_d2h={} logical_d2h={} read_batches={} read_points={} cache_eq={} cache_fe={} fr_elements={} g1_elements={}",
                    usize::from(step<2),if addition {"addition"} else {"multiplier"},t.total_ns,t.rebuild_ns[6],t.cache_detail.ns[0],t.cache_detail.ns[4],t.cache_detail.ns[5],r.rebuild_units,r.allowance,
                    proof.generation,proof.anchor_generation,after.bytes_allocated,d.physical_d2h_bytes-diag.physical_d2h_bytes,after.d2h_bytes-before.d2h_bytes,
                    t.cache_detail.read_batches,t.cache_detail.read_points,t.cache_detail.equations,t.cache_detail.final_exponentiations,
                    after.fr_transform_elements-before.fr_transform_elements,after.g1_transform_elements-before.g1_transform_elements);
                assert!(p.verifying_key().verify_reference(&i,&proof).unwrap());
                eprintln!("GD_VERIFY n={n} repeat={repeat} route={route} step={step}");
            }
            for (route,_,_) in ROUTES {
                assert_eq!(sessions[route].proof(),sessions["host_full"].proof());
                assert_eq!(sessions[route].last_receipt(),sessions["host_full"].last_receipt());
            }
        }
        let s=&sessions["device_bounded"];
        assert_eq!(s.proof().anchor,p.prove_anchor(&i,s.current_witness()).unwrap().anchor);
        eprintln!("GD_DENSE n={n} repeat={repeat}");
    }}
}

pub fn pending() {
    for n in [256,1024] {
        let c=ToyCircuit::new(n).unwrap();let(i,w)=c.valid_instance();
        let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(29700+n as u64)).unwrap();
        let mut sessions=BTreeMap::new();let mut history=BTreeMap::from([(0,w.clone())]);
        let order=if n==256 {[false,true]} else {[true,false]};
        for bounded in order {
            let started=Instant::now();let s=p.start_incremental_session(&i,&w,config(16),make(false,bounded)).unwrap();
            eprintln!("GDP_INITIAL n={n} bounded={} ns={} resident={}",usize::from(bounded),started.elapsed().as_nanos(),s.backend().stats().bytes_allocated);
            sessions.insert(bounded,s);
        }
        for step in 0..12 {
            let root=step+1;let mut rng=StdRng::seed_from_u64(29800+n as u64+step as u64);
            let value=Fr::rand(&mut rng);let current=sessions[&false].current_witness();
            assert!((value-current.wires[4*n+root]).into_bigint().num_bits()>128);
            let u=Update {indices:vec![4*n+root,5*n+root],values:vec![value,current.wires[3*n+root]*value]};
            for offset in 0..2 {
                let bounded=(step+offset)%2==1;let s=sessions.get_mut(&bounded).unwrap();let allocated=s.backend().stats().bytes_allocated;
                let started=Instant::now();let proof=s.update(&u).unwrap();let wall=started.elapsed().as_nanos();let t=s.update_timings();let r=s.last_receipt();let progress=s.rebuild_progress();
                assert!(!r.large_batch);assert_eq!(r.changed,2);assert!(r.rebuild_units<=r.allowance);assert_eq!(s.backend().stats().bytes_allocated,allocated);
                eprintln!("GDP_SAMPLE n={n} bounded={} step={step} warm={} wall_ns={wall} read_ns={} work={} allowance={} pending={} command={} commands={} generation={} anchor_generation={} swapped={} resident={allocated}",
                    usize::from(bounded),usize::from(step<2),t.cache_detail.ns[0],r.rebuild_units,r.allowance,usize::from(progress.is_some()),
                    progress.map_or(0,|p|p.command),progress.map_or(0,|p|p.commands),proof.generation,proof.anchor_generation,usize::from(r.swapped));
                assert!(p.verifying_key().verify_reference(&i,&proof).unwrap());eprintln!("GDP_VERIFY n={n} bounded={} step={step}",usize::from(bounded));
            }
            assert_eq!(sessions[&false].proof(),sessions[&true].proof());assert_eq!(sessions[&false].last_receipt(),sessions[&true].last_receipt());
            assert_eq!(sessions[&false].rebuild_progress(),sessions[&true].rebuild_progress());
            history.insert(sessions[&false].proof().generation,sessions[&false].current_witness().clone());
        }
        for bounded in [false,true] {
            let s=&sessions[&bounded];let generation=s.proof().anchor_generation;
            assert_eq!(s.proof().anchor,p.prove_anchor(&i,&history[&generation]).unwrap().anchor);
            eprintln!("GDP_END n={n} bounded={} pending={} generation={} anchor_generation={generation}",usize::from(bounded),usize::from(s.rebuild_progress().is_some()),s.proof().generation);
        }
    }
}
