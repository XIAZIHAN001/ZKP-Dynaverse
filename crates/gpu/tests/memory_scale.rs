#![cfg(feature="icicle")]
use ark_bn254::{Fr,G1Affine,G1Projective};
use ark_ec::{CurveGroup,Group,VariableBaseMSM};
use dynaverse_gpu::{G2ValidationMode,IcicleBackend,SetupMode};
use dynaverse_protocol::incremental_rebuild::operators::{RebuildBackend,BufferKind};

fn scale_sizes(value:Option<&str>)->anyhow::Result<Vec<usize>> {
    let Some(value)=value else {return Ok(vec![1024,4096,16384]);};
    let mut sizes=Vec::new();
    for part in value.split(',') {
        let n:usize=part.parse()?;
        anyhow::ensure!((16..=1<<20).contains(&n) && n.is_power_of_two(),
            "scale must be a power of two between 16 and 2^20");
        anyhow::ensure!(!sizes.contains(&n),"duplicate scale");
        sizes.push(n);
    }
    Ok(sizes)
}

fn scale_limits(n:usize)->(u64,u64) {
    let limit=(16384*n as u64+(256<<20)).max(2<<30);
    let arena=(1472*n as u64).next_power_of_two().max(64<<20);
    (limit,arena)
}

#[test]
fn scale_options_validate_sizes_before_gpu_allocation() {
    assert_eq!(scale_sizes(None).unwrap(),vec![1024,4096,16384]);
    assert_eq!(scale_sizes(Some("65536,131072,262144,524288,1048576")).unwrap(),
        vec![65536,131072,262144,524288,1048576]);
    for invalid in ["", "0", "3", "2097152", "65536,", "65536,65536", "-1"] {
        assert!(scale_sizes(Some(invalid)).is_err(),"accepted {invalid}");
    }
}

#[test]
fn scale_quotas_cover_unique_undo_and_program_envelope() {
    for (n,arena_mib) in [(65536,128),(131072,256),(262144,512),(524288,1024),(1048576,2048)] {
        let (limit,arena)=scale_limits(n);
        assert_eq!(arena,arena_mib<<20);
        assert!(arena>=1472*n as u64);
        assert!(limit>=16384*n as u64+(256<<20));
        assert!(arena<limit);
    }
    assert_eq!(scale_limits(16384),(2<<30,64<<20));
}

#[test]
fn read_only_msm_bases_use_affine_storage_and_match_reference() {
    let mut backend=IcicleBackend::with_resident_limits(1<<30,1<<20).unwrap();
    let points:Vec<G1Affine>=(0..33).map(|i|(G1Projective::generator()*Fr::from(i as u64)).into_affine()).collect();
    let scalars:Vec<Fr>=(0..33).map(|i|Fr::from((i+11) as u64)).collect();
    let before=backend.stats().bytes_allocated;
    let bases=backend.register_msm_bases(&points).unwrap();
    assert_eq!(backend.stats().bytes_allocated-before,33*64,"bases expanded to projective storage");
    let values=backend.alloc(BufferKind::Fr,33,true).unwrap();
    backend.write_fields(values,0,&scalars).unwrap();
    backend.seal();
    assert_eq!(backend.msm_g1(bases,0,values,0,33).unwrap(),
        G1Projective::msm_unchecked(&points,&scalars).into_affine());
    let mut read=vec![G1Affine::identity();33];backend.read_points(bases,0,&mut read).unwrap();
    assert_eq!(read,points);
    assert!(backend.write_points(bases,0,&points[..1]).is_err());
}

#[test]
fn first_write_pages_preserve_originals_across_overlaps_and_transactions() {
    let mut backend=IcicleBackend::with_resident_limits(1<<30,512).unwrap();
    let id=backend.alloc(BufferKind::Fr,17,true).unwrap();
    let original:Vec<Fr>=(0..17).map(|i|Fr::from(i as u64)).collect();
    backend.write_fields(id,0,&original).unwrap();backend.seal();
    backend.begin_transaction().unwrap();
    for _ in 0..20 {backend.write_fields(id,7,&[Fr::from(99u64);2]).unwrap();}
    assert_eq!(backend.resident_diagnostics().unwrap().journal_peak_bytes,64);
    backend.rollback_transaction().unwrap();
    let mut read=vec![Fr::from(0u64);17];backend.read_fields(id,0,&mut read).unwrap();
    assert_eq!(read,original);
    backend.begin_transaction().unwrap();backend.write_fields(id,16,&[Fr::from(88u64)]).unwrap();backend.commit_transaction();
    backend.begin_transaction().unwrap();backend.write_fields(id,16,&[Fr::from(77u64)]).unwrap();backend.rollback_transaction().unwrap();
    backend.read_fields(id,0,&mut read).unwrap();assert_eq!(read[16],Fr::from(88u64));
}

#[test]
#[ignore="explicit complete-circuit scale probe, includes CPU proof oracle and full paid rebuild"]
fn complete_circuit_memory_scale_probe() {
    use dynaverse_protocol::{dynaverse::DynaverseParams,toy_circuit::ToyCircuit,Update,
        incremental_rebuild::IncrementalConfig};
    use rand::{rngs::StdRng,SeedableRng};
    use std::time::Instant;
    use std::io::Write;
    use dynaverse_protocol::trace::{CpuPhase,CpuTrace};
    let requested=std::env::var("DVGPU_SCALE_N").ok();
    for n in scale_sizes(requested.as_deref()).unwrap() {
        let total_started=Instant::now();
        let started=Instant::now();
        let route=std::env::var("DVGPU_SCALE_G2_MODE").unwrap_or_else(|_|"cpu_reference".into());
        let mode=match route.as_str() {
            "cpu_reference"=>G2ValidationMode::CpuReference,
            "gpu63"=>G2ValidationMode::GpuFrobenius63,
            _=>panic!("DVGPU_SCALE_G2_MODE must be cpu_reference or gpu63"),
        };
        let mut output=std::env::var("DVGPU_G2_OUTPUT_DIR").ok().map(|value| {
            let path=std::path::PathBuf::from(value);
            assert!(path.is_absolute(),"scale output must be absolute");
            std::fs::create_dir_all(&path).unwrap();
            let mut file=std::fs::OpenOptions::new().write(true).create_new(true)
                .open(path.join(format!("scale-{n}-{route}-timings.csv"))).unwrap();
            writeln!(file,"n,total_gates,route,level,parent,phase,ns").unwrap();file
        });
        // Flush after every completed phase so a timeout preserves partial data.
        let mut record=|level:&str,parent:&str,phase:&str,ns:u128| {
            eprintln!("SCALE_TIME n={n} total_gates={} route={route} level={level} parent={parent} phase={phase} ns={ns}",2*n);
            if let Some(file)=&mut output {
                writeln!(file,"{n},{},{route},{level},{parent},{phase},{ns}",2*n).unwrap();
                file.flush().unwrap();
            }
        };
        let mut accounted=0u128;
        let (limit,arena)=scale_limits(n);
        let circuit=ToyCircuit::new(n).unwrap();let (instance,witness)=circuit.valid_instance();
        let backend=IcicleBackend::with_resident_limits(limit,arena).unwrap();
        backend.set_setup_mode(SetupMode::TableBatchNormalize).unwrap();
        backend.set_g2_validation_mode(mode).unwrap();
        let ns=started.elapsed().as_nanos();record("exclusive","command","fixture_backend",ns);accounted+=ns;
        eprintln!("SCALE_BEGIN n={n} phase=setup memory_limit={limit} undo_arena={arena}");
        let mut setup_trace=CpuTrace::new();
        let started=Instant::now();
        let params=DynaverseParams::setup_with_backend_traced(circuit,&mut StdRng::seed_from_u64(7500+n as u64),&backend,&mut setup_trace).unwrap();
        let setup=started.elapsed();
        record("exclusive","command","setup",setup.as_nanos());accounted+=setup.as_nanos();
        assert!(setup_trace.recorded_total()<=setup);
        for phase in CpuPhase::ALL {
            record("detail","setup",phase.name(),setup_trace.elapsed(phase).as_nanos());
        }
        record("detail","setup","unattributed",(setup-setup_trace.recorded_total()).as_nanos());
        eprintln!("SCALE_SETUP_STATS n={n} route={route} device={:?} host={:?} g2={:?} traffic={:?}",
            backend.setup_stats().unwrap(),backend.setup_host_stats(),backend.g2_validation_stats().unwrap(),backend.g2_validation_traffic().unwrap());
        eprintln!("SCALE_PHASE n={n} phase=initial setup_ns={}",setup.as_nanos());
        let started=Instant::now();
        let mut session=params.start_incremental_session(&instance,&witness,
            IncrementalConfig {memory_limit_bytes:limit,..Default::default()},backend).unwrap();
        let initial=started.elapsed();
        record("exclusive","command","initial",initial.as_nanos());accounted+=initial.as_nanos();
        let started=Instant::now();
        let resident=session.backend().stats().bytes_allocated;
        let certificate=session.resource_plan().transaction_bytes;
        let ns=started.elapsed().as_nanos();record("exclusive","command","initial_diagnostics",ns);accounted+=ns;
        eprintln!("SCALE_PHASE n={n} phase=oracle initial_ns={} resident={resident} certificate={certificate}",initial.as_nanos());
        let started=Instant::now();let expected=params.prove_with_aux(&instance,&witness).unwrap();
        let ns=started.elapsed().as_nanos();record("exclusive","command","cpu_initial_oracle",ns);accounted+=ns;
        let started=Instant::now();assert_eq!(*session.proof(),expected.0);drop(expected);
        let ns=started.elapsed().as_nanos();record("exclusive","command","initial_compare_cleanup",ns);accounted+=ns;
        let started=Instant::now();
        assert!(params.verifying_key().verify_reference(&instance,session.proof()).unwrap());
        let ns=started.elapsed().as_nanos();record("exclusive","command","initial_public_verify",ns);accounted+=ns;
        let started=Instant::now();
        let mut indices=Vec::with_capacity(2*n);let mut values=Vec::with_capacity(2*n);
        for i in 0..n {
            let next=witness.wires[4*n+i]+Fr::from(1u64);
            indices.push(4*n+i);values.push(next);
            indices.push(5*n+i);values.push(witness.wires[3*n+i]*next);
        }
        let ns=started.elapsed().as_nanos();record("exclusive","command","prepare_update",ns);accounted+=ns;
        eprintln!("SCALE_PHASE n={n} phase=paid_rebuild");
        let started=Instant::now();let proof=session.update(&Update {indices,values}).unwrap();
        let rebuilt=started.elapsed();
        record("exclusive","command","paid_rebuild",rebuilt.as_nanos());accounted+=rebuilt.as_nanos();
        eprintln!("SCALE_REBUILD_BLOCKS n={n} count={} total_ns={} max_ns={}",
            session.block_timings().count,session.block_timings().total_ns,session.block_timings().max_ns);
        let started=Instant::now();
        assert!(session.last_receipt().large_batch && session.last_receipt().swapped);
        assert!(session.last_receipt().rebuild_units<=session.last_receipt().allowance);
        assert_eq!(session.backend().stats().bytes_allocated,resident);
        let ns=started.elapsed().as_nanos();record("exclusive","command","rebuild_receipt_checks",ns);accounted+=ns;
        let started=Instant::now();let expected=params.prove_anchor(&instance,session.current_witness()).unwrap();
        let ns=started.elapsed().as_nanos();record("exclusive","command","cpu_rebuild_anchor_oracle",ns);accounted+=ns;
        let started=Instant::now();assert_eq!(proof.anchor,expected.anchor);drop(expected);
        let ns=started.elapsed().as_nanos();record("exclusive","command","rebuild_compare_cleanup",ns);accounted+=ns;
        let started=Instant::now();
        assert!(params.verifying_key().verify_reference(&instance,&proof).unwrap());
        let ns=started.elapsed().as_nanos();record("exclusive","command","rebuild_public_verify",ns);accounted+=ns;
        let started=Instant::now();
        let journal=session.backend().resident_diagnostics().unwrap().journal_peak_bytes;
        assert!(journal<=certificate);
        let ns=started.elapsed().as_nanos();record("exclusive","command","final_diagnostics",ns);accounted+=ns;
        let started=Instant::now();drop(session);drop(params);drop(proof);drop(instance);drop(witness);
        let ns=started.elapsed().as_nanos();record("exclusive","command","cleanup",ns);accounted+=ns;
        let total=total_started.elapsed().as_nanos();
        assert!(accounted<=total);
        record("exclusive","command","probe_overhead",total-accounted);
        record("total","command","probe_total",total);
        eprintln!("COMPLETE_SCALE_OK n={n} setup_ns={} initial_ns={} rebuild_ns={} resident={resident} certificate={certificate} journal_peak={journal}",
            setup.as_nanos(),initial.as_nanos(),rebuilt.as_nanos());
    }
}
