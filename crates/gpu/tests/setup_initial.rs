#![cfg(feature = "icicle")]
use ark_bn254::{Fr, G1Projective, G2Projective};
use ark_ec::{CurveGroup, Group};
use ark_ff::{One, Zero};
use dynaverse_gpu::IcicleBackend;
use dynaverse_protocol::{dynaverse::DynaverseParams, toy_circuit::ToyCircuit, OperatorBackend};
use dynaverse_protocol::incremental_rebuild::{IncrementalConfig, operators::RebuildBackend};
use rand::{rngs::StdRng, SeedableRng};

#[test]
fn gpu_fixed_generator_arrays_match_both_cpu_groups() {
    let gpu = IcicleBackend::new().unwrap();
    for n in [0, 1, 2, 33, 257, 4097] {
        let mut scalars = Vec::new();
        for i in 0..n { scalars.push(Fr::from(i as u64) - Fr::one()); }
        if n > 1 { scalars[0] = Fr::zero(); scalars[1] = -Fr::one(); }
        let g1 = gpu.fixed_base_g1(&scalars).unwrap();
        let g2 = gpu.fixed_base_g2(&scalars).unwrap();
        assert_eq!(g1.len(), n); assert_eq!(g2.len(), n);
        for i in 0..n {
            assert_eq!(g1[i], (G1Projective::generator()*scalars[i]).into_affine());
            assert_eq!(g2[i], (G2Projective::generator()*scalars[i]).into_affine());
        }
    }
    assert_eq!(gpu.setup_point_counts(), (4390,4390));
}

#[test]
fn reservation_covers_public_batch_larger_than_sparse_window() {
    let mut backend=IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap();
    backend.reserve_sparse_msm(4,1024).unwrap();
    let before=backend.stats().bytes_allocated;
    let points=vec![ark_bn254::G1Affine::identity();1024];
    let scalars=vec![Fr::zero();1024];
    let sets:[&[ark_bn254::G1Affine];1]=[&points];
    backend.msm_batch(&dynaverse_protocol::BatchMsmRequest {point_sets:&sets,
        shared_scalars:&scalars,offsets:&[0],lengths:&[1024]}).unwrap();
    assert_eq!(backend.stats().bytes_allocated,before,"public batch grew retained buffers");
}

#[test]
fn gpu_setup_and_resident_initial_anchor_match_cpu_reference() {
    for n in [1, 2, 16] {
        let circuit = ToyCircuit::new(n).unwrap();
        let (instance, witness) = circuit.valid_instance();
        let cpu = DynaverseParams::setup(circuit.clone(), &mut StdRng::seed_from_u64(6400+n as u64)).unwrap();
        let gpu = IcicleBackend::with_resident_limits(1<<30, 64<<20).unwrap();
        let params = DynaverseParams::setup_with_backend(circuit,
            &mut StdRng::seed_from_u64(6400+n as u64), &gpu).unwrap();
        let counts = gpu.setup_point_counts();
        assert!(counts.0 > 0 && counts.1 > 0);
        let expected = cpu.prove_with_aux(&instance, &witness).unwrap().0;
        let session = params.start_incremental_session(&instance, &witness,
            IncrementalConfig::default(), gpu).unwrap();
        assert_eq!(*session.proof(), expected);
        assert!(cpu.verifying_key().verify(&instance, session.proof()).unwrap());
        assert!(cpu.verifying_key().verify_reference(&instance, session.proof()).unwrap());
        let stats = session.backend().stats();
        assert!(stats.msm_g1_points>0 && stats.msm_g2_points>0 && stats.fr_transform_elements>0);
    }
}

#[test]
#[ignore = "explicit GPU setup and larger full-circuit initial proof probe"]
fn gpu_setup_large_initial_probe() {
    for n in [256,1024] {
        let circuit=ToyCircuit::new(n).unwrap();
        let (instance,witness)=circuit.valid_instance();
        let started=std::time::Instant::now();
        let cpu=DynaverseParams::setup(circuit.clone(),&mut StdRng::seed_from_u64(6600+n as u64)).unwrap();
        let cpu_setup=started.elapsed();
        let backend=IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap();
        let started=std::time::Instant::now();
        let params=DynaverseParams::setup_with_backend(circuit,&mut StdRng::seed_from_u64(6600+n as u64),&backend).unwrap();
        let gpu_setup=started.elapsed();
        let setup_points=backend.setup_point_counts();
        let started=std::time::Instant::now();
        let session=params.start_incremental_session(&instance,&witness,IncrementalConfig::default(),backend).unwrap();
        let initial=started.elapsed();
        let expected=cpu.prove_with_aux(&instance,&witness).unwrap().0;
        assert_eq!(*session.proof(),expected);
        assert!(cpu.verifying_key().verify_reference(&instance,session.proof()).unwrap());
        eprintln!("GPU_SETUP_INITIAL n={n} cpu_setup_ns={} gpu_setup_ns={} initial_ns={} g1_setup={} g2_setup={} resident={} journal_peak={}",
            cpu_setup.as_nanos(),gpu_setup.as_nanos(),initial.as_nanos(),setup_points.0,setup_points.1,
            session.backend().stats().bytes_allocated,session.backend().resident_diagnostics().unwrap().journal_peak_bytes);
    }
}
