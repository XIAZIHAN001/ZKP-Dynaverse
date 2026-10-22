use ark_bn254::{Fr, G1Projective, G2Projective};
use ark_ec::{CurveGroup, Group};
use ark_ff::{One, UniformRand, Zero};
use dynaverse_gpu::FixedBaseCpuBackend;
use dynaverse_protocol::OperatorBackend;
use rand::{rngs::StdRng, SeedableRng};

#[test]
fn cpu_fixed_tables_reuse_and_match_independent_scalar_multiplication() {
    let backend = FixedBaseCpuBackend::default();
    assert!(backend.fixed_base_g1(&[]).unwrap().is_empty());
    assert_eq!(backend.table_build_counts(), (0, 0));
    let mut rng = StdRng::seed_from_u64(17001);
    let mut scalars = vec![Fr::zero(), Fr::one(), -Fr::one()];
    for _ in 0..257 { scalars.push(Fr::rand(&mut rng)); }
    for _ in 0..2 {
        let g1 = backend.fixed_base_g1(&scalars).unwrap();
        let g2 = backend.fixed_base_g2(&scalars).unwrap();
        for i in 0..scalars.len() {
            assert_eq!(g1[i], (G1Projective::generator()*scalars[i]).into_affine());
            assert_eq!(g2[i], (G2Projective::generator()*scalars[i]).into_affine());
        }
    }
    assert_eq!(backend.table_build_counts(), (1, 1));
}

#[test]
fn cpu_fixed_table_setup_proof_matches_naive_setup() {
    use dynaverse_protocol::{dynaverse::DynaverseParams, toy_circuit::ToyCircuit};
    let circuit = ToyCircuit::new(8).unwrap();
    let (instance, witness) = circuit.valid_instance();
    let reference = DynaverseParams::setup(circuit.clone(), &mut StdRng::seed_from_u64(17002)).unwrap();
    let optimized = DynaverseParams::setup_with_backend(circuit,
        &mut StdRng::seed_from_u64(17002), &FixedBaseCpuBackend::default()).unwrap();
    let proof = optimized.prove_with_aux(&instance, &witness).unwrap().0;
    assert_eq!(proof, reference.prove_with_aux(&instance, &witness).unwrap().0);
    assert!(reference.verifying_key().verify_reference(&instance, &proof).unwrap());
}

#[test]
fn setup_trace_covers_exclusive_srs_and_public_preparation_phases() {
    use dynaverse_protocol::{dynaverse::DynaverseParams, toy_circuit::ToyCircuit, trace::{CpuTrace,CpuPhase}};
    let mut trace=CpuTrace::new();
    let started=std::time::Instant::now();
    DynaverseParams::setup_with_backend_traced(ToyCircuit::new(4).unwrap(),
        &mut StdRng::seed_from_u64(17004),&FixedBaseCpuBackend::default(),&mut trace).unwrap();
    assert!(trace.recorded_total()<=started.elapsed());
    for phase in [CpuPhase::SetupTrapdoors,CpuPhase::SetupDynamo,CpuPhase::SetupKzg,
        CpuPhase::SetupConsistency,CpuPhase::SetupSparse,CpuPhase::SetupPublic] {
        assert!(!trace.elapsed(phase).is_zero(),"missing {}",phase.name());
    }
}
