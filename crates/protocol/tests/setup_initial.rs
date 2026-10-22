use ark_bn254::Fr;
use ark_ff::One;
use dynaverse_protocol::{dynaverse::DynaverseParams, toy_circuit::ToyCircuit, workcount};
use rand::{rngs::StdRng, SeedableRng};
use dynaverse_protocol::{BatchMsmRequest, OperatorBackend};
use dynaverse_protocol::incremental_rebuild::{IncrementalConfig, operators::CpuRebuildBackend};
use ark_bn254::G1Affine;

struct RejectSetup;
impl OperatorBackend for RejectSetup {
    fn msm(&self, _: &[G1Affine], _: &[Fr]) -> anyhow::Result<G1Affine> { anyhow::bail!("unexpected MSM") }
    fn msm_batch(&self, _: &BatchMsmRequest<'_>) -> anyhow::Result<Vec<G1Affine>> { anyhow::bail!("unexpected batch") }
    fn ntt(&self, _: &mut [Fr], _: bool) -> anyhow::Result<()> { anyhow::bail!("unexpected NTT") }
    fn fixed_base_g1(&self, _: &[Fr]) -> anyhow::Result<Vec<G1Affine>> { anyhow::bail!("setup device failed") }
}

#[test]
fn setup_backend_errors_propagate_without_reference_fallback() {
    let error = DynaverseParams::setup_with_backend(ToyCircuit::new(2).unwrap(),
        &mut StdRng::seed_from_u64(6102), &RejectSetup).err().unwrap();
    assert!(error.to_string().contains("setup device failed"));
}

// Exact proof comparison uses the retained independent dense prover. A field
// or MSM routing error in the new initial program changes these commitments.
#[test]
fn resident_initial_proof_matches_dense_reference_and_uses_resident_msms() {
    for n in [1, 2, 16] {
        let circuit = ToyCircuit::new(n).unwrap();
        let (instance, witness) = circuit.valid_instance();
        let params = DynaverseParams::setup(circuit, &mut StdRng::seed_from_u64(6300+n as u64)).unwrap();
        let expected = params.prove_with_aux(&instance, &witness).unwrap().0;
        let session = params.start_incremental_session(&instance, &witness,
            IncrementalConfig::default(), CpuRebuildBackend::default()).unwrap();
        assert_eq!(*session.proof(), expected);
        use dynaverse_protocol::incremental_rebuild::operators::RebuildBackend;
        eprintln!("INITIAL_RESIDENT n={n} stats={:?}",session.backend().stats());
        if n==16 {assert!(session.backend().stats().calls<600,"initialization uses update-sized submissions");}
        assert!(session.backend().stats().msm_g1_points > 0, "initial proof bypassed resident program");
        assert!(session.backend().stats().msm_g2_points > 0);
        assert!(params.verifying_key().verify_reference(&instance, session.proof()).unwrap());
    }
}

#[test]
fn balanced_public_divisor_matches_sequential_oracle_and_checks_indices() {
    use dynaverse_protocol::domain::{radix2_domain, vanishing_polynomial, vanishing_polynomial_balanced};
    let domain=radix2_domain(256).unwrap();
    for indices in [vec![],vec![255],vec![0,7,100],(192..256).collect()] {
        assert_eq!(vanishing_polynomial_balanced(&domain,&indices).unwrap(),
            vanishing_polynomial(&domain,&indices).unwrap());
    }
    assert!(vanishing_polynomial_balanced(&domain,&[1,1]).is_err());
    assert!(vanishing_polynomial_balanced(&domain,&[256]).is_err());
    let (_,fast)=workcount::measure(||vanishing_polynomial_balanced(&domain,&(0..256).collect::<Vec<_>>()).unwrap());
    let (_,slow)=workcount::measure(||vanishing_polynomial(&domain,&(0..256).collect::<Vec<_>>()).unwrap());
    assert!(fast.poly_elements < slow.poly_elements/2);
}

#[test]
fn invalid_initial_witness_is_rejected_before_allocation() {
    let circuit=ToyCircuit::new(2).unwrap();
    let (instance,mut witness)=circuit.valid_instance();
    let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(6104)).unwrap();
    witness.wires[1]+=Fr::one();
    assert!(params.start_incremental_session(&instance,&witness,
        IncrementalConfig::default(),CpuRebuildBackend::new()).is_err());
}

// Reintroducing n dense public-base MSMs makes this linear envelope fail.
#[test]
fn setup_does_not_commit_n_dense_public_vectors() {
    let n = 64;
    let (params, count) = workcount::measure(|| {
        DynaverseParams::setup(ToyCircuit::new(n).unwrap(), &mut StdRng::seed_from_u64(6101)).unwrap()
    });
    assert!(count.msm_points + count.g2_msm_points < (64 * n) as u64,
        "setup still performs quadratic public MSM work: {count:?}");
    let (instance, witness) = ToyCircuit::new(n).unwrap().valid_instance();
    let proof = params.prove_with_aux(&instance, &witness).unwrap().0;
    assert!(params.verifying_key().verify(&instance, &proof).unwrap());
    let mut wrong = instance;
    wrong.public_inputs[n-1] += Fr::one();
    assert!(!params.verifying_key().verify(&wrong, &proof).unwrap());
}

#[test]
#[ignore = "explicit setup and legacy initial timing baseline"]
fn setup_initial_timing_probe() {
    for n in [16, 64, 256] {
        let circuit = ToyCircuit::new(n).unwrap();
        let (instance, witness) = circuit.valid_instance();
        let start = std::time::Instant::now();
        let (params, work) = workcount::measure(||
            DynaverseParams::setup(circuit, &mut StdRng::seed_from_u64(6200+n as u64)).unwrap());
        let setup = start.elapsed();
        let start = std::time::Instant::now();
        let proof = params.prove_with_aux(&instance, &witness).unwrap().0;
        let initial = start.elapsed();
        assert!(params.verifying_key().verify(&instance, &proof).unwrap());
        eprintln!("SETUP_INITIAL n={n} setup_ns={} legacy_initial_ns={} msm={} g2={} poly={} ntt={}",
            setup.as_nanos(), initial.as_nanos(), work.msm_points, work.g2_msm_points,
            work.poly_elements, work.ntt_elements);
    }
}
