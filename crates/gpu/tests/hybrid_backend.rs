#![cfg(feature = "icicle")]

use ark_bn254::{Fq, Fr, G1Affine, G1Projective};
use ark_ec::{CurveGroup, Group, VariableBaseMSM};
use ark_ff::{UniformRand, Zero};
use dynaverse_gpu::{HybridBackend, HybridThresholds, IcicleBackend, MAX_MSM_SIZE};
use dynaverse_protocol::{AnchorOpeningPolicy, BatchMsmRequest, BatchOpenMode, OperatorBackend, Update, VerificationMode};
use dynaverse_protocol::dynaverse::DynaverseParams;
use dynaverse_protocol::toy_circuit::ToyCircuit;
use dynaverse_protocol::trace::CpuTrace;
use rand::{rngs::StdRng, SeedableRng};

#[test]
fn hybrid_timeline_preserves_known_math_routes_and_event_domains() {
    let gpu = IcicleBackend::new().unwrap();
    let backend = HybridBackend::new(&gpu, 2).unwrap();
    let generator = G1Projective::generator();
    let points = [generator.into_affine(); 2];
    let label = std::ffi::CString::new("dvgpu/test/operation=prove/step=0/call=0/size=2").unwrap();
    for (scalars, multiple, expect_gpu) in [
        (vec![Fr::from(9u64)], 9u64, false),
        (vec![Fr::from(0u64), Fr::from(1u64)], 1, false),
        (vec![Fr::from(2u64), Fr::from(3u64)], 5, true),
    ] {
        let (actual, diagnostic) = backend.msm_timeline(&points[..scalars.len()], &scalars, &label).unwrap();
        assert_eq!(actual, (generator * Fr::from(multiple)).into_affine());
        assert_eq!(diagnostic.is_some(), expect_gpu);
        if let Some((profile, timeline)) = diagnostic {
            assert_eq!(timeline.bytes[1], 128);
            assert_eq!(timeline.bytes[2], 64);
            assert_eq!(timeline.event_calls[1], 1);
            assert_eq!(timeline.event_calls[2], 1);
            assert!(timeline.event_ns[3] > 0);
            assert_eq!(timeline.event_calls[4], 0); // Host wait has no GPU event cost.
            let mut wall_sum = timeline.instrumentation_inclusive_ns;
            for ns in timeline.wall_ns { wall_sum += ns; }
            assert_eq!(wall_sum as u128, profile.icicle_inclusive.as_nanos());
            let mut kernel_sum = 0;
            for ns in timeline.kernel_ns { kernel_sum += ns; }
            assert_eq!(kernel_sum, timeline.event_ns[3]);
        }
    }
    assert!(backend.msm_timeline(&points, &[Fr::from(2u64)], &label).is_err());
    let (empty, profile, timeline) = gpu.msm_timeline(&[], &[], &label).unwrap();
    assert_eq!(empty, G1Affine::identity());
    assert_eq!(profile.icicle_inclusive.as_nanos(), 0);
    assert_eq!(timeline.event_calls, [0; 9]);
}

#[test]
#[ignore = "requires real CUDA allocation failure fixture at allocation 1"]
fn calibrated_ntt_uses_gpu_only_for_enabled_direction_and_measured_size() {
    let gpu = IcicleBackend::new().unwrap();
    let hybrid = HybridBackend::with_thresholds(&gpu, HybridThresholds {
        dense_msm_min: None, sparse_msm_min: None,
        forward_ntt_min: Some(32), inverse_ntt_min: None, max_measured_size: 64,
    }).unwrap();
    let mut short = vec![Fr::from(1u64); 16];
    hybrid.ntt(&mut short, false).unwrap();
    let mut large = vec![Fr::from(1u64); 128];
    hybrid.ntt(&mut large, false).unwrap();
    let mut input = vec![Fr::from(2u64); 32];
    let before = input.clone();
    assert!(hybrid.ntt(&mut input, false).is_err(), "enabled calibrated NTT must execute GPU");
    assert_eq!(input, before);
    assert!(hybrid.ntt(&mut input, false).unwrap_err().to_string().contains("retired"));
    // Disabled inverse still independently uses the CPU after owner failure.
    hybrid.ntt(&mut input, true).unwrap();
    assert_eq!(input[0], Fr::from(2u64));
    assert!(input[1..].iter().all(|value| value.is_zero()));
}

#[test]
fn calibration_thresholds_reject_unmeasured_minima() {
    let valid = HybridThresholds { dense_msm_min: Some(16), sparse_msm_min: None,
        forward_ntt_min: Some(32), inverse_ntt_min: None, max_measured_size: 64 };
    valid.validate().unwrap();
    for invalid in [
        HybridThresholds { max_measured_size: 0, ..valid },
        HybridThresholds { max_measured_size: MAX_MSM_SIZE + 1, ..valid },
        HybridThresholds { inverse_ntt_min: Some(0), ..valid },
        HybridThresholds { sparse_msm_min: Some(65), ..valid },
    ] { assert!(invalid.validate().is_err()); }
}

#[test]
fn calibrated_msm_separates_sparse_binary_dense_and_unmeasured_sizes() {
    let gpu = IcicleBackend::new().unwrap();
    let hybrid = HybridBackend::with_thresholds(&gpu, HybridThresholds {
        dense_msm_min: Some(8), sparse_msm_min: None,
        forward_ntt_min: None, inverse_ntt_min: None, max_measured_size: 64,
    }).unwrap();
    let generator = G1Projective::generator();
    for (length, nonzero, scalar, expect_gpu) in [
        (64usize, 1usize, 2u64, false), (64, 64, 1, false),
        (64, 64, 2, true), (128, 128, 2, false), (4, 4, 2, false),
    ] {
        let points = vec![generator.into_affine(); length];
        let mut scalars = vec![Fr::zero(); length];
        for value in scalars.iter_mut().take(nonzero) { *value = Fr::from(scalar); }
        let (actual, profile) = hybrid.msm_profiled(&points, &scalars).unwrap();
        assert_eq!(actual, (generator * Fr::from(nonzero as u64 * scalar)).into_affine());
        assert_eq!(profile.is_some(), expect_gpu);
    }
}

#[test]
#[ignore = "requires real allocation/event failure fixture and a quiet GPU"]
fn hybrid_timeline_propagates_native_failure_and_keeps_cpu_route() {
    assert!(std::env::var_os("DVGPU_TEST_FAIL_ALLOC_AT").is_some()
        || std::env::var_os("DVGPU_TEST_FAIL_EVENT_AT").is_some());
    let gpu = IcicleBackend::new().unwrap();
    let backend = HybridBackend::new(&gpu, 2).unwrap();
    let generator = G1Projective::generator();
    let points = [generator.into_affine(); 2];
    let label = std::ffi::CString::new("dvgpu/failure/operation=prove/step=0/call=0/size=2").unwrap();
    let binary = [Fr::from(0u64), Fr::from(1u64)];
    let dense = [Fr::from(2u64), Fr::from(3u64)];
    assert!(backend.msm_timeline(&points, &binary, &label).unwrap().1.is_none());
    assert!(backend.msm_timeline(&points, &dense, &label).is_err());
    let retired = backend.msm_timeline(&points, &dense, &label).unwrap_err();
    assert!(retired.to_string().contains("retired"));
    assert_eq!(backend.msm_timeline(&points, &binary, &label).unwrap().0, generator.into_affine());
}

#[test]
fn hybrid_diagnostics_follow_real_routes_and_preserve_cpu_equality() {
    let gpu = IcicleBackend::new().unwrap();
    let backend = HybridBackend::new(&gpu, 2).unwrap();
    let generator = G1Projective::generator();
    let points = [generator.into_affine(); 2];
    for (scalars, expected_multiple, expect_gpu) in [
        (vec![Fr::from(9u64)], 9u64, false),
        (vec![Fr::from(0u64), Fr::from(1u64)], 1, false),
        (vec![Fr::from(2u64), Fr::from(3u64)], 5, true),
    ] {
        let (actual, profile) = backend.msm_profiled(&points[..scalars.len()], &scalars).unwrap();
        assert_eq!(actual, (generator * Fr::from(expected_multiple)).into_affine());
        assert_eq!(profile.is_some(), expect_gpu);
        if let Some(profile) = profile { assert!(profile.icicle_inclusive.as_nanos() > 0); }
    }
    assert!(backend.msm_profiled(&points, &[Fr::from(2u64)]).is_err());
    let invalid = G1Affine::new_unchecked(Fq::from(1u64), Fq::from(1u64));
    assert!(backend.msm_profiled(&[invalid], &[Fr::zero()]).is_err());
}

#[test]
#[ignore = "requires native allocation failure fixture with FAIL_ALLOC_AT=1 and a quiet GPU"]
fn hybrid_diagnostics_propagate_gpu_failure_and_retirement() {
    let gpu = IcicleBackend::new().unwrap();
    let backend = HybridBackend::new(&gpu, 2).unwrap();
    let generator = G1Projective::generator();
    let points = [generator.into_affine(); 2];
    let binary = [Fr::from(0u64), Fr::from(1u64)];
    let dense = [Fr::from(2u64), Fr::from(3u64)];
    let (actual, profile) = backend.msm_profiled(&points, &binary).unwrap();
    assert_eq!(actual, generator.into_affine());
    assert!(profile.is_none());
    assert!(backend.msm_profiled(&points, &dense).is_err());
    assert!(backend.msm_profiled(&points, &dense).is_err());
    assert_eq!(backend.msm_profiled(&points, &binary).unwrap().0, generator.into_affine());
}

#[test]
fn hybrid_msm_matches_cpu_across_threshold_and_preserves_rejections() {
    let gpu = IcicleBackend::new().unwrap();
    let backend = HybridBackend::new(&gpu, 17).unwrap();
    let mut rng = StdRng::seed_from_u64(2930);
    for length in [0usize, 1, 16, 17, 18, 1024] {
        let mut points = Vec::new();
        let mut scalars = Vec::new();
        for _ in 0..length {
            points.push(G1Projective::rand(&mut rng).into_affine());
            scalars.push(Fr::rand(&mut rng));
        }
        let expected = G1Projective::msm_unchecked(&points, &scalars).into_affine();
        assert_eq!(backend.msm(&points, &scalars).unwrap(), expected);
    }
    let point = G1Projective::generator().into_affine();
    assert!(backend.msm(&[point], &[]).is_err());
    let invalid = G1Affine::new_unchecked(Fq::from(1u64), Fq::from(1u64));
    assert!(backend.msm(&[invalid], &[Fr::zero()]).is_err());
    assert!(HybridBackend::new(&gpu, 0).is_err());
    assert!(HybridBackend::new(&gpu, MAX_MSM_SIZE + 1).is_err());
    let oversized = vec![G1Affine::identity(); MAX_MSM_SIZE + 1];
    assert!(backend.msm(&oversized, &vec![Fr::zero(); oversized.len()]).is_err());
}

#[test]
fn hybrid_batch_routes_each_checked_slice() {
    let gpu = IcicleBackend::new().unwrap();
    let backend = HybridBackend::new(&gpu, 2).unwrap();
    let generator = G1Projective::generator();
    let points = [generator.into_affine(); 3];
    let sets: [&[G1Affine]; 2] = [&points, &points];
    let scalars = [Fr::from(9u64), Fr::from(2u64), Fr::from(3u64)];
    assert_eq!(backend.msm_batch(&BatchMsmRequest { point_sets: &sets, shared_scalars: &scalars,
        offsets: &[1, 0], lengths: &[2, 1] }).unwrap(),
        vec![(generator * Fr::from(5u64)).into_affine(), (generator * Fr::from(9u64)).into_affine()]);
    assert!(backend.msm_batch(&BatchMsmRequest { point_sets: &sets, shared_scalars: &scalars,
        offsets: &[usize::MAX, 0], lengths: &[2, 1] }).is_err());
}

fn compare_proofs(n: usize, min_points: usize, rebuild: bool) {
    compare_configured_proofs(n, min_points, rebuild, 0);
}

fn compare_configured_proofs(n: usize, min_points: usize, rebuild: bool, factor: u32) {
    let gpu = IcicleBackend::with_large_bucket_factor(factor).unwrap();
    let backend = HybridBackend::new(&gpu, min_points).unwrap();
    let circuit = ToyCircuit::new(n).unwrap();
    let (instance, witness) = circuit.valid_instance();
    let mut rng = StdRng::seed_from_u64(2931 + n as u64);
    let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
    let vk = params.verifying_key();
    for policy in [AnchorOpeningPolicy::Dense, AnchorOpeningPolicy::Precompute] {
        let (mut expected, mut cpu_aux) = params.prove_with_aux_policy(&instance, &witness, policy).unwrap();
        let (mut actual, mut aux) = params.prove_with_aux_traced_with_backend(
            &instance, &witness, policy, &backend, &mut CpuTrace::new()).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(aux, cpu_aux);
        // A fixed number of small updates cannot reach the sqrt(n) rebuild
        // threshold at larger sizes. Include a real batch crossing that bound,
        // between repeated updates and subsequent incremental updates.
        let mut rebuild_span = 1;
        while rebuild_span * rebuild_span < n { rebuild_span += 1; }
        let update_gates = if rebuild {
            vec![vec![0usize], vec![0], (0..rebuild_span).collect(), vec![1], vec![2], vec![3]]
        } else { vec![vec![0usize], vec![1]] };
        let mut rebuilds = 0;
        for gates in update_gates {
            let old = cpu_aux.current_witness().clone();
            let mut update = Update { indices: Vec::new(), values: Vec::new() };
            for gate in gates {
                let value = old.wires[4 * n + gate] + Fr::from(1u64);
                update.indices.extend([4 * n + gate, 5 * n + gate]);
                update.values.extend([value, old.wires[3 * n + gate] * value]);
            }
            let previous_anchor = actual.anchor_generation;
            expected = params.update_proof(&instance, &instance, &old, &update, &expected, &mut cpu_aux).unwrap();
            actual = params.update_proof_traced_with_backend(&instance, &instance, &old, &update, &actual,
                &mut aux, VerificationMode::Product, BatchOpenMode::Prepared, &backend, &mut CpuTrace::new()).unwrap();
            assert_eq!(actual, expected);
            assert_eq!(aux, cpu_aux);
            if actual.anchor_generation > previous_anchor { rebuilds += 1; }
            assert!(vk.verify_reference(&instance, &actual).unwrap());
            assert!(vk.verify(&instance, &actual).unwrap());
        }
        if rebuild { assert!(rebuilds > 0); }
        eprintln!("HYBRID_PROOF_OK n={n} policy={policy:?} rebuild={rebuild} factor={factor} rebuilds={rebuilds}");
    }
}

#[test]
fn hybrid_proofs_updates_and_rebuilds_equal_original_cpu_path() { compare_proofs(16, 32, true); }

#[test]
#[ignore = "run explicitly for n=256/512/1024 hybrid proof and aux equality"]
fn hybrid_proofs_equal_cpu_at_scale() {
    for n in [256usize, 512, 1024] { compare_proofs(n, 512, false); }
}

#[test]
#[ignore = "explicit factor=0/10 CPU proof/aux equality, updates and rebuilds through n=2048"]
fn configured_hybrid_proofs_equal_cpu_across_sizes_and_rebuilds() {
    for factor in [0u32, 10] {
        for n in [16usize, 256, 512, 1024, 2048] {
            compare_configured_proofs(n, if n == 16 { 32 } else { 512 }, true, factor);
        }
    }
}

#[test]
#[ignore = "requires native allocation failure fixture with FAIL_ALLOC_AT=1 and a quiet GPU"]
fn hybrid_cpu_routes_avoid_cuda_and_gpu_failures_are_not_hidden() {
    let gpu = IcicleBackend::new().unwrap();
    let backend = HybridBackend::new(&gpu, 2).unwrap();
    let generator = G1Projective::generator();
    let points = [generator.into_affine(); 2];
    assert_eq!(backend.msm(&points[..1], &[Fr::from(9u64)]).unwrap(), (generator * Fr::from(9u64)).into_affine());
    assert_eq!(backend.msm(&points, &[Fr::from(0u64), Fr::from(0u64)]).unwrap(), G1Affine::identity());
    assert_eq!(backend.msm(&points, &[Fr::from(1u64), Fr::from(1u64)]).unwrap(), (generator + generator).into_affine());
    assert_eq!(backend.msm(&points, &[Fr::from(0u64), Fr::from(1u64)]).unwrap(), generator.into_affine());
    let error = backend.msm(&points, &[Fr::from(2u64), Fr::from(3u64)]).unwrap_err();
    assert!(error.to_string().contains("ICICLE CUDA error"), "{error}");
    // No transparent fallback follows the failed GPU dispatch. The CPU route
    // remains independent of the native owner's retired state.
    assert!(backend.msm(&points, &[Fr::from(2u64), Fr::from(3u64)]).is_err());
    assert_eq!(backend.msm(&points[..1], &[Fr::from(9u64)]).unwrap(), (generator * Fr::from(9u64)).into_affine());
}
