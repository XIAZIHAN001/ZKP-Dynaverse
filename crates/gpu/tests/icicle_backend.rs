#![cfg(feature = "icicle")]

use ark_bn254::{Fq, Fr, G1Affine, G1Projective};
use ark_ec::{CurveGroup, Group, VariableBaseMSM};
use ark_ff::{UniformRand, Zero};
use dynaverse_gpu::IcicleBackend;
use dynaverse_protocol::{AnchorOpeningPolicy, BatchMsmRequest, BatchOpenMode, OperatorBackend, Update, VerificationMode};
use dynaverse_protocol::dynaverse::DynaverseParams;
use dynaverse_protocol::toy_circuit::ToyCircuit;
use dynaverse_protocol::trace::CpuTrace;
use rand::{rngs::StdRng, SeedableRng};

// Only this ignored failure gate uses runtime observation APIs. They are not
// part of the production FFI or the protocol's public surface.
extern "C" {
    fn cudaFree(pointer: *mut std::ffi::c_void) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaDeviceGetDefaultMemPool(pool: *mut *mut std::ffi::c_void, device: i32) -> i32;
    fn cudaMemPoolTrimTo(pool: *mut std::ffi::c_void, minimum: usize) -> i32;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
}

fn available_device_memory() -> usize {
    let mut pool = std::ptr::null_mut();
    let mut free = 0;
    let mut total = 0;
    // SAFETY: output pointers refer to initialized local storage; pool is the
    // runtime's borrowed default pool. Synchronization precedes trimming.
    unsafe {
        assert_eq!(cudaFree(std::ptr::null_mut()), 0);
        assert_eq!(cudaDeviceSynchronize(), 0);
        assert_eq!(cudaDeviceGetDefaultMemPool(&mut pool, 0), 0);
        assert_eq!(cudaMemPoolTrimTo(pool, 0), 0);
        assert_eq!(cudaMemGetInfo(&mut free, &mut total), 0);
    }
    free
}

#[test]
#[ignore = "run with native failure LD_PRELOAD fixture; requires a quiet GPU"]
fn allocation_failure_releases_context_and_new_backend_can_run() {
    assert_eq!(std::env::var("DVGPU_TEST_FAIL_ALLOC_AT").unwrap(), "3");
    let point = G1Projective::generator().into_affine();
    let points = vec![point; 16384];
    let scalars = vec![Fr::from(1u64); points.len()];
    let healthy = IcicleBackend::new().unwrap();
    let before = available_device_memory();
    let backend = IcicleBackend::new().unwrap();
    assert!(backend.msm_profiled(&points, &scalars).is_err());
    // A failed context is retired, so callers cannot retry inside damaged or
    // partially allocated native state. A fresh backend is the recovery path.
    assert!(backend.msm(&points[..32], &scalars[..32]).is_err());
    drop(backend);
    let after = available_device_memory();
    eprintln!("ALLOCATION_FAILURE_MEMORY before={before} after={after}");
    assert!(after + 1024 * 1024 >= before, "failed context retained device memory");
    assert_eq!(healthy.msm(&points[..32], &scalars[..32]).unwrap(),
        (G1Projective::generator() * Fr::from(32u64)).into_affine());
    let recovered = IcicleBackend::new().unwrap();
    assert_eq!(recovered.msm_profiled(&points[..32], &scalars[..32]).unwrap().0,
        (G1Projective::generator() * Fr::from(32u64)).into_affine());
}

#[test]
fn real_icicle_msm_matches_arkworks_across_sizes_and_zero_cases() {
    // Enabling the feature requires real CUDA execution. Missing GPU is a test
    // failure, never a success obtained by skipping or silently using CPU.
    let backend = IcicleBackend::new().unwrap();
    let mut rng = StdRng::seed_from_u64(2905);
    for n in [0usize, 1, 2, 17, 1024, 8192] {
        eprintln!("ICICLE_MSM_CASE length={n}");
        let mut points = Vec::new();
        let mut scalars = Vec::new();
        for i in 0..n {
            let mut point = (G1Projective::generator() * Fr::rand(&mut rng)).into_affine();
            if i % 7 == 0 { point = G1Affine::identity(); }
            points.push(point);
            let scalar = if i % 11 == 0 { Fr::zero() } else { Fr::rand(&mut rng) };
            scalars.push(scalar);
        }
        let expected = G1Projective::msm_unchecked(&points, &scalars).into_affine();
        assert_eq!(backend.msm(&points, &scalars).unwrap(), expected, "length {n}");
        assert_eq!(backend.msm(&points, &vec![Fr::zero(); n]).unwrap(), G1Affine::identity());
        // Repeating in the same process also exercises resource cleanup.
        assert_eq!(backend.msm(&points, &scalars).unwrap(), expected);
    }
    assert_eq!(backend.msm(&[G1Projective::generator().into_affine()], &[Fr::from(3u64)]).unwrap(),
        (G1Projective::generator() * Fr::from(3u64)).into_affine());
    let point = G1Projective::generator().into_affine();
    assert_eq!(backend.msm(&[point, point], &[Fr::from(1u64), -Fr::from(1u64)]).unwrap(),
        G1Affine::identity());
}

#[test]
fn configured_owners_match_cpu_across_real_window_sizes_and_random_seeds() {
    // The two live owners exercise independent CUDA contexts and options.
    let baseline = IcicleBackend::new().unwrap();
    let candidate = IcicleBackend::with_large_bucket_factor(10).unwrap();
    for seed in [7u64, 17, 2026] {
        let mut rng = StdRng::seed_from_u64(seed);
        for n in [0usize, 1, 2, 17, 511, 512, 1022, 1023, 1024, 2048, 6144, 8192] {
            let mut points = Vec::new();
            let mut scalars = Vec::new();
            for i in 0..n {
                points.push(if i % 7 == 0 { G1Affine::identity() } else { G1Projective::rand(&mut rng).into_affine() });
                scalars.push(if i % 11 == 0 { Fr::zero() } else { Fr::rand(&mut rng) });
            }
            let expected = G1Projective::msm_unchecked(&points, &scalars).into_affine();
            for (factor, backend) in [(0, &baseline), (10, &candidate)] {
                assert_eq!(backend.msm(&points, &scalars).unwrap(), expected, "seed={seed} n={n} factor={factor}");
                assert_eq!(backend.msm(&points, &vec![Fr::zero(); n]).unwrap(), G1Affine::identity());
            }
            eprintln!("CONFIGURED_MSM_CPU_OK seed={seed} n={n}");
        }
    }
}

#[test]
fn configured_profile_and_timeline_match_cpu_on_large_bucket_inputs() {
    let point = G1Projective::generator().into_affine();
    let label = std::ffi::CString::new("configured_large_bucket_cpu_check").unwrap();
    for factor in [0, 1, 10, 100] {
        let backend = IcicleBackend::with_large_bucket_factor(factor).unwrap();
        for n in [1024, 8192] {
            let points = vec![point; n];
            let scalars = vec![Fr::from(3u64); n];
            let expected = (G1Projective::generator() * Fr::from(3 * n as u64)).into_affine();
            assert_eq!(backend.msm(&points, &scalars).unwrap(), expected);
            assert_eq!(backend.msm_profiled(&points, &scalars).unwrap().0, expected);
            let (actual, profile, timeline) = backend.msm_timeline(&points, &scalars, &label).unwrap();
            assert_eq!(actual, expected);
            assert!(profile.native >= profile.native_conversion + profile.icicle_inclusive + profile.native_output);
            let mut wall_sum = timeline.instrumentation_inclusive_ns as u128;
            for value in timeline.wall_ns { wall_sum += value as u128; }
            assert_eq!(wall_sum, profile.icicle_inclusive.as_nanos());
            let mut kernel_sum = 0;
            for value in timeline.kernel_ns { kernel_sum += value; }
            assert_eq!(kernel_sum, timeline.event_ns[3]);
            assert_eq!(timeline.c, if n == 1024 { 6 } else { 9 });
            assert_eq!(timeline.bitsize, 254);
            eprintln!("CONFIGURED_THREE_ENTRIES_OK factor={factor} n={n}");
        }
    }
}

#[test]
fn invalid_large_bucket_factor_is_rejected_before_cuda_initialization() {
    if std::env::var_os("DVGPU_TEST_INVALID_FACTOR").is_some() {
        let error = IcicleBackend::with_large_bucket_factor(u32::MAX).err().unwrap();
        assert!(error.to_string().contains("large bucket factor"), "{error}");
        return;
    }
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "invalid_large_bucket_factor_is_rejected_before_cuda_initialization", "--nocapture"])
        .env("DVGPU_TEST_INVALID_FACTOR", "1").env("CUDA_VISIBLE_DEVICES", "").status().unwrap();
    assert!(status.success());
}

#[test]
fn profiled_msm_matches_cpu_and_reports_nonoverlapping_host_spans() {
    let backend = IcicleBackend::new().unwrap();
    let point = G1Projective::generator().into_affine();
    let points = [point, point, G1Affine::identity()];
    let scalars = [Fr::from(2u64), Fr::from(3u64), Fr::from(99u64)];
    let (actual, profile) = backend.msm_profiled(&points, &scalars).unwrap();
    assert_eq!(actual, (G1Projective::generator() * Fr::from(5u64)).into_affine());
    assert!(profile.total >= profile.validation + profile.encoding + profile.native + profile.decode);
    assert!(profile.native >= profile.native_conversion + profile.icicle_inclusive + profile.native_output);
    assert!(!profile.icicle_inclusive.is_zero());
    assert_eq!(backend.msm(&points, &scalars).unwrap(), actual);
}

#[test]
fn profiled_msm_preserves_empty_and_rejection_boundaries() {
    let backend = IcicleBackend::new().unwrap();
    let (actual, profile) = backend.msm_profiled(&[], &[]).unwrap();
    assert_eq!(actual, G1Affine::identity());
    assert!(profile.native.is_zero());
    assert!(profile.icicle_inclusive.is_zero());
    let point = G1Projective::generator().into_affine();
    assert!(backend.msm_profiled(&[point], &[]).is_err());
    let invalid = G1Affine::new_unchecked(Fq::from(1u64), Fq::from(1u64));
    assert!(backend.msm_profiled(&[invalid], &[Fr::zero()]).is_err());
    // A Rust validation failure must not retire the healthy native owner.
    assert_eq!(backend.msm_profiled(&[point], &[Fr::from(1u64)]).unwrap().0, point);
}

#[test]
fn unavailable_gpu_returns_error_without_cpu_fallback() {
    if std::env::var_os("DYNAVERSE_TEST_NO_GPU").is_some() {
        assert!(IcicleBackend::new().is_err());
        return;
    }
    // A child process gets an isolated CUDA visibility setting; changing the
    // current process environment could race with other parallel GPU tests.
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "unavailable_gpu_returns_error_without_cpu_fallback", "--nocapture"])
        .env("DYNAVERSE_TEST_NO_GPU", "1").env("CUDA_VISIBLE_DEVICES", "")
        .status().unwrap();
    assert!(status.success());
}

#[test]
fn icicle_rejects_invalid_inputs_and_obeys_batch_scalar_ranges() {
    let backend = IcicleBackend::new().unwrap();
    let point = G1Projective::generator().into_affine();
    assert!(backend.msm(&[point], &[]).is_err());
    let invalid = G1Affine::new_unchecked(Fq::from(1u64), Fq::from(1u64));
    assert!(backend.msm(&[invalid], &[Fr::from(1u64)]).is_err());
    let scalars = [Fr::from(9u64), Fr::from(2u64), Fr::from(3u64)];
    let first = [point, point];
    let second = [point];
    let sets: [&[G1Affine]; 2] = [&first, &second];
    let request = BatchMsmRequest { point_sets: &sets, shared_scalars: &scalars,
        offsets: &[1, 0], lengths: &[2, 1] };
    assert_eq!(backend.msm_batch(&request).unwrap(), vec![
        (G1Projective::generator() * Fr::from(5u64)).into_affine(),
        (G1Projective::generator() * Fr::from(9u64)).into_affine()]);
    for (offsets, lengths) in [(&[usize::MAX, 0][..], &[2, 1][..]),
        (&[2, 0][..], &[2, 1][..]), (&[0][..], &[2, 1][..]),
        (&[0, 0][..], &[3, 1][..])] {
        assert!(backend.msm_batch(&BatchMsmRequest { point_sets: &sets,
            shared_scalars: &scalars, offsets, lengths }).is_err());
    }
}

#[test]
fn icicle_rejects_oversized_msm_before_native_allocation() {
    let backend = IcicleBackend::new().unwrap();
    // The baseline contract stops at 2^20. Larger sizes can overflow ICICLE's
    // internal 32-bit indexing, even before its public int size limit.
    let points = vec![G1Affine::identity(); (1 << 20) + 1];
    let scalars = vec![Fr::zero(); points.len()];
    let error = backend.msm(&points, &scalars).unwrap_err();
    assert!(error.to_string().contains("size bound"), "{error}");
}

#[test]
fn icicle_kzg_path_preserves_initial_updates_rebuild_and_aux() {
    let backend = IcicleBackend::new().unwrap();
    let n = 16;
    let circuit = ToyCircuit::new(n).unwrap();
    let (instance, witness) = circuit.valid_instance();
    let mut rng = StdRng::seed_from_u64(2906);
    let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
    let vk = params.verifying_key();
    for policy in [AnchorOpeningPolicy::Dense, AnchorOpeningPolicy::Precompute] {
        let (mut cpu, mut cpu_aux) = params.prove_with_aux_policy(&instance, &witness, policy).unwrap();
        let (mut gpu, mut gpu_aux) = params.prove_with_aux_traced_with_backend(
            &instance, &witness, policy, &backend, &mut CpuTrace::new()).unwrap();
        assert_eq!(cpu, gpu);
        assert_eq!(cpu_aux, gpu_aux);
        for gate in [0usize, 0, 1, 2, 3, 4] {
            let old = cpu_aux.current_witness().clone();
            let value = old.wires[4 * n + gate] + Fr::from(1u64);
            let update = Update { indices: vec![4 * n + gate, 5 * n + gate],
                values: vec![value, old.wires[3 * n + gate] * value] };
            cpu = params.update_proof(&instance, &instance, &old, &update, &cpu, &mut cpu_aux).unwrap();
            gpu = params.update_proof_traced_with_backend(&instance, &instance, &old, &update,
                &gpu, &mut gpu_aux, VerificationMode::Product, BatchOpenMode::Prepared,
                &backend, &mut CpuTrace::new()).unwrap();
            assert_eq!(cpu, gpu);
            assert_eq!(cpu_aux, gpu_aux);
            assert!(vk.verify_reference(&instance, &gpu).unwrap());
        }
        assert!(gpu.anchor_generation > 0);
    }
}

#[test]
#[ignore = "run explicitly for the real GPU n=256/512/1024 proof equality gate"]
fn icicle_kzg_proofs_equal_cpu_at_profiled_scale_sizes() {
    let backend = IcicleBackend::new().unwrap();
    for n in [256usize, 512, 1024] {
        let circuit = ToyCircuit::new(n).unwrap();
        let (instance, witness) = circuit.valid_instance();
        let mut rng = StdRng::seed_from_u64(2908 + n as u64);
        let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
        let vk = params.verifying_key();
        for policy in [AnchorOpeningPolicy::Dense, AnchorOpeningPolicy::Precompute] {
            let (cpu, mut cpu_aux) = params.prove_with_aux_policy(&instance, &witness, policy).unwrap();
            let (gpu, mut gpu_aux) = params.prove_with_aux_traced_with_backend(
                &instance, &witness, policy, &backend, &mut CpuTrace::new()).unwrap();
            assert_eq!(cpu, gpu, "initial n={n}");
            assert_eq!(cpu_aux, gpu_aux);
            let update = Update { indices: vec![4 * n], values: vec![witness.wires[4 * n] + Fr::from(1u64)] };
            let cpu = params.update_proof(&instance, &instance, &witness, &update, &cpu, &mut cpu_aux).unwrap();
            let gpu = params.update_proof_traced_with_backend(&instance, &instance, &witness, &update,
                &gpu, &mut gpu_aux, VerificationMode::Product, BatchOpenMode::Prepared,
                &backend, &mut CpuTrace::new()).unwrap();
            assert_eq!(cpu, gpu, "updated n={n}");
            assert_eq!(cpu_aux, gpu_aux);
            assert!(vk.verify_reference(&instance, &gpu).unwrap());
            assert!(vk.verify(&instance, &gpu).unwrap());
            eprintln!("ICICLE_PROOF_SCALE_OK n={n} policy={policy:?}");
        }
    }
}
