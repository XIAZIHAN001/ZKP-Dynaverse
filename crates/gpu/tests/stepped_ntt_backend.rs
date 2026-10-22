#![cfg(feature = "icicle")]

use ark_bn254::{Fr, G1Affine, G1Projective, G2Projective};
use ark_ec::{CurveGroup, Group};
use ark_ff::{FftField, Field, Zero};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use dynaverse_gpu::IcicleBackend;
use dynaverse_protocol::OperatorBackend;
use dynaverse_protocol::incremental_rebuild::operators::{
    BufferId, BufferKind, FieldBinaryOp, RebuildBackend,
};

// A wrong root, transpose, row offset, batch stride or inverse normalization
// changes this independent full-transform expectation.
fn four_step<B: RebuildBackend>(backend: &mut B, kind: BufferKind,
    input: BufferId, a: BufferId, b: BufferId, n: usize, inverse: bool) {
    let log = n.trailing_zeros();
    let rows = 1usize << (log / 2);
    let cols = n / rows;
    let mut root = Fr::get_root_of_unity(n as u64).unwrap();
    if inverse { root = root.inverse().unwrap(); }
    for start in (0..n).step_by(7) {
        backend.transpose(kind, input, a, rows, cols, start, 7.min(n - start), None).unwrap();
    }
    for row in (0..cols).step_by(3) {
        backend.transform_rows(kind, a, b, rows, row, 3.min(cols - row), inverse).unwrap();
    }
    for start in (0..n).step_by(11) {
        backend.transpose(kind, b, a, cols, rows, start, 11.min(n - start), Some(root)).unwrap();
    }
    for row in (0..rows).step_by(2) {
        backend.transform_rows(kind, a, b, cols, row, 2.min(rows - row), inverse).unwrap();
    }
    for start in (0..n).step_by(13) {
        backend.transpose(kind, b, a, rows, cols, start, 13.min(n - start), None).unwrap();
    }
}

#[test]
fn resident_four_step_fr_and_ecntt_match_independent_cpu() {
    let mut backend = IcicleBackend::with_resident_limits(512 << 20, 16 << 20).unwrap();
    let mut buffers = Vec::new();
    for kind in [BufferKind::Fr, BufferKind::G1] {
        for _ in 0..3 { buffers.push(backend.alloc(kind, 1024, true).unwrap()); }
    }
    backend.seal();
    let allocated = backend.stats().bytes_allocated;
    for n in [1usize, 2, 8, 16, 32, 128, 256, 1024] {
        let input: Vec<Fr> = (0..n).map(|i| Fr::from((i * i + 11) as u64)).collect();
        let points: Vec<G1Affine> = input.iter().enumerate().map(|(i, value)| {
            if i % 7 == 0 { G1Affine::identity() }
            else { (G1Projective::generator() * value).into_affine() }
        }).collect();
        let domain = Radix2EvaluationDomain::<Fr>::new(n).unwrap();
        for inverse in [false, true] {
            backend.write_fields(buffers[0], 0, &input).unwrap();
            backend.write_points(buffers[3], 0, &points).unwrap();
            let mut expected = input.clone();
            let mut expected_points: Vec<G1Projective> = points.iter().map(|p| (*p).into()).collect();
            if inverse { domain.ifft_in_place(&mut expected); domain.ifft_in_place(&mut expected_points); }
            else { domain.fft_in_place(&mut expected); domain.fft_in_place(&mut expected_points); }
            four_step(&mut backend, BufferKind::Fr, buffers[0], buffers[1], buffers[2], n, inverse);
            four_step(&mut backend, BufferKind::G1, buffers[3], buffers[4], buffers[5], n, inverse);
            let mut actual = vec![Fr::zero(); n];
            let mut actual_points = vec![G1Affine::identity(); n];
            backend.read_fields(buffers[1], 0, &mut actual).unwrap();
            backend.read_points(buffers[4], 0, &mut actual_points).unwrap();
            assert_eq!(actual, expected, "Fr n={n} inverse={inverse}");
            assert_eq!(actual_points, G1Projective::normalize_batch(&expected_points), "G1 n={n} inverse={inverse}");
        }
        eprintln!("RESIDENT_FOUR_STEP_CPU_EXACT n={n}");
    }
    assert_eq!(backend.stats().bytes_allocated, allocated, "resident allocations grow after seal");
}

#[test]
fn resident_partial_msm_elementwise_and_touched_range_rollback() {
    let mut backend = IcicleBackend::with_resident_limits(256 << 20, 1 << 20).unwrap();
    let fields = backend.alloc(BufferKind::Fr, 32, true).unwrap();
    let points = backend.alloc(BufferKind::G1, 32, true).unwrap();
    let output = backend.alloc(BufferKind::G1, 32, true).unwrap();
    let readonly = backend.alloc(BufferKind::Fr, 8, false).unwrap();
    let values: Vec<Fr> = (0..32).map(|i| Fr::from(i as u64 + 1)).collect();
    let g1: Vec<G1Affine> = values.iter().map(|s| (G1Projective::generator() * s).into_affine()).collect();
    backend.write_fields(fields, 0, &values).unwrap();
    backend.write_points(points, 0, &g1).unwrap();
    backend.seal();
    assert!(backend.alloc(BufferKind::Fr, 1, true).is_err());
    assert!(backend.write_fields(readonly, 0, &[Fr::from(1u64)]).is_err());
    backend.begin_transaction().unwrap();
    backend.field_zip(FieldBinaryOp::Mul, fields, 3, fields, 3, fields, 3, 10).unwrap();
    backend.write_fields(fields, 7, &[Fr::from(999u64); 4]).unwrap();
    backend.scale_points(points, 0, fields, 0, output, 0, 16).unwrap();
    let mut changed = values.clone();
    for i in 3..13 { changed[i].square_in_place(); }
    changed[7..11].fill(Fr::from(999u64));
    let mut scaled = vec![G1Affine::identity(); 16];
    backend.read_points(output, 0, &mut scaled).unwrap();
    for i in 0..16 { assert_eq!(scaled[i], (g1[i] * changed[i]).into_affine()); }
    assert!(backend.copy_buffer(fields, 0, fields, 1, 8).is_err());
    backend.rollback_transaction().unwrap();
    let mut recovered = vec![Fr::zero(); 32];
    backend.read_fields(fields, 0, &mut recovered).unwrap();
    assert_eq!(recovered, values);
    for (offset, len) in [(0, 1), (3, 7), (7, 16)] {
        let mut expected = G1Projective::zero();
        for i in offset..offset + len { expected += g1[i] * values[i]; }
        assert_eq!(backend.msm_g1(points, offset, fields, offset, len).unwrap(), expected.into_affine());
    }
    let g2: Vec<_> = (0..8).map(|i| (G2Projective::generator() * values[i]).into_affine()).collect();
    let mut expected = G2Projective::zero();
    for i in 0..8 { expected += g2[i] * values[4 + i]; }
    assert_eq!(backend.msm_g2(&g2, fields, 4).unwrap(), expected.into_affine());
    backend.begin_transaction().unwrap();
    backend.write_fields(fields, 2, &[Fr::from(777u64)]).unwrap();
    backend.commit_transaction();
    backend.read_fields(fields, 2, &mut recovered[..1]).unwrap();
    assert_eq!(recovered[0], Fr::from(777u64));
}

#[test]
fn resident_reverse_alias_checks_and_owner_tags_match_cpu_contract() {
    let mut backend = IcicleBackend::with_resident_limits(256 << 20, 1 << 20).unwrap();
    let mut other = IcicleBackend::with_resident_limits(256 << 20, 1 << 20).unwrap();
    let fields = backend.alloc(BufferKind::Fr, 32, true).unwrap();
    let output = backend.alloc(BufferKind::Fr, 32, true).unwrap();
    let points = backend.alloc(BufferKind::G1, 32, true).unwrap();
    let point_output = backend.alloc(BufferKind::G1, 32, true).unwrap();
    let foreign = other.alloc(BufferKind::Fr, 32, true).unwrap();
    assert!(backend.buffer_len(foreign).is_err());
    let values: Vec<_> = (0..32).map(|i| Fr::from(i as u64 + 1)).collect();
    let g1: Vec<_> = values.iter().map(|s| (G1Projective::generator() * s).into_affine()).collect();
    backend.write_fields(fields, 0, &values).unwrap();
    backend.write_points(points, 0, &g1).unwrap();
    backend.seal();
    backend.reverse_copy(BufferKind::Fr, fields, 3, output, 9, 11).unwrap();
    backend.reverse_copy(BufferKind::G1, points, 3, point_output, 9, 11).unwrap();
    let mut actual = vec![Fr::zero(); 11];
    let mut actual_points = vec![G1Affine::identity(); 11];
    backend.read_fields(output, 9, &mut actual).unwrap();
    backend.read_points(point_output, 9, &mut actual_points).unwrap();
    assert_eq!(actual, values[3..14].iter().copied().rev().collect::<Vec<_>>());
    assert_eq!(actual_points, g1[3..14].iter().copied().rev().collect::<Vec<_>>());
    assert!(backend.reverse_copy(BufferKind::Fr, fields, 0, fields, 0, 8).is_err());
    assert!(backend.field_zip(FieldBinaryOp::Add, fields, 0, fields, 0, fields, 1, 8).is_err());
    for op in [FieldBinaryOp::Add, FieldBinaryOp::Sub, FieldBinaryOp::Mul] {
        backend.field_zip(op, fields, 2, fields, 7, output, 4, 8).unwrap();
        backend.read_fields(output, 4, &mut actual[..8]).unwrap();
        for i in 0..8 {
            let expected = match op {
                FieldBinaryOp::Add => values[2 + i] + values[7 + i],
                FieldBinaryOp::Sub => values[2 + i] - values[7 + i],
                FieldBinaryOp::Mul => values[2 + i] * values[7 + i],
            };
            assert_eq!(actual[i], expected);
        }
    }
    assert_eq!(backend.msm_g1(points, 32, fields, 32, 0).unwrap(), G1Affine::identity());
    backend.transform_rows(BufferKind::Fr, fields, output, 8, 4, 0, false).unwrap();
    backend.transpose(BufferKind::Fr, fields, output, 4, 8, 32, 0, None).unwrap();
}

#[test]
fn resident_partial_msms_cover_full_width_scalars_and_g2_component_order() {
    use ark_ff::UniformRand;
    use rand::{rngs::StdRng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(20261015);
    let mut backend = IcicleBackend::with_resident_limits(256 << 20, 1 << 20).unwrap();
    let scalars = backend.alloc(BufferKind::Fr, 128, true).unwrap();
    let points = backend.alloc(BufferKind::G1, 128, true).unwrap();
    let mut values: Vec<_> = (0..128).map(|_| Fr::rand(&mut rng)).collect();
    values[7] = -Fr::from(1u64); values[9] = Fr::zero();
    let mut g1 = Vec::new(); let mut g2 = Vec::new();
    for i in 0..128 {
        let coefficient = Fr::rand(&mut rng);
        g1.push(if i % 11 == 0 { G1Affine::identity() } else { (G1Projective::generator() * coefficient).into_affine() });
        g2.push(if i % 13 == 0 { ark_bn254::G2Affine::identity() } else { (G2Projective::generator() * coefficient).into_affine() });
    }
    backend.write_fields(scalars, 0, &values).unwrap();
    backend.write_points(points, 0, &g1).unwrap(); backend.seal();
    for len in [1usize, 7, 16, 65] {
        let mut expected_g1 = G1Projective::zero(); let mut expected_g2 = G2Projective::zero();
        for i in 0..len { expected_g1 += g1[i + 3] * values[i + 7]; expected_g2 += g2[i + 3] * values[i + 7]; }
        assert_eq!(backend.msm_g1(points, 3, scalars, 7, len).unwrap(), expected_g1.into_affine());
        assert_eq!(backend.msm_g2(&g2[3..3 + len], scalars, 7).unwrap(), expected_g2.into_affine());
        assert_eq!(backend.msm_g2_host(&g2[3..3 + len], &values[7..7 + len]).unwrap(), expected_g2.into_affine());
    }
    assert_eq!(backend.msm_g2_host(&[], &[]).unwrap(), ark_bn254::G2Affine::identity());
    assert!(backend.msm_g2_host(&g2[..1], &[]).is_err());
    let bad=ark_bn254::G2Affine::new_unchecked(ark_bn254::Fq2::from(1u64),ark_bn254::Fq2::from(1u64));
    assert!(backend.msm_g2_host(&[bad], &values[..1]).is_err());
}

#[test]
#[ignore = "explicit latency probe; CPU and device-event clocks reported separately"]
fn resident_independent_ecntt_latency_profile() {
    use std::time::Instant;
    fn quantiles(mut samples: Vec<u64>) -> (u64, u64, u64) {
        samples.sort_unstable();
        (samples[samples.len() / 2], samples[(samples.len() * 99 / 100).min(samples.len() - 1)], *samples.last().unwrap())
    }
    let mut backend = IcicleBackend::with_resident_limits(512 << 20, 16 << 20).unwrap();
    let input = backend.alloc(BufferKind::G1, 4096, true).unwrap();
    let output = backend.alloc(BufferKind::G1, 4096, true).unwrap();
    backend.seal();
    for n in [256usize, 1024, 4096] {
        let points: Vec<_> = (0..n).map(|i| (G1Projective::generator() * Fr::from(i as u64 + 1)).into_affine()).collect();
        let domain = Radix2EvaluationDomain::<Fr>::new(n).unwrap();
        let mut result = vec![G1Affine::identity(); n];
        let mut cpu_ns = Vec::new(); let mut total_ns = Vec::new(); let mut gpu_ns = Vec::new();
        let mut upload_ns = Vec::new(); let mut download_ns = Vec::new();
        for iteration in 0..21 {
            let mut expected: Vec<G1Projective> = points.iter().map(|p| (*p).into()).collect();
            let start = Instant::now(); domain.fft_in_place(&mut expected); let cpu = start.elapsed().as_nanos() as u64;
            let start = Instant::now();
            backend.write_points(input, 0, &points).unwrap(); let uploaded = start.elapsed().as_nanos() as u64;
            let before = backend.resident_diagnostics().unwrap();
            backend.transform_rows(BufferKind::G1, input, output, n, 0, 1, false).unwrap();
            let after = backend.resident_diagnostics().unwrap();
            let read_start = Instant::now(); backend.read_points(output, 0, &mut result).unwrap();
            let downloaded = read_start.elapsed().as_nanos() as u64; let total = start.elapsed().as_nanos() as u64;
            assert_eq!(result, G1Projective::normalize_batch(&expected));
            if iteration > 0 {
                cpu_ns.push(cpu); total_ns.push(total); gpu_ns.push(after.device_operation_ns - before.device_operation_ns);
                upload_ns.push(uploaded); download_ns.push(downloaded);
            }
        }
        eprintln!("ECNTT_PROFILE n={n} samples=20 cpu_ns={:?} total_h2d_transform_d2h_ns={:?} gpu_event_transform_ns={:?} upload_host_ns={:?} download_decode_host_ns={:?} stats={:?} diagnostics={:?}",
            quantiles(cpu_ns), quantiles(total_ns), quantiles(gpu_ns), quantiles(upload_ns), quantiles(download_ns), backend.stats(), backend.resident_diagnostics().unwrap());
    }
}

#[test]
#[ignore = "run only with native allocation-failure LD_PRELOAD fixture"]
fn resident_fatal_cuda_error_retires_owner_after_prior_successful_step() {
    assert_eq!(std::env::var("DVGPU_TEST_RECORD_FIXTURE").unwrap(), "1");
    let mut backend = IcicleBackend::with_resident_limits(256 << 20, 1 << 20).unwrap();
    let input = backend.alloc(BufferKind::Fr, 16, true).unwrap();
    let output = backend.alloc(BufferKind::Fr, 16, true).unwrap();
    backend.write_fields(input, 0, &[Fr::from(5u64); 16]).unwrap();
    backend.seal(); backend.begin_transaction().unwrap();
    backend.write_fields(input, 3, &[Fr::from(7u64); 2]).unwrap();
    std::env::set_var("DVGPU_TEST_FAIL_RECORD_AT", "1");
    assert!(backend.transform_rows(BufferKind::Fr, input, output, 8, 0, 2, false).is_err());
    std::env::remove_var("DVGPU_TEST_FAIL_RECORD_AT");
    assert!(backend.rollback_transaction().is_err(), "lost CUDA owner cannot promise physical rollback");
    assert!(backend.write_fields(input, 0, &[Fr::from(1u64)]).is_err());
    drop(backend);
    let mut recovered = IcicleBackend::with_resident_limits(256 << 20, 1 << 20).unwrap();
    assert!(recovered.alloc(BufferKind::Fr, 8, true).is_ok(), "recovery requires a new explicit owner");
}

#[test]
fn resident_quota_rejects_before_update_and_journal_exhaustion_is_recoverable() {
    let mut backend = IcicleBackend::with_resident_limits(128 << 20, 64).unwrap();
    assert!(backend.alloc(BufferKind::G1, 1 << 21, true).is_err());
    let id = backend.alloc(BufferKind::Fr, 16, true).unwrap();
    backend.write_fields(id, 0, &[Fr::from(5u64); 16]).unwrap();
    backend.seal();
    backend.begin_transaction().unwrap();
    backend.write_fields(id, 0, &[Fr::from(9u64); 2]).unwrap();
    assert!(backend.write_fields(id, 2, &[Fr::from(9u64)]).is_err());
    backend.rollback_transaction().unwrap();
    let mut result = [Fr::zero(); 16];
    backend.read_fields(id, 0, &mut result).unwrap();
    assert_eq!(result, [Fr::from(5u64); 16]);
}

#[test]
fn foreground_gpu_calls_share_the_owner_memory_cap_and_diagnostics() {
    use dynaverse_protocol::BatchMsmRequest;
    let backend=IcicleBackend::with_resident_limits(112<<20,64).unwrap();
    let n=1<<20;
    let points=vec![G1Affine::identity();n];
    let scalars=vec![Fr::from(1u64);n];
    let sets=[points.as_slice()]; let offsets=[0]; let lengths=[n];
    let request=BatchMsmRequest {point_sets:&sets,shared_scalars:&scalars,offsets:&offsets,lengths:&lengths};
    let before=backend.batch_stats().unwrap();
    assert!(backend.msm_batch(&request).is_err(),"foreground retained arrays exceed this owner's cap");
    assert_eq!(backend.batch_stats().unwrap(),before,"quota rejection must precede batch growth");
    assert!(backend.memory_info().is_ok(),"a preflight quota refusal keeps the owner healthy");
    let mut values=vec![Fr::from(3u64);256];
    let before=backend.resident_diagnostics().unwrap();
    backend.ntt(&mut values,false).unwrap();
    let ntt=backend.resident_diagnostics().unwrap();
    assert!(ntt.temporary_allocations>before.temporary_allocations);
    assert_eq!(backend.msm(&points[..8],&scalars[..8]).unwrap(),G1Affine::identity());
    let g1=backend.resident_diagnostics().unwrap();
    assert!(g1.temporary_allocations>ntt.temporary_allocations);
    assert_eq!(backend.msm_g2_host(&[ark_bn254::G2Affine::identity();8],&scalars[..8]).unwrap(),ark_bn254::G2Affine::identity());
    let after=backend.resident_diagnostics().unwrap();
    assert!(after.temporary_allocations>g1.temporary_allocations,
        "foreground FFT/MSM scratch must participate in the same quota and counters");
    assert!(backend.stats().bytes_allocated+after.temporary_peak_bytes<=112<<20);
    let before_bytes=backend.stats().bytes_allocated;
    let short_lengths=[8];
    let request=BatchMsmRequest {lengths:&short_lengths,..request};
    assert_eq!(backend.msm_batch(&request).unwrap(),vec![G1Affine::identity()]);
    assert_eq!(backend.stats().bytes_allocated,before_bytes+8*96);
}
