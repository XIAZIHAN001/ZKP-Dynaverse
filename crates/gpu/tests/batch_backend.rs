#![cfg(feature = "icicle")]
use ark_bn254::{Fq, Fr, G1Affine, G1Projective};
use ark_ec::{CurveGroup, Group, VariableBaseMSM};
use ark_ff::{UniformRand, Zero};
use dynaverse_gpu::IcicleBackend;
use dynaverse_protocol::{BatchMsmRequest, OperatorBackend};
use rand::{rngs::StdRng, SeedableRng};

#[test]
fn real_icicle_batch_preserves_order_and_reuses_input_buffers() {
    let owner = IcicleBackend::with_large_bucket_factor(10).unwrap();
    let mut rng = StdRng::seed_from_u64(20602);
    let mut points = Vec::new();
    for _ in 0..64 { points.push(G1Projective::rand(&mut rng).into_affine()); }
    let scalars: Vec<_> = (0..128).map(|_| Fr::rand(&mut rng)).collect();
    let sets: [&[G1Affine]; 3] = [&points, &points, &points];
    let request = BatchMsmRequest { point_sets: &sets, shared_scalars: &scalars,
        offsets: &[7, 0, 80], lengths: &[64, 53, 40] };
    let expected: Vec<_> = (0..3).map(|i| G1Projective::msm_unchecked(
        &points[..request.lengths[i]], &scalars[request.offsets[i]..request.offsets[i] + request.lengths[i]])
        .into_affine()).collect();
    assert_eq!(owner.msm_batch(&request).unwrap(), expected);
    let first = owner.batch_stats().unwrap();
    assert_eq!(first.max_batch_size, 3, "must execute ICICLE with batch_size > 1");
    assert_eq!(first.icicle_calls, 1);
    assert_eq!(first.buffer_grows, 1);
    for _ in 0..4 { assert_eq!(owner.msm_batch(&request).unwrap(), expected); }
    let reused = owner.batch_stats().unwrap();
    assert_eq!(reused.buffer_grows, first.buffer_grows);
    assert_eq!(reused.input_capacity, first.input_capacity);
    assert_eq!(reused.icicle_calls, 5);
    // Interleave NTT without invalidating the cached MSM inputs.
    let mut transform = vec![Fr::from(3u64); 256];
    owner.ntt(&mut transform, false).unwrap();
    assert_eq!(owner.msm_batch(&request).unwrap(), expected);
    assert_eq!(owner.batch_stats().unwrap().buffer_grows, 1);
}

#[test]
fn batch_handles_heterogeneous_empty_shared_offsets_and_zero_after_dense() {
    let owner = IcicleBackend::new().unwrap();
    let generator = G1Projective::generator();
    let points = vec![generator.into_affine(); 129];
    let sets: [&[G1Affine]; 6] = [&points, &points, &[], &points, &points, &points];
    let scalars: Vec<_> = (0..140).map(|i| Fr::from(i as u64 + 1)).collect();
    let request = BatchMsmRequest { point_sets: &sets, shared_scalars: &scalars,
        offsets: &[0, 5, 140, 0, 9, 0], lengths: &[1, 17, 0, 128, 7, 129] };
    let actual = owner.msm_batch(&request).unwrap();
    for i in 0..sets.len() {
        let end = request.offsets[i] + request.lengths[i];
        let sum: Fr = scalars[request.offsets[i]..end].iter().copied().sum();
        assert_eq!(actual[i], (generator * sum).into_affine());
    }
    let zeros = vec![Fr::zero(); scalars.len()];
    let zero_request = BatchMsmRequest { shared_scalars: &zeros, ..request };
    assert_eq!(owner.msm_batch(&zero_request).unwrap(), vec![G1Affine::identity(); 6]);
    let invalid = G1Affine::new_unchecked(Fq::from(1u64), Fq::from(1u64));
    let bad_sets: [&[G1Affine]; 2] = [&points, std::slice::from_ref(&invalid)];
    let before = owner.batch_stats().unwrap();
    assert!(owner.msm_batch(&BatchMsmRequest { point_sets: &bad_sets, shared_scalars: &scalars,
        offsets: &[0, 0], lengths: &[17, 1] }).is_err());
    assert_eq!(owner.batch_stats().unwrap(), before, "validate every point before first native group");
    assert!(owner.msm_batch(&BatchMsmRequest { point_sets: &sets, shared_scalars: &scalars,
        offsets: &[usize::MAX; 6], lengths: request.lengths }).is_err());
    assert_eq!(owner.batch_stats().unwrap(), before);
}

#[test]
fn batches_split_at_work_bounds_and_reuse_grown_capacity() {
    let owner = IcicleBackend::new().unwrap();
    let generator = G1Projective::generator();
    let points = vec![generator.into_affine(); 1024];
    let scalars = vec![Fr::from(11u64); 1024];
    let sets: Vec<&[G1Affine]> = vec![&points; 70];
    let offsets = vec![0; 70];
    let lengths = vec![32; 70];
    let request = BatchMsmRequest { point_sets: &sets, shared_scalars: &scalars,
        offsets: &offsets, lengths: &lengths };
    assert_eq!(owner.msm_batch(&request).unwrap(), vec![(generator * Fr::from(352u64)).into_affine(); 70]);
    let first = owner.batch_stats().unwrap();
    assert_eq!(first.icicle_calls, 2);
    assert_eq!(first.max_batch_size, 64);
    assert_eq!(first.buffer_grows, 1);
    let larger = BatchMsmRequest { point_sets: &sets[..3], shared_scalars: &scalars,
        offsets: &[0; 3], lengths: &[1024; 3] };
    assert_eq!(owner.msm_batch(&larger).unwrap(), vec![(generator * Fr::from(11264u64)).into_affine(); 3]);
    let grown = owner.batch_stats().unwrap();
    assert_eq!(grown.buffer_grows, 2);
    assert!(grown.input_capacity > first.input_capacity);
    assert_eq!(owner.msm_batch(&request).unwrap().len(), 70);
    assert_eq!(owner.batch_stats().unwrap().buffer_grows, 2);
}

#[test]
#[ignore = "explicit full-scale batch equality and inclusive performance experiment"]
fn batch_cross_scale_equality_and_inclusive_timing() {
    use std::{fs::File, io::Write, time::Instant};
    let destination = std::env::var("DVGPU_BATCH_CSV").expect("set explicit output path");
    let mut output = File::create(destination).unwrap();
    writeln!(output, "length,count,backend,phase,sample,order,elapsed_us,cpu_match,buffer_grows,input_capacity,icicle_calls").unwrap();
    let owner = IcicleBackend::with_large_bucket_factor(10).unwrap();
    let mut rng = StdRng::seed_from_u64(20603);
    for length in [16, 1024, 8192, 65536, 262144, 524288, 1048576] {
        let generator = G1Projective::generator();
        let mut next = generator;
        let mut projective = Vec::with_capacity(length);
        for _ in 0..length { projective.push(next); next += generator; }
        let points = G1Projective::normalize_batch(&projective);
        let scalars: Vec<_> = (0..3 * length).map(|_| Fr::rand(&mut rng)).collect();
        let sets: [&[G1Affine]; 3] = [&points; 3];
        let offsets = [0, length, 2 * length];
        let lengths = [length; 3];
        let request = BatchMsmRequest { point_sets: &sets, shared_scalars: &scalars,
            offsets: &offsets, lengths: &lengths };
        let expected: Vec<_> = offsets.iter().map(|&offset| G1Projective::msm_unchecked(
            &points, &scalars[offset..offset + length]).into_affine()).collect();
        for sample in 0..=5 {
            for order in 0..2 {
                let batch = (sample + order) % 2 == 0;
                let started = Instant::now();
                let actual = if batch { owner.msm_batch(&request).unwrap() }
                else { offsets.iter().map(|&offset| owner.msm(&points,
                    &scalars[offset..offset + length]).unwrap()).collect() };
                let micros = started.elapsed().as_secs_f64() * 1e6;
                assert_eq!(actual, expected, "length={length} batch={batch}");
                let stats = owner.batch_stats().unwrap();
                assert!(stats.input_capacity <= (1 << 20));
                writeln!(output, "{length},3,{},{},{sample},{order},{micros:.3},true,{},{},{}",
                    if batch { "batch" } else { "serial" },
                    if sample == 0 { "first_at_scale" } else { "sample" },
                    stats.buffer_grows, stats.input_capacity, stats.icicle_calls).unwrap();
            }
        }
        output.flush().unwrap();
        eprintln!("BATCH_SCALE_CPU_EXACT length={length}");
    }
}

#[test]
#[ignore = "requires real CUDA allocation failure fixture"]
fn batch_failure_retires_only_its_owner_and_returns_no_partial_results() {
    let owner = IcicleBackend::new().unwrap();
    let survivor = IcicleBackend::new().unwrap();
    let generator = G1Projective::generator();
    let points = vec![generator.into_affine(); 32];
    let scalars = vec![Fr::from(7u64); 64];
    let sets: [&[G1Affine]; 2] = [&points, &points];
    let request = BatchMsmRequest { point_sets: &sets, shared_scalars: &scalars,
        offsets: &[0, 16], lengths: &[32, 32] };
    assert!(owner.msm_batch(&request).is_err());
    assert!(owner.msm_batch(&request).unwrap_err().to_string().contains("retired"));
    assert!(owner.msm(&points, &scalars[..32]).is_err());
    drop(owner);
    let actual = survivor.msm_batch(&request).unwrap();
    assert_eq!(actual, vec![(generator * Fr::from(224u64)).into_affine(); 2]);
    eprintln!("BATCH_FAILURE_ISOLATION_RECOVERY_OK");
}
