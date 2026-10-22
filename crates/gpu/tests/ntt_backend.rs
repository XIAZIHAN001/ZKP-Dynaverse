#![cfg(feature = "icicle")]

use ark_bn254::{Fr, G1Projective};
use ark_ec::{CurveGroup, Group};
use ark_ff::{UniformRand, Zero};
use dynaverse_gpu::{CpuBackend, IcicleBackend};
use dynaverse_protocol::OperatorBackend;
use rand::{rngs::StdRng, SeedableRng};

#[test]
#[ignore = "run with allocation-failure fixture; proves NTT executes CUDA rather than CPU"]
fn ntt_allocation_failure_retires_owner_without_changing_input() {
    let fail_at: usize = std::env::var("DVGPU_TEST_FAIL_ALLOC_AT").unwrap().parse().unwrap();
    assert!((1..=8).contains(&fail_at));
    let input: Vec<Fr> = (0..1024).map(|i| Fr::from(i as u64 + 1)).collect();
    let owner = IcicleBackend::new().unwrap();
    let mut actual = input.clone();
    assert!(owner.ntt(&mut actual, false).is_err(), "NTT must execute CUDA and propagate its allocation failure");
    assert_eq!(actual, input, "failed NTT must preserve caller input");
    let point = G1Projective::generator().into_affine();
    assert!(owner.msm(&[point], &[Fr::from(3u64)]).is_err(), "execution failure must retire the shared owner");
    assert!(owner.ntt(&mut actual, true).is_err());
    drop(owner);
    let recovered = IcicleBackend::new().unwrap();
    let mut expected = input.clone();
    CpuBackend.ntt(&mut expected, false).unwrap();
    recovered.ntt(&mut actual, false).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(recovered.msm(&[point], &[Fr::from(3u64)]).unwrap(),
        (G1Projective::generator() * Fr::from(3u64)).into_affine());
    eprintln!("NTT_ALLOCATION_FAILURE_RECOVERY_OK");
}

#[test]
fn ntt_matches_independent_cpu_forward_and_inverse_across_sizes() {
    let baseline = IcicleBackend::new().unwrap();
    let configured = IcicleBackend::with_large_bucket_factor(10).unwrap();
    for seed in [7u64, 17, 2026] {
        let mut rng = StdRng::seed_from_u64(seed);
        // Include both algorithm branches and changing transform sizes within
        // an owner; expectations come from arkworks rather than a GPU roundtrip.
        for n in [1usize, 2, 8, 16, 128, 256, 1024, 4096, 16384] {
            let mut input: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
            input[0] = -Fr::from(1u64);
            for inverse in [false, true] {
                let mut expected = input.clone();
                CpuBackend.ntt(&mut expected, inverse).unwrap();
                for owner in [&baseline, &configured] {
                    let mut actual = input.clone();
                    owner.ntt(&mut actual, inverse).unwrap();
                    assert_eq!(actual, expected, "seed={seed} n={n} inverse={inverse}");
                }
            }
            let mut roundtrip = input.clone();
            baseline.ntt(&mut roundtrip, false).unwrap();
            baseline.ntt(&mut roundtrip, true).unwrap();
            assert_eq!(roundtrip, input);
            eprintln!("NTT_CPU_EXACT_OK seed={seed} n={n}");
        }
    }
}

#[test]
fn ntt_known_values_and_invalid_lengths_preserve_healthy_owner() {
    let owner = IcicleBackend::new().unwrap();
    let mut impulse = vec![Fr::zero(); 8];
    impulse[0] = Fr::from(1u64);
    owner.ntt(&mut impulse, false).unwrap();
    assert_eq!(impulse, vec![Fr::from(1u64); 8]);
    let mut constant = vec![Fr::from(5u64); 8];
    owner.ntt(&mut constant, false).unwrap();
    assert_eq!(constant, vec![Fr::from(40u64), Fr::zero(), Fr::zero(), Fr::zero(),
        Fr::zero(), Fr::zero(), Fr::zero(), Fr::zero()]);
    owner.ntt(&mut constant, true).unwrap();
    assert_eq!(constant, vec![Fr::from(5u64); 8]);
    for n in [0usize, 3, 17] {
        let mut invalid = vec![Fr::from(7u64); n];
        let before = invalid.clone();
        assert!(owner.ntt(&mut invalid, false).is_err());
        assert_eq!(invalid, before);
    }
    let mut healthy = vec![Fr::from(1u64); 16];
    owner.ntt(&mut healthy, false).unwrap();
    assert_eq!(healthy[0], Fr::from(16u64));
    assert!(healthy[1..].iter().all(|value| value.is_zero()));
}

#[test]
fn ntt_dropping_one_owner_does_not_invalidate_another_domain() {
    let survivor = IcicleBackend::new().unwrap();
    let mut values = vec![Fr::from(3u64); 1024];
    survivor.ntt(&mut values, false).unwrap();
    for _ in 0..3 {
        let temporary = IcicleBackend::new().unwrap();
        let mut independent = vec![Fr::from(2u64); 256];
        temporary.ntt(&mut independent, false).unwrap();
        assert_eq!(independent[0], Fr::from(512u64));
        drop(temporary);
        survivor.ntt(&mut values, true).unwrap();
        assert_eq!(values, vec![Fr::from(3u64); 1024]);
        survivor.ntt(&mut values, false).unwrap();
    }
}

#[test]
fn ntt_parallel_owners_can_initialize_and_drop_domains_independently() {
    use std::sync::{Arc, Barrier};
    let barrier = Arc::new(Barrier::new(3));
    let mut threads = Vec::new();
    for worker in 0..3u64 {
        let barrier = barrier.clone();
        threads.push(std::thread::spawn(move || {
            for iteration in 0..4 {
                // Construct each owner on its thread; never send a native
                // handle. Overlap domain initialization with other GPU work.
                let owner = IcicleBackend::new().unwrap();
                let value = Fr::from(worker + iteration + 1);
                let mut values = vec![value; 1024];
                barrier.wait();
                owner.ntt(&mut values, false).unwrap();
                assert_eq!(values[0], value * Fr::from(1024u64));
                assert!(values[1..].iter().all(|value| value.is_zero()));
                owner.ntt(&mut values, true).unwrap();
                assert_eq!(values, vec![value; 1024]);
                drop(owner);
                barrier.wait();
            }
        }));
    }
    for thread in threads { thread.join().unwrap(); }
}

#[test]
#[ignore = "explicit full-scale CPU/GPU NTT gate through 2^20"]
fn ntt_full_scale_matches_cpu_in_both_directions() {
    let owner = IcicleBackend::new().unwrap();
    let mut rng = StdRng::seed_from_u64(20420);
    for log_n in 10..=20 {
        let n = 1usize << log_n;
        let input: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        for inverse in [false, true] {
            let mut actual = input.clone();
            let mut expected = input.clone();
            CpuBackend.ntt(&mut expected, inverse).unwrap();
            owner.ntt(&mut actual, inverse).unwrap();
            assert_eq!(actual, expected, "n={n} inverse={inverse}");
        }
        eprintln!("NTT_FULL_SCALE_EXACT_OK n={n}");
    }
}
