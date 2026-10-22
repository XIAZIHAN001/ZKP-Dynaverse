#![cfg(feature = "icicle")]
use ark_bn254::{Fr, G1Projective, G2Projective};
use ark_ec::{CurveGroup, Group};
use ark_ff::{One, UniformRand, Zero};
use dynaverse_gpu::{IcicleBackend, SetupMode};
use dynaverse_protocol::OperatorBackend;
use rand::{rngs::StdRng, SeedableRng};

#[test]
fn setup_modes_match_oracle_reuse_public_tables_and_profile() {
    let mut rng = StdRng::seed_from_u64(17003);
    let mut scalars = vec![Fr::zero(), Fr::one(), -Fr::one()];
    for _ in 0..4094 { scalars.push(Fr::rand(&mut rng)); }
    let expected_g1: Vec<_> = scalars.iter().map(|s|(G1Projective::generator()*s).into_affine()).collect();
    let expected_g2: Vec<_> = scalars.iter().map(|s|(G2Projective::generator()*s).into_affine()).collect();
    for mode in [SetupMode::Legacy, SetupMode::Table, SetupMode::TableBatchNormalize] {
        let gpu = IcicleBackend::new().unwrap();
        gpu.set_setup_mode(mode).unwrap();
        assert!(gpu.fixed_base_g1(&[]).unwrap().is_empty());
        assert_eq!(gpu.setup_stats().unwrap().table_builds, [0,0]);
        for length in [1, 127, 128, 129, 4097] {
            assert_eq!(gpu.fixed_base_g1(&scalars[..length]).unwrap(), expected_g1[..length]);
            assert_eq!(gpu.fixed_base_g2(&scalars[..length]).unwrap(), expected_g2[..length]);
        }
        let stats = gpu.setup_stats().unwrap();
        assert_eq!(stats.table_builds, if mode == SetupMode::Legacy {[0,0]} else {[1,1]});
        assert_eq!(stats.points, [4482,4482]);
        assert_eq!(stats.calls, [6,6]);
        assert!(stats.native_wall_ns > 0 && stats.device_ns[2] > 0);
        assert!(gpu.setup_host_stats().validate_ns > 0);
        if mode != SetupMode::Legacy { assert_eq!(stats.table_bytes, 294912); }
    }
}

#[test]
fn setup_table_bytes_share_resident_quota_with_batch_reservation() {
    use dynaverse_protocol::incremental_rebuild::operators::RebuildBackend;
    let mut gpu = IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap();
    let base=gpu.stats().bytes_allocated;
    gpu.fixed_base_g1(&[Fr::one()]).unwrap();
    let table = gpu.setup_stats().unwrap().table_bytes;
    assert_eq!(gpu.stats().bytes_allocated, base+table);
    gpu.reserve_sparse_msm(4,1024).unwrap();
    let after_batch = gpu.stats().bytes_allocated;
    assert!(after_batch >= table+1024*96);
    gpu.fixed_base_g2(&[Fr::zero()]).unwrap();
    assert_eq!(gpu.stats().bytes_allocated, after_batch+196608);
    let fresh = IcicleBackend::new().unwrap();
    assert_eq!(fresh.setup_stats().unwrap().table_builds, [0,0]);
}

#[test]
fn setup_table_quota_refusal_does_not_allocate_or_retire_owner() {
    // Matches the fixed 2^20-domain reservation in icicle_resident.cu.
    let domain_bytes=((1u64<<20)+1+6+64)*32;
    let gpu=IcicleBackend::with_resident_limits(domain_bytes+(64<<20)+1024,0).unwrap();
    let before=gpu.setup_stats().unwrap();
    assert!(gpu.fixed_base_g1(&[Fr::one()]).unwrap_err().to_string().contains("quota"));
    assert_eq!(gpu.setup_stats().unwrap().table_bytes,before.table_bytes);
    assert_eq!(gpu.setup_point_counts(),(0,0));
    assert!(gpu.memory_info().is_ok());
    gpu.set_setup_mode(SetupMode::Legacy).unwrap();
    assert_eq!(gpu.fixed_base_g1(&[Fr::one()]).unwrap()[0],G1Projective::generator().into_affine());
}

#[test]
fn resident_configuration_after_setup_keeps_table_accounted() {
    use dynaverse_protocol::incremental_rebuild::operators::{RebuildBackend,BufferKind};
    let mut gpu=IcicleBackend::new().unwrap();
    gpu.fixed_base_g1(&[Fr::one()]).unwrap();
    let table=gpu.setup_stats().unwrap().table_bytes;
    gpu.alloc(BufferKind::Fr,1,true).unwrap();
    assert!(gpu.stats().bytes_allocated>=table+((1u64<<20)+71)*32+(64<<20));
}

#[test]
fn setup_batch_normalization_accepts_all_identity_and_repeated_points() {
    let gpu=IcicleBackend::new().unwrap();
    for scalar in [Fr::zero(),Fr::one()] {
        let values=vec![scalar;129];
        assert_eq!(gpu.fixed_base_g1(&values).unwrap(),vec![(G1Projective::generator()*scalar).into_affine();129]);
        assert_eq!(gpu.fixed_base_g2(&values).unwrap(),vec![(G2Projective::generator()*scalar).into_affine();129]);
    }
}
