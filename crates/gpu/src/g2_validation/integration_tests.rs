use super::*;
use crate::{IcicleBackend,SetupMode};
use ark_ec::CurveGroup;
use ark_ff::One;
use dynaverse_protocol::{OperatorBackend,dynaverse::DynaverseParams,toy_circuit::ToyCircuit};
use rand::{rngs::StdRng,SeedableRng};

const MODES:[G2ValidationMode;6]=[G2ValidationMode::CpuReference,G2ValidationMode::CpuFrobenius127,
    G2ValidationMode::CpuFrobenius63,G2ValidationMode::GpuFullOrder,
    G2ValidationMode::GpuFrobenius127,G2ValidationMode::GpuFrobenius63];
extern "C" {fn dvgpu_g2_test_corrupt_next(owner:*mut std::ffi::c_void,index:u64)->i32;}
fn inject(gpu:&IcicleBackend,index:usize) {
    // SAFETY: explicit test-only ABI, one bounded index; no environment switch.
    assert_eq!(unsafe {dvgpu_g2_test_corrupt_next(gpu.handle.as_ptr(),index as u64)},0);
}

#[test]
fn checked_setup_matches_cpu_points_and_full_proof() {
    for n in [1,2,16,64] {
        let circuit=ToyCircuit::new(n).unwrap();let (instance,witness)=circuit.valid_instance();
        let cpu=DynaverseParams::setup(circuit.clone(),&mut StdRng::seed_from_u64(17400+n as u64)).unwrap();
        let expected=cpu.prove_with_aux(&instance,&witness).unwrap().0;
        for mode in MODES {
            let gpu=IcicleBackend::new().unwrap();gpu.set_g2_validation_mode(mode).unwrap();
            let params=DynaverseParams::setup_with_backend(circuit.clone(),&mut StdRng::seed_from_u64(17400+n as u64),&gpu).unwrap();
            let proof=params.prove_with_aux(&instance,&witness).unwrap().0;
            assert_eq!(proof,expected,"n={n} mode={mode:?}");
            assert!(cpu.verifying_key().verify_reference(&instance,&proof).unwrap());
            let scalars=[Fr::zero(),Fr::one(),-Fr::one(),Fr::from(47)];
            let actual=gpu.fixed_base_g2(&scalars).unwrap();
            for i in 0..scalars.len() {assert_eq!(actual[i],(G2Projective::generator()*scalars[i]).into_affine());}
        }
    }
}

#[test]
fn cpu_setup_modes_preserve_legacy_acceptance_for_bad_wire() {
    let gpu=IcicleBackend::new().unwrap();
    for bytes in super::tests::acceptance_corpus() {
        for mode in &MODES[..3] {
            let expected=decode_g2_reference(&bytes);let actual=gpu.cpu_g2_chunk(&bytes,*mode,0);
            assert_eq!(actual.is_ok(),expected.is_ok());if let Ok(point)=expected {assert_eq!(actual.unwrap(),[point]);}
        }
    }
}

#[test]
fn checked_setup_uses_existing_affine_buffer_without_point_reupload() {
    let gpu=IcicleBackend::new().unwrap();assert_eq!(gpu.g2_validation_mode(),G2ValidationMode::CpuReference);
    gpu.set_g2_validation_mode(G2ValidationMode::GpuFrobenius63).unwrap();
    for n in [0,1,127,128,129,4097] {
        let before=gpu.g2_validation_traffic().unwrap();let stats=gpu.g2_validation_stats().unwrap();
        let scalars=vec![Fr::from(13);n];let points=gpu.fixed_base_g2(&scalars).unwrap();
        assert_eq!(points,vec![(G2Projective::generator()*Fr::from(13)).into_affine();n]);
        let after=gpu.g2_validation_traffic().unwrap();let done=gpu.g2_validation_stats().unwrap();
        assert_eq!(after.point_h2d_bytes-before.point_h2d_bytes,0);
        assert_eq!(after.scalar_h2d_bytes-before.scalar_h2d_bytes,n as u64*32);
        assert_eq!(after.point_d2h_bytes-before.point_d2h_bytes,n as u64*128);
        assert_eq!(after.status_d2h_bytes-before.status_d2h_bytes,((n+4095)/4096) as u64*32);
        assert_eq!(done.accepted_points-stats.accepted_points,n as u64);
    }
    assert_eq!(gpu.setup_stats().unwrap().table_builds,[0,1]);
}

#[test]
fn invalid_g2_chunk_never_publishes_partial_srs_and_can_retry() {
    for mode in [G2ValidationMode::GpuFullOrder,G2ValidationMode::GpuFrobenius127,G2ValidationMode::GpuFrobenius63] {
        for (n,bad) in [(129,0),(129,127),(129,128),(4096,4095)] {
            let gpu=IcicleBackend::new().unwrap();gpu.set_g2_validation_mode(mode).unwrap();
            let counts=gpu.setup_point_counts();let before=gpu.g2_validation_stats().unwrap();
            inject(&gpu,bad);let error=gpu.fixed_base_g2(&vec![Fr::one();n]).unwrap_err().to_string();
            assert!(error.contains(&format!("index {bad}")),"{error}");
            assert_eq!(gpu.setup_point_counts(),counts);
            assert_eq!(gpu.g2_validation_stats().unwrap().accepted_points,before.accepted_points);
            assert_eq!(gpu.setup_stats().unwrap().table_bytes,0,"rejected cold call retained a table");
            assert!(gpu.memory_info().is_ok());assert_eq!(gpu.fixed_base_g2(&[Fr::one()]).unwrap(),[G2Affine::generator()]);
        }
    }
}

#[test]
fn native_content_rejection_preserves_output_and_rust_reports_global_index() {
    let gpu=IcicleBackend::new().unwrap();let mut output=vec![0x55u8;129*128];let mut error=[0u8;256];
    let input=crate::codec::encode_fr(&Fr::one()).repeat(129);let mut report=G2ValidationReport::default();inject(&gpu,127);
    // SAFETY: explicit bounded test ABI produces content failure in a staged
    // output; only the rejection report is allowed to be published.
    let status=unsafe {dvgpu_bn254_fixed_base_g2_checked(gpu.handle.as_ptr(),input.as_ptr(),129,3,
        output.as_mut_ptr(),&mut report,error.as_mut_ptr().cast(),error.len())};
    assert_eq!(status,-5);assert_eq!(report.invalid_points,1);assert_eq!(report.first_invalid,127);
    assert!(output.iter().all(|&b|b==0x55));assert_eq!(gpu.g2_validation_stats().unwrap().accepted_points,0);
    inject(&gpu,0);assert!(gpu.checked_g2_chunk(&[Fr::one()],3,4096).unwrap_err().to_string().contains("index 4096"));
    assert!(gpu.memory_info().is_ok());
}

#[test]
fn validated_chunk_rejects_wrong_report_or_mutated_data_source() {
    let bytes=super::tests::wire(G2Affine::generator()).to_vec();
    let report=G2ValidationReport{checked_points:1,invalid_points:0,first_invalid:u64::MAX,failure_mask:0,algorithm:3};
    for changed in [G2ValidationReport{algorithm:2,..report},G2ValidationReport{checked_points:2,..report},
        G2ValidationReport{invalid_points:1,..report},G2ValidationReport{failure_mask:4,..report},G2ValidationReport{first_invalid:0,..report}] {
        assert!(ValidatedG2Chunk::new(bytes.clone(),changed,3,1).is_err());
    }
    assert!(ValidatedG2Chunk::new(bytes[..127].to_vec(),report,3,1).is_err());
    let mut source=bytes.clone();let certificate=ValidatedG2Chunk::new(source.clone(),report,3,1).unwrap();source.fill(0xff);
    assert_eq!(certificate.decode_validated().unwrap(),[G2Affine::generator()]);
}

#[test]
fn checked_setup_quota_and_independent_owners_do_not_grow_on_rejection() {
    use dynaverse_protocol::incremental_rebuild::operators::RebuildBackend;
    let domain=((1u64<<20)+71)*32;
    let limited=IcicleBackend::with_resident_limits(domain+(64<<20)+256,0).unwrap();
    limited.set_setup_mode(SetupMode::Legacy).unwrap();limited.set_g2_validation_mode(G2ValidationMode::GpuFrobenius63).unwrap();
    let before=limited.stats().bytes_allocated;
    assert!(limited.fixed_base_g2(&[Fr::one()]).unwrap_err().to_string().contains("quota"));
    assert_eq!(limited.stats().bytes_allocated,before);assert!(limited.memory_info().is_ok());
    let gpu=IcicleBackend::with_resident_limits(1<<30,0).unwrap();gpu.set_g2_validation_mode(G2ValidationMode::GpuFrobenius63).unwrap();
    gpu.fixed_base_g2(&[Fr::one()]).unwrap();let before=gpu.stats().bytes_allocated;
    for _ in 0..3 {inject(&gpu,0);assert!(gpu.fixed_base_g2(&[Fr::one()]).is_err());assert_eq!(gpu.stats().bytes_allocated,before);}
    assert_eq!(gpu.fixed_base_g2(&[Fr::one()]).unwrap(),[G2Affine::generator()]);
    assert_eq!(limited.g2_validation_stats().unwrap().accepted_points,0);
}

#[test]
#[ignore="explicit CUDA allocation/event/stream failure LD_PRELOAD fixture"]
fn checked_setup_cuda_failure_is_all_or_error() {
    assert!(std::env::var_os("DVGPU_TEST_FAIL_ALLOC_AT").is_some()
        || std::env::var_os("DVGPU_TEST_FAIL_EVENT_DESTROY_AT").is_some()
        || std::env::var_os("DVGPU_TEST_FAIL_STREAM_DESTROY_AT").is_some());
    let gpu=IcicleBackend::new().unwrap();gpu.set_g2_validation_mode(G2ValidationMode::GpuFrobenius63).unwrap();
    let input=crate::codec::encode_fr(&Fr::one());let mut output=[0x55u8;128];let mut error=[0u8;256];
    let original=G2ValidationReport{checked_points:91,invalid_points:92,first_invalid:93,failure_mask:94,algorithm:95};
    let mut report=original;
    // SAFETY: bounded canonical input and exact-sized sentinel outputs; injected
    // CUDA failure must not publish either output, and native catches exceptions.
    let status=unsafe {dvgpu_bn254_fixed_base_g2_checked(gpu.handle.as_ptr(),input.as_ptr(),1,3,
        output.as_mut_ptr(),&mut report,error.as_mut_ptr().cast(),error.len())};
    assert_eq!(status,-2);assert_eq!(output,[0x55;128]);assert_eq!(report,original);
    assert_eq!(gpu.g2_validation_stats().unwrap().accepted_points,0);
    assert_eq!(gpu.setup_point_counts(),(0,0));assert!(gpu.memory_info().is_err());
    assert!(gpu.set_g2_validation_mode(G2ValidationMode::CpuReference).is_err());
    assert_eq!(gpu.g2_validation_mode(),G2ValidationMode::GpuFrobenius63);
    let fresh=IcicleBackend::new().unwrap();fresh.set_g2_validation_mode(G2ValidationMode::GpuFrobenius63).unwrap();
    assert_eq!(fresh.fixed_base_g2(&[Fr::one()]).unwrap(),[G2Affine::generator()]);
}
