use super::*;
use crate::{IcicleBackend,SetupMode};
use ark_ec::CurveGroup;
use ark_ff::{One,UniformRand};
use rand::{rngs::StdRng,SeedableRng};
use dynaverse_protocol::OperatorBackend;
extern "C" {fn dvgpu_g2_test_corrupt_next(owner:*mut std::ffi::c_void,index:u64)->i32;}

#[test]
fn all_tuning_configs_preserve_acceptance_and_fused_rejection() {
    let gpu=IcicleBackend::new().unwrap();gpu.set_setup_mode(SetupMode::TableBatchNormalize).unwrap();
    let mut bytes=Vec::new();for point in super::tests::differential_points() {bytes.extend_from_slice(&super::tests::wire(*point));}
    for point in super::tests::acceptance_corpus() {bytes.extend_from_slice(&point);}
    let (expected,_)=profile_cpu_full_order(&bytes).unwrap();
    let mut rng=StdRng::seed_from_u64(1750002);let mut scalars=Vec::new();
    for _ in 0..129 {scalars.push(Fr::rand(&mut rng));}scalars[0]=Fr::zero();scalars[1]=-Fr::one();scalars[2]=Fr::one();
    let points:Vec<_>=scalars.iter().map(|s|(G2Projective::generator()*s).into_affine()).collect();
    for threads in [64,128,256] {for schedule in [0,1] {for layout in [0,1] {for fused in [0,1] {
        let config=G2KernelConfig{threads,schedule,layout,fused};gpu.set_g2_kernel_config(config).unwrap();
        assert_eq!(gpu.g2_kernel_config().unwrap(),config);
        for algorithm in [1,2,3] {
            let (masks,report)=gpu.inspect_g2_validation_batched(&bytes,algorithm).unwrap();
            assert_eq!(masks,expected,"config={config:?} algorithm={algorithm}");
            assert_eq!(report.checked_points,expected.len() as u64);
            assert_eq!(report.invalid_points,expected.iter().filter(|&&m|m!=0).count() as u64);
            assert_eq!(report.first_invalid,expected.iter().position(|&m|m!=0).unwrap() as u64);
            let resources=gpu.g2_kernel_resources(algorithm).unwrap();
            assert!(resources.registers>0 && resources.active_blocks>0 && resources.max_threads_per_sm>=threads);
            let mode=match algorithm {1=>G2ValidationMode::GpuFullOrder,2=>G2ValidationMode::GpuFrobenius127,_=>G2ValidationMode::GpuFrobenius63};
            gpu.set_g2_validation_mode(mode).unwrap();assert_eq!(gpu.fixed_base_g2(&scalars).unwrap(),points);
            for bad in [0,128] {
                let accepted=gpu.g2_validation_stats().unwrap().accepted_points;
                // SAFETY: explicit bounded test corruption before both fused and separate curve checks.
                assert_eq!(unsafe {dvgpu_g2_test_corrupt_next(gpu.handle.as_ptr(),bad)},0);
                assert!(gpu.fixed_base_g2(&scalars).is_err());
                assert_eq!(gpu.g2_validation_stats().unwrap().accepted_points,accepted);
            }
        }
    }}}}
}

#[test]
fn fused_compact_soa_setup_preserves_full_proof_and_all_normalizers() {
    use dynaverse_protocol::{dynaverse::DynaverseParams,toy_circuit::ToyCircuit};
    let gpu=IcicleBackend::new().unwrap();
    gpu.set_g2_validation_mode(G2ValidationMode::GpuFrobenius63).unwrap();
    gpu.set_g2_kernel_config(G2KernelConfig{threads:64,schedule:1,layout:1,fused:1}).unwrap();
    for setup_mode in [SetupMode::Legacy,SetupMode::Table,SetupMode::TableBatchNormalize] {
        gpu.set_setup_mode(setup_mode).unwrap();
        let mut scalars=Vec::new();for i in 0..129 {scalars.push(Fr::from(i as u64));}
        let actual=gpu.fixed_base_g2(&scalars).unwrap();
        for i in 0..129 {assert_eq!(actual[i],(G2Projective::generator()*scalars[i]).into_affine());}
    }
    gpu.set_setup_mode(SetupMode::TableBatchNormalize).unwrap();
    let circuit=ToyCircuit::new(16).unwrap();let (instance,witness)=circuit.valid_instance();
    let cpu=DynaverseParams::setup(circuit.clone(),&mut StdRng::seed_from_u64(1750016)).unwrap();
    let params=DynaverseParams::setup_with_backend(circuit,&mut StdRng::seed_from_u64(1750016),&gpu).unwrap();
    let proof=params.prove_with_aux(&instance,&witness).unwrap().0;
    assert_eq!(proof,cpu.prove_with_aux(&instance,&witness).unwrap().0);
    assert!(cpu.verifying_key().verify_reference(&instance,&proof).unwrap());
}

#[test]
fn invalid_tuning_config_preserves_previous_selection() {
    let gpu=IcicleBackend::new().unwrap();let before=gpu.g2_kernel_config().unwrap();
    for bad in [G2KernelConfig{threads:0,..before},G2KernelConfig{threads:32,..before},
        G2KernelConfig{schedule:2,..before},G2KernelConfig{layout:2,..before},G2KernelConfig{fused:2,..before}] {
        assert!(gpu.set_g2_kernel_config(bad).is_err());assert_eq!(gpu.g2_kernel_config().unwrap(),before);
    }
    assert!(gpu.memory_info().is_ok());
}

#[test]
fn threaded_cpu_wire_baseline_matches_full_order_and_legacy() {
    let mut bytes=Vec::new();for point in super::tests::acceptance_corpus() {bytes.extend_from_slice(&point);}
    let (expected,_)=profile_cpu_full_order(&bytes).unwrap();
    for mode in [G2ValidationMode::CpuReference,G2ValidationMode::CpuFrobenius127,G2ValidationMode::CpuFrobenius63] {
        for threads in [1,8] {assert_eq!(inspect_g2_cpu(&bytes,mode,threads).unwrap(),expected);}
    }
    assert!(inspect_g2_cpu(&bytes,G2ValidationMode::GpuFullOrder,1).is_err());
}
