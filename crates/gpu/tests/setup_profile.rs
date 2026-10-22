#![cfg(feature="icicle")]
//! Explicit bounded diagnostic run, separate from correctness tests.
use ark_bn254::Fr;
use ark_ff::UniformRand;
use dynaverse_gpu::{CpuBackend,FixedBaseCpuBackend,IcicleBackend,SetupMode};
use dynaverse_protocol::{OperatorBackend,dynaverse::DynaverseParams,toy_circuit::ToyCircuit,trace::{CpuPhase,CpuTrace}};
use rand::{rngs::StdRng,SeedableRng};
use std::time::Instant;

#[test]
#[ignore="explicit local fixed-base ablation and setup phase profile"]
fn setup_ablation_profile() {
    let mut rng=StdRng::seed_from_u64(17021);
    let scalars:Vec<Fr>=(0..4096).map(|_|Fr::rand(&mut rng)).collect();
    let started=Instant::now();
    let oracle1=CpuBackend.fixed_base_g1(&scalars).unwrap();
    let oracle2=CpuBackend.fixed_base_g2(&scalars).unwrap();
    println!("OP_CSV,cpu_naive,0,4096,{},0,0,0,0,0,0,0,0",started.elapsed().as_nanos());
    for repeat in 1..4 {
        let started=Instant::now();
        let a=CpuBackend.fixed_base_g1(&scalars).unwrap();let b=CpuBackend.fixed_base_g2(&scalars).unwrap();
        let wall=started.elapsed().as_nanos(); assert_eq!(a,oracle1);assert_eq!(b,oracle2);
        println!("OP_CSV,cpu_naive,{repeat},4096,{wall},0,0,0,0,0,0,0,0");
    }
    let cpu=FixedBaseCpuBackend::default();
    for repeat in 0..4 {
        let started=Instant::now();
        let a=cpu.fixed_base_g1(&scalars).unwrap();let b=cpu.fixed_base_g2(&scalars).unwrap();
        let wall=started.elapsed().as_nanos();
        assert_eq!(a,oracle1);assert_eq!(b,oracle2);
        println!("OP_CSV,cpu_table,{repeat},4096,{wall},0,0,0,0,0,0,0,0");
    }
    for mode in [SetupMode::Legacy,SetupMode::Table,SetupMode::TableBatchNormalize] {
        let gpu=IcicleBackend::new().unwrap();gpu.set_setup_mode(mode).unwrap();
        println!("DEVICE,{}",gpu.device_identity().unwrap());
        for repeat in 0..4 {
            let before=gpu.setup_stats().unwrap();let host_before=gpu.setup_host_stats();
            let started=Instant::now();
            let a=gpu.fixed_base_g1(&scalars).unwrap();let b=gpu.fixed_base_g2(&scalars).unwrap();
            let wall=started.elapsed().as_nanos();
            assert_eq!(a,oracle1);assert_eq!(b,oracle2);
            let after=gpu.setup_stats().unwrap();let host_after=gpu.setup_host_stats();
            let mut device=[0;5];for j in 0..5 {device[j]=after.device_ns[j]-before.device_ns[j];}
            println!("OP_CSV,{mode:?},{repeat},4096,{wall},{},{},{},{},{},{},{},{}",
                after.native_wall_ns-before.native_wall_ns,host_after.encode_ns-host_before.encode_ns,
                host_after.validate_ns-host_before.validate_ns,device[0],device[1],device[2],device[3],device[4]);
            println!("VALIDATE_CSV,{mode:?},{repeat},{},{}",host_after.g1_validate_ns-host_before.g1_validate_ns,
                host_after.g2_validate_ns-host_before.g2_validate_ns);
        }
    }
    for n in [64,256] {
        for mode in [None,Some(SetupMode::Legacy),Some(SetupMode::Table),Some(SetupMode::TableBatchNormalize)] {
            let cpu=FixedBaseCpuBackend::default();
            let gpu=if let Some(mode)=mode {let g=IcicleBackend::new().unwrap();g.set_setup_mode(mode).unwrap();Some(g)} else {None};
            let backend:&dyn OperatorBackend=match &gpu {Some(g)=>g,None=>&cpu};
            let mut trace=CpuTrace::new();
            let circuit=ToyCircuit::new(n).unwrap();let (instance,witness)=circuit.valid_instance();
            let started=Instant::now();
            let params=DynaverseParams::setup_with_backend_traced(circuit,&mut StdRng::seed_from_u64(17022),backend,&mut trace).unwrap();
            let wall=started.elapsed().as_nanos();
            println!("SETUP_CSV,{mode:?},{n},{wall},{}",trace.recorded_total().as_nanos());
            for phase in CpuPhase::ALL {let ns=trace.elapsed(phase).as_nanos();if ns>0 {println!("PHASE_CSV,{mode:?},{n},{},{ns}",phase.name());}}
            if let Some(g)=gpu {println!("SETUP_STATS,{mode:?},{n},{:?},{:?}",g.setup_stats().unwrap(),g.setup_host_stats());}
            let proof=params.prove_with_aux(&instance,&witness).unwrap().0;
            assert!(params.verifying_key().verify_reference(&instance,&proof).unwrap());
        }
    }
}
