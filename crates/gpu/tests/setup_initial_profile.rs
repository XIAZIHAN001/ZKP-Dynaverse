#![cfg(feature = "icicle")]
use dynaverse_gpu::IcicleBackend;
use dynaverse_protocol::{dynaverse::DynaverseParams, toy_circuit::ToyCircuit,
    incremental_rebuild::{IncrementalConfig,operators::RebuildBackend}};
use rand::{rngs::StdRng,SeedableRng};
use std::time::Instant;

// Uses only pre-change public APIs, so the identical probe runs against the
// saved source snapshot. One cold sample per size; includes Program preparation,
// proof/cache construction and its validation, excludes context creation/setup.
#[test]
#[ignore = "explicit matched snapshot/current initial-session timing"]
fn initial_session_timing_probe() {
    for n in [16,64,256] {
        let circuit=ToyCircuit::new(n).unwrap();
        let (instance,witness)=circuit.valid_instance();
        let started=Instant::now();
        let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(6500+n as u64)).unwrap();
        let setup=started.elapsed();
        let backend=IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap();
        let started=Instant::now();
        let session=params.start_incremental_session(&instance,&witness,IncrementalConfig::default(),backend).unwrap();
        let initial=started.elapsed();
        let expected=params.prove_with_aux(&instance,&witness).unwrap().0;
        assert_eq!(*session.proof(),expected);
        assert!(params.verifying_key().verify_reference(&instance,session.proof()).unwrap());
        let stats=session.backend().stats();
        eprintln!("INITIAL_SESSION n={n} setup_ns={} initial_ns={} resident={} calls={} fr={} ec={} g1={} g2={}",
            setup.as_nanos(),initial.as_nanos(),stats.bytes_allocated,stats.calls,
            stats.fr_transform_elements,stats.g1_transform_elements,stats.msm_g1_points,stats.msm_g2_points);
    }
}
