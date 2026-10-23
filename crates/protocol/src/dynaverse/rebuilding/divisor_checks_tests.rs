use super::*;
use crate::incremental_rebuild::operators::CpuRebuildBackend;
use rand::{rngs::StdRng,SeedableRng};

fn config(enabled:bool)->IncrementalConfig {
    IncrementalConfig {window:Some(4),trace_updates:true,rebuild_divisor_cache:enabled,
        rebuild_cache_mode:VerificationMode::Product,rebuild_prepared_cache_roots:Some(3),
        rebuild_cache_readback_roots:Some(16),unchanged_cache_reuse:true,
        ..Default::default()}
}

#[test]
fn full_divisor_initial_checker_reuses_construction_without_hiding_equations() {
    let circuit=ToyCircuit::new(16).unwrap();let(instance,witness)=circuit.valid_instance();
    let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(27200)).unwrap();
    let baseline=params.start_incremental_session(&instance,&witness,config(false),CpuRebuildBackend::default()).unwrap();
    let cached=params.start_incremental_session(&instance,&witness,config(true),CpuRebuildBackend::default()).unwrap();
    assert_eq!(baseline.proof(),cached.proof());
    assert_eq!(baseline.initial_cache_stats(),InitialCacheStats {equations:48,divisors_formed:16,divisor_cache_hits:0});
    assert_eq!(cached.initial_cache_stats(),InitialCacheStats {equations:48,divisors_formed:0,divisor_cache_hits:16});
    assert_eq!(cached.cache_divisor_stats().unwrap().construction_divisors,16);
}

#[test]
fn full_divisor_update_eliminates_construction_but_keeps_independent_pairings() {
    let n=16;let circuit=ToyCircuit::new(n).unwrap();let(instance,witness)=circuit.valid_instance();
    let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(27201)).unwrap();
    let mut baseline=params.start_incremental_session(&instance,&witness,config(false),CpuRebuildBackend::default()).unwrap();
    let mut cached=params.start_incremental_session(&instance,&witness,config(true),CpuRebuildBackend::default()).unwrap();
    let mut update=Update {indices:Vec::new(),values:Vec::new()};
    for i in 1..=8 {
        let v=witness.wires[4*n+i]+Fr::from(3u64);
        update.indices.extend([4*n+i,5*n+i]);update.values.extend([v,witness.wires[3*n+i]*v]);
    }
    baseline.update(&update).unwrap();cached.update(&update).unwrap();
    assert_eq!(baseline.proof(),cached.proof());assert_eq!(baseline.current_witness(),cached.current_witness());
    let old=baseline.update_timings().cache_detail;let new=cached.update_timings().cache_detail;
    assert_eq!(old.divisors_formed,26);assert_eq!(new.divisors_formed,0);
    assert_eq!(new.divisor_cache_hits,26);
    assert_eq!(new.equations,32);assert_eq!(new.pairing_terms,64);
    assert_eq!(new.miller_calls,32);assert_eq!(new.final_exponentiations,32);
    assert_eq!(new.g2_prepare_builds,old.g2_prepare_builds);
    assert!(cached.last_receipt().swapped && cached.rebuild_progress().is_none());
    assert!(cached.last_receipt().rebuild_units<=cached.last_receipt().allowance);
    assert!(params.verifying_key().verify_reference(&instance,cached.proof()).unwrap());
}

#[test]
fn full_divisor_cache_rejects_changed_equations_and_cancelling_errors() {
    use ark_ff::UniformRand;
    let n=8;let circuit=ToyCircuit::new(n).unwrap();let(_,mut witness)=circuit.valid_instance();
    let params=DynaverseParams::setup(circuit,&mut StdRng::seed_from_u64(27202)).unwrap();
    let mut rng=StdRng::seed_from_u64(27203);
    for v in &mut witness.wires[3*n..4*n] {*v=Fr::rand(&mut rng);}
    let domain=radix2_domain(n).unwrap();let coeff=domain.ifft(&witness.wires[3*n..4*n]);
    let commitment=G1Projective::from(params.srs.kzg_x.commit(&coeff).unwrap().point);
    let points=params.srs.kzg_x.precompute_single_openings(&coeff,&domain).unwrap();
    let mut storage=AnchorStorage {witness,cache:KzgPrecomputation {q4:vec![],q5:vec![],q6:vec![]}};
    let bound=divisors::PublicDivisors::host_bound(n,true,VerificationMode::Product).unwrap();
    let table=divisors::PublicDivisors::new(&params,bound).unwrap();
    for cap in [None,Some(3),Some(8)] {
        let prepared=cap.map(|cap| {
            let b=preparation::PublicPreparation::host_bound(n,Some(cap),VerificationMode::Product).unwrap();
            preparation::PublicPreparation::new(&params,cap,b).unwrap()
        });
        let check=|storage:&AnchorStorage,commitment,start,points:&[G1Affine]| {
            validate_cache_block_cached(&params,storage,commitment,0,start,points,VerificationMode::Product,
                &mut VerificationTrace::disabled(),&mut Default::default(),prepared.as_ref(),Some(&table))
        };
        for (start,len) in [(0,8),(2,3),(7,1),(8,0)] {check(&storage,commitment,start,&points[start..start+len]).unwrap();}
        assert!(check(&storage,commitment+G1Projective::generator(),0,&points).is_err());
        storage.witness.wires[3*n+1]+=Fr::from(1u64);
        assert!(check(&storage,commitment,0,&points).is_err());
        storage.witness.wires[3*n+1]-=Fr::from(1u64);
        let mut corrupt=points.clone();corrupt.swap(0,1);
        assert!(check(&storage,commitment,0,&corrupt).is_err());
        let powers=params.srs.kzg_x.rebuild_g1_powers();
        let mut corrupt=points.clone();
        corrupt[0]=(G1Projective::from(corrupt[0])+G1Projective::from(powers[1])
            -G1Projective::from(powers[0])*params.srs.sparse.small_points[1]).into_affine();
        corrupt[1]=(G1Projective::from(corrupt[1])-G1Projective::from(powers[1])
            +G1Projective::from(powers[0])*params.srs.sparse.small_points[0]).into_affine();
        assert!(check(&storage,commitment,0,&corrupt).is_err());
        assert!(check(&storage,commitment,8,&points[..1]).is_err());
        let mut constant=storage.witness.clone();constant.wires[3*n..4*n].fill(Fr::from(7u64));
        let constant=AnchorStorage {witness:constant,cache:KzgPrecomputation {q4:vec![],q5:vec![],q6:vec![]}};
        check(&constant,G1Projective::from(powers[0])*Fr::from(7u64),0,&[G1Affine::identity();8]).unwrap();
    }
}
