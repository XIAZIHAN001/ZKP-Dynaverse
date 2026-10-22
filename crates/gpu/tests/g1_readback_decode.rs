#![cfg(feature = "icicle")]
use ark_bn254::{Fr,G1Affine};
use ark_ec::{AffineRepr,CurveGroup};
use dynaverse_gpu::IcicleBackend;
use dynaverse_protocol::incremental_rebuild::operators::{BufferKind,RebuildBackend};
use rand::{rngs::StdRng,SeedableRng};
#[allow(dead_code)]
#[path="../../protocol/tests/support/cache_preparation_profile.rs"]
mod fixture;

#[test]
fn bounded_profile_uses_the_selected_decoder_and_falls_back_above_cap() {
    let mut b=IcicleBackend::with_resident_limits(1<<30,1<<20).unwrap();
    let mut points=Vec::new();
    for i in 0..270 {
        points.push(if i%7==0 {G1Affine::identity()}
            else {(G1Affine::generator()*Fr::from(i as u64+1)).into_affine()});
    }
    let id=b.alloc(BufferKind::G1,points.len(),true).unwrap();
    b.write_points(id,0,&points).unwrap();b.seal();
    let mut first=[G1Affine::identity();1];
    let default=b.profile_read_points(id,3,&mut first).unwrap();
    assert_eq!(default.g1_full_decode_points,1);
    assert_eq!(default.g1_cofactor_decode_points,0);
    for device in [false,true] {for bounded in [false,true] {
        b.set_device_affine_readback(device);b.set_bounded_g1_decode(bounded);
        for n in [0,1,16,64,256,257] {
            let mut output=vec![G1Affine::identity();n];
            let before=b.resident_diagnostics().unwrap();
            let p=b.profile_read_points(id,3,&mut output).unwrap();
            assert_eq!(output,points[3..3+n]);
            let selected=bounded && n<=256;
            assert_eq!(p.g1_cofactor_decode_points,if selected {n as u64} else {0},
                "bounded={bounded} device={device} n={n}");
            assert_eq!(p.g1_full_decode_points,if selected {0} else {n as u64});
            let width=if device && n<=256 {64} else {96};
            assert_eq!(b.resident_diagnostics().unwrap().physical_d2h_bytes-before.physical_d2h_bytes,n as u64*width);
            let mut normal=vec![G1Affine::identity();n];
            b.read_points(id,3,&mut normal).unwrap();assert_eq!(normal,output);
        }
    }}
}

#[test]
fn bounded_decode_epochs_match_full_gpu_cpu_receipts_and_dense_anchors() {
    use dynaverse_protocol::{dynaverse::DynaverseParams,toy_circuit::ToyCircuit,
        incremental_rebuild::{IncrementalConfig,SparseDispatchPolicy,operators::CpuRebuildBackend}};
    for n in [16,64] {for device in [false,true] {
        let c=ToyCircuit::new(n).unwrap();let(i,w)=c.valid_instance();
        let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(29200+n as u64)).unwrap();
        let make=|bounded| {
            let mut b=IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap();
            b.set_device_affine_readback(device);b.set_bounded_g1_decode(bounded);b
        };
        let cfg=IncrementalConfig {rebuild_prepared_cache_roots:Some(3),rebuild_cache_readback_roots:Some(7),
            ..fixture::config("prepared",SparseDispatchPolicy::ForceGpu)};
        let mut new=p.start_incremental_session(&i,&w,cfg.clone(),make(true)).unwrap();
        let mut old=p.start_incremental_session(&i,&w,cfg.clone(),make(false)).unwrap();
        let mut cpu=p.start_incremental_session(&i,&w,IncrementalConfig {sparse:SparseDispatchPolicy::CpuReference,..cfg},CpuRebuildBackend::default()).unwrap();
        let allocated=new.backend().stats().bytes_allocated;
        for step in 0..2 {
            let addition=step%2==1;
            let u=fixture::edits(new.current_witness(),n,if addition {4} else {8},addition,29300+step);
            let proof=new.update(&u).unwrap();assert_eq!(proof,old.update(&u).unwrap());
            assert_eq!(proof,cpu.update(&u).unwrap());
            assert_eq!(new.last_receipt(),old.last_receipt());
            assert_eq!(new.resource_plan(),old.resource_plan());assert_eq!(new.budget(),old.budget());
            assert_eq!(new.update_timings().cache_detail.equations,2*n as u64);
            assert_eq!(new.update_timings().cache_detail.final_exponentiations,2*n as u64);
            assert_eq!(new.backend().stats().bytes_allocated,allocated);
            assert!(p.verifying_key().verify_reference(&i,&proof).unwrap());
            assert_eq!(proof.anchor,p.prove_anchor(&i,new.current_witness()).unwrap().anchor);
        }
    }}
}

#[allow(dead_code)]
#[path="support/g1_readback_profile.rs"]
mod profile;

#[test]
#[ignore="explicit serial four-route readback clock diagnostic"]
fn local_g1_decode_micro() {profile::micro();}

#[test]
#[ignore="explicit serial counterbalanced complete update comparison"]
fn local_g1_decode_scaling() {profile::scaling();}

#[test]
#[ignore="explicit serial small update equal-work diagnostic"]
fn local_g1_decode_pending() {profile::pending();}
