#![cfg(feature = "icicle")]
use ark_bn254::{Fr, G1Affine};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::UniformRand;
use dynaverse_gpu::IcicleBackend;
use dynaverse_protocol::incremental_rebuild::operators::{BufferId, BufferKind, RebuildBackend};
use rand::{rngs::StdRng, SeedableRng};

// Scaling on the device gives nontrivial projective Z, including identities.
// The independent expected values are computed by arkworks on the CPU.
fn fixture(b: &mut IcicleBackend, n: usize) -> (BufferId, Vec<G1Affine>) {
    let mut rng = StdRng::seed_from_u64(28100);
    let mut points = Vec::new();
    let mut scalars = Vec::new();
    let mut expected = Vec::new();
    for i in 0..n {
        let point = if i % 11 == 0 { G1Affine::identity() }
            else { (G1Affine::generator() * Fr::from(i as u64 + 1)).into_affine() };
        let scalar = if i % 13 == 0 { Fr::from(0u64) } else { Fr::rand(&mut rng) };
        expected.push((point * scalar).into_affine());
        points.push(point); scalars.push(scalar);
    }
    let input = b.alloc(BufferKind::G1, n, false).unwrap();
    let fields = b.alloc(BufferKind::Fr, n, false).unwrap();
    let output = b.alloc(BufferKind::G1, n, true).unwrap();
    b.write_points(input, 0, &points).unwrap();
    b.write_fields(fields, 0, &scalars).unwrap();
    b.scale_points(input, 0, fields, 0, output, 0, n).unwrap();
    (output, expected)
}

#[test]
fn bounded_device_readback_matches_cpu_and_reduces_physical_transfer() {
    let mut b = IcicleBackend::with_resident_limits(1 << 30, 1 << 20).unwrap();
    let (id, expected) = fixture(&mut b, 1032);
    b.seal();
    let allocated = b.stats().bytes_allocated;
    let allocations = b.resident_diagnostics().unwrap().resident_allocations;
    for enabled in [false, true] {
        b.set_device_affine_readback(enabled);
        for n in [0, 1, 3, 15, 16, 63, 64, 128, 255, 256, 257, 1024] {
            let mut output = vec![G1Affine::identity(); n];
            let before = b.resident_diagnostics().unwrap();
            let logical = b.stats().d2h_bytes;
            b.read_points(id, 3, &mut output).unwrap();
            assert_eq!(output, expected[3..3+n]);
            let after = b.resident_diagnostics().unwrap();
            let width = if enabled && n <= 256 { 64 } else { 96 };
            assert_eq!(after.physical_d2h_bytes-before.physical_d2h_bytes, n as u64 * width,
                "enabled={enabled} n={n}");
            assert_eq!(b.stats().d2h_bytes-logical, n as u64 * 64);
            assert_eq!(b.stats().bytes_allocated, allocated);
            assert_eq!(after.resident_allocations, allocations);
        }
    }
}

#[test]
fn packed_fields_invalid_ranges_and_rollback_keep_their_contracts() {
    let mut b = IcicleBackend::with_resident_limits(1 << 30, 1 << 20).unwrap();
    let (id, expected) = fixture(&mut b, 32);
    let packed_values = [G1Affine::identity(), G1Affine::generator(), -G1Affine::generator()];
    let packed = b.register_msm_bases(&packed_values).unwrap();
    let field = b.alloc(BufferKind::Fr, 3, true).unwrap();
    b.write_fields(field, 0, &[Fr::from(1u64); 3]).unwrap();
    let mut other = IcicleBackend::with_resident_limits(1 << 30, 1 << 20).unwrap();
    let foreign = other.alloc(BufferKind::G1, 3, true).unwrap();
    b.set_device_affine_readback(true);
    b.seal();
    let allocated = b.stats().bytes_allocated;
    let mut output = [G1Affine::generator(); 3];
    let before = b.resident_diagnostics().unwrap();
    b.read_points(packed, 0, &mut output).unwrap();
    assert_eq!(output, packed_values);
    assert_eq!(b.resident_diagnostics().unwrap().physical_d2h_bytes-before.physical_d2h_bytes, 3*64);
    let mut fields = [Fr::from(0u64); 3];
    b.read_fields(field, 0, &mut fields).unwrap();
    assert_eq!(fields, [Fr::from(1u64); 3]);

    // A bad read must not overwrite caller output, transfer point data, or
    // retire this owner. Follow every error with a valid profiled read.
    for (bad, offset) in [(foreign, 0), (id, 31), (field, 0), (id, usize::MAX)] {
        let sentinel = [G1Affine::generator(); 3];
        output = sentinel;
        let before = b.resident_diagnostics().unwrap();
        assert!(b.read_points(bad, offset, &mut output).is_err());
        assert_eq!(output, sentinel);
        assert_eq!(b.resident_diagnostics().unwrap().physical_d2h_bytes, before.physical_d2h_bytes);
        assert!(b.profile_read_points(bad, offset, &mut output).is_err());
        assert_eq!(output, sentinel);
        let p = b.profile_read_points(id, 3, &mut output).unwrap();
        assert_eq!(output, expected[3..6]);
        assert_eq!(p.native.d2h_bytes, 3*64);
    }

    b.begin_transaction().unwrap();
    b.write_points(id, 3, &packed_values).unwrap();
    b.read_points(id, 3, &mut output).unwrap();
    assert_eq!(output, packed_values);
    b.rollback_transaction().unwrap();
    b.read_points(id, 3, &mut output).unwrap();
    assert_eq!(output, expected[3..6]);
    let mut sum = G1Affine::identity().into_group();
    for point in &expected[3..6] { sum += point; }
    assert_eq!(b.msm_g1(id, 3, field, 0, 3).unwrap(), sum.into_affine());
    // MSM normalizes into the same scratch; a subsequent read must remain
    // independent of those earlier scratch contents.
    b.read_points(id, 3, &mut output).unwrap();
    assert_eq!(output, expected[3..6]);
    b.begin_transaction().unwrap();
    b.write_points(id, 3, &packed_values).unwrap();
    b.commit_transaction();
    b.read_points(id, 3, &mut output).unwrap();
    assert_eq!(output, packed_values);
    assert_eq!(b.stats().bytes_allocated, allocated);
}

#[test]
#[ignore = "explicit serial host/device readback clock diagnostic"]
fn local_affine_readback_baseline() {
    let mut b = IcicleBackend::with_resident_limits(1 << 30, 1 << 20).unwrap();
    let (id, expected) = fixture(&mut b, 1032);
    b.seal();
    for n in [0, 1, 16, 64, 128, 256, 257, 1024] {
        for repeat in 0..8 {
            let mut output = vec![G1Affine::identity(); n];
            let before = b.resident_diagnostics().unwrap();
            let p = b.profile_read_points(id, 3, &mut output).unwrap();
            let after = b.resident_diagnostics().unwrap();
            assert_eq!(output, expected[3..3+n]);
            assert_eq!(after.physical_d2h_bytes-before.physical_d2h_bytes, n as u64 * 96);
            eprintln!("AR_BASE n={n} repeat={repeat} warm={} total_ns={} ffi_ns={} decode_ns={} native_ns={} conversion_ns={} operation_host_ns={} wait_ns={} event_ns={} event_operation_ns={} physical_d2h={}",
                usize::from(repeat<2), p.total_ns,p.ffi_ns,p.decoding_ns,p.native.native_wall_ns,
                p.native.conversion_host_ns,p.native.operation_host_ns,p.native.wait_host_ns,
                p.native.event_total_ns,p.native.event_operation_ns,after.physical_d2h_bytes-before.physical_d2h_bytes);
        }
    }
}

#[test]
#[ignore = "explicit serial paired readback clock diagnostic"]
fn local_affine_readback_comparison() {
    let mut b = IcicleBackend::with_resident_limits(1 << 30, 1 << 20).unwrap();
    let (id, expected) = fixture(&mut b, 1032);
    b.seal();
    for n in [0, 1, 16, 64, 128, 256, 257, 1024] {
        for repeat in 0..8 {
            for offset in 0..2 {
                let enabled = (repeat + offset) % 2 == 1;
                b.set_device_affine_readback(enabled);
                let mut output = vec![G1Affine::identity(); n];
                let before = b.resident_diagnostics().unwrap();
                let p = b.profile_read_points(id, 3, &mut output).unwrap();
                let after = b.resident_diagnostics().unwrap();
                assert_eq!(output, expected[3..3+n]);
                let width = if enabled && n <= 256 { 64 } else { 96 };
                assert_eq!(after.physical_d2h_bytes-before.physical_d2h_bytes, n as u64 * width);
                eprintln!("AR_MICRO n={n} enabled={} repeat={repeat} warm={} total_ns={} ffi_ns={} decode_ns={} native_ns={} conversion_ns={} operation_host_ns={} wait_ns={} event_ns={} event_operation_ns={} physical_d2h={}",
                    usize::from(enabled), usize::from(repeat<2), p.total_ns,p.ffi_ns,p.decoding_ns,p.native.native_wall_ns,
                    p.native.conversion_host_ns,p.native.operation_host_ns,p.native.wait_host_ns,
                    p.native.event_total_ns,p.native.event_operation_ns,after.physical_d2h_bytes-before.physical_d2h_bytes);
            }
        }
    }
}

#[allow(dead_code)]
#[path = "../../protocol/tests/support/cache_preparation_profile.rs"]
mod protocol_fixture;

#[test]
fn device_affine_epochs_match_host_gpu_cpu_and_dense_anchors() {
    use dynaverse_protocol::{dynaverse::DynaverseParams,toy_circuit::ToyCircuit,
        incremental_rebuild::{IncrementalConfig,SparseDispatchPolicy,operators::CpuRebuildBackend}};
    for n in [16,64] {for cap in [None,Some(7),Some(64)] {
        let c=ToyCircuit::new(n).unwrap();let(i,w)=c.valid_instance();
        let p=DynaverseParams::setup(c,&mut StdRng::seed_from_u64(28600+n as u64)).unwrap();
        let mut device=IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap();
        device.set_device_affine_readback(true);
        let cfg=IncrementalConfig {rebuild_prepared_cache_roots:Some(3),rebuild_cache_readback_roots:cap,
            ..protocol_fixture::config("prepared",SparseDispatchPolicy::ForceGpu)};
        let mut new=p.start_incremental_session(&i,&w,cfg.clone(),device).unwrap();
        let mut old=p.start_incremental_session(&i,&w,cfg.clone(),IcicleBackend::with_resident_limits(1<<30,64<<20).unwrap()).unwrap();
        let mut cpu=p.start_incremental_session(&i,&w,IncrementalConfig {sparse:SparseDispatchPolicy::CpuReference,..cfg},CpuRebuildBackend::default()).unwrap();
        let allocated=new.backend().stats().bytes_allocated;
        for step in 0..2 {
            let addition=step%2==1;
            let u=protocol_fixture::edits(new.current_witness(),n,if addition {4} else {8},addition,28700+step);
            let proof=new.update(&u).unwrap();
            assert_eq!(proof,old.update(&u).unwrap());assert_eq!(proof,cpu.update(&u).unwrap());
            assert_eq!(new.backend().stats().bytes_allocated,allocated);
            assert_eq!(new.update_timings().cache_detail.equations,2*n as u64);
            assert_eq!(new.update_timings().cache_detail.final_exponentiations,2*n as u64);
            assert!(new.last_receipt().rebuild_units<=new.last_receipt().allowance);
            assert!(p.verifying_key().verify_reference(&i,&proof).unwrap());
            assert_eq!(proof.anchor,p.prove_anchor(&i,new.current_witness()).unwrap().anchor);
        }
    }}
}

#[allow(dead_code)]
#[path = "support/affine_readback_profile.rs"]
mod full_profile;

#[test]
#[ignore = "explicit serial counterbalanced full update scan"]
fn local_device_affine_scaling() { full_profile::scaling(); }

#[test]
#[ignore = "explicit serial small update diagnostic"]
fn local_device_affine_pending() { full_profile::pending(); }
