//! Raw ABI tests to execute only after the source-stage approval gate opens.
use super::*;

#[test]
fn raw_invalid_arguments_leave_outputs_and_owner_usable() {
    let backend = crate::IcicleBackend::new().unwrap();
    let left = [0u8; 64];
    let right = [0u8; 64];
    // field, operation, power, left bytes, right bytes, output bytes, statuses.
    // Each supplied length fits its backing buffer, including rejected calls.
    for (degree,operation,power,left_bytes,right_bytes,output_bytes,statuses_count) in [
        (0,1,0,64,64,64,1), (6,1,0,64,64,64,1),
        (2,0,0,64,64,64,1), (2,7,0,64,64,64,1),
        (2,1,1,64,64,64,1), (2,6,13,64,0,64,1),
        (2,4,0,63,0,63,0), (2,1,0,64,63,64,1),
        (2,4,0,64,64,64,1), (2,4,0,64,0,63,1),
        (2,4,0,64,0,64,0), (2,4,0,0,0,0,1),
    ] {
        let mut output = [0xa5u8; 64];
        let mut statuses = [0xdeadbeefu32; 1];
        let mut error = [0u8; 256];
        // SAFETY: buffers outlive the call and cover every declared length.
        // The deliberately inconsistent shapes must be rejected before CUDA.
        let status = unsafe { dvgpu_bn254_tower_inspect(backend.handle.as_ptr(),degree,operation,power,
            left.as_ptr(),left_bytes,right.as_ptr(),right_bytes,output.as_mut_ptr(),output_bytes,
            statuses.as_mut_ptr(),statuses_count,error.as_mut_ptr().cast(),error.len()) };
        assert_eq!(status,-1);
        assert_eq!(output,[0xa5; 64]);
        assert_eq!(statuses,[0xdeadbeef]);
        backend.memory_info().unwrap();
    }
    let retry = backend.inspect_bn254_tower(Bn254TowerField::Fq2,
        Bn254TowerOperation::Add,&left,&right).unwrap();
    assert_eq!(retry.statuses,[0]);
    assert_eq!(retry.output,vec![0;64]);
}

#[test]
fn raw_oversized_batch_does_not_publish_output() {
    let backend = crate::IcicleBackend::new().unwrap();
    let left = vec![0u8; 65*64];
    let mut output = vec![0xa5u8; left.len()];
    let mut statuses = vec![0xdeadbeefu32; 65];
    let mut error = [0u8;256];
    // SAFETY: full backing allocations cover even this rejected batch; no
    // dangling or undersized pointer is used to manufacture the rejection.
    let status = unsafe { dvgpu_bn254_tower_inspect(backend.handle.as_ptr(),2,4,0,
        left.as_ptr(),left.len(),std::ptr::null(),0,output.as_mut_ptr(),output.len(),
        statuses.as_mut_ptr(),statuses.len(),error.as_mut_ptr().cast(),error.len()) };
    assert_eq!(status,-1);
    assert_eq!(output,vec![0xa5; left.len()]);
    assert_eq!(statuses,vec![0xdeadbeef;65]);
    backend.memory_info().unwrap();
}

#[test]
fn resident_quota_rejection_preserves_both_outputs_and_owner() {
    // An empty resident owner already reserves the pinned Fr domain, journal
    // and a 64 MiB workspace margin. Use its exact viable quota, leaving no
    // space for the tower probe's additional I/O scratch above that margin.
    let domain_bytes = (crate::MAX_NTT_SIZE as u64 + 1 + 6 + 64) * 32;
    let journal_bytes = 1u64 << 20;
    let backend = crate::IcicleBackend::with_resident_limits(
        domain_bytes + (64u64 << 20) + journal_bytes, journal_bytes).unwrap();
    let before = backend.resident_diagnostics().unwrap();
    let left = [0u8;64];
    let mut output = [0xa5u8;64];
    let mut statuses = [0xdeadbeefu32;1];
    let mut error = [0u8;256];
    // SAFETY: valid single-item buffers. The configured owner has less than
    // the probe's conservative scratch envelope, so preflight must reject.
    let status = unsafe { dvgpu_bn254_tower_inspect(backend.handle.as_ptr(),2,4,0,
        left.as_ptr(),left.len(),std::ptr::null(),0,output.as_mut_ptr(),output.len(),
        statuses.as_mut_ptr(),statuses.len(),error.as_mut_ptr().cast(),error.len()) };
    assert_eq!(status,-1);
    assert_eq!(output,[0xa5;64]);
    assert_eq!(statuses,[0xdeadbeef]);
    let after = backend.resident_diagnostics().unwrap();
    assert_eq!(after.temporary_allocations,before.temporary_allocations);
    assert_eq!(after.temporary_peak_bytes,before.temporary_peak_bytes);
    backend.memory_info().unwrap();
}
