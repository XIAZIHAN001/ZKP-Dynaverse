//! Source-stage raw ABI tests; execute only after a later test approval.
use super::*;

#[test]
fn raw_bad_lengths_and_nulls_preserve_all_outputs_and_owner() {
    let backend = crate::IcicleBackend::new().unwrap();
    let input = [0u8;128];
    for (input_bytes,output_bytes,count_len,status_len,null_input,null_output,null_counts,null_status) in [
        (127,BN254_LINE_ITEM_BYTES,1,1,false,false,false,false),
        (128,BN254_LINE_ITEM_BYTES-1,1,1,false,false,false,false),
        (128,BN254_LINE_ITEM_BYTES,0,1,false,false,false,false),
        (128,BN254_LINE_ITEM_BYTES,1,0,false,false,false,false),
        (0,0,1,0,false,false,false,false),
        (128,BN254_LINE_ITEM_BYTES,1,1,true,false,false,false),
        (128,BN254_LINE_ITEM_BYTES,1,1,false,true,false,false),
        (128,BN254_LINE_ITEM_BYTES,1,1,false,false,true,false),
        (128,BN254_LINE_ITEM_BYTES,1,1,false,false,false,true),
    ] {
        let mut output = vec![0xa5u8;BN254_LINE_ITEM_BYTES];
        let mut counts = [0xdeadbeefu32]; let mut statuses = [0xdeadbeefu32]; let mut error=[0u8;256];
        // SAFETY: each non-null buffer covers its declared length; nulls and
        // inconsistent shapes must be rejected before any buffer/CUDA access.
        let status = unsafe { dvgpu_bn254_lines_inspect(backend.handle.as_ptr(),
            if null_input {std::ptr::null()} else {input.as_ptr()},input_bytes,
            if null_output {std::ptr::null_mut()} else {output.as_mut_ptr()},output_bytes,
            if null_counts {std::ptr::null_mut()} else {counts.as_mut_ptr()},count_len,
            if null_status {std::ptr::null_mut()} else {statuses.as_mut_ptr()},status_len,
            error.as_mut_ptr().cast(),error.len()) };
        assert_eq!(status,-1); assert!(output.iter().all(|&b|b==0xa5));
        assert_eq!(counts,[0xdeadbeef]); assert_eq!(statuses,[0xdeadbeef]);
        backend.memory_info().unwrap();
    }
    let retry = backend.inspect_bn254_lines(&input).unwrap();
    assert_eq!(retry.statuses,[BN254_LINES_INFINITY]); assert_eq!(retry.counts,[0]);
}

#[test]
fn raw_oversized_batch_preserves_complete_backing_buffers() {
    let backend = crate::IcicleBackend::new().unwrap();
    let input = vec![0u8;9*128]; let mut output = vec![0xa5;9*BN254_LINE_ITEM_BYTES];
    let mut counts = [0xdeadbeefu32;9]; let mut statuses = [0xdeadbeefu32;9]; let mut error=[0u8;256];
    // SAFETY: full nine-item allocations even though the native limit is eight.
    let status = unsafe { dvgpu_bn254_lines_inspect(backend.handle.as_ptr(),input.as_ptr(),input.len(),
        output.as_mut_ptr(),output.len(),counts.as_mut_ptr(),counts.len(),
        statuses.as_mut_ptr(),statuses.len(),error.as_mut_ptr().cast(),error.len()) };
    assert_eq!(status,-1); assert!(output.iter().all(|&b|b==0xa5));
    assert_eq!(counts,[0xdeadbeef;9]); assert_eq!(statuses,[0xdeadbeef;9]);
    backend.memory_info().unwrap();
}

#[test]
fn quota_rejection_precedes_allocations_and_preserves_three_outputs() {
    let domain_bytes = (crate::MAX_NTT_SIZE as u64+1+6+64)*32;
    let journal_bytes=1u64<<20;
    let backend = crate::IcicleBackend::with_resident_limits(
        domain_bytes+(64u64<<20)+journal_bytes,journal_bytes).unwrap();
    let before = backend.resident_diagnostics().unwrap();
    let input=[0u8;128]; let mut output=vec![0xa5u8;BN254_LINE_ITEM_BYTES];
    let mut counts=[0xdeadbeefu32]; let mut statuses=[0xdeadbeefu32]; let mut error=[0u8;256];
    // SAFETY: all buffers valid; the exact viable owner quota must reject the
    // additional line scratch envelope without touching outputs or allocating.
    let status = unsafe { dvgpu_bn254_lines_inspect(backend.handle.as_ptr(),input.as_ptr(),input.len(),
        output.as_mut_ptr(),output.len(),counts.as_mut_ptr(),counts.len(),
        statuses.as_mut_ptr(),statuses.len(),error.as_mut_ptr().cast(),error.len()) };
    assert_eq!(status,-1); assert!(output.iter().all(|&b|b==0xa5));
    assert_eq!(counts,[0xdeadbeef]); assert_eq!(statuses,[0xdeadbeef]);
    let after=backend.resident_diagnostics().unwrap();
    assert_eq!(before.temporary_allocations,after.temporary_allocations);
    assert_eq!(before.temporary_peak_bytes,after.temporary_peak_bytes);
    backend.memory_info().unwrap();
}
