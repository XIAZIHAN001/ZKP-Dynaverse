//! Raw ABI admission source; no execution before separate user approval.
use super::*;

#[test]
fn raw_shapes_and_nulls_preserve_output_and_metadata() {
    let backend=crate::IcicleBackend::new().unwrap();
    let p=[0u8;128];let q=[0u8;256];
    for (pl,ql,ol,ml,nulls) in [
        (63,128,BN254_MILLER_OUTPUT_BYTES,3,0),(64,127,BN254_MILLER_OUTPUT_BYTES,3,0),
        (128,256,BN254_MILLER_OUTPUT_BYTES,3,0),(64,128,BN254_MILLER_OUTPUT_BYTES-1,3,0),
        (64,128,BN254_MILLER_OUTPUT_BYTES,2,0),(0,0,BN254_MILLER_OUTPUT_BYTES,3,0),
        (64,128,BN254_MILLER_OUTPUT_BYTES,3,1),(64,128,BN254_MILLER_OUTPUT_BYTES,3,2),
        (64,128,BN254_MILLER_OUTPUT_BYTES,3,4),(64,128,BN254_MILLER_OUTPUT_BYTES,3,8),
    ] {
        let mut output=vec![0xa5;BN254_MILLER_OUTPUT_BYTES];let mut meta=[0xdeadbeefu32;3];let mut error=[0u8;256];
        // SAFETY: backing buffers cover all lengths; invalid shapes/nulls must
        // be rejected before CUDA access or writes to either destination.
        let status=unsafe{dvgpu_bn254_miller_inspect(backend.handle.as_ptr(),
            if nulls&1!=0{std::ptr::null()}else{p.as_ptr()},pl,
            if nulls&2!=0{std::ptr::null()}else{q.as_ptr()},ql,
            if nulls&4!=0{std::ptr::null_mut()}else{output.as_mut_ptr()},ol,
            if nulls&8!=0{std::ptr::null_mut()}else{meta.as_mut_ptr()},ml,error.as_mut_ptr().cast(),error.len())};
        assert_eq!(status,-1);assert!(output.iter().all(|&b|b==0xa5));assert_eq!(meta,[0xdeadbeef;3]);
        backend.memory_info().unwrap();
    }
    assert_eq!(backend.inspect_bn254_miller(&p[..64],&q[..128]).unwrap().count,0);
}

#[test]
fn raw_null_owner_preserves_both_destinations() {
    let p=[0u8;64];let q=[0u8;128];let mut output=vec![0xa5;BN254_MILLER_OUTPUT_BYTES];
    let mut meta=[0xdeadbeefu32;3];let mut error=[0u8;256];
    // SAFETY: null owner is explicitly rejected, all other pointers are valid.
    let status=unsafe{dvgpu_bn254_miller_inspect(std::ptr::null_mut(),p.as_ptr(),p.len(),q.as_ptr(),q.len(),
        output.as_mut_ptr(),output.len(),meta.as_mut_ptr(),meta.len(),error.as_mut_ptr().cast(),error.len())};
    assert_eq!(status,-4);assert!(output.iter().all(|&b|b==0xa5));assert_eq!(meta,[0xdeadbeef;3]);
}

#[test]
fn quota_rejection_precedes_allocation_and_publication() {
    let domain_bytes=(crate::MAX_NTT_SIZE as u64+1+6+64)*32;let journal_bytes=1u64<<20;
    let backend=crate::IcicleBackend::with_resident_limits(domain_bytes+(64u64<<20)+journal_bytes,journal_bytes).unwrap();
    let before=backend.resident_diagnostics().unwrap();let p=[0u8;64];let q=[0u8;128];
    let mut output=vec![0xa5;BN254_MILLER_OUTPUT_BYTES];let mut meta=[0xdeadbeefu32;3];let mut error=[0u8;256];
    // SAFETY: valid exact buffers; viable owner quota cannot fit Miller scratch.
    let status=unsafe{dvgpu_bn254_miller_inspect(backend.handle.as_ptr(),p.as_ptr(),p.len(),q.as_ptr(),q.len(),
        output.as_mut_ptr(),output.len(),meta.as_mut_ptr(),meta.len(),error.as_mut_ptr().cast(),error.len())};
    assert_eq!(status,-1);assert!(output.iter().all(|&b|b==0xa5));assert_eq!(meta,[0xdeadbeef;3]);
    let after=backend.resident_diagnostics().unwrap();
    assert_eq!(before.temporary_allocations,after.temporary_allocations);
    assert_eq!(before.temporary_peak_bytes,after.temporary_peak_bytes);backend.memory_info().unwrap();
}
