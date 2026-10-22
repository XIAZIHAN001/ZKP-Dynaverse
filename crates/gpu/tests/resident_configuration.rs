#![cfg(feature = "icicle")]

//! Exercise the legacy-to-resident transition through the public native ABI.
use ark_bn254::Fr;
use ark_ff::FftField;
use dynaverse_gpu::{encode_fr, BatchStats, MemoryInfo};
use std::ffi::{c_char, c_void};
use std::ptr::NonNull;

#[repr(C)]
#[derive(Default)]
struct ResidentStats {
    h2d_bytes: u64,
    d2h_bytes: u64,
    bytes_allocated: u64,
    fr_transform_elements: u64,
    g1_transform_elements: u64,
    msm_g1_points: u64,
    msm_g2_points: u64,
    calls: u64,
}

extern "C" {
    fn dvgpu_backend_create(owner: *mut *mut c_void, error: *mut c_char, capacity: usize) -> i32;
    fn dvgpu_backend_destroy(owner: *mut c_void);
    fn dvgpu_bn254_msm_batch(
        owner: *mut c_void, points: *const u8, scalars: *const u8,
        length: usize, count: usize, result: *mut u8,
        error: *mut c_char, capacity: usize,
    ) -> i32;
    fn dvgpu_backend_batch_stats(owner: *mut c_void, stats: *mut BatchStats) -> i32;
    fn dvgpu_backend_memory_info(
        owner: *mut c_void, info: *mut MemoryInfo, error: *mut c_char, capacity: usize,
    ) -> i32;
    fn dvgpu_resident_configure(
        owner: *mut c_void, limit: u64, journal: u64, root: *const u8,
        error: *mut c_char, capacity: usize,
    ) -> i32;
    fn dvgpu_resident_stats(owner: *mut c_void, stats: *mut ResidentStats) -> i32;
}

struct NativeOwner(NonNull<c_void>);

impl NativeOwner {
    fn new() -> Self {
        let mut pointer = std::ptr::null_mut();
        let mut error = [0u8; 256];
        // SAFETY: native initializes the opaque owner and bounded error buffer;
        // the successful result is owned exclusively by this RAII guard.
        let status = unsafe {
            dvgpu_backend_create(&mut pointer, error.as_mut_ptr().cast(), error.len())
        };
        assert_eq!(status, 0, "create native owner: {}", error_text(&error));
        Self(NonNull::new(pointer).expect("successful creation returned a null owner"))
    }

    fn pointer(&self) -> *mut c_void { self.0.as_ptr() }
}

impl Drop for NativeOwner {
    fn drop(&mut self) {
        // SAFETY: this guard is the sole owner, all calls are synchronous, and
        // native destruction accepts both healthy and retired owners.
        unsafe { dvgpu_backend_destroy(self.pointer()); }
    }
}

fn error_text(error: &[u8]) -> String {
    let end = error.iter().position(|&byte| byte == 0).unwrap_or(error.len());
    String::from_utf8_lossy(&error[..end]).into_owned()
}

#[test]
fn rejected_legacy_to_resident_configuration_can_retry_and_counts_retained_inputs() {
    const LENGTH: usize = 1 << 20;
    const JOURNAL_BYTES: u64 = 64;
    const RETAINED_BYTES: u64 = LENGTH as u64 * (32 + 64);
    let owner = NativeOwner::new();
    // Canonical BN254 G1 infinity is x||y = 64 zero bytes. Every scalar is
    // canonical Fr one. The legacy batch retains both device input arrays.
    let points = vec![0u8; LENGTH * 64];
    let mut scalars = vec![0u8; LENGTH * 32];
    let one = encode_fr(&Fr::from(1u64));
    for scalar in scalars.chunks_exact_mut(32) { scalar.copy_from_slice(&one); }
    let mut result = [0xa5u8; 64];
    let mut error = [0u8; 256];
    // SAFETY: owned canonical arrays contain LENGTH points/scalars and the
    // result contains one affine point. Native retains no borrowed pointer.
    let status = unsafe {
        dvgpu_bn254_msm_batch(owner.pointer(), points.as_ptr(), scalars.as_ptr(),
            LENGTH, 1, result.as_mut_ptr(), error.as_mut_ptr().cast(), error.len())
    };
    assert_eq!(status, 0, "legacy batch: {}", error_text(&error));
    assert_eq!(result, [0u8; 64]);
    let mut batch = BatchStats::default();
    // SAFETY: repr(C) BatchStats matches the native initialized output layout.
    assert_eq!(unsafe { dvgpu_backend_batch_stats(owner.pointer(), &mut batch) }, 0);
    assert_eq!(batch.input_capacity, LENGTH as u64);
    assert_eq!(batch.buffer_grows, 1);

    let root = encode_fr(&Fr::get_root_of_unity(LENGTH as u64).unwrap());
    error.fill(0);
    // SAFETY: the 32-byte root and error buffer remain live for synchronous
    // initialization. Existing retained inputs make this cap insufficient.
    let status = unsafe {
        dvgpu_resident_configure(owner.pointer(), 112 << 20, JOURNAL_BYTES,
            root.as_ptr(), error.as_mut_ptr().cast(), error.len())
    };
    assert_ne!(status, 0, "configuration must include the retained 96 MiB");

    let mut memory = MemoryInfo::default();
    error.fill(0);
    // SAFETY: repr(C) MemoryInfo and bounded error storage match the native ABI.
    let status = unsafe {
        dvgpu_backend_memory_info(owner.pointer(), &mut memory,
            error.as_mut_ptr().cast(), error.len())
    };
    assert_eq!(status, 0, "quota rejection retired the owner: {}", error_text(&error));
    assert!(memory.total_bytes > 0);

    error.fill(0);
    // SAFETY: the same live owner and root are retried after a recoverable
    // refusal. A failed initialization must not publish resident state.
    let status = unsafe {
        dvgpu_resident_configure(owner.pointer(), 512 << 20, JOURNAL_BYTES,
            root.as_ptr(), error.as_mut_ptr().cast(), error.len())
    };
    assert_eq!(status, 0, "retry resident configuration: {}", error_text(&error));
    let mut resident = ResidentStats::default();
    // SAFETY: this repr(C) output mirrors all eight native u64 fields exactly.
    assert_eq!(unsafe { dvgpu_resident_stats(owner.pointer(), &mut resident) }, 0);
    assert!(resident.bytes_allocated >= RETAINED_BYTES + LENGTH as u64 * 32 + JOURNAL_BYTES,
        "resident accounting omitted retained legacy inputs: {}", resident.bytes_allocated);
    assert!(resident.bytes_allocated <= 512 << 20);

    result.fill(0xa5);
    error.fill(0);
    // SAFETY: the first eight canonical entries are valid borrowed inputs.
    // This smaller call reuses the original retained arrays after configuration.
    let status = unsafe {
        dvgpu_bn254_msm_batch(owner.pointer(), points.as_ptr(), scalars.as_ptr(),
            8, 1, result.as_mut_ptr(), error.as_mut_ptr().cast(), error.len())
    };
    assert_eq!(status, 0, "batch after configuration: {}", error_text(&error));
    assert_eq!(result, [0u8; 64]);
    let mut after = BatchStats::default();
    // SAFETY: the owner remains live and the output has the native layout.
    assert_eq!(unsafe { dvgpu_backend_batch_stats(owner.pointer(), &mut after) }, 0);
    assert_eq!(after.input_capacity, batch.input_capacity);
    assert_eq!(after.buffer_grows, batch.buffer_grows);
}
