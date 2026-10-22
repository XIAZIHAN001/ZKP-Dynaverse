//! Thin synchronous wrapper. Native code borrows encoded host buffers only
//! until the call returns; no Rust pointer is retained on the device or in C++.
use crate::codec::{decode_fr, decode_g1, encode_fr, encode_g1, FR_BYTES, G1_BYTES};
use crate::MsmTimeline;
use anyhow::{ensure, Result};
use ark_bn254::{Fr, G1Affine, G2Affine};
use ark_ec::CurveConfig;
use ark_ff::FftField;
use dynaverse_protocol::{BatchMsmRequest, GpuCapabilities, OperatorBackend, SparseMsmRequest};
use ark_serialize::{CanonicalDeserialize, Compress, Validate};
use std::ffi::{c_char, c_void, CStr};
use std::cell::{Cell,RefCell};
use std::ptr::NonNull;
use std::time::{Duration, Instant};

/// Optional host-clock diagnostics. Native spans are nested inside `native`;
/// ICICLE's span includes allocation/transfers/kernels/sync, not kernel alone.
#[derive(Clone, Copy, Debug, Default)]
pub struct MsmProfile {
    pub total: Duration,
    pub validation: Duration,
    pub encoding: Duration,
    pub native: Duration,
    pub decode: Duration,
    pub native_conversion: Duration,
    pub icicle_inclusive: Duration,
    pub native_output: Duration,
}

// repr(C) and fixed-width fields match DvgpuMsmTimings in the C header.
#[repr(C)]
#[derive(Default)]
struct NativeTimings { conversion_ns: u64, icicle_ns: u64, output_ns: u64 }

extern "C" {
    fn dvgpu_backend_create_with_large_bucket_factor(factor: u32, backend: *mut *mut c_void,
        error: *mut c_char, error_capacity: usize) -> i32;
    fn dvgpu_backend_destroy(backend: *mut c_void);
    fn dvgpu_backend_device_identity(backend: *mut c_void, output: *mut c_char, capacity: usize,
        error: *mut c_char, error_capacity: usize) -> i32;
    fn dvgpu_backend_batch_stats(backend: *mut c_void, stats: *mut BatchStats) -> i32;
    fn dvgpu_backend_memory_info(backend: *mut c_void, info: *mut MemoryInfo,
        error: *mut c_char, error_capacity: usize) -> i32;
    fn dvgpu_bn254_msm_batch(backend: *mut c_void, points: *const u8, scalars: *const u8,
        length: usize, count: usize, result: *mut u8, error: *mut c_char, error_capacity: usize) -> i32;
    fn dvgpu_bn254_msm(backend: *mut c_void, points: *const u8, scalars: *const u8, length: usize,
        result: *mut u8, error: *mut c_char, error_capacity: usize) -> i32;
    fn dvgpu_bn254_g2_msm(backend: *mut c_void, points: *const u8, scalars: *const u8, length: usize,
        result: *mut u8, error: *mut c_char, error_capacity: usize) -> i32;
    fn dvgpu_bn254_ntt(backend: *mut c_void, values: *const u8, length: usize, inverse: i32,
        root: *const u8, result: *mut u8, error: *mut c_char, error_capacity: usize) -> i32;
    fn dvgpu_bn254_msm_profiled(backend: *mut c_void, points: *const u8, scalars: *const u8, length: usize,
        result: *mut u8, timings: *mut NativeTimings, error: *mut c_char, error_capacity: usize) -> i32;
    fn dvgpu_bn254_msm_timeline(backend: *mut c_void, points: *const u8, scalars: *const u8, length: usize,
        result: *mut u8, label: *const c_char, timings: *mut NativeTimings, timeline: *mut MsmTimeline,
        error: *mut c_char, error_capacity: usize) -> i32;
}

/// Baseline supports the planned scan through 2^20. This also prevents
/// overflow of ICICLE's internal 32-bit bucket/index counts at larger sizes.
pub const MAX_MSM_SIZE: usize = 1 << 20;
pub const MAX_NTT_SIZE: usize = 1 << 20;
pub const MAX_MSM_BATCH_SIZE: usize = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BatchStats {
    pub icicle_calls: u64,
    pub buffer_grows: u64,
    pub max_batch_size: u64,
    pub input_capacity: u64,
}

/// Device-wide availability, including other owners/users. Diagnostic only.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MemoryInfo { pub free_bytes: u64, pub total_bytes: u64 }

// This optimization is specific to BN254 G1. Cofactor one means every
// on-curve point belongs to the prime-order group; multiplying each point by
// r adds no validation. Fail compilation if an upstream config changes this
// premise. It must not be generalized to G2 or to curves with other cofactors.
const _: () = {
    assert!(ark_bn254::g1::Config::COFACTOR.len() == 1);
    assert!(ark_bn254::g1::Config::COFACTOR[0] == 1);
};

fn check_bn254_g1_points(points: &[G1Affine]) -> Result<()> {
    for point in points { ensure!(point.is_on_curve(), "invalid G1 point"); }
    Ok(())
}

// NonNull does not implement Send/Sync. One backend belongs to one caller
// thread, preventing simultaneous mutation of its opaque native context.
#[derive(Default)]
struct BatchEncoding { points: Vec<u8>, scalars: Vec<u8>, output: Vec<u8> }

pub struct IcicleBackend {
    pub(crate) handle: NonNull<c_void>, large_bucket_factor: u32,
    pub(crate) resident_ready: bool,
    pub(crate) resident_sealed: bool,
    pub(crate) resident_limits: Option<(u64, u64)>,
    pub(crate) device_affine_readback: bool,
    pub(crate) bounded_g1_decode: bool,
    batch_encoding: RefCell<BatchEncoding>,
    pub(crate) setup_counts: Cell<(u64,u64)>,
    pub(crate) setup_host: Cell<crate::setup::SetupHostStats>,
    pub(crate) g2_mode: Cell<crate::g2_validation::G2ValidationMode>,
    pub(crate) g2_host: Cell<crate::g2_validation::G2ValidationHostStats>,
}

impl IcicleBackend {
    pub fn memory_info(&self) -> Result<MemoryInfo> {
        let mut info = MemoryInfo::default();
        let mut error = [0u8; 256];
        // SAFETY: repr(C) output and bounded error buffer remain alive during
        // the native context-scoped query; native retains no borrowed pointer.
        let status = unsafe { dvgpu_backend_memory_info(self.handle.as_ptr(), &mut info,
            error.as_mut_ptr().cast(), error.len()) };
        check_status(status, &error)?;
        Ok(info)
    }
    pub fn batch_stats(&self) -> Result<BatchStats> {
        let mut stats = BatchStats::default();
        // SAFETY: repr(C) fixed-width initialized output matches the C struct.
        let status = unsafe { dvgpu_backend_batch_stats(self.handle.as_ptr(), &mut stats) };
        ensure!(status == 0, "invalid native batch stats owner");
        Ok(stats)
    }

    /// Initializes device 0. Failures are explicit; no implicit CPU fallback.
    /// The opaque native owner holds an isolated CUDA context. Drop releases
    /// it; an execution error retires it without resetting other GPU users.
    pub fn new() -> Result<Self> {
        Self::with_large_bucket_factor(0)
    }

    /// Explicit per-owner ICICLE tuning, independent of process environment.
    /// Zero disables large buckets; 1..=100 are threshold multipliers. This
    /// does not change automatic c, full scalar bits or big-triangle reduction.
    pub fn with_large_bucket_factor(factor: u32) -> Result<Self> {
        ensure!(factor <= 100, "large bucket factor must be within 0..=100");
        let mut error = [0u8; 256];
        let mut handle = std::ptr::null_mut();
        // SAFETY: native code borrows the output pointer and error buffer only
        // for this call, writes within their bounds, and catches exceptions.
        let status = unsafe { dvgpu_backend_create_with_large_bucket_factor(factor,
            &mut handle, error.as_mut_ptr().cast(), error.len()) };
        check_status(status, &error)?;
        let handle = NonNull::new(handle).ok_or_else(|| anyhow::anyhow!("native backend returned a null owner"))?;
        Ok(Self { handle, large_bucket_factor: factor, resident_ready: false, resident_sealed: false,
            resident_limits: None, device_affine_readback: false, bounded_g1_decode: false,
            batch_encoding: RefCell::default(), setup_counts: Cell::new((0,0)),
            setup_host: Cell::default(), g2_mode:Cell::default(),g2_host:Cell::default() })
    }

    pub fn large_bucket_factor(&self) -> u32 { self.large_bucket_factor }

    pub fn device_identity(&self) -> Result<String> {
        let mut output = [0u8; 320];
        let mut error = [0u8; 256];
        // SAFETY: both writable buffers are initialized and capacity-bounded;
        // native code copies only its owned identity and retains no pointers.
        let status = unsafe { dvgpu_backend_device_identity(self.handle.as_ptr(), output.as_mut_ptr().cast(),
            output.len(), error.as_mut_ptr().cast(), error.len()) };
        check_status(status, &error)?;
        let end = output.iter().position(|&byte| byte == 0).unwrap_or(output.len());
        Ok(String::from_utf8(output[..end].to_vec())?)
    }
}

impl Drop for IcicleBackend {
    fn drop(&mut self) {
        // SAFETY: create returned this unique owner. No Clone/Send/Sync is
        // exposed, calls are synchronous, and Drop destroys it exactly once.
        unsafe { dvgpu_backend_destroy(self.handle.as_ptr()); }
    }
}

pub(crate) fn check_status(status: i32, error: &[u8]) -> Result<()> {
    if status == 0 { return Ok(()); }
    let end = error.iter().position(|&byte| byte == 0).unwrap_or(error.len());
    anyhow::bail!("ICICLE CUDA error {status}: {}", String::from_utf8_lossy(&error[..end]))
}

fn start_span(profiling: bool) -> Option<Instant> {
    if profiling { Some(Instant::now()) } else { None }
}

fn elapsed(start: Option<Instant>) -> Duration {
    match start { Some(start) => start.elapsed(), None => Duration::ZERO }
}

impl IcicleBackend {
    /// Diagnostic form of the same synchronous operation. Ordinary trait
    /// dispatch uses the original ABI and does not read diagnostic clocks.
    pub fn msm_profiled(&self, points: &[G1Affine], scalars: &[Fr]) -> Result<(G1Affine, MsmProfile)> {
        let (point, profile, _) = self.msm_impl(points, scalars, true, None)?;
        Ok((point, profile))
    }

    /// Dedicated diagnostic path. label is a borrowed NUL-terminated string;
    /// native code finishes its NVTX ranges and all CUDA work before return.
    pub fn msm_timeline(&self, points: &[G1Affine], scalars: &[Fr], label: &CStr)
        -> Result<(G1Affine, MsmProfile, MsmTimeline)> {
        self.msm_impl(points, scalars, true, Some(label))
    }

    fn msm_impl(&self, points: &[G1Affine], scalars: &[Fr], profiling: bool, label: Option<&CStr>)
        -> Result<(G1Affine, MsmProfile, MsmTimeline)> {
        let total_started = start_span(profiling);
        let mut profile = MsmProfile::default();
        let mut timeline = MsmTimeline::default();
        ensure!(points.len() == scalars.len(), "MSM length mismatch");
        ensure!(points.len() <= MAX_MSM_SIZE, "MSM exceeds baseline size bound (2^20)");
        if points.is_empty() {
            profile.total = elapsed(total_started);
            return Ok((G1Affine::identity(), profile, timeline));
        }
        let point_bytes = points.len().checked_mul(G1_BYTES)
            .ok_or_else(|| anyhow::anyhow!("point buffer size overflow"))?;
        let scalar_bytes = scalars.len().checked_mul(FR_BYTES)
            .ok_or_else(|| anyhow::anyhow!("scalar buffer size overflow"))?;
        let started = start_span(profiling);
        check_bn254_g1_points(points)?;
        profile.validation = elapsed(started);
        let started = start_span(profiling);
        let mut encoded_points = Vec::new();
        let mut encoded_scalars = Vec::new();
        // Reservation failures propagate as errors before any native dispatch.
        encoded_points.try_reserve_exact(point_bytes)?;
        encoded_scalars.try_reserve_exact(scalar_bytes)?;
        for point in points {
            encoded_points.extend_from_slice(&encode_g1(point));
        }
        for scalar in scalars { encoded_scalars.extend_from_slice(&encode_fr(scalar)); }
        profile.encoding = elapsed(started);
        let mut result = [0u8; G1_BYTES];
        let mut error = [0u8; 256];
        let mut native_timings = NativeTimings::default();
        let started = start_span(profiling);
        // SAFETY: vectors contain exactly length * 64/32 initialized bytes.
        // Native code only reads them, writes result/error within their sizes,
        // and synchronizes before returning. The vectors stay owned here.
        // The optional repr(C) timings output is initialized stack storage
        // matching the C struct; native code retains none of these pointers.
        // A CStr label stays borrowed and alive through the synchronous call.
        let status = unsafe {
            if let Some(label) = label {
                dvgpu_bn254_msm_timeline(self.handle.as_ptr(), encoded_points.as_ptr(), encoded_scalars.as_ptr(), points.len(),
                    result.as_mut_ptr(), label.as_ptr(), &mut native_timings, &mut timeline, error.as_mut_ptr().cast(), error.len())
            } else if profiling {
                dvgpu_bn254_msm_profiled(self.handle.as_ptr(), encoded_points.as_ptr(), encoded_scalars.as_ptr(), points.len(),
                    result.as_mut_ptr(), &mut native_timings, error.as_mut_ptr().cast(), error.len())
            } else {
                dvgpu_bn254_msm(self.handle.as_ptr(), encoded_points.as_ptr(), encoded_scalars.as_ptr(), points.len(),
                    result.as_mut_ptr(), error.as_mut_ptr().cast(), error.len())
            }
        };
        profile.native = elapsed(started);
        check_status(status, &error)?;
        let started = start_span(profiling);
        let point = decode_g1(&result)?;
        profile.decode = elapsed(started);
        profile.native_conversion = Duration::from_nanos(native_timings.conversion_ns);
        profile.icicle_inclusive = Duration::from_nanos(native_timings.icicle_ns);
        profile.native_output = Duration::from_nanos(native_timings.output_ns);
        // The call has synchronized, so buffers can be freed before stopping
        // total. This includes their deallocation in the diagnostic total.
        drop(encoded_points);
        drop(encoded_scalars);
        profile.total = elapsed(total_started);
        Ok((point, profile, timeline))
    }
}

impl IcicleBackend {
    // Callers validate every input before this helper. The closure writes one
    // unpadded row into owned byte slices; it cannot retain a caller borrow in
    // BatchEncoding. Both dense and indexed inputs use the same native layout.
    fn submit_encoded_batches(&self, lengths: &[usize],
        mut encode: impl FnMut(usize, &mut [u8], &mut [u8])) -> Result<Vec<G1Affine>> {
        let count = lengths.len();
        ensure!(count <= MAX_MSM_SIZE, "too many batch sets (maximum 2^20)");
        let mut bin_counts = [0usize; 21];
        for &length in lengths {
            ensure!(length <= MAX_MSM_SIZE, "MSM exceeds baseline size bound (2^20)");
            if length > 0 { bin_counts[length.next_power_of_two().trailing_zeros() as usize] += 1; }
        }
        let mut bins: [Vec<usize>; 21] = std::array::from_fn(|_| Vec::new());
        for bin in 0..21 { bins[bin].try_reserve_exact(bin_counts[bin])?; }
        for (i, &length) in lengths.iter().enumerate() {
            if length > 0 { bins[length.next_power_of_two().trailing_zeros() as usize].push(i); }
        }
        let mut results = Vec::new();
        results.try_reserve_exact(count)?;
        results.resize(count, G1Affine::identity());
        // One shared receiver, one caller thread. RefCell holds reusable owned
        // bytes, not caller borrows; its guard outlives the synchronous FFI call.
        let mut buffers = self.batch_encoding.try_borrow_mut()
            .map_err(|_| anyhow::anyhow!("batch encoding is already borrowed"))?;
        for (bin, indices) in bins.iter().enumerate() {
            let padded_length = 1usize << bin;
            let chunk_size = MAX_MSM_BATCH_SIZE.min(MAX_MSM_SIZE / padded_length);
            for group in indices.chunks(chunk_size) {
                let total = padded_length * group.len();
                buffers.points.clear();
                buffers.points.try_reserve_exact(total * G1_BYTES)?;
                buffers.points.resize(total * G1_BYTES, 0);
                buffers.scalars.clear();
                buffers.scalars.try_reserve_exact(total * FR_BYTES)?;
                buffers.scalars.resize(total * FR_BYTES, 0);
                buffers.output.clear();
                buffers.output.try_reserve_exact(group.len() * G1_BYTES)?;
                buffers.output.resize(group.len() * G1_BYTES, 0);
                for (slot, &index) in group.iter().enumerate() {
                    let start = slot * padded_length;
                    let end = start + lengths[index];
                    // Split the owned fields into disjoint mutable borrows.
                    // Padding was cleared above and is never exposed to encode.
                    let BatchEncoding { points, scalars, .. } = &mut *buffers;
                    encode(index, &mut points[start * G1_BYTES..end * G1_BYTES],
                        &mut scalars[start * FR_BYTES..end * FR_BYTES]);
                }
                let mut error = [0u8; 256];
                // SAFETY: padded canonical arrays have total * 64/32 bytes;
                // output has group.len()*64. Native owns its copies/buffers,
                // retains no borrowed pointer, and synchronizes before return.
                let status = unsafe { dvgpu_bn254_msm_batch(self.handle.as_ptr(), buffers.points.as_ptr(),
                    buffers.scalars.as_ptr(), padded_length, group.len(), buffers.output.as_mut_ptr(),
                    error.as_mut_ptr().cast(), error.len()) };
                check_status(status, &error)?;
                for (slot, &index) in group.iter().enumerate() {
                    results[index] = decode_g1(&buffers.output[slot * G1_BYTES..(slot + 1) * G1_BYTES])?;
                }
            }
        }
        Ok(results)
    }
}

impl OperatorBackend for IcicleBackend {
    fn fixed_base_g1(&self,scalars:&[Fr])->Result<Vec<G1Affine>> {self.setup_g1(scalars)}
    fn fixed_base_g2(&self,scalars:&[Fr])->Result<Vec<G2Affine>> {self.setup_g2(scalars)}
    fn msm(&self, points: &[G1Affine], scalars: &[Fr]) -> Result<G1Affine> {
        Ok(self.msm_impl(points, scalars, false, None)?.0)
    }

    fn msm_batch(&self, request: &BatchMsmRequest<'_>) -> Result<Vec<G1Affine>> {
        let count = request.point_sets.len();
        ensure!(request.offsets.len() == count && request.lengths.len() == count,
            "batch shape mismatch");
        ensure!(count <= MAX_MSM_SIZE, "too many batch sets (maximum 2^20)");
        // All ranges/points are checked before the first native group. Empty
        // sets are identity and never consume a GPU batch slot.
        for i in 0..count {
            let length = request.lengths[i];
            let end = request.offsets[i].checked_add(length)
                .ok_or_else(|| anyhow::anyhow!("scalar range overflow"))?;
            ensure!(end <= request.shared_scalars.len(), "scalar range out of bounds");
            ensure!(length <= request.point_sets[i].len(), "point range out of bounds");
            ensure!(length <= MAX_MSM_SIZE, "MSM exceeds baseline size bound (2^20)");
            check_bn254_g1_points(&request.point_sets[i][..length])?;
        }
        self.submit_encoded_batches(request.lengths, |index, points, scalars| {
            for j in 0..request.lengths[index] {
                points[j * G1_BYTES..(j + 1) * G1_BYTES]
                    .copy_from_slice(&encode_g1(&request.point_sets[index][j]));
                scalars[j * FR_BYTES..(j + 1) * FR_BYTES]
                    .copy_from_slice(&encode_fr(&request.shared_scalars[request.offsets[index] + j]));
            }
        })
    }

    fn msm_sparse_batch(&self, requests: &[SparseMsmRequest<'_>]) -> Result<Vec<G1Affine>> {
        ensure!(requests.len() <= MAX_MSM_SIZE, "too many batch sets (maximum 2^20)");
        let mut lengths = Vec::new();
        lengths.try_reserve_exact(requests.len())?;
        for request in requests {
            ensure!(request.entries.len() <= MAX_MSM_SIZE, "MSM exceeds baseline size bound (2^20)");
            request.validate()?;
            for &(index, _) in request.entries {
                ensure!(request.bases[index].is_on_curve(), "invalid G1 point");
            }
            lengths.push(request.entries.len());
        }
        // No Vec<G1Affine> gather and no intermediate Vec<Fr> expansion.
        // Selected caller data is encoded directly into retained FFI buffers.
        self.submit_encoded_batches(&lengths, |index, points, scalars| {
            let request = &requests[index];
            for (j, &(base, value)) in request.entries.iter().enumerate() {
                points[j * G1_BYTES..(j + 1) * G1_BYTES]
                    .copy_from_slice(&encode_g1(&request.bases[base]));
                scalars[j * FR_BYTES..(j + 1) * FR_BYTES].copy_from_slice(&encode_fr(&value));
            }
        })
    }

    fn gpu_capabilities(&self) -> GpuCapabilities {
        GpuCapabilities { g1_msm: true, fr_ntt: true, g2_msm: true }
    }

    fn msm_g2_host(&self, points: &[G2Affine], scalars: &[Fr]) -> Result<G2Affine> {
        ensure!(points.len() == scalars.len(), "G2 MSM length mismatch");
        ensure!(points.len() <= MAX_MSM_SIZE, "G2 MSM exceeds baseline size bound (2^20)");
        let mut bases = Vec::new();
        bases.try_reserve_exact(points.len() * 128)?;
        let mut encoded_scalars = Vec::new();
        encoded_scalars.try_reserve_exact(scalars.len() * FR_BYTES)?;
        for point in points {
            ensure!(point.is_on_curve() && point.is_in_correct_subgroup_assuming_on_curve(),
                "invalid G2 point");
            bases.extend_from_slice(&crate::resident::encode_g2(point));
        }
        for scalar in scalars { encoded_scalars.extend_from_slice(&encode_fr(scalar)); }
        let mut output = [0u8; 128];
        let mut error = [0u8; 256];
        // SAFETY: canonical x.c0/x.c1/y.c0/y.c1 and scalar arrays contain
        // length*128/32 bytes. Native synchronizes and retains no host pointer.
        let status = unsafe { dvgpu_bn254_g2_msm(self.handle.as_ptr(), bases.as_ptr(),
            encoded_scalars.as_ptr(), points.len(), output.as_mut_ptr(), error.as_mut_ptr().cast(), error.len()) };
        check_status(status, &error)?;
        if output.iter().all(|byte| *byte == 0) { return Ok(G2Affine::identity()); }
        Ok(G2Affine::deserialize_with_mode(&mut &output[..], Compress::No, Validate::Yes)?)
    }

    /// Canonical Fr, natural order, and normalized inverse match arkworks.
    /// Scratch output preserves the caller's values on native/decode failure.
    fn ntt(&self, values: &mut [Fr], inverse: bool) -> Result<()> {
        ensure!(values.len().is_power_of_two(), "NTT size is not a power of two");
        ensure!(values.len() <= MAX_NTT_SIZE, "NTT exceeds baseline size bound (2^20)");
        let bytes = values.len().checked_mul(FR_BYTES)
            .ok_or_else(|| anyhow::anyhow!("NTT buffer size overflow"))?;
        let mut input = Vec::new();
        input.try_reserve_exact(bytes)?;
        for value in values.iter() { input.extend_from_slice(&encode_fr(value)); }
        let mut output = Vec::new();
        output.try_reserve_exact(bytes)?;
        output.resize(bytes, 0);
        // Pass the CPU reference's root explicitly; this fixes orientation
        // even if ICICLE and arkworks choose different subgroup generators.
        let root = encode_fr(&Fr::get_root_of_unity(MAX_NTT_SIZE as u64)
            .ok_or_else(|| anyhow::anyhow!("missing BN254 NTT root"))?);
        let mut error = [0u8; 256];
        // SAFETY: initialized input/output contain length * 32 bytes; root
        // contains 32. The unique owner retains only its own domain storage.
        // All borrowed buffers remain alive until synchronized native return.
        let status = unsafe { dvgpu_bn254_ntt(self.handle.as_ptr(), input.as_ptr(), values.len(),
            i32::from(inverse), root.as_ptr(), output.as_mut_ptr(), error.as_mut_ptr().cast(), error.len()) };
        check_status(status, &error)?;
        let mut decoded = Vec::new();
        decoded.try_reserve_exact(values.len())?;
        for encoded in output.chunks_exact(FR_BYTES) { decoded.push(decode_fr(encoded)?); }
        values.copy_from_slice(&decoded);
        Ok(())
    }
}

#[cfg(test)]
mod point_check_tests {
    use super::*;
    use ark_bn254::{Fq, G1Projective};
    use ark_ec::{CurveGroup, Group};

    #[test]
    fn bn254_g1_check_accepts_curve_points_and_rejects_off_curve_coordinates() {
        let generator = G1Projective::generator();
        let mut valid = vec![G1Affine::identity()];
        for scalar in 1..=64u64 {
            let point = (generator * Fr::from(scalar)).into_affine();
            // Independent slow check pins equivalence at the boundary.
            assert!(point.is_on_curve() && point.is_in_correct_subgroup_assuming_on_curve());
            valid.push(point);
        }
        check_bn254_g1_points(&valid).unwrap();
        check_bn254_g1_points(&[]).unwrap();
        for (x, y) in [(0u64, 0u64), (1, 1), (2, 2), (1, 3)] {
            let invalid = G1Affine::new_unchecked(Fq::from(x), Fq::from(y));
            assert!(!invalid.is_on_curve());
            assert!(check_bn254_g1_points(&[invalid]).is_err());
        }
    }
}
