//! Thin resident adapter. Native owns device storage, bounds, the serial
//! blocking stream, transform domain, and touched-range undo journal. Rust
//! borrows canonical host bytes only until a synchronous FFI call returns.
use crate::codec::{decode_fr, decode_g1, encode_fr, encode_g1};
use crate::icicle::check_status;
use crate::{IcicleBackend, MAX_NTT_SIZE};
use anyhow::{ensure, Result};
use ark_bn254::{Fq, Fr, G1Affine, G2Affine};
use ark_ec::AffineRepr;
use ark_ff::{FftField, PrimeField};
use ark_serialize::{CanonicalDeserialize, Compress, Validate};
use dynaverse_protocol::incremental_rebuild::operators::{
    BufferId, BufferKind, FieldBinaryOp, RebuildBackend, RebuildStats,
};
use std::ffi::{c_char, c_void};
#[path = "resident_profile.rs"]
mod profile;
pub use profile::{ResidentCallProfile, ResidentNativeProfile};

#[derive(Default)]
struct G1ReadbackDecodeCounts { full: u64, cofactor_one: u64 }

const ALLOC: u32 = 1;
const LENGTH: u32 = 2;
const KIND: u32 = 3;
const BEGIN: u32 = 4;
const ROLLBACK: u32 = 5;
const WRITE: u32 = 6;
const READ: u32 = 7;
const COPY: u32 = 8;
const REVERSE: u32 = 9;
const TRANSPOSE: u32 = 10;
const TRANSFORM: u32 = 11;
const ZIP: u32 = 12;
const SCALE: u32 = 13;
const MSM_G1: u32 = 14;
const MSM_G2: u32 = 15;
const ALLOC_BASES:u32=16;
const TRANSFORM_PACKET: u32 = 17;

#[repr(C)]
struct NativeOp {
    opcode: u32, kind: u32, flags: u32, reserved: u32,
    a: u64, b: u64, c: u64,
    a_offset: u64, b_offset: u64, c_offset: u64, length: u64, rows: u64, cols: u64,
    input: *const u8, output: *mut u8,
}
impl NativeOp {
    fn new(opcode: u32) -> Self {
        Self { opcode, kind: 0, flags: 0, reserved: 0, a: 0, b: 0, c: 0,
            a_offset: 0, b_offset: 0, c_offset: 0, length: 0, rows: 0, cols: 0,
            input: std::ptr::null(), output: std::ptr::null_mut() }
    }
}
#[repr(C)]
#[derive(Default)]
struct NativeStats {
    h2d_bytes: u64, d2h_bytes: u64, bytes_allocated: u64,
    fr_transform_elements: u64, g1_transform_elements: u64,
    msm_g1_points: u64, msm_g2_points: u64, calls: u64,
}
/// Device-event time includes stream work submitted by a resident primitive,
/// including its journal/transfer work. ICICLE scratch allocations are counted
/// separately from sealed resident allocations. Clock totals are diagnostics.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ResidentDiagnostics {
    pub resident_allocations: u64,
    pub temporary_allocations: u64,
    pub temporary_peak_bytes: u64,
    pub journal_saved_bytes: u64,
    pub journal_saved_ranges: u64,
    pub journal_peak_bytes: u64,
    pub device_operation_ns: u64,
    pub device_operations: u64,
    pub physical_h2d_bytes: u64,
    pub physical_d2h_bytes: u64,
    pub explicit_d2d_bytes: u64,
    pub journal_metadata_capacity_bytes: u64,
}
extern "C" {
    fn dvgpu_resident_configure(owner: *mut c_void, limit: u64, journal: u64,
        root: *const u8, error: *mut c_char, capacity: usize) -> i32;
    fn dvgpu_reserve_batch(owner: *mut c_void, points: usize,
        error: *mut c_char, capacity: usize) -> i32;
    fn dvgpu_resident_call(owner: *mut c_void, operation: *const NativeOp,
        output: *mut u64, error: *mut c_char, capacity: usize) -> i32;
    fn dvgpu_resident_seal(owner: *mut c_void);
    fn dvgpu_resident_commit(owner: *mut c_void);
    fn dvgpu_resident_stats(owner: *mut c_void, output: *mut NativeStats) -> i32;
    fn dvgpu_resident_diagnostics(owner: *mut c_void, output: *mut ResidentDiagnostics) -> i32;
}
fn kind_code(kind: BufferKind) -> u32 {
    match kind { BufferKind::Fr => 0, BufferKind::G1 => 1 }
}

#[cfg(test)]
mod g1_readback_error_tests {
    use super::*;
    #[test]
    fn invalid_curve_readback_rejects_then_rolls_back_and_retries_in_each_mode() {
        let mut b=IcicleBackend::with_resident_limits(1<<30,1<<20).unwrap();
        let id=b.alloc(BufferKind::G1,1,true).unwrap();
        b.write_points(id,0,&[G1Affine::generator()]).unwrap();b.seal();
        let allocated=b.stats().bytes_allocated;
        for device in [false,true] {for bounded in [false,true] {
            b.set_device_affine_readback(device);b.set_bounded_g1_decode(bounded);
            b.begin_transaction().unwrap();
            // Raw ABI deliberately bypasses the safe upload's on-curve check
            // to model a bad device result: canonical Fq (1,3), off-curve.
            let mut invalid=[0u8;64];invalid[0]=1;invalid[32]=3;
            let mut op=NativeOp::new(WRITE);op.kind=1;op.a=id.raw();op.length=1;op.input=invalid.as_ptr();
            b.resident_call(&op).unwrap();
            let sentinel=[-G1Affine::generator()];let mut output=sentinel;
            assert!(b.read_points(id,0,&mut output).is_err());assert_eq!(output,sentinel);
            assert!(b.profile_read_points(id,0,&mut output).is_err());assert_eq!(output,sentinel);
            b.rollback_transaction().unwrap();
            let p=b.profile_read_points(id,0,&mut output).unwrap();
            assert_eq!(output,[G1Affine::generator()]);
            assert_eq!(p.g1_cofactor_decode_points,u64::from(bounded));
            assert_eq!(p.g1_full_decode_points,u64::from(!bounded));
            assert_eq!(b.stats().bytes_allocated,allocated);
        }}
    }
}

#[cfg(test)]
mod packed_ffi_tests {
    use super::*;
    #[test]
    fn raw_read_flags_reject_unknown_or_field_modes_without_publishing() {
        let mut backend = IcicleBackend::with_resident_limits(1<<30,1<<20).unwrap();
        let points = backend.alloc(BufferKind::G1,1,true).unwrap();
        let fields = backend.alloc(BufferKind::Fr,1,true).unwrap();
        for (id,kind,flags) in [(points,1,2),(fields,0,1)] {
            let mut output=[0xa5u8;64];
            let mut op=NativeOp::new(READ);op.a=id.raw();op.kind=kind;
            op.flags=flags;op.length=1;op.output=output.as_mut_ptr();
            let before=backend.resident_diagnostics().unwrap();
            assert!(backend.resident_call(&op).is_err());
            assert_eq!(output,[0xa5u8;64]);
            assert_eq!(backend.resident_diagnostics().unwrap().physical_d2h_bytes,before.physical_d2h_bytes);
        }
        backend.set_device_affine_readback(true);
        let mut output=[G1Affine::generator()];
        backend.read_points(points,0,&mut output).unwrap();
        assert_eq!(output,[G1Affine::identity()]);
    }
    #[test]
    fn raw_abi_rejects_packed_handles_for_projective_only_operations() {
        let mut backend=IcicleBackend::with_resident_limits(1<<30,1<<20).unwrap();
        let a=backend.register_msm_bases(&[G1Affine::identity();2]).unwrap();
        let c=backend.register_msm_bases(&[G1Affine::identity();2]).unwrap();
        // Zero lengths exercise raw argument validation without launching an
        // invalid-width kernel in the pre-fix RED run.
        for opcode in [TRANSPOSE,TRANSFORM,REVERSE,COPY,WRITE] {
            let mut op=NativeOp::new(opcode);op.kind=2;op.a=a.raw();op.c=c.raw();
            op.cols=1;op.rows=if opcode==TRANSFORM {0} else {1};
            assert!(backend.resident_call(&op).is_err(),"raw packed opcode {opcode} accepted");
        }
        assert!(backend.memory_info().is_ok());
    }
}
impl IcicleBackend {
    /// Opt into cofactor-one BN254 G1 decoding for reads of at most 256
    /// points. Default is false; larger reads use the full-validation oracle.
    /// Coordinate/flag decoding and the on-curve check remain mandatory.
    pub fn set_bounded_g1_decode(&mut self, enabled: bool) {
        self.bounded_g1_decode = enabled;
    }

    fn decode_readback_points(&self, bytes: &[u8], values: &mut [G1Affine]) -> Result<G1ReadbackDecodeCounts> {
        ensure!(values.len().checked_mul(64) == Some(bytes.len()), "resident G1 decode length mismatch");
        let mut counts = G1ReadbackDecodeCounts::default();
        let bounded = self.bounded_g1_decode && values.len() <= 256;
        for (value, encoded) in values.iter_mut().zip(bytes.chunks_exact(64)) {
            if bounded {
                *value = crate::codec::decode_g1_cofactor_one(encoded)?;
                counts.cofactor_one += 1;
            } else {
                *value = decode_g1(encoded)?;
                counts.full += 1;
            }
        }
        Ok(counts)
    }

    /// Opt into device normalization for reads of at most 256 projective G1
    /// points. Default is false. Larger reads retain host normalization;
    /// packed bases retain their direct affine copy. This reserves no memory:
    /// native reuses the owner-local affine workspace allocated before seal.
    pub fn set_device_affine_readback(&mut self, enabled: bool) {
        self.device_affine_readback = enabled;
    }

    /// Reserve the owner-local transaction arena and the pinned 2^20 domain
    /// before serving updates. Native also checks a conservative ICICLE scratch
    /// allowance against this cap and actual free device memory before alloc.
    pub fn with_resident_limits(memory_limit_bytes: u64, transaction_bytes: u64) -> Result<Self> {
        let mut backend = Self::new()?;
        backend.configure_resident(memory_limit_bytes, transaction_bytes)?;
        Ok(backend)
    }
    fn configure_resident(&mut self, limit: u64, journal: u64) -> Result<()> {
        let root = encode_fr(&Fr::get_root_of_unity(MAX_NTT_SIZE as u64)
            .ok_or_else(|| anyhow::anyhow!("missing resident NTT root"))?);
        let mut error = [0u8; 256];
        // SAFETY: fixed-size root/error bytes remain live until synchronous
        // native initialization finishes; native retains only its own copies.
        let status = unsafe { dvgpu_resident_configure(self.handle.as_ptr(), limit,
            journal, root.as_ptr(), error.as_mut_ptr().cast(), error.len()) };
        check_status(status, &error)?;
        self.resident_ready = true;
        self.resident_limits = Some((limit, journal));
        Ok(())
    }
    /// Configured device cap and device undo-arena size, or None before the
    /// first resident allocation on a legacy-created owner.
    pub fn resident_limits(&self) -> Option<(u64, u64)> { self.resident_limits }

    /// Call at session initialization using the unique writable-buffer bound.
    /// Exact first-write markers retain each original element at most once.
    pub fn preflight_resident_transaction(&self, required_bytes: u64) -> Result<()> {
        let (_, journal) = self.resident_limits
            .ok_or_else(|| anyhow::anyhow!("resident backend is not configured"))?;
        ensure!(required_bytes <= journal, "resident transaction requires {required_bytes} bytes but journal has {journal}");
        Ok(())
    }
    fn resident_call(&self, operation: &NativeOp) -> Result<u64> {
        let mut error = [0u8; 256];
        let mut result = 0;
        // SAFETY: repr(C) matches the header. Every byte pointer is backed by
        // the bounded owned encoding buffer in its caller. Native validates all
        // handles/ranges, synchronizes, and retains no borrowed Rust pointer.
        let status = unsafe { dvgpu_resident_call(self.handle.as_ptr(), operation,
            &mut result, error.as_mut_ptr().cast(), error.len()) };
        check_status(status, &error)?;
        Ok(result)
    }
    pub fn resident_diagnostics(&self) -> Result<ResidentDiagnostics> {
        let mut output = ResidentDiagnostics::default();
        // SAFETY: initialized repr(C) output matches the native diagnostic ABI.
        let status = unsafe { dvgpu_resident_diagnostics(self.handle.as_ptr(), &mut output) };
        ensure!(status == 0, "invalid resident diagnostics owner");
        Ok(output)
    }
}

impl RebuildBackend for IcicleBackend {
    fn register_msm_bases(&mut self,points:&[G1Affine])->Result<BufferId> {
        ensure!(!self.resident_sealed,"basis registration after seal");
        if !self.resident_ready {self.configure_resident(4u64<<30,64u64<<20)?;}
        let mut bytes=Vec::with_capacity(points.len()*64);
        for point in points {ensure!(point.is_on_curve(),"invalid G1 basis");bytes.extend_from_slice(&encode_g1(point));}
        let mut op=NativeOp::new(ALLOC_BASES);op.length=points.len() as u64;op.input=bytes.as_ptr();
        Ok(BufferId::from_raw(self.resident_call(&op)?))
    }
    fn check_write_capacity(&self,unique:u64,_repeated:u64)->Result<u64> {
        self.preflight_resident_transaction(unique)?;Ok(unique)
    }
    fn reserve_sparse_msm(&mut self,max_entries:usize,public_entries:usize)->Result<()> {
        ensure!(!self.resident_sealed,"foreground reservation after seal");
        let max_h=max_entries.checked_mul(2)
            .and_then(|n|n.checked_next_power_of_two())
            .ok_or_else(||anyhow::anyhow!("sparse capacity overflow"))?;
        // Native groups contain <=64 sets and <=2^20 padded points in total.
        let public=public_entries.checked_next_power_of_two()
            .ok_or_else(||anyhow::anyhow!("public capacity overflow"))?;
        ensure!(public<=crate::MAX_MSM_SIZE,"public MSM exceeds GPU capacity");
        let total=max_h.saturating_mul(crate::MAX_MSM_BATCH_SIZE)
            .min(crate::MAX_MSM_SIZE).max(public);
        let mut error=[0u8;256];
        // SAFETY: owner-scoped allocation only, no borrowed input pointers.
        let status=unsafe {dvgpu_reserve_batch(self.handle.as_ptr(),total,
            error.as_mut_ptr().cast(),error.len())};
        check_status(status,&error)
    }
    fn check_transaction_capacity(&self, required_bytes: u64) -> Result<()> {
        self.preflight_resident_transaction(required_bytes)
    }
    fn alloc(&mut self, kind: BufferKind, len: usize, writable: bool) -> Result<BufferId> {
        ensure!(!self.resident_sealed, "resident allocation after seal");
        if !self.resident_ready { self.configure_resident(4u64 << 30, 64u64 << 20)?; }
        let mut op = NativeOp::new(ALLOC);
        op.kind = kind_code(kind); op.length = len as u64; op.flags = u32::from(writable);
        Ok(BufferId::from_raw(self.resident_call(&op)?))
    }
    fn seal(&mut self) {
        self.resident_sealed = true;
        // SAFETY: metadata-only native operation; no allocation or CUDA work.
        unsafe { dvgpu_resident_seal(self.handle.as_ptr()); }
    }
    fn buffer_len(&self, id: BufferId) -> Result<usize> {
        let mut op = NativeOp::new(LENGTH); op.a = id.raw();
        Ok(self.resident_call(&op)? as usize)
    }
    fn buffer_kind(&self, id: BufferId) -> Result<BufferKind> {
        let mut op = NativeOp::new(KIND); op.a = id.raw();
        match self.resident_call(&op)? {
            0 => Ok(BufferKind::Fr), 1|2 => Ok(BufferKind::G1),
            _ => anyhow::bail!("unknown native resident kind"),
        }
    }
    fn begin_transaction(&mut self) -> Result<()> {
        self.resident_call(&NativeOp::new(BEGIN))?; Ok(())
    }
    fn commit_transaction(&mut self) {
        // SAFETY: only clears already-completed undo metadata; cannot fail.
        unsafe { dvgpu_resident_commit(self.handle.as_ptr()); }
    }
    fn rollback_transaction(&mut self) -> Result<()> {
        self.resident_call(&NativeOp::new(ROLLBACK))?; Ok(())
    }
    fn write_fields(&mut self, id: BufferId, offset: usize, values: &[Fr]) -> Result<()> {
        let mut bytes = Vec::with_capacity(values.len() * 32);
        for value in values { bytes.extend_from_slice(&encode_fr(value)); }
        let mut op = NativeOp::new(WRITE); op.a = id.raw(); op.a_offset = offset as u64;
        op.length = values.len() as u64; op.input = bytes.as_ptr();
        self.resident_call(&op)?; Ok(())
    }
    fn read_fields(&mut self, id: BufferId, offset: usize, values: &mut [Fr]) -> Result<()> {
        let mut bytes = vec![0u8; values.len() * 32];
        let mut op = NativeOp::new(READ); op.a = id.raw(); op.a_offset = offset as u64;
        op.length = values.len() as u64; op.output = bytes.as_mut_ptr();
        self.resident_call(&op)?;
        for (value, encoded) in values.iter_mut().zip(bytes.chunks_exact(32)) { *value = decode_fr(encoded)?; }
        Ok(())
    }
    fn write_points(&mut self, id: BufferId, offset: usize, values: &[G1Affine]) -> Result<()> {
        let mut bytes = Vec::with_capacity(values.len() * 64);
        for point in values {
            ensure!(point.is_on_curve(), "invalid resident G1 point");
            bytes.extend_from_slice(&encode_g1(point));
        }
        let mut op = NativeOp::new(WRITE); op.kind = 1; op.a = id.raw(); op.a_offset = offset as u64;
        op.length = values.len() as u64; op.input = bytes.as_ptr();
        self.resident_call(&op)?; Ok(())
    }
    fn read_points(&mut self, id: BufferId, offset: usize, values: &mut [G1Affine]) -> Result<()> {
        let mut bytes = vec![0u8; values.len() * 64];
        let mut op = NativeOp::new(READ); op.kind = 1; op.a = id.raw(); op.a_offset = offset as u64;
        op.flags = u32::from(self.device_affine_readback);
        op.length = values.len() as u64; op.output = bytes.as_mut_ptr();
        self.resident_call(&op)?;
        self.decode_readback_points(&bytes, values)?;
        Ok(())
    }
    fn copy_buffer(&mut self, src: BufferId, src_offset: usize, dst: BufferId, dst_offset: usize, len: usize) -> Result<()> {
        let mut op = NativeOp::new(COPY); op.a = src.raw(); op.c = dst.raw();
        op.a_offset = src_offset as u64; op.c_offset = dst_offset as u64; op.length = len as u64;
        self.resident_call(&op)?; Ok(())
    }
    fn reverse_copy(&mut self, kind: BufferKind, src: BufferId, src_offset: usize, dst: BufferId, dst_offset: usize, len: usize) -> Result<()> {
        let mut op = NativeOp::new(REVERSE); op.kind = kind_code(kind); op.a = src.raw(); op.c = dst.raw();
        op.a_offset = src_offset as u64; op.c_offset = dst_offset as u64; op.length = len as u64;
        self.resident_call(&op)?; Ok(())
    }
    fn transpose(&mut self, kind: BufferKind, src: BufferId, dst: BufferId, rows: usize, cols: usize, start: usize, len: usize, twiddle: Option<Fr>) -> Result<()> {
        let root = encode_fr(&twiddle.unwrap_or_default());
        let mut op = NativeOp::new(TRANSPOSE); op.kind = kind_code(kind); op.a = src.raw(); op.c = dst.raw();
        op.rows = rows as u64; op.cols = cols as u64; op.a_offset = start as u64; op.length = len as u64;
        op.flags = u32::from(twiddle.is_some()); op.input = root.as_ptr();
        self.resident_call(&op)?; Ok(())
    }
    fn transform_rows(&mut self, kind: BufferKind, src: BufferId, dst: BufferId, row_len: usize, first_row: usize, row_count: usize, inverse: bool) -> Result<()> {
        let mut op = NativeOp::new(TRANSFORM); op.kind = kind_code(kind); op.a = src.raw(); op.c = dst.raw();
        op.cols = row_len as u64; op.rows = row_count as u64; op.a_offset = first_row as u64; op.flags = u32::from(inverse);
        self.resident_call(&op)?; Ok(())
    }
    fn transform_four_step(&mut self, kind: BufferKind, input: BufferId, scratch: BufferId, output: BufferId, len: usize, inverse: bool) -> Result<()> {
        let mut op = NativeOp::new(TRANSFORM_PACKET);
        op.kind = kind_code(kind); op.a = input.raw(); op.b = scratch.raw(); op.c = output.raw();
        op.length = len as u64; op.flags = u32::from(inverse);
        self.resident_call(&op)?;
        Ok(())
    }
    fn field_zip(&mut self, operator: FieldBinaryOp, left: BufferId, left_offset: usize, right: BufferId, right_offset: usize, dst: BufferId, dst_offset: usize, len: usize) -> Result<()> {
        let mut op = NativeOp::new(ZIP);
        op.flags = match operator { FieldBinaryOp::Add => 0, FieldBinaryOp::Sub => 1, FieldBinaryOp::Mul => 2 };
        op.a = left.raw(); op.b = right.raw(); op.c = dst.raw(); op.a_offset = left_offset as u64;
        op.b_offset = right_offset as u64; op.c_offset = dst_offset as u64; op.length = len as u64;
        self.resident_call(&op)?; Ok(())
    }
    fn scale_points(&mut self, points: BufferId, point_offset: usize, scalars: BufferId, scalar_offset: usize, dst: BufferId, dst_offset: usize, len: usize) -> Result<()> {
        let mut op = NativeOp::new(SCALE); op.a = points.raw(); op.b = scalars.raw(); op.c = dst.raw();
        op.a_offset = point_offset as u64; op.b_offset = scalar_offset as u64; op.c_offset = dst_offset as u64; op.length = len as u64;
        self.resident_call(&op)?; Ok(())
    }
    fn msm_g1(&mut self, points: BufferId, point_offset: usize, scalars: BufferId, scalar_offset: usize, len: usize) -> Result<G1Affine> {
        let mut output = [0u8; 64];
        let mut op = NativeOp::new(MSM_G1); op.a = points.raw(); op.b = scalars.raw(); op.a_offset = point_offset as u64;
        op.b_offset = scalar_offset as u64; op.length = len as u64; op.output = output.as_mut_ptr();
        self.resident_call(&op)?; Ok(decode_g1(&output)?)
    }
    fn msm_g2(&mut self, points: &[G2Affine], scalars: BufferId, scalar_offset: usize) -> Result<G2Affine> {
        let mut bytes = Vec::with_capacity(points.len() * 128);
        for point in points {
            ensure!(point.is_on_curve() && point.is_in_correct_subgroup_assuming_on_curve(), "invalid resident G2 base");
            bytes.extend_from_slice(&encode_g2(point));
        }
        let mut output = [0u8; 128];
        let mut op = NativeOp::new(MSM_G2); op.b = scalars.raw(); op.b_offset = scalar_offset as u64;
        op.length = points.len() as u64; op.input = bytes.as_ptr(); op.output = output.as_mut_ptr();
        self.resident_call(&op)?;
        if output.iter().all(|byte| *byte == 0) { return Ok(G2Affine::identity()); }
        Ok(G2Affine::deserialize_with_mode(&mut &output[..], Compress::No, Validate::Yes)?)
    }
    fn stats(&self) -> RebuildStats {
        let mut output = NativeStats::default();
        // SAFETY: initialized repr(C) storage has the exact native layout. The
        // owner is live for this synchronous, host-metadata-only query.
        unsafe { dvgpu_resident_stats(self.handle.as_ptr(), &mut output); }
        RebuildStats { h2d_bytes: output.h2d_bytes, d2h_bytes: output.d2h_bytes,
            bytes_allocated: output.bytes_allocated, fr_transform_elements: output.fr_transform_elements,
            g1_transform_elements: output.g1_transform_elements, msm_g1_points: output.msm_g1_points,
            msm_g2_points: output.msm_g2_points, calls: output.calls }
    }
}
pub(crate) fn encode_g2(point: &G2Affine) -> [u8; 128] {
    let mut output = [0u8; 128];
    if let Some((x, y)) = point.xy() {
        let coordinates: [Fq; 4] = [x.c0, x.c1, y.c0, y.c1];
        for (coordinate, value) in coordinates.iter().enumerate() {
            for (limb, integer) in value.into_bigint().0.iter().enumerate() {
                let start = coordinate * 32 + limb * 8;
                output[start..start + 8].copy_from_slice(&integer.to_le_bytes());
            }
        }
    }
    output
}
