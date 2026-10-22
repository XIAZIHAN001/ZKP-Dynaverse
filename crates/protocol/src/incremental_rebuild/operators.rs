//! Bounded, resident arithmetic for incremental anchor reconstruction.

use anyhow::{ensure, Context, Result};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{Field, Zero};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{BatchMsmRequest, OperatorBackend};

/// Owner-local handle. A backend must reject handles that it did not allocate.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BufferId(u64);

impl BufferId {
    /// For backend implementors; callers should retain the handle returned by `alloc`.
    pub fn from_raw(raw: u64) -> Self { Self(raw) }
    pub fn raw(self) -> u64 { self.0 }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BufferKind { Fr, G1 }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FieldBinaryOp { Add, Sub, Mul }

/// Hard bound for the optional synchronous complete-transform submission.
pub const MAX_TRANSFORM_PACKET_LEN:usize=4096;

/// Cumulative submitted work, including work in subsequently rolled-back calls.
/// Transfers use the canonical boundary widths (Fr=32, G1=64, G2=128 bytes).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RebuildStats {
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub bytes_allocated: u64,
    pub fr_transform_elements: u64,
    pub g1_transform_elements: u64,
    pub msm_g1_points: u64,
    pub msm_g2_points: u64,
    pub calls: u64,
}

/// Synchronous resident operations. Successful return means all work has finished.
///
/// All allocations occur before `seal`. A buffer with `writable=false` is writable
/// during initialization and immutable after sealing. A transaction journals only
/// touched output ranges, retaining the committed checkpoint until whole-call
/// success. `commit_transaction` only publishes completed work and cannot fail.
/// A native fatal error may make rollback fail and must poison the owning session.
pub trait RebuildBackend: OperatorBackend {
    /// Opaque immutable MSM bases. Supports msm_g1 and read_points; callers
    /// must not use this handle as a general transform/copy/scale buffer.
    /// Native backends may keep affine64; the CPU oracle uses its normal pool.
    fn register_msm_bases(&mut self, points:&[G1Affine])->Result<BufferId> {
        let id=self.alloc(BufferKind::G1,points.len(),false)?;
        self.write_points(id,0,points)?;
        Ok(id)
    }
    /// First-write implementations may admit a unique-buffer bound. Backends
    /// with a repeated-write journal keep the conservative legacy bound.
    fn check_write_capacity(&self,unique:u64,repeated:u64)->Result<u64> {
        let _=unique;
        self.check_transaction_capacity(repeated)?;Ok(repeated)
    }
    /// Reserve retained foreground batch storage at initialization. A sparse
    /// h vector can have twice max_entries nonzeros; GPU padding/grouping is
    /// backend-specific. CPU references need no retained device input buffers.
    fn reserve_sparse_msm(&mut self, _max_entries: usize, _public_entries: usize) -> Result<()> { Ok(()) }
    fn alloc(&mut self, kind: BufferKind, len: usize, writable: bool) -> Result<BufferId>;
    fn seal(&mut self);
    fn buffer_len(&self, id: BufferId) -> Result<usize>;
    fn buffer_kind(&self, id: BufferId) -> Result<BufferKind>;
    fn begin_transaction(&mut self) -> Result<()>;
    fn commit_transaction(&mut self);
    fn rollback_transaction(&mut self) -> Result<()>;
    fn write_fields(&mut self, id: BufferId, offset: usize, values: &[Fr]) -> Result<()>;
    fn read_fields(&mut self, id: BufferId, offset: usize, values: &mut [Fr]) -> Result<()>;
    fn write_points(&mut self, id: BufferId, offset: usize, values: &[G1Affine]) -> Result<()>;
    fn read_points(&mut self, id: BufferId, offset: usize, values: &mut [G1Affine]) -> Result<()>;
    fn copy_buffer(&mut self, src: BufferId, src_offset: usize, dst: BufferId, dst_offset: usize, len: usize) -> Result<()>;
    fn reverse_copy(&mut self, kind: BufferKind, src: BufferId, src_offset: usize, dst: BufferId, dst_offset: usize, len: usize) -> Result<()>;
    /// Maps source `i` to `(i % cols) * rows + i / cols`. Optional twiddle is
    /// `root^((i / cols) * (i % cols))`, multiplying Fr or acting on a G1 point.
    fn transpose(&mut self, kind: BufferKind, src: BufferId, dst: BufferId, rows: usize, cols: usize, start: usize, len: usize, twiddle: Option<Fr>) -> Result<()>;
    fn transform_rows(&mut self, kind: BufferKind, src: BufferId, dst: BufferId, row_len: usize, first_row: usize, row_count: usize, inverse: bool) -> Result<()>;
    /// Complete natural-order FFT with a bounded input and distinct device
    /// buffers. Default uses the existing five stages; native backends may
    /// queue them together but must complete before returning success.
    fn transform_four_step(&mut self,kind:BufferKind,input:BufferId,scratch:BufferId,
        output:BufferId,len:usize,inverse:bool)->Result<()> {
        ensure!(len.is_power_of_two() && len<=MAX_TRANSFORM_PACKET_LEN,"transform packet length outside1..4096 power-of-two bound");
        let rows=1usize<<(len.trailing_zeros()/2);let cols=len/rows;
        let mut plan=FourStepPlan::new(kind,len,input,scratch,output,len,rows.max(cols),inverse)?;
        // Zero-length self copies validate both outputs' write permissions
        // before the first real stage can modify either one.
        self.copy_buffer(scratch,0,scratch,0,0)?;
        self.copy_buffer(output,0,output,0,0)?;
        while !plan.next_step(self)? {}
        Ok(())
    }
    fn field_zip(&mut self, op: FieldBinaryOp, left: BufferId, left_offset: usize, right: BufferId, right_offset: usize, dst: BufferId, dst_offset: usize, len: usize) -> Result<()>;
    fn scale_points(&mut self, points: BufferId, point_offset: usize, scalars: BufferId, scalar_offset: usize, dst: BufferId, dst_offset: usize, len: usize) -> Result<()>;
    fn msm_g1(&mut self, points: BufferId, point_offset: usize, scalars: BufferId, scalar_offset: usize, len: usize) -> Result<G1Affine>;
    fn msm_g2(&mut self, points: &[G2Affine], scalars: BufferId, scalar_offset: usize) -> Result<G2Affine>;
    fn stats(&self) -> RebuildStats;
    /// Native backends preflight their fixed device undo arena at session
    /// initialization. CPU references have a growable host journal.
    fn check_transaction_capacity(&self, _required_bytes: u64) -> Result<()> { Ok(()) }
}

enum BufferData {
    Fields(Vec<Fr>),
    Points(Vec<G1Affine>),
}

struct CpuBuffer {
    data: BufferData,
    writable: bool,
    // Allocated with the buffer. Begin never scans or clones candidate-sized data.
    saved: Vec<bool>,
}

enum UndoEntry {
    Field { buffer: usize, index: usize, old: Fr },
    Point { buffer: usize, index: usize, old: G1Affine },
}

/// Arkworks oracle with a preallocated buffer pool and a touched-entry undo log.
/// CPU row temporaries are bounded by the submitted row size, never a full FFT.
pub struct CpuRebuildBackend {
    owner: u64,
    buffers: Vec<CpuBuffer>,
    sealed: bool,
    in_transaction: bool,
    undo: Vec<UndoEntry>,
    stats: RebuildStats,
}

impl Default for CpuRebuildBackend {
    fn default() -> Self { Self::new() }
}

impl CpuRebuildBackend {
    pub fn new() -> Self {
        static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);
        Self {
            owner: NEXT_OWNER.fetch_add(1, Ordering::Relaxed),
            buffers: Vec::new(), sealed: false, in_transaction: false,
            undo: Vec::new(), stats: RebuildStats::default(),
        }
    }

    fn index(&self, id: BufferId) -> Result<usize> {
        ensure!(id.raw() >> 32 == self.owner, "buffer belongs to another owner");
        let index = usize::try_from(id.raw() & u32::MAX as u64).context("buffer handle overflow")?;
        ensure!(index < self.buffers.len(), "unknown buffer handle");
        Ok(index)
    }

    fn validate(&self, id: BufferId, kind: BufferKind, offset: usize, len: usize, write: bool) -> Result<usize> {
        let index = self.index(id)?;
        let buffer = &self.buffers[index];
        let (actual_kind, actual_len) = match &buffer.data {
            BufferData::Fields(values) => (BufferKind::Fr, values.len()),
            BufferData::Points(values) => (BufferKind::G1, values.len()),
        };
        ensure!(actual_kind == kind, "buffer kind mismatch");
        let end = offset.checked_add(len).context("buffer range overflow")?;
        ensure!(end <= actual_len, "buffer range out of bounds");
        ensure!(!write || !self.sealed || buffer.writable, "buffer is sealed read-only");
        Ok(index)
    }

    fn validate_alias(src: BufferId, src_offset: usize, dst: BufferId, dst_offset: usize, len: usize, exact_allowed: bool) -> Result<()> {
        if src != dst || len == 0 { return Ok(()); }
        let source_end = src_offset.checked_add(len).context("source range overflow")?;
        let destination_end = dst_offset.checked_add(len).context("destination range overflow")?;
        let overlaps = src_offset < destination_end && dst_offset < source_end;
        ensure!(!overlaps || (exact_allowed && src_offset == dst_offset), "overlapping buffer alias");
        Ok(())
    }

    fn field(&self, buffer: usize, index: usize) -> Fr {
        match &self.buffers[buffer].data {
            BufferData::Fields(values) => values[index],
            BufferData::Points(_) => unreachable!("validated Fr buffer"),
        }
    }

    fn point(&self, buffer: usize, index: usize) -> G1Affine {
        match &self.buffers[buffer].data {
            BufferData::Points(values) => values[index],
            BufferData::Fields(_) => unreachable!("validated G1 buffer"),
        }
    }

    fn put_field(&mut self, buffer: usize, index: usize, value: Fr) {
        if self.in_transaction && !self.buffers[buffer].saved[index] {
            self.undo.push(UndoEntry::Field { buffer, index, old: self.field(buffer, index) });
            self.buffers[buffer].saved[index] = true;
        }
        match &mut self.buffers[buffer].data {
            BufferData::Fields(values) => values[index] = value,
            BufferData::Points(_) => unreachable!("validated Fr buffer"),
        }
    }

    fn put_point(&mut self, buffer: usize, index: usize, value: G1Affine) {
        if self.in_transaction && !self.buffers[buffer].saved[index] {
            self.undo.push(UndoEntry::Point { buffer, index, old: self.point(buffer, index) });
            self.buffers[buffer].saved[index] = true;
        }
        match &mut self.buffers[buffer].data {
            BufferData::Points(values) => values[index] = value,
            BufferData::Fields(_) => unreachable!("validated G1 buffer"),
        }
    }

    #[cfg(test)]
    fn transaction_journal_len(&self) -> usize { self.undo.len() }
}

impl OperatorBackend for CpuRebuildBackend {
    fn msm(&self, points: &[G1Affine], scalars: &[Fr]) -> Result<G1Affine> {
        crate::dynaverse::ArkReferenceBackend.msm(points, scalars)
    }

    fn msm_batch(&self, request: &BatchMsmRequest<'_>) -> Result<Vec<G1Affine>> {
        crate::dynaverse::ArkReferenceBackend.msm_batch(request)
    }

    fn ntt(&self, values: &mut [Fr], inverse: bool) -> Result<()> {
        crate::dynaverse::ArkReferenceBackend.ntt(values, inverse)
    }
}

impl RebuildBackend for CpuRebuildBackend {
    fn check_write_capacity(&self,unique:u64,_repeated:u64)->Result<u64> {Ok(unique)}
    fn alloc(&mut self, kind: BufferKind, len: usize, writable: bool) -> Result<BufferId> {
        ensure!(!self.sealed, "buffer allocation after seal");
        ensure!(!self.in_transaction, "buffer allocation in transaction");
        ensure!(self.owner <= u32::MAX as u64 && self.buffers.len() <= u32::MAX as usize, "CPU buffer handle space exhausted");
        let old_entries: usize = self.buffers.iter().map(|buffer| buffer.saved.len()).sum();
        let entries = old_entries.checked_add(len).context("buffer pool size overflow")?;
        let width = match kind { BufferKind::Fr => std::mem::size_of::<Fr>(), BufferKind::G1 => std::mem::size_of::<G1Affine>() };
        let bytes = len.checked_mul(width).context("buffer byte size overflow")?;
        let bytes = u64::try_from(bytes).context("buffer byte size overflow")?;
        let allocated = self.stats.bytes_allocated.checked_add(bytes).context("allocation statistics overflow")?;
        // Reserve once at initialization: each pool entry can enter the journal at most once.
        self.undo.try_reserve(entries.saturating_sub(self.undo.len())).context("cannot reserve CPU undo journal")?;
        let data = match kind {
            BufferKind::Fr => {
                let mut values = Vec::new();
                values.try_reserve_exact(len).context("cannot allocate Fr buffer")?;
                values.resize(len, Fr::zero());
                BufferData::Fields(values)
            }
            BufferKind::G1 => {
                let mut values = Vec::new();
                values.try_reserve_exact(len).context("cannot allocate G1 buffer")?;
                values.resize(len, G1Affine::zero());
                BufferData::Points(values)
            }
        };
        let mut saved = Vec::new();
        saved.try_reserve_exact(len).context("cannot allocate journal markers")?;
        saved.resize(len, false);
        let handle = BufferId::from_raw((self.owner << 32) | self.buffers.len() as u64);
        self.buffers.push(CpuBuffer { data, writable, saved });
        self.stats.bytes_allocated = allocated;
        self.stats.calls += 1;
        Ok(handle)
    }

    fn seal(&mut self) { self.sealed = true; }

    fn buffer_len(&self, id: BufferId) -> Result<usize> {
        let index = self.index(id)?;
        Ok(self.buffers[index].saved.len())
    }

    fn buffer_kind(&self, id: BufferId) -> Result<BufferKind> {
        let index = self.index(id)?;
        Ok(match self.buffers[index].data { BufferData::Fields(_) => BufferKind::Fr, BufferData::Points(_) => BufferKind::G1 })
    }

    fn begin_transaction(&mut self) -> Result<()> {
        ensure!(!self.in_transaction, "transaction already active");
        self.in_transaction = true;
        self.stats.calls += 1;
        Ok(())
    }

    fn commit_transaction(&mut self) {
        for entry in &self.undo {
            let (buffer, index) = match entry { UndoEntry::Field { buffer, index, .. } | UndoEntry::Point { buffer, index, .. } => (*buffer, *index) };
            self.buffers[buffer].saved[index] = false;
        }
        self.undo.clear();
        self.in_transaction = false;
        self.stats.calls += 1;
    }

    fn rollback_transaction(&mut self) -> Result<()> {
        ensure!(self.in_transaction, "no transaction to roll back");
        // Pop only touched entries. No full-buffer clearing, cloning or checkpoint scan.
        while let Some(entry) = self.undo.pop() {
            match entry {
                UndoEntry::Field { buffer, index, old } => {
                    if let BufferData::Fields(values) = &mut self.buffers[buffer].data { values[index] = old; }
                    self.buffers[buffer].saved[index] = false;
                }
                UndoEntry::Point { buffer, index, old } => {
                    if let BufferData::Points(values) = &mut self.buffers[buffer].data { values[index] = old; }
                    self.buffers[buffer].saved[index] = false;
                }
            }
        }
        self.in_transaction = false;
        self.stats.calls += 1;
        Ok(())
    }

    fn write_fields(&mut self, id: BufferId, offset: usize, values: &[Fr]) -> Result<()> {
        let buffer = self.validate(id, BufferKind::Fr, offset, values.len(), true)?;
        for (i, value) in values.iter().enumerate() { self.put_field(buffer, offset + i, *value); }
        self.stats.h2d_bytes += values.len() as u64 * 32;
        self.stats.calls += 1;
        Ok(())
    }

    fn read_fields(&mut self, id: BufferId, offset: usize, values: &mut [Fr]) -> Result<()> {
        let buffer = self.validate(id, BufferKind::Fr, offset, values.len(), false)?;
        for (i, value) in values.iter_mut().enumerate() { *value = self.field(buffer, offset + i); }
        self.stats.d2h_bytes += values.len() as u64 * 32;
        self.stats.calls += 1;
        Ok(())
    }

    fn write_points(&mut self, id: BufferId, offset: usize, values: &[G1Affine]) -> Result<()> {
        let buffer = self.validate(id, BufferKind::G1, offset, values.len(), true)?;
        for (i, value) in values.iter().enumerate() { self.put_point(buffer, offset + i, *value); }
        self.stats.h2d_bytes += values.len() as u64 * 64;
        self.stats.calls += 1;
        Ok(())
    }

    fn read_points(&mut self, id: BufferId, offset: usize, values: &mut [G1Affine]) -> Result<()> {
        let buffer = self.validate(id, BufferKind::G1, offset, values.len(), false)?;
        for (i, value) in values.iter_mut().enumerate() { *value = self.point(buffer, offset + i); }
        self.stats.d2h_bytes += values.len() as u64 * 64;
        self.stats.calls += 1;
        Ok(())
    }

    fn copy_buffer(&mut self, src: BufferId, src_offset: usize, dst: BufferId, dst_offset: usize, len: usize) -> Result<()> {
        let kind = self.buffer_kind(src)?;
        let source = self.validate(src, kind, src_offset, len, false)?;
        let destination = self.validate(dst, kind, dst_offset, len, true)?;
        Self::validate_alias(src, src_offset, dst, dst_offset, len, true)?;
        for i in 0..len {
            match kind {
                BufferKind::Fr => self.put_field(destination, dst_offset + i, self.field(source, src_offset + i)),
                BufferKind::G1 => self.put_point(destination, dst_offset + i, self.point(source, src_offset + i)),
            }
        }
        self.stats.calls += 1;
        Ok(())
    }

    fn reverse_copy(&mut self, kind: BufferKind, src: BufferId, src_offset: usize, dst: BufferId, dst_offset: usize, len: usize) -> Result<()> {
        let source = self.validate(src, kind, src_offset, len, false)?;
        let destination = self.validate(dst, kind, dst_offset, len, true)?;
        Self::validate_alias(src, src_offset, dst, dst_offset, len, len <= 1)?;
        for i in 0..len {
            match kind {
                BufferKind::Fr => self.put_field(destination, dst_offset + i, self.field(source, src_offset + len - 1 - i)),
                BufferKind::G1 => self.put_point(destination, dst_offset + i, self.point(source, src_offset + len - 1 - i)),
            }
        }
        self.stats.calls += 1;
        Ok(())
    }

    fn transpose(&mut self, kind: BufferKind, src: BufferId, dst: BufferId, rows: usize, cols: usize, start: usize, len: usize, twiddle: Option<Fr>) -> Result<()> {
        ensure!(rows > 0 && cols > 0, "transpose dimensions must be positive");
        let size = rows.checked_mul(cols).context("transpose size overflow")?;
        let end = start.checked_add(len).context("transpose range overflow")?;
        ensure!(end <= size, "transpose range out of bounds");
        let source = self.validate(src, kind, 0, size, false)?;
        let destination = self.validate(dst, kind, 0, size, true)?;
        ensure!(src != dst || len == 0, "transpose requires distinct buffers");
        for i in start..end {
            let row = i / cols;
            let column = i % cols;
            let target = column * rows + row;
            let factor = match twiddle { Some(root) => root.pow([(row * column) as u64]), None => Fr::from(1u64) };
            match kind {
                BufferKind::Fr => self.put_field(destination, target, self.field(source, i) * factor),
                BufferKind::G1 => {
                    let point = self.point(source, i);
                    let result = if twiddle.is_some() { (point * factor).into_affine() } else { point };
                    self.put_point(destination, target, result);
                }
            }
        }
        self.stats.calls += 1;
        Ok(())
    }

    fn transform_rows(&mut self, kind: BufferKind, src: BufferId, dst: BufferId, row_len: usize, first_row: usize, row_count: usize, inverse: bool) -> Result<()> {
        ensure!(row_len.is_power_of_two(), "transform row length must be a power of two");
        let start = first_row.checked_mul(row_len).context("row range overflow")?;
        let count = row_count.checked_mul(row_len).context("row count overflow")?;
        let source = self.validate(src, kind, start, count, false)?;
        let destination = self.validate(dst, kind, start, count, true)?;
        ensure!(src != dst || count == 0, "row transform requires distinct buffers");
        let domain = Radix2EvaluationDomain::<Fr>::new(row_len).context("unsupported transform root order")?;
        for row in 0..row_count {
            let offset = start + row * row_len;
            match kind {
                BufferKind::Fr => {
                    let mut values = Vec::with_capacity(row_len);
                    for i in 0..row_len { values.push(self.field(source, offset + i)); }
                    self.ntt(&mut values, inverse)?;
                    for (i, value) in values.iter().enumerate() { self.put_field(destination, offset + i, *value); }
                }
                BufferKind::G1 => {
                    let mut values = Vec::with_capacity(row_len);
                    for i in 0..row_len { values.push(self.point(source, offset + i).into_group()); }
                    if inverse { domain.ifft_in_place(&mut values); } else { domain.fft_in_place(&mut values); }
                    // Normalize this row only; no whole-transform affine conversion.
                    let affine = G1Projective::normalize_batch(&values);
                    for (i, value) in affine.iter().enumerate() { self.put_point(destination, offset + i, *value); }
                }
            }
        }
        match kind {
            BufferKind::Fr => self.stats.fr_transform_elements += count as u64,
            BufferKind::G1 => self.stats.g1_transform_elements += count as u64,
        }
        self.stats.calls += 1;
        Ok(())
    }

    fn field_zip(&mut self, op: FieldBinaryOp, left: BufferId, left_offset: usize, right: BufferId, right_offset: usize, dst: BufferId, dst_offset: usize, len: usize) -> Result<()> {
        let left_buffer = self.validate(left, BufferKind::Fr, left_offset, len, false)?;
        let right_buffer = self.validate(right, BufferKind::Fr, right_offset, len, false)?;
        let destination = self.validate(dst, BufferKind::Fr, dst_offset, len, true)?;
        Self::validate_alias(left, left_offset, dst, dst_offset, len, true)?;
        Self::validate_alias(right, right_offset, dst, dst_offset, len, true)?;
        for i in 0..len {
            let a = self.field(left_buffer, left_offset + i);
            let b = self.field(right_buffer, right_offset + i);
            let result = match op { FieldBinaryOp::Add => a + b, FieldBinaryOp::Sub => a - b, FieldBinaryOp::Mul => a * b };
            self.put_field(destination, dst_offset + i, result);
        }
        self.stats.calls += 1;
        Ok(())
    }

    fn scale_points(&mut self, points: BufferId, point_offset: usize, scalars: BufferId, scalar_offset: usize, dst: BufferId, dst_offset: usize, len: usize) -> Result<()> {
        let source = self.validate(points, BufferKind::G1, point_offset, len, false)?;
        let factors = self.validate(scalars, BufferKind::Fr, scalar_offset, len, false)?;
        let destination = self.validate(dst, BufferKind::G1, dst_offset, len, true)?;
        Self::validate_alias(points, point_offset, dst, dst_offset, len, true)?;
        for i in 0..len {
            let result = (self.point(source, point_offset + i) * self.field(factors, scalar_offset + i)).into_affine();
            self.put_point(destination, dst_offset + i, result);
        }
        self.stats.calls += 1;
        Ok(())
    }

    fn msm_g1(&mut self, points: BufferId, point_offset: usize, scalars: BufferId, scalar_offset: usize, len: usize) -> Result<G1Affine> {
        let source = self.validate(points, BufferKind::G1, point_offset, len, false)?;
        let factors = self.validate(scalars, BufferKind::Fr, scalar_offset, len, false)?;
        // Both slices are borrowed only during the synchronous reference call.
        let point_slice = match &self.buffers[source].data { BufferData::Points(values) => &values[point_offset..point_offset + len], _ => unreachable!() };
        let scalar_slice = match &self.buffers[factors].data { BufferData::Fields(values) => &values[scalar_offset..scalar_offset + len], _ => unreachable!() };
        let result = self.msm(point_slice, scalar_slice)?;
        self.stats.msm_g1_points += len as u64;
        self.stats.d2h_bytes += 64;
        self.stats.calls += 1;
        Ok(result)
    }

    fn msm_g2(&mut self, points: &[G2Affine], scalars: BufferId, scalar_offset: usize) -> Result<G2Affine> {
        let factors = self.validate(scalars, BufferKind::Fr, scalar_offset, points.len(), false)?;
        let scalar_slice = match &self.buffers[factors].data { BufferData::Fields(values) => &values[scalar_offset..scalar_offset + points.len()], _ => unreachable!() };
        let result = G2Projective::msm_unchecked(points, scalar_slice).into_affine();
        self.stats.msm_g2_points += points.len() as u64;
        self.stats.h2d_bytes += points.len() as u64 * 128;
        self.stats.d2h_bytes += 128;
        self.stats.calls += 1;
        Ok(result)
    }

    fn stats(&self) -> RebuildStats { self.stats }
}

/// Cheaply cloned cursor for the natural-order four-step FFT/ECNTT.
/// A caller keeps its committed clone until the encompassing transaction commits.
#[derive(Clone, Debug)]
pub struct FourStepPlan {
    kind: BufferKind,
    len: usize,
    a: usize,
    b: usize,
    input: BufferId,
    scratch: BufferId,
    output: BufferId,
    tile_len: usize,
    row_batch: usize,
    inverse: bool,
    root: Fr,
    phase: u8,
    cursor: usize,
    total_steps: usize,
}

impl FourStepPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(kind: BufferKind, len: usize, input: BufferId, scratch: BufferId, output: BufferId, tile_len: usize, row_batch: usize, inverse: bool) -> Result<Self> {
        ensure!(len.is_power_of_two(), "four-step length must be a power of two");
        ensure!(tile_len > 0 && row_batch > 0, "four-step block bounds must be positive");
        ensure!(input != scratch && input != output && scratch != output, "four-step buffers must be distinct");
        let domain = Radix2EvaluationDomain::<Fr>::new(len).context("unsupported four-step root order")?;
        let a = 1usize << (len.trailing_zeros() / 2);
        let b = len / a;
        let tiles = len.div_ceil(tile_len);
        let total_steps = tiles.checked_mul(3)
            .and_then(|count| count.checked_add(a.div_ceil(row_batch)))
            .and_then(|count| count.checked_add(b.div_ceil(row_batch)))
            .context("four-step plan size overflow")?;
        Ok(Self {
            kind, len, a, b, input, scratch, output, tile_len, row_batch,
            inverse, root: if inverse { domain.group_gen_inv } else { domain.group_gen },
            phase: 0, cursor: 0, total_steps,
        })
    }

    pub fn total_steps(&self) -> usize { self.total_steps }
    pub fn is_complete(&self) -> bool { self.phase == 5 }
    pub fn dimensions(&self) -> (usize, usize) { (self.a, self.b) }

    pub fn execute_packet<B:RebuildBackend+?Sized>(&self,backend:&mut B)->Result<()> {
        ensure!(self.phase==0 && self.cursor==0,"transform packet must start at an untouched plan");
        backend.transform_four_step(self.kind,self.input,self.scratch,self.output,self.len,self.inverse)
    }

    /// Initialization has no per-update latency budget. Batch contiguous rows
    /// while capping each submission at 65536 elements. Rebuild cursors keep
    /// their original small bounds and their original work certificate.
    pub(crate) fn initialization_plan(&self) -> Result<Self> {
        let cap=65536usize;
        Self::new(self.kind,self.len,self.input,self.scratch,self.output,
            self.len.min(cap),(cap/self.a.max(self.b)).max(1),self.inverse)
    }

    /// Execute one tile or row batch. Errors leave this cursor unchanged.
    /// The final real operation returns true; no extra completion operation exists.
    pub fn next_step<B: RebuildBackend + ?Sized>(&mut self, backend: &mut B) -> Result<bool> {
        if self.is_complete() { return Ok(true); }
        // Validate the entire plan before its first mutation, including the root's
        // full order rather than merely checking the shorter row transforms.
        if self.phase == 0 && self.cursor == 0 {
            for buffer in [self.input, self.scratch, self.output] {
                ensure!(backend.buffer_kind(buffer)? == self.kind, "four-step buffer kind mismatch");
                ensure!(backend.buffer_len(buffer)? >= self.len, "four-step buffer too short");
            }
        }
        let phase_len = match self.phase { 0 | 2 | 4 => self.len, 1 => self.b, 3 => self.a, _ => unreachable!() };
        let block = if self.phase == 1 || self.phase == 3 { self.row_batch } else { self.tile_len };
        let count = block.min(phase_len - self.cursor);
        match self.phase {
            0 => backend.transpose(self.kind, self.input, self.output, self.a, self.b, self.cursor, count, None)?,
            1 => backend.transform_rows(self.kind, self.output, self.scratch, self.a, self.cursor, count, self.inverse)?,
            2 => backend.transpose(self.kind, self.scratch, self.output, self.b, self.a, self.cursor, count, Some(self.root))?,
            3 => backend.transform_rows(self.kind, self.output, self.scratch, self.b, self.cursor, count, self.inverse)?,
            4 => backend.transpose(self.kind, self.scratch, self.output, self.a, self.b, self.cursor, count, None)?,
            _ => unreachable!(),
        }
        self.cursor += count;
        if self.cursor == phase_len { self.phase += 1; self.cursor = 0; }
        Ok(self.is_complete())
    }
}

#[cfg(test)]
#[path = "operators_tests.rs"]
mod tests;
