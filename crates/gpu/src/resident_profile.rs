//! Explicit diagnostic entry points. Normal RebuildBackend calls stay unchanged.
use super::*;
use std::time::Instant;

/// Separate host and CUDA-event clock domains. Event operation is an envelope,
/// including ICICLE submission gaps; it is not a sum of pure kernel busy times.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ResidentNativeProfile {
    pub native_wall_ns: u64,
    pub journal_host_ns: u64,
    pub conversion_host_ns: u64,
    pub operation_host_ns: u64,
    pub wait_host_ns: u64,
    pub event_total_ns: u64,
    pub event_journal_ns: u64,
    pub event_operation_ns: u64,
    pub event_mask: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub d2d_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ResidentCallProfile {
    pub total_ns: u64,
    pub encoding_ns: u64,
    pub ffi_ns: u64,
    pub decoding_ns: u64,
    /// Successfully decoded G1 points, counted in the shared actual decoder
    /// branches. These are host diagnostics and do not alter the native ABI.
    pub g1_full_decode_points: u64,
    pub g1_cofactor_decode_points: u64,
    pub native: ResidentNativeProfile,
}

fn elapsed(start: Instant) -> u64 { start.elapsed().as_nanos() as u64 }

extern "C" {
    fn dvgpu_resident_call_profiled(owner:*mut c_void,operation:*const NativeOp,result:*mut u64,
        profile:*mut ResidentNativeProfile,error:*mut c_char,capacity:usize)->i32;
}
const _: [();96]=[();std::mem::size_of::<ResidentNativeProfile>()];

impl IcicleBackend {
    fn profile_call(&self, op: &NativeOp) -> Result<ResidentCallProfile> {
        let start=Instant::now();let mut native=ResidentNativeProfile::default();
        let mut result=0;let mut error=[0u8;256];
        // SAFETY: C layouts are fixed-size. Encoded inputs/outputs and profile
        // live through synchronous completion; native clears its stack borrow
        // on success and every exception and retains only its own CUDA events.
        let status=unsafe {dvgpu_resident_call_profiled(self.handle.as_ptr(),op,&mut result,
            &mut native,error.as_mut_ptr().cast(),error.len())};
        check_status(status,&error)?;
        Ok(ResidentCallProfile {ffi_ns:elapsed(start),native,..Default::default()})
    }

    pub fn profile_write_points(&mut self, id: BufferId, offset: usize, values: &[G1Affine]) -> Result<ResidentCallProfile> {
        let start = Instant::now();
        let encoded = Instant::now();
        let mut bytes = Vec::with_capacity(values.len() * 64);
        for point in values {
            ensure!(point.is_on_curve(), "invalid resident G1 point");
            bytes.extend_from_slice(&encode_g1(point));
        }
        let encoding_ns = elapsed(encoded);
        let mut op = NativeOp::new(WRITE); op.kind=1; op.a=id.raw(); op.a_offset=offset as u64;
        op.length=values.len() as u64; op.input=bytes.as_ptr();
        let mut result=self.profile_call(&op)?;result.encoding_ns=encoding_ns;result.total_ns=elapsed(start);
        Ok(result)
    }

    pub fn profile_read_points(&mut self, id: BufferId, offset: usize, values: &mut [G1Affine]) -> Result<ResidentCallProfile> {
        let start=Instant::now();let mut bytes=vec![0u8;values.len()*64];
        let mut op=NativeOp::new(READ);op.kind=1;op.a=id.raw();op.a_offset=offset as u64;
        op.flags=u32::from(self.device_affine_readback);
        op.length=values.len() as u64;op.output=bytes.as_mut_ptr();
        let mut result=self.profile_call(&op)?;let decoded=Instant::now();
        let counts=self.decode_readback_points(&bytes,values)?;
        result.g1_full_decode_points=counts.full;
        result.g1_cofactor_decode_points=counts.cofactor_one;
        result.decoding_ns=elapsed(decoded);result.total_ns=elapsed(start);Ok(result)
    }

    pub fn profile_write_fields(&mut self,id:BufferId,offset:usize,values:&[Fr])->Result<ResidentCallProfile> {
        let start=Instant::now();let encoded=Instant::now();let mut bytes=Vec::with_capacity(values.len()*32);
        for value in values {bytes.extend_from_slice(&encode_fr(value));}
        let encoding_ns=elapsed(encoded);let mut op=NativeOp::new(WRITE);
        op.a=id.raw();op.a_offset=offset as u64;op.length=values.len() as u64;op.input=bytes.as_ptr();
        let mut result=self.profile_call(&op)?;result.encoding_ns=encoding_ns;result.total_ns=elapsed(start);Ok(result)
    }

    pub fn profile_read_fields(&mut self,id:BufferId,offset:usize,values:&mut[Fr])->Result<ResidentCallProfile> {
        let start=Instant::now();let mut bytes=vec![0u8;values.len()*32];let mut op=NativeOp::new(READ);
        op.a=id.raw();op.a_offset=offset as u64;op.length=values.len() as u64;op.output=bytes.as_mut_ptr();
        let mut result=self.profile_call(&op)?;let decoded=Instant::now();
        for (value,encoded) in values.iter_mut().zip(bytes.chunks_exact(32)) {*value=decode_fr(encoded)?;}
        result.decoding_ns=elapsed(decoded);result.total_ns=elapsed(start);Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn profile_transform_rows(&mut self,kind:BufferKind,src:BufferId,dst:BufferId,row_len:usize,first_row:usize,row_count:usize,inverse:bool)->Result<ResidentCallProfile> {
        let start=Instant::now();let mut op=NativeOp::new(TRANSFORM);op.kind=kind_code(kind);op.a=src.raw();op.c=dst.raw();
        op.cols=row_len as u64;op.rows=row_count as u64;op.a_offset=first_row as u64;op.flags=u32::from(inverse);
        let mut result=self.profile_call(&op)?;result.total_ns=elapsed(start);Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn profile_transpose(&mut self,kind:BufferKind,src:BufferId,dst:BufferId,rows:usize,cols:usize,start:usize,len:usize,twiddle:Option<Fr>)->Result<ResidentCallProfile> {
        let entered=Instant::now();let root=encode_fr(&twiddle.unwrap_or_default());let mut op=NativeOp::new(TRANSPOSE);
        op.kind=kind_code(kind);op.a=src.raw();op.c=dst.raw();op.rows=rows as u64;op.cols=cols as u64;
        op.a_offset=start as u64;op.length=len as u64;op.flags=u32::from(twiddle.is_some());op.input=root.as_ptr();
        let mut result=self.profile_call(&op)?;result.total_ns=elapsed(entered);Ok(result)
    }

    pub fn profile_transform_four_step(&mut self, kind: BufferKind, input: BufferId, scratch: BufferId, output: BufferId, len: usize, inverse: bool) -> Result<ResidentCallProfile> {
        let entered = Instant::now();
        let mut op = NativeOp::new(TRANSFORM_PACKET);
        op.kind = kind_code(kind); op.a = input.raw(); op.b = scratch.raw(); op.c = output.raw();
        op.length = len as u64; op.flags = u32::from(inverse);
        let mut result = self.profile_call(&op)?;
        result.total_ns = elapsed(entered);
        Ok(result)
    }
}
