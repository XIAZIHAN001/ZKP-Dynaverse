//! Explicit single-pair Miller diagnostic. No final exponentiation or protocol use.
use anyhow::{ensure, Result};
use std::ffi::{c_char,c_void};

pub const BN254_MILLER_LINES: usize = 91;
pub const BN254_MILLER_OUTPUT_BYTES: usize = (1+2*BN254_MILLER_LINES)*384;
pub const BN254_MILLER_INFINITY: u32 = 8;
const TRACE_BYTES: usize = BN254_MILLER_LINES*384;

// BN254 G1 curve membership implies subgroup membership because h=1.
// This assertion must not be generalized to G2; its full-order check is retained.
const _: () = {
    use ark_ec::CurveConfig;
    assert!(ark_bn254::g1::Config::COFACTOR.len()==1);
    assert!(ark_bn254::g1::Config::COFACTOR[0]==1);
};

#[derive(Debug,PartialEq,Eq)]
pub struct Bn254MillerProbe {
    /// Canonical non-Montgomery Fq12: final value, 91 evaluated lines, then
    /// 91 post-line accumulators. Fq12 order is c0/c1, Fq6 c0/c1/c2, Fq2 c0/c1;
    /// every Fq is LE32. Rejected pairs are entirely zero. For an omitted
    /// infinity pair only the final value is one; both traces stay zero.
    pub output: Vec<u8>,
    pub count: u32,
    /// 0 finite, 1 encoding, 2 curve, 8 infinity. Never a protocol certificate.
    pub g1_status: u32,
    /// 0 finite, 1 encoding, 2 curve, 4 subgroup, 8 infinity.
    pub g2_status: u32,
}
impl Bn254MillerProbe {
    pub fn value(&self)->&[u8] {&self.output[..384]}
    pub fn evaluated_lines(&self)->&[u8] {&self.output[384..384+TRACE_BYTES]}
    pub fn accumulations(&self)->&[u8] {&self.output[384+TRACE_BYTES..]}
}

extern "C" {
    fn dvgpu_bn254_miller_inspect(owner:*mut c_void,p:*const u8,p_bytes:usize,q:*const u8,q_bytes:usize,
        output:*mut u8,output_bytes:usize,metadata:*mut u32,metadata_count:usize,
        error:*mut c_char,error_capacity:usize)->i32;
}
impl crate::IcicleBackend {
    /// Exactly one G1 LE64 and one G2 LE128, using retained zero/flag handling.
    /// CUDA validates both points, evaluates D-twist lines and accumulates Fq12.
    /// This raw Miller value is not a pairing or a validated GT element.
    pub fn inspect_bn254_miller(&self,p:&[u8],q:&[u8])->Result<Bn254MillerProbe> {
        ensure!(p.len()==64 && q.len()==128,"BN254 Miller requires exactly one G1/G2 pair");
        let mut output=vec![0u8;BN254_MILLER_OUTPUT_BYTES];let mut metadata=[0u32;3];let mut error=[0u8;256];
        // SAFETY: disjoint exact-sized initialized buffers live through this
        // synchronous call. Native retains no pointers and publishes only after
        // successful synchronization, allocation release and stream cleanup.
        let status=unsafe{dvgpu_bn254_miller_inspect(self.handle.as_ptr(),p.as_ptr(),p.len(),q.as_ptr(),q.len(),
            output.as_mut_ptr(),output.len(),metadata.as_mut_ptr(),metadata.len(),error.as_mut_ptr().cast(),error.len())};
        crate::icicle::check_status(status,&error)?;
        ensure!(matches!(metadata[1],0|1|2|BN254_MILLER_INFINITY),"invalid Miller G1 status");
        ensure!(matches!(metadata[2],0|1|2|4|BN254_MILLER_INFINITY),"invalid Miller G2 status");
        let finite=metadata[1]==0 && metadata[2]==0;
        ensure!(metadata[0]==if finite{BN254_MILLER_LINES as u32}else{0},"invalid Miller trace count");
        Ok(Bn254MillerProbe{output,count:metadata[0],g1_status:metadata[1],g2_status:metadata[2]})
    }
}
#[cfg(test)]
mod tests;
