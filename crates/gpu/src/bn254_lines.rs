//! Bounded G2 Miller-line preparation diagnostic, not a pairing backend.
//! Rust owns/validates buffer shapes; CUDA C++ prepares all coefficients.
use anyhow::{ensure, Result};
use std::ffi::{c_char,c_void};

pub const MAX_BN254_LINE_ITEMS: usize = 8;
pub const BN254_LINE_COUNT: usize = 91;
pub const BN254_LINE_ITEM_BYTES: usize = BN254_LINE_COUNT*3*64;
pub const BN254_LINES_INFINITY: u32 = 8;

#[derive(Debug,PartialEq,Eq)]
pub struct Bn254LinesProbe {
    /// Point-major fixed slots: 91 line triples of canonical non-Montgomery
    /// LE Fq2 (c0,c1), in locked arkworks G2Prepared order. Unused slots zero.
    pub coefficients: Vec<u8>,
    /// 91 for a finite accepted point; zero for infinity or a rejected item.
    pub counts: Vec<u32>,
    /// 0 finite, 1 encoding, 2 curve, 4 subgroup, 8 accepted infinity.
    /// This diagnostic record is never a protocol/SRS validation certificate.
    pub statuses: Vec<u32>,
}

extern "C" {
    fn dvgpu_bn254_lines_inspect(owner:*mut c_void,points:*const u8,points_bytes:usize,
        output:*mut u8,output_bytes:usize,counts:*mut u32,counts_count:usize,
        statuses:*mut u32,statuses_count:usize,error:*mut c_char,error_capacity:usize)->i32;
}

impl crate::IcicleBackend {
    /// Prepares at most eight canonical uncompressed G2 wire points. Reuses
    /// the retained parser/curve/full-integer-order checks on the GPU, including
    /// legacy zero/flagged infinity handling. No Miller accumulation or final
    /// exponentiation runs; protocol/backend/default selections are unchanged.
    pub fn inspect_bn254_lines(&self,points:&[u8])->Result<Bn254LinesProbe> {
        ensure!(points.len()%128==0,"invalid BN254 line point length");
        let count=points.len()/128;
        ensure!(count<=MAX_BN254_LINE_ITEMS,"BN254 line batch exceeds eight points");
        let mut result=Bn254LinesProbe {coefficients:vec![0;count*BN254_LINE_ITEM_BYTES],
            counts:vec![0;count],statuses:vec![0;count]};
        let mut error=[0u8;256];
        // SAFETY: disjoint, exact-length initialized outputs and input outlive
        // this synchronous call. Native retains no pointer and publishes only
        // after stream synchronization and all device/stream cleanup succeed.
        let status=unsafe {dvgpu_bn254_lines_inspect(self.handle.as_ptr(),points.as_ptr(),points.len(),
            result.coefficients.as_mut_ptr(),result.coefficients.len(),result.counts.as_mut_ptr(),result.counts.len(),
            result.statuses.as_mut_ptr(),result.statuses.len(),error.as_mut_ptr().cast(),error.len())};
        crate::icicle::check_status(status,&error)?;
        for index in 0..count {
            let item=result.statuses[index];
            ensure!(matches!(item,0|1|2|4|BN254_LINES_INFINITY),"invalid BN254 line status");
            let expected=if item==0 {BN254_LINE_COUNT as u32} else {0};
            ensure!(result.counts[index]==expected,"invalid BN254 line count");
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests;
