//! Setup-only GPU generator arrays. Bounded chunks avoid retaining the full
//! SRS on device. Each native call finishes before its Rust buffers are dropped.
use crate::{IcicleBackend, codec::encode_fr, icicle::check_status};
use anyhow::{ensure,Result};
use ark_bn254::{Fr,G1Affine,G2Affine};
use ark_serialize::{CanonicalDeserialize,Compress,Validate};
use std::ffi::{c_char,c_void};
use std::time::Instant;

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
#[repr(i32)]
pub enum SetupMode { Legacy=0, Table=1, TableBatchNormalize=2 }

/// Completed device-stage diagnostics; a later cleanup failure may retire the
/// owner with these counters preserved. Accepted outputs: setup_point_counts.
/// Events and host wall use different clocks. Empty calls do not count.
#[repr(C)]
#[derive(Clone,Copy,Debug,Default)]
pub struct SetupStats {
    pub table_builds:[u64;2], pub calls:[u64;2], pub points:[u64;2],
    pub table_bytes:u64,
    /// Stream event ranges: table, H2D, generation, normalization, D2H.
    pub device_ns:[u64;5], pub native_wall_ns:u64,
}
#[derive(Clone,Copy,Debug,Default)]
pub struct SetupHostStats {
    pub encode_ns:u64, pub validate_ns:u64,
    pub g1_validate_ns:u64, pub g2_validate_ns:u64,
}

// On-curve equals prime-order subgroup membership only for concrete BN254 G1.
// Field deserialization still rejects noncanonical coordinates with Validate::No.
// Preserve codec::decode_g1 as an independent full-validation oracle.
const _:()={
    use ark_ec::CurveConfig;
    assert!(ark_bn254::g1::Config::COFACTOR.len()==1);
    assert!(ark_bn254::g1::Config::COFACTOR[0]==1);
};
fn decode_setup_g1(bytes:&[u8])->Result<G1Affine> {
    ensure!(bytes.len()==64,"G1 point must be exactly 64 bytes");
    if bytes.iter().all(|&b|b==0) {return Ok(G1Affine::identity());}
    let point=G1Affine::deserialize_with_mode(&mut &bytes[..],Compress::No,Validate::No)?;
    ensure!(point.is_on_curve(),"invalid or off-curve G1 bytes");
    Ok(point)
}

extern "C" {
    fn dvgpu_setup_mode(owner:*mut c_void,mode:i32)->i32;
    fn dvgpu_setup_stats(owner:*mut c_void,stats:*mut SetupStats)->i32;
    fn dvgpu_bn254_fixed_base(owner:*mut c_void,scalars:*const u8,length:usize,
        g2:i32,output:*mut u8,error:*mut c_char,capacity:usize)->i32;
}

#[cfg(test)]
mod decode_tests {
    use super::*;
    use crate::codec::{encode_g1,decode_g1};
    use ark_bn254::{G1Projective,Fq};
    use ark_ec::{CurveGroup,Group};
    use ark_ff::{One,PrimeField};
    #[test]
    fn setup_g1_decoder_matches_full_validation_acceptance() {
        let mut samples=vec![vec![],vec![0;63],vec![0;64],vec![255;64]];
        for s in [Fr::from(1),Fr::from(17),-Fr::one()] {
            let point=(G1Projective::generator()*s).into_affine();
            let bytes=encode_g1(&point);samples.push(bytes.to_vec());
            for i in 0..64 {let mut bad=bytes;bad[i]^=0x80;samples.push(bad.to_vec());}
        }
        // Explicit noncanonical x=Fq modulus, canonical finite y=2.
        let mut bad=vec![0u8;64];
        for (i,limb) in Fq::MODULUS.0.iter().enumerate() {bad[8*i..8*i+8].copy_from_slice(&limb.to_le_bytes());}
        bad[32]=2;samples.push(bad);
        for bytes in samples {
            let reference=decode_g1(&bytes);let optimized=decode_setup_g1(&bytes);
            assert_eq!(optimized.is_ok(),reference.is_ok(),"different acceptance: {bytes:?}");
            if let Ok(point)=reference {assert_eq!(optimized.unwrap(),point);}
        }
    }
}

impl IcicleBackend {
    pub fn set_setup_mode(&self,mode:SetupMode)->Result<()> {
        // SAFETY: synchronous owner metadata change, enum is ABI-checked.
        let status=unsafe {dvgpu_setup_mode(self.handle.as_ptr(),mode as i32)};
        ensure!(status==0,"invalid or retired setup owner"); Ok(())
    }
    pub fn setup_stats(&self)->Result<SetupStats> {
        let mut stats=SetupStats::default();
        // SAFETY: fixed-width repr(C) output is borrowed only during this call.
        let status=unsafe {dvgpu_setup_stats(self.handle.as_ptr(),&mut stats)};
        ensure!(status==0,"invalid setup stats owner"); Ok(stats)
    }
    pub fn setup_host_stats(&self)->SetupHostStats {self.setup_host.get()}
    /// Successfully returned setup points (G1, G2), excluding verifying constants.
    pub fn setup_point_counts(&self)->(u64,u64) {self.setup_counts.get()}

    fn fixed_base_chunk(&self,scalars:&[Fr],g2:bool)->Result<Vec<u8>> {
        let started=Instant::now();
        let mut input=Vec::with_capacity(32*scalars.len());
        for scalar in scalars {input.extend_from_slice(&encode_fr(scalar));}
        let mut output=vec![0u8;(if g2 {128} else {64})*scalars.len()];
        let mut stats=self.setup_host.get();
        stats.encode_ns+=started.elapsed().as_nanos() as u64; self.setup_host.set(stats);
        let mut error=[0u8;256];
        // SAFETY: input/output lengths match the checked <=4096-point native
        // operation; canonical non-Montgomery scalars, no retained host pointers.
        let status=unsafe {dvgpu_bn254_fixed_base(self.handle.as_ptr(),input.as_ptr(),
            scalars.len(),i32::from(g2),output.as_mut_ptr(),error.as_mut_ptr().cast(),error.len())};
        check_status(status,&error)?;
        Ok(output)
    }

    pub(crate) fn setup_g1(&self,scalars:&[Fr])->Result<Vec<G1Affine>> {
        let mut points=Vec::with_capacity(scalars.len());
        for chunk in scalars.chunks(4096) {
            let bytes=self.fixed_base_chunk(chunk,false)?;
            let started=Instant::now();
            for point in bytes.chunks_exact(64) {points.push(decode_setup_g1(point)?);}
            let mut stats=self.setup_host.get();
            let ns=started.elapsed().as_nanos() as u64;
            stats.validate_ns+=ns; stats.g1_validate_ns+=ns; self.setup_host.set(stats);
        }
        let (g1,g2)=self.setup_counts.get();
        self.setup_counts.set((g1+scalars.len() as u64,g2));
        Ok(points)
    }

    pub(crate) fn setup_g2(&self,scalars:&[Fr])->Result<Vec<G2Affine>> {
        let mut points=Vec::with_capacity(scalars.len());
        let mode=self.g2_validation_mode();
        for (block,chunk) in scalars.chunks(4096).enumerate() {
            let decoded=if let Some(algorithm)=mode.gpu_algorithm() {
                self.checked_g2_chunk(chunk,algorithm,block*4096)?
            } else {
                let bytes=self.fixed_base_chunk(chunk,true)?;
                self.cpu_g2_chunk(&bytes,mode,block*4096)?
            };
            points.extend(decoded);
        }
        let (g1,g2)=self.setup_counts.get();
        self.setup_counts.set((g1,g2+scalars.len() as u64));
        Ok(points)
    }
}
