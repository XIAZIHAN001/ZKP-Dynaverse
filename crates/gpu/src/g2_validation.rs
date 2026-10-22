//! Deterministic BN254 G2 validators. Production defaults to CPU full validation;
//! explicit GPU modes validate the producer's existing device affine buffer.
use anyhow::{ensure,Result};
use ark_bn254::{Fq,Fq2,Fr,G2Affine,G2Projective};
use ark_ec::{AffineRepr,Group};
use ark_ff::{Field,PrimeField,Zero};
use ark_serialize::{CanonicalDeserialize,Compress,Validate};
use std::time::Instant;

#[derive(Clone,Copy,Debug,Default)]
pub struct CpuValidationProfile {
    pub parse_ns:u64,pub curve_ns:u64,pub subgroup_ns:u64,pub wall_ns:u64,
}

#[derive(Clone,Copy,Debug,Default,PartialEq,Eq)]
pub enum G2ValidationMode {
    #[default] CpuReference, CpuFrobenius127, CpuFrobenius63,
    GpuFullOrder, GpuFrobenius127, GpuFrobenius63,
}
impl G2ValidationMode {
    pub fn gpu_algorithm(self)->Option<u32> {
        match self {Self::GpuFullOrder=>Some(1),Self::GpuFrobenius127=>Some(2),Self::GpuFrobenius63=>Some(3),_=>None}
    }
}
#[derive(Clone,Copy,Debug,Default)]
pub struct G2ValidationHostStats {
    pub host_parse_ns:u64,pub host_curve_ns:u64,pub host_subgroup_ns:u64,pub host_publish_decode_ns:u64,
}
#[repr(C)]
#[derive(Clone,Copy,Debug,Default,PartialEq,Eq)]
pub struct G2ValidationTraffic {
    pub scalar_h2d_bytes:u64,pub point_h2d_bytes:u64,pub point_d2h_bytes:u64,pub status_d2h_bytes:u64,
}
#[repr(C)]
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct G2KernelConfig {pub threads:u32,pub schedule:u32,pub layout:u32,pub fused:u32}
impl Default for G2KernelConfig {fn default()->Self {Self{threads:128,schedule:0,layout:0,fused:0}}}
#[repr(C)]
#[derive(Clone,Copy,Debug,Default)]
pub struct G2ExperimentStats {pub repack_device_ns:u64,pub subgroup_kernel_ns:u64,pub logical_repack_bytes:u64}
#[repr(C)]
#[derive(Clone,Copy,Debug,Default)]
pub struct G2KernelResources {
    pub registers:u32,pub block_threads:u32,pub active_blocks:u32,pub max_threads_per_sm:u32,
    pub local_bytes:u64,pub shared_bytes:u64,pub normalize_registers:u32,pub normalize_active_blocks:u32,
    pub normalize_local_bytes:u64,pub normalize_shared_bytes:u64,
}
const _:()={assert!(std::mem::size_of::<G2KernelConfig>()==16);assert!(std::mem::size_of::<G2KernelResources>()==56);
    assert!(std::mem::size_of::<G2ExperimentStats>()==24);};

// Only this module constructs and consumes the certificate. It owns the exact
// immutable native output; there is no public bool/report + replaceable-bytes API.
struct ValidatedG2Chunk {bytes:Vec<u8>}
impl ValidatedG2Chunk {
    fn new(bytes:Vec<u8>,report:G2ValidationReport,algorithm:u32,n:usize)->Result<Self> {
        ensure!((1..=3).contains(&algorithm) && n<=4096 && bytes.len()==n*128,"invalid checked G2 byte length");
        ensure!(report.algorithm==algorithm && report.checked_points==n as u64 && report.invalid_points==0
            && report.failure_mask==0 && report.first_invalid==u64::MAX,"inconsistent checked G2 report");
        Ok(Self{bytes})
    }
    fn decode_validated(self)->Result<Vec<G2Affine>> {
        let mut points=Vec::with_capacity(self.bytes.len()/128);
        for bytes in self.bytes.chunks_exact(128) {points.push(parse_wire(bytes)?);}
        Ok(points)
    }
}

#[repr(C)]
#[derive(Clone,Copy,Debug,Default,PartialEq,Eq)]
pub struct G2ValidationReport {
    pub checked_points:u64,pub invalid_points:u64,pub first_invalid:u64,
    pub failure_mask:u32,pub algorithm:u32,
}
#[repr(C)]
#[derive(Clone,Copy,Debug,Default,PartialEq,Eq)]
pub struct G2ValidationStats {
    pub parse_curve_device_ns:u64,pub subgroup_device_ns:u64,pub reduce_device_ns:u64,
    pub h2d_device_ns:u64,pub d2h_device_ns:u64,pub native_wall_ns:u64,
    pub checked_points:u64,pub accepted_points:u64,
}
const _:()={assert!(std::mem::size_of::<G2ValidationReport>()==32);assert!(std::mem::size_of::<G2ValidationStats>()==64);};

#[cfg(feature="icicle")]
extern "C" {
    fn dvgpu_bn254_g2_validation_inspect(owner:*mut std::ffi::c_void,wire:*const u8,n:usize,
        algorithm:u32,masks:*mut u32,report:*mut G2ValidationReport,error:*mut std::ffi::c_char,capacity:usize)->i32;
    fn dvgpu_g2_validation_stats(owner:*mut std::ffi::c_void,stats:*mut G2ValidationStats)->i32;
    fn dvgpu_g2_validation_traffic(owner:*mut std::ffi::c_void,stats:*mut G2ValidationTraffic)->i32;
    fn dvgpu_bn254_fixed_base_g2_checked(owner:*mut std::ffi::c_void,scalars:*const u8,n:usize,
        algorithm:u32,output:*mut u8,report:*mut G2ValidationReport,error:*mut std::ffi::c_char,capacity:usize)->i32;
    fn dvgpu_g2_kernel_config_set(owner:*mut std::ffi::c_void,config:*const G2KernelConfig)->i32;
    fn dvgpu_g2_kernel_config_get(owner:*mut std::ffi::c_void,config:*mut G2KernelConfig)->i32;
    fn dvgpu_g2_kernel_resources(owner:*mut std::ffi::c_void,algorithm:u32,resources:*mut G2KernelResources)->i32;
    fn dvgpu_g2_experiment_stats(owner:*mut std::ffi::c_void,stats:*mut G2ExperimentStats)->i32;
}
#[cfg(feature="icicle")]
impl crate::IcicleBackend {
    pub fn set_g2_kernel_config(&self,config:G2KernelConfig)->Result<()> {
        // SAFETY: repr(C) metadata is borrowed synchronously, never retained.
        let status=unsafe {dvgpu_g2_kernel_config_set(self.handle.as_ptr(),&config)};
        ensure!(status==0,"invalid or retired G2 kernel config ({status})");Ok(())
    }
    pub fn g2_kernel_config(&self)->Result<G2KernelConfig> {
        let mut config=G2KernelConfig::default();
        // SAFETY: initialized repr(C) output, synchronous metadata query.
        let status=unsafe {dvgpu_g2_kernel_config_get(self.handle.as_ptr(),&mut config)};
        ensure!(status==0,"invalid G2 kernel config owner");Ok(config)
    }
    pub fn g2_kernel_resources(&self,algorithm:u32)->Result<G2KernelResources> {
        let mut output=G2KernelResources::default();
        // SAFETY: native owner scopes the runtime attribute/occupancy queries;
        // repr(C) output outlives the synchronous call. Occupancy is theoretical.
        let status=unsafe {dvgpu_g2_kernel_resources(self.handle.as_ptr(),algorithm,&mut output)};
        ensure!(status==0,"G2 kernel resource query failed ({status})");Ok(output)
    }
    pub fn g2_experiment_stats(&self)->Result<G2ExperimentStats> {
        let mut output=G2ExperimentStats::default();
        // SAFETY: initialized fixed-width repr(C) output, no retained pointers.
        let status=unsafe {dvgpu_g2_experiment_stats(self.handle.as_ptr(),&mut output)};
        ensure!(status==0,"invalid G2 experiment owner");Ok(output)
    }
    /// Status-only batched inspector; first_invalid is global to these bytes.
    /// This diagnostic report never authorizes unchecked typed-point decoding.
    pub fn inspect_g2_validation_batched(&self,bytes:&[u8],algorithm:u32)->Result<(Vec<u32>,G2ValidationReport)> {
        ensure!(bytes.len()%128==0 && (1..=3).contains(&algorithm),"invalid batched G2 input");
        let mut masks=Vec::with_capacity(bytes.len()/128);
        let mut combined=G2ValidationReport{first_invalid:u64::MAX,algorithm,..G2ValidationReport::default()};
        for (block,chunk) in bytes.chunks(4096*128).enumerate() {
            let (status,report)=self.inspect_g2_validation(chunk,algorithm)?;
            combined.checked_points+=report.checked_points;combined.invalid_points+=report.invalid_points;
            combined.failure_mask|=report.failure_mask;
            if report.invalid_points>0 {combined.first_invalid=combined.first_invalid.min(block as u64*4096+report.first_invalid);}
            masks.extend(status);
        }Ok((masks,combined))
    }
    pub fn set_g2_validation_mode(&self,mode:G2ValidationMode)->Result<()> {
        // Identity is cached even after retirement; the context-scoped memory
        // query checks that this owner is still healthy before mode mutation.
        self.memory_info()?;self.g2_mode.set(mode);Ok(())
    }
    pub fn g2_validation_mode(&self)->G2ValidationMode {self.g2_mode.get()}
    pub fn g2_validation_host_stats(&self)->G2ValidationHostStats {self.g2_host.get()}
    pub fn g2_validation_traffic(&self)->Result<G2ValidationTraffic> {
        let mut stats=G2ValidationTraffic::default();
        // SAFETY: fixed-width repr(C), borrowed initialized output only.
        let status=unsafe {dvgpu_g2_validation_traffic(self.handle.as_ptr(),&mut stats)};
        ensure!(status==0,"invalid G2 traffic owner");Ok(stats)
    }
    pub(crate) fn checked_g2_chunk(&self,scalars:&[Fr],algorithm:u32,start_index:usize)->Result<Vec<G2Affine>> {
        let started=Instant::now();let mut input=Vec::with_capacity(scalars.len()*32);
        for scalar in scalars {input.extend_from_slice(&crate::codec::encode_fr(scalar));}
        let mut host=self.setup_host.get();host.encode_ns+=started.elapsed().as_nanos() as u64;self.setup_host.set(host);
        let mut bytes=vec![0;scalars.len()*128];let mut report=G2ValidationReport::default();let mut error=[0u8;256];
        // SAFETY: <=4096 canonical scalars and exact output sizes; native stages
        // output until validation, synchronization and cleanup all succeed.
        let status=unsafe {dvgpu_bn254_fixed_base_g2_checked(self.handle.as_ptr(),input.as_ptr(),scalars.len(),
            algorithm,bytes.as_mut_ptr(),&mut report,error.as_mut_ptr().cast(),error.len())};
        if status == -5 {
            ensure!(report.first_invalid<scalars.len() as u64,"invalid G2 rejection report");
            anyhow::bail!("G2 validation rejected index {} (mask {})",start_index+report.first_invalid as usize,report.failure_mask);
        }
        crate::icicle::check_status(status,&error)?;
        let started=Instant::now();
        let points=ValidatedG2Chunk::new(bytes,report,algorithm,scalars.len())?.decode_validated()?;
        let ns=started.elapsed().as_nanos() as u64;
        let mut stats=self.g2_host.get();stats.host_publish_decode_ns+=ns;self.g2_host.set(stats);
        let mut host=self.setup_host.get();host.validate_ns+=ns;host.g2_validate_ns+=ns;self.setup_host.set(host);
        Ok(points)
    }
    pub(crate) fn cpu_g2_chunk(&self,bytes:&[u8],mode:G2ValidationMode,start_index:usize)->Result<Vec<G2Affine>> {
        let wall=Instant::now();let mut stats=self.g2_host.get();let mut points=Vec::with_capacity(bytes.len()/128);
        for (i,bytes) in bytes.chunks_exact(128).enumerate() {
            let started=Instant::now();let point=parse_wire(bytes)?;stats.host_parse_ns+=started.elapsed().as_nanos() as u64;
            let started=Instant::now();let valid=point.is_on_curve();stats.host_curve_ns+=started.elapsed().as_nanos() as u64;
            ensure!(valid,"off-curve G2 index {}",start_index+i);
            let started=Instant::now();let valid=match mode {
                G2ValidationMode::CpuReference=>subgroup_full_order(&point),
                G2ValidationMode::CpuFrobenius127=>subgroup_frobenius_127(&point),
                G2ValidationMode::CpuFrobenius63=>subgroup_frobenius_63(&point),
                _=>anyhow::bail!("GPU mode in CPU G2 decoder"),
            };
            stats.host_subgroup_ns+=started.elapsed().as_nanos() as u64;
            ensure!(valid,"non-subgroup G2 index {}",start_index+i);points.push(point);
        }
        self.g2_host.set(stats);let ns=wall.elapsed().as_nanos() as u64;
        let mut host=self.setup_host.get();host.validate_ns+=ns;host.g2_validate_ns+=ns;self.setup_host.set(host);
        Ok(points)
    }
    /// Standalone status inspection. The report does not authorize unchecked
    /// decode; no SRS or typed-point publication consumes this diagnostic API.
    pub fn inspect_g2_validation(&self,bytes:&[u8],algorithm:u32)->Result<(Vec<u32>,G2ValidationReport)> {
        ensure!(bytes.len()%128==0 && bytes.len()/128<=4096,"invalid G2 validation byte length");
        ensure!((1..=3).contains(&algorithm),"unsupported G2 validation algorithm");
        let n=bytes.len()/128;let mut masks=vec![0;n];let mut report=G2ValidationReport::default();
        let mut error=[0;256];
        // SAFETY: bounded initialized outputs match repr(C); the synchronous
        // owner call borrows all buffers only until completion and retains none.
        let status=unsafe {dvgpu_bn254_g2_validation_inspect(self.handle.as_ptr(),bytes.as_ptr(),n,
            algorithm,masks.as_mut_ptr(),&mut report,error.as_mut_ptr().cast(),error.len())};
        crate::icicle::check_status(status,&error)?;
        ensure!(report.checked_points==n as u64 && report.algorithm==algorithm,"inconsistent G2 inspection report");
        Ok((masks,report))
    }
    pub fn g2_validation_stats(&self)->Result<G2ValidationStats> {
        let mut stats=G2ValidationStats::default();
        // SAFETY: fixed-width repr(C) output is writable for the synchronous query.
        let status=unsafe {dvgpu_g2_validation_stats(self.handle.as_ptr(),&mut stats)};
        ensure!(status==0,"invalid G2 validation stats owner");Ok(stats)
    }
}

/// Strong CPU wire baseline: same canonical parsing/curve/subgroup conditions.
/// Scoped worker creation and ordered result merging are part of caller wall.
pub fn inspect_g2_cpu(bytes:&[u8],mode:G2ValidationMode,threads:usize)->Result<Vec<u32>> {
    ensure!(bytes.len()%128==0 && [1,2,4,8].contains(&threads),"invalid CPU G2 input or threads");
    let predicate:fn(&G2Affine)->bool=match mode {
        G2ValidationMode::CpuReference=>subgroup_full_order,G2ValidationMode::CpuFrobenius127=>subgroup_frobenius_127,
        G2ValidationMode::CpuFrobenius63=>subgroup_frobenius_63,_=>anyhow::bail!("GPU mode in CPU inspector"),
    };
    if bytes.is_empty() {return Ok(Vec::new());}
    let chunk_size=((bytes.len()/128+threads-1)/threads)*128;
    Ok(std::thread::scope(|scope| {
        let mut workers=Vec::new();
        for chunk in bytes.chunks(chunk_size) {workers.push(scope.spawn(move || {
            let mut result=Vec::with_capacity(chunk.len()/128);
            for bytes in chunk.chunks_exact(128) {
                let mask=match parse_wire(bytes) {Err(_)=>1,Ok(point)=>{
                    if !point.is_on_curve() {2} else if !predicate(&point) {4} else {0}
                }};result.push(mask);
            }result
        }));}
        let mut result=Vec::with_capacity(bytes.len()/128);for worker in workers {result.extend(worker.join().expect("G2 CPU worker panicked"));}result
    }))
}

/// Frozen setup wire oracle, including its all-zero infinity convention.
pub fn decode_g2_reference(bytes:&[u8])->Result<G2Affine> {
    ensure!(bytes.len()==128,"G2 point must be exactly 128 bytes");
    if bytes.iter().all(|&b|b==0) {return Ok(G2Affine::identity());}
    Ok(G2Affine::deserialize_with_mode(&mut &bytes[..],Compress::No,Validate::Yes)?)
}

fn parse_wire(bytes:&[u8])->Result<G2Affine> {
    ensure!(bytes.len()==128,"G2 point must be exactly 128 bytes");
    if bytes.iter().all(|&b|b==0) {return Ok(G2Affine::identity());}
    // No subgroup check here. This private value is not published as validated.
    Ok(G2Affine::deserialize_with_mode(&mut &bytes[..],Compress::No,Validate::No)?)
}

pub(crate) fn subgroup_full_order(point:&G2Affine)->bool {
    // Integer limbs, never Fr::from(r): that would turn r into zero. Pinned
    // arkworks 0.4 mul_bigint uses general curve arithmetic, not subgroup GLS.
    point.mul_bigint(Fr::MODULUS).is_zero()
}

const BN_SEED:u64=4965661367192848881;
fn lambda_limbs()->[u64;2] {
    let lambda=6u128*u128::from(BN_SEED)*u128::from(BN_SEED);
    [lambda as u64,(lambda>>64) as u64]
}

fn psi_constants()->(Fq2,Fq2) {
    static CONSTANTS:std::sync::OnceLock<(Fq2,Fq2)>=std::sync::OnceLock::new();
    *CONSTANTS.get_or_init(|| {
        // Divide the INTEGER q-1, without ever reducing it through Fq/Fr.
        let mut q_minus_one=Fq::MODULUS.0;q_minus_one[0]-=1;
        let divide=|divisor:u64| {
            let mut out=[0;4];let mut remainder=0u128;
            for i in (0..4).rev() {
                let value=(remainder<<64)|u128::from(q_minus_one[i]);
                out[i]=(value/u128::from(divisor)) as u64;remainder=value%u128::from(divisor);
            }
            assert_eq!(remainder,0);out
        };
        let xi=Fq2::new(Fq::from(9u64),Fq::from(1u64));
        (xi.pow(divide(3)),xi.pow(divide(2)))
    })
}

fn conjugate(value:Fq2)->Fq2 {Fq2::new(value.c0,-value.c1)}

/// Conjugate of q-Frobenius under the sextic untwist. Works on all twist
/// points, including the cofactor component. Arkworks uses Jacobian coordinates.
fn psi(point:G2Projective)->G2Projective {
    if point.is_zero() {return point;}
    let (cx,cy)=psi_constants();
    G2Projective::new_unchecked(conjugate(point.x)*cx,conjugate(point.y)*cy,conjugate(point.z))
}

/// Precondition: point is on E'(Fq^2). See MATH_AUDIT.md for both implications.
pub(crate) fn subgroup_frobenius_127(point:&G2Affine)->bool {
    psi(point.into_group())==point.mul_bigint(lambda_limbs())
}

/// The 63-bit BN-seed relation. All multipliers are unreduced integers.
pub(crate) fn subgroup_frobenius_63(point:&G2Affine)->bool {
    let a=point.mul_bigint([BN_SEED]);let b=psi(a);let c=psi(b);
    a+point.into_group()+b+c==psi(c).double()
}

/// Equivalent decomposed CPU baseline. Masks: 1 encoding, 2 curve, 4 subgroup.
/// Profiling clocks are deliberately opt-in, outside the production setup.
pub fn profile_cpu_full_order(bytes:&[u8])->Result<(Vec<u32>,CpuValidationProfile)> {
    ensure!(bytes.len()%128==0,"G2 byte length must be a multiple of 128");
    let wall=Instant::now();let mut stats=CpuValidationProfile::default();
    let mut masks=Vec::with_capacity(bytes.len()/128);
    for chunk in bytes.chunks_exact(128) {
        let start=Instant::now();let parsed=parse_wire(chunk);
        stats.parse_ns+=start.elapsed().as_nanos() as u64;
        let point=match parsed {Ok(p)=>p,Err(_)=>{masks.push(1);continue;}};
        let start=Instant::now();let on_curve=point.is_on_curve();
        stats.curve_ns+=start.elapsed().as_nanos() as u64;
        if !on_curve {masks.push(2);continue;}
        let start=Instant::now();let in_group=subgroup_full_order(&point);
        stats.subgroup_ns+=start.elapsed().as_nanos() as u64;
        masks.push(if in_group {0}else{4});
    }
    stats.wall_ns=wall.elapsed().as_nanos() as u64;Ok((masks,stats))
}
#[cfg(test)]
mod tests;
#[cfg(all(test,feature="icicle"))]
mod integration_tests;
#[cfg(all(test,feature="icicle"))]
mod tuning_tests;
#[cfg(all(test,feature="icicle"))]
mod profile;
