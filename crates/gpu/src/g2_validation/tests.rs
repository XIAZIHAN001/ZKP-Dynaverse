use super::*;
use ark_bn254::{Fq,Fq2,Fr,G2Affine,G2Projective};
use ark_ec::{AffineRepr,CurveGroup,Group,CurveConfig};
use ark_ff::{PrimeField,UniformRand};
use ark_serialize::{CanonicalDeserialize,Compress,Validate};
use rand::{rngs::StdRng,SeedableRng};
// Test artifacts must follow the same explicit output directory as the probes.
// In immutable remote deliveries this points outside the source tree.
#[path="../../tests/support/g2_artifacts.rs"]
mod test_artifacts;

fn old_decoder(bytes:&[u8])->anyhow::Result<G2Affine> {
    anyhow::ensure!(bytes.len()==128,"G2 length");
    if bytes.iter().all(|&b|b==0) {return Ok(G2Affine::identity());}
    Ok(G2Affine::deserialize_with_mode(&mut &bytes[..],Compress::No,Validate::Yes)?)
}
pub(super) fn wire(point:G2Affine)->[u8;128] {
    let mut out=[0;128];
    if point.is_zero() {return out;}
    let fields=[point.x.c0,point.x.c1,point.y.c0,point.y.c1];
    for j in 0..4 {for i in 0..4 {out[j*32+i*8..j*32+i*8+8].copy_from_slice(&fields[j].into_bigint().0[i].to_le_bytes());}}
    out
}
pub(super) fn twist_points(count:usize)->Vec<G2Affine> {
    let mut rng=StdRng::seed_from_u64(1720001);
    let mut points=Vec::new();
    while points.len()<count {
        if let Some(p)=G2Affine::get_point_from_x_unchecked(Fq2::rand(&mut rng),false) {points.push(p);}
    }
    points
}
pub(super) fn acceptance_corpus()->Vec<[u8;128]> {
    let mut out=vec![[0;128]];
    for p in [G2Affine::generator(),(-G2Projective::generator()).into_affine()] {
        let b=wire(p);out.push(b);
        for flag in [0,0x40,0x80,0xc0] {let mut changed=b;changed[127]|=flag;out.push(changed);}
        for i in 0..128 {let mut changed=b;changed[i]^=0x80;out.push(changed);}
    }
    for flag in [0x40,0x80,0xc0] {let mut b=[0;128];b[127]=flag;out.push(b);}
    let q=Fq::MODULUS;
    for field in 0..4 {for adjustment in [-1i8,0,1] {
        let mut qbytes=[0;32];
        for i in 0..4 {qbytes[8*i..8*i+8].copy_from_slice(&q.0[i].to_le_bytes());}
        qbytes[0]=(i16::from(qbytes[0])+i16::from(adjustment)) as u8;
        let mut b=wire(G2Affine::generator());b[32*field..32*field+32].copy_from_slice(&qbytes);out.push(b);
    }}
    for p in twist_points(32) {out.push(wire(p));}
    out
}

#[test]
fn legacy_g2_wire_acceptance_matches_frozen_parser() {
    assert!(decode_g2_reference(&[0;127]).is_err());
    assert!(decode_g2_reference(&[0;129]).is_err());
    for bytes in acceptance_corpus() {
        let a=old_decoder(&bytes);let b=decode_g2_reference(&bytes);
        assert_eq!(a.is_ok(),b.is_ok());if let Ok(p)=a {assert_eq!(p,b.unwrap());}
    }
    for n in [127,128,129] {
        let bytes=wire(G2Affine::generator()).repeat(n);
        let (masks,_)=profile_cpu_full_order(&bytes).unwrap();assert_eq!(masks,vec![0;n]);
    }
}

#[test]
fn g2_profile_separates_parse_curve_and_subgroup() {
    let mut bytes=Vec::new();let corpus=acceptance_corpus();
    for p in &corpus {bytes.extend_from_slice(p);}
    let (masks,timing)=profile_cpu_full_order(&bytes).unwrap();
    for (i,p) in corpus.iter().enumerate() {assert_eq!(masks[i]==0,old_decoder(p).is_ok());}
    assert!(timing.parse_ns>0 && timing.curve_ns>0 && timing.subgroup_ns>0);
    assert!(timing.parse_ns+timing.curve_ns+timing.subgroup_ns<=timing.wall_ns);
}

#[test]
#[ignore="explicit deterministic public corpus and split CPU baseline"]
fn write_public_corpus_and_host_profile() {
    let mut rng=StdRng::seed_from_u64(1720002);let mut bytes=Vec::new();
    for _ in 0..4096 {bytes.extend_from_slice(&wire((G2Projective::generator()*Fr::rand(&mut rng)).into_affine()));}
    for p in twist_points(256) {bytes.extend_from_slice(&wire(p));}
    let (masks,t)=profile_cpu_full_order(&bytes).unwrap();
    assert!(masks[..4096].iter().all(|&m|m==0));assert!(masks[4096..].iter().any(|&m|m==4));
    let dir=test_artifacts::directory().unwrap();
    std::fs::write(dir.join("public-corpus.bin"),&bytes).unwrap();
    std::fs::write(dir.join("host-split-baseline.csv"),format!("points,parse_ns,curve_ns,subgroup_ns,wall_ns\n4352,{},{},{},{}\n",t.parse_ns,t.curve_ns,t.subgroup_ns,t.wall_ns)).unwrap();
    eprintln!("G2_HOST_SPLIT points=4352 parse={} curve={} subgroup={} wall={}",t.parse_ns,t.curve_ns,t.subgroup_ns,t.wall_ns);
}

pub(super) fn differential_points()->&'static Vec<G2Affine> {
    static POINTS:std::sync::OnceLock<Vec<G2Affine>>=std::sync::OnceLock::new();
    POINTS.get_or_init(|| {
        let mut points=vec![G2Affine::identity()];
        let mut rng=StdRng::seed_from_u64(1720003);
        for _ in 0..2500 {points.push((G2Projective::generator()*Fr::rand(&mut rng)).into_affine());}
        let twist=twist_points(5000);
        for p in &twist {points.push(*p);}
        for p in &twist[..1250] {
            let q=p.mul_bigint(Fr::MODULUS);
            points.push(q.into_affine());points.push((q+G2Projective::generator()).into_affine());
        }
        points
    })
}

#[test]
fn frobenius_predicates_match_full_order_on_twist_points() {
    let points=differential_points();assert!(points.len()>=10000);
    for (i,p) in points.iter().enumerate() {
        assert!(p.is_on_curve());let expected=subgroup_full_order(p);
        assert_eq!(subgroup_frobenius_127(p),expected,"127 index={i}");
        assert_eq!(subgroup_frobenius_63(p),expected,"63 index={i}");
    }
}

#[test]
fn non_subgroup_corpus_is_independent_of_fast_formula() {
    for p in twist_points(32) {
        let q=p.mul_bigint(Fr::MODULUS).into_affine();
        assert!(!q.is_zero());assert!(!subgroup_full_order(&q));
        for bad in [q,(-q.into_group()).into_affine(),(q+G2Projective::generator()).into_affine()] {
            assert!(!subgroup_frobenius_127(&bad));assert!(!subgroup_frobenius_63(&bad));
        }
    }
}

#[test]
fn psi_constants_match_twist_and_homogeneous_reference() {
    let mut points=twist_points(32);points.push(G2Affine::generator());points.push(G2Affine::identity());
    for p in points {
        let image=psi(p.into_group());assert!(image.into_affine().is_on_curve());
        let t=6u128*u128::from(BN_SEED)*u128::from(BN_SEED)+1;
        let characteristic=psi(image)-image.mul_bigint([t as u64,(t>>64) as u64])+p.mul_bigint(Fq::MODULUS);
        assert!(characteristic.is_zero());
    }
    let g=G2Affine::generator();assert_eq!(psi(g.into_group()),g.mul_bigint(lambda_limbs()));
    let (cx,cy)=psi_constants();
    let mut bytes=Vec::new();
    for field in [cx.c0,cx.c1,cy.c0,cy.c1] {for limb in field.into_bigint().0 {bytes.extend_from_slice(&limb.to_le_bytes());}}
    let dir=test_artifacts::directory().unwrap();
    std::fs::write(dir.join("psi-constants.bin"),bytes).unwrap();
    let mut cofactor=Vec::new();
    for limb in ark_bn254::g2::Config::COFACTOR {cofactor.extend_from_slice(&limb.to_le_bytes());}
    std::fs::write(dir.join("cofactor.bin"),cofactor).unwrap();
}

#[test]
#[ignore="explicit CPU thread-count benchmark, all data generation outside timing"]
fn cpu_frobenius_thread_profile() {
    let points=differential_points();
    let expected=points.iter().filter(|p|subgroup_full_order(p)).count();
    let algorithms:[(&str,fn(&G2Affine)->bool);3]=[("full_r",subgroup_full_order),("frob127",subgroup_frobenius_127),("frob63",subgroup_frobenius_63)];
    let mut csv=String::from("algorithm,threads,points,sample,wall_ns,accepted\n");
    for (name,algorithm) in algorithms {for threads in [1,2,4,8] {for sample in 0..3 {
        let start=Instant::now();
        let accepted=std::thread::scope(|scope| {
            let mut workers=Vec::new();
            for chunk in points.chunks((points.len()+threads-1)/threads) {
                workers.push(scope.spawn(move || {let mut count=0;for point in chunk {if algorithm(point) {count+=1;}}count}));
            }
            let mut total=0;for worker in workers {total+=worker.join().unwrap();}total
        });
        let ns=start.elapsed().as_nanos();assert_eq!(accepted,expected);
        csv.push_str(&format!("{name},{threads},{},{sample},{ns},{accepted}\n",points.len()));
    }}}
    let dir=test_artifacts::directory().unwrap();
    std::fs::write(dir.join("cpu-thread-profile.csv"),csv).unwrap();
}

#[cfg(feature="icicle")]
mod gpu {
    use super::*;
    use crate::IcicleBackend;

    fn check(gpu:&IcicleBackend,bytes:&[u8],algorithm:u32)->Vec<u32> {
        let (masks,report)=gpu.inspect_g2_validation(bytes,algorithm).unwrap();
        let (expected,_)=profile_cpu_full_order(bytes).unwrap();assert_eq!(masks,expected);
        let invalid=masks.iter().filter(|&&m|m!=0).count() as u64;
        assert_eq!(report.checked_points,bytes.len() as u64/128);
        assert_eq!(report.invalid_points,invalid);
        assert_eq!(report.first_invalid,masks.iter().position(|&m|m!=0).map(|i|i as u64).unwrap_or(u64::MAX));
        assert_eq!(report.failure_mask,masks.iter().fold(0,|a,m|a|m));assert_eq!(report.algorithm,algorithm);
        masks
    }

    #[test]
    fn gpu_full_order_matches_legacy_decoder_and_reports_first_error() {
        let gpu=IcicleBackend::new().unwrap();let corpus=acceptance_corpus();
        let mut bytes=Vec::new();for p in &corpus {bytes.extend_from_slice(p);}
        let masks=check(&gpu,&bytes,1);
        for (i,p) in corpus.iter().enumerate() {assert_eq!(masks[i]==0,old_decoder(p).is_ok());}
        for n in [0,1,127,128,129,4096] {check(&gpu,&wire(G2Affine::generator()).repeat(n),1);}
        for bad in [0,127,128,4095] {
            let mut bytes=wire(G2Affine::generator()).repeat(4096);bytes[bad*128]^=1;
            let masks=check(&gpu,&bytes,1);assert_ne!(masks[bad],0);
        }
        let stats=gpu.g2_validation_stats().unwrap();assert!(stats.checked_points>0);assert_eq!(stats.accepted_points,0);
        assert!(stats.subgroup_device_ns>0 && stats.native_wall_ns>0);
        assert!(gpu.memory_info().is_ok()); // invalid content did not retire this owner
    }

    #[test]
    fn integer_group_order_is_not_reduced_to_fr_zero() {
        let gpu=IcicleBackend::new().unwrap();
        for p in twist_points(16) {
            let q=p.mul_bigint(Fr::MODULUS).into_affine();assert!(!subgroup_full_order(&q));
            assert_eq!(check(&gpu,&wire(q),1),[4]);
        }
    }

    #[test]
    fn gpu_status_reduction_does_not_cancel_bad_points() {
        let gpu=IcicleBackend::new().unwrap();let q=twist_points(1)[0].mul_bigint(Fr::MODULUS).into_affine();
        for (a,b) in [(0,1),(127,128),(4095,4096)] {
            let mut bytes=wire(G2Affine::generator()).repeat(4097);
            bytes[a*128..a*128+128].copy_from_slice(&wire(q));
            bytes[b*128..b*128+128].copy_from_slice(&wire((-q.into_group()).into_affine()));
            let mut invalid=0;for chunk in bytes.chunks(4096*128) {invalid+=check(&gpu,chunk,1).iter().filter(|&&m|m!=0).count();}
            assert_eq!(invalid,2);
        }
    }

    #[test]
    fn inspect_lengths_quota_and_retry_preserve_owner() {
        let gpu=IcicleBackend::new().unwrap();
        assert!(gpu.inspect_g2_validation(&[0;127],1).is_err());
        assert!(gpu.inspect_g2_validation(&vec![0;4097*128],1).is_err());
        for algorithm in [0,4,u32::MAX] {assert!(gpu.inspect_g2_validation(&[],algorithm).is_err());}
        check(&gpu,&wire(G2Affine::generator()),1);
        let domain_bytes=((1u64<<20)+1+6+64)*32;
        // Owner construction reserves a mandatory 64 MiB workspace margin.
        // Leave 256 extra bytes: below this operator's n=1 admission scratch.
        let limited=IcicleBackend::with_resident_limits(domain_bytes+(64<<20)+256,0).unwrap();
        let before=limited.g2_validation_stats().unwrap();
        assert!(limited.inspect_g2_validation(&wire(G2Affine::generator()),1).unwrap_err().to_string().contains("quota"));
        assert_eq!(limited.g2_validation_stats().unwrap(),before);assert!(limited.memory_info().is_ok());
        check(&limited,&[],1);
        let healthy=IcicleBackend::with_resident_limits(domain_bytes+(64<<20)+4096*512,0).unwrap();
        check(&healthy,&wire(G2Affine::generator()).repeat(129),1);
        let counts=healthy.setup_point_counts();
        for _ in 0..4 {check(&healthy,&wire(q_point()),1);}
        assert_eq!(healthy.setup_point_counts(),counts);
    }
    #[test]
    fn native_inspect_rejects_bad_arguments_without_publishing() {
        let gpu=IcicleBackend::new().unwrap();let mut error=[0u8;256];
        for (n,algorithm) in [(1,1),(4097,1),(0,0),(0,4)] {
            let mut report=G2ValidationReport{checked_points:91,invalid_points:92,first_invalid:93,failure_mask:94,algorithm:95};
            let original=report;
            // SAFETY: intentionally null borrowed data buffers; native validates
            // n/algorithm/null before dereferencing or allocating any CUDA data.
            let status=unsafe {dvgpu_bn254_g2_validation_inspect(gpu.handle.as_ptr(),std::ptr::null(),n,
                algorithm,std::ptr::null_mut(),&mut report,error.as_mut_ptr().cast(),error.len())};
            assert_eq!(status,-1);assert_eq!(report,original);
        }
        assert!(gpu.memory_info().is_ok());check(&gpu,&wire(G2Affine::generator()),1);
    }
    fn q_point()->G2Affine {twist_points(1)[0].mul_bigint(Fr::MODULUS).into_affine()}

    #[test]
    fn all_gpu_predicates_match_full_order_corpus() {
        let gpu=IcicleBackend::new().unwrap();let points=differential_points();
        assert!(points.len()>=10000);
        for chunk in points.chunks(4096) {
            let mut bytes=Vec::new();for p in chunk {bytes.extend_from_slice(&wire(*p));}
            let full=check(&gpu,&bytes,1);
            for algorithm in [2,3] {assert_eq!(check(&gpu,&bytes,algorithm),full);}
        }
        let mut bytes=Vec::new();for p in acceptance_corpus() {bytes.extend_from_slice(&p);}
        for algorithm in [2,3] {check(&gpu,&bytes,algorithm);}
    }

    extern "C" {
        fn dvgpu_g2_psi_probe(owner:*mut std::ffi::c_void,points:*const u8,n:usize,
            output:*mut u8,error:*mut std::ffi::c_char,capacity:usize)->i32;
    }
    fn probe(gpu:&IcicleBackend,bytes:&[u8])->(i32,Vec<u8>) {
        assert_eq!(bytes.len()%192,0);let mut output=vec![0x55;bytes.len()*3];let mut error=[0u8;256];
        // SAFETY: six canonical Fq per homogeneous point, bounded output for
        // three images; native borrows until synchronous completion only.
        let status=unsafe {dvgpu_g2_psi_probe(gpu.handle.as_ptr(),bytes.as_ptr(),bytes.len()/192,
            output.as_mut_ptr(),error.as_mut_ptr().cast(),error.len())};
        (status,output)
    }
    fn encode_homogeneous(fields:[Fq2;3])->Vec<u8> {
        let mut bytes=Vec::new();
        for field in fields {for fq in [field.c0,field.c1] {for limb in fq.into_bigint().0 {bytes.extend_from_slice(&limb.to_le_bytes());}}}
        bytes
    }
    fn decode_homogeneous(bytes:&[u8])->[Fq2;3] {
        let mut fields=Vec::new();
        for fq in bytes.chunks_exact(32) {
            let mut limbs=[0;4];for i in 0..4 {limbs[i]=u64::from_le_bytes(fq[8*i..8*i+8].try_into().unwrap());}
            fields.push(Fq::from_bigint(ark_ff::BigInt(limbs)).expect("noncanonical probe output"));
        }
        [Fq2::new(fields[0],fields[1]),Fq2::new(fields[2],fields[3]),Fq2::new(fields[4],fields[5])]
    }
    #[test]
    fn gpu_psi_matches_cpu_homogeneous_map() {
        use ark_ff::One;
        let gpu=IcicleBackend::new().unwrap();let mut points=twist_points(32);
        points.extend_from_slice(&[G2Affine::generator(),(-G2Projective::generator()).into_affine(),G2Affine::identity()]);
        let mut rng=StdRng::seed_from_u64(1720004);let mut bytes=Vec::new();
        for p in &points {
            let mut z=Fq2::rand(&mut rng);if z.is_zero() {z=Fq2::one();}
            bytes.extend_from_slice(&encode_homogeneous(if p.is_zero() {[Fq2::zero(),z,Fq2::zero()]} else {[p.x*z,p.y*z,z]}));
        }
        let (status,output)=probe(&gpu,&bytes);assert_eq!(status,0);
        for (i,p) in points.iter().enumerate() {
            let mut expected=p.into_group();
            for power in 0..3 {
                expected=psi(expected);let affine=expected.into_affine();
                let fields=decode_homogeneous(&output[(3*i+power)*192..(3*i+power+1)*192]);
                if affine.is_zero() {assert!(fields[0].is_zero() && fields[2].is_zero() && !fields[1].is_zero());}
                else {assert!(!fields[2].is_zero());assert_eq!(fields[0],affine.x*fields[2]);assert_eq!(fields[1],affine.y*fields[2]);}
            }
        }
        let (status,output)=probe(&gpu,&[0;192]);assert_eq!(status,-1);assert!(output.iter().all(|&b|b==0x55));
        let mut noncanonical=bytes[..192].to_vec();noncanonical[..32].fill(0xff);
        let (status,output)=probe(&gpu,&noncanonical);assert_eq!(status,-1);assert!(output.iter().all(|&b|b==0x55));
        assert!(gpu.memory_info().is_ok());
    }

    #[test]
    #[ignore="explicit bounded standalone GPU timing, not integrated setup"]
    fn gpu_validation_resource_profile() {
        let gpu=IcicleBackend::new().unwrap();
        let dir=test_artifacts::directory().unwrap();
        let corpus=std::fs::read(dir.join("public-corpus.bin")).unwrap();
        let mut csv=String::from("algorithm,points,sample,wall_ns,parse_curve_ns,subgroup_ns,reduce_ns,h2d_ns,d2h_ns,native_wall_ns\n");
        for n in [128,1024,4096] {for algorithm in [1,2,3] {
            for sample in -4i32..10 {
                let before=gpu.g2_validation_stats().unwrap();let start=Instant::now();
                let (_,report)=gpu.inspect_g2_validation(&corpus[..n*128],algorithm).unwrap();
                let wall=start.elapsed().as_nanos();assert_eq!(report.invalid_points,0);
                let after=gpu.g2_validation_stats().unwrap();
                if sample==-4 || sample>=0 {csv.push_str(&format!("{algorithm},{n},{sample},{wall},{},{},{},{},{},{}\n",
                    after.parse_curve_device_ns-before.parse_curve_device_ns,after.subgroup_device_ns-before.subgroup_device_ns,
                    after.reduce_device_ns-before.reduce_device_ns,after.h2d_device_ns-before.h2d_device_ns,
                    after.d2h_device_ns-before.d2h_device_ns,after.native_wall_ns-before.native_wall_ns));}
            }
        }}
        std::fs::write(dir.join("gpu-standalone-profile.csv"),csv).unwrap();
    }
}
