#![cfg(feature = "icicle")]
//! Caller-paid inclusive operator timings; never used as the work ledger.
use ark_bn254::{Fr,G1Projective,G2Projective};
use ark_ec::{CurveGroup,Group};
use dynaverse_gpu::{ArkworksBackend,IcicleBackend};
use dynaverse_protocol::OperatorBackend;
use std::time::Instant;

fn quantiles(mut values:Vec<u128>)->(u128,u128,u128) {
    values.sort_unstable();
    (values[values.len()/2],values[(values.len()*99).div_ceil(100)-1],*values.last().unwrap())
}

#[test]
#[ignore = "explicit local CPU/GPU inclusive Fr/G1/G2 operator profile"]
fn stage5_operator_latency_profile() {
    let gpu=IcicleBackend::with_resident_limits(512<<20,16<<20).unwrap();
    let cpu=ArkworksBackend;
    for n in [16,256,4096] {
        let original:Vec<_>=(0..n).map(|i|Fr::from(i as u64+2)).collect();
        let mut cpu_ns=Vec::new(); let mut gpu_ns=Vec::new();
        for iteration in 0..21 {
            let mut expected=original.clone(); let mut actual=original.clone();
            let start=Instant::now(); cpu.ntt(&mut expected,false).unwrap(); let c=start.elapsed().as_nanos();
            let start=Instant::now(); gpu.ntt(&mut actual,false).unwrap(); let g=start.elapsed().as_nanos();
            assert_eq!(actual,expected);
            if iteration>0 {cpu_ns.push(c);gpu_ns.push(g);}
        }
        eprintln!("FR_PROFILE n={n} samples=20 cpu_ns={:?} gpu_inclusive_ns={:?}",quantiles(cpu_ns),quantiles(gpu_ns));
    }
    for n in [2,16,128] {
        let scalars:Vec<_>=(0..n).map(|i|-Fr::from(i as u64+7)).collect();
        let g1:Vec<_>=(0..n).map(|i|(G1Projective::generator()*Fr::from(i as u64+3)).into_affine()).collect();
        let g2:Vec<_>=(0..n).map(|i|(G2Projective::generator()*Fr::from(i as u64+3)).into_affine()).collect();
        let mut c1=Vec::new(); let mut d1=Vec::new(); let mut e1=Vec::new(); let mut v1=Vec::new();
        let mut c2=Vec::new(); let mut d2=Vec::new();
        for iteration in 0..21 {
            let start=Instant::now(); let expected=cpu.msm(&g1,&scalars).unwrap(); let c=start.elapsed().as_nanos();
            let start=Instant::now(); let (actual,profile)=gpu.msm_profiled(&g1,&scalars).unwrap(); let g=start.elapsed().as_nanos();
            assert_eq!(actual,expected);
            let start=Instant::now(); let expected=cpu.msm_g2_host(&g2,&scalars).unwrap(); let c_g2=start.elapsed().as_nanos();
            let start=Instant::now(); let actual=gpu.msm_g2_host(&g2,&scalars).unwrap(); let g_g2=start.elapsed().as_nanos();
            assert_eq!(actual,expected);
            if iteration>0 {
                c1.push(c);d1.push(g);e1.push(profile.encoding.as_nanos());v1.push(profile.validation.as_nanos());
                c2.push(c_g2);d2.push(g_g2);
            }
        }
        eprintln!("G1_PROFILE n={n} samples=20 cpu_ns={:?} gpu_inclusive_ns={:?} encoding_ns={:?} validation_ns={:?}",
            quantiles(c1),quantiles(d1),quantiles(e1),quantiles(v1));
        eprintln!("G2_PROFILE n={n} samples=20 cpu_ns={:?} gpu_inclusive_ns={:?}",quantiles(c2),quantiles(d2));
    }
}
