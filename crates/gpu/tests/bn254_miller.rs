//! Admission tests prepared during the source-only stage; not yet executed.
#![cfg(feature = "icicle")]

use ark_bn254::{Bn254, Config, Fq, Fq2, Fq6, Fq12, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{bn::{BnConfig, G2Prepared}, pairing::Pairing, AffineRepr, CurveGroup};
use ark_ff::{BigInteger, Field, PrimeField, UniformRand, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use dynaverse_gpu::{bn254_miller::{Bn254MillerProbe, BN254_MILLER_LINES, BN254_MILLER_INFINITY},
    codec::{decode_g1, encode_g1}, IcicleBackend};
use rand::{rngs::StdRng, SeedableRng};

fn qwire(point: G2Affine) -> Vec<u8> {
    let mut bytes=Vec::new(); point.serialize_uncompressed(&mut bytes).unwrap(); bytes
}
fn field(bytes: &[u8]) -> Fq12 {
    let mut input=bytes;
    let value=Fq12::deserialize_uncompressed(&mut input).unwrap();
    assert!(input.is_empty()); value
}
fn dense(a: Fq2,b: Fq2,c: Fq2) -> Fq12 {
    Fq12::new(Fq6::new(a,Fq2::ZERO,Fq2::ZERO),Fq6::new(b,c,Fq2::ZERO))
}
fn scaled(coeff: (Fq2,Fq2,Fq2),p: G1Affine) -> (Fq2,Fq2,Fq2) {
    let (mut a,mut b,c)=coeff;
    a.mul_assign_by_fp(&p.y); b.mul_assign_by_fp(&p.x); (a,b,c)
}
fn assert_trace(result: &Bn254MillerProbe,p: G1Affine,q: G2Affine) {
    assert_eq!(result.g1_status,if p.is_zero(){BN254_MILLER_INFINITY}else{0});
    assert_eq!(result.g2_status,if q.is_zero(){BN254_MILLER_INFINITY}else{0});
    assert_eq!(field(result.value()),Bn254::multi_miller_loop([p],[q]).0);
    if p.is_zero() || q.is_zero() {
        assert_eq!(result.count,0);
        assert!(result.evaluated_lines().iter().chain(result.accumulations()).all(|&b|b==0));
        return;
    }
    assert_eq!(result.count as usize,BN254_MILLER_LINES);
    let prepared=G2Prepared::<Config>::from(q);
    let mut f=Fq12::ONE; let mut next=0;
    // CPU sparse multiplication is independent of the CUDA general dense
    // product. Compare every post-line snapshot, then the original CPU result.
    let mut consume=|f: &mut Fq12| {
        let (a,b,c)=scaled(prepared.ell_coeffs[next],p);
        assert_eq!(field(&result.evaluated_lines()[next*384..(next+1)*384]),dense(a,b,c));
        f.mul_by_034(&a,&b,&c);
        assert_eq!(field(&result.accumulations()[next*384..(next+1)*384]),*f,"line {next}");
        next+=1;
    };
    let digits=<Config as BnConfig>::ATE_LOOP_COUNT;
    assert!(!<Config as BnConfig>::X_IS_NEGATIVE);
    for i in (1..digits.len()).rev() {
        if i!=digits.len()-1 {f.square_in_place();}
        consume(&mut f);
        if digits[i-1]!=0 {consume(&mut f);}
    }
    consume(&mut f); consume(&mut f);
    assert_eq!(next,BN254_MILLER_LINES);
    assert_eq!(f,field(result.value()));
}

#[test]
fn miller_trace_and_final_match_locked_cpu() {
    let backend=IcicleBackend::new().unwrap();
    let p=G1Affine::generator(); let q=G2Affine::generator();
    let mut pairs=vec![(p,q),(-p,q),(p,-q),(-p,-q)];
    let mut rng=StdRng::seed_from_u64(0x2026_1021_21);
    for _ in 0..8 {pairs.push((G1Projective::rand(&mut rng).into_affine(),G2Projective::rand(&mut rng).into_affine()));}
    for (p,q) in pairs {
        let result=backend.inspect_bn254_miller(&encode_g1(&p),&qwire(q)).unwrap();
        assert_trace(&result,p,q);
    }
}

#[test]
fn d_twist_first_line_uses_034_and_g1_xy() {
    let backend=IcicleBackend::new().unwrap();
    let p=(G1Affine::generator().into_group()*Fr::from(7u64)).into_affine();
    let q=G2Affine::generator();
    let b=Fq2::from(3u64)/Fq2::new(Fq::from(9u64),Fq::ONE);
    // Independently derived tangent, avoiding G2Prepared for this assertion.
    let (a,c,d)=scaled((-q.y.double(),q.x.square()*Fq2::from(3u64),b*Fq2::from(3u64)-q.y.square()),p);
    let expected=dense(a,c,d);
    let result=backend.inspect_bn254_miller(&encode_g1(&p),&qwire(q)).unwrap();
    assert_eq!(field(&result.evaluated_lines()[..384]),expected);
    assert_eq!(field(&result.accumulations()[..384]),expected);
    assert!(!expected.is_zero());
}

#[test]
fn infinity_encodings_omit_pair_but_still_validate_partner() {
    let backend=IcicleBackend::new().unwrap();
    let p=G1Affine::generator();let q=G2Affine::generator();
    let pw=encode_g1(&p);let qw=qwire(q);
    let mut flagged_p=pw;flagged_p[63]=(flagged_p[63]&0x3f)|0x40;
    let mut flagged_q=qw.clone();flagged_q[127]=(flagged_q[127]&0x3f)|0x40;
    let mut serialized_p=Vec::new();G1Affine::identity().serialize_uncompressed(&mut serialized_p).unwrap();
    for pin in [vec![0;64],serialized_p,flagged_p.to_vec()] {
        assert!(decode_g1(&pin).unwrap().is_zero());
        assert_trace(&backend.inspect_bn254_miller(&pin,&qw).unwrap(),G1Affine::identity(),q);
    }
    for qin in [vec![0;128],qwire(G2Affine::identity()),flagged_q] {
        assert_trace(&backend.inspect_bn254_miller(&pw,&qin).unwrap(),p,G2Affine::identity());
    }
    assert_trace(&backend.inspect_bn254_miller(&[0;64],&[0;128]).unwrap(),G1Affine::identity(),G2Affine::identity());
    let bad_p=encode_g1(&G1Affine::new_unchecked(Fq::ONE,Fq::ONE));
    let rejected=backend.inspect_bn254_miller(&bad_p,&[0;128]).unwrap();
    assert_eq!((rejected.g1_status,rejected.g2_status,rejected.count),(2,8,0));
    assert!(rejected.output.iter().all(|&b|b==0));
    let rejected=backend.inspect_bn254_miller(&[0;64],&qwire(non_subgroup())).unwrap();
    assert_eq!((rejected.g1_status,rejected.g2_status,rejected.count),(8,4,0));
    assert!(rejected.output.iter().all(|&b|b==0));
}

fn non_subgroup() -> G2Affine {
    let mut rng=StdRng::seed_from_u64(0x2026_1021_22);
    let b=Fq2::from(3u64)/Fq2::new(Fq::from(9u64),Fq::ONE);
    for _ in 0..1024 {
        let x=Fq2::rand(&mut rng);
        if let Some(y)=(x.square()*x+b).sqrt() {
            let q=G2Affine::new_unchecked(x,y);
            if !q.mul_bigint(Fr::MODULUS).is_zero() {return q;}
        }
    }
    panic!("fixed-seed non-subgroup fixture search exhausted");
}

#[test]
fn invalid_encodings_curves_and_subgroup_zero_every_output_then_retry() {
    let backend=IcicleBackend::new().unwrap();
    let p=G1Affine::generator();let q=G2Affine::generator();
    let pw=encode_g1(&p);let qw=qwire(q);let mut cases=Vec::new();
    for coordinate in 0..2 {
        let mut bad=pw;bad[coordinate*32..(coordinate+1)*32].copy_from_slice(&Fq::MODULUS.to_bytes_le());
        assert!(decode_g1(&bad).is_err());cases.push((bad.to_vec(),qw.clone(),1,0));
    }
    for coordinate in 0..4 {
        let mut bad=qw.clone();bad[coordinate*32..(coordinate+1)*32].copy_from_slice(&Fq::MODULUS.to_bytes_le());
        cases.push((pw.to_vec(),bad,0,1));
    }
    let mut bad_p=pw;bad_p[63]|=0xc0;cases.push((bad_p.to_vec(),qw.clone(),1,0));
    let mut bad_q=qw.clone();bad_q[127]|=0xc0;cases.push((pw.to_vec(),bad_q,0,1));
    cases.push((encode_g1(&G1Affine::new_unchecked(Fq::ONE,Fq::ONE)).to_vec(),qw.clone(),2,0));
    cases.push((pw.to_vec(),qwire(G2Affine::new_unchecked(Fq2::ONE,Fq2::ONE)),0,2));
    cases.push((pw.to_vec(),qwire(non_subgroup()),0,4));
    // Infinity flags never excuse a noncanonical coordinate. Check every
    // component, including the final field that carries the flag bits.
    for coordinate in 0..2 {
        let mut bad=pw;bad[coordinate*32..(coordinate+1)*32].copy_from_slice(&Fq::MODULUS.to_bytes_le());
        bad[63]=(bad[63]&0x3f)|0x40;
        assert!(decode_g1(&bad).is_err());cases.push((bad.to_vec(),qw.clone(),1,0));
    }
    for coordinate in 0..4 {
        let mut bad=qw.clone();bad[coordinate*32..(coordinate+1)*32].copy_from_slice(&Fq::MODULUS.to_bytes_le());
        bad[127]=(bad[127]&0x3f)|0x40;cases.push((pw.to_vec(),bad,0,1));
    }
    cases.push((encode_g1(&G1Affine::new_unchecked(Fq::ONE,Fq::ONE)).to_vec(),
        qwire(G2Affine::new_unchecked(Fq2::ONE,Fq2::ONE)),2,2));
    for (pin,qin,ps,qs) in cases {
        let result=backend.inspect_bn254_miller(&pin,&qin).unwrap();
        assert_eq!((result.g1_status,result.g2_status,result.count),(ps,qs,0));
        assert!(result.output.iter().all(|&b|b==0));
        assert_trace(&backend.inspect_bn254_miller(&pw,&qw).unwrap(),p,q);
    }
    // Uncompressed sign bits are accepted by the retained arkworks codec.
    let mut signed_p=pw;signed_p[63]|=0x80;
    let mut signed_q=qw;signed_q[127]|=0x80;
    assert_eq!(decode_g1(&signed_p).unwrap(),p);
    assert_trace(&backend.inspect_bn254_miller(&signed_p,&signed_q).unwrap(),p,q);
}

#[test]
fn exact_single_pair_shapes_preserve_owner() {
    let backend=IcicleBackend::new().unwrap();
    for (p,q) in [(vec![],vec![]),(vec![0;63],vec![0;128]),(vec![0;64],vec![0;127]),
        (vec![0;128],vec![0;256])] {
        assert!(backend.inspect_bn254_miller(&p,&q).is_err());
    }
    backend.memory_info().unwrap();
    assert_trace(&backend.inspect_bn254_miller(&encode_g1(&G1Affine::generator()),&qwire(G2Affine::generator())).unwrap(),
        G1Affine::generator(),G2Affine::generator());
}
