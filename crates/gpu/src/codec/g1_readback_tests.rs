use super::*;
use ark_bn254::Fq;
use ark_ec::CurveGroup;
use ark_ff::UniformRand;
use rand::{rngs::StdRng,RngCore,SeedableRng};

fn compare(bytes: &[u8]) -> bool {
    let reference=decode_g1(bytes);let bounded=decode_g1_cofactor_one(bytes);
    assert_eq!(reference.is_ok(),bounded.is_ok(),"different acceptance: {bytes:?}");
    if let Ok(point)=reference {assert_eq!(bounded.unwrap(),point);true} else {false}
}

#[test]
fn cofactor_one_decoder_preserves_literal_wire_and_flag_acceptance() {
    for len in [0,1,31,32,63,65,128] {assert!(!compare(&vec![0;len]));}
    assert_eq!(decode_g1_cofactor_one(&[0;64]).unwrap(),G1Affine::identity());
    for flags in [0,0x40,0x80,0xc0] {
        let mut valid=[0u8;64];valid[0]=1;valid[32]=2;valid[63]=flags;
        let want=match flags {0|0x80=>Some(G1Affine::generator()),0x40=>Some(G1Affine::identity()),_=>None};
        let actual=decode_g1_cofactor_one(&valid);
        assert_eq!(actual.is_ok(),want.is_some());
        if let Some(point)=want {assert_eq!(actual.unwrap(),point);}
        compare(&valid);
        // The full oracle accepts flagged infinity with canonical nonzero
        // coordinates, even when that finite (x,y) is off-curve. Preserve it.
        valid[32]=3;
        assert_eq!(compare(&valid),flags==0x40);
    }
    for coordinate in [0,32] {for flags in [0,0x40,0x80,0xc0] {
        let mut invalid=[0u8;64];invalid[0]=1;invalid[32]=2;
        for (i,limb) in Fq::MODULUS.0.iter().enumerate() {
            invalid[coordinate+8*i..coordinate+8*i+8].copy_from_slice(&limb.to_le_bytes());
        }
        invalid[63]|=flags;
        assert!(!compare(&invalid),"noncanonical coordinate accepted");
    }}
}

#[test]
fn cofactor_one_decoder_matches_full_oracle_on_mutations_boundaries_and_random_bytes() {
    let mut rng=StdRng::seed_from_u64(29100);let mut count=0;let mut accepted=0;
    for _ in 0..64 {
        let point=(G1Affine::generator()*Fr::rand(&mut rng)).into_affine();
        let bytes=encode_g1(&point);
        assert_eq!(decode_g1_cofactor_one(&bytes).unwrap(),point);
        for flags in [0,0x40,0x80,0xc0] {
            let mut flagged=bytes;flagged[63]=(flagged[63]&0x3f)|flags;
            accepted+=usize::from(compare(&flagged));count+=1;
        }
        for at in 0..64 {for mask in [1,0x80] {
            let mut bad=bytes;bad[at]^=mask;
            accepted+=usize::from(compare(&bad));count+=1;
        }}
    }
    for _ in 0..4096 {
        let mut bytes=[0u8;64];rng.fill_bytes(&mut bytes);
        accepted+=usize::from(compare(&bytes));count+=1;
    }
    // Literal low/carry/high limb boundaries, with all four flag patterns.
    for x in [0,1,2,u64::MAX] {for y in [0,1,2,3,u64::MAX] {for flags in [0,0x40,0x80,0xc0] {
        let mut bytes=[0u8;64];bytes[..8].copy_from_slice(&x.to_le_bytes());
        bytes[32..40].copy_from_slice(&y.to_le_bytes());bytes[63]=flags;
        accepted+=usize::from(compare(&bytes));count+=1;
    }}}
    eprintln!("G1_DECODE_CORPUS total={count} accepted={accepted} rejected={}",count-accepted);
}
