//! Codec boundary tests (day 2). Written BEFORE implementation (TDD).
//!
//! Encoding contract (frozen):
//! - Scalars: 32 bytes, little-endian, canonical integer form (ICICLE host flag is false).
//! - G1Affine points: 64 bytes = x (32 LE bytes) || y (32 LE bytes),
//!   in canonical integer form; infinity encoded as all-zero 64 bytes.
//! - Non-canonical scalars (>= field modulus) are REJECTED.
use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::{CurveGroup, AffineRepr};
use ark_ff::{Field, PrimeField};
use dynaverse_gpu::codec::{decode_fr, decode_g1, encode_fr, encode_g1};

#[test]
fn fr_roundtrip_small_values() {
    for x in [0u64, 1, 17, 42, 1024, u64::MAX] {
        let f = Fr::from(x);
        let bytes = encode_fr(&f);
        assert_eq!(bytes.len(), 32);
        assert_eq!(decode_fr(&bytes).unwrap(), f);
    }
}

#[test]
fn fr_roundtrip_random_elements() {
    // deterministic PRNG via field rand with fixed seed is overkill here;
    // derive pseudo-random values from a simple LCG over Fr.
    let mut state = Fr::from(0x2026_0923u64);
    for _ in 0..64 {
        state = state + state.double() + Fr::from(7u64); // cheap mixing
        let bytes = encode_fr(&state);
        assert_eq!(decode_fr(&bytes).unwrap(), state);
    }
}

#[test]
fn fr_rejects_noncanonical() {
    // [0xff;32] is >= modulus -> must be rejected (plan doc requirement).
    assert!(decode_fr(&[0xffu8; 32]).is_err());

    // modulus itself is also non-canonical
    let r = <Fr as PrimeField>::MODULUS;
    let r_bytes = {
        let mut b = [0u8; 32];
        for (i, limb) in r.0.iter().enumerate() {
            b[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
        }
        b
    };
    assert!(decode_fr(&r_bytes).is_err());

    // modulus - 1 is canonical (construct without crossing the modulus)
    let r_minus_1_bytes = {
        let mut limbs = r.0;
        // r - 1 on limbs (r > 0, so no borrow beyond limb 0)
        limbs[0] -= 1;
        let mut b = [0u8; 32];
        for (i, limb) in limbs.iter().enumerate() {
            b[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
        }
        b
    };
    assert!(decode_fr(&r_minus_1_bytes).is_ok());

    // wrong length rejected
    assert!(decode_fr(&[0u8; 31]).is_err());
    assert!(decode_fr(&[0u8; 33]).is_err());
}

#[test]
fn g1_roundtrip_generator_and_random() {
    let mut state = Fr::from(0xABCDEFu64);
    for i in 0..32u64 {
        state = state * Fr::from(2u64) + Fr::from(i);
        let p: G1Affine = (G1Projective::from(G1Affine::generator()) * state).into_affine();
        let bytes = encode_g1(&p);
        assert_eq!(bytes.len(), 64);
        assert_eq!(decode_g1(&bytes).unwrap(), p);
    }
}

#[test]
fn g1_infinity_roundtrip() {
    let inf = G1Affine::identity();
    let bytes = encode_g1(&inf);
    // infinity = all-zero 64 bytes
    assert!(bytes.iter().all(|&b| b == 0));
    let back = decode_g1(&bytes).unwrap();
    assert_eq!(back, inf);
}

#[test]
fn g1_rejects_bad_length_and_off_curve() {
    // wrong length
    assert!(decode_g1(&[0u8; 63]).is_err());
    // point not on curve must be rejected
    let mut bad = [0u8; 64];
    bad[0] = 1; // x = 1
    bad[32] = 1; // y = 1
    // (1,1) is not on bn254 G1 (y^2 = x^3 + 3)
    assert!(decode_g1(&bad).is_err());
}
