//! Phase A day-1 smoke: arkworks BN254 scalar multiplication + pairing.
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup, Group};
use ark_ff::{Field, PrimeField};

#[test]
fn bn254_scalar_mul_matches_generator_identity() {
    // 7*G computed two ways must agree: scalar mul vs G + 2G + 4G.
    let g = G1Projective::from(G1Affine::generator());
    let lhs = (g * Fr::from(7u64)).into_affine();
    let rhs = (g + g.double() + g.double().double()).into_affine();
    assert_eq!(lhs, rhs);
}

#[test]
fn bn254_pairing_bilinearity_small() {
    // e(aG, bH) == e(G, H)^(ab)
    let g = G1Affine::generator();
    let h = G2Affine::generator();
    let a = Fr::from(3u64);
    let b = Fr::from(7u64);

    let e_ag_bh = Bn254::pairing(
        (G1Projective::from(g) * a).into_affine(),
        (G2Projective::from(h) * b).into_affine(),
    );
    let e_g_h = Bn254::pairing(g, h);

    let ab = a * b;
    let expected = ark_ec::pairing::PairingOutput(e_g_h.0.pow(ab.into_bigint()));
    // PairingOutput is a newtype over Fp12; construct expected via field pow then wrap.
    assert_eq!(e_ag_bh, expected);
}

#[test]
fn bn254_field_ops_smoke() {
    // (x + 1)^2 == x^2 + 2x + 1 in Fr
    let x = Fr::from(12345u64);
    let one = Fr::from(1u64);
    let two = Fr::from(2u64);
    assert_eq!((x + one) * (x + one), x * x + two * x + one);
    // sqrt of a square is one of the two roots
    let y = x * x;
    let r = y.sqrt().expect("square has a sqrt in Fr (p = 1 mod 4)");
    assert!(r == x || r == -x);
}
