//! Pending source-stage admission tests. No performance or GPU pairing claim.
#![cfg(feature = "icicle")]

use ark_bn254::{Bn254, Config, Fq, Fq2, Fr, G1Affine, G2Affine, G2Projective};
use ark_ec::{bn::G2Prepared, pairing::Pairing, AffineRepr, CurveGroup};
use ark_ff::{BigInteger, Field, PrimeField, UniformRand, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use dynaverse_gpu::{bn254_lines::{Bn254LinesProbe, BN254_LINE_COUNT, BN254_LINE_ITEM_BYTES,
    BN254_LINES_INFINITY, MAX_BN254_LINE_ITEMS}, IcicleBackend};
use rand::{rngs::StdRng, SeedableRng};

fn wire(points: &[G2Affine]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for point in points { point.serialize_uncompressed(&mut bytes).unwrap(); }
    bytes
}

fn coefficients(result: &Bn254LinesProbe, item: usize) -> Vec<(Fq2, Fq2, Fq2)> {
    let offset = item * BN254_LINE_ITEM_BYTES;
    let mut values = Vec::new();
    for line in 0..result.counts[item] as usize {
        let mut input = &result.coefficients[offset+line*192..offset+(line+1)*192];
        let a = Fq2::deserialize_uncompressed(&mut input).unwrap();
        let b = Fq2::deserialize_uncompressed(&mut input).unwrap();
        let c = Fq2::deserialize_uncompressed(&mut input).unwrap();
        assert!(input.is_empty());
        values.push((a,b,c));
    }
    values
}

fn assert_point(result: &Bn254LinesProbe, item: usize, point: G2Affine) {
    let expected = G2Prepared::<Config>::from(point);
    assert_eq!(result.statuses[item], if point.is_zero() { BN254_LINES_INFINITY } else { 0 });
    assert_eq!(result.counts[item] as usize, expected.ell_coeffs.len());
    assert_eq!(coefficients(result,item), expected.ell_coeffs);
    let used = result.counts[item] as usize * 192;
    let slot = &result.coefficients[item*BN254_LINE_ITEM_BYTES..(item+1)*BN254_LINE_ITEM_BYTES];
    assert!(slot[used..].iter().all(|&b|b==0));
}

// Catches NAF sign/order, homogeneous-vs-Jacobian, twist and q1/q2 tail errors.
// Expected coefficients come from the retained, locked arkworks preparation.
#[test]
fn all_lines_match_cpu_for_generator_negation_and_seeded_points() {
    let backend = IcicleBackend::new().unwrap();
    let generator = G2Affine::generator();
    let mut points = vec![generator, -generator,
        (generator.into_group()*Fr::from(2u64)).into_affine()];
    let mut rng = StdRng::seed_from_u64(0x2026_1021_11);
    for _ in 0..17 { points.push(G2Projective::rand(&mut rng).into_affine()); }
    for chunk in points.chunks(MAX_BN254_LINE_ITEMS) {
        let result = backend.inspect_bn254_lines(&wire(chunk)).unwrap();
        for (i,&point) in chunk.iter().enumerate() { assert_point(&result,i,point); }
    }
}

// Independent first-tangent relation: (-2y, 3x^2, 3B-y^2) on the D twist.
#[test]
fn first_tangent_has_independently_derived_coefficients() {
    let backend = IcicleBackend::new().unwrap();
    let point = G2Affine::generator();
    let result = backend.inspect_bn254_lines(&wire(&[point])).unwrap();
    let b = Fq2::from(3u64) / Fq2::new(Fq::from(9u64),Fq::ONE);
    let expected = (-point.y.double(),point.x.square()*Fq2::from(3u64),
        b*Fq2::from(3u64)-point.y.square());
    assert_eq!(coefficients(&result,0)[0],expected);
}

// A real downstream consumer catches missing/extra/reordered line streams.
// Accumulation remains CPU arkworks; this is not a GPU Miller implementation.
#[test]
fn gpu_prepared_lines_drive_the_retained_cpu_miller_consumer() {
    let backend = IcicleBackend::new().unwrap();
    let mut rng = StdRng::seed_from_u64(0x2026_1021_12);
    for q in [G2Affine::generator(),-G2Affine::generator(),G2Affine::identity(),
        G2Projective::rand(&mut rng).into_affine()] {
        let p = G1Affine::generator();
        let result = backend.inspect_bn254_lines(&wire(&[q])).unwrap();
        assert_point(&result,0,q);
        let prepared = G2Prepared::<Config> { ell_coeffs: coefficients(&result,0),
            infinity: result.statuses[0]==BN254_LINES_INFINITY };
        assert_eq!(Bn254::multi_miller_loop([p],[prepared]),Bn254::multi_miller_loop([p],[q]));
    }
}

fn non_subgroup() -> G2Affine {
    let mut rng = StdRng::seed_from_u64(0x2026_1021_13);
    let b = Fq2::from(3u64) / Fq2::new(Fq::from(9u64),Fq::ONE);
    for _ in 0..1024 {
        let x = Fq2::rand(&mut rng);
        if let Some(y) = (x.square()*x+b).sqrt() {
            let point = G2Affine::new_unchecked(x,y);
            // Full INTEGER r, not Fr::from(r), and no subgroup-only shortcut.
            if !point.mul_bigint(Fr::MODULUS).is_zero() { return point; }
        }
    }
    panic!("fixed-seed non-subgroup fixture search exhausted");
}

// Catches omitted parse/curve/full-order checks and partial invalid output.
#[test]
fn malformed_curve_and_non_subgroup_items_are_zeroed_then_retry_exactly() {
    let backend = IcicleBackend::new().unwrap();
    let generator = G2Affine::generator();
    let good = wire(&[generator]);
    let mut bad_cases = Vec::new();
    for coordinate in 0..4 {
        let mut bad = good.clone();
        bad[coordinate*32..(coordinate+1)*32].copy_from_slice(&Fq::MODULUS.to_bytes_le());
        bad_cases.push((bad,1));
    }
    let mut flags = good.clone(); flags[127] |= 0xc0; bad_cases.push((flags,1));
    bad_cases.push((wire(&[G2Affine::new_unchecked(Fq2::ONE,Fq2::ONE)]),2));
    let outside = non_subgroup(); assert!(outside.is_on_curve());
    bad_cases.push((wire(&[outside]),4));
    for (bad,mask) in bad_cases {
        let mut input = good.clone(); input.extend_from_slice(&bad); input.extend_from_slice(&good);
        let result = backend.inspect_bn254_lines(&input).unwrap();
        assert_eq!(result.statuses,[0,mask,0]);
        assert_eq!(result.counts,[BN254_LINE_COUNT as u32,0,BN254_LINE_COUNT as u32]);
        assert_point(&result,0,generator); assert_point(&result,2,generator);
        assert!(result.coefficients[BN254_LINE_ITEM_BYTES..2*BN254_LINE_ITEM_BYTES].iter().all(|&b|b==0));
        let retry = backend.inspect_bn254_lines(&wire(&[generator,generator,generator])).unwrap();
        for i in 0..3 { assert_point(&retry,i,generator); }
    }
}

// Catches infinity-slot leaks, broken limits, and retirement on host rejection.
#[test]
fn infinity_empty_and_maximum_batches_preserve_padding_and_owner() {
    let backend = IcicleBackend::new().unwrap();
    let empty = backend.inspect_bn254_lines(&[]).unwrap();
    assert!(empty.coefficients.is_empty() && empty.statuses.is_empty() && empty.counts.is_empty());
    let mut flagged = wire(&[G2Affine::generator()]); flagged[127]=(flagged[127]&0x3f)|0x40;
    let mut input = vec![0;128]; input.extend_from_slice(&wire(&[G2Affine::identity()])); input.extend_from_slice(&flagged);
    let result = backend.inspect_bn254_lines(&input).unwrap();
    assert_eq!(result.statuses,vec![BN254_LINES_INFINITY;3]);
    assert_eq!(result.counts,vec![0;3]); assert!(result.coefficients.iter().all(|&b|b==0));
    assert!(backend.inspect_bn254_lines(&[0;127]).is_err());
    assert!(backend.inspect_bn254_lines(&vec![0;9*128]).is_err());
    let points = vec![G2Affine::generator();MAX_BN254_LINE_ITEMS];
    let result = backend.inspect_bn254_lines(&wire(&points)).unwrap();
    for (i,&point) in points.iter().enumerate() { assert_point(&result,i,point); }
    backend.memory_info().unwrap();
}
