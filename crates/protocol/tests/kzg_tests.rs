use ark_bn254::{Fr, G1Projective};
use ark_ec::{CurveGroup, Group};
use dynaverse_protocol::domain::radix2_domain;
use dynaverse_protocol::kzg::{combine_single_openings, KzgParams};
use ark_poly::EvaluationDomain;
use rand::{rngs::StdRng, SeedableRng};

fn fixture() -> (KzgParams, Vec<Fr>) {
    let mut rng = StdRng::seed_from_u64(7);
    let params = KzgParams::setup(16, &mut rng).unwrap();
    let coeffs = vec![
        Fr::from(3u64),
        Fr::from(2u64),
        Fr::from(5u64),
        Fr::from(7u64),
    ];
    (params, coeffs)
}

#[test]
fn single_opening_accepts_valid_and_rejects_tampered_value() {
    let (params, coeffs) = fixture();
    let domain = radix2_domain(16).unwrap();
    let commitment = params.commit(&coeffs).unwrap();
    let opening = params.open_single(&coeffs, &domain, 4).unwrap();
    assert!(params
        .verify_single(&commitment, &domain, 4, &opening)
        .unwrap());
    let mut bad = opening;
    bad.values[0] += Fr::from(1u64);
    assert!(!params.verify_single(&commitment, &domain, 4, &bad).unwrap());
}

#[test]
fn opening_reports_a_hand_calculated_evaluation() {
    let (params, coeffs) = fixture();
    let domain = radix2_domain(16).unwrap();
    // The first domain point is 1; f(1) = 3 + 2 + 5 + 7 = 17.
    let opening = params.open_single(&coeffs, &domain, 0).unwrap();
    assert_eq!(opening.values, vec![Fr::from(17u64)]);
}

#[test]
fn kzg_batch_opening_accepts_valid_and_rejects_tampered_value() {
    let (params, coeffs) = fixture();
    let domain = radix2_domain(16).unwrap();
    let commitment = params.commit(&coeffs).unwrap();
    let indices = [9usize, 1, 4];
    let opening = params.open_batch(&coeffs, &domain, &indices).unwrap();
    assert!(params
        .verify_batch(&commitment, &domain, &indices, &opening)
        .unwrap());
    let mut bad = opening.clone();
    bad.values[1] += Fr::from(1u64);
    assert!(!params
        .verify_batch(&commitment, &domain, &indices, &bad)
        .unwrap());

    let mut bad_proof = opening.clone();
    bad_proof.witness =
        (G1Projective::from(bad_proof.witness) + G1Projective::generator()).into_affine();
    assert!(!params
        .verify_batch(&commitment, &domain, &indices, &bad_proof)
        .unwrap());

    let mut bad_commitment = commitment;
    bad_commitment.point =
        (G1Projective::from(bad_commitment.point) + G1Projective::generator()).into_affine();
    assert!(!params
        .verify_batch(&bad_commitment, &domain, &indices, &opening)
        .unwrap());
}

#[test]
fn batch_opening_rejects_empty_duplicate_and_out_of_range_indices() {
    let (params, coeffs) = fixture();
    let domain = radix2_domain(16).unwrap();
    assert!(params.open_batch(&coeffs, &domain, &[]).is_err());
    assert!(params.open_batch(&coeffs, &domain, &[2, 2]).is_err());
    assert!(params.open_batch(&coeffs, &domain, &[16]).is_err());
}

#[test]
fn commitment_and_opening_enforce_srs_degree_bound() {
    let mut rng = StdRng::seed_from_u64(17);
    let params = KzgParams::setup(2, &mut rng).unwrap();
    assert!(params.commit(&vec![Fr::from(1u64); 4]).is_err());
    let domain = radix2_domain(4).unwrap();
    assert!(params
        .open_batch(&[Fr::from(1u64)], &domain, &[0, 1, 2])
        .is_err());
}

#[test]
fn precomputed_openings_match_dense_single_and_batch_witnesses() {
    let mut rng = StdRng::seed_from_u64(2801);
    for n in [1usize, 2, 8, 32] {
        let params = KzgParams::setup(n, &mut rng).unwrap();
        let domain = radix2_domain(n).unwrap();
        for coefficients in [vec![], vec![Fr::from(11u64)],
            (0..n).map(|i| Fr::from((i * i + 3 * i + 7) as u64)).collect()] {
            let cache = params.precompute_single_openings(&coefficients, &domain).unwrap();
            assert_eq!(cache.len(), n);
            for i in 0..n {
                assert_eq!(cache[i], params.open_single(&coefficients, &domain, i).unwrap().witness);
            }
            let indices: Vec<usize> = (0..n).rev().step_by(2).collect();
            let dense = params.open_batch(&coefficients, &domain, &indices).unwrap();
            assert_eq!(combine_single_openings(&domain, &indices, &cache).unwrap(), dense.witness);
            let full: Vec<usize> = (0..n).rev().collect();
            assert_eq!(combine_single_openings(&domain, &full, &cache).unwrap(),
                params.open_batch(&coefficients, &domain, &full).unwrap().witness);
        }
    }
}

#[test]
fn cached_openings_reject_bad_shapes_and_cannot_hide_corruption() {
    let (params, coefficients) = fixture();
    let domain = radix2_domain(16).unwrap();
    let mut cache = params.precompute_single_openings(&coefficients, &domain).unwrap();
    for indices in [vec![], vec![2, 2], vec![domain.size()]] {
        assert!(combine_single_openings(&domain, &indices, &cache).is_err());
    }
    assert!(combine_single_openings(&domain, &[2], &cache[..15]).is_err());
    assert!(params.precompute_single_openings(&vec![Fr::from(1u64); 17], &domain).is_err());
    let mut opening = params.open_batch(&coefficients, &domain, &[2, 5]).unwrap();
    cache[2] = (G1Projective::from(cache[2]) + G1Projective::generator()).into_affine();
    opening.witness = combine_single_openings(&domain, &[2, 5], &cache).unwrap();
    assert!(!params.verify_batch(&params.commit(&coefficients).unwrap(), &domain, &[2, 5], &opening).unwrap());
}
