use ark_bn254::Fr;
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use dynaverse_protocol::{dynamo::DynamoPermutation, srs::SharedSrs};
use rand::{rngs::StdRng, SeedableRng};

#[test]
fn shared_x_commitment_matches_dynamo_linear_commitment() {
    let mut rng = StdRng::seed_from_u64(7);
    let sigma = DynamoPermutation::try_new((0..16).collect()).unwrap();
    let srs = SharedSrs::setup(16, 2, sigma, &mut rng).unwrap();
    let values: Vec<Fr> = (0..16)
        .map(|i| Fr::from((i as u64 * i as u64 + 3) % 23))
        .collect();
    let domain = Radix2EvaluationDomain::<Fr>::new(16).unwrap();
    let coefficients = domain.ifft(&values);

    let dynamo_commitment = srs.dynamo.commit_z_dense_unmasked(&values).unwrap();
    let kzg_commitment = srs.kzg_x.commit(&coefficients).unwrap();
    assert_eq!(dynamo_commitment, kzg_commitment.point);
}

#[test]
fn shared_setup_rejects_invalid_shapes() {
    let mut rng = StdRng::seed_from_u64(17);
    let four = DynamoPermutation::try_new(vec![0, 1, 2, 3]).unwrap();
    assert!(SharedSrs::setup(0, 2, four.clone(), &mut rng).is_err());
    assert!(SharedSrs::setup(16, 3, four.clone(), &mut rng).is_err());
    assert!(SharedSrs::setup(16, 32, four.clone(), &mut rng).is_err());
    assert!(SharedSrs::setup(16, 2, four, &mut rng).is_err());
}
