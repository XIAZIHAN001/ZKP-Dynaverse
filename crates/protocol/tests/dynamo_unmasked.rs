use anyhow::Result;
use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::{CurveGroup, VariableBaseMSM};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use dynaverse_protocol::{
    dynamo::{derive_h_sparse, dynamo_prove_unmasked, dynamo_verify_with_key, DynamoPermutation, SparseVector},
    srs::SharedSrs,
    BatchMsmRequest, OperatorBackend,
};
use rand::{rngs::StdRng, SeedableRng};

struct ArkBackend;
impl OperatorBackend for ArkBackend {
    fn msm(&self, points: &[G1Affine], scalars: &[Fr]) -> Result<G1Affine> {
        Ok(G1Projective::msm_unchecked(points, scalars).into_affine())
    }
    fn msm_batch(&self, request: &BatchMsmRequest<'_>) -> Result<Vec<G1Affine>> {
        request.point_sets.iter().zip(request.offsets.iter().zip(request.lengths.iter()))
            .map(|(points, (&offset, &len))| self.msm(&points[..len], &request.shared_scalars[offset..offset + len]))
            .collect()
    }
    fn ntt(&self, values: &mut [Fr], inverse: bool) -> Result<()> {
        let domain = Radix2EvaluationDomain::<Fr>::new(values.len()).unwrap();
        let mut buffer = values.to_vec();
        if inverse { domain.ifft_in_place(&mut buffer); } else { domain.fft_in_place(&mut buffer); }
        values.copy_from_slice(&buffer);
        Ok(())
    }
}

#[test]
fn unmasked_instances_match_shared_kzg_and_use_only_vk() {
    let mut rng = StdRng::seed_from_u64(123);
    let sigma = DynamoPermutation::try_new((0..64).map(|i| if i == 0 { 16 } else if i == 16 { 0 } else { i }).collect()).unwrap();
    let srs = SharedSrs::setup(64, 8, sigma.clone(), &mut rng).unwrap();
    let z = SparseVector::try_new(64, vec![(0, Fr::from(3u64)), (3, Fr::from(9u64))]).unwrap();
    let h = derive_h_sparse(&z, &sigma).unwrap();
    let proof = dynamo_prove_unmasked(&srs.dynamo, &z, &h, &ArkBackend).unwrap();
    let mut dense_z = vec![Fr::from(0u64); 64];
    let mut dense_h = vec![Fr::from(0u64); 64];
    for &(i, v) in z.entries() { dense_z[i] = v; }
    for &(i, v) in h.entries() { dense_h[i] = v; }
    let domain = Radix2EvaluationDomain::<Fr>::new(64).unwrap();
    assert_eq!(proof.instance_z, srs.kzg_x.commit(&domain.ifft(&dense_z)).unwrap().point);
    assert_eq!(proof.instance_h, srs.kzg_y.commit(&domain.ifft(&dense_h)).unwrap().point);
    let vk = srs.dynamo.verifying_key();
    assert!(dynamo_verify_with_key(&vk, &proof).unwrap());
    let wrong_h = SparseVector::try_new(64, vec![(0, Fr::from(5u64))]).unwrap();
    let bad = dynamo_prove_unmasked(&srs.dynamo, &z, &wrong_h, &ArkBackend).unwrap();
    assert!(!dynamo_verify_with_key(&vk, &bad).unwrap());
}
