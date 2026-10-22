use anyhow::{ensure, Result};
use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::{CurveGroup, Group, VariableBaseMSM};
use ark_ff::Zero;
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use dynaverse_protocol::{
    dynamo::DynamoPermutation, srs::SharedSrs, BatchMsmRequest, ConsistencyProof,
    OperatorBackend,
};
use rand::{rngs::StdRng, SeedableRng};

struct ArkBackend;

impl OperatorBackend for ArkBackend {
    fn msm(&self, points: &[G1Affine], scalars: &[Fr]) -> Result<G1Affine> {
        ensure!(points.len() == scalars.len(), "MSM length mismatch");
        Ok(G1Projective::msm_unchecked(points, scalars).into_affine())
    }

    fn msm_batch(&self, request: &BatchMsmRequest<'_>) -> Result<Vec<G1Affine>> {
        ensure!(
            request.point_sets.len() == request.offsets.len()
                && request.point_sets.len() == request.lengths.len(),
            "batch shape mismatch"
        );
        request
            .point_sets
            .iter()
            .zip(request.offsets.iter().zip(request.lengths.iter()))
            .map(|(points, (&offset, &len))| {
                ensure!(len <= points.len(), "point range out of bounds");
                ensure!(offset.checked_add(len).is_some_and(|end| end <= request.shared_scalars.len()),
                    "scalar range out of bounds");
                self.msm(&points[..len], &request.shared_scalars[offset..offset + len])
            })
            .collect()
    }

    fn ntt(&self, values: &mut [Fr], inverse: bool) -> Result<()> {
        let domain = Radix2EvaluationDomain::<Fr>::new(values.len())
            .ok_or_else(|| anyhow::anyhow!("invalid NTT length"))?;
        let mut buffer = values.to_vec();
        if inverse {
            domain.ifft_in_place(&mut buffer);
        } else {
            domain.fft_in_place(&mut buffer);
        }
        values.copy_from_slice(&buffer);
        Ok(())
    }
}

fn setup_for(m: usize, n: usize) -> SharedSrs {
    let mut rng = StdRng::seed_from_u64(2026);
    let sigma = DynamoPermutation::try_new((0..m).collect()).unwrap();
    SharedSrs::setup(m, n, sigma, &mut rng).unwrap()
}

fn setup() -> SharedSrs {
    setup_for(64, 8)
}

fn commitment(srs: &SharedSrs, values: &[Fr]) -> G1Affine {
    let domain = Radix2EvaluationDomain::<Fr>::new(values.len()).unwrap();
    srs.kzg_x.commit(&domain.ifft(values)).unwrap().point
}

#[test]
fn slice_prover_accepts_values_outside_slice() {
    let srs = setup();
    let mut full = vec![Fr::zero(); 64];
    full[3] = Fr::from(4u64);
    full[16] = Fr::from(7u64);
    full[23] = Fr::from(9u64);
    let small = full[16..24].to_vec();
    let proof = srs.consistency[2].prove(&full, &small, &ArkBackend).unwrap();
    assert_ne!(proof.qac, G1Affine::identity());
    assert_ne!(proof.d, G1Affine::identity());
    assert_ne!(proof.a_degree, G1Affine::identity());
}

#[test]
fn zero_slice_and_sparse_delta_have_well_formed_commitments() {
    let srs = setup();
    let mut full = vec![Fr::zero(); 64];
    full[2] = Fr::from(11u64);
    let zero_small = vec![Fr::zero(); 8];
    let zero_proof = srs.consistency[2]
        .prove(&full, &zero_small, &ArkBackend)
        .unwrap();
    assert_eq!(zero_proof.d, G1Affine::identity());
    assert_eq!(zero_proof.b_degree, G1Affine::identity());
    assert_ne!(zero_proof.qac, G1Affine::identity());

    full[20] = Fr::from(5u64);
    let mut delta_small = vec![Fr::zero(); 8];
    delta_small[4] = Fr::from(5u64);
    let delta_proof = srs.consistency[2]
        .prove(&full, &delta_small, &ArkBackend)
        .unwrap();
    assert_ne!(delta_proof.d, G1Affine::identity());
}

#[test]
fn public_verifier_accepts_matching_slice_and_rejects_every_changed_point() {
    let srs = setup();
    let mut full = vec![Fr::zero(); 64];
    full[3] = Fr::from(4u64);
    full[16] = Fr::from(7u64);
    full[23] = Fr::from(9u64);
    let small = full[16..24].to_vec();
    let a = commitment(&srs, &full);
    let b = commitment(&srs, &small);
    let proof = srs.consistency[2].prove(&full, &small, &ArkBackend).unwrap();
    let vk = &srs.consistency[2].verifying;
    assert!(vk.verify(a, b, &proof));

    let g = G1Projective::generator();
    for field in 0..7 {
        let mut changed = proof;
        let point = match field {
            0 => &mut changed.qac,
            1 => &mut changed.qcd,
            2 => &mut changed.qbd,
            3 => &mut changed.d,
            4 => &mut changed.a_degree,
            5 => &mut changed.b_degree,
            _ => &mut changed.d_degree,
        };
        *point = (G1Projective::from(*point) + g).into_affine();
        assert!(!vk.verify(a, b, &changed), "field {field} was accepted");
    }
}

#[test]
fn public_verifier_rejects_wrong_slice_offset_and_srs() {
    let srs = setup();
    let mut full = vec![Fr::zero(); 64];
    full[8] = Fr::from(3u64);
    full[16] = Fr::from(7u64);
    full[23] = Fr::from(9u64);
    let small = full[16..24].to_vec();
    let a = commitment(&srs, &full);
    let b = commitment(&srs, &small);
    let proof = srs.consistency[2].prove(&full, &small, &ArkBackend).unwrap();
    assert!(srs.consistency[2].verifying.verify(a, b, &proof));

    let mut wrong = small.clone();
    wrong[0] += Fr::from(1u64);
    let wrong_b = commitment(&srs, &wrong);
    assert!(!srs.consistency[2].verifying.verify(a, wrong_b, &proof));
    let wrong_proof = srs.consistency[2].prove(&full, &wrong, &ArkBackend).unwrap();
    assert!(!srs.consistency[2].verifying.verify(a, wrong_b, &wrong_proof));
    assert!(!srs.consistency[1].verifying.verify(a, b, &proof));

    let mut other_rng = StdRng::seed_from_u64(17);
    let sigma = DynamoPermutation::try_new((0..64).collect()).unwrap();
    let other = SharedSrs::setup(64, 8, sigma, &mut other_rng).unwrap();
    assert!(!other.consistency[2].verifying.verify(a, b, &proof));
}

#[test]
fn six_slice_verification_uses_only_public_commitments() {
    for (m, n) in [(64, 8), (128, 16), (256, 32)] {
        let srs = setup_for(m, n);
        let mut full = vec![Fr::zero(); m];
        full[3] = Fr::from(7u64);
        full[2 * n + 4] = Fr::from(11u64);
        full[5 * n + 3] = Fr::from(13u64);
        let a = commitment(&srs, &full);
        let slices: [Vec<Fr>; 6] =
            std::array::from_fn(|t| full[t * n..(t + 1) * n].to_vec());
        let bs: [G1Affine; 6] = std::array::from_fn(|t| commitment(&srs, &slices[t]));
        let proof = ConsistencyProof {
            slices: std::array::from_fn(|t| {
                srs.consistency[t].prove(&full, &slices[t], &ArkBackend).unwrap()
            }),
        };
        let vk = srs.consistency_verifying_key();
        assert!(vk.verify_six(a, &bs, &proof), "m={m}, n={n}");
        let mut wrong_metadata = vk.clone();
        wrong_metadata.slices[2].offset += n;
        assert!(!wrong_metadata.verify_six(a, &bs, &proof), "offset metadata must bind");
        let mut wrong_metadata = vk.clone();
        wrong_metadata.slices[5].n = n / 2;
        assert!(!wrong_metadata.verify_six(a, &bs, &proof), "size metadata must bind");
        let mut changed = bs;
        changed[4] =
            (G1Projective::from(changed[4]) + G1Projective::generator()).into_affine();
        assert!(!vk.verify_six(a, &changed, &proof), "m={m}, n={n}");
    }
}
