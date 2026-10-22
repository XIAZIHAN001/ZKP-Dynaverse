use ark_bn254::{Fq, Fr, G1Affine, G1Projective};
use ark_ec::{CurveGroup, Group};
use dynaverse_gpu::ArkworksBackend;
use dynaverse_protocol::{BatchMsmRequest, OperatorBackend};

#[test]
fn arkworks_msm_matches_group_identity_and_rejects_invalid_inputs() {
    let generator = G1Projective::generator();
    let points = [generator.into_affine(), (generator + generator).into_affine(), G1Affine::identity()];
    let scalars = [Fr::from(2u64), Fr::from(3u64), Fr::from(999u64)];
    // 2*G + 3*(2*G) + 999*identity = 8*G, independent of MSM code.
    assert_eq!(ArkworksBackend.msm(&points, &scalars).unwrap(), (generator * Fr::from(8u64)).into_affine());
    assert_eq!(ArkworksBackend.msm(&[], &[]).unwrap(), G1Affine::identity());
    assert!(ArkworksBackend.msm(&points, &scalars[..2]).is_err());
    let invalid = G1Affine::new_unchecked(Fq::from(1u64), Fq::from(1u64));
    assert!(ArkworksBackend.msm(&[invalid], &[Fr::from(0u64)]).is_err());
}

#[test]
fn arkworks_batch_obeys_lengths_offsets_and_rejects_bad_ranges() {
    let generator = G1Projective::generator();
    let points = [generator.into_affine(); 3];
    let sets: [&[G1Affine]; 2] = [&points, &points];
    let scalars = [Fr::from(9u64), Fr::from(2u64), Fr::from(3u64)];
    let request = BatchMsmRequest { point_sets: &sets, shared_scalars: &scalars, offsets: &[1, 0], lengths: &[2, 1] };
    assert_eq!(ArkworksBackend.msm_batch(&request).unwrap(), vec![
        (generator * Fr::from(5u64)).into_affine(), (generator * Fr::from(9u64)).into_affine()]);
    for (offsets, lengths) in [(&[usize::MAX, 0][..], &[2, 1][..]),
        (&[2, 0][..], &[2, 1][..]), (&[0][..], &[2, 1][..]), (&[0, 0][..], &[4, 1][..])] {
        assert!(ArkworksBackend.msm_batch(&BatchMsmRequest {
            point_sets: &sets, shared_scalars: &scalars, offsets, lengths }).is_err());
    }
}
