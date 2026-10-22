//! Pending GPU arithmetic admission tests. These are not performance claims.
#![cfg(feature = "icicle")]

use ark_bn254::{Fq, Fq2, Fq6, Fq12};
use ark_ff::{BigInteger, Field, PrimeField, UniformRand};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use dynaverse_gpu::bn254_tower::{
    Bn254TowerField, Bn254TowerOperation, BN254_TOWER_NON_CANONICAL,
    BN254_TOWER_ZERO_INVERSE, MAX_BN254_TOWER_ITEMS,
};
use dynaverse_gpu::IcicleBackend;
use rand::{rngs::StdRng, SeedableRng};

fn encode<T: CanonicalSerialize>(values: &[T]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for value in values {
        value.serialize_uncompressed(&mut bytes).unwrap();
    }
    bytes
}

fn decode<T: CanonicalDeserialize>(bytes: &[u8]) -> T {
    // The independent decoder rejects noncanonical output coordinates.
    let mut input = bytes;
    let value = T::deserialize_uncompressed(&mut input).unwrap();
    assert!(input.is_empty());
    value
}

// Catches component order, nonresidue, general multiplication/inversion and
// Frobenius mistakes. Expectations use only the retained arkworks CPU oracle.
fn compare_field<T: Field + UniformRand>(field: Bn254TowerField, seed: u64) {
    let backend = IcicleBackend::new().unwrap();
    let mut rng = StdRng::seed_from_u64(seed);
    let maximum = encode(&[-Fq::ONE]);
    let mut all_maximum = Vec::new();
    for _ in 0..field.element_bytes()/32 {
        all_maximum.extend_from_slice(&maximum);
    }
    let mut left = vec![T::ZERO, T::ONE, -T::ONE, decode::<T>(&all_maximum)];
    let mut right = vec![T::ONE, T::ZERO, T::ONE, T::ONE];
    for _ in 0..31 {
        left.push(T::rand(&mut rng));
        right.push(T::rand(&mut rng));
    }
    let left_bytes = encode(&left);
    let right_bytes = encode(&right);
    let mut operations = vec![
        Bn254TowerOperation::Add,
        Bn254TowerOperation::Subtract,
        Bn254TowerOperation::Multiply,
        Bn254TowerOperation::Square,
        Bn254TowerOperation::Inverse,
    ];
    for power in 0..=12 {
        operations.push(Bn254TowerOperation::Frobenius(power));
    }
    for operation in operations {
        let binary = matches!(operation, Bn254TowerOperation::Add
            | Bn254TowerOperation::Subtract | Bn254TowerOperation::Multiply);
        let rhs = if binary { right_bytes.as_slice() } else { &[] };
        let result = backend.inspect_bn254_tower(field, operation, &left_bytes, rhs).unwrap();
        assert_eq!(result.statuses.len(), left.len());
        assert_eq!(result.output.len(), left_bytes.len());
        for index in 0..left.len() {
            let a = left[index];
            let b = right[index];
            let expected = match operation {
                Bn254TowerOperation::Add => Some(a + b),
                Bn254TowerOperation::Subtract => Some(a - b),
                Bn254TowerOperation::Multiply => Some(a * b),
                Bn254TowerOperation::Square => Some(a.square()),
                Bn254TowerOperation::Inverse => a.inverse(),
                Bn254TowerOperation::Frobenius(power) => {
                    let mut value = a;
                    value.frobenius_map_in_place(power as usize);
                    Some(value)
                }
            };
            let width = field.element_bytes();
            let actual = decode::<T>(&result.output[index * width..(index + 1) * width]);
            if let Some(expected) = expected {
                assert_eq!(result.statuses[index], 0, "{field:?} {operation:?} item {index}");
                assert_eq!(actual, expected, "{field:?} {operation:?} item {index}");
            } else {
                assert_eq!(result.statuses[index], BN254_TOWER_ZERO_INVERSE);
                assert_eq!(actual, T::ZERO);
            }
        }
    }
    // This checks the Frobenius independently of its coefficient tables.
    let sample = left[4];
    let result = backend.inspect_bn254_tower(field, Bn254TowerOperation::Frobenius(1),
        &encode(&[sample]), &[]).unwrap();
    assert_eq!(result.statuses, [0]);
    assert_eq!(decode::<T>(&result.output), sample.pow(Fq::MODULUS));
}

#[test]
fn fq2_full_width_arithmetic_matches_cpu() {
    compare_field::<Fq2>(Bn254TowerField::Fq2, 31002);
}

#[test]
fn fq6_full_width_arithmetic_matches_cpu() {
    compare_field::<Fq6>(Bn254TowerField::Fq6, 31006);
}

#[test]
fn fq12_full_width_arithmetic_matches_cpu() {
    compare_field::<Fq12>(Bn254TowerField::Fq12, 31012);
}

#[test]
fn tower_basis_relations_have_literal_expected_values() {
    let backend = IcicleBackend::new().unwrap();
    let u = Fq2::new(Fq::ZERO, Fq::ONE);
    let result = backend.inspect_bn254_tower(Bn254TowerField::Fq2,
        Bn254TowerOperation::Square, &encode(&[u]), &[]).unwrap();
    assert_eq!(result.statuses, [0]);
    assert_eq!(decode::<Fq2>(&result.output), Fq2::new(-Fq::ONE, Fq::ZERO));

    let v = Fq6::new(Fq2::ZERO, Fq2::ONE, Fq2::ZERO);
    let square = backend.inspect_bn254_tower(Bn254TowerField::Fq6,
        Bn254TowerOperation::Square, &encode(&[v]), &[]).unwrap();
    assert_eq!(square.statuses, [0]);
    let cube = backend.inspect_bn254_tower(Bn254TowerField::Fq6,
        Bn254TowerOperation::Multiply, &square.output, &encode(&[v])).unwrap();
    assert_eq!(cube.statuses, [0]);
    assert_eq!(decode::<Fq6>(&cube.output),
        Fq6::new(Fq2::new(Fq::from(9u64), Fq::ONE), Fq2::ZERO, Fq2::ZERO));

    let w = Fq12::new(Fq6::ZERO, Fq6::ONE);
    let result = backend.inspect_bn254_tower(Bn254TowerField::Fq12,
        Bn254TowerOperation::Square, &encode(&[w]), &[]).unwrap();
    assert_eq!(result.statuses, [0]);
    assert_eq!(decode::<Fq12>(&result.output), Fq12::new(v, Fq6::ZERO));
}

#[test]
fn every_coefficient_rejects_modulus_and_high_flags_without_losing_other_items() {
    let backend = IcicleBackend::new().unwrap();
    let modulus = Fq::MODULUS.to_bytes_le();
    for field in [Bn254TowerField::Fq2, Bn254TowerField::Fq6, Bn254TowerField::Fq12] {
        let width = field.element_bytes();
        // Four AoS items: valid one, bad left, bad right, valid one.
        for coefficient in 0..width / 32 {
            let mut left = vec![0; 4 * width];
            let mut right = vec![0; 4 * width];
            left[0] = 1;
            left[3 * width] = 1;
            let mut expected = vec![0; 4 * width];
            expected[0] = 1;
            expected[3 * width] = 1;
            let offset = width + coefficient * 32;
            left[offset..offset + 32].copy_from_slice(&modulus);
            right[2 * width + coefficient * 32 + 31] = 0xc0;
            let result = backend.inspect_bn254_tower(field, Bn254TowerOperation::Add,
                &left, &right).unwrap();
            assert_eq!(result.statuses, [0, BN254_TOWER_NON_CANONICAL,
                BN254_TOWER_NON_CANONICAL, 0]);
            assert_eq!(result.output, expected);
            left[offset..offset + 32].fill(0);
            right[2 * width + coefficient * 32 + 31] = 0;
            let retry = backend.inspect_bn254_tower(field, Bn254TowerOperation::Add,
                &left, &right).unwrap();
            assert_eq!(retry.statuses, [0; 4]);
            assert_eq!(retry.output, expected);
        }
    }
}

#[test]
fn bounded_empty_and_rejected_calls_preserve_the_owner() {
    let backend = IcicleBackend::new().unwrap();
    for field in [Bn254TowerField::Fq2, Bn254TowerField::Fq6, Bn254TowerField::Fq12] {
        let empty = backend.inspect_bn254_tower(field, Bn254TowerOperation::Add, &[], &[]).unwrap();
        assert!(empty.output.is_empty() && empty.statuses.is_empty());
        let width = field.element_bytes();
        let left = vec![0; width * MAX_BN254_TOWER_ITEMS];
        let result = backend.inspect_bn254_tower(field, Bn254TowerOperation::Square, &left, &[]).unwrap();
        assert_eq!(result.output, left);
        assert_eq!(result.statuses, vec![0; MAX_BN254_TOWER_ITEMS]);
        assert!(backend.inspect_bn254_tower(field, Bn254TowerOperation::Square,
            &vec![0; width * (MAX_BN254_TOWER_ITEMS + 1)], &[]).is_err());
        assert!(backend.inspect_bn254_tower(field, Bn254TowerOperation::Square, &[0], &[]).is_err());
        assert!(backend.inspect_bn254_tower(field, Bn254TowerOperation::Multiply, &left, &[]).is_err());
        assert!(backend.inspect_bn254_tower(field, Bn254TowerOperation::Square, &left, &left).is_err());
        assert!(backend.inspect_bn254_tower(field, Bn254TowerOperation::Frobenius(13), &left, &[]).is_err());
        backend.memory_info().unwrap();
    }
}
