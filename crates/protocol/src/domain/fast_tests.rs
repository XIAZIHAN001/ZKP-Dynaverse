use super::*;
use ark_std::UniformRand;
use rand::{rngs::StdRng, SeedableRng};

// These oracles use the retained quadratic algorithms, not tree intermediates.
fn naive_interpolate(points: &[Fr], values: &[Fr]) -> Vec<Fr> {
    let mut result = vec![Fr::zero(); points.len().max(1)];
    for i in 0..points.len() {
        let mut numerator = vec![Fr::one()];
        let mut denominator = Fr::one();
        for j in 0..points.len() {
            if i != j {
                numerator = multiply_polynomials(&numerator, &[-points[j], Fr::one()]);
                denominator *= points[i] - points[j];
            }
        }
        let weight = values[i] * denominator.inverse().unwrap();
        for j in 0..numerator.len() { result[j] += weight * numerator[j]; }
    }
    trim(&mut result);
    result
}

#[test]
fn tree_product_evaluation_and_interpolation_match_independent_oracles() {
    let mut rng = StdRng::seed_from_u64(15201);
    for size in [0usize, 1, 2, 3, 16, 17, 33, 65] {
        let points: Vec<Fr> = (0..size).map(|i| Fr::from((3 * i + 2) as u64)).collect();
        let tree = SubproductTree::new(&points).unwrap();
        let mut vanishing = vec![Fr::one()];
        for &x in &points { vanishing = multiply_polynomials(&vanishing, &[-x, Fr::one()]); }
        assert_eq!(tree.vanishing(), vanishing);
        for degree in [0usize, size, 2 * size + 3] {
            let polynomial: Vec<Fr> = (0..=degree).map(|_| Fr::rand(&mut rng)).collect();
            let values: Vec<Fr> = points.iter().map(|&x| evaluate_polynomial(&polynomial, x)).collect();
            assert_eq!(tree.multipoint_eval(&polynomial).unwrap(), values);
            assert_eq!(tree.interpolate(&values).unwrap(), naive_interpolate(&points, &values));
        }
        assert_eq!(tree.interpolate(&vec![Fr::zero(); size]).unwrap(), vec![Fr::zero()]);
        assert!(tree.interpolate(&vec![Fr::zero(); size + 1]).is_err());
    }
    assert!(SubproductTree::new(&[Fr::one(), Fr::one()]).is_err());
}

#[test]
fn cauchy_sums_exclude_the_diagonal_in_input_order() {
    let mut rng = StdRng::seed_from_u64(15202);
    for size in [0usize, 1, 2, 3, 17, 32, 65] {
        let points: Vec<Fr> = (0..size).map(|i| Fr::from((2 * (size - i) + 1) as u64)).collect();
        let weights: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut rng)).collect();
        let mut expected = vec![Fr::zero(); size];
        for i in 0..size {
            for j in 0..size {
                if i != j { expected[i] += weights[j] / (points[i] - points[j]); }
            }
        }
        let tree = SubproductTree::new(&points).unwrap();
        assert_eq!(tree.cauchy_sums(&weights).unwrap(), expected);
        assert_eq!(tree.cauchy_sums(&vec![Fr::zero(); size]).unwrap(), vec![Fr::zero(); size]);
        assert!(tree.cauchy_sums(&vec![Fr::zero(); size + 1]).is_err());
    }
}

#[test]
fn newton_division_matches_dense_quotients_and_rejects_each_remainder() {
    let mut rng = StdRng::seed_from_u64(15203);
    for degree in [0usize, 1, 3, 17, 33, 65] {
        let mut denominator: Vec<Fr> = (0..degree).map(|_| Fr::rand(&mut rng)).collect();
        denominator.push(Fr::from(3u64));
        for len in [1usize, 3, 17, 67] {
            let quotient: Vec<Fr> = (0..len).map(|_| Fr::rand(&mut rng)).collect();
            let numerator = multiply_polynomials(&denominator, &quotient);
            assert_eq!(newton_divide_exact(&numerator, &denominator).unwrap(), quotient);
            assert_eq!(newton_divide_exact(&numerator, &denominator).unwrap(),
                divide_exact(&numerator, &denominator).unwrap());
            for index in 0..degree {
                let mut corrupted = numerator.clone();
                corrupted[index] += Fr::one();
                assert!(newton_divide_exact(&corrupted, &denominator).is_err());
            }
        }
    }
    assert_eq!(newton_divide_exact(&[], &[Fr::one()]).unwrap(), vec![Fr::zero()]);
    assert!(newton_divide_exact(&[], &[]).is_err());
    assert!(newton_divide_exact(&[Fr::one()], &[Fr::zero(), Fr::one()]).is_err());
}

#[test]
fn fft_product_records_all_three_transform_lengths() {
    let left = vec![Fr::one(); 17];
    let right = vec![Fr::from(2u64); 17];
    let (product, work) = crate::workcount::measure(|| fft_multiply(&left, &right));
    assert_eq!(product.unwrap(), multiply_polynomials(&left, &right));
    assert_eq!(work.ntt_elements, 3 * 64, "two forward FFTs and one inverse FFT, length 64 each");
    assert_eq!(work.max_operand, 64);
}
