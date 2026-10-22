use super::*;
use crate::{BatchMsmRequest, OperatorBackend};
use ark_bn254::G1Affine;
use std::cell::{Cell, RefCell};

#[derive(Default)]
struct SpyBackend {
    calls: RefCell<Vec<(usize, bool)>>,
    fail_at: Cell<Option<usize>>,
}

impl SpyBackend {
    fn reset(&self, fail_at: Option<usize>) {
        self.calls.borrow_mut().clear();
        self.fail_at.set(fail_at);
    }
}

impl OperatorBackend for SpyBackend {
    fn msm(&self, points: &[G1Affine], scalars: &[Fr]) -> Result<G1Affine> {
        crate::dynaverse::ArkReferenceBackend.msm(points, scalars)
    }

    fn msm_batch(&self, request: &BatchMsmRequest<'_>) -> Result<Vec<G1Affine>> {
        crate::dynaverse::ArkReferenceBackend.msm_batch(request)
    }

    fn ntt(&self, values: &mut [Fr], inverse: bool) -> Result<()> {
        let call = {
            let mut calls = self.calls.borrow_mut();
            calls.push((values.len(), inverse));
            calls.len()
        };
        ensure!(self.fail_at.get() != Some(call), "injected FFT failure at call {call}");
        crate::dynaverse::ArkReferenceBackend.ntt(values, inverse)
    }
}

// Independent quadratic products/Horner evaluation do not use tree state or
// backend results to derive the expected answers.
fn product_oracle(left: &[Fr], right: &[Fr]) -> Vec<Fr> {
    if left.is_empty() || right.is_empty() { return vec![Fr::zero()]; }
    let mut product = vec![Fr::zero(); left.len() + right.len() - 1];
    for i in 0..left.len() {
        for j in 0..right.len() { product[i + j] += left[i] * right[j]; }
    }
    while product.len() > 1 && product.last().unwrap().is_zero() { product.pop(); }
    product
}

fn evaluate_oracle(polynomial: &[Fr], point: Fr) -> Fr {
    let mut value = Fr::zero();
    for &coefficient in polynomial.iter().rev() { value = value * point + coefficient; }
    value
}

fn coefficients(len: usize, offset: u64) -> Vec<Fr> {
    (0..len).map(|i| Fr::from(offset + i as u64)).collect()
}

#[test]
fn tiny_and_padded_products_submit_two_forward_and_one_inverse_ntt() {
    let backend = SpyBackend::default();
    for (left_len, right_len, fft_len) in [(1, 1, 1), (2, 3, 4), (16, 17, 32), (17, 33, 64), (33, 65, 128)] {
        let left = coefficients(left_len, 2);
        let right = coefficients(right_len, 5);
        backend.reset(None);
        let (result, work) = crate::workcount::measure(|| fft_multiply_with_backend(&left, &right, &backend));
        assert_eq!(result.unwrap(), product_oracle(&left, &right));
        assert_eq!(*backend.calls.borrow(), vec![(fft_len, false), (fft_len, false), (fft_len, true)]);
        assert_eq!(work.ntt_elements, (3 * fft_len) as u64);
        assert_eq!(work.max_operand, fft_len as u64);
    }
    backend.reset(None);
    assert_eq!(fft_multiply_with_backend(&[], &[Fr::one()], &backend).unwrap(), vec![Fr::zero()]);
    assert!(backend.calls.borrow().is_empty());
    assert_eq!(fft_multiply_with_backend(&[Fr::zero()], &[Fr::one()], &backend).unwrap(), vec![Fr::zero()]);
    assert_eq!(backend.calls.borrow().len(), 3, "a nonempty zero product still honors explicit backend dispatch");
}

#[test]
fn backend_trees_preserve_nonpower_point_order_and_all_polynomial_results() {
    let backend = SpyBackend::default();
    for size in [0usize, 1, 2, 3, 7, 17, 33] {
        let points: Vec<Fr> = (0..size).map(|i| Fr::from((5 * (size - i) + 2) as u64)).collect();
        let cpu = SubproductTree::new(&points).unwrap();
        backend.reset(None);
        let selected = SubproductTree::new_with_backend(&points, &backend).unwrap();
        assert_eq!(selected.vanishing(), cpu.vanishing());
        assert_eq!(selected.inverse_derivatives(), cpu.inverse_derivatives());
        if size > 1 { assert!(!backend.calls.borrow().is_empty()); }
        for degree in [0usize, size, 2 * size + 3] {
            let polynomial = coefficients(degree + 1, 3);
            let expected: Vec<Fr> = points.iter().map(|&point| evaluate_oracle(&polynomial, point)).collect();
            backend.reset(None);
            assert_eq!(selected.multipoint_eval_with_backend(&polynomial, &backend).unwrap(), expected);
            assert_eq!(selected.multipoint_eval_with_backend(&polynomial, &backend).unwrap(), cpu.multipoint_eval(&polynomial).unwrap());
            backend.reset(None);
            assert_eq!(selected.interpolate_with_backend(&expected, &backend).unwrap(), cpu.interpolate(&expected).unwrap());
            if size > 1 { assert!(!backend.calls.borrow().is_empty()); }
        }
        let weights = coefficients(size, 7);
        let mut expected = vec![Fr::zero(); size];
        for i in 0..size {
            for j in 0..size {
                if i != j { expected[i] += weights[j] / (points[i] - points[j]); }
            }
        }
        backend.reset(None);
        assert_eq!(selected.cauchy_sums_with_backend(&weights, &backend).unwrap(), expected);
        assert_eq!(selected.cauchy_sums_with_backend(&weights, &backend).unwrap(), cpu.cauchy_sums(&weights).unwrap());
        if size > 1 { assert!(!backend.calls.borrow().is_empty()); }
    }
}

#[test]
fn newton_reciprocal_products_also_use_the_selected_backend() {
    let backend = SpyBackend::default();
    let denominator = vec![-Fr::from(2u64), Fr::from(3u64), Fr::one()];
    let quotient = vec![Fr::from(5u64), Fr::from(7u64), Fr::from(11u64), Fr::from(13u64)];
    let numerator = product_oracle(&denominator, &quotient);
    assert_eq!(newton_divide_exact_with_backend(&numerator, &denominator, &backend).unwrap(), quotient);
    // Inverse lengths 1->2->4 require four products (2,2,4,8), followed
    // by quotient recovery and complete remainder validation (8,8).
    let sizes = [2usize, 2, 4, 8, 8, 8];
    let mut expected = Vec::new();
    for size in sizes { expected.extend([(size, false), (size, false), (size, true)]); }
    assert_eq!(*backend.calls.borrow(), expected);

    for (den_len, quotient_len) in [(1, 1), (1, 17), (3, 19), (17, 3), (19, 35)] {
        let denominator = coefficients(den_len, 2);
        let quotient = coefficients(quotient_len, 7);
        let numerator = product_oracle(&denominator, &quotient);
        backend.reset(None);
        assert_eq!(newton_divide_exact_with_backend(&numerator, &denominator, &backend).unwrap(), quotient);
        assert_eq!(newton_divide_exact_with_backend(&numerator, &denominator, &backend).unwrap(), newton_divide_exact(&numerator, &denominator).unwrap());
        assert!(!backend.calls.borrow().is_empty());
        if den_len > 1 {
            let mut corrupted = numerator;
            corrupted[0] += Fr::one();
            assert!(newton_divide_exact_with_backend(&corrupted, &denominator, &backend).is_err());
        }
    }
}

#[test]
fn transform_errors_propagate_at_every_product_phase_without_cpu_fallback() {
    let backend = SpyBackend::default();
    let left = coefficients(17, 2);
    let right = coefficients(17, 9);
    for fail_at in 1..=3 {
        backend.reset(Some(fail_at));
        let (result, work) = crate::workcount::measure(|| fft_multiply_with_backend(&left, &right, &backend));
        assert!(result.unwrap_err().to_string().contains("injected FFT failure"));
        assert_eq!(backend.calls.borrow().len(), fail_at);
        assert_eq!(work.ntt_elements, (fail_at * 64) as u64, "count only submitted transforms");
    }
    let points = coefficients(3, 11);
    let cpu = SubproductTree::new(&points).unwrap();
    backend.reset(Some(1));
    assert!(SubproductTree::new_with_backend(&points, &backend).unwrap_err().to_string().contains("injected FFT failure"));
    backend.reset(Some(1));
    assert!(cpu.multipoint_eval_with_backend(&coefficients(9, 1), &backend).unwrap_err().to_string().contains("injected FFT failure"));
    backend.reset(Some(1));
    assert!(cpu.interpolate_with_backend(&coefficients(3, 1), &backend).unwrap_err().to_string().contains("injected FFT failure"));
    backend.reset(Some(1));
    assert!(cpu.cauchy_sums_with_backend(&coefficients(3, 1), &backend).unwrap_err().to_string().contains("injected FFT failure"));
    let denominator = coefficients(3, 2);
    let numerator = product_oracle(&denominator, &coefficients(5, 1));
    for fail_at in [1, 4, 7, 10, 13, 16] {
        backend.reset(Some(fail_at));
        assert!(newton_divide_exact_with_backend(&numerator, &denominator, &backend).unwrap_err().to_string().contains("injected FFT failure"));
        assert_eq!(backend.calls.borrow().len(), fail_at);
    }
}

#[test]
fn invalid_inputs_fail_before_dispatch_and_cpu_small_products_keep_their_counters() {
    let backend = SpyBackend::default();
    let points = coefficients(3, 1);
    let cpu = SubproductTree::new(&points).unwrap();
    assert!(SubproductTree::new_with_backend(&[Fr::one(), Fr::one()], &backend).is_err());
    assert!(cpu.interpolate_with_backend(&[], &backend).is_err());
    assert!(cpu.cauchy_sums_with_backend(&[], &backend).is_err());
    assert!(newton_divide_exact_with_backend(&[Fr::one()], &[], &backend).is_err());
    assert!(newton_divide_exact_with_backend(&[Fr::one()], &[Fr::zero(), Fr::one()], &backend).is_err());
    assert_eq!(newton_divide_exact_with_backend(&[], &[Fr::one()], &backend).unwrap(), vec![Fr::zero()]);
    assert!(backend.calls.borrow().is_empty());

    let (product, work) = crate::workcount::measure(|| fft_multiply(&[Fr::one(); 2], &[Fr::one(); 2]));
    assert_eq!(product.unwrap(), vec![Fr::one(), Fr::from(2u64), Fr::one()]);
    assert_eq!(work.ntt_elements, 0);
    assert_eq!(work.poly_elements, 4);
    assert_eq!(work.max_operand, 2);
}
