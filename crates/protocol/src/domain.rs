//! Polynomial-domain helpers used by the CPU Dynaverse reference prover.

use anyhow::{bail, ensure, Result};
use ark_bn254::Fr;
use ark_ff::{Field, One, Zero};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use std::collections::HashSet;

mod fast;
pub use fast::{fft_multiply_with_backend, newton_divide_exact, newton_divide_exact_with_backend, SubproductTree};

#[cfg(test)]
#[path = "domain/fast_tests.rs"]
mod fast_tests;

pub fn radix2_domain(size: usize) -> Result<Radix2EvaluationDomain<Fr>> {
    ensure!(
        size != 0 && size.is_power_of_two(),
        "domain size must be a nonzero power of two"
    );
    Radix2EvaluationDomain::<Fr>::new(size)
        .ok_or_else(|| anyhow::anyhow!("BN254 does not support a domain of size {size}"))
}

/// Pad a value vector to a nonempty radix-2 domain, preserving every input.
pub fn pad_to_power_of_two(values: &[Fr]) -> Result<Vec<Fr>> {
    let size = values
        .len()
        .max(1)
        .checked_next_power_of_two()
        .ok_or_else(|| anyhow::anyhow!("value count is too large to pad"))?;
    let mut out = values.to_vec();
    out.resize(size, Fr::zero());
    Ok(out)
}

pub fn index_points(domain: &Radix2EvaluationDomain<Fr>, indices: &[usize]) -> Result<Vec<Fr>> {
    let mut seen = HashSet::with_capacity(indices.len());
    let mut points = Vec::with_capacity(indices.len());
    for &index in indices {
        ensure!(
            index < domain.size(),
            "domain index {index} is out of range"
        );
        ensure!(seen.insert(index), "duplicate domain index {index}");
        points.push(domain.element(index));
    }
    Ok(points)
}

/// Coefficients of A_I(X) = product_{i in I}(X - omega^i), low degree first.
pub fn vanishing_polynomial(
    domain: &Radix2EvaluationDomain<Fr>,
    indices: &[usize],
) -> Result<Vec<Fr>> {
    let roots = index_points(domain, indices)?;
    let mut out = vec![Fr::one()];
    for root in roots {
        out = multiply_polynomials(&out, &[-root, Fr::one()]);
    }
    Ok(out)
}

/// Product-only balanced tree: O(M(s) log s) work and O(s) live
/// coefficients. Setup needs no multipoint evaluations or derivative cache.
pub fn vanishing_polynomial_balanced(
    domain: &Radix2EvaluationDomain<Fr>, indices: &[usize],
) -> Result<Vec<Fr>> {
    let roots = index_points(domain, indices)?;
    let mut level: Vec<Vec<Fr>> = roots.iter().map(|&x| vec![-x, Fr::one()]).collect();
    if level.is_empty() { return Ok(vec![Fr::one()]); }
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        for pair in level.chunks(2) {
            next.push(if pair.len() == 2 { fft_multiply(&pair[0], &pair[1])? }
                else { pair[0].clone() });
        }
        level = next;
    }
    Ok(level.pop().unwrap())
}

/// Split the small-domain vanishing polynomial into the affected roots and
/// their complement. The empty affected set has A_I=1.
pub fn affected_set_factorization(
    domain: &Radix2EvaluationDomain<Fr>,
    affected: &[usize],
) -> Result<(Vec<Fr>, Vec<Fr>)> {
    let ai = vanishing_polynomial(domain, affected)?;
    let mut all = vec![Fr::zero(); domain.size() + 1];
    all[0] = -Fr::one();
    all[domain.size()] = Fr::one();
    let bi = divide_exact(&all, &ai)?;
    Ok((ai, bi))
}

pub fn vanishing_derivative_values(
    domain: &Radix2EvaluationDomain<Fr>,
    indices: &[usize],
) -> Result<Vec<Fr>> {
    let points = index_points(domain, indices)?;
    let a = vanishing_polynomial(domain, indices)?;
    crate::workcount::record_poly(points.len() * a.len(), points.len());
    let derivative: Vec<Fr> = a
        .iter()
        .enumerate()
        .skip(1)
        .map(|(degree, coefficient)| *coefficient * Fr::from(degree as u64))
        .collect();
    Ok(points
        .into_iter()
        .map(|x| evaluate_polynomial(&derivative, x))
        .collect())
}

/// Interpolate the unique polynomial of degree < |I| at selected domain points.
pub fn interpolate_at_indices(
    domain: &Radix2EvaluationDomain<Fr>,
    indices: &[usize],
    values: &[Fr],
) -> Result<Vec<Fr>> {
    let points = index_points(domain, indices)?;
    ensure!(
        points.len() == values.len(),
        "index and value counts differ"
    );
    if points.is_empty() {
        return Ok(vec![Fr::zero()]);
    }
    crate::workcount::record_poly(points.len() * points.len(), points.len());
    // Build A_I once, then obtain each Lagrange numerator A_I/(X-x_j)
    // by synthetic division. This keeps interpolation quadratic in |I|.
    let mut vanishing = vec![Fr::one()];
    for &point in &points {
        vanishing = multiply_polynomials(&vanishing, &[-point, Fr::one()]);
    }
    let derivative: Vec<Fr> = vanishing
        .iter()
        .enumerate()
        .skip(1)
        .map(|(degree, coefficient)| *coefficient * Fr::from(degree as u64))
        .collect();
    let mut result = vec![Fr::zero(); points.len()];
    for (j, &x_j) in points.iter().enumerate() {
        let mut basis = vec![Fr::zero(); points.len()];
        basis[points.len() - 1] = vanishing[points.len()];
        for degree in (1..points.len()).rev() {
            basis[degree - 1] = vanishing[degree] + x_j * basis[degree];
        }
        let inverse = evaluate_polynomial(&derivative, x_j)
            .inverse()
            .ok_or_else(|| anyhow::anyhow!("interpolation points are not distinct"))?;
        let scale = values[j] * inverse;
        for (degree, coefficient) in basis.iter().enumerate() {
            result[degree] += *coefficient * scale;
        }
    }
    trim(&mut result);
    Ok(result)
}

pub fn evaluate_polynomial(coefficients: &[Fr], x: Fr) -> Fr {
    coefficients
        .iter()
        .rev()
        .fold(Fr::zero(), |acc, coefficient| acc * x + coefficient)
}

pub(crate) fn subtract_polynomials(left: &[Fr], right: &[Fr]) -> Vec<Fr> {
    let mut result = vec![Fr::zero(); left.len().max(right.len()).max(1)];
    for (i, &value) in left.iter().enumerate() {
        result[i] += value;
    }
    for (i, &value) in right.iter().enumerate() {
        result[i] -= value;
    }
    trim(&mut result);
    result
}

pub(crate) fn divide_exact(numerator: &[Fr], denominator: &[Fr]) -> Result<Vec<Fr>> {
    let mut remainder = numerator.to_vec();
    let mut divisor = denominator.to_vec();
    trim(&mut remainder);
    trim(&mut divisor);
    ensure!(
        !divisor.last().unwrap().is_zero(),
        "cannot divide by zero polynomial"
    );
    if remainder.len() < divisor.len() {
        if remainder.iter().all(Zero::is_zero) {
            return Ok(vec![Fr::zero()]);
        }
        bail!("polynomial division has a nonzero remainder");
    }
    crate::workcount::record_poly(remainder.len() * divisor.len(), remainder.len());
    let leading_inverse = divisor.last().unwrap().inverse().unwrap();
    let mut quotient = vec![Fr::zero(); remainder.len() - divisor.len() + 1];
    while remainder.len() >= divisor.len() && !remainder.iter().all(Zero::is_zero) {
        let offset = remainder.len() - divisor.len();
        let factor = *remainder.last().unwrap() * leading_inverse;
        quotient[offset] = factor;
        for (i, &coefficient) in divisor.iter().enumerate() {
            remainder[offset + i] -= factor * coefficient;
        }
        trim(&mut remainder);
    }
    ensure!(
        remainder.iter().all(Zero::is_zero),
        "polynomial division has a nonzero remainder"
    );
    trim(&mut quotient);
    Ok(quotient)
}

/// A fixed denominator and the reciprocal of its reversed polynomial.
/// This is prover-local CPU data, never part of a proof or verifier key.
#[derive(Clone, Debug)]
pub(crate) struct PreparedExactDivision {
    divisor: Vec<Fr>,
    reversed_inverse: Vec<Fr>,
    max_numerator_len: usize,
}

impl PreparedExactDivision {
    pub(crate) fn new(denominator: &[Fr], max_numerator_len: usize) -> Result<Self> {
        let mut divisor = denominator.to_vec();
        trim(&mut divisor);
        ensure!(!divisor.last().unwrap().is_zero(), "cannot divide by zero polynomial");
        ensure!(max_numerator_len >= divisor.len(), "division bound is below divisor length");
        let inverse_len = max_numerator_len - divisor.len() + 1;
        let mut reversed = divisor.clone();
        reversed.reverse();
        let mut inverse = vec![reversed[0].inverse().unwrap()];
        while inverse.len() < inverse_len {
            let doubled = inverse.len().checked_mul(2)
                .ok_or_else(|| anyhow::anyhow!("reciprocal length overflow"))?;
            let next_len = doubled.min(inverse_len);
            // Newton iteration g <- g*(2-f*g) mod X^next_len doubles the
            // number of correct reciprocal coefficients over the field.
            let mut correction = fft_multiply(&reversed[..reversed.len().min(next_len)], &inverse)?;
            correction.resize(next_len, Fr::zero());
            for coefficient in &mut correction { *coefficient = -*coefficient; }
            correction[0] += Fr::from(2u64);
            inverse = fft_multiply(&inverse, &correction)?;
            inverse.resize(next_len, Fr::zero());
        }
        // FFT multiplication allocated a larger buffer; release its spare
        // capacity rather than retaining it for the parameter object's life.
        inverse.shrink_to_fit();
        Ok(Self { divisor, reversed_inverse: inverse, max_numerator_len })
    }

    pub(crate) fn divisor(&self) -> &[Fr] { &self.divisor }

    pub(crate) fn divide_exact(&self, numerator: &[Fr]) -> Result<Vec<Fr>> {
        let mut normalized = numerator.to_vec();
        trim(&mut normalized);
        ensure!(normalized.len() <= self.max_numerator_len, "numerator exceeds prepared division bound");
        if normalized.iter().all(Zero::is_zero) { return Ok(vec![Fr::zero()]); }
        ensure!(normalized.len() >= self.divisor.len(), "polynomial division has a nonzero remainder");
        let quotient_len = normalized.len() - self.divisor.len() + 1;
        let mut reversed_numerator = Vec::with_capacity(quotient_len);
        for i in 0..quotient_len { reversed_numerator.push(normalized[normalized.len() - 1 - i]); }
        // Only the highest quotient_len coefficients are needed to recover
        // the quotient; reversing again restores low-degree-first layout.
        let mut quotient = fft_multiply(&reversed_numerator, &self.reversed_inverse[..quotient_len])?;
        quotient.resize(quotient_len, Fr::zero());
        quotient.reverse();
        trim(&mut quotient);
        // Do not trust a truncated series as evidence of exact divisibility.
        // Multiplication in the field checks every numerator coefficient.
        ensure!(fft_multiply(&quotient, &self.divisor)? == normalized,
            "polynomial division has a nonzero remainder");
        Ok(quotient)
    }
}

/// Exact division by the fixed small-domain polynomial X^n - 1.
pub(crate) fn divide_exact_by_xn_minus_one(numerator: &[Fr], n: usize) -> Result<Vec<Fr>> {
    ensure!(n > 0, "vanishing polynomial degree must be nonzero");
    let mut remainder = numerator.to_vec();
    trim(&mut remainder);
    crate::workcount::record_poly(remainder.len(), remainder.len());
    if remainder.len() <= n {
        ensure!(remainder.iter().all(Zero::is_zero), "polynomial division has a nonzero remainder");
        return Ok(vec![Fr::zero()]);
    }
    let mut quotient = vec![Fr::zero(); remainder.len() - n];
    // Subtract c*X^(degree-n)*(X^n-1): clear the high term and add c
    // to the low term. Descending order also handles degrees above 2n.
    for degree in (n..remainder.len()).rev() {
        let coefficient = remainder[degree];
        quotient[degree - n] = coefficient;
        remainder[degree - n] += coefficient;
        remainder[degree] = Fr::zero();
    }
    ensure!(remainder.iter().all(Zero::is_zero), "polynomial division has a nonzero remainder");
    trim(&mut quotient);
    Ok(quotient)
}

pub(crate) fn fft_multiply(left: &[Fr], right: &[Fr]) -> Result<Vec<Fr>> {
    if left.len().min(right.len()) <= 16 { return Ok(multiply_polynomials(left, right)); }
    let result_len = left.len().checked_add(right.len()).and_then(|len| len.checked_sub(1))
        .ok_or_else(|| anyhow::anyhow!("polynomial product length overflow"))?;
    let size = result_len.checked_next_power_of_two()
        .ok_or_else(|| anyhow::anyhow!("polynomial FFT size overflow"))?;
    let domain = radix2_domain(size)?;
    crate::workcount::record_poly(size * (size.trailing_zeros() as usize + 1), size);
    // Record the actual transforms separately, including their true maximum.
    crate::workcount::record_ntt(size);
    crate::workcount::record_ntt(size);
    crate::workcount::record_ntt(size);
    let mut product = domain.fft(left);
    let right_values = domain.fft(right);
    for i in 0..size { product[i] *= right_values[i]; }
    domain.ifft_in_place(&mut product);
    product.truncate(result_len);
    trim(&mut product);
    Ok(product)
}

pub(crate) fn multiply_polynomials(left: &[Fr], right: &[Fr]) -> Vec<Fr> {
    if left.is_empty() || right.is_empty() {
        return vec![Fr::zero()];
    }
    crate::workcount::record_poly(left.len() * right.len(), left.len().max(right.len()));
    let mut result = vec![Fr::zero(); left.len() + right.len() - 1];
    for (i, &a) in left.iter().enumerate() {
        for (j, &b) in right.iter().enumerate() {
            result[i + j] += a * b;
        }
    }
    trim(&mut result);
    result
}

fn trim(coefficients: &mut Vec<Fr>) {
    while coefficients.len() > 1 && coefficients.last().unwrap().is_zero() {
        coefficients.pop();
    }
    if coefficients.is_empty() {
        coefficients.push(Fr::zero());
    }
}

#[cfg(test)]
mod prepared_division_tests {
    use super::*;
    use ark_std::UniformRand;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn prepared_division_matches_hand_calculated_quotients_and_rejects_remainders() {
        let minus_one = -Fr::one();
        let prepared = PreparedExactDivision::new(&[minus_one, Fr::zero(), Fr::one()], 8).unwrap();
        let numerator = [-Fr::from(2u64), minus_one, Fr::from(2u64), Fr::one()];
        assert_eq!(prepared.divide_exact(&numerator).unwrap(), vec![Fr::from(2u64), Fr::one()]);
        let nonmonic = PreparedExactDivision::new(&[-Fr::from(2u64), Fr::zero(), Fr::from(2u64)], 8).unwrap();
        assert_eq!(nonmonic.divide_exact(&[-Fr::from(4u64), -Fr::from(2u64), Fr::from(4u64), Fr::from(2u64)]).unwrap(),
            vec![Fr::from(2u64), Fr::one()]);
        assert!(prepared.divide_exact(&[minus_one, minus_one, Fr::from(2u64), Fr::one()]).is_err());
        assert!(prepared.divide_exact(&[Fr::one()]).is_err());
        assert_eq!(prepared.divide_exact(&[Fr::zero()]).unwrap(), vec![Fr::zero()]);
        let constant = PreparedExactDivision::new(&[Fr::from(3u64)], 8).unwrap();
        assert_eq!(constant.divide_exact(&[Fr::from(6u64), Fr::from(9u64)]).unwrap(), vec![Fr::from(2u64), Fr::from(3u64)]);
        assert!(PreparedExactDivision::new(&[], 8).is_err());
        assert!(PreparedExactDivision::new(&[Fr::zero()], 8).is_err());
        assert!(PreparedExactDivision::new(&[minus_one, Fr::zero(), Fr::one()], 2).is_err());
        assert!(prepared.divide_exact(&[Fr::one(); 9]).is_err());
    }

    #[test]
    fn prepared_division_matches_original_long_division_across_fft_sizes() {
        let mut rng = StdRng::seed_from_u64(2901);
        for degree in [1usize, 3, 17, 65, 129] {
            let mut divisor = Vec::new();
            for _ in 0..degree { divisor.push(Fr::rand(&mut rng)); }
            divisor.push(Fr::from(7u64));
            let prepared = PreparedExactDivision::new(&divisor, 4 * degree + 8).unwrap();
            for quotient_size in [1usize, 3, degree + 1, 2 * degree + 3] {
                let mut quotient = Vec::new();
                for _ in 0..quotient_size { quotient.push(Fr::rand(&mut rng)); }
                let numerator = multiply_polynomials(&divisor, &quotient);
                assert_eq!(prepared.divide_exact(&numerator).unwrap(), divide_exact(&numerator, &divisor).unwrap());
                let mut corrupted = numerator;
                corrupted[0] += Fr::one();
                assert!(prepared.divide_exact(&corrupted).is_err());
            }
        }
    }
}

#[cfg(test)]
mod multiplication_domain_tests {
    use super::*;
    use ark_std::UniformRand;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn fixed_vanishing_division_matches_literal_quotients_and_boundaries() {
        let f = Fr::from;
        assert_eq!(divide_exact_by_xn_minus_one(&[-f(2u64), -f(1u64), f(2u64), f(1u64)], 2).unwrap(),
            vec![f(2u64), f(1u64)]);
        // Degree above 2n requires repeated folding, with a positive low term.
        assert_eq!(divide_exact_by_xn_minus_one(&[-f(2u64), -f(3u64), -f(3u64), -f(4u64), -f(6u64), f(7u64), f(11u64)], 2).unwrap(),
            vec![f(2u64), f(3u64), f(5u64), f(7u64), f(11u64)]);
        assert_eq!(divide_exact_by_xn_minus_one(&[-f(2u64), -f(1u64), -f(2u64), f(5u64)], 1).unwrap(),
            vec![f(2u64), f(3u64), f(5u64)]);
        for n in [1usize, 2, 64, usize::MAX] {
            assert_eq!(divide_exact_by_xn_minus_one(&[], n).unwrap(), vec![Fr::zero()]);
            assert_eq!(divide_exact_by_xn_minus_one(&[Fr::zero(); 4], n).unwrap(), vec![Fr::zero()]);
            assert!(divide_exact_by_xn_minus_one(&[Fr::one()], n).is_err());
        }
        assert!(divide_exact_by_xn_minus_one(&[], 0).is_err());
        assert!(divide_exact_by_xn_minus_one(&[Fr::zero()], 0).is_err());
        assert!(divide_exact_by_xn_minus_one(&[Fr::one(); 3], 0).is_err());
        assert!(divide_exact_by_xn_minus_one(&[-f(2u64), Fr::zero(), f(2u64), f(1u64)], 2).is_err());
    }

    #[test]
    fn fixed_vanishing_division_matches_dense_oracle_and_checks_every_remainder() {
        let mut rng = StdRng::seed_from_u64(3101);
        for n in [1usize, 2, 3, 8, 17, 64] {
            let mut divisor = vec![Fr::zero(); n + 1];
            divisor[0] = -Fr::one();
            divisor[n] = Fr::one();
            for len in [1usize, n, 2 * n + 3] {
                let mut quotient = Vec::new();
                for _ in 0..len { quotient.push(Fr::rand(&mut rng)); }
                // Preserve the original quadratic routines as the independent oracle.
                let numerator = multiply_polynomials(&divisor, &quotient);
                let expected = divide_exact(&numerator, &divisor).unwrap();
                assert_eq!(divide_exact_by_xn_minus_one(&numerator, n).unwrap(), expected);
                let mut padded = numerator.clone();
                padded.extend([Fr::zero(); 3]);
                assert_eq!(divide_exact_by_xn_minus_one(&padded, n).unwrap(), expected);
                for index in 0..n {
                    let mut corrupted = numerator.clone();
                    corrupted[index] += Fr::one();
                    assert!(divide_exact(&corrupted, &divisor).is_err());
                    assert!(divide_exact_by_xn_minus_one(&corrupted, n).is_err(),
                        "missed nonzero remainder at degree {index} for n={n}");
                }
            }
        }
    }

    #[test]
    fn fft_product_matches_dense_oracle_across_fallback_and_padding_boundaries() {
        let mut rng = StdRng::seed_from_u64(3102);
        for (left_len, right_len) in [(0usize, 32usize), (1, 65), (16, 33), (17, 17),
            (17, 33), (32, 33), (33, 64), (64, 65), (65, 129), (129, 130)] {
            let mut left = Vec::new();
            let mut right = Vec::new();
            for _ in 0..left_len { left.push(Fr::rand(&mut rng)); }
            for _ in 0..right_len { right.push(Fr::rand(&mut rng)); }
            if let Some(last) = left.last_mut() { *last = Fr::one(); }
            if let Some(last) = right.last_mut() { *last = Fr::one(); }
            assert_eq!(fft_multiply(&left, &right).unwrap(), multiply_polynomials(&left, &right));
            left.extend([Fr::zero(); 2]);
            right.extend([Fr::zero(); 3]);
            assert_eq!(fft_multiply(&left, &right).unwrap(), multiply_polynomials(&left, &right));
        }
        assert_eq!(fft_multiply(&[Fr::zero(); 33], &[Fr::one(); 65]).unwrap(), vec![Fr::zero()]);
        assert_eq!(fft_multiply(&[], &[]).unwrap(), vec![Fr::zero()]);
    }
}
