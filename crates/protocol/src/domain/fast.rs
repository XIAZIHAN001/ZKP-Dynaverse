//! Balanced polynomial tools whose allocation/work depends on the selected set.
use super::{fft_multiply, subtract_polynomials, trim, PreparedExactDivision};
use anyhow::{ensure, Result};
use ark_bn254::Fr;
use ark_ff::{batch_inversion, Field, One, Zero};
use std::collections::HashSet;

/// A balanced subproduct tree over distinct points, retaining input order.
/// Level zero holds X-x_i; an unpaired node is carried to the next level.
#[derive(Clone, Debug)]
pub struct SubproductTree {
    points: Vec<Fr>,
    levels: Vec<Vec<Vec<Fr>>>,
    inverse_derivatives: Vec<Fr>,
}

impl SubproductTree {
    pub fn new_with_backend(points: &[Fr], backend: &dyn crate::OperatorBackend) -> Result<Self> {
        Self::new_impl(points, Some(backend))
    }

    pub fn multipoint_eval_with_backend(&self, polynomial: &[Fr], backend: &dyn crate::OperatorBackend) -> Result<Vec<Fr>> {
        self.eval_impl(polynomial, Some(backend))
    }

    pub fn interpolate_with_backend(&self, values: &[Fr], backend: &dyn crate::OperatorBackend) -> Result<Vec<Fr>> {
        self.interpolate_impl(values, Some(backend))
    }

    pub fn cauchy_sums_with_backend(&self, weights: &[Fr], backend: &dyn crate::OperatorBackend) -> Result<Vec<Fr>> {
        self.cauchy_impl(weights, Some(backend))
    }

    pub fn new(points: &[Fr]) -> Result<Self> {
        Self::new_impl(points, None)
    }

    fn new_impl(points: &[Fr], backend: Option<&dyn crate::OperatorBackend>) -> Result<Self> {
        let mut seen = HashSet::with_capacity(points.len());
        for &point in points { ensure!(seen.insert(point), "duplicate interpolation point"); }
        crate::workcount::record_scan(points.len());
        let mut leaves = Vec::with_capacity(points.len().max(1));
        for &point in points { leaves.push(vec![-point, Fr::one()]); }
        if leaves.is_empty() { leaves.push(vec![Fr::one()]); }
        let mut levels = vec![leaves];
        while levels.last().unwrap().len() > 1 {
            let previous = levels.last().unwrap();
            let mut next = Vec::with_capacity(previous.len().div_ceil(2));
            for pair in previous.chunks(2) {
                next.push(if pair.len() == 2 { product(&pair[0], &pair[1], backend)? }
                    else { pair[0].clone() });
            }
            levels.push(next);
        }
        let mut tree = Self { points: points.to_vec(), levels, inverse_derivatives: Vec::new() };
        let derivative = differentiate(tree.vanishing());
        let mut derivatives = tree.eval_impl(&derivative, backend)?;
        ensure!(derivatives.iter().all(|value| !value.is_zero()), "repeated interpolation root");
        batch_inversion(&mut derivatives);
        tree.inverse_derivatives = derivatives;
        Ok(tree)
    }

    pub fn vanishing(&self) -> &[Fr] { &self.levels.last().unwrap()[0] }

    /// Weights 1/A'(x_i), also shared by all three KZG opening combinations.
    pub fn inverse_derivatives(&self) -> &[Fr] { &self.inverse_derivatives }

    pub fn multipoint_eval(&self, polynomial: &[Fr]) -> Result<Vec<Fr>> {
        self.eval_impl(polynomial, None)
    }

    fn eval_impl(&self, polynomial: &[Fr], backend: Option<&dyn crate::OperatorBackend>) -> Result<Vec<Fr>> {
        if self.points.is_empty() { return Ok(Vec::new()); }
        let mut remainders = vec![div_rem_impl(polynomial, self.vanishing(), backend)?.1];
        for level in (0..self.levels.len() - 1).rev() {
            let mut next = Vec::with_capacity(self.levels[level].len());
            for (i, divisor) in self.levels[level].iter().enumerate() {
                next.push(div_rem_impl(&remainders[i / 2], divisor, backend)?.1);
            }
            remainders = next;
        }
        Ok(remainders.iter().map(|remainder| remainder[0]).collect())
    }

    pub fn interpolate(&self, values: &[Fr]) -> Result<Vec<Fr>> {
        self.interpolate_impl(values, None)
    }

    fn interpolate_impl(&self, values: &[Fr], backend: Option<&dyn crate::OperatorBackend>) -> Result<Vec<Fr>> {
        ensure!(values.len() == self.points.len(), "interpolation value count differs from points");
        if values.is_empty() { return Ok(vec![Fr::zero()]); }
        crate::workcount::record_poly(values.len(), values.len());
        let mut polynomials = Vec::with_capacity(values.len());
        for (i, &value) in values.iter().enumerate() {
            polynomials.push(vec![value * self.inverse_derivatives[i]]);
        }
        for level in 0..self.levels.len() - 1 {
            let mut next = Vec::with_capacity(polynomials.len().div_ceil(2));
            for i in (0..polynomials.len()).step_by(2) {
                if i + 1 == polynomials.len() { next.push(polynomials[i].clone()); continue; }
                let left = product(&polynomials[i], &self.levels[level][i + 1], backend)?;
                let right = product(&polynomials[i + 1], &self.levels[level][i], backend)?;
                let mut combined = vec![Fr::zero(); left.len().max(right.len())];
                for j in 0..left.len() { combined[j] += left[j]; }
                for j in 0..right.len() { combined[j] += right[j]; }
                trim(&mut combined);
                next.push(combined);
            }
            polynomials = next;
        }
        Ok(polynomials.pop().unwrap())
    }

    /// sum_{j != i} weights[j]/(x_i-x_j), excluding the singular diagonal.
    /// N = sum w_j A/(X-x_j), so the regular term of N/A at x_i is
    /// N'(x_i)/A'(x_i) - w_i A''(x_i)/(2 A'(x_i)).
    pub fn cauchy_sums(&self, weights: &[Fr]) -> Result<Vec<Fr>> {
        self.cauchy_impl(weights, None)
    }

    fn cauchy_impl(&self, weights: &[Fr], backend: Option<&dyn crate::OperatorBackend>) -> Result<Vec<Fr>> {
        ensure!(weights.len() == self.points.len(), "Cauchy weight count differs from points");
        let mut numerator_values = Vec::with_capacity(weights.len());
        for (i, &weight) in weights.iter().enumerate() {
            numerator_values.push(weight / self.inverse_derivatives[i]);
        }
        let numerator = self.interpolate_impl(&numerator_values, backend)?;
        let first = self.eval_impl(&differentiate(&numerator), backend)?;
        let second = self.eval_impl(&differentiate(&differentiate(self.vanishing())), backend)?;
        let half = Fr::from(2u64).inverse().unwrap();
        let mut sums = Vec::with_capacity(weights.len());
        crate::workcount::record_poly(4 * weights.len(), weights.len());
        for i in 0..weights.len() {
            sums.push((first[i] - weights[i] * second[i] * half) * self.inverse_derivatives[i]);
        }
        Ok(sums)
    }
}

fn differentiate(polynomial: &[Fr]) -> Vec<Fr> {
    crate::workcount::record_poly(polynomial.len(), polynomial.len());
    let mut derivative = Vec::with_capacity(polynomial.len().saturating_sub(1));
    for (i, &coefficient) in polynomial.iter().enumerate().skip(1) {
        derivative.push(coefficient * Fr::from(i as u64));
    }
    trim(&mut derivative);
    derivative
}

/// Newton reciprocal division with a checked remainder. Uses O(M(n)) field
/// work; the quadratic reference division remains available unchanged.
pub fn newton_divide_exact(numerator: &[Fr], denominator: &[Fr]) -> Result<Vec<Fr>> {
    let (quotient, remainder) = div_rem(numerator, denominator)?;
    ensure!(remainder.iter().all(Zero::is_zero), "polynomial division has a nonzero remainder");
    Ok(quotient)
}

fn div_rem(numerator: &[Fr], denominator: &[Fr]) -> Result<(Vec<Fr>, Vec<Fr>)> {
    div_rem_impl(numerator, denominator, None)
}

fn div_rem_impl(numerator: &[Fr], denominator: &[Fr], backend: Option<&dyn crate::OperatorBackend>) -> Result<(Vec<Fr>, Vec<Fr>)> {
    crate::workcount::record_scan(numerator.len() + denominator.len());
    let mut num = numerator.to_vec();
    let mut den = denominator.to_vec();
    trim(&mut num);
    trim(&mut den);
    ensure!(!den.last().unwrap().is_zero(), "cannot divide by zero polynomial");
    if num.len() < den.len() || num.iter().all(Zero::is_zero) {
        return Ok((vec![Fr::zero()], num));
    }
    let len = num.len() - den.len() + 1;
    // Keep the reference path unchanged. The selected path must also submit
    // every reciprocal-Newton product, not only the final quotient product.
    let reversed_inverse = if backend.is_none() {
        PreparedExactDivision::new(&den, num.len())?.reversed_inverse
    } else {
        let reversed: Vec<Fr> = den.iter().rev().copied().collect();
        let mut inverse = vec![reversed[0].inverse().unwrap()];
        while inverse.len() < len {
            let next = inverse.len().checked_mul(2)
                .ok_or_else(|| anyhow::anyhow!("reciprocal length overflow"))?.min(len);
            let mut correction = product(&reversed[..reversed.len().min(next)], &inverse, backend)?;
            correction.resize(next, Fr::zero());
            for coefficient in &mut correction { *coefficient = -*coefficient; }
            correction[0] += Fr::from(2u64);
            inverse = product(&inverse, &correction, backend)?;
            inverse.resize(next, Fr::zero());
        }
        inverse
    };
    let reversed: Vec<Fr> = num.iter().rev().take(len).copied().collect();
    let mut quotient = product(&reversed, &reversed_inverse[..len], backend)?;
    quotient.resize(len, Fr::zero());
    quotient.reverse();
    trim(&mut quotient);
    let remainder = subtract_polynomials(&num, &product(&quotient, &den, backend)?);
    ensure!(remainder.len() < den.len() || remainder.iter().all(Zero::is_zero),
        "Newton division failed to cancel high coefficients");
    Ok((quotient, remainder))
}

fn product(left: &[Fr], right: &[Fr], backend: Option<&dyn crate::OperatorBackend>) -> Result<Vec<Fr>> {
    match backend {
        Some(backend) => fft_multiply_with_backend(left, right, backend),
        None => fft_multiply(left, right),
    }
}

/// Explicit dispatch includes tiny nonempty products: policy and thresholds
/// belong to the caller's backend adapter. Errors never trigger CPU fallback.
pub fn fft_multiply_with_backend(left: &[Fr], right: &[Fr], backend: &dyn crate::OperatorBackend) -> Result<Vec<Fr>> {
    if left.is_empty() || right.is_empty() { return Ok(vec![Fr::zero()]); }
    let result_len = left.len().checked_add(right.len()).and_then(|v| v.checked_sub(1))
        .ok_or_else(|| anyhow::anyhow!("polynomial product length overflow"))?;
    let size = result_len.checked_next_power_of_two()
        .ok_or_else(|| anyhow::anyhow!("polynomial FFT size overflow"))?;
    crate::workcount::record_poly(size * (size.trailing_zeros() as usize + 1), size);
    let mut result = left.to_vec();
    let mut right_values = right.to_vec();
    result.resize(size, Fr::zero());
    right_values.resize(size, Fr::zero());
    crate::workcount::record_ntt(size);
    backend.ntt(&mut result, false)?;
    crate::workcount::record_ntt(size);
    backend.ntt(&mut right_values, false)?;
    for i in 0..size { result[i] *= right_values[i]; }
    crate::workcount::record_ntt(size);
    backend.ntt(&mut result, true)?;
    result.truncate(result_len);
    trim(&mut result);
    Ok(result)
}

pub fn newton_divide_exact_with_backend(numerator: &[Fr], denominator: &[Fr], backend: &dyn crate::OperatorBackend) -> Result<Vec<Fr>> {
    let (quotient, remainder) = div_rem_impl(numerator, denominator, Some(backend))?;
    ensure!(remainder.iter().all(Zero::is_zero), "polynomial division has a nonzero remainder");
    Ok(quotient)
}

#[cfg(test)]
#[path = "fast_backend_tests.rs"]
mod backend_tests;
