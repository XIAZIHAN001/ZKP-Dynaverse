//! Setup arithmetic shared by SRS constructors. Trapdoor-derived scalars live
//! only in these calls; neither the backend nor the returned SRS retains them.
use anyhow::{ensure, Result};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{CurveGroup, Group};
use ark_ff::{batch_inversion, Field, One, Zero};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use crate::OperatorBackend;

#[cfg(test)]
thread_local! { pub(crate) static LAGRANGE_LENGTHS: std::cell::RefCell<Vec<usize>> = const {std::cell::RefCell::new(Vec::new())}; }

pub(crate) fn cpu_g1(scalars: &[Fr]) -> Result<Vec<G1Affine>> {
    let mut result = Vec::with_capacity(scalars.len());
    // Bound projective temporaries independently of the complete SRS size.
    for chunk in scalars.chunks(4096) {
        let mut points = Vec::with_capacity(chunk.len());
        for &scalar in chunk { points.push(G1Projective::generator()*scalar); }
        result.extend(G1Projective::normalize_batch(&points));
    }
    Ok(result)
}
pub(crate) fn cpu_g2(scalars: &[Fr]) -> Result<Vec<G2Affine>> {
    let mut result = Vec::with_capacity(scalars.len());
    for chunk in scalars.chunks(4096) {
        let mut points = Vec::with_capacity(chunk.len());
        for &scalar in chunk { points.push(G2Projective::generator()*scalar); }
        result.extend(G2Projective::normalize_batch(&points));
    }
    Ok(result)
}
pub(crate) fn g1(scalars: &[Fr], backend: Option<&dyn OperatorBackend>) -> Result<Vec<G1Affine>> {
    let points = match backend { Some(b) => b.fixed_base_g1(scalars)?, None => cpu_g1(scalars)? };
    ensure!(points.len()==scalars.len(), "setup backend returned wrong G1 count");
    Ok(points)
}
pub(crate) fn g2(scalars: &[Fr], backend: Option<&dyn OperatorBackend>) -> Result<Vec<G2Affine>> {
    let points = match backend { Some(b) => b.fixed_base_g2(scalars)?, None => cpu_g2(scalars)? };
    ensure!(points.len()==scalars.len(), "setup backend returned wrong G2 count");
    Ok(points)
}

/// L_i(tau)=(tau^s-1)*omega^i/(s*(tau-omega^i)). One inversion
/// per vector replaces s inversions; successive roots take s multiplies.
pub(crate) fn lagrange_values(tau: Fr, domain: &Radix2EvaluationDomain<Fr>) -> Result<Vec<Fr>> {
    let size = domain.size();
    #[cfg(test)]
    LAGRANGE_LENGTHS.with(|lengths|lengths.borrow_mut().push(size));
    let numerator = tau.pow([size as u64])-Fr::one();
    ensure!(!numerator.is_zero(), "Lagrange evaluation point lies on domain");
    let scale = numerator * domain.size_inv;
    let mut roots = Vec::with_capacity(size);
    let mut denominators = Vec::with_capacity(size);
    let mut root = Fr::one();
    for _ in 0..size {
        roots.push(root); denominators.push(tau-root); root *= domain.group_gen;
    }
    batch_inversion(&mut denominators);
    for i in 0..size { denominators[i] *= scale*roots[i]; }
    Ok(denominators)
}
