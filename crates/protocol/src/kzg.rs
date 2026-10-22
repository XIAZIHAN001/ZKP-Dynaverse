//! Minimal circuit-specific KZG commitments over BN254.
//!
//! Setup is for the research reference prover: its trapdoor is sampled in
//! memory and discarded. No production trusted-setup ceremony is claimed.

use crate::domain::{
    divide_exact, evaluate_polynomial, index_points, interpolate_at_indices, subtract_polynomials,
    vanishing_polynomial, vanishing_derivative_values, radix2_domain,
};
use crate::types::{BatchMsmRequest, OperatorBackend, SparseMsmRequest};
use anyhow::{ensure, Result};
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{pairing::Pairing, CurveGroup, VariableBaseMSM};
use ark_ff::{batch_inversion, One, Zero};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_std::{rand::Rng, UniformRand};
use std::sync::{Arc, Mutex};

#[cfg(test)]
#[path = "kzg_sparse_tests.rs"]
mod sparse_tests;

#[derive(Debug)]
struct PreparedOpeningBasis {
    degree: usize,
    fft: Arc<Vec<G1Projective>>,
}

#[derive(Clone, Debug)]
pub struct KzgParams {
    max_degree: usize,
    g1_powers: Arc<Vec<G1Affine>>,
    g2_powers: Arc<Vec<G2Affine>>,
    // Clones of this immutable SRS share at most one prepared degree.
    opening_basis: Arc<Mutex<Option<PreparedOpeningBasis>>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KzgCommitment {
    pub point: G1Affine,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KzgOpening {
    pub witness: G1Affine,
    /// Evaluation at each requested index, in the caller's index order.
    pub values: Vec<Fr>,
}

/// Submit selected rows without copying the base table into gathered vectors.
/// Count before dispatch so CPU and GPU calls have the same protocol cost.
pub(crate) fn gather_msm_batch(
    requests: &[(&[G1Affine], &[(usize, Fr)])], backend: &dyn OperatorBackend,
) -> Result<Vec<G1Affine>> {
    let mut sparse = Vec::new();
    sparse.try_reserve_exact(requests.len())?;
    for &(bases, entries) in requests {
        let request = SparseMsmRequest { bases, entries };
        request.validate()?;
        crate::workcount::record_scan(entries.len());
        crate::workcount::record_msm(entries.len());
        sparse.push(request);
    }
    let result = backend.msm_sparse_batch(&sparse)?;
    ensure!(result.len() == requests.len(), "backend returned wrong sparse commitment count");
    Ok(result)
}

/// Combine q_i = [(f(X)-f(x_i))/(X-x_i)] into [(f-r_I)/A_I].
/// The caller must supply openings of the same polynomial and SRS at every
/// domain point. Public verification remains responsible for binding the result.
pub fn combine_single_openings(
    domain: &Radix2EvaluationDomain<Fr>,
    indices: &[usize],
    single_openings: &[G1Affine],
) -> Result<G1Affine> {
    ensure!(!indices.is_empty(), "cached opening index set must not be empty");
    ensure!(single_openings.len() == domain.size(), "opening cache size differs from domain");
    let mut weights = vanishing_derivative_values(domain, indices)?;
    // Distinct domain points make every A'_I(x_i) nonzero.
    batch_inversion(&mut weights);
    let points: Vec<G1Affine> = indices.iter().map(|&i| single_openings[i]).collect();
    crate::workcount::record_msm(points.len());
    Ok(G1Projective::msm_unchecked(&points, &weights).into_affine())
}

impl KzgParams {
    pub(crate) fn rebuild_g1_powers(&self) -> &[G1Affine] {
        &self.g1_powers
    }

    pub(crate) fn rebuild_g2_powers(&self) -> &[G2Affine] {
        &self.g2_powers
    }
    pub fn setup<R: Rng + ?Sized>(max_degree: usize, rng: &mut R) -> Result<Self> {
        let mut tau = Fr::rand(rng);
        while tau.is_zero() {
            tau = Fr::rand(rng);
        }
        Self::from_tau(max_degree, tau)
    }

    pub(crate) fn from_tau(max_degree: usize, tau: Fr) -> Result<Self> {
        Self::from_tau_with_backend(max_degree, tau, None)
    }

    pub(crate) fn from_tau_with_backend(max_degree: usize, tau: Fr,
        backend: Option<&dyn OperatorBackend>) -> Result<Self> {
        ensure!(!tau.is_zero(), "KZG trapdoor must be nonzero");
        let capacity = max_degree
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("SRS degree is too large"))?;
        let mut scalars = Vec::with_capacity(capacity);
        let mut tau_power = Fr::one();
        for _ in 0..capacity {
            scalars.push(tau_power);
            tau_power *= tau;
        }
        let g1_powers = crate::setup::g1(&scalars, backend)?;
        let g2_powers = crate::setup::g2(&scalars, backend)?;
        Ok(Self {
            max_degree,
            g1_powers: Arc::new(g1_powers),
            g2_powers: Arc::new(g2_powers),
            opening_basis: Arc::new(Mutex::new(None)),
        })
    }

    pub fn max_degree(&self) -> usize {
        self.max_degree
    }

    pub fn commit(&self, coefficients: &[Fr]) -> Result<KzgCommitment> {
        ensure!(
            coefficients.len() <= self.g1_powers.len(),
            "polynomial exceeds KZG SRS degree bound"
        );
        if coefficients.is_empty() {
            return Ok(KzgCommitment {
                point: G1Projective::zero().into_affine(),
            });
        }
        crate::workcount::record_msm(coefficients.len());
        let point =
            G1Projective::msm_unchecked(&self.g1_powers[..coefficients.len()], coefficients)
                .into_affine();
        Ok(KzgCommitment { point })
    }

    /// Dispatch only the G1 MSM. The backend is borrowed for this synchronous
    /// call; no device handle or backend reference is stored in the SRS.
    pub fn commit_with_backend(
        &self, coefficients: &[Fr], backend: &dyn OperatorBackend,
    ) -> Result<KzgCommitment> {
        ensure!(coefficients.len() <= self.g1_powers.len(),
            "polynomial exceeds KZG SRS degree bound");
        if coefficients.is_empty() {
            return Ok(KzgCommitment { point: G1Affine::identity() });
        }
        // Counted before dispatch so a GPU backend reports the same work.
        crate::workcount::record_msm(coefficients.len());
        // Trait dispatch keeps the protocol independent of CUDA types/crates.
        let point = backend.msm(&self.g1_powers[..coefficients.len()], coefficients)?;
        Ok(KzgCommitment { point })
    }

    /// Independent polynomial commitments in caller order. Every degree is
    /// checked before dispatch; shared SRS slices stay borrowed for the call.
    pub fn commit_batch_with_backend(
        &self, coefficients: &[&[Fr]], backend: &dyn OperatorBackend,
    ) -> Result<Vec<KzgCommitment>> {
        let mut total = 0usize;
        for polynomial in coefficients {
            ensure!(polynomial.len() <= self.g1_powers.len(), "polynomial exceeds KZG SRS degree bound");
            total = total.checked_add(polynomial.len())
                .ok_or_else(|| anyhow::anyhow!("batch scalar size overflow"))?;
        }
        let mut scalars = Vec::new();
        scalars.try_reserve_exact(total)?;
        let mut sets = Vec::new();
        let mut offsets = Vec::new();
        let mut lengths = Vec::new();
        sets.try_reserve_exact(coefficients.len())?;
        offsets.try_reserve_exact(coefficients.len())?;
        lengths.try_reserve_exact(coefficients.len())?;
        for polynomial in coefficients {
            crate::workcount::record_msm(polynomial.len());
            offsets.push(scalars.len());
            lengths.push(polynomial.len());
            sets.push(&self.g1_powers[..polynomial.len()]);
            scalars.extend_from_slice(polynomial);
        }
        let points = backend.msm_batch(&BatchMsmRequest { point_sets: &sets,
            shared_scalars: &scalars, offsets: &offsets, lengths: &lengths })?;
        ensure!(points.len() == coefficients.len(), "batch backend returned wrong commitment count");
        let mut commitments = Vec::new();
        commitments.try_reserve_exact(points.len())?;
        for point in points { commitments.push(KzgCommitment { point }); }
        Ok(commitments)
    }

    pub fn open_single(
        &self,
        coefficients: &[Fr],
        domain: &Radix2EvaluationDomain<Fr>,
        index: usize,
    ) -> Result<KzgOpening> {
        self.open_batch(coefficients, domain, &[index])
    }

    /// Compute every domain-point opening for a polynomial of degree < n.
    /// A scalar/group convolution forms H_j = sum_{l>j} f_l [tau^(l-1-j)],
    /// then a group FFT evaluates sum_j H_j x^j at all n domain points.
    /// This avoids n independent dense quotient commitments during reanchoring.
    /// The SRS forward FFT is prepared lazily and shared by parameter clones.
    /// Only one coefficient degree is retained; polynomial-dependent work is local.
    pub fn precompute_single_openings(
        &self,
        coefficients: &[Fr],
        domain: &Radix2EvaluationDomain<Fr>,
    ) -> Result<Vec<G1Affine>> {
        ensure!(coefficients.len() <= domain.size(), "cached polynomial degree must be below domain size");
        ensure!(coefficients.len() <= self.g1_powers.len(), "polynomial exceeds KZG SRS degree bound");
        let degree = coefficients.len().saturating_sub(1);
        if degree == 0 {
            return Ok(vec![G1Affine::identity(); domain.size()]);
        }
        let convolution_size = degree.checked_mul(2)
            .and_then(|size| size.checked_sub(1))
            .and_then(usize::checked_next_power_of_two)
            .ok_or_else(|| anyhow::anyhow!("opening convolution size overflow"))?;
        let convolution_domain = radix2_domain(convolution_size)?;
        let reversed: Vec<Fr> = coefficients[1..].iter().rev().copied().collect();
        // One scalar FFT, one group IFFT and one group FFT over the domain.
        crate::workcount::record_ntt(convolution_size);
        crate::workcount::record_ntt(convolution_size);
        crate::workcount::record_ntt(domain.size());
        let scalar_fft = convolution_domain.fft(&reversed);
        let prepared = self.prepared_opening_basis(degree, &convolution_domain);
        // Copy the immutable SRS transform before polynomial-specific mutation.
        let mut convolution = prepared.as_ref().clone();
        for (point, scalar) in convolution.iter_mut().zip(&scalar_fft) {
            *point *= scalar;
        }
        convolution_domain.ifft_in_place(&mut convolution);
        convolution.truncate(degree);
        convolution.reverse();
        let openings = domain.fft(&convolution);
        Ok(G1Projective::normalize_batch(&openings))
    }

    fn prepared_opening_basis(
        &self, degree: usize, convolution_domain: &Radix2EvaluationDomain<Fr>,
    ) -> Arc<Vec<G1Projective>> {
        let mut cache = match self.opening_basis.lock() {
            Ok(cache) => cache,
            // A panic during preparation cannot publish a partial transform:
            // the entry is assigned only after FFT completes, and points are immutable.
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some(prepared) = cache.as_ref() {
            if prepared.degree == degree {
                return Arc::clone(&prepared.fft);
            }
        }
        let mut powers = Vec::with_capacity(degree);
        for point in &self.g1_powers[..degree] {
            powers.push(G1Projective::from(*point));
        }
        crate::workcount::record_ntt(convolution_domain.size());
        let fft = Arc::new(convolution_domain.fft(&powers));
        // Holding the lock through construction avoids duplicate concurrent work.
        *cache = Some(PreparedOpeningBasis { degree, fft: Arc::clone(&fft) });
        fft
    }

    pub fn open_batch(
        &self,
        coefficients: &[Fr],
        domain: &Radix2EvaluationDomain<Fr>,
        indices: &[usize],
    ) -> Result<KzgOpening> {
        self.check_opening_indices(indices)?;
        ensure!(
            coefficients.len() <= self.g1_powers.len(),
            "polynomial exceeds KZG SRS degree bound"
        );
        let points = index_points(domain, indices)?;
        let values: Vec<Fr> = points
            .into_iter()
            .map(|point| evaluate_polynomial(coefficients, point))
            .collect();
        let interpolation = interpolate_at_indices(domain, indices, &values)?;
        let vanishing = vanishing_polynomial(domain, indices)?;
        let numerator = subtract_polynomials(coefficients, &interpolation);
        let quotient = divide_exact(&numerator, &vanishing)?;
        let witness = self.commit(&quotient)?.point;
        Ok(KzgOpening { witness, values })
    }

    pub fn verify_single(
        &self,
        commitment: &KzgCommitment,
        domain: &Radix2EvaluationDomain<Fr>,
        index: usize,
        opening: &KzgOpening,
    ) -> Result<bool> {
        self.verify_batch(commitment, domain, &[index], opening)
    }

    pub fn verify_batch(
        &self,
        commitment: &KzgCommitment,
        domain: &Radix2EvaluationDomain<Fr>,
        indices: &[usize],
        opening: &KzgOpening,
    ) -> Result<bool> {
        self.check_opening_indices(indices)?;
        ensure!(
            opening.values.len() == indices.len(),
            "opening value count differs from index count"
        );
        let interpolation = interpolate_at_indices(domain, indices, &opening.values)?;
        let vanishing = vanishing_polynomial(domain, indices)?;
        let interpolation_commitment = self.commit(&interpolation)?.point;
        let vanishing_commitment = self.commit_g2(&vanishing)?;
        let difference = (G1Projective::from(commitment.point)
            - G1Projective::from(interpolation_commitment))
        .into_affine();
        Ok(Bn254::pairing(difference, self.g2_powers[0])
            == Bn254::pairing(opening.witness, vanishing_commitment))
    }

    fn check_opening_indices(&self, indices: &[usize]) -> Result<()> {
        ensure!(
            !indices.is_empty(),
            "KZG opening index set must not be empty"
        );
        ensure!(
            indices.len() <= self.max_degree,
            "opening set exceeds G2 SRS degree bound"
        );
        Ok(())
    }

    pub(crate) fn commit_g2(&self, coefficients: &[Fr]) -> Result<G2Affine> {
        ensure!(
            coefficients.len() <= self.g2_powers.len(),
            "polynomial exceeds G2 SRS degree bound"
        );
        crate::workcount::record_g2_msm(coefficients.len());
        Ok(
            G2Projective::msm_unchecked(&self.g2_powers[..coefficients.len()], coefficients)
                .into_affine(),
        )
    }

    pub(crate) fn commit_g2_with_backend(&self,coefficients:&[Fr],backend:&dyn OperatorBackend)
        ->Result<G2Affine> {
        ensure!(coefficients.len()<=self.g2_powers.len(),"polynomial exceeds G2 SRS degree bound");
        crate::workcount::record_g2_msm(coefficients.len());
        backend.msm_g2_host(&self.g2_powers[..coefficients.len()],coefficients)
    }
}

#[cfg(test)]
mod opening_basis_tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    #[test]
    fn opening_work_counts_cold_srs_fft_and_separate_transform_lengths() {
        let params = KzgParams::from_tau(16, Fr::from(7u64)).unwrap();
        let domain = radix2_domain(8).unwrap();
        let polynomial = coefficients(8, 7);
        let (first, cold) = crate::workcount::measure(|| params.precompute_single_openings(&polynomial, &domain));
        let (second, warm) = crate::workcount::measure(|| params.precompute_single_openings(&polynomial, &domain));
        assert_eq!(first.unwrap(), second.unwrap());
        assert_eq!(cold.ntt_elements, 16 + 16 + 16 + 8);
        assert_eq!(warm.ntt_elements, 16 + 16 + 8);
        assert_eq!(cold.max_operand, 16);
        assert_eq!(warm.max_operand, 16);
    }

    fn coefficients(length: usize, offset: u64) -> Vec<Fr> {
        let mut result = Vec::new();
        for i in 0..length {
            result.push(Fr::from((i * i + 3 * i) as u64 + offset));
        }
        result
    }

    // Test utilities inspect retained ownership without adding a production API.
    fn retained_basis(params: &KzgParams) -> Option<(usize, Arc<Vec<G1Projective>>)> {
        let cache = match params.opening_basis.lock() {
            Ok(cache) => cache,
            Err(poisoned) => poisoned.into_inner(),
        };
        cache.as_ref().map(|basis| (basis.degree, Arc::clone(&basis.fft)))
    }

    fn assert_dense_oracle(
        params: &KzgParams, polynomial: &[Fr], domain: &Radix2EvaluationDomain<Fr>,
        openings: &[G1Affine],
    ) {
        assert_eq!(openings.len(), domain.size());
        for i in 0..domain.size() {
            assert_eq!(openings[i], params.open_single(polynomial, domain, i).unwrap().witness);
        }
    }

    #[test]
    fn same_degree_reuses_basis_without_mutating_it() {
        let params = KzgParams::from_tau(16, Fr::from(7u64)).unwrap();
        let domain = radix2_domain(8).unwrap();
        let first_polynomial = coefficients(8, 7);
        let first_openings = params.precompute_single_openings(&first_polynomial, &domain).unwrap();
        assert_dense_oracle(&params, &first_polynomial, &domain, &first_openings);
        let first = retained_basis(&params).expect("first opening call did not retain the SRS FFT");
        let unchanged_points = first.1.as_ref().clone();
        let second_polynomial = coefficients(8, 19);
        let second_openings = params.precompute_single_openings(&second_polynomial, &domain).unwrap();
        assert_dense_oracle(&params, &second_polynomial, &domain, &second_openings);
        let second = retained_basis(&params).unwrap();
        assert!(Arc::ptr_eq(&first.1, &second.1), "same degree rebuilt the fixed SRS FFT");
        assert_eq!(first.1.as_ref(), &unchanged_points, "pointwise multiplication mutated shared SRS data");
    }

    #[test]
    fn degree_switch_replaces_one_entry_and_keeps_openings_correct() {
        let params = KzgParams::from_tau(16, Fr::from(7u64)).unwrap();
        let domain = radix2_domain(16).unwrap();
        let mut previous = None;
        // Degree 6 and 7 have the same FFT size: the degree itself must be keyed.
        for length in [8, 7, 4, 8] {
            let polynomial = coefficients(length, 11);
            let openings = params.precompute_single_openings(&polynomial, &domain).unwrap();
            assert_dense_oracle(&params, &polynomial, &domain, &openings);
            let current = retained_basis(&params).expect("degree switch did not retain a basis");
            if let Some(old) = previous {
                assert!(!Arc::ptr_eq(&old, &current.1), "a different-degree basis was reused");
            }
            previous = Some(current.1);
        }
    }

    #[test]
    fn parameter_clones_share_basis_and_distinct_srs_do_not() {
        let params = KzgParams::from_tau(16, Fr::from(7u64)).unwrap();
        let clone = params.clone();
        let other = KzgParams::from_tau(16, Fr::from(11u64)).unwrap();
        let domain = radix2_domain(8).unwrap();
        let polynomial = coefficients(8, 5);
        let first_openings = params.precompute_single_openings(&polynomial, &domain).unwrap();
        let first = retained_basis(&params).expect("original parameters have no prepared basis");
        let cloned_openings = clone.precompute_single_openings(&polynomial, &domain).unwrap();
        let cloned = retained_basis(&clone).unwrap();
        assert!(Arc::ptr_eq(&first.1, &cloned.1));
        assert_eq!(first_openings, cloned_openings);
        assert!(retained_basis(&other).is_none());
        let other_openings = other.precompute_single_openings(&polynomial, &domain).unwrap();
        let other_basis = retained_basis(&other).unwrap();
        assert!(!Arc::ptr_eq(&first.1, &other_basis.1));
        assert_ne!(first.1.as_ref(), other_basis.1.as_ref());
        assert_dense_oracle(&other, &polynomial, &domain, &other_openings);
    }

    #[test]
    fn constants_and_invalid_inputs_do_not_prepare_or_evict_basis() {
        let params = KzgParams::from_tau(8, Fr::from(7u64)).unwrap();
        let domain = radix2_domain(8).unwrap();
        for polynomial in [vec![], vec![Fr::from(3u64)]] {
            assert_eq!(params.precompute_single_openings(&polynomial, &domain).unwrap(),
                vec![G1Affine::identity(); 8]);
        }
        assert!(params.precompute_single_openings(&coefficients(9, 3), &domain).is_err());
        let larger_domain = radix2_domain(16).unwrap();
        assert!(params.precompute_single_openings(&coefficients(10, 3), &larger_domain).is_err());
        assert!(retained_basis(&params).is_none());
        let polynomial = coefficients(8, 9);
        let openings = params.precompute_single_openings(&polynomial, &domain).unwrap();
        assert_dense_oracle(&params, &polynomial, &domain, &openings);
        let first = retained_basis(&params).expect("valid input did not retain a basis");
        params.precompute_single_openings(&[], &domain).unwrap();
        assert!(params.precompute_single_openings(&coefficients(9, 3), &domain).is_err());
        let second = retained_basis(&params).unwrap();
        assert!(Arc::ptr_eq(&first.1, &second.1));
    }

    #[test]
    fn opening_domain_and_coset_do_not_change_the_convolution_basis() {
        let params = KzgParams::from_tau(16, Fr::from(7u64)).unwrap();
        let polynomial = coefficients(7, 13);
        let domain = radix2_domain(8).unwrap();
        let first_openings = params.precompute_single_openings(&polynomial, &domain).unwrap();
        assert_dense_oracle(&params, &polynomial, &domain, &first_openings);
        let first = retained_basis(&params).expect("original domain did not retain a basis");
        let coset = radix2_domain(16).unwrap().get_coset(Fr::from(5u64)).unwrap();
        let other_openings = params.precompute_single_openings(&polynomial, &coset).unwrap();
        assert_dense_oracle(&params, &polynomial, &coset, &other_openings);
        let other = retained_basis(&params).unwrap();
        assert!(Arc::ptr_eq(&first.1, &other.1));
    }

    #[test]
    fn simultaneous_callers_get_independent_correct_openings() {
        let params = Arc::new(KzgParams::from_tau(16, Fr::from(7u64)).unwrap());
        let barrier = Arc::new(Barrier::new(3));
        let mut threads = Vec::new();
        for offset in [3, 17] {
            let caller = Arc::clone(&params);
            let ready = Arc::clone(&barrier);
            threads.push(std::thread::spawn(move || {
                let domain = radix2_domain(8).unwrap();
                let polynomial = coefficients(8, offset);
                ready.wait();
                let openings = caller.precompute_single_openings(&polynomial, &domain).unwrap();
                assert_dense_oracle(&caller, &polynomial, &domain, &openings);
                retained_basis(&caller).expect("concurrent caller did not retain a basis").1
            }));
        }
        barrier.wait();
        let first = threads.remove(0).join().unwrap();
        let second = threads.remove(0).join().unwrap();
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn poisoned_cache_lock_preserves_published_srs_and_valid_openings() {
        let params = Arc::new(KzgParams::from_tau(16, Fr::from(7u64)).unwrap());
        let domain = radix2_domain(8).unwrap();
        let polynomial = coefficients(8, 3);
        params.precompute_single_openings(&polynomial, &domain).unwrap();
        let first = retained_basis(&params).expect("initial basis was not retained");
        let caller = Arc::clone(&params);
        assert!(std::thread::spawn(move || {
            let _guard = caller.opening_basis.lock().unwrap();
            panic!("intentional test panic while holding the immutable-basis cache lock");
        }).join().is_err());
        let openings = params.precompute_single_openings(&polynomial, &domain).unwrap();
        assert_dense_oracle(&params, &polynomial, &domain, &openings);
        let after = retained_basis(&params).unwrap();
        assert!(Arc::ptr_eq(&first.1, &after.1));
    }
}
