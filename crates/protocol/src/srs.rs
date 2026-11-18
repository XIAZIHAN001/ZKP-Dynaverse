//! Shared setup for the CPU Dynaverse reference proof.
//!
//! Trapdoors are sampled and discarded in setup. They are never stored in
//! public proving or verifying parameters.

use crate::dynamo::{DynamoParams, DynamoPermutation};
use crate::consistency::{ConsistencyVerifyingKey, SliceConsistencyKey};
use crate::kzg::KzgParams;
use crate::trace::{CpuPhase, CpuTrace};
use crate::OperatorBackend;
use anyhow::{ensure, Result};
use ark_bn254::{Fr, G1Affine, G2Affine};
#[cfg(test)]
use ark_bn254::{G1Projective, G2Projective};
#[cfg(test)]
use ark_ec::CurveGroup;
use ark_ff::{batch_inversion, Field, One, Zero};
use ark_poly::EvaluationDomain;
use std::sync::Arc;
use ark_std::{rand::Rng, UniformRand};

#[derive(Clone, Debug)]
pub struct SharedSrs {
    pub dynamo: DynamoParams,
    pub kzg_x: KzgParams,
    pub kzg_y: KzgParams,
    pub consistency: [SliceConsistencyKey; 6],
    pub m: usize,
    pub n: usize,
    /// Prover-only bases for sublinear updates; never part of a verifying key.
    pub sparse: SparseUpdateBases,
}

/// Core material shared by both setups. Dynaverse has no Y monomial table.
#[derive(Clone, Debug)]
pub(crate) struct DynaverseSrs {
    pub(crate) dynamo: DynamoParams,
    pub(crate) kzg_x: KzgParams,
    pub(crate) consistency: [SliceConsistencyKey; 6],
    pub(crate) sparse: SparseUpdateBases,
}

/// Prover-only bases evaluated at tau_x before the trapdoor is dropped.
/// They are linear in the trapdoor-evaluated polynomials, so a sparse MSM over
/// them equals the KZG commitment of the corresponding dense polynomial.
#[derive(Clone, Debug)]
pub struct SparseUpdateBases {
    /// omega^j of the small (size n) domain, j in [0, n).
    pub(crate) small_points: Arc<Vec<Fr>>,
    /// [L'_j(tau_x)]_1: small-domain Lagrange basis.
    pub(crate) small_lagrange_g1: Arc<Vec<G1Affine>>,
    /// [L'_j(tau_x)]_2: small-domain Lagrange basis in G2.
    pub(crate) small_lagrange_g2: Arc<Vec<G2Affine>>,
    /// [u_j(tau_x)]_1 with u_j = (L'_j - 1)/(X - omega^j) (single-point opening of L'_j).
    pub(crate) opening_u_g1: Arc<Vec<G1Affine>>,
    /// [L_j(tau_x)/I_x(tau_x)]_1 for big-domain j in [0, 6n), where I_x vanishes
    /// on the public roots [6n, m). Each L_j/I_x is a polynomial for j < 6n.
    pub(crate) public_quotient_g1: Arc<Vec<G1Affine>>,
}

impl SparseUpdateBases {
    #[cfg(test)]
    pub(crate) fn from_trapdoor(m: usize, n: usize, tau_x: Fr) -> Result<Self> {
        Self::from_trapdoor_with_backend(m, n, tau_x, None)
    }

    fn from_trapdoor_with_backend(m: usize, n: usize, tau_x: Fr,
        backend: Option<&dyn crate::OperatorBackend>) -> Result<Self> {
        ensure!(n.checked_mul(6).is_some_and(|six_n| six_n <= m), "m must fit six wire slices");
        let small = crate::domain::radix2_domain(n)?;
        let big = crate::domain::radix2_domain(m)?;
        let tau_n = tau_x.pow([n as u64]);
        let tau_m = tau_x.pow([m as u64]);
        ensure!(tau_n != Fr::one() && tau_m != Fr::one(), "tau_x must be outside both domains");
        let inv_n = Fr::from(n as u64).inverse().expect("nonzero domain size");
        let inv_m = Fr::from(m as u64).inverse().expect("nonzero domain size");

        // Small domain: L'_j(tau) = (tau^n - 1) * omega^j / (n * (tau - omega^j)).
        let mut small_points = Vec::with_capacity(n);
        let mut point = Fr::one();
        for _ in 0..n {
            small_points.push(point);
            point *= small.group_gen;
        }
        let mut small_inverses = Vec::with_capacity(n);
        for &x in &small_points { small_inverses.push(tau_x - x); }
        batch_inversion(&mut small_inverses);
        let mut lagrange = Vec::with_capacity(n);
        let mut opening = Vec::with_capacity(n);
        for j in 0..n {
            let value = (tau_n - Fr::one()) * small_points[j] * inv_n * small_inverses[j];
            lagrange.push(value);
            opening.push((value - Fr::one()) * small_inverses[j]);
        }

        // Big domain: L_j(tau)/I_x(tau), I_x(tau) = prod_{i in [6n, m)} (tau - phi^i).
        let mut big_points = Vec::with_capacity(m);
        let mut point = Fr::one();
        for _ in 0..m {
            big_points.push(point);
            point *= big.group_gen;
        }
        let mut public_vanishing = Fr::one();
        for &x in &big_points[6 * n..] { public_vanishing *= tau_x - x; }
        let public_inverse = public_vanishing.inverse()
            .ok_or_else(|| anyhow::anyhow!("tau_x lies on a public root"))?;
        let mut big_inverses = Vec::with_capacity(6 * n);
        for &x in &big_points[..6 * n] { big_inverses.push(tau_x - x); }
        batch_inversion(&mut big_inverses);
        let mut public_quotient = Vec::with_capacity(6 * n);
        for j in 0..6 * n {
            let l_j = (tau_m - Fr::one()) * big_points[j] * inv_m * big_inverses[j];
            public_quotient.push(l_j * public_inverse);
        }

        let mut l1 = Vec::with_capacity(n);
        let mut l2 = Vec::with_capacity(n);
        let mut u1 = Vec::with_capacity(n);
        for j in 0..n {
            l1.push(lagrange[j]);
            l2.push(lagrange[j]);
            u1.push(opening[j]);
        }
        let mut q1 = Vec::with_capacity(6 * n);
        for &value in &public_quotient { q1.push(value); }
        Ok(Self {
            small_points: Arc::new(small_points),
            small_lagrange_g1: Arc::new(crate::setup::g1(&l1, backend)?),
            small_lagrange_g2: Arc::new(crate::setup::g2(&l2, backend)?),
            opening_u_g1: Arc::new(crate::setup::g1(&u1, backend)?),
            public_quotient_g1: Arc::new(crate::setup::g1(&q1, backend)?),
        })
    }
}

impl SharedSrs {
    pub fn consistency_verifying_key(&self) -> ConsistencyVerifyingKey {
        consistency_verifying_key(&self.consistency)
    }

    pub fn setup<R: Rng + ?Sized>(
        m: usize,
        n: usize,
        sigma: DynamoPermutation,
        rng: &mut R,
    ) -> Result<Self> {
        Self::setup_with_backend(m, n, sigma, rng, None)
    }

    pub(crate) fn setup_with_backend<R: Rng + ?Sized>(
        m: usize,
        n: usize,
        sigma: DynamoPermutation,
        rng: &mut R,
        backend: Option<&dyn OperatorBackend>,
    ) -> Result<Self> {
        let (core, kzg_y) = DynaverseSrs::setup_impl(
            m, n, sigma, rng, backend, &mut CpuTrace::disabled(), true,
        )?;
        let DynaverseSrs { dynamo, kzg_x, consistency, sparse } = core;
        let kzg_y = kzg_y.expect("full shared setup requested Y monomial powers");
        Ok(Self { dynamo, kzg_x, kzg_y, consistency, m, n, sparse })
    }
}

impl DynaverseSrs {
    pub(crate) fn consistency_verifying_key(&self) -> ConsistencyVerifyingKey {
        consistency_verifying_key(&self.consistency)
    }

    pub(crate) fn setup_traced<R: Rng + ?Sized>(
        m: usize,
        n: usize,
        sigma: DynamoPermutation,
        rng: &mut R,
        backend: Option<&dyn OperatorBackend>,
        trace: &mut CpuTrace,
    ) -> Result<Self> {
        let (core, _) = Self::setup_impl(m, n, sigma, rng, backend, trace, false)?;
        Ok(core)
    }

    fn setup_impl<R: Rng + ?Sized>(
        m: usize,
        n: usize,
        sigma: DynamoPermutation,
        rng: &mut R,
        backend: Option<&dyn OperatorBackend>,
        trace: &mut CpuTrace,
        include_kzg_y: bool,
    ) -> Result<(Self, Option<KzgParams>)> {
        let started=trace.start();
        ensure!(m > 0 && m.is_power_of_two(), "m must be a nonzero power of two");
        ensure!(n > 0 && n.is_power_of_two(), "n must be a nonzero power of two");
        ensure!(n.checked_mul(6).is_some_and(|six_n| six_n <= m), "m must fit six wire slices");
        ensure!(sigma.len() == m, "permutation size differs from m");
        let domain = crate::domain::radix2_domain(m)?;
        let ratio = m / n;
        let phi = domain.element(ratio);
        let mut tau_x;
        let mut tau_y;
        loop {
            tau_x = Fr::rand(rng);
            tau_y = Fr::rand(rng);
            if tau_x.is_zero()
                || tau_y.is_zero()
                || tau_x.pow([m as u64]) == Fr::one()
                || tau_y.pow([m as u64]) == Fr::one()
                || tau_x == tau_y
            {
                continue;
            }
            let y_r = tau_y.pow([ratio as u64]);
            if (0..6).any(|t| {
                let offset = t * n;
                let lambda = phi.pow([(n - offset % n) as u64]);
                tau_x == lambda * y_r
            }) {
                continue;
            }
            break;
        }
        trace.finish(CpuPhase::SetupTrapdoors,started);
        let started=trace.start();
        let dynamo = DynamoParams::from_trapdoors_with_backend(m, sigma, tau_x, tau_y, backend)?;
        trace.finish(CpuPhase::SetupDynamo,started);
        let started=trace.start();
        let kzg_x = KzgParams::from_tau_with_backend(m - 1, tau_x, backend)?;
        // Dynamo and consistency still use tau_y. Only the Y monomial table
        // is optional, and SharedSrs builds it here before the remaining bases.
        let kzg_y = if include_kzg_y {
            Some(KzgParams::from_tau_with_backend(m - 1, tau_y, backend)?)
        } else {
            None
        };
        trace.finish(CpuPhase::SetupKzg,started);
        let started=trace.start();
        let consistency = SliceConsistencyKey::setup_six(m,n,tau_x,tau_y,backend)?;
        trace.finish(CpuPhase::SetupConsistency,started);
        let started=trace.start();
        let sparse = SparseUpdateBases::from_trapdoor_with_backend(m, n, tau_x, backend)?;
        trace.finish(CpuPhase::SetupSparse,started);
        Ok((Self { dynamo, kzg_x, consistency, sparse }, kzg_y))
    }
}

fn consistency_verifying_key(keys: &[SliceConsistencyKey; 6]) -> ConsistencyVerifyingKey {
    ConsistencyVerifyingKey { slices: std::array::from_fn(|i| keys[i].verifying.clone()) }
}

#[cfg(test)]
mod sparse_basis_tests {
    use super::*;
    use crate::domain::{divide_exact, vanishing_polynomial};
    use ark_ec::VariableBaseMSM;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn immutable_srs_clones_and_common_consistency_columns_share_storage() {
        let srs=SharedSrs::setup(64,8,DynamoPermutation::try_new((0..64).collect()).unwrap(),
            &mut StdRng::seed_from_u64(7101)).unwrap();
        let cloned=srs.clone();
        assert_eq!(srs.kzg_x.rebuild_g1_powers().as_ptr(),cloned.kzg_x.rebuild_g1_powers().as_ptr(),
            "KZG clone copied immutable SRS");
        assert_eq!(srs.dynamo.rebuild_bases()[0].as_ptr(),cloned.dynamo.rebuild_bases()[0].as_ptr());
        for column in 1..7 {
            for key in &srs.consistency[1..] {
                assert_eq!(srs.consistency[0].rebuild_bases()[column].as_ptr(),key.rebuild_bases()[column].as_ptr(),
                    "common column {column} duplicated");
            }
        }
        assert_ne!(srs.consistency[0].rebuild_bases()[0].as_ptr(),srs.consistency[1].rebuild_bases()[0].as_ptr());
    }

    #[test]
    fn small_lagrange_bases_match_dense_kzg_in_both_groups() {
        let mut rng = StdRng::seed_from_u64(15101);
        for n in [1usize, 2, 8, 32] {
            let tau = Fr::from(171u64);
            let bases = SparseUpdateBases::from_trapdoor(8 * n, n, tau).unwrap();
            let kzg = KzgParams::from_tau(8 * n - 1, tau).unwrap();
            let domain = crate::domain::radix2_domain(n).unwrap();
            for nonzero in [false, true] {
                let mut values = vec![Fr::zero(); n];
                if nonzero {
                    for i in [0, n / 2, n - 1] { values[i] = Fr::rand(&mut rng); }
                }
                let polynomial = domain.ifft(&values);
                assert_eq!(G1Projective::msm_unchecked(&bases.small_lagrange_g1, &values).into_affine(),
                    kzg.commit(&polynomial).unwrap().point);
                assert_eq!(G2Projective::msm_unchecked(&bases.small_lagrange_g2, &values).into_affine(),
                    kzg.commit_g2(&polynomial).unwrap());
            }
            assert_eq!(*bases.small_points, domain.elements().collect::<Vec<_>>());
        }
    }

    #[test]
    fn opening_and_public_quotient_bases_match_dense_exact_division() {
        let mut rng = StdRng::seed_from_u64(15102);
        for n in [1usize, 2, 8, 32] {
            let tau = Fr::from(173u64);
            let m = 8 * n;
            let bases = SparseUpdateBases::from_trapdoor(m, n, tau).unwrap();
            let kzg = KzgParams::from_tau(m - 1, tau).unwrap();
            let small = crate::domain::radix2_domain(n).unwrap();
            for i in [0, n / 2, n - 1] {
                let mut values = vec![Fr::zero(); n];
                values[i] = Fr::one();
                let mut numerator = small.ifft(&values);
                numerator[0] -= Fr::one();
                let quotient = divide_exact(&numerator, &[-small.element(i), Fr::one()]).unwrap();
                assert_eq!(bases.opening_u_g1[i], kzg.commit(&quotient).unwrap().point);
            }
            let big = crate::domain::radix2_domain(m).unwrap();
            let divisor = vanishing_polynomial(&big, &(6 * n..m).collect::<Vec<_>>()).unwrap();
            for nonzero in [false, true] {
                let mut values = vec![Fr::zero(); m];
                if nonzero {
                    for i in [0, n - 1, 3 * n, 6 * n - 1] { values[i] = Fr::rand(&mut rng); }
                }
                let quotient = divide_exact(&big.ifft(&values), &divisor).unwrap();
                let sparse_commitment = G1Projective::msm_unchecked(
                    &bases.public_quotient_g1, &values[..6 * n]).into_affine();
                assert_eq!(sparse_commitment, kzg.commit(&quotient).unwrap().point);
            }
        }
    }

    #[test]
    fn sparse_bases_reject_invalid_domains_and_domain_trapdoors() {
        for (m, n, tau) in [(8, 0, 7u64), (8, 3, 7), (4, 1, 7), (8, 1, 1)] {
            assert!(SparseUpdateBases::from_trapdoor(m, n, Fr::from(tau)).is_err());
        }
    }
}

#[cfg(test)]
mod setup_tests {
    use super::*;
    use crate::dynamo::dynamo_verify_with_key;
    use crate::dynaverse::DynaverseParams;
    use crate::{toy_circuit::ToyCircuit, BatchMsmRequest};
    use rand::{rngs::StdRng, RngCore, SeedableRng};
    use std::cell::Cell;

    #[derive(Default)]
    struct FixedBaseSpy {
        g1_points: Cell<usize>,
        g2_points: Cell<usize>,
    }

    impl OperatorBackend for FixedBaseSpy {
        fn fixed_base_g1(&self, scalars: &[Fr]) -> Result<Vec<G1Affine>> {
            self.g1_points.set(self.g1_points.get() + scalars.len());
            crate::setup::cpu_g1(scalars)
        }

        fn fixed_base_g2(&self, scalars: &[Fr]) -> Result<Vec<G2Affine>> {
            self.g2_points.set(self.g2_points.get() + scalars.len());
            crate::setup::cpu_g2(scalars)
        }

        fn msm(&self, _: &[G1Affine], _: &[Fr]) -> Result<G1Affine> {
            anyhow::bail!("unexpected setup MSM dispatch")
        }

        fn msm_batch(&self, _: &BatchMsmRequest<'_>) -> Result<Vec<G1Affine>> {
            anyhow::bail!("unexpected setup batch MSM dispatch")
        }

        fn ntt(&self, _: &mut [Fr], _: bool) -> Result<()> {
            anyhow::bail!("unexpected setup NTT dispatch")
        }
    }

    #[test]
    fn dynaverse_setup_skips_y_monomial_powers() {
        let circuit = ToyCircuit::new(2).unwrap();
        let m = circuit.m();
        let n = circuit.n();
        let full_backend = FixedBaseSpy::default();
        let private_backend = FixedBaseSpy::default();
        let mut full_rng = StdRng::seed_from_u64(7102);
        let mut private_rng = StdRng::seed_from_u64(7102);
        let full = SharedSrs::setup_with_backend(
            m, n, circuit.permutation(), &mut full_rng, Some(&full_backend),
        ).unwrap();
        let params = DynaverseParams::setup_with_backend(
            circuit.clone(), &mut private_rng, &private_backend,
        ).unwrap();

        // Count fixed-base dispatch only, not the CPU-only divisor and VK
        // preparation in DynaverseParams setup.
        assert_eq!(full_backend.g1_points.get(), private_backend.g1_points.get() + m);
        assert_eq!(full_backend.g2_points.get(), private_backend.g2_points.get() + m);
        assert_eq!(full_rng.next_u64(), private_rng.next_u64());
        assert_eq!(full.kzg_y.max_degree(), m - 1);
        assert_eq!(full.kzg_y.rebuild_g1_powers().len(), m);
        assert_eq!(full.kzg_y.rebuild_g2_powers().len(), m);

        // The full public SRS is still an independent dense commitment oracle
        // for the private X commitment and Dynamo's unmasked Y commitment.
        let (instance, witness) = circuit.valid_instance();
        let proof = params.prove_anchor(&instance, &witness).unwrap();
        let anchor = &proof.anchor;
        let mut w = witness.wires;
        w.resize(m, Fr::zero());
        let big = crate::domain::radix2_domain(m).unwrap();
        let sigma = circuit.permutation_ref().as_slice();
        let h: Vec<Fr> = (0..m).map(|i| w[i] - w[sigma[i]]).collect();
        assert_eq!(anchor.w_commitment, full.kzg_x.commit(&big.ifft(&w)).unwrap().point);
        assert_eq!(anchor.h_commitment, full.kzg_y.commit(&big.ifft(&h)).unwrap().point);
        let small = crate::domain::radix2_domain(n).unwrap();
        let s5_coefficients = small.ifft(&w[4 * n..5 * n]);
        assert_eq!(anchor.s5_g2, full.kzg_x.commit_g2(&s5_coefficients).unwrap());

        assert!(dynamo_verify_with_key(&full.dynamo.verifying_key(), &anchor.dynamo).unwrap());
        let wires = anchor.wires;
        assert!(full.consistency_verifying_key().verify_six(
            anchor.w_commitment,
            &[wires.s1, wires.s2, wires.s3, wires.s4, wires.s5, wires.s6],
            &anchor.consistency,
        ));
        let vk = params.verifying_key();
        assert!(vk.verify_anchor(&instance, &proof).unwrap());
        let mut wrong = instance;
        wrong.public_inputs[0] += Fr::one();
        assert!(!vk.verify_anchor(&wrong, &proof).unwrap());
    }
}
