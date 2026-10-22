//! CPU Dynamo permutation relation and sparse proof helpers.

use crate::types::{BatchMsmRequest, DynamoCommitments, DynamoProof, OperatorBackend};
use crate::verification::PairingChecks;
use anyhow::{ensure, Result};
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{pairing::Pairing, CurveGroup, Group, VariableBaseMSM};
use ark_ff::{Field, One, Zero};
use ark_std::{rand::Rng, UniformRand};
use std::collections::BTreeMap;
use std::sync::Arc;

/// A field vector with strictly increasing, nonzero entries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseVector {
    len: usize,
    entries: Vec<(usize, Fr)>,
}

impl SparseVector {
    pub fn try_new(len: usize, entries: Vec<(usize, Fr)>) -> Result<Self> {
        ensure!(len > 0, "sparse vector length must be positive");
        let mut previous = None;
        for &(index, value) in &entries {
            ensure!(index < len, "sparse index {index} is out of range");
            ensure!(!value.is_zero(), "sparse entries must be nonzero");
            ensure!(previous.is_none_or(|last| index > last), "sparse indices must increase strictly");
            previous = Some(index);
        }
        Ok(Self { len, entries })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn entries(&self) -> &[(usize, Fr)] {
        &self.entries
    }
}

/// A bijection on a nonempty radix-2 domain, with its inverse cached.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DynamoPermutation {
    sigma: Vec<usize>,
    inverse: Vec<usize>,
}

impl DynamoPermutation {
    pub fn try_new(sigma: Vec<usize>) -> Result<Self> {
        let m = sigma.len();
        ensure!(m > 0 && m.is_power_of_two(), "permutation size must be a nonzero power of two");
        let mut inverse = vec![usize::MAX; m];
        for (index, &image) in sigma.iter().enumerate() {
            ensure!(image < m, "permutation value {image} is out of range");
            ensure!(inverse[image] == usize::MAX, "permutation contains duplicate value {image}");
            inverse[image] = index;
        }
        Ok(Self { sigma, inverse })
    }

    pub fn len(&self) -> usize {
        self.sigma.len()
    }

    pub fn as_slice(&self) -> &[usize] {
        &self.sigma
    }
}

/// Precomputed non-universal Dynamo proving bases for one permutation.
#[derive(Clone, Debug)]
pub struct DynamoParams {
    m: usize,
    alpha_bases: Arc<Vec<G1Affine>>,
    beta_bases: Arc<Vec<G1Affine>>,
    z_bases: Arc<Vec<G1Affine>>,
    h_bases: Arc<Vec<G1Affine>>,
    b_bases: Arc<Vec<G1Affine>>,
    lx_bases: Arc<Vec<G1Affine>>,
    ly_bases: Arc<Vec<G1Affine>>,
    proving: ProvingConstants,
    verifying: DynamoVerifyingKey,
}

#[derive(Clone, Debug)]
struct ProvingConstants {
    u: G1Affine,
    x: G1Affine,
    x2: G1Affine,
    xm: G1Affine,
    ym: G1Affine,
    xm_ym: G1Affine,
    x2_ym: G1Affine,
}

#[derive(Clone, Debug)]
pub struct DynamoVerifyingKey {
    m: usize,
    u: G2Affine,
    x: G2Affine,
    x2: G2Affine,
    xm: G2Affine,
    ym: G2Affine,
    inv_m: G2Affine,
}

impl DynamoVerifyingKey {
    pub fn size(&self) -> usize {
        self.m
    }
}

impl DynamoParams {
    /// Read-only linear bases used by the bounded anchor builder. No setup
    /// material is copied on an update; device registration happens once.
    pub(crate) fn rebuild_bases(&self) -> [&[G1Affine]; 6] {
        [&self.alpha_bases, &self.beta_bases, &self.z_bases,
            &self.h_bases, &self.b_bases, &self.lx_bases]
    }

    /// h[i]=z[i]-z[sigma[i]], hence its commitment is linear in z. Moving
    /// this fixed permutation into the bases avoids a hot dense gather.
    pub(crate) fn rebuild_h_bases(&self, sigma: &DynamoPermutation) -> Result<Vec<G1Affine>> {
        ensure!(sigma.len() == self.m, "rebuild permutation size mismatch");
        let mut points = Vec::with_capacity(self.m);
        for i in 0..self.m {
            points.push(G1Projective::from(self.ly_bases[i])
                - G1Projective::from(self.ly_bases[sigma.inverse[i]]));
        }
        Ok(G1Projective::normalize_batch(&points))
    }

    /// A public unit vector at j has h[j]=1 and h[sigma^-1(j)]=-1.
    /// Form only the requested points instead of n full-vector commitments.
    pub(crate) fn public_h_bases(&self, sigma: &DynamoPermutation,
        indices: std::ops::Range<usize>) -> Result<Vec<G1Affine>> {
        ensure!(sigma.len() == self.m && indices.end <= self.m,
            "public permutation range mismatch");
        crate::workcount::record_scan(2 * indices.len());
        let mut points = Vec::with_capacity(indices.len());
        for i in indices {
            points.push(G1Projective::from(self.ly_bases[i])
                - G1Projective::from(self.ly_bases[sigma.inverse[i]]));
        }
        Ok(G1Projective::normalize_batch(&points))
    }
    pub fn setup<R: Rng + ?Sized>(
        m: usize,
        sigma: DynamoPermutation,
        rng: &mut R,
    ) -> Result<Self> {
        ensure!(m > 0 && m.is_power_of_two(), "Dynamo size must be a nonzero power of two");
        ensure!(sigma.len() == m, "permutation size differs from Dynamo size");
        let tau_x = sample_outside_domain(m, rng);
        let tau_y = sample_outside_domain(m, rng);
        Self::from_trapdoors(m, sigma, tau_x, tau_y)
    }

    pub(crate) fn from_trapdoors(
        m: usize,
        sigma: DynamoPermutation,
        tau_x: Fr,
        tau_y: Fr,
    ) -> Result<Self> {
        Self::from_trapdoors_with_backend(m, sigma, tau_x, tau_y, None)
    }

    pub(crate) fn from_trapdoors_with_backend(m: usize, sigma: DynamoPermutation,
        tau_x: Fr, tau_y: Fr, backend: Option<&dyn OperatorBackend>) -> Result<Self> {
        ensure!(m > 0 && m.is_power_of_two(), "Dynamo size must be a nonzero power of two");
        ensure!(sigma.len() == m, "permutation size differs from Dynamo size");
        ensure!(
            !tau_x.is_zero() && tau_x.pow([m as u64]) != Fr::one(),
            "tau_x must be outside the domain"
        );
        ensure!(
            !tau_y.is_zero() && tau_y.pow([m as u64]) != Fr::one(),
            "tau_y must be outside the domain"
        );
        let domain = crate::domain::radix2_domain(m)?;
        let inv_m = Fr::from(m as u64).inverse().expect("nonzero domain size");
        let tau_x_m = tau_x.pow([m as u64]);
        let tau_y_m = tau_y.pow([m as u64]);
        let x_vanishing_inv = (tau_x_m - Fr::one()).inverse().expect("tau_x outside domain");
        let tau_x_inv = tau_x.inverse().expect("tau_x nonzero");

        let lx = crate::setup::lagrange_values(tau_x, &domain)?;
        let ly = crate::setup::lagrange_values(tau_y, &domain)?;
        let y: Vec<Fr> = (0..m)
            .map(|i| ly[i] - ly[sigma.inverse[i]])
            .collect();
        let u = lx.iter().zip(&y).fold(Fr::zero(), |sum, (&l, &yi)| sum + l * yi);
        let g1 = G1Projective::generator();
        let mut alpha_bases = Vec::with_capacity(m);
        let mut beta_bases = Vec::with_capacity(m);
        let mut z_bases = Vec::with_capacity(m);
        let mut h_bases = Vec::with_capacity(m);
        let mut b_bases = Vec::with_capacity(m);
        let mut lx_bases = Vec::with_capacity(m);
        let mut ly_bases = Vec::with_capacity(m);
        for i in 0..m {
            let common = y[i] * (lx[i] - inv_m);
            alpha_bases.push(lx[i] * (u - y[i]) * x_vanishing_inv);
            beta_bases.push(common * tau_x_inv);
            z_bases.push(tau_y_m * lx[i]);
            h_bases.push(tau_x_m * y[i]);
            b_bases.push(common * tau_x);
            lx_bases.push(lx[i]);
            ly_bases.push(ly[i]);
        }
        let alpha_bases = crate::setup::g1(&alpha_bases, backend)?;
        let beta_bases = crate::setup::g1(&beta_bases, backend)?;
        let z_bases = crate::setup::g1(&z_bases, backend)?;
        let h_bases = crate::setup::g1(&h_bases, backend)?;
        let b_bases = crate::setup::g1(&b_bases, backend)?;
        let lx_bases = crate::setup::g1(&lx_bases, backend)?;
        let ly_bases = crate::setup::g1(&ly_bases, backend)?;
        let x2 = tau_x * tau_x;
        let proving = ProvingConstants {
            u: (g1 * u).into_affine(),
            x: (g1 * tau_x).into_affine(),
            x2: (g1 * x2).into_affine(),
            xm: (g1 * tau_x_m).into_affine(),
            ym: (g1 * tau_y_m).into_affine(),
            xm_ym: (g1 * (tau_x_m * tau_y_m)).into_affine(),
            x2_ym: (g1 * (x2 * tau_y_m)).into_affine(),
        };
        let g2 = G2Projective::generator();
        let verifying = DynamoVerifyingKey {
            m,
            u: (g2 * u).into_affine(),
            x: (g2 * tau_x).into_affine(),
            x2: (g2 * x2).into_affine(),
            xm: (g2 * tau_x_m).into_affine(),
            ym: (g2 * tau_y_m).into_affine(),
            inv_m: (g2 * inv_m).into_affine(),
        };
        Ok(Self {
            m,
            alpha_bases: Arc::new(alpha_bases),
            beta_bases: Arc::new(beta_bases),
            z_bases: Arc::new(z_bases),
            h_bases: Arc::new(h_bases),
            b_bases: Arc::new(b_bases),
            lx_bases: Arc::new(lx_bases),
            ly_bases: Arc::new(ly_bases),
            proving,
            verifying,
        })
    }

    pub fn commit_five_sparse(
        &self,
        z: &SparseVector,
        backend: &impl OperatorBackend,
    ) -> Result<DynamoCommitments> {
        ensure!(z.len() == self.m, "vector size differs from Dynamo setup");
        let mut alpha = Vec::with_capacity(z.entries().len());
        let mut beta = Vec::with_capacity(z.entries().len());
        let mut z_points = Vec::with_capacity(z.entries().len());
        let mut h_points = Vec::with_capacity(z.entries().len());
        let mut b = Vec::with_capacity(z.entries().len());
        let mut scalars = Vec::with_capacity(z.entries().len());
        for &(index, scalar) in z.entries() {
            alpha.push(self.alpha_bases[index]);
            beta.push(self.beta_bases[index]);
            z_points.push(self.z_bases[index]);
            h_points.push(self.h_bases[index]);
            b.push(self.b_bases[index]);
            scalars.push(scalar);
        }
        let point_sets: [&[G1Affine]; 5] = [&alpha, &beta, &z_points, &h_points, &b];
        let lengths = [scalars.len(); 5];
        let offsets = [0usize; 5];
        let request = BatchMsmRequest {
            point_sets: &point_sets,
            shared_scalars: &scalars,
            offsets: &offsets,
            lengths: &lengths,
        };
        let commitments = backend.msm_batch(&request)?;
        ensure!(commitments.len() == 5, "backend returned wrong Dynamo commitment count");
        Ok(DynamoCommitments {
            alpha: commitments[0],
            beta: commitments[1],
            z: commitments[2],
            h: commitments[3],
            b: commitments[4],
        })
    }

    /// Dense CPU reference and fallback for checking the sparse gather path.
    pub fn commit_five_dense_reference(&self, z: &[Fr]) -> Result<DynamoCommitments> {
        ensure!(z.len() == self.m, "vector size differs from Dynamo setup");
        crate::workcount::record_msm(5 * z.len());
        let msm = |bases: &[G1Affine]| G1Projective::msm_unchecked(bases, z).into_affine();
        Ok(DynamoCommitments {
            alpha: msm(&self.alpha_bases),
            beta: msm(&self.beta_bases),
            z: msm(&self.z_bases),
            h: msm(&self.h_bases),
            b: msm(&self.b_bases),
        })
    }

    /// Unmasked X-domain instance commitment for binding to the shared KZG SRS.
    pub fn commit_z_dense_unmasked(&self, z: &[Fr]) -> Result<G1Affine> {
        ensure!(z.len() == self.m, "vector size differs from Dynamo setup");
        crate::workcount::record_msm(z.len());
        Ok(G1Projective::msm_unchecked(&self.lx_bases, z).into_affine())
    }

    #[cfg(test)]
    pub(crate) fn commit_h_dense_unmasked(&self, h: &[Fr]) -> Result<G1Affine> {
        ensure!(h.len() == self.m, "vector size differs from Dynamo setup");
        crate::workcount::record_msm(h.len());
        Ok(G1Projective::msm_unchecked(&self.ly_bases, h).into_affine())
    }

    /// Public verification material; no proving bases are included.
    pub fn verifying_key(&self) -> DynamoVerifyingKey {
        self.verifying.clone()
    }
}

/// Build a Dynamo proof without masks so its instances bind directly to the
/// shared X/Y KZG commitments.
pub fn dynamo_prove_unmasked(
    params: &DynamoParams,
    z: &SparseVector,
    h: &SparseVector,
    backend: &impl OperatorBackend,
) -> Result<DynamoProof> {
    ensure!(z.len() == params.m && h.len() == params.m, "witness size differs from Dynamo setup");
    let commitments = params.commit_five_sparse(z, backend)?;
    let instance_z = sparse_msm(&params.lx_bases, z, backend)?;
    let instance_h = sparse_msm(&params.ly_bases, h, backend)?;
    Ok(DynamoProof {
        commitments,
        q: G1Affine::identity(),
        instance_z,
        instance_h,
    })
}

/// Build a non-universal Dynamo proof from sparse z and h vectors.
pub fn dynamo_prove_sparse<R: Rng + ?Sized>(
    params: &DynamoParams,
    z: &SparseVector,
    h: &SparseVector,
    backend: &impl OperatorBackend,
    rng: &mut R,
) -> Result<DynamoProof> {
    ensure!(z.len() == params.m && h.len() == params.m, "witness size differs from Dynamo setup");
    let raw = params.commit_five_sparse(z, backend)?;
    let z_lin = sparse_msm(&params.lx_bases, z, backend)?;
    let h_lin = sparse_msm(&params.ly_bases, h, backend)?;
    let rho_z = Fr::rand(rng);
    let rho_h = Fr::rand(rng);
    let rho_qv = Fr::rand(rng);
    let rho_qh = Fr::rand(rng);
    let inv_m = Fr::from(params.m as u64).inverse().expect("nonzero domain size");
    let g1 = G1Projective::generator();
    let pk = &params.proving;
    let xm1 = G1Projective::from(pk.xm) - g1;
    let ym1 = G1Projective::from(pk.ym) - g1;
    let instance_z = (G1Projective::from(z_lin) + xm1 * rho_z).into_affine();
    let instance_h = (G1Projective::from(h_lin) + ym1 * rho_h).into_affine();
    let q = (g1 * (-rho_h * inv_m) + xm1 * rho_qv + G1Projective::from(pk.x) * rho_qh)
        .into_affine();
    let commitments = DynamoCommitments {
        alpha: (G1Projective::from(raw.alpha) + G1Projective::from(pk.u) * rho_z - ym1 * rho_qv)
            .into_affine(),
        beta: (G1Projective::from(raw.beta) - ym1 * rho_qh).into_affine(),
        z: (G1Projective::from(raw.z)
            + (G1Projective::from(pk.xm_ym) - G1Projective::from(pk.ym)) * rho_z)
            .into_affine(),
        h: (G1Projective::from(raw.h)
            + (G1Projective::from(pk.xm_ym) - G1Projective::from(pk.xm)) * rho_h)
            .into_affine(),
        b: (G1Projective::from(raw.b)
            - (G1Projective::from(pk.x2_ym) - G1Projective::from(pk.x2)) * rho_qh)
            .into_affine(),
    };
    Ok(DynamoProof { commitments, q, instance_z, instance_h })
}

/// Verify all four non-universal Dynamo pairing identities.
pub fn dynamo_verify(params: &DynamoParams, proof: &DynamoProof) -> Result<bool> {
    dynamo_verify_with_key(&params.verifying, proof)
}

/// Verify with only the public G2 constants.
pub fn dynamo_verify_with_key(vk: &DynamoVerifyingKey, proof: &DynamoProof) -> Result<bool> {
    ensure!(vk.m > 0 && vk.m.is_power_of_two(), "invalid Dynamo verifying key size");
    let g2 = G2Projective::generator();
    let xm1 = (G2Projective::from(vk.xm) - g2).into_affine();
    let ym1 = (G2Projective::from(vk.ym) - g2).into_affine();
    let c = &proof.commitments;
    let main_left = Bn254::pairing(proof.instance_z, vk.u);
    let main_right = Bn254::pairing(c.alpha, xm1)
        + Bn254::pairing(c.beta, vk.x)
        + Bn254::pairing(proof.instance_h, vk.inv_m)
        + Bn254::pairing(proof.q, ym1);
    if main_left != main_right {
        return Ok(false);
    }
    let g2_affine = g2.into_affine();
    Ok(Bn254::pairing(proof.instance_z, vk.ym) == Bn254::pairing(c.z, g2_affine)
        && Bn254::pairing(proof.instance_h, vk.xm) == Bn254::pairing(c.h, g2_affine)
        && Bn254::pairing(c.beta, vk.x2) == Bn254::pairing(c.b, g2_affine))
}

pub(crate) fn dynamo_verify_with_checks(
    vk: &DynamoVerifyingKey, proof: &DynamoProof, checks: &mut PairingChecks<'_>,
) -> Result<bool> {
    ensure!(vk.m > 0 && vk.m.is_power_of_two(), "invalid Dynamo verifying key size");
    let g2 = G2Projective::generator();
    let xm1 = (G2Projective::from(vk.xm) - g2).into_affine();
    let ym1 = (G2Projective::from(vk.ym) - g2).into_affine();
    let c = &proof.commitments;
    if !checks.equal((proof.instance_z, vk.u), &[
        (c.alpha, xm1), (c.beta, vk.x), (proof.instance_h, vk.inv_m), (proof.q, ym1),
    ]) { return Ok(false); }
    let g2_affine = g2.into_affine();
    Ok(checks.equal((proof.instance_z, vk.ym), &[(c.z, g2_affine)])
        && checks.equal((proof.instance_h, vk.xm), &[(c.h, g2_affine)])
        && checks.equal((c.beta, vk.x2), &[(c.b, g2_affine)]))
}

fn sparse_msm(
    bases: &[G1Affine],
    values: &SparseVector,
    backend: &impl OperatorBackend,
) -> Result<G1Affine> {
    let mut points = Vec::with_capacity(values.entries().len());
    let mut scalars = Vec::with_capacity(values.entries().len());
    for &(index, value) in values.entries() {
        points.push(bases[index]);
        scalars.push(value);
    }
    backend.msm(&points, &scalars)
}

fn sample_outside_domain<R: Rng + ?Sized>(m: usize, rng: &mut R) -> Fr {
    loop {
        let value = Fr::rand(rng);
        if !value.is_zero() && value.pow([m as u64]) != Fr::one() {
            return value;
        }
    }
}

#[cfg(test)]
mod setup_basis_tests {
    use super::*;
    #[test]
    fn direct_public_h_bases_equal_independent_dense_commitments() {
        let m=32;
        let mut permutation:Vec<usize>=(0..m).collect();
        permutation.swap(0,24); permutation.swap(3,27);
        let sigma=DynamoPermutation::try_new(permutation).unwrap();
        let params=DynamoParams::from_trapdoors(m,sigma.clone(),Fr::from(19u64),Fr::from(23u64)).unwrap();
        let actual=params.public_h_bases(&sigma,24..28).unwrap();
        for (i,&point) in actual.iter().enumerate() {
            let unit=SparseVector::try_new(m,vec![(24+i,Fr::one())]).unwrap();
            let h=derive_h_sparse(&unit,&sigma).unwrap();
            let mut dense=vec![Fr::zero();m];
            for &(j,value) in h.entries() {dense[j]=value;}
            assert_eq!(point,params.commit_h_dense_unmasked(&dense).unwrap());
        }
    }
}

/// Compute h[i] = z[i] - z[sigma[i]] without scanning the dense domain.
pub fn derive_h_sparse(z: &SparseVector, sigma: &DynamoPermutation) -> Result<SparseVector> {
    ensure!(z.len == sigma.len(), "vector and permutation lengths differ");
    let mut values = BTreeMap::<usize, Fr>::new();
    for &(index, value) in &z.entries {
        *values.entry(index).or_insert_with(Fr::zero) += value;
        *values.entry(sigma.inverse[index]).or_insert_with(Fr::zero) -= value;
    }
    SparseVector::try_new(
        z.len,
        values.into_iter().filter(|(_, value)| !value.is_zero()).collect(),
    )
}
