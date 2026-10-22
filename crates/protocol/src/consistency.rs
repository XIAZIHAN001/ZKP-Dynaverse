//! Unmasked slice-consistency proof for a large-domain vector and one wire.
//!
//! Setup evaluates fixed polynomial bases at private trapdoors, then discards
//! the trapdoors. Proving is linear in the supplied vectors; it cannot evaluate
//! an arbitrary quotient at the trapdoors.

use crate::domain::radix2_domain;
use crate::types::{ConsistencyProof, OperatorBackend, SliceConsistencyProof};
use crate::verification::PairingChecks;
use anyhow::{ensure, Result};
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{pairing::Pairing, CurveGroup, Group};
use ark_ff::{Field, One, Zero};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct SliceConsistencyKey {
    m: usize,
    n: usize,
    offset: usize,
    qac_bases: Arc<Vec<G1Affine>>,
    qcd_bases: Arc<Vec<G1Affine>>,
    qbd_bases: Arc<Vec<G1Affine>>,
    d_bases: Arc<Vec<G1Affine>>,
    a_degree_bases: Arc<Vec<G1Affine>>,
    b_degree_bases: Arc<Vec<G1Affine>>,
    d_degree_bases: Arc<Vec<G1Affine>>,
    pub verifying: SliceConsistencyVerifyingKey,
}

#[derive(Clone, Debug)]
pub struct SliceConsistencyVerifyingKey {
    pub m: usize,
    pub n: usize,
    pub offset: usize,
    pub(crate) z_in: G2Affine,
    pub(crate) x_minus_y: G2Affine,
    pub(crate) x_minus_lambda_y_r: G2Affine,
    pub(crate) x_m: G2Affine,
    pub(crate) y_m: G2Affine,
}

#[derive(Clone, Debug)]
pub struct ConsistencyVerifyingKey {
    pub slices: [SliceConsistencyVerifyingKey; 6],
}

impl ConsistencyVerifyingKey {
    pub(crate) fn verify_six_with_checks(
        &self, a: G1Affine, bs: &[G1Affine; 6], proof: &ConsistencyProof,
        checks: &mut PairingChecks<'_>,
    ) -> bool {
        let m = self.slices[0].m;
        let n = self.slices[0].n;
        if m == 0 || !m.is_power_of_two() || n == 0 || !n.is_power_of_two()
            || n.checked_mul(6).is_none_or(|six_n| six_n > m) { return false; }
        for i in 0..6 {
            let vk = &self.slices[i];
            if vk.m != m || vk.n != n || vk.offset != i * n { return false; }
        }
        for i in 0..6 {
            if !self.slices[i].verify_with_checks(a, bs[i], &proof.slices[i], checks) {
                return false;
            }
        }
        true
    }

    pub fn verify_six(
        &self,
        a: G1Affine,
        bs: &[G1Affine; 6],
        proof: &ConsistencyProof,
    ) -> bool {
        let m = self.slices[0].m;
        let n = self.slices[0].n;
        if m == 0 || !m.is_power_of_two() || n == 0 || !n.is_power_of_two()
            || n.checked_mul(6).is_none_or(|six_n| six_n > m)
            || self.slices.iter().enumerate().any(|(i, vk)| {
                vk.m != m || vk.n != n || vk.offset != i * n
            })
        {
            return false;
        }
        self.slices
            .iter()
            .zip(bs.iter().zip(proof.slices.iter()))
            .all(|(vk, (&b, slice_proof))| vk.verify(a, b, slice_proof))
    }
}

impl SliceConsistencyVerifyingKey {
    fn verify_with_checks(
        &self, a: G1Affine, b: G1Affine, proof: &SliceConsistencyProof,
        checks: &mut PairingChecks<'_>,
    ) -> bool {
        let g2 = G2Projective::generator().into_affine();
        let a_minus_d = (G1Projective::from(a) - G1Projective::from(proof.d)).into_affine();
        let b_minus_d = (G1Projective::from(b) - G1Projective::from(proof.d)).into_affine();
        checks.equal((a_minus_d, g2), &[(proof.qac, self.z_in), (proof.qcd, self.x_minus_y)])
            && checks.equal((b_minus_d, g2), &[(proof.qbd, self.x_minus_lambda_y_r)])
            && checks.equal((a, self.y_m), &[(proof.a_degree, g2)])
            && checks.equal((b, self.y_m), &[(proof.b_degree, g2)])
            && checks.equal((proof.d, self.x_m), &[(proof.d_degree, g2)])
    }

    /// Verify Fig. 8's two relation checks and three degree checks.
    pub fn verify(&self, a: G1Affine, b: G1Affine, proof: &SliceConsistencyProof) -> bool {
        let g2 = G2Projective::generator().into_affine();
        let a_minus_d =
            (G1Projective::from(a) - G1Projective::from(proof.d)).into_affine();
        let b_minus_d =
            (G1Projective::from(b) - G1Projective::from(proof.d)).into_affine();
        Bn254::pairing(a_minus_d, g2)
            == Bn254::pairing(proof.qac, self.z_in)
                + Bn254::pairing(proof.qcd, self.x_minus_y)
            && Bn254::pairing(b_minus_d, g2)
                == Bn254::pairing(proof.qbd, self.x_minus_lambda_y_r)
            && Bn254::pairing(a, self.y_m) == Bn254::pairing(proof.a_degree, g2)
            && Bn254::pairing(b, self.y_m) == Bn254::pairing(proof.b_degree, g2)
            && Bn254::pairing(proof.d, self.x_m)
                == Bn254::pairing(proof.d_degree, g2)
    }
}

// Setup-local only: no static cache, no Arc, no trapdoor-derived scalar in SRS.
struct ConsistencyScalarPreparation {
    m:usize, n:usize, tau_x:Fr, tau_y:Fr,
    l_big_x:Vec<Fr>, l_small_x:Vec<Fr>, c_x:Vec<Fr>, c_y:Vec<Fr>,
}
impl ConsistencyScalarPreparation {
    fn new(m:usize,n:usize,tau_x:Fr,tau_y:Fr)->Result<Self> {
        ensure!(n>0 && m>=n && m%n==0,"n must divide m");
        let big=radix2_domain(m)?; let small=radix2_domain(n)?;
        let ratio=m/n;
        Ok(Self {m,n,tau_x,tau_y,
            l_big_x:lagrange_values(tau_x,&big)?,
            l_small_x:lagrange_values(tau_x,&small)?,
            c_x:lagrange_values(tau_x.pow([ratio as u64]),&small)?,
            c_y:lagrange_values(tau_y.pow([ratio as u64]),&small)?})
    }
}

impl SliceConsistencyKey {
    /// Borrowed in the same order as SliceConsistencyProof's named fields.
    pub(crate) fn rebuild_bases(&self) -> [&[G1Affine]; 7] {
        [&self.qac_bases, &self.qcd_bases, &self.qbd_bases, &self.d_bases,
            &self.a_degree_bases, &self.b_degree_bases, &self.d_degree_bases]
    }
    #[cfg(test)]
    pub(crate) fn from_trapdoors(
        m: usize,
        n: usize,
        offset: usize,
        tau_x: Fr,
        tau_y: Fr,
        backend: Option<&dyn OperatorBackend>,
    ) -> Result<Self> {
        Self::from_impl(m,n,offset,tau_x,tau_y,backend,None,None)
    }

    /// Only this constructor shares columns, with identical domains/trapdoors
    /// and n-aligned offsets. lambda=phi^n=1, so six columns are identical.
    pub(crate) fn setup_six(m:usize,n:usize,tau_x:Fr,tau_y:Fr,
        backend:Option<&dyn OperatorBackend>)->Result<[Self;6]> {
        ensure!(n.checked_mul(6).is_some_and(|end|end<=m),"m must fit six wire slices");
        let prepared=ConsistencyScalarPreparation::new(m,n,tau_x,tau_y)?;
        let mut keys=Vec::with_capacity(6);
        keys.push(Self::from_impl(m,n,0,tau_x,tau_y,backend,None,Some(&prepared))?);
        for t in 1..6 {keys.push(Self::from_impl(m,n,t*n,tau_x,tau_y,backend,Some(&keys[0]),Some(&prepared))?);}
        keys.try_into().map_err(|_|anyhow::anyhow!("expected six consistency keys"))
    }

    #[allow(clippy::too_many_arguments)]
    fn from_impl(m:usize,n:usize,offset:usize,tau_x:Fr,tau_y:Fr,
        backend:Option<&dyn OperatorBackend>,common:Option<&Self>,
        prepared:Option<&ConsistencyScalarPreparation>)->Result<Self> {
        ensure!(m > 0 && m.is_power_of_two(), "m must be a nonzero power of two");
        ensure!(n > 0 && n.is_power_of_two() && n <= m, "n must divide m");
        ensure!(m % n == 0, "n must divide m");
        ensure!(offset % n == 0, "slice offset must be n-aligned");
        ensure!(offset.checked_add(n).is_some_and(|end| end <= m), "slice exceeds m");
        let big_domain = radix2_domain(m)?;
        let ratio = m / n;
        let phi = big_domain.element(ratio);
        let lambda = phi.pow([(n - offset % n) as u64]);
        let y_r = tau_y.pow([ratio as u64]);
        let x_m = tau_x.pow([m as u64]);
        let y_m = tau_y.pow([m as u64]);
        ensure!(!tau_x.is_zero() && !tau_y.is_zero(), "trapdoors must be nonzero");
        ensure!(x_m != Fr::one() && y_m != Fr::one(), "trapdoors must be outside m-domain");
        let xy_inverse = (tau_x - tau_y)
            .inverse()
            .ok_or_else(|| anyhow::anyhow!("tau_x equals tau_y"))?;
        let substituted_inverse = (tau_x - lambda * y_r)
            .inverse()
            .ok_or_else(|| anyhow::anyhow!("consistency divisor vanishes"))?;
        let mut z_in = Fr::one();
        for i in offset..offset + n {
            z_in *= tau_x - big_domain.element(i);
        }
        let z_in_inverse = z_in
            .inverse()
            .ok_or_else(|| anyhow::anyhow!("tau_x lies on selected slice"))?;

        // Every accepted offset is n-aligned; phi has order n, hence lambda=1.
        // Borrow shared vectors for six slices; standalone setup owns a local
        // preparation with the same lifetime boundary.
        let local;
        let prepared=match prepared {
            Some(p)=>p,
            None=>{local=ConsistencyScalarPreparation::new(m,n,tau_x,tau_y)?; &local}
        };
        ensure!(prepared.m==m && prepared.n==n && prepared.tau_x==tau_x
            && prepared.tau_y==tau_y && lambda==Fr::one(),"incompatible consistency preparation");
        let l_big_x=&prepared.l_big_x; let l_small_x=&prepared.l_small_x;
        let c_x=&prepared.c_x; let c_y=&prepared.c_y;
        let to_g1 = |scalars: &[Fr]| crate::setup::g1(scalars, backend);

        let qac_scalars: Vec<Fr> = (0..m)
            .map(|i| {
                let mapped = if (offset..offset + n).contains(&i) {
                    c_x[i - offset]
                } else {
                    Fr::zero()
                };
                (l_big_x[i] - mapped) * z_in_inverse
            })
            .collect();
        let qac_bases = Arc::new(to_g1(&qac_scalars)?);
        let (qcd_bases,qbd_bases,d_bases,a_degree_bases,b_degree_bases,d_degree_bases)=
            if let Some(key)=common {
                (key.qcd_bases.clone(),key.qbd_bases.clone(),key.d_bases.clone(),
                    key.a_degree_bases.clone(),key.b_degree_bases.clone(),key.d_degree_bases.clone())
            } else {
                (Arc::new(to_g1(&(0..n).map(|j|(c_x[j]-c_y[j])*xy_inverse).collect::<Vec<_>>())?),
                 Arc::new(to_g1(&(0..n).map(|j|(l_small_x[j]-c_y[j])*substituted_inverse).collect::<Vec<_>>())?),
                 Arc::new(to_g1(&c_y)?),
                 Arc::new(to_g1(&l_big_x.iter().map(|&v|v*y_m).collect::<Vec<_>>())?),
                 Arc::new(to_g1(&l_small_x.iter().map(|&v|v*y_m).collect::<Vec<_>>())?),
                 Arc::new(to_g1(&c_y.iter().map(|&v|v*x_m).collect::<Vec<_>>())?))
            };

        let g2 = G2Projective::generator();
        let verifying = SliceConsistencyVerifyingKey {
            m,
            n,
            offset,
            z_in: (g2 * z_in).into_affine(),
            x_minus_y: (g2 * (tau_x - tau_y)).into_affine(),
            x_minus_lambda_y_r: (g2 * (tau_x - lambda * y_r)).into_affine(),
            x_m: (g2 * x_m).into_affine(),
            y_m: (g2 * y_m).into_affine(),
        };
        Ok(Self {
            m,
            n,
            offset,
            qac_bases,
            qcd_bases,
            qbd_bases,
            d_bases,
            a_degree_bases,
            b_degree_bases,
            d_degree_bases,
            verifying,
        })
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Linear consistency witnesses from the selected full/slice evaluations.
    /// Neither the proving bases nor the domain is scanned or cloned.
    pub fn prove_sparse(
        &self, full: &crate::dynamo::SparseVector, small: &crate::dynamo::SparseVector,
        backend: &dyn OperatorBackend,
    ) -> Result<SliceConsistencyProof> {
        ensure!(full.len() == self.m && small.len() == self.n, "sparse consistency domain mismatch");
        let expected: Vec<(usize, Fr)> = full.entries().iter()
            .filter(|&&(i, _)| (self.offset..self.offset + self.n).contains(&i))
            .map(|&(i, v)| (i - self.offset, v)).collect();
        ensure!(expected == small.entries(), "sparse slice differs from full vector");
        let f = full.entries();
        let s = small.entries();
        let points = crate::kzg::gather_msm_batch(&[
            (&self.qac_bases, f), (&self.qcd_bases, s), (&self.qbd_bases, s),
            (&self.d_bases, s), (&self.a_degree_bases, f),
            (&self.b_degree_bases, s), (&self.d_degree_bases, s),
        ], backend)?;
        Ok(SliceConsistencyProof { qac: points[0], qcd: points[1], qbd: points[2],
            d: points[3], a_degree: points[4], b_degree: points[5], d_degree: points[6] })
    }

    pub fn prove(
        &self,
        full: &[Fr],
        small: &[Fr],
        backend: &impl OperatorBackend,
    ) -> Result<SliceConsistencyProof> {
        ensure!(full.len() == self.m, "full vector length differs from m");
        ensure!(small.len() == self.n, "slice length differs from n");
        Ok(SliceConsistencyProof {
            qac: backend.msm(&self.qac_bases, full)?,
            qcd: backend.msm(&self.qcd_bases, small)?,
            qbd: backend.msm(&self.qbd_bases, small)?,
            d: backend.msm(&self.d_bases, small)?,
            a_degree: backend.msm(&self.a_degree_bases, full)?,
            b_degree: backend.msm(&self.b_degree_bases, small)?,
            d_degree: backend.msm(&self.d_degree_bases, small)?,
        })
    }
}

fn lagrange_values(tau: Fr, domain: &Radix2EvaluationDomain<Fr>) -> Result<Vec<Fr>> {
    crate::setup::lagrange_values(tau, domain)
}

#[cfg(test)]
mod setup_sharing_tests {
    use super::*;
    #[test]
    fn six_slices_prepare_four_vectors_once_and_match_standalone_keys() {
        for n in [1,8] {
            let m=8*n; let x=Fr::from(17011u64); let y=Fr::from(17012u64);
            crate::setup::LAGRANGE_LENGTHS.with(|v|v.borrow_mut().clear());
            let shared=SliceConsistencyKey::setup_six(m,n,x,y,None).unwrap();
            let sizes=crate::setup::LAGRANGE_LENGTHS.with(|v|v.borrow().clone());
            assert_eq!(sizes,vec![m,n,n,n],"duplicated trapdoor-derived Lagrange vectors");
            for t in 0..6 {
                let reference=SliceConsistencyKey::from_trapdoors(m,n,t*n,x,y,None).unwrap();
                assert_eq!(shared[t].rebuild_bases(),reference.rebuild_bases());
                let a=&shared[t].verifying; let b=&reference.verifying;
                assert_eq!((a.z_in,a.x_minus_y,a.x_minus_lambda_y_r,a.x_m,a.y_m),
                    (b.z_in,b.x_minus_y,b.x_minus_lambda_y_r,b.x_m,b.y_m));
            }
            let changed=SliceConsistencyKey::setup_six(m,n,x+Fr::from(2u64),y,None).unwrap();
            assert_ne!(changed[0].rebuild_bases()[0],shared[0].rebuild_bases()[0]);
        }
    }
}
