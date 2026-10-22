//! Fixed-anchor sparse proof. All temporary polynomials have degree O(a),
//! and all gathered commitment inputs have length O(d+a).
use super::*;
use crate::domain::{newton_divide_exact, SubproductTree};
use crate::kzg::gather_msm_batch;
use ark_ff::{Field, batch_inversion};
use std::collections::BTreeSet;
use std::sync::Arc;
use super::sparse_geometry::SparseGeometry;

/// Existing Dynamo helpers are generic over a sized backend. This adapter
/// keeps dispatch borrowed and instruments their calls before any GPU work.
pub(super) struct CountedBackend<'a>(pub(super) &'a dyn OperatorBackend);
impl OperatorBackend for CountedBackend<'_> {
    fn msm(&self, points: &[G1Affine], scalars: &[Fr]) -> Result<G1Affine> {
        crate::workcount::record_msm(scalars.len());
        self.0.msm(points, scalars)
    }
    fn msm_batch(&self, request: &BatchMsmRequest<'_>) -> Result<Vec<G1Affine>> {
        for &len in request.lengths { crate::workcount::record_msm(len); }
        self.0.msm_batch(request)
    }
    fn msm_sparse_batch(&self, requests: &[crate::SparseMsmRequest<'_>]) -> Result<Vec<G1Affine>> {
        for request in requests { crate::workcount::record_msm(request.entries.len()); }
        self.0.msm_sparse_batch(requests)
    }
    fn ntt(&self, values: &mut [Fr], inverse: bool) -> Result<()> {
        crate::workcount::record_ntt(values.len());
        self.0.ntt(values, inverse)
    }
    fn msm_g2_host(&self,points:&[G2Affine],scalars:&[Fr])->Result<G2Affine> {
        self.0.msm_g2_host(points,scalars)
    }
    fn gpu_capabilities(&self)->crate::GpuCapabilities {self.0.gpu_capabilities()}
}

impl DynaverseParams {
    pub(super) fn prove_delta_sparse(
        &self, anchor: &Witness, entries: &[(usize, Fr)], anchor_wires: &WireCommitments,
        cache: &KzgPrecomputation, backend: &dyn OperatorBackend,
    ) -> Result<DeltaProof> {
        self.prove_delta_sparse_with_field_backend(anchor,entries,anchor_wires,cache,backend,None)
    }

    /// Stage-5 opts into explicit small Fr/G2 dispatch. The retained Stage-4
    /// wrapper above preserves its CPU arithmetic and work-count baseline.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prove_delta_sparse_with_field_backend(
        &self,anchor:&Witness,entries:&[(usize,Fr)],anchor_wires:&WireCommitments,
        cache:&KzgPrecomputation,backend:&dyn OperatorBackend,
        field_backend:Option<&dyn OperatorBackend>,
    )->Result<DeltaProof> {
        self.prove_delta_sparse_with_linear(anchor,entries,anchor_wires,cache,backend,field_backend,None)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn prove_delta_sparse_with_linear(
        &self,anchor:&Witness,entries:&[(usize,Fr)],anchor_wires:&WireCommitments,
        cache:&KzgPrecomputation,backend:&dyn OperatorBackend,
        field_backend:Option<&dyn OperatorBackend>,fixed:Option<&super::sparse_linear::FixedLinear>,
    )->Result<DeltaProof> {
        Ok(self.prove_delta_sparse_cached(anchor,entries,anchor_wires,cache,backend,field_backend,
            fixed,None,0,false)?.0)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn prove_delta_sparse_cached(
        &self,anchor:&Witness,entries:&[(usize,Fr)],anchor_wires:&WireCommitments,
        cache:&KzgPrecomputation,backend:&dyn OperatorBackend,
        field_backend:Option<&dyn OperatorBackend>,fixed:Option<&super::sparse_linear::FixedLinear>,
        previous:Option<&Arc<SparseGeometry>>,anchor_generation:u64,reuse_geometry:bool,
    )->Result<(DeltaProof,Option<Arc<SparseGeometry>>,bool)> {
        let mut trace=crate::incremental_rebuild::SparseProofTimings::default();
        self.prove_delta_sparse_traced(anchor,entries,anchor_wires,cache,backend,field_backend,
            fixed,previous,anchor_generation,reuse_geometry,&mut trace)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn prove_delta_sparse_traced(
        &self,anchor:&Witness,entries:&[(usize,Fr)],anchor_wires:&WireCommitments,
        cache:&KzgPrecomputation,backend:&dyn OperatorBackend,
        field_backend:Option<&dyn OperatorBackend>,fixed:Option<&super::sparse_linear::FixedLinear>,
        previous:Option<&Arc<SparseGeometry>>,anchor_generation:u64,reuse_geometry:bool,
        trace:&mut crate::incremental_rebuild::SparseProofTimings,
    )->Result<(DeltaProof,Option<Arc<SparseGeometry>>,bool)> {
        let started=trace.start();
        let n = self.circuit.n();
        let m = self.circuit.m();
        ensure!(anchor.wires.len() == 6 * n, "anchor witness length differs from circuit");
        ensure!(cache.q4.len() == n && cache.q5.len() == n && cache.q6.len() == n,
            "sparse updates require a complete precomputed anchor opening cache");
        ensure!(entries.iter().all(|&(i, _)| i < 6 * n), "sparse delta includes a public root");
        let changed = SparseVector::try_new(m, entries.to_vec())?;
        let fixed_fields=fixed.map(|state|state.fields());
        let dynamo=if let Some(f)=&fixed_fields {f.dynamo.clone()} else {
            let h=derive_h_sparse(&changed,self.circuit.permutation_ref())?;
            dynamo_prove_unmasked(&self.srs.dynamo,&changed,&h,&CountedBackend(backend))?
        };
        let mut slices: [Vec<(usize, Fr)>; 6] = std::array::from_fn(|_| Vec::new());
        let mut affected = BTreeSet::new();
        crate::workcount::record_scan(entries.len());
        for &(index, value) in entries {
            slices[index / n].push((index % n, value));
            if index >= 3 * n { affected.insert(index % n); }
        }
        let basis = &self.srs.sparse;
        let wire_requests: Vec<(&[G1Affine], &[(usize, Fr)])> = slices.iter()
            .map(|slice| (basis.small_lagrange_g1.as_slice(), slice.as_slice())).collect();
        let (changed_wires,consistency,public_zero_quotient)=if let Some(f)=fixed_fields {
            (f.wires,f.consistency,f.public_zero_quotient)
        } else {
            let wires=gather_msm_batch(&wire_requests,backend)?;
            let changed_wires=WireCommitments {s1:wires[0],s2:wires[1],s3:wires[2],s4:wires[3],s5:wires[4],s6:wires[5]};
            let mut consistency=Vec::with_capacity(6);
            for (i,slice) in slices.iter().enumerate() {
                let small=SparseVector::try_new(n,slice.clone())?;
                consistency.push(self.srs.consistency[i].prove_sparse(&changed,&small,backend)?);
            }
            let consistency=ConsistencyProof {slices:consistency.try_into().map_err(|_|anyhow::anyhow!("expected six consistency proofs"))?};
            let public=gather_msm_batch(&[(&basis.public_quotient_g1,entries)],backend)?[0];
            (changed_wires,consistency,public)
        };
        trace.finish(0,started);
        let started=trace.start();
        let affected: Vec<usize> = affected.into_iter().collect();
        let cached=previous.filter(|g|reuse_geometry && g.anchor_generation==anchor_generation && g.affected==affected);
        let points:Vec<Fr>=if let Some(g)=cached {g.points.clone()}
            else {affected.iter().map(|&i|basis.small_points[i]).collect()};
        let tree=if let Some(g)=cached {g.tree.clone()} else {Arc::new(match field_backend {
            Some(selected)=>SubproductTree::new_with_backend(&points,selected)?,
            None=>SubproductTree::new(&points)?,
        })};
        let weights=tree.inverse_derivatives();
        let ai=tree.vanishing();
        trace.finish(1,started);
        let started=trace.start();
        let affected_divisor_g1=if let Some(g)=cached {g.affected_g1}
            else {self.srs.kzg_x.commit_with_backend(ai,backend)?.point};
        trace.finish(4,started);
        let started=trace.start();
        let affected_divisor_g2=if let Some(g)=cached {g.affected_g2} else {match field_backend {
            Some(selected)=>self.srs.kzg_x.commit_g2_with_backend(ai,selected)?,
            None=>self.srs.kzg_x.commit_g2(ai)?,
        }};
        trace.finish(5,started);
        let started=trace.start();
        let mut inverses=if let Some(g)=cached {g.inverses.clone()} else {points.clone()};
        if cached.is_none() {batch_inversion(&mut inverses);}
        // Z'(x_i)=n/x_i. B=Z/A=sum Z'(x_i)/A'(x_i) L_i.
        let b_entries:Vec<(usize,Fr)>=affected.iter().enumerate()
            .map(|(j,&i)|(i,Fr::from(n as u64)*inverses[j]*weights[j])).collect();
        trace.finish(1,started);
        let (unaffected_divisor_g1,unaffected_divisor_g2)=if let Some(g)=cached {
            (g.unaffected_g1,g.unaffected_g2)
        } else if affected.is_empty() {
            (self.verifying.multiplication_vanishing_g1,self.verifying.multiplication_divisor)
        } else {
            let started=trace.start();
            let b1=gather_msm_batch(&[(&basis.small_lagrange_g1,&b_entries)],backend)?[0];
            trace.finish(4,started);
            let started=trace.start();
            let g2_points:Vec<G2Affine>=affected.iter().map(|&i|basis.small_lagrange_g2[i]).collect();
            let scalars:Vec<Fr>=b_entries.iter().map(|&(_,v)|v).collect();
            crate::workcount::record_g2_msm(scalars.len());
            let b2=match field_backend {
                Some(selected)=>selected.msm_g2_host(&g2_points,&scalars)?,
                None=>G2Projective::msm_unchecked(&g2_points,&scalars).into_affine(),
            };
            trace.finish(5,started);
            (b1,b2)
        };
        let started=trace.start();
        let cauchy_weights=if let Some(g)=cached {g.cauchy_weights.clone()} else {cauchy(&tree,weights,field_backend)?};
        trace.finish(2,started);
        let mut anchor_local: [Vec<Fr>; 3] = std::array::from_fn(|_| Vec::new());
        let mut changed_local: [Vec<Fr>; 3] = std::array::from_fn(|_| Vec::new());
        let mut changed_openings = Vec::with_capacity(3);
        let mut zero_quotients: [Vec<Fr>; 3] = std::array::from_fn(|_| Vec::new());
        for t in 0..3 {
            let started=trace.start();
            let mut anchor_values = Vec::with_capacity(affected.len());
            let mut values = Vec::with_capacity(affected.len());
            for &i in &affected {
                anchor_values.push(anchor.wires[(t + 3) * n + i]);
                let value = slices[t + 3].binary_search_by_key(&i, |&(index, _)| index)
                    .map_or(Fr::zero(), |j| slices[t + 3][j].1);
                values.push(value);
            }
            anchor_local[t]=if let Some(g)=cached {g.anchor_local[t].clone()}
                else {interpolate(&tree,&anchor_values,field_backend)?};
            if reuse_geometry && values.iter().all(Zero::is_zero) {
                // Exact zero polynomial/quotient/opening, regardless of the
                // nonempty affected set contributed by other wires.
                changed_local[t]=vec![Fr::zero()];zero_quotients[t]=vec![Fr::zero()];
                changed_openings.push(G1Affine::identity());trace.finish(2,started);continue;
            }
            changed_local[t] = interpolate(&tree,&values,field_backend)?;
            let weighted_x: Vec<Fr> = values.iter().zip(&points).map(|(&v, &x)| v * x).collect();
            let cauchy_values = cauchy(&tree,&weighted_x,field_backend)?;
            let mut lagrange_entries = Vec::with_capacity(affected.len());
            let mut opening_entries = Vec::with_capacity(affected.len());
            let mut zero_values = Vec::with_capacity(affected.len());
            let inv_n = Fr::from(n as u64).inverse().unwrap();
            for (j, &i) in affected.iter().enumerate() {
                // For i != j, L_j/(X-x_i)=(L_j-(x_j/x_i)L_i)/(x_j-x_i).
                // Aggregate all single openings with 1/A'(x_i); the two
                // diagonal-excluding Cauchy sums give the L_i coefficients.
                lagrange_entries.push((i, values[j] * cauchy_weights[j]
                    + weights[j] * inverses[j] * cauchy_values[j]));
                opening_entries.push((i, values[j] * weights[j]));
                // f/B has degree < a and value f(x_i) A'(x_i)/Z'(x_i).
                zero_values.push(values[j] / weights[j] * points[j] * inv_n);
            }
            trace.finish(2,started);
            let started=trace.start();
            let q = gather_msm_batch(&[(&basis.small_lagrange_g1, &lagrange_entries),
                (&basis.opening_u_g1, &opening_entries)], backend)?;
            changed_openings.push((G1Projective::from(q[0]) + q[1]).into_affine());
            trace.finish(4,started);
            let started=trace.start();
            zero_quotients[t] = interpolate(&tree,&zero_values,field_backend)?;
            trace.finish(2,started);
        }
        let started=trace.start();
        let mut updated = [Vec::new(), Vec::new(), Vec::new()];
        for t in 0..3 { updated[t] = add_polynomials(&anchor_local[t], &changed_local[t]); }
        let product=match field_backend {
            Some(selected)=>crate::domain::fft_multiply_with_backend(&updated[0],&updated[1],selected)?,
            None=>fft_multiply(&updated[0],&updated[1])?,
        };
        let local_numerator = subtract_polynomials(&product,&updated[2]);
        let local_quotient = match field_backend {
            Some(selected)=>crate::domain::newton_divide_exact_with_backend(&local_numerator,ai,selected)?,
            None=>newton_divide_exact(&local_numerator,ai)?,
        };
        trace.finish(3,started);
        let started=trace.start();
        let anchor_openings=if let Some(g)=cached {g.anchor_openings} else if affected.is_empty() {
            MultiplicationWireCommitments { s4: anchor_wires.s4, s5: anchor_wires.s5, s6: anchor_wires.s6 }
        } else {
            let selected: Vec<(usize, Fr)> = affected.iter().copied().zip(weights.iter().copied()).collect();
            let q = gather_msm_batch(&[(&cache.q4, &selected), (&cache.q5, &selected), (&cache.q6, &selected)], backend)?;
            MultiplicationWireCommitments { s4: q[0], s5: q[1], s6: q[2] }
        };
        let anchor_local_commitments=if let Some(g)=cached {g.anchor_local_commitments}
            else {self.commit_triple(&anchor_local,Some(backend))?};
        let changed_local_commitments=self.commit_triple(&changed_local,Some(backend))?;
        let zero_commitments=self.commit_triple(&zero_quotients,Some(backend))?;
        let quotient_commitment=self.srs.kzg_x.commit_with_backend(&local_quotient,backend)?.point;
        trace.finish(4,started);
        let started=trace.start();
        let updated_s5_g2=match field_backend {
            Some(selected)=>self.srs.kzg_x.commit_g2_with_backend(&updated[1],selected)?,
            None=>self.srs.kzg_x.commit_g2(&updated[1])?,
        };
        trace.finish(5,started);
        let proof=DeltaProof {
            changed_wires, changed_w_commitment: dynamo.instance_z,
            changed_h_commitment: dynamo.instance_h, dynamo, consistency, public_zero_quotient,
            affected_divisor_g1, affected_divisor_g2, unaffected_divisor_g1, unaffected_divisor_g2,
            anchor_local: anchor_local_commitments,
            changed_local:changed_local_commitments,updated_s5_g2,anchor_openings,
            changed_openings: MultiplicationWireCommitments { s4: changed_openings[0],
                s5: changed_openings[1], s6: changed_openings[2] },
            unaffected_zero_quotients:zero_commitments,
            local_multiplication_quotient:quotient_commitment,
        };
        let hit=cached.is_some();
        let next=if !reuse_geometry {None} else if let Some(g)=cached {Some(g.clone())} else {
            Some(Arc::new(SparseGeometry {anchor_generation,affected,points,tree,inverses,cauchy_weights,
                affected_g1:affected_divisor_g1,affected_g2:affected_divisor_g2,
                unaffected_g1:unaffected_divisor_g1,unaffected_g2:unaffected_divisor_g2,
                anchor_local,anchor_local_commitments,anchor_openings}))
        };
        Ok((proof,next,hit))
    }
}

fn interpolate(tree:&SubproductTree,values:&[Fr],backend:Option<&dyn OperatorBackend>)->Result<Vec<Fr>> {
    match backend {
        Some(selected)=>tree.interpolate_with_backend(values,selected),
        None=>tree.interpolate(values),
    }
}

fn cauchy(tree:&SubproductTree,values:&[Fr],backend:Option<&dyn OperatorBackend>)->Result<Vec<Fr>> {
    match backend {
        Some(selected)=>tree.cauchy_sums_with_backend(values,selected),
        None=>tree.cauchy_sums(values),
    }
}
