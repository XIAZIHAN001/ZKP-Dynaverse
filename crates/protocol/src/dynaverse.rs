//! CPU Fig. 4 initial relation proof using one shared, unmasked SRS.

use crate::consistency::ConsistencyVerifyingKey;
use crate::domain::{
    affected_set_factorization, divide_exact, divide_exact_by_xn_minus_one, fft_multiply,
    interpolate_at_indices, multiply_polynomials,
    radix2_domain, subtract_polynomials, vanishing_polynomial, PreparedExactDivision,
};
use crate::dynamo::{
    derive_h_sparse, dynamo_prove_unmasked, dynamo_verify_with_key, dynamo_verify_with_checks, DynamoVerifyingKey,
    SparseVector,
};
use crate::srs::SharedSrs;
use crate::kzg::combine_single_openings;
use crate::toy_circuit::ToyCircuit;
use crate::trace::{BatchOpenPhase, CpuPhase, CpuTrace, VerificationPhase, VerificationTrace};
use crate::verification::PairingChecks;
use crate::types::{
    AnchorOpeningPolicy, AnchorProof, BatchMsmRequest, BatchOpenMode, ConsistencyProof, DeltaProof, DynaverseApi, DynaverseProof, Instance,
    KzgPrecomputation, MultiplicationWireCommitments, OperatorBackend, ProverAux,
    SchedulerState, Update, VerificationMode, WireCommitments, Witness,
};
use anyhow::{ensure, Result};
use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{pairing::Pairing, CurveGroup, Group, VariableBaseMSM};
use ark_ff::{One, Zero};
use ark_poly::EvaluationDomain;
use ark_std::rand::Rng;

mod sparse;
mod sparse_linear;
mod sparse_geometry;
mod session;
mod rebuilding;
pub use session::UpdateSession;
pub use rebuilding::IncrementalSession;

#[cfg(test)]
#[path = "dynaverse/sparse_tests.rs"]
mod sparse_tests;

#[derive(Clone, Debug)]
pub struct DynaverseParams {
    circuit: ToyCircuit,
    srs: SharedSrs,
    verifying: DynaverseVerifyingKey,
    // Fixed to this circuit's public root set; immutable and prover-owned.
    public_division: PreparedExactDivision,
}

// Owned intermediates cross the timing boundary without retaining a borrow
// of the trace or making a partly constructed public proof.
struct DeltaBatchRelations {
    public_zero_quotient: G1Affine,
    affected: Vec<usize>,
    ai: Vec<Fr>,
    bi: Vec<Fr>,
    affected_divisor_g1: G1Affine,
    affected_divisor_g2: G2Affine,
    unaffected_divisor_g1: G1Affine,
    unaffected_divisor_g2: G2Affine,
    anchor_local_coefficients: [Vec<Fr>; 3],
    changed_local_coefficients: [Vec<Fr>; 3],
    local_multiplication_quotient: G1Affine,
    updated_s5_g2: G2Affine,
}

/// All material needed to check the initial proof, with no witness or
/// prover-side commitment bases.
#[derive(Clone, Debug)]
pub struct DynaverseVerifyingKey {
    n: usize,
    m: usize,
    dynamo: DynamoVerifyingKey,
    consistency: ConsistencyVerifyingKey,
    public_h_bases: Vec<G1Affine>,
    public_zero_divisor: G2Affine,
    multiplication_vanishing_g1: G1Affine,
    multiplication_divisor: G2Affine,
}

impl DynaverseParams {
    pub fn setup<R: Rng + ?Sized>(circuit: ToyCircuit, rng: &mut R) -> Result<Self> {
        Self::setup_traced(circuit, rng, &mut CpuTrace::disabled())
    }

    pub fn setup_traced<R: Rng + ?Sized>(circuit: ToyCircuit, rng: &mut R, trace: &mut CpuTrace) -> Result<Self> {
        Self::setup_impl(circuit, rng, trace, None)
    }

    /// Generate prover SRS arrays through the selected setup backend. Small
    /// verifying-key constants and public polynomial preparation remain CPU.
    pub fn setup_with_backend<R: Rng + ?Sized>(circuit: ToyCircuit, rng: &mut R,
        backend: &dyn OperatorBackend) -> Result<Self> {
        Self::setup_impl(circuit, rng, &mut CpuTrace::disabled(), Some(backend))
    }

    /// Exclusive setup phase timings include backend work and host validation.
    pub fn setup_with_backend_traced<R:Rng+?Sized>(circuit:ToyCircuit,rng:&mut R,
        backend:&dyn OperatorBackend,trace:&mut CpuTrace)->Result<Self> {
        Self::setup_impl(circuit,rng,trace,Some(backend))
    }

    fn setup_impl<R: Rng + ?Sized>(circuit: ToyCircuit, rng: &mut R,
        trace: &mut CpuTrace, backend: Option<&dyn OperatorBackend>) -> Result<Self> {
        let n = circuit.n();
        let m = circuit.m();
        let sigma = circuit.permutation();
        let srs = SharedSrs::setup_with_backend_traced(m,n,sigma.clone(),rng,backend,trace)?;
        let started=trace.start();
        let big = radix2_domain(m)?;
        let public_indices: Vec<usize> = (6 * n..m).collect();
        let ix = crate::domain::vanishing_polynomial_balanced(&big, &public_indices)?;
        let public_zero_divisor = srs.kzg_x.commit_g2(&ix)?;
        trace.finish(CpuPhase::SetupPublic,started);
        let started = trace.start();
        let public_division = PreparedExactDivision::new(&ix, m);
        trace.finish(CpuPhase::PublicDivisionPrecompute, started);
        let public_division = public_division?;
        let started=trace.start();
        let mut multiplication_vanishing = vec![Fr::zero(); n + 1];
        multiplication_vanishing[0] = -Fr::one();
        multiplication_vanishing[n] = Fr::one();
        let multiplication_divisor = srs.kzg_x.commit_g2(&multiplication_vanishing)?;
        let multiplication_vanishing_g1 = srs.kzg_x.commit(&multiplication_vanishing)?.point;
        let public_h_bases = srs.dynamo.public_h_bases(&sigma, 6*n..7*n)?;
        let verifying = DynaverseVerifyingKey {
            n,
            m,
            dynamo: srs.dynamo.verifying_key(),
            consistency: srs.consistency_verifying_key(),
            public_h_bases,
            public_zero_divisor,
            multiplication_vanishing_g1,
            multiplication_divisor,
        };
        trace.finish(CpuPhase::SetupPublic,started);
        Ok(Self { circuit, srs, verifying, public_division })
    }

    pub fn verifying_key(&self) -> DynaverseVerifyingKey {
        self.verifying.clone()
    }

    pub fn prove_anchor(&self, instance: &Instance, witness: &Witness) -> Result<DynaverseProof> {
        self.prove_anchor_traced(instance, witness, None, &mut CpuTrace::disabled())
    }

    fn prove_anchor_traced(
        &self, instance: &Instance, witness: &Witness,
        kzg_backend: Option<&dyn OperatorBackend>, trace: &mut CpuTrace,
    ) -> Result<DynaverseProof> {
        let started = trace.start();
        let n = self.circuit.n();
        let m = self.circuit.m();
        ensure!(instance.public_inputs.len() == n, "public input length differs from circuit");
        ensure!(witness.wires.len() == 6 * n, "witness length differs from circuit");
        trace.finish(CpuPhase::Validation, started);
        let backend = ArkReferenceBackend;
        let mut w = witness.wires.clone();
        w.resize(m, Fr::zero());
        let started = trace.start();
        let w_coefficients = self.interpolate(&w, kzg_backend)?;
        trace.finish(CpuPhase::Interpolation, started);
        let started = trace.start();
        let w_commitment = self.commit_g1(&w_coefficients, kzg_backend)?;
        trace.finish(CpuPhase::KzgCommit, started);
        let started = trace.start();
        let w_sparse = sparse_from_dense(&w)?;
        let h_sparse = derive_h_sparse(&w_sparse, &self.circuit.permutation())?;
        let dynamo = dynamo_prove_unmasked(&self.srs.dynamo, &w_sparse, &h_sparse, &backend)?;
        let h_commitment = dynamo.instance_h;
        ensure!(dynamo.instance_z == w_commitment, "shared X commitment mismatch");
        trace.finish(CpuPhase::DynamoMsm, started);

        let started = trace.start();
        let mut wire_coefficients: [Vec<Fr>; 6] = std::array::from_fn(|_| Vec::new());
        for i in 0..6 {
            wire_coefficients[i] = self.interpolate(&w[i * n..(i + 1) * n], kzg_backend)?;
        }
        trace.finish(CpuPhase::Interpolation, started);
        let started = trace.start();
        let wire_points: [G1Affine; 6] = self.commit_many_g1(&wire_coefficients, kzg_backend)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("expected six wire commitments"))?;
        let wires = WireCommitments {
            s1: wire_points[0],
            s2: wire_points[1],
            s3: wire_points[2],
            s4: wire_points[3],
            s5: wire_points[4],
            s6: wire_points[5],
        };
        let s5_g2 = self.srs.kzg_x.commit_g2(&wire_coefficients[4])?;
        trace.finish(CpuPhase::KzgCommit, started);
        let started = trace.start();
        let slices = (0..6)
            .map(|i| self.srs.consistency[i].prove(&w, &w[i * n..(i + 1) * n], &backend))
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| anyhow::anyhow!("expected six consistency proofs"))?;
        let consistency = ConsistencyProof { slices };
        trace.finish(CpuPhase::Consistency, started);

        let started = trace.start();
        // Setup prepared this circuit's fixed public divisor. Reuse its
        // reciprocal while still checking every coefficient for exact division.
        let qx = self.public_division.divide_exact(&w_coefficients)?;
        let public_zero_quotient = self.commit_g1(&qx, kzg_backend)?;

        let product = fft_multiply(&wire_coefficients[3], &wire_coefficients[4])?;
        let numerator = subtract_polynomials(&product, &wire_coefficients[5]);
        let quotient = divide_exact_by_xn_minus_one(&numerator, n)?;
        let multiplication_quotient = self.commit_g1(&quotient, kzg_backend)?;
        trace.finish(CpuPhase::BatchOpen, started);

        Ok(DynaverseProof {
            anchor: AnchorProof {
                wires,
                s5_g2,
                w_commitment,
                h_commitment,
                dynamo,
                consistency,
                public_zero_quotient,
                multiplication_quotient,
            },
            delta: None,
            anchor_generation: 0,
            generation: 0,
        })
    }

    pub fn prove_with_aux(
        &self,
        instance: &Instance,
        witness: &Witness,
    ) -> Result<(DynaverseProof, ProverAux)> {
        self.prove_with_aux_policy(instance, witness, AnchorOpeningPolicy::Precompute)
    }

    pub fn prove_with_aux_policy(
        &self, instance: &Instance, witness: &Witness, policy: AnchorOpeningPolicy,
    ) -> Result<(DynaverseProof, ProverAux)> {
        self.prove_with_aux_traced(instance, witness, policy, &mut CpuTrace::disabled())
    }

    /// Accumulates phase durations in the caller's recorder. Use a fresh
    /// recorder when measuring one operation; the stable API disables timing.
    pub fn prove_with_aux_traced(
        &self, instance: &Instance, witness: &Witness,
        policy: AnchorOpeningPolicy, trace: &mut CpuTrace,
    ) -> Result<(DynaverseProof, ProverAux)> {
        self.prove_with_aux_backend(instance, witness, policy, None, trace)
    }

    /// Select a backend for prover Fr interpolation and KZG G1 commitments.
    /// The borrow lasts only for this operation; EC opening caches remain CPU.
    pub fn prove_with_aux_traced_with_backend(
        &self, instance: &Instance, witness: &Witness, policy: AnchorOpeningPolicy,
        backend: &dyn OperatorBackend, trace: &mut CpuTrace,
    ) -> Result<(DynaverseProof, ProverAux)> {
        self.prove_with_aux_backend(instance, witness, policy, Some(backend), trace)
    }

    fn prove_with_aux_backend(
        &self, instance: &Instance, witness: &Witness, policy: AnchorOpeningPolicy,
        kzg_backend: Option<&dyn OperatorBackend>, trace: &mut CpuTrace,
    ) -> Result<(DynaverseProof, ProverAux)> {
        let started = trace.start();
        self.circuit.validate(instance, witness)?;
        trace.finish(CpuPhase::Validation, started);
        let proof = self.prove_anchor_traced(instance, witness, kzg_backend, trace)?;
        let kzg = self.prepare_opening_cache(witness, policy, kzg_backend, trace)?;
        let rebuild_interval = ceil_sqrt(self.circuit.n()) as u64;
        let aux = ProverAux {
            anchor_witness: witness.clone(),
            current_witness: witness.clone(),
            current_instance: instance.clone(),
            current_proof: proof.clone(),
            kzg,
            opening_policy: policy,
            scheduler: SchedulerState {
                generation: 0,
                updates_since_rebuild: 0,
                rebuild_interval,
            },
        };
        Ok((proof, aux))
    }

    fn prepare_opening_cache(
        &self, witness: &Witness, policy: AnchorOpeningPolicy,
        backend: Option<&dyn OperatorBackend>, trace: &mut CpuTrace,
    ) -> Result<KzgPrecomputation> {
        if policy == AnchorOpeningPolicy::Dense {
            return Ok(KzgPrecomputation { q4: Vec::new(), q5: Vec::new(), q6: Vec::new() });
        }
        let started = trace.start();
        let cache = self.precompute_anchor_openings(witness, backend)?;
        trace.finish(CpuPhase::CachePrecompute, started);
        Ok(cache)
    }

    fn precompute_anchor_openings(&self, witness: &Witness, backend: Option<&dyn OperatorBackend>) -> Result<KzgPrecomputation> {
        let n = self.circuit.n();
        let small = radix2_domain(n)?;
        let openings = (3..6).map(|t| {
            let coefficients = self.interpolate(&witness.wires[t * n..(t + 1) * n], backend)?;
            self.srs.kzg_x.precompute_single_openings(&coefficients, &small)
        }).collect::<Result<Vec<_>>>()?;
        let [q4, q5, q6]: [Vec<G1Affine>; 3] = openings.try_into()
            .map_err(|_| anyhow::anyhow!("expected three opening caches"))?;
        Ok(KzgPrecomputation { q4, q5, q6 })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update_proof(
        &self,
        old_instance: &Instance,
        new_instance: &Instance,
        old_witness: &Witness,
        update: &Update,
        proof: &DynaverseProof,
        aux: &mut ProverAux,
    ) -> Result<DynaverseProof> {
        self.update_proof_traced(old_instance, new_instance, old_witness, update,
            proof, aux, &mut CpuTrace::disabled())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update_proof_traced(
        &self,
        old_instance: &Instance, new_instance: &Instance, old_witness: &Witness,
        update: &Update, proof: &DynaverseProof, aux: &mut ProverAux,
        trace: &mut CpuTrace,
    ) -> Result<DynaverseProof> {
        self.update_proof_traced_with_verifier(old_instance, new_instance, old_witness,
            update, proof, aux, VerificationMode::Product, trace)
    }

    /// Select the candidate verifier for CPU comparisons. State is committed
    /// only after the selected verifier accepts and replacement caches succeed.
    #[allow(clippy::too_many_arguments)]
    pub fn update_proof_traced_with_verifier(
        &self, old_instance: &Instance, new_instance: &Instance, old_witness: &Witness,
        update: &Update, proof: &DynaverseProof, aux: &mut ProverAux,
        mode: VerificationMode, trace: &mut CpuTrace,
    ) -> Result<DynaverseProof> {
        self.update_proof_traced_with_modes(old_instance, new_instance, old_witness,
            update, proof, aux, mode, BatchOpenMode::Prepared, trace)
    }

    /// Explicit CPU arithmetic modes for comparisons using identical inputs.
    #[allow(clippy::too_many_arguments)]
    pub fn update_proof_traced_with_modes(
        &self, old_instance: &Instance, new_instance: &Instance, old_witness: &Witness,
        update: &Update, proof: &DynaverseProof, aux: &mut ProverAux,
        mode: VerificationMode, batch_mode: BatchOpenMode, trace: &mut CpuTrace,
    ) -> Result<DynaverseProof> {
        self.update_proof_backend(old_instance, new_instance, old_witness, update,
            proof, aux, mode, batch_mode, None, trace)
    }

    /// Use the selected G1 backend for both delta commitments and reanchoring.
    /// Errors propagate before replacement aux state is committed.
    #[allow(clippy::too_many_arguments)]
    pub fn update_proof_traced_with_backend(
        &self, old_instance: &Instance, new_instance: &Instance, old_witness: &Witness,
        update: &Update, proof: &DynaverseProof, aux: &mut ProverAux,
        mode: VerificationMode, batch_mode: BatchOpenMode,
        backend: &dyn OperatorBackend, trace: &mut CpuTrace,
    ) -> Result<DynaverseProof> {
        self.update_proof_backend(old_instance, new_instance, old_witness, update,
            proof, aux, mode, batch_mode, Some(backend), trace)
    }

    #[allow(clippy::too_many_arguments)]
    fn update_proof_backend(
        &self, old_instance: &Instance, new_instance: &Instance, old_witness: &Witness,
        update: &Update, proof: &DynaverseProof, aux: &mut ProverAux,
        mode: VerificationMode, batch_mode: BatchOpenMode,
        kzg_backend: Option<&dyn OperatorBackend>, trace: &mut CpuTrace,
    ) -> Result<DynaverseProof> {
        let started = trace.start();
        ensure!(
            old_instance == &aux.current_instance
                && old_witness == &aux.current_witness
                && proof == &aux.current_proof
                && proof.generation == aux.scheduler.generation,
            "stale update state"
        );
        ensure!(
            aux.anchor_witness.wires.len() == 6 * self.circuit.n(),
            "anchor snapshot length differs from circuit"
        );
        // Comparing old_witness and the instance with the stored state reads
        // every wire and public input.
        crate::workcount::record_scan(old_witness.wires.len() + old_instance.public_inputs.len());
        crate::workcount::record_scan(self.circuit.m());
        let mut anchor_padded = aux.anchor_witness.wires.clone();
        anchor_padded.resize(self.circuit.m(), Fr::zero());
        ensure!(
            self.srs.dynamo.commit_z_dense_unmasked(&anchor_padded)?
                == proof.anchor.w_commitment,
            "anchor snapshot differs from proof"
        );
        trace.finish(CpuPhase::StateCheck, started);
        let started = trace.start();
        let normalized = update.clone().canonicalize(old_witness.wires.len())?;
        crate::workcount::record_scan(old_witness.wires.len());
        let mut new_witness = old_witness.clone();
        for (&index, &value) in normalized.indices.iter().zip(&normalized.values) {
            new_witness.wires[index] = value;
        }
        self.circuit.validate(new_instance, &new_witness)?;
        // The no-op comparison and the anchor distance each read all wires.
        crate::workcount::record_scan(2 * new_witness.wires.len());
        if new_witness == *old_witness && new_instance == old_instance {
            trace.finish(CpuPhase::Validation, started);
            return Ok(proof.clone());
        }
        let true_delta = new_witness.wires.iter().zip(&aux.anchor_witness.wires)
            .filter(|(new, anchor)| new != anchor).count();
        let generation = proof.generation.checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("proof generation overflow"))?;
        let rebuilt = true_delta >= ceil_sqrt(self.circuit.n());
        trace.finish(CpuPhase::Validation, started);
        let updated = if rebuilt {
            let started = trace.start();
            // The rebuild is one exclusive phase: nested anchor timing is off.
            let mut rebuilt = self.prove_anchor_traced(new_instance, &new_witness,
                kzg_backend, &mut CpuTrace::disabled())?;
            rebuilt.generation = generation;
            rebuilt.anchor_generation = generation;
            trace.finish(CpuPhase::AnchorRebuild, started);
            rebuilt
        } else {
            let delta = self.prove_delta(&aux.anchor_witness, &new_witness,
                &proof.anchor.wires, &aux.kzg, aux.opening_policy, batch_mode, kzg_backend, trace)?;
            let mut updated = proof.clone();
            updated.delta = Some(delta);
            updated.generation = generation;
            updated
        };
        let started = trace.start();
        let verification = self.verifying.verify_traced(new_instance, &updated, mode, trace.verification_mut());
        // End the outer interval before propagating failure, so rejected
        // candidates still contain a complete parent for the nested timings.
        trace.finish(CpuPhase::PairingVerify, started);
        ensure!(verification?, "updated proof failed public verification");
        // Prepare all fallible replacement state before changing the live aux.
        // Committing new state clones the full witness and instance.
        crate::workcount::record_scan(new_witness.wires.len() + new_instance.public_inputs.len());
        let refreshed_cache = if rebuilt {
            Some(self.prepare_opening_cache(&new_witness, aux.opening_policy, kzg_backend, trace)?)
        } else { None };
        if rebuilt {
            aux.anchor_witness = new_witness.clone();
            aux.kzg = refreshed_cache.unwrap();
            aux.scheduler.updates_since_rebuild = 0;
        } else {
            aux.scheduler.updates_since_rebuild += 1;
        }
        aux.current_instance = new_instance.clone();
        aux.current_witness = new_witness;
        aux.current_proof = updated.clone();
        aux.scheduler.generation = generation;
        Ok(updated)
    }

    pub(crate) fn check_background_state(&self, proof: &DynaverseProof, aux: &ProverAux) -> Result<()> {
        ensure!(proof == &aux.current_proof && proof.generation == aux.scheduler.generation
            && proof.anchor_generation <= proof.generation, "stale background state");
        ensure!(aux.anchor_witness.wires.len() == 6 * self.circuit.n(), "invalid anchor snapshot length");
        self.circuit.validate(&aux.current_instance, &aux.current_witness)?;
        Ok(())
    }

    pub(crate) fn background_rebuild_due(&self, aux: &ProverAux) -> Result<bool> {
        self.check_background_state(&aux.current_proof, aux)?;
        if aux.current_proof.anchor_generation == aux.scheduler.generation { return Ok(false); }
        let half = ceil_sqrt(self.circuit.n()).div_ceil(2);
        let changed = aux.current_witness.wires.iter().zip(&aux.anchor_witness.wires)
            .filter(|(current, anchor)| current != anchor).count();
        Ok(changed >= half || aux.scheduler.updates_since_rebuild >= half as u64)
    }

    pub(crate) fn validate_background_candidate(&self, candidate: &ProverAux) -> Result<()> {
        self.check_background_state(&candidate.current_proof, candidate)?;
        ensure!(candidate.current_proof.delta.is_none()
            && candidate.current_proof.generation == 0 && candidate.current_proof.anchor_generation == 0
            && candidate.anchor_witness == candidate.current_witness, "background candidate is not a complete anchor");
        ensure!(self.verifying.verify(&candidate.current_instance, &candidate.current_proof)?,
            "background anchor failed public verification");
        // Public anchor verification does not cover the private opening cache.
        // Check the whole cache against the independent CPU construction before
        // publishing it. This extra CPU work runs on the background thread.
        let expected = self.prepare_opening_cache(&candidate.anchor_witness,
            candidate.opening_policy, None, &mut CpuTrace::disabled())?;
        ensure!(candidate.kzg == expected, "background opening cache differs from CPU reference");
        Ok(())
    }

    pub(crate) fn install_background_candidate(&self, source_generation: u64, candidate: ProverAux,
        proof: &mut DynaverseProof, aux: &mut ProverAux,
        backend: Option<&dyn OperatorBackend>) -> Result<bool> {
        self.check_background_state(proof, aux)?;
        ensure!(source_generation <= proof.generation, "background candidate is from a future generation");
        if source_generation <= proof.anchor_generation { return Ok(false); }
        ensure!(candidate.opening_policy == aux.opening_policy, "background opening policy changed");
        ensure!(candidate.anchor_witness.wires.len() == 6 * self.circuit.n(), "invalid background witness length");
        let changed = aux.current_witness.wires.iter().zip(&candidate.anchor_witness.wires)
            .filter(|(current, anchor)| current != anchor).count();
        if changed >= ceil_sqrt(self.circuit.n()) { return Ok(false); }
        // Match the existing update API's binding between active snapshot, SRS
        // and proof before replacing it with a candidate from this worker.
        let mut anchor_padded = aux.anchor_witness.wires.clone();
        anchor_padded.resize(self.circuit.m(), Fr::zero());
        ensure!(self.srs.dynamo.commit_z_dense_unmasked(&anchor_padded)? == proof.anchor.w_commitment,
            "active anchor snapshot differs from proof");
        // Generation gaps require a delta even when the current witness has
        // reverted exactly to this older snapshot; prove_delta handles zeros.
        ensure!(source_generation != proof.generation || changed == 0,
            "background snapshot differs within the same generation");
        let delta = if source_generation == proof.generation { None } else {
            Some(self.prove_delta(&candidate.anchor_witness, &aux.current_witness,
                &candidate.current_proof.anchor.wires, &candidate.kzg, candidate.opening_policy,
                BatchOpenMode::Prepared, backend, &mut CpuTrace::disabled())?)
        };
        let replacement = DynaverseProof { anchor: candidate.current_proof.anchor,
            delta, anchor_generation: source_generation, generation: proof.generation };
        ensure!(self.verifying.verify(&aux.current_instance, &replacement)?,
            "background replacement failed public verification");
        // All fallible work finished. Current witness/instance/generation stay
        // live; only anchor/cache and their delta representation are replaced.
        aux.anchor_witness = candidate.anchor_witness;
        aux.kzg = candidate.kzg;
        aux.scheduler.updates_since_rebuild = proof.generation - source_generation;
        aux.current_proof = replacement.clone();
        *proof = replacement;
        Ok(true)
    }

    fn prove_delta(
        &self, anchor: &Witness, current: &Witness,
        anchor_wires: &WireCommitments, cache: &KzgPrecomputation,
        policy: AnchorOpeningPolicy, batch_mode: BatchOpenMode,
        kzg_backend: Option<&dyn OperatorBackend>, trace: &mut CpuTrace,
    ) -> Result<DeltaProof> {
        let n = self.circuit.n();
        let m = self.circuit.m();
        ensure!(anchor.wires.len() == 6 * n && current.wires.len() == 6 * n,
            "witness length differs from circuit");
        let backend = ArkReferenceBackend;
        // Building and later sparsifying the dense m-length difference.
        crate::workcount::record_scan(2 * m);
        let mut changed: Vec<Fr> = current.wires.iter().zip(&anchor.wires)
            .map(|(new, old)| *new - *old).collect();
        changed.resize(m, Fr::zero());
        let big = radix2_domain(m)?;
        let small = radix2_domain(n)?;
        let started = trace.start();
        let changed_coefficients = self.interpolate(&changed, kzg_backend)?;
        trace.finish(CpuPhase::Interpolation, started);
        let started = trace.start();
        let changed_w_commitment = self.commit_g1(&changed_coefficients, kzg_backend)?;
        trace.finish(CpuPhase::KzgCommit, started);
        let started = trace.start();
        let changed_sparse = sparse_from_dense(&changed)?;
        let changed_h_sparse = derive_h_sparse(&changed_sparse, &self.circuit.permutation())?;
        let dynamo = dynamo_prove_unmasked(
            &self.srs.dynamo, &changed_sparse, &changed_h_sparse, &backend
        )?;
        ensure!(dynamo.instance_z == changed_w_commitment, "changed X commitment mismatch");
        let changed_h_commitment = dynamo.instance_h;
        trace.finish(CpuPhase::DynamoMsm, started);
        let started = trace.start();
        let mut delta_coefficients: [Vec<Fr>; 6] = std::array::from_fn(|_| Vec::new());
        for i in 0..6 {
            delta_coefficients[i] = self.interpolate(&changed[i * n..(i + 1) * n], kzg_backend)?;
        }
        trace.finish(CpuPhase::Interpolation, started);
        let started = trace.start();
        let delta_points: [G1Affine; 6] = self.commit_many_g1(&delta_coefficients, kzg_backend)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("expected six changed wire commitments"))?;
        let changed_wires = WireCommitments {
            s1: delta_points[0], s2: delta_points[1], s3: delta_points[2],
            s4: delta_points[3], s5: delta_points[4], s6: delta_points[5],
        };
        trace.finish(CpuPhase::KzgCommit, started);
        let started = trace.start();
        let slices = (0..6)
            .map(|i| self.srs.consistency[i].prove(
                &changed, &changed[i * n..(i + 1) * n], &backend
            ))
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| anyhow::anyhow!("expected six changed consistency proofs"))?;
        let consistency = ConsistencyProof { slices };
        trace.finish(CpuPhase::Consistency, started);
        let started = trace.start();
        // This immediately invoked closure borrows self and trace only during
        // this block. Record the parent before propagating any substep error.
        let relations = (|| -> Result<DeltaBatchRelations> {
            let part_started = trace.start();
            // The Option owns the reference polynomial for this block. Prepared
            // mode borrows the immutable denominator from this parameter object.
            let reference_ix = if batch_mode == BatchOpenMode::Reference {
                let public_indices: Vec<usize> = (6 * n..m).collect();
                Some(vanishing_polynomial(&big, &public_indices)?)
            } else { None };
            let ix = match &reference_ix {
                Some(coefficients) => coefficients.as_slice(),
                None => self.public_division.divisor(),
            };
            trace.finish_batch_open(BatchOpenPhase::PublicVanishing, part_started);
            let part_started = trace.start();
            let public_quotient = match batch_mode {
                BatchOpenMode::Reference => divide_exact(&changed_coefficients, ix)?,
                BatchOpenMode::Prepared => self.public_division.divide_exact(&changed_coefficients)?,
            };
            trace.finish_batch_open(BatchOpenPhase::PublicQuotient, part_started);
            let part_started = trace.start();
            let public_zero_quotient = self.commit_g1(&public_quotient, kzg_backend)?;
            trace.finish_batch_open(BatchOpenPhase::PublicCommit, part_started);
            let part_started = trace.start();
            let mut affected = Vec::new();
            for i in 0..n {
                if !changed[3 * n + i].is_zero() || !changed[4 * n + i].is_zero()
                    || !changed[5 * n + i].is_zero() {
                    affected.push(i);
                }
            }
            let (ai, bi) = affected_set_factorization(&small, &affected)?;
            trace.finish_batch_open(BatchOpenPhase::Factors, part_started);
            let part_started = trace.start();
            let affected_divisor_g1 = self.commit_g1(&ai, kzg_backend)?;
            let affected_divisor_g2 = self.srs.kzg_x.commit_g2(&ai)?;
            let unaffected_divisor_g1 = self.commit_g1(&bi, kzg_backend)?;
            let unaffected_divisor_g2 = self.srs.kzg_x.commit_g2(&bi)?;
            trace.finish_batch_open(BatchOpenPhase::DivisorCommit, part_started);
            let part_started = trace.start();
            let mut anchor_local_coefficients = [Vec::new(), Vec::new(), Vec::new()];
            let mut changed_local_coefficients = [Vec::new(), Vec::new(), Vec::new()];
            for t in 0..3 {
                let mut anchor_values = Vec::with_capacity(affected.len());
                let mut changed_values = Vec::with_capacity(affected.len());
                for &i in &affected {
                    anchor_values.push(anchor.wires[(t + 3) * n + i]);
                    changed_values.push(changed[(t + 3) * n + i]);
                }
                anchor_local_coefficients[t] = interpolate_at_indices(&small, &affected, &anchor_values)?;
                changed_local_coefficients[t] = interpolate_at_indices(&small, &affected, &changed_values)?;
            }
            trace.finish_batch_open(BatchOpenPhase::LocalInterpolation, part_started);
            let part_started = trace.start();
            let mut updated_local_coefficients = [Vec::new(), Vec::new(), Vec::new()];
            for t in 0..3 {
                updated_local_coefficients[t] = add_polynomials(
                    &anchor_local_coefficients[t], &changed_local_coefficients[t]);
            }
            let local_product = multiply_polynomials(
                &updated_local_coefficients[0], &updated_local_coefficients[1]
            );
            let local_numerator = subtract_polynomials(
                &local_product, &updated_local_coefficients[2]
            );
            let local_quotient = divide_exact(&local_numerator, &ai)?;
            trace.finish_batch_open(BatchOpenPhase::LocalArithmetic, part_started);
            let part_started = trace.start();
            let local_multiplication_quotient = self.commit_g1(&local_quotient, kzg_backend)?;
            let updated_s5_g2 = self.srs.kzg_x.commit_g2(&updated_local_coefficients[1])?;
            trace.finish_batch_open(BatchOpenPhase::LocalCommit, part_started);
            Ok(DeltaBatchRelations { public_zero_quotient, affected, ai, bi,
                affected_divisor_g1, affected_divisor_g2, unaffected_divisor_g1, unaffected_divisor_g2,
                anchor_local_coefficients, changed_local_coefficients,
                local_multiplication_quotient, updated_s5_g2 })
        })();
        trace.finish(CpuPhase::BatchOpen, started);
        let DeltaBatchRelations { public_zero_quotient, affected, ai, bi,
            affected_divisor_g1, affected_divisor_g2, unaffected_divisor_g1, unaffected_divisor_g2,
            anchor_local_coefficients, changed_local_coefficients,
            local_multiplication_quotient, updated_s5_g2 } = relations?;
        let started = trace.start();
        let anchor_openings = if affected.is_empty() {
            // A_empty=1 and r_empty=0, so the quotient is the anchor itself.
            MultiplicationWireCommitments {
                s4: anchor_wires.s4, s5: anchor_wires.s5, s6: anchor_wires.s6,
            }
        } else if policy == AnchorOpeningPolicy::Precompute {
            MultiplicationWireCommitments {
                s4: combine_single_openings(&small, &affected, &cache.q4)?,
                s5: combine_single_openings(&small, &affected, &cache.q5)?,
                s6: combine_single_openings(&small, &affected, &cache.q6)?,
            }
        } else {
            // Reconstruct the original dense quotient using known local values,
            // so this baseline does not add generic open_batch evaluation work.
            let mut quotients = [Vec::new(), Vec::new(), Vec::new()];
            for t in 0..3 {
                let coefficients = self.interpolate(&anchor.wires[(t + 3) * n..(t + 4) * n], kzg_backend)?;
                let numerator = subtract_polynomials(&coefficients, &anchor_local_coefficients[t]);
                quotients[t] = divide_exact(&numerator, &ai)?;
            }
            self.commit_triple(&quotients, kzg_backend)?
        };
        trace.finish(CpuPhase::AnchorOpenings, started);
        let started = trace.start();
        // Keep the same error/timing boundary for this second exclusive block.
        let delta = (|| -> Result<DeltaProof> {
            let part_started = trace.start();
            let mut changed_opening_coefficients = [Vec::new(), Vec::new(), Vec::new()];
            let mut unaffected_zero_coefficients = [Vec::new(), Vec::new(), Vec::new()];
            for t in 0..3 {
                let numerator = subtract_polynomials(&delta_coefficients[t + 3], &changed_local_coefficients[t]);
                changed_opening_coefficients[t] = divide_exact(&numerator, &ai)?;
                unaffected_zero_coefficients[t] = divide_exact(&delta_coefficients[t + 3], &bi)?;
            }
            trace.finish_batch_open(BatchOpenPhase::OpeningQuotients, part_started);
            let part_started = trace.start();
            let delta = DeltaProof {
                changed_wires,
                changed_w_commitment,
                changed_h_commitment,
                dynamo,
                consistency,
                public_zero_quotient,
                affected_divisor_g1,
                affected_divisor_g2,
                unaffected_divisor_g1,
                unaffected_divisor_g2,
                anchor_local: self.commit_triple(&anchor_local_coefficients, kzg_backend)?,
                changed_local: self.commit_triple(&changed_local_coefficients, kzg_backend)?,
                updated_s5_g2,
                anchor_openings,
                changed_openings: self.commit_triple(&changed_opening_coefficients, kzg_backend)?,
                unaffected_zero_quotients: self.commit_triple(&unaffected_zero_coefficients, kzg_backend)?,
                local_multiplication_quotient,
            };
            trace.finish_batch_open(BatchOpenPhase::OpeningCommit, part_started);
            Ok(delta)
        })();
        trace.finish(CpuPhase::BatchOpen, started);
        delta
    }

    fn interpolate(&self, values: &[Fr], backend: Option<&dyn OperatorBackend>) -> Result<Vec<Fr>> {
        // Counted here, before dispatch, so CPU and GPU backends report alike.
        crate::workcount::record_ntt(values.len());
        let mut coefficients = values.to_vec();
        match backend {
            Some(backend) => backend.ntt(&mut coefficients, true)?,
            None => ArkReferenceBackend.ntt(&mut coefficients, true)?,
        }
        Ok(coefficients)
    }

    fn commit_g1(&self, coefficients: &[Fr], backend: Option<&dyn OperatorBackend>) -> Result<G1Affine> {
        // Option contains a borrowed trait object, so forwarding it copies only
        // the reference and never transfers ownership of the backend.
        let commitment = match backend {
            Some(backend) => self.srs.kzg_x.commit_with_backend(coefficients, backend)?,
            None => self.srs.kzg_x.commit(coefficients)?,
        };
        Ok(commitment.point)
    }

    fn commit_triple(&self, coefficients: &[Vec<Fr>; 3], backend: Option<&dyn OperatorBackend>)
        -> Result<MultiplicationWireCommitments>
    {
        let points = self.commit_many_g1(coefficients, backend)?;
        Ok(MultiplicationWireCommitments {
            s4: points[0], s5: points[1], s6: points[2],
        })
    }

    fn commit_many_g1(&self, coefficients: &[Vec<Fr>], backend: Option<&dyn OperatorBackend>) -> Result<Vec<G1Affine>> {
        let mut points = Vec::new();
        if let Some(backend) = backend {
            let mut slices = Vec::new();
            for polynomial in coefficients { slices.push(polynomial.as_slice()); }
            let commitments = self.srs.kzg_x.commit_batch_with_backend(&slices, backend)?;
            for commitment in commitments { points.push(commitment.point); }
        } else {
            for polynomial in coefficients { points.push(self.commit_g1(polynomial, None)?); }
        }
        Ok(points)
    }
}

impl DynaverseApi for DynaverseParams {
    fn prove(&self, instance: &Instance, witness: &Witness)
        -> Result<(DynaverseProof, ProverAux)>
    {
        self.prove_with_aux(instance, witness)
    }

    fn update(
        &self,
        old_instance: &Instance,
        new_instance: &Instance,
        old_witness: &Witness,
        update: &Update,
        proof: &DynaverseProof,
        aux: &mut ProverAux,
    ) -> Result<DynaverseProof> {
        self.update_proof(
            old_instance, new_instance, old_witness, update, proof, aux
        )
    }

    fn verify(&self, instance: &Instance, proof: &DynaverseProof) -> Result<bool> {
        self.verifying.verify(instance, proof)
    }
}

fn add_polynomials(left: &[Fr], right: &[Fr]) -> Vec<Fr> {
    let mut out = vec![Fr::zero(); left.len().max(right.len()).max(1)];
    for (i, value) in left.iter().enumerate() { out[i] += value; }
    for (i, value) in right.iter().enumerate() { out[i] += value; }
    out
}

fn ceil_sqrt(n: usize) -> usize {
    let mut root = 1usize;
    while root < n / root || (root == n / root && n % root != 0) {
        crate::workcount::record_poly(1, 1);
        root += 1;
    }
    root
}

impl DynaverseVerifyingKey {
    pub fn verify_anchor(&self, instance: &Instance, proof: &DynaverseProof) -> Result<bool> {
        if proof.delta.is_some() {
            return Ok(false);
        }
        self.verify(instance, proof)
    }

    pub fn verify(&self, instance: &Instance, proof: &DynaverseProof) -> Result<bool> {
        self.verify_traced(instance, proof, VerificationMode::Product, &mut VerificationTrace::disabled())
    }

    /// Original verifier retained with individual arkworks pairing calls.
    pub fn verify_reference(&self, instance: &Instance, proof: &DynaverseProof) -> Result<bool> {
        if instance.public_inputs.len() != self.n
            || proof.anchor_generation > proof.generation
            || proof.delta.is_some() != (proof.anchor_generation < proof.generation)
            || self.dynamo.size() != self.m
            || self.public_h_bases.len() != self.n
            || self.consistency.slices.iter().enumerate().any(|(i, vk)| {
                vk.m != self.m || vk.n != self.n || vk.offset != i * self.n
            })
            || !self.verify_anchor_core(&proof.anchor)?
        {
            return Ok(false);
        }
        if let Some(delta) = &proof.delta {
            if !self.verify_delta_core(&proof.anchor, delta)? {
                return Ok(false);
            }
        }
        crate::workcount::record_msm(self.public_h_bases.len());
        let public_h =
            G1Projective::msm_unchecked(&self.public_h_bases, &instance.public_inputs);
        let delta_h = proof.delta.as_ref()
            .map_or(G1Projective::zero(), |d| G1Projective::from(d.changed_h_commitment));
        Ok(G1Projective::from(proof.anchor.h_commitment) + delta_h + public_h
            == G1Projective::zero())
    }

    fn verify_anchor_core(&self, anchor: &AnchorProof) -> Result<bool> {
        if anchor.w_commitment != anchor.dynamo.instance_z
            || anchor.h_commitment != anchor.dynamo.instance_h
            || !dynamo_verify_with_key(&self.dynamo, &anchor.dynamo)?
        {
            return Ok(false);
        }
        let wires = [
            anchor.wires.s1,
            anchor.wires.s2,
            anchor.wires.s3,
            anchor.wires.s4,
            anchor.wires.s5,
            anchor.wires.s6,
        ];
        if !self.consistency.verify_six(
            anchor.w_commitment,
            &wires,
            &anchor.consistency,
        ) {
            return Ok(false);
        }
        let g1 = G1Projective::generator().into_affine();
        let g2 = G2Projective::generator().into_affine();
        if Bn254::pairing(anchor.public_zero_quotient, self.public_zero_divisor)
            != Bn254::pairing(anchor.w_commitment, g2)
            || G1Projective::from(wires[0]) + G1Projective::from(wires[1])
                != G1Projective::from(wires[2])
            || Bn254::pairing(wires[4], g2) != Bn254::pairing(g1, anchor.s5_g2)
            || Bn254::pairing(wires[3], anchor.s5_g2)
                != Bn254::pairing(wires[5], g2)
                    + Bn254::pairing(
                        anchor.multiplication_quotient,
                        self.multiplication_divisor,
                    )
        {
            return Ok(false);
        }
        Ok(true)
    }

    fn verify_delta_core(&self, anchor: &AnchorProof, delta: &DeltaProof) -> Result<bool> {
        if delta.changed_w_commitment != delta.dynamo.instance_z
            || delta.changed_h_commitment != delta.dynamo.instance_h
            || !dynamo_verify_with_key(&self.dynamo, &delta.dynamo)?
        {
            return Ok(false);
        }
        let changed_wires = [
            delta.changed_wires.s1, delta.changed_wires.s2, delta.changed_wires.s3,
            delta.changed_wires.s4, delta.changed_wires.s5, delta.changed_wires.s6,
        ];
        if !self.consistency.verify_six(
            delta.changed_w_commitment, &changed_wires, &delta.consistency
        ) {
            return Ok(false);
        }
        let g1 = G1Projective::generator().into_affine();
        let g2 = G2Projective::generator().into_affine();
        if Bn254::pairing(delta.public_zero_quotient, self.public_zero_divisor)
                != Bn254::pairing(delta.changed_w_commitment, g2)
            || G1Projective::from(changed_wires[0])
                + G1Projective::from(changed_wires[1])
                != G1Projective::from(changed_wires[2])
            || Bn254::pairing(delta.affected_divisor_g1, g2)
                != Bn254::pairing(g1, delta.affected_divisor_g2)
            || Bn254::pairing(delta.unaffected_divisor_g1, g2)
                != Bn254::pairing(g1, delta.unaffected_divisor_g2)
            || Bn254::pairing(delta.affected_divisor_g1, delta.unaffected_divisor_g2)
                != Bn254::pairing(self.multiplication_vanishing_g1, g2)
        {
            return Ok(false);
        }
        let r = triple_points(&delta.anchor_local);
        let r_star = triple_points(&delta.changed_local);
        let r_updated: [G1Affine; 3] = std::array::from_fn(|i| {
            (G1Projective::from(r[i]) + G1Projective::from(r_star[i])).into_affine()
        });
        if Bn254::pairing(r_updated[1], g2) != Bn254::pairing(g1, delta.updated_s5_g2)
            || Bn254::pairing(r_updated[0], delta.updated_s5_g2)
                != Bn254::pairing(r_updated[2], g2)
                    + Bn254::pairing(
                        delta.local_multiplication_quotient,
                        delta.affected_divisor_g2,
                    )
        {
            return Ok(false);
        }
        let anchor_wires = [anchor.wires.s4, anchor.wires.s5, anchor.wires.s6];
        let changed_mul_wires = [
            delta.changed_wires.s4, delta.changed_wires.s5, delta.changed_wires.s6
        ];
        let anchor_openings = triple_points(&delta.anchor_openings);
        let changed_openings = triple_points(&delta.changed_openings);
        let unaffected_zero = triple_points(&delta.unaffected_zero_quotients);
        for i in 0..3 {
            let anchor_difference =
                (G1Projective::from(anchor_wires[i]) - G1Projective::from(r[i])).into_affine();
            let changed_difference =
                (G1Projective::from(changed_mul_wires[i])
                    - G1Projective::from(r_star[i])).into_affine();
            if Bn254::pairing(anchor_openings[i], delta.affected_divisor_g2)
                    != Bn254::pairing(anchor_difference, g2)
                || Bn254::pairing(changed_openings[i], delta.affected_divisor_g2)
                    != Bn254::pairing(changed_difference, g2)
                || Bn254::pairing(unaffected_zero[i], delta.unaffected_divisor_g2)
                    != Bn254::pairing(changed_mul_wires[i], g2)
            {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

impl DynaverseVerifyingKey {
    pub fn verify_reference_traced(
        &self, instance: &Instance, proof: &DynaverseProof, trace: &mut VerificationTrace,
    ) -> Result<bool> {
        self.verify_traced(instance, proof, VerificationMode::Reference, trace)
    }

    /// Accumulates non-overlapping verification phases. Miller and final-exp
    /// times are nested breakdowns within phases, not additional wall time.
    pub fn verify_traced(
        &self, instance: &Instance, proof: &DynaverseProof,
        mode: VerificationMode, trace: &mut VerificationTrace,
    ) -> Result<bool> {
        let mut checks = PairingChecks::new(mode, trace);
        let started = checks.begin(VerificationPhase::Structure);
        let valid_shape = instance.public_inputs.len() == self.n
            && proof.anchor_generation <= proof.generation
            && proof.delta.is_some() == (proof.anchor_generation < proof.generation)
            && self.dynamo.size() == self.m
            && self.public_h_bases.len() == self.n;
        let mut valid_key = true;
        for i in 0..6 {
            let vk = &self.consistency.slices[i];
            if vk.m != self.m || vk.n != self.n || vk.offset != i * self.n {
                valid_key = false;
                break;
            }
        }
        checks.end(started);
        if !valid_shape || !valid_key { return Ok(false); }
        if !self.verify_anchor_with_checks(&proof.anchor, &mut checks)? { return Ok(false); }
        if let Some(delta) = &proof.delta {
            if !self.verify_delta_with_checks(&proof.anchor, delta, &mut checks)? { return Ok(false); }
        }
        let started = checks.begin(VerificationPhase::PublicInputs);
        crate::workcount::record_msm(self.public_h_bases.len());
        let public_h = G1Projective::msm_unchecked(&self.public_h_bases, &instance.public_inputs);
        let delta_h = match &proof.delta {
            Some(delta) => G1Projective::from(delta.changed_h_commitment),
            None => G1Projective::zero(),
        };
        let accepted = G1Projective::from(proof.anchor.h_commitment) + delta_h + public_h
            == G1Projective::zero();
        checks.end(started);
        Ok(accepted)
    }

    fn verify_anchor_with_checks(&self, anchor: &AnchorProof, checks: &mut PairingChecks<'_>) -> Result<bool> {
        let started = checks.begin(VerificationPhase::AnchorDynamo);
        let accepted = anchor.w_commitment == anchor.dynamo.instance_z
            && anchor.h_commitment == anchor.dynamo.instance_h
            && dynamo_verify_with_checks(&self.dynamo, &anchor.dynamo, checks)?;
        checks.end(started);
        if !accepted { return Ok(false); }
        let wires = [anchor.wires.s1, anchor.wires.s2, anchor.wires.s3,
            anchor.wires.s4, anchor.wires.s5, anchor.wires.s6];
        let started = checks.begin(VerificationPhase::AnchorConsistency);
        let accepted = self.consistency.verify_six_with_checks(anchor.w_commitment, &wires, &anchor.consistency, checks);
        checks.end(started);
        if !accepted { return Ok(false); }
        let started = checks.begin(VerificationPhase::AnchorRelations);
        let g1 = G1Projective::generator().into_affine();
        let g2 = G2Projective::generator().into_affine();
        let accepted = checks.equal((anchor.public_zero_quotient, self.public_zero_divisor), &[(anchor.w_commitment, g2)])
            && G1Projective::from(wires[0]) + G1Projective::from(wires[1]) == G1Projective::from(wires[2])
            && checks.equal((wires[4], g2), &[(g1, anchor.s5_g2)])
            && checks.equal((wires[3], anchor.s5_g2), &[
                (wires[5], g2), (anchor.multiplication_quotient, self.multiplication_divisor),
            ]);
        checks.end(started);
        Ok(accepted)
    }

    fn verify_delta_with_checks(
        &self, anchor: &AnchorProof, delta: &DeltaProof, checks: &mut PairingChecks<'_>,
    ) -> Result<bool> {
        let started = checks.begin(VerificationPhase::DeltaDynamo);
        let accepted = delta.changed_w_commitment == delta.dynamo.instance_z
            && delta.changed_h_commitment == delta.dynamo.instance_h
            && dynamo_verify_with_checks(&self.dynamo, &delta.dynamo, checks)?;
        checks.end(started);
        if !accepted { return Ok(false); }
        let wires = [delta.changed_wires.s1, delta.changed_wires.s2, delta.changed_wires.s3,
            delta.changed_wires.s4, delta.changed_wires.s5, delta.changed_wires.s6];
        let started = checks.begin(VerificationPhase::DeltaConsistency);
        let accepted = self.consistency.verify_six_with_checks(delta.changed_w_commitment, &wires, &delta.consistency, checks);
        checks.end(started);
        if !accepted { return Ok(false); }
        let started = checks.begin(VerificationPhase::DeltaDivisors);
        let g1 = G1Projective::generator().into_affine();
        let g2 = G2Projective::generator().into_affine();
        let accepted = checks.equal((delta.public_zero_quotient, self.public_zero_divisor), &[(delta.changed_w_commitment, g2)])
            && G1Projective::from(wires[0]) + G1Projective::from(wires[1]) == G1Projective::from(wires[2])
            && checks.equal((delta.affected_divisor_g1, g2), &[(g1, delta.affected_divisor_g2)])
            && checks.equal((delta.unaffected_divisor_g1, g2), &[(g1, delta.unaffected_divisor_g2)])
            && checks.equal((delta.affected_divisor_g1, delta.unaffected_divisor_g2), &[(self.multiplication_vanishing_g1, g2)]);
        checks.end(started);
        if !accepted { return Ok(false); }
        let started = checks.begin(VerificationPhase::DeltaMultiplication);
        let r = triple_points(&delta.anchor_local);
        let r_star = triple_points(&delta.changed_local);
        let mut r_updated = [G1Affine::identity(); 3];
        for i in 0..3 {
            r_updated[i] = (G1Projective::from(r[i]) + G1Projective::from(r_star[i])).into_affine();
        }
        let accepted = checks.equal((r_updated[1], g2), &[(g1, delta.updated_s5_g2)])
            && checks.equal((r_updated[0], delta.updated_s5_g2), &[
                (r_updated[2], g2), (delta.local_multiplication_quotient, delta.affected_divisor_g2),
            ]);
        checks.end(started);
        if !accepted { return Ok(false); }
        let started = checks.begin(VerificationPhase::DeltaOpenings);
        let anchor_wires = [anchor.wires.s4, anchor.wires.s5, anchor.wires.s6];
        let openings = triple_points(&delta.anchor_openings);
        let changed_openings = triple_points(&delta.changed_openings);
        let unaffected_zero = triple_points(&delta.unaffected_zero_quotients);
        let mut accepted = true;
        for i in 0..3 {
            let anchor_difference = (G1Projective::from(anchor_wires[i]) - G1Projective::from(r[i])).into_affine();
            let changed_difference = (G1Projective::from(wires[i + 3]) - G1Projective::from(r_star[i])).into_affine();
            if !checks.equal((openings[i], delta.affected_divisor_g2), &[(anchor_difference, g2)])
                || !checks.equal((changed_openings[i], delta.affected_divisor_g2), &[(changed_difference, g2)])
                || !checks.equal((unaffected_zero[i], delta.unaffected_divisor_g2), &[(wires[i + 3], g2)]) {
                accepted = false;
                break;
            }
        }
        checks.end(started);
        Ok(accepted)
    }
}

fn triple_points(values: &MultiplicationWireCommitments) -> [G1Affine; 3] {
    [values.s4, values.s5, values.s6]
}

fn sparse_from_dense(values: &[Fr]) -> Result<SparseVector> {
    SparseVector::try_new(
        values.len(),
        values.iter().copied().enumerate()
            .filter(|(_, value)| !value.is_zero())
            .collect(),
    )
}

pub(crate) struct ArkReferenceBackend;

impl OperatorBackend for ArkReferenceBackend {
    fn msm(&self, points: &[G1Affine], scalars: &[Fr]) -> Result<G1Affine> {
        ensure!(points.len() == scalars.len(), "MSM length mismatch");
        crate::workcount::record_msm(points.len());
        Ok(G1Projective::msm_unchecked(points, scalars).into_affine())
    }

    fn msm_batch(&self, request: &BatchMsmRequest<'_>) -> Result<Vec<G1Affine>> {
        ensure!(request.point_sets.len() == request.offsets.len()
            && request.point_sets.len() == request.lengths.len(), "batch shape mismatch");
        request.point_sets.iter().zip(request.offsets.iter().zip(request.lengths.iter()))
            .map(|(points, (&offset, &len))| {
                ensure!(len <= points.len(), "point range out of bounds");
                ensure!(offset.checked_add(len).is_some_and(|end| end <= request.shared_scalars.len()),
                    "scalar range out of bounds");
                self.msm(&points[..len], &request.shared_scalars[offset..offset + len])
            })
            .collect()
    }

    fn ntt(&self, values: &mut [Fr], inverse: bool) -> Result<()> {
        let domain = radix2_domain(values.len())?;
        let mut result = values.to_vec();
        if inverse { domain.ifft_in_place(&mut result); }
        else { domain.fft_in_place(&mut result); }
        values.copy_from_slice(&result);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn anchor_multiplication_relation_matches_original_dense_constraint() {
        use ark_std::UniformRand;

        let n = 32;
        let circuit = ToyCircuit::new(n).unwrap();
        let params = DynaverseParams::setup(circuit.clone(), &mut StdRng::seed_from_u64(3103)).unwrap();
        let small = radix2_domain(n).unwrap();
        let mut divisor = vec![Fr::zero(); n + 1];
        divisor[0] = -Fr::one();
        divisor[n] = Fr::one();
        let mut rng = StdRng::seed_from_u64(3104);
        for all_zero in [true, false] {
            let mut instance = Instance { public_inputs: vec![Fr::zero(); n] };
            let mut witness = Witness { wires: vec![Fr::zero(); 6 * n] };
            if !all_zero {
                for i in 0..n {
                    let left = Fr::rand(&mut rng);
                    let right = Fr::rand(&mut rng);
                    let multiplier = Fr::rand(&mut rng);
                    instance.public_inputs[i] = left;
                    witness.wires[i] = left;
                    witness.wires[n + i] = right;
                    witness.wires[2 * n + i] = left + right;
                    witness.wires[3 * n + i] = left + right;
                    witness.wires[4 * n + i] = multiplier;
                    witness.wires[5 * n + i] = (left + right) * multiplier;
                }
            }
            let s4 = small.ifft(&witness.wires[3 * n..4 * n]);
            let s5 = small.ifft(&witness.wires[4 * n..5 * n]);
            let s6 = small.ifft(&witness.wires[5 * n..6 * n]);
            let product = multiply_polynomials(&s4, &s5);
            let numerator = subtract_polynomials(&product, &s6);
            let expected = divide_exact(&numerator, &divisor).unwrap();
            let proof = params.prove_anchor(&instance, &witness).unwrap();
            assert_eq!(proof.anchor.multiplication_quotient, params.srs.kzg_x.commit(&expected).unwrap().point);
            assert!(params.verifying.verify(&instance, &proof).unwrap());
            assert!(params.verifying.verify_reference(&instance, &proof).unwrap());
        }
    }

    #[test]
    fn anchor_public_relation_keeps_exact_public_root_constraint() {
        use ark_std::UniformRand;

        let n = 32;
        let circuit = ToyCircuit::new(n).unwrap();
        let params = DynaverseParams::setup(circuit.clone(), &mut StdRng::seed_from_u64(3010)).unwrap();
        let domain = radix2_domain(8 * n).unwrap();
        let public_indices: Vec<usize> = (6 * n..8 * n).collect();
        let divisor = vanishing_polynomial(&domain, &public_indices).unwrap();
        let mut rng = StdRng::seed_from_u64(3011);

        for all_zero in [true, false] {
            let mut instance = Instance { public_inputs: vec![Fr::zero(); n] };
            let mut witness = Witness { wires: vec![Fr::zero(); 6 * n] };
            if !all_zero {
                for i in 0..n {
                    let left = Fr::rand(&mut rng);
                    let right = Fr::rand(&mut rng);
                    let multiplier = Fr::rand(&mut rng);
                    instance.public_inputs[i] = left;
                    witness.wires[i] = left;
                    witness.wires[n + i] = right;
                    witness.wires[2 * n + i] = left + right;
                    witness.wires[3 * n + i] = left + right;
                    witness.wires[4 * n + i] = multiplier;
                    witness.wires[5 * n + i] = (left + right) * multiplier;
                }
            }
            circuit.validate(&instance, &witness).unwrap();
            let mut padded = witness.wires.clone();
            padded.resize(8 * n, Fr::zero());
            let coefficients = domain.ifft(&padded);
            // Dense long division is independent of the prepared FFT division.
            let expected = divide_exact(&coefficients, &divisor).unwrap();
            let proof = params.prove_anchor(&instance, &witness).unwrap();
            assert_eq!(proof.anchor.public_zero_quotient, params.srs.kzg_x.commit(&expected).unwrap().point);
            assert!(params.verifying.verify(&instance, &proof).unwrap());
            assert!(params.verifying.verify_reference(&instance, &proof).unwrap());

            // A nonzero evaluation at a public root must never be accepted by
            // the exact division reused by the anchor path.
            padded[6 * n] = Fr::one();
            let corrupted = domain.ifft(&padded);
            assert!(divide_exact(&corrupted, &divisor).is_err());
            assert!(params.public_division.divide_exact(&corrupted).is_err());
        }
    }

    #[test]
    fn cached_anchor_openings_match_dense_batch_witnesses() {
        let n = 32;
        let circuit = ToyCircuit::new(n).unwrap();
        let (instance, witness) = circuit.valid_instance();
        let mut rng = StdRng::seed_from_u64(2802);
        let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
        let (proof, mut aux) = params.prove_with_aux(&instance, &witness).unwrap();
        let small = radix2_domain(n).unwrap();
        for (t, cache) in [&aux.kzg.q4, &aux.kzg.q5, &aux.kzg.q6].iter().enumerate() {
            let coefficients = small.ifft(&witness.wires[(t + 3) * n..(t + 4) * n]);
            for affected in [vec![0], vec![7, 1]] {
                assert_eq!(crate::kzg::combine_single_openings(&small, &affected, cache).unwrap(),
                    params.srs.kzg_x.open_batch(&coefficients, &small, &affected).unwrap().witness);
            }
        }
        let update = Update {
            indices: vec![4 * n + 1, 5 * n + 1, 4 * n + 7, 5 * n + 7],
            values: vec![witness.wires[4 * n + 1] + Fr::one(),
                witness.wires[5 * n + 1] + witness.wires[3 * n + 1],
                witness.wires[4 * n + 7] + Fr::one(),
                witness.wires[5 * n + 7] + witness.wires[3 * n + 7]],
        };
        let changed = params.update_proof(&instance, &instance, &witness, &update, &proof, &mut aux).unwrap();
        let delta = changed.delta.unwrap();
        for (t, point) in [delta.anchor_openings.s4, delta.anchor_openings.s5, delta.anchor_openings.s6].iter().enumerate() {
            let coefficients = small.ifft(&witness.wires[(t + 3) * n..(t + 4) * n]);
            assert_eq!(*point, params.srs.kzg_x.open_batch(&coefficients, &small, &[1, 7]).unwrap().witness);
        }
    }

    #[test]
    fn reanchor_refreshes_openings_for_the_next_sparse_update() {
        let n = 8;
        let circuit = ToyCircuit::new(n).unwrap();
        let (instance, witness) = circuit.valid_instance();
        let mut rng = StdRng::seed_from_u64(2803);
        let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
        let (proof, mut aux) = params.prove_with_aux(&instance, &witness).unwrap();
        let original_cache = aux.kzg.clone();
        let update = Update {
            indices: vec![4 * n + 1, 5 * n + 1, 4 * n + 2, 5 * n + 2],
            values: vec![witness.wires[4 * n + 1] + Fr::one(),
                witness.wires[5 * n + 1] + witness.wires[3 * n + 1],
                witness.wires[4 * n + 2] + Fr::one(),
                witness.wires[5 * n + 2] + witness.wires[3 * n + 2]],
        };
        let rebuilt = params.update_proof(&instance, &instance, &witness, &update, &proof, &mut aux).unwrap();
        assert!(rebuilt.delta.is_none());
        assert_ne!(aux.kzg, original_cache);
        let small = radix2_domain(n).unwrap();
        let coefficients = small.ifft(&aux.anchor_witness.wires[4 * n..5 * n]);
        assert_eq!(aux.kzg.q5[0], params.srs.kzg_x.open_single(&coefficients, &small, 0).unwrap().witness);
        let current = aux.current_witness.clone();
        let sparse = Update { indices: vec![4 * n], values: vec![current.wires[4 * n] + Fr::one()] };
        let changed = params.update_proof(&instance, &instance, &current, &sparse, &rebuilt, &mut aux).unwrap();
        assert_eq!(changed.delta.as_ref().unwrap().anchor_openings.s5, aux.kzg.q5[0]);
        assert!(params.verifying.verify(&instance, &changed).unwrap());
    }

    #[test]
    fn damaged_opening_cache_does_not_commit_new_state() {
        let n = 8;
        let circuit = ToyCircuit::new(n).unwrap();
        let (instance, witness) = circuit.valid_instance();
        let mut rng = StdRng::seed_from_u64(2804);
        let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
        let (proof, aux) = params.prove_with_aux(&instance, &witness).unwrap();
        assert_eq!(aux.kzg.q5.len(), n, "anchor opening cache is missing");
        for truncate in [false, true] {
            let mut damaged = aux.clone();
            if truncate { damaged.kzg.q5.pop(); }
            else { damaged.kzg.q5[0] = (G1Projective::from(damaged.kzg.q5[0]) + G1Projective::generator()).into_affine(); }
            let snapshot = damaged.clone();
            let update = Update { indices: vec![4 * n], values: vec![witness.wires[4 * n] + Fr::one()] };
            let mut trace = CpuTrace::new();
            assert!(params.update_proof_traced(&instance, &instance, &witness,
                &update, &proof, &mut damaged, &mut trace).is_err());
            assert_eq!(damaged, snapshot);
            if !truncate {
                assert!(trace.verification().counts().equations > 0);
                assert!(trace.verification().recorded_total() <= trace.elapsed(CpuPhase::PairingVerify));
            }
        }
    }

    #[test]
    fn corrupted_anchor_snapshot_does_not_commit_new_state() {
        let n = 8;
        let circuit = ToyCircuit::new(n).unwrap();
        let (instance, witness) = circuit.valid_instance();
        let mut rng = StdRng::seed_from_u64(1006);
        let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
        let (proof, mut aux) = params.prove_with_aux(&instance, &witness).unwrap();
        aux.anchor_witness.wires[4 * n] += Fr::one();
        let snapshot = aux.clone();
        let update = Update {
            indices: vec![4 * n],
            values: vec![witness.wires[4 * n] + Fr::from(2u64)],
        };
        assert!(params.update_proof(
            &instance, &instance, &witness, &update, &proof, &mut aux
        ).is_err());
        assert_eq!(aux, snapshot);
    }
}
