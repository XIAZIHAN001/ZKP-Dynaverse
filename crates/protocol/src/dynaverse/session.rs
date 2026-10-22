//! Encapsulated prover session. Initialization and explicit reanchoring are
//! dense; a fixed-anchor update reads only the supplied edits and delta map.
use super::*;
use std::collections::{BTreeMap, BTreeSet};

/// The parameter borrow binds a session to its circuit and SRS. Its private,
/// owned state cannot be replaced with caller-supplied stale aux/proof data.
/// Public inputs are also edited sparsely; getters borrow complete snapshots
/// without making copying them part of the update's complexity boundary.
pub struct UpdateSession<'p> {
    params: &'p DynaverseParams,
    aux: ProverAux,
    delta: BTreeMap<usize, Fr>,
    public_h: G1Projective,
}

impl DynaverseParams {
    pub fn start_update_session(&self, instance: &Instance, witness: &Witness) -> Result<UpdateSession<'_>> {
        let (proof, aux) = self.prove_with_aux(instance, witness)?;
        ensure!(self.verifying.verify(instance, &proof)?, "initial session proof failed verification");
        crate::workcount::record_msm(instance.public_inputs.len());
        let public_h = G1Projective::msm_unchecked(&self.verifying.public_h_bases, &instance.public_inputs);
        Ok(UpdateSession { params: self, aux, delta: BTreeMap::new(), public_h })
    }
}

impl UpdateSession<'_> {
    pub fn proof(&self) -> &DynaverseProof { &self.aux.current_proof }
    pub fn current_witness(&self) -> &Witness { &self.aux.current_witness }
    pub fn current_instance(&self) -> &Instance { &self.aux.current_instance }
    pub fn anchor_distance(&self) -> usize { self.delta.len() }

    pub fn update(&mut self, update: &Update) -> Result<DynaverseProof> {
        self.update_with_public_inputs(update, &Update { indices: Vec::new(), values: Vec::new() })
    }

    /// Public edits use public-input indices, separately from witness indices.
    /// A valid public change must also update the circuit's matching wires.
    pub fn update_with_public_inputs(&mut self, update: &Update, public_update: &Update) -> Result<DynaverseProof> {
        self.apply(update, public_update, None)
    }

    /// Only G1 MSMs are dispatched. Field polynomial arithmetic and G2 remain
    /// CPU reference work; no device handle is retained by the session.
    pub fn update_with_backend(
        &mut self, update: &Update, public_update: &Update, backend: &dyn OperatorBackend,
    ) -> Result<DynaverseProof> {
        self.apply(update, public_update, Some(backend))
    }

    fn apply(&mut self, update: &Update, public_update: &Update,
        backend: Option<&dyn OperatorBackend>) -> Result<DynaverseProof> {
        let n = self.params.circuit.n();
        let normalized = update.clone().canonicalize(6 * n)?;
        let public = public_update.clone().canonicalize(n)?;
        crate::workcount::record_scan(normalized.indices.len() + public.indices.len());
        let edits: BTreeMap<usize, Fr> = normalized.indices.iter().copied()
            .zip(normalized.values.iter().copied()).collect();
        let public_edits: BTreeMap<usize, Fr> = public.indices.iter().copied()
            .zip(public.values.iter().copied()).collect();
        let mut dirty = BTreeSet::new();
        let mut changed = false;
        for (&index, &value) in &edits {
            if value != self.aux.current_witness.wires[index] { changed = true; dirty.insert(index % n); }
        }
        for (&index, &value) in &public_edits {
            if value != self.aux.current_instance.public_inputs[index] { changed = true; dirty.insert(index); }
        }
        if !changed { return Ok(self.aux.current_proof.clone()); }
        // These closures borrow the staged maps and read through to live state.
        // No edit is published until validation, proving and self-check finish.
        let wire = |i: usize| edits.get(&i).copied().unwrap_or(self.aux.current_witness.wires[i]);
        let input = |i: usize| public_edits.get(&i).copied().unwrap_or(self.aux.current_instance.public_inputs[i]);
        crate::workcount::record_gates(dirty.len());
        for &i in &dirty {
            ensure!(wire(i) == input(i), "public copy constraint fails at {i}");
            ensure!(wire(2 * n + i) == wire(3 * n + i), "wire copy constraint fails at {i}");
            ensure!(wire(i) + wire(n + i) == wire(2 * n + i), "addition gate fails at {i}");
            ensure!(wire(3 * n + i) * wire(4 * n + i) == wire(5 * n + i), "multiplication gate fails at {i}");
        }
        let generation = self.aux.current_proof.generation.checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("proof generation overflow"))?;
        crate::workcount::record_scan(self.delta.len());
        let mut delta = self.delta.clone();
        for (&index, &value) in &edits {
            let difference = value - self.aux.anchor_witness.wires[index];
            if difference.is_zero() { delta.remove(&index); }
            else { delta.insert(index, difference); }
        }
        // The threshold was computed during initialization. Recomputing it
        // here with ceil_sqrt's loop would cost Theta(sqrt(n)) per update.
        if delta.len() as u64 >= self.aux.scheduler.rebuild_interval {
            // Explicitly preserve the existing synchronous rebuild semantics.
            // These O(n) copies/validation/cache work are NOT a de-amortized step.
            crate::workcount::record_scan(2 * self.aux.current_instance.public_inputs.len()
                + self.aux.current_witness.wires.len());
            let old_instance = self.aux.current_instance.clone();
            let old_witness = self.aux.current_witness.clone();
            let mut instance = old_instance.clone();
            for (&i, &value) in &public_edits { instance.public_inputs[i] = value; }
            let old_proof = self.aux.current_proof.clone();
            let result = self.params.update_proof_backend(&old_instance, &instance, &old_witness,
                &normalized, &old_proof, &mut self.aux, VerificationMode::Product,
                BatchOpenMode::Prepared, backend, &mut CpuTrace::disabled())?;
            crate::workcount::record_msm(n);
            self.public_h = G1Projective::msm_unchecked(&self.params.verifying.public_h_bases, &instance.public_inputs);
            self.delta.clear();
            return Ok(result);
        }
        let entries: Vec<(usize, Fr)> = delta.iter().map(|(&i, &v)| (i, v)).collect();
        let selected: &dyn OperatorBackend = backend.unwrap_or(&SparseCpuBackend);
        let proof_delta = self.params.prove_delta_sparse(&self.aux.anchor_witness, &entries,
            &self.aux.current_proof.anchor.wires, &self.aux.kzg, selected)?;
        let mut public_h = self.public_h;
        let mut public_differences = Vec::with_capacity(public_edits.len());
        for (&i, &value) in &public_edits {
            let difference = value - self.aux.current_instance.public_inputs[i];
            if !difference.is_zero() { public_differences.push((i, difference)); }
        }
        if !public_differences.is_empty() {
            public_h += crate::kzg::gather_msm_batch(
                &[(&self.params.verifying.public_h_bases, &public_differences)], selected)?[0];
        }
        let updated = DynaverseProof { anchor: self.aux.current_proof.anchor.clone(), delta: Some(proof_delta),
            anchor_generation: self.aux.current_proof.anchor_generation, generation };
        let vk = &self.params.verifying;
        ensure!(vk.verify_anchor_core(&updated.anchor)?
            && vk.verify_delta_core(&updated.anchor, updated.delta.as_ref().unwrap())?
            && G1Projective::from(updated.anchor.h_commitment)
                + updated.delta.as_ref().unwrap().changed_h_commitment + public_h == G1Projective::zero(),
            "sparse session proof failed verification");
        // All fallible operations are over. Publish O(k) edits, the O(d) map
        // and the fixed-size proof together; no full witness/instance clone.
        for (&i, &value) in &edits { self.aux.current_witness.wires[i] = value; }
        for (&i, &value) in &public_edits { self.aux.current_instance.public_inputs[i] = value; }
        self.aux.scheduler.generation = generation;
        self.aux.scheduler.updates_since_rebuild = generation - updated.anchor_generation;
        self.aux.current_proof = updated.clone();
        self.delta = delta;
        self.public_h = public_h;
        Ok(updated)
    }
}

/// Uninstrumented arithmetic backend: the sparse protocol counts inputs before
/// dispatch, once, just as it does when a real GPU backend is selected.
struct SparseCpuBackend;
impl OperatorBackend for SparseCpuBackend {
    fn msm(&self, points: &[G1Affine], scalars: &[Fr]) -> Result<G1Affine> {
        ensure!(points.len() == scalars.len(), "MSM length mismatch");
        Ok(G1Projective::msm_unchecked(points, scalars).into_affine())
    }
    fn msm_batch(&self, request: &BatchMsmRequest<'_>) -> Result<Vec<G1Affine>> {
        ensure!(request.point_sets.len() == request.offsets.len()
            && request.point_sets.len() == request.lengths.len(), "batch shape mismatch");
        let mut output = Vec::with_capacity(request.point_sets.len());
        for i in 0..request.point_sets.len() {
            let offset = request.offsets[i];
            let len = request.lengths[i];
            ensure!(len <= request.point_sets[i].len()
                && offset.checked_add(len).is_some_and(|end| end <= request.shared_scalars.len()),
                "batch range out of bounds");
            output.push(self.msm(&request.point_sets[i][..len], &request.shared_scalars[offset..offset + len])?);
        }
        Ok(output)
    }
    fn ntt(&self, values: &mut [Fr], inverse: bool) -> Result<()> {
        ArkReferenceBackend.ntt(values, inverse)
    }
}
