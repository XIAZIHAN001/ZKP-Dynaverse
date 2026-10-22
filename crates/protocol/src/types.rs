//! Stable protocol types. Frozen at end of day 3 (three-week plan).
//! Every commitment uses a distinct named type - position-based
//! `Vec<G1Affine>` interpretation is forbidden.

use ark_bn254::{Fr, G1Affine, G2Affine};

/// Public circuit inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Instance {
    pub public_inputs: Vec<Fr>,
}

/// Full witness (all wires).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Witness {
    pub wires: Vec<Fr>,
}

/// Sparse update: indices into the witness with new values.
/// Canonical form: indices strictly increasing, no duplicates, values in field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Update {
    pub indices: Vec<usize>,
    pub values: Vec<Fr>,
}

impl Update {
    /// Normalize: sort by index, merge duplicates (last wins), check lengths.
    /// Returns error on indices out of range of `witness_len`.
    pub fn canonicalize(mut self, witness_len: usize) -> anyhow::Result<Self> {
        if self.indices.len() != self.values.len() {
            anyhow::bail!("update indices/values length mismatch");
        }
        // sort pairs by index
        let mut pairs: Vec<(usize, Fr)> = self
            .indices
            .drain(..)
            .zip(self.values.drain(..))
            .collect();
        pairs.sort_by_key(|(i, _)| *i);
        // dedup keeping the LAST value for each index: sort is stable, so
        // remove each duplicate keeping the final occurrence.
        let mut deduped: Vec<(usize, Fr)> = Vec::with_capacity(pairs.len());
        for (i, v) in pairs {
            if deduped.last().map(|(li, _)| *li) == Some(i) {
                deduped.pop();
            }
            deduped.push((i, v));
        }
        let (indices, values): (Vec<_>, Vec<_>) = deduped.into_iter().unzip();
        if let Some(&last) = indices.last() {
            if last >= witness_len {
                anyhow::bail!("update index {} out of range (witness len {})", last, witness_len);
            }
        }
        self.indices = indices;
        self.values = values;
        Ok(self)
    }

    /// k = Hamming distance of this update.
    pub fn k(&self) -> usize {
        self.indices.len()
    }
}

/// Six wire commitments s1..s6 (fan-in 2 Plonkish layout).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WireCommitments {
    pub s1: G1Affine,
    pub s2: G1Affine,
    pub s3: G1Affine,
    pub s4: G1Affine,
    pub s5: G1Affine,
    pub s6: G1Affine,
}

/// Five Dynamo sparse-permutation commitments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DynamoCommitments {
    pub alpha: G1Affine,
    pub beta: G1Affine,
    pub h: G1Affine,
    pub z: G1Affine,
    pub b: G1Affine,
}

/// Sub-proof placeholders: concrete structs land with their protocol modules
/// (day 5-8); the proof envelope is part of the frozen interface.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DynamoProof {
    pub commitments: DynamoCommitments,
    pub q: G1Affine,
    pub instance_z: G1Affine,
    pub instance_h: G1Affine,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiplicationProof {
    pub a_quotient_commitment: G1Affine,
    pub opening_evaluations: Vec<Fr>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsistencyProof {
    pub slices: [SliceConsistencyProof; 6],
}

/// Seven commitments for one large-domain/small-domain slice relation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SliceConsistencyProof {
    pub qac: G1Affine,
    pub qcd: G1Affine,
    pub qbd: G1Affine,
    pub d: G1Affine,
    pub a_degree: G1Affine,
    pub b_degree: G1Affine,
    pub d_degree: G1Affine,
}

/// Initial Fig. 4 relation proof, before sparse-update and masking extensions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnchorProof {
    pub wires: WireCommitments,
    pub s5_g2: G2Affine,
    pub w_commitment: G1Affine,
    pub h_commitment: G1Affine,
    pub dynamo: DynamoProof,
    pub consistency: ConsistencyProof,
    pub public_zero_quotient: G1Affine,
    pub multiplication_quotient: G1Affine,
}

/// Three commitments indexed by their multiplication-wire role.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MultiplicationWireCommitments {
    pub s4: G1Affine,
    pub s5: G1Affine,
    pub s6: G1Affine,
}

/// Fig. 4's proof that a changed witness preserves all relations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeltaProof {
    pub changed_wires: WireCommitments,
    pub changed_w_commitment: G1Affine,
    pub changed_h_commitment: G1Affine,
    pub dynamo: DynamoProof,
    pub consistency: ConsistencyProof,
    pub public_zero_quotient: G1Affine,
    pub affected_divisor_g1: G1Affine,
    pub affected_divisor_g2: G2Affine,
    pub unaffected_divisor_g1: G1Affine,
    pub unaffected_divisor_g2: G2Affine,
    pub anchor_local: MultiplicationWireCommitments,
    pub changed_local: MultiplicationWireCommitments,
    pub updated_s5_g2: G2Affine,
    pub anchor_openings: MultiplicationWireCommitments,
    pub changed_openings: MultiplicationWireCommitments,
    pub unaffected_zero_quotients: MultiplicationWireCommitments,
    pub local_multiplication_quotient: G1Affine,
}

/// The full Dynaverse proof object.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DynaverseProof {
    pub anchor: AnchorProof,
    pub delta: Option<DeltaProof>,
    /// Generation when the current anchor was last rebuilt.
    pub anchor_generation: u64,
    pub generation: u64,
}

/// Single-point KZG opening commitments for the current anchor's three
/// multiplication wires, indexed by the small radix-2 domain. These are
/// witness-dependent and must be replaced whenever the anchor is rebuilt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KzgPrecomputation {
    pub q4: Vec<G1Affine>,
    pub q5: Vec<G1Affine>,
    pub q6: Vec<G1Affine>,
}

/// Local prover policy; it never changes the public proof or verifying key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnchorOpeningPolicy {
    Precompute,
    Dense,
}

/// CPU verification computation only; public proof and key formats are shared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerificationMode {
    Reference,
    Product,
}

/// Local public-zero quotient calculation; proof and aux formats are unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchOpenMode {
    Reference,
    Prepared,
}

/// De-amortization scheduler state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SchedulerState {
    pub generation: u64,
    pub updates_since_rebuild: u64,
    pub rebuild_interval: u64,
}

/// Prover side-state carried across updates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProverAux {
    pub(crate) anchor_witness: Witness,
    pub(crate) current_witness: Witness,
    pub(crate) current_instance: Instance,
    pub(crate) current_proof: DynaverseProof,
    pub(crate) kzg: KzgPrecomputation,
    pub(crate) opening_policy: AnchorOpeningPolicy,
    pub(crate) scheduler: SchedulerState,
}

impl ProverAux {
    pub fn opening_policy(&self) -> AnchorOpeningPolicy {
        self.opening_policy
    }

    pub fn cached_opening_points(&self) -> usize {
        self.kzg.q4.len() + self.kzg.q5.len() + self.kzg.q6.len()
    }

    pub fn anchor_witness(&self) -> &Witness {
        &self.anchor_witness
    }

    pub fn current_witness(&self) -> &Witness {
        &self.current_witness
    }

    pub fn current_instance(&self) -> &Instance {
        &self.current_instance
    }

    pub fn scheduler(&self) -> SchedulerState {
        self.scheduler
    }
}

/// Batched MSM request: several point sets sharing the same sparse scalar
/// vector via offsets/lengths into `shared_scalars`.
#[derive(Clone, Debug)]
pub struct BatchMsmRequest<'a> {
    pub point_sets: &'a [&'a [ark_bn254::G1Affine]],
    pub shared_scalars: &'a [ark_bn254::Fr],
    pub offsets: &'a [usize],
    pub lengths: &'a [usize],
}

/// Borrowed sparse inputs. Each entry contributes `value * bases[index]`;
/// order, repeated indices and zero/full-width scalars are preserved.
#[derive(Clone, Copy, Debug)]
pub struct SparseMsmRequest<'a> {
    pub bases: &'a [ark_bn254::G1Affine],
    pub entries: &'a [(usize, ark_bn254::Fr)],
}

impl SparseMsmRequest<'_> {
    /// Validate the whole selection before a backend can dispatch any group.
    /// Only selected bases belong to this operation; no full-SRS scan is needed.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.entries.iter().all(|&(index, _)| index < self.bases.len()),
            "sparse MSM index out of range");
        Ok(())
    }
}

/// Operator backend: CPU and GPU (ICICLE) implementations must be
/// element-for-element identical on all inputs.
pub trait OperatorBackend {
    /// Setup-only fixed-generator arrays, preserving order (including zero).
    /// Defaults keep existing CPU backends source-compatible. GPU implementations
    /// must bound temporary storage and propagate errors without CPU fallback.
    fn fixed_base_g1(&self, scalars: &[ark_bn254::Fr])
        -> anyhow::Result<Vec<ark_bn254::G1Affine>> {
        crate::setup::cpu_g1(scalars)
    }

    fn fixed_base_g2(&self, scalars: &[ark_bn254::Fr])
        -> anyhow::Result<Vec<ark_bn254::G2Affine>> {
        crate::setup::cpu_g2(scalars)
    }

    fn msm(
        &self,
        points: &[ark_bn254::G1Affine],
        scalars: &[ark_bn254::Fr],
    ) -> anyhow::Result<ark_bn254::G1Affine>;

    fn msm_batch(&self, request: &BatchMsmRequest<'_>) -> anyhow::Result<Vec<ark_bn254::G1Affine>>;

    /// Additive indexed interface. Existing backends keep their real dense
    /// batch implementation; GPU backends can encode selected bases directly.
    fn msm_sparse_batch(&self, requests: &[SparseMsmRequest<'_>])
        -> anyhow::Result<Vec<ark_bn254::G1Affine>> {
        for request in requests { request.validate()?; }
        if requests.is_empty() { return Ok(Vec::new()); }
        let mut gathered = Vec::new();
        let mut scalars = Vec::new();
        let mut offsets = Vec::new();
        let mut lengths = Vec::new();
        gathered.try_reserve_exact(requests.len())?;
        offsets.try_reserve_exact(requests.len())?;
        lengths.try_reserve_exact(requests.len())?;
        let mut shared_offsets = std::collections::HashMap::new();
        shared_offsets.try_reserve(requests.len())?;
        for request in requests {
            let mut points = Vec::new();
            points.try_reserve_exact(request.entries.len())?;
            for &(index, _) in request.entries { points.push(request.bases[index]); }
            // Raw pointers are identity keys only, never dereferenced. The
            // borrowed slices stay alive throughout this synchronous call.
            // Equal length or equal indices do not imply shared scalar values.
            let key = (request.entries.as_ptr(), request.entries.len());
            let offset = match shared_offsets.get(&key) {
                Some(&offset) => offset,
                None => {
                    let offset = scalars.len();
                    scalars.try_reserve(request.entries.len())?;
                    for &(_, value) in request.entries { scalars.push(value); }
                    shared_offsets.insert(key, offset);
                    offset
                }
            };
            gathered.push(points);
            offsets.push(offset);
            lengths.push(request.entries.len());
        }
        let point_sets: Vec<&[ark_bn254::G1Affine]> = gathered.iter().map(Vec::as_slice).collect();
        let result = self.msm_batch(&BatchMsmRequest { point_sets: &point_sets,
            shared_scalars: &scalars, offsets: &offsets, lengths: &lengths })?;
        anyhow::ensure!(result.len() == requests.len(), "backend returned wrong sparse commitment count");
        Ok(result)
    }

    fn ntt(&self, values: &mut [ark_bn254::Fr], inverse: bool) -> anyhow::Result<()>;

    /// Explicit host-bound G2 operation. Older CPU backends retain their
    /// reference behavior; GPU selection checks capabilities before dispatch.
    fn msm_g2_host(&self,points:&[ark_bn254::G2Affine],scalars:&[ark_bn254::Fr])
        ->anyhow::Result<ark_bn254::G2Affine> {
        use ark_ec::{CurveGroup,VariableBaseMSM};
        anyhow::ensure!(points.len()==scalars.len(),"G2 MSM length mismatch");
        Ok(ark_bn254::G2Projective::msm_unchecked(points,scalars).into_affine())
    }

    fn gpu_capabilities(&self)->GpuCapabilities {GpuCapabilities::default()}
}

#[derive(Clone,Copy,Debug,Default,PartialEq,Eq)]
pub struct GpuCapabilities {
    pub g1_msm:bool,
    pub fr_ntt:bool,
    pub g2_msm:bool,
}

/// Top-level API: prove / update / verify.
pub trait DynaverseApi {
    fn prove(
        &self,
        instance: &Instance,
        witness: &Witness,
    ) -> anyhow::Result<(DynaverseProof, ProverAux)>;

    #[allow(clippy::too_many_arguments)]
    fn update(
        &self,
        old_instance: &Instance,
        new_instance: &Instance,
        old_witness: &Witness,
        update: &Update,
        proof: &DynaverseProof,
        aux: &mut ProverAux,
    ) -> anyhow::Result<DynaverseProof>;

    fn verify(&self, instance: &Instance, proof: &DynaverseProof) -> anyhow::Result<bool>;
}
