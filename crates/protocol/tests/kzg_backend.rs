use anyhow::{ensure, Result};
use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::{CurveGroup, VariableBaseMSM};
use dynaverse_protocol::dynaverse::DynaverseParams;
use dynaverse_protocol::kzg::KzgParams;
use dynaverse_protocol::toy_circuit::ToyCircuit;
use dynaverse_protocol::trace::CpuTrace;
use dynaverse_protocol::{AnchorOpeningPolicy, BatchMsmRequest, BatchOpenMode, OperatorBackend, Update, VerificationMode};
use rand::{rngs::StdRng, SeedableRng};
use std::cell::Cell;

// Runs actual arkworks MSM. The switch represents a device failure so that the
// protocol's error/transaction boundary can be tested without requiring CUDA.
struct CheckedBackend {
    calls: Cell<usize>, fail: Cell<bool>,
    ntt_calls: Cell<usize>, fail_ntt: Cell<bool>,
    batch_calls: Cell<usize>, fail_batch: Cell<bool>,
    short_batch: Cell<bool>,
}

impl CheckedBackend {
    fn new() -> Self { Self {
        calls: Cell::new(0), fail: Cell::new(false),
        ntt_calls: Cell::new(0), fail_ntt: Cell::new(false),
        batch_calls: Cell::new(0), fail_batch: Cell::new(false),
        short_batch: Cell::new(false),
    } }
}

impl OperatorBackend for CheckedBackend {
    fn msm(&self, points: &[G1Affine], scalars: &[Fr]) -> Result<G1Affine> {
        self.calls.set(self.calls.get() + 1);
        ensure!(!self.fail.get(), "injected device failure");
        ensure!(points.len() == scalars.len(), "MSM length mismatch");
        Ok(G1Projective::msm_unchecked(points, scalars).into_affine())
    }

    fn msm_batch(&self, request: &BatchMsmRequest<'_>) -> Result<Vec<G1Affine>> {
        self.batch_calls.set(self.batch_calls.get() + 1);
        ensure!(!self.fail_batch.get(), "injected batch failure");
        let count = request.point_sets.len();
        ensure!(request.offsets.len() == count && request.lengths.len() == count, "invalid batch shape");
        let mut result = Vec::new();
        for i in 0..count {
            let offset = request.offsets[i];
            let length = request.lengths[i];
            result.push(self.msm(&request.point_sets[i][..length], &request.shared_scalars[offset..offset + length])?);
        }
        if self.short_batch.get() { result.pop(); }
        Ok(result)
    }

    fn ntt(&self, values: &mut [Fr], inverse: bool) -> Result<()> {
        use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
        self.ntt_calls.set(self.ntt_calls.get() + 1);
        ensure!(!self.fail_ntt.get(), "injected NTT failure");
        let domain = Radix2EvaluationDomain::<Fr>::new(values.len()).unwrap();
        let mut result = values.to_vec();
        if inverse { domain.ifft_in_place(&mut result); }
        else { domain.fft_in_place(&mut result); }
        values.copy_from_slice(&result);
        Ok(())
    }
}

#[test]
fn backend_commit_matches_reference_and_checks_degree_before_dispatch() {
    let mut rng = StdRng::seed_from_u64(2903);
    let params = KzgParams::setup(17, &mut rng).unwrap();
    let backend = CheckedBackend::new();
    for length in [0, 1, 2, 17, 18] {
        let mut coefficients = Vec::new();
        for i in 0..length { coefficients.push(Fr::from((i + 1) as u64)); }
        assert_eq!(params.commit(&coefficients).unwrap(),
            params.commit_with_backend(&coefficients, &backend).unwrap());
    }
    let calls = backend.calls.get();
    assert!(calls > 0);
    assert!(params.commit_with_backend(&vec![Fr::from(1u64); 19], &backend).is_err());
    assert_eq!(backend.calls.get(), calls);
    backend.fail.set(true);
    assert!(params.commit_with_backend(&[Fr::from(1u64)], &backend).is_err());
    assert_eq!(params.commit_with_backend(&[], &backend).unwrap().point, G1Affine::identity());
}

#[test]
fn batch_commit_checks_all_degrees_and_backend_result_count() {
    let mut rng = StdRng::seed_from_u64(20604);
    let params = KzgParams::setup(17, &mut rng).unwrap();
    let backend = CheckedBackend::new();
    let one = [Fr::from(5u64)];
    let dense = vec![Fr::from(7u64); 18];
    let polynomials: [&[Fr]; 3] = [&one, &[], &dense];
    let expected = polynomials.iter().map(|p| params.commit(p).unwrap()).collect::<Vec<_>>();
    assert_eq!(params.commit_batch_with_backend(&polynomials, &backend).unwrap(), expected);
    let calls = backend.batch_calls.get();
    let too_large = vec![Fr::from(1u64); 19];
    assert!(params.commit_batch_with_backend(&[&one, &too_large], &backend).is_err());
    assert_eq!(backend.batch_calls.get(), calls);
    backend.short_batch.set(true);
    assert!(params.commit_batch_with_backend(&polynomials, &backend).is_err());
}

#[test]
fn injected_g1_backend_preserves_proofs_rebuilds_and_failure_state() {
    let n = 16;
    let circuit = ToyCircuit::new(n).unwrap();
    let (instance, witness) = circuit.valid_instance();
    let mut rng = StdRng::seed_from_u64(2904);
    let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
    let backend = CheckedBackend::new();
    let vk = params.verifying_key();
    for policy in [AnchorOpeningPolicy::Dense, AnchorOpeningPolicy::Precompute] {
        let (mut reference, mut reference_aux) = params.prove_with_aux_policy(&instance, &witness, policy).unwrap();
        let (mut candidate, mut candidate_aux) = params.prove_with_aux_traced_with_backend(
            &instance, &witness, policy, &backend, &mut CpuTrace::new()).unwrap();
        assert_eq!(reference, candidate);
        assert_eq!(reference_aux, candidate_aux);
        for gate in [0usize, 0, 1, 2, 3, 4] {
            let old = reference_aux.current_witness().clone();
            let multiplier = 4 * n + gate;
            let value = old.wires[multiplier] + Fr::from(1u64);
            let update = Update { indices: vec![multiplier, 5 * n + gate],
                values: vec![value, old.wires[3 * n + gate] * value] };
            let snapshot = candidate_aux.clone();
            backend.fail.set(true);
            assert!(params.update_proof_traced_with_backend(&instance, &instance, &old, &update,
                &candidate, &mut candidate_aux, VerificationMode::Product, BatchOpenMode::Prepared,
                &backend, &mut CpuTrace::new()).is_err());
            assert_eq!(candidate_aux, snapshot);
            backend.fail.set(false);
            reference = params.update_proof(&instance, &instance, &old, &update,
                &reference, &mut reference_aux).unwrap();
            candidate = params.update_proof_traced_with_backend(&instance, &instance, &old, &update,
                &candidate, &mut candidate_aux, VerificationMode::Product, BatchOpenMode::Prepared,
                &backend, &mut CpuTrace::new()).unwrap();
            assert_eq!(reference, candidate);
            assert_eq!(reference_aux, candidate_aux);
            assert!(vk.verify_reference(&instance, &candidate).unwrap());
        }
        assert!(candidate.anchor_generation > 0, "sequence must include a rebuild");
        backend.fail.set(true);
        let old = candidate_aux.current_witness().clone();
        let snapshot = candidate_aux.clone();
        let noop = Update { indices: vec![], values: vec![] };
        assert_eq!(params.update_proof_traced_with_backend(&instance, &instance, &old, &noop,
            &candidate, &mut candidate_aux, VerificationMode::Product, BatchOpenMode::Prepared,
            &backend, &mut CpuTrace::new()).unwrap(), candidate);
        assert_eq!(candidate_aux, snapshot);
        backend.fail.set(false);
    }
}

#[test]
fn selected_backend_runs_protocol_interpolation_and_ntt_failure_is_transactional() {
    let n = 16;
    let circuit = ToyCircuit::new(n).unwrap();
    let (instance, witness) = circuit.valid_instance();
    let mut rng = StdRng::seed_from_u64(2914);
    let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
    let backend = CheckedBackend::new();
    let (proof, mut aux) = params.prove_with_aux_traced_with_backend(
        &instance, &witness, AnchorOpeningPolicy::Dense, &backend, &mut CpuTrace::new()).unwrap();
    assert!(backend.ntt_calls.get() >= 7, "selected operator backend must run large and six wire IFFTs");
    assert_eq!(proof, params.prove_anchor(&instance, &witness).unwrap());
    let value = witness.wires[4 * n] + Fr::from(1u64);
    let update = Update { indices: vec![4 * n, 5 * n],
        values: vec![value, witness.wires[3 * n] * value] };
    let snapshot = aux.clone();
    backend.fail_ntt.set(true);
    let calls = backend.calls.get();
    assert!(params.update_proof_traced_with_backend(&instance, &instance, &witness, &update,
        &proof, &mut aux, VerificationMode::Product, BatchOpenMode::Prepared,
        &backend, &mut CpuTrace::new()).is_err());
    assert_eq!(aux, snapshot);
    assert_eq!(backend.calls.get(), calls, "NTT failure must stop before commitment dispatch");
}

#[test]
fn selected_backend_batches_wire_commitments_and_keeps_failure_transactional() {
    let n = 16;
    let circuit = ToyCircuit::new(n).unwrap();
    let (instance, witness) = circuit.valid_instance();
    let mut rng = StdRng::seed_from_u64(29314);
    let params = DynaverseParams::setup(circuit, &mut rng).unwrap();
    let backend = CheckedBackend::new();
    let (proof, mut aux) = params.prove_with_aux_traced_with_backend(&instance, &witness,
        AnchorOpeningPolicy::Dense, &backend, &mut CpuTrace::new()).unwrap();
    assert!(backend.batch_calls.get() > 0, "wire commitments must call batch backend");
    assert_eq!(proof, params.prove_anchor(&instance, &witness).unwrap());
    let value = witness.wires[4 * n] + Fr::from(1u64);
    let update = Update { indices: vec![4 * n, 5 * n], values: vec![value, witness.wires[3 * n] * value] };
    let snapshot = aux.clone();
    backend.fail_batch.set(true);
    assert!(params.update_proof_traced_with_backend(&instance, &instance, &witness, &update,
        &proof, &mut aux, VerificationMode::Product, BatchOpenMode::Prepared,
        &backend, &mut CpuTrace::new()).is_err());
    assert_eq!(aux, snapshot);
}
