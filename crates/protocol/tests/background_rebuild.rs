use anyhow::{ensure, Result};
use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::{CurveGroup, VariableBaseMSM};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use dynaverse_protocol::background::{BackgroundRebuilder, SwapOutcome};
use dynaverse_protocol::dynaverse::DynaverseParams;
use dynaverse_protocol::toy_circuit::ToyCircuit;
use dynaverse_protocol::{AnchorOpeningPolicy, BatchMsmRequest, DynaverseProof, Instance, OperatorBackend, ProverAux, Update};
use rand::{rngs::StdRng, SeedableRng};
use std::{cell::Cell, rc::Rc, sync::{mpsc, Arc, atomic::{AtomicBool, AtomicUsize, Ordering}}, thread, time::{Duration, Instant}};

// Constructed within the worker: Rc deliberately makes this backend !Send.
struct LocalOracle { _local: Rc<()>, fail: bool, dropped: Option<Arc<AtomicUsize>>,
    corrupt_cache: bool, ntt_calls: Cell<usize> }
impl LocalOracle { fn new() -> Self { Self { _local: Rc::new(()), fail: false, dropped: None,
    corrupt_cache: false, ntt_calls: Cell::new(0) } } }
impl Drop for LocalOracle {
    fn drop(&mut self) { if let Some(count) = &self.dropped { count.fetch_add(1, Ordering::SeqCst); } }
}
impl OperatorBackend for LocalOracle {
    fn msm(&self, points: &[G1Affine], scalars: &[Fr]) -> Result<G1Affine> {
        ensure!(!self.fail, "injected worker operator failure");
        Ok(G1Projective::msm_unchecked(points, scalars).into_affine())
    }
    fn msm_batch(&self, request: &BatchMsmRequest<'_>) -> Result<Vec<G1Affine>> {
        let mut result = Vec::new();
        for i in 0..request.point_sets.len() { let offset = request.offsets[i]; let length = request.lengths[i];
            result.push(self.msm(&request.point_sets[i][..length], &request.shared_scalars[offset..offset + length])?); }
        Ok(result)
    }
    fn ntt(&self, values: &mut [Fr], inverse: bool) -> Result<()> {
        ensure!(!self.fail, "injected worker operator failure");
        self.ntt_calls.set(self.ntt_calls.get() + 1);
        if self.corrupt_cache && self.ntt_calls.get() > 7 { values.fill(Fr::from(0u64)); return Ok(()); }
        let domain = Radix2EvaluationDomain::<Fr>::new(values.len()).unwrap();
        let transformed = if inverse { domain.ifft(values) } else { domain.fft(values) };
        values.copy_from_slice(&transformed); Ok(())
    }
}
fn setup(policy: AnchorOpeningPolicy) -> (Arc<DynaverseParams>, Instance, DynaverseProof, ProverAux) {
    setup_at(16, policy)
}
fn setup_at(n: usize, policy: AnchorOpeningPolicy) -> (Arc<DynaverseParams>, Instance, DynaverseProof, ProverAux) {
    let circuit = ToyCircuit::new(n).unwrap(); let (instance, witness) = circuit.valid_instance();
    let mut rng = StdRng::seed_from_u64(20605);
    let params = Arc::new(DynaverseParams::setup(circuit, &mut rng).unwrap());
    let (proof, aux) = params.prove_with_aux_policy(&instance, &witness, policy).unwrap();
    (params, instance, proof, aux)
}
fn update(params: &DynaverseParams, instance: &Instance, proof: &mut DynaverseProof, aux: &mut ProverAux, gate: usize) {
    let n = instance.public_inputs.len();
    let old = aux.current_witness().clone(); let value = old.wires[4 * n + gate] + Fr::from(1u64);
    let edit = Update { indices: vec![4 * n + gate, 5 * n + gate], values: vec![value, old.wires[3 * n + gate] * value] };
    *proof = params.update_proof(instance, instance, &old, &edit, proof, aux).unwrap();
}
fn poll(worker: &mut BackgroundRebuilder, proof: &mut DynaverseProof, aux: &mut ProverAux) -> Result<SwapOutcome> {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop { let outcome = worker.try_swap(proof, aux, None)?;
        if !matches!(outcome, SwapOutcome::Building { .. }) { return Ok(outcome); }
        assert!(Instant::now() < deadline, "background worker did not finish");
        thread::sleep(Duration::from_millis(1)); }
}

#[test]
fn delayed_snapshot_rebases_current_state_and_preserves_caches() {
    for policy in [AnchorOpeningPolicy::Dense, AnchorOpeningPolicy::Precompute] {
        let (params, instance, mut proof, mut aux) = setup(policy);
        update(&params, &instance, &mut proof, &mut aux, 1);
        let snapshot_witness = aux.current_witness().clone();
        let (ready_tx, ready_rx) = mpsc::sync_channel(1); let (release_tx, release_rx) = mpsc::sync_channel(1);
        let mut worker = BackgroundRebuilder::new(params.clone(), move || {
            ready_tx.send(()).unwrap(); release_rx.recv_timeout(Duration::from_secs(15)).unwrap(); Ok(LocalOracle::new())
        }).unwrap();
        assert!(worker.start(&aux).unwrap(), "approved background job must actually start");
        ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(!worker.start(&aux).unwrap(), "only one snapshot may be outstanding");
        update(&params, &instance, &mut proof, &mut aux, 1);
        let current = aux.current_witness().clone(); release_tx.send(()).unwrap();
        assert_eq!(poll(&mut worker, &mut proof, &mut aux).unwrap(), SwapOutcome::Swapped { anchor_generation: 1, current_generation: 2 });
        assert_eq!(*aux.current_witness(), current);
        assert_eq!(*aux.anchor_witness(), snapshot_witness);
        assert_eq!(aux.scheduler().generation, 2);
        assert_eq!(aux.scheduler().updates_since_rebuild, 1);
        assert!(params.verifying_key().verify_reference(&instance, &proof).unwrap());
        update(&params, &instance, &mut proof, &mut aux, 1);
        assert!(params.verifying_key().verify_reference(&instance, &proof).unwrap());
        assert_eq!(proof.anchor_generation, 1);
    }
}

#[test]
fn candidate_older_than_foreground_rebuild_is_discarded() {
    let (params, instance, mut proof, mut aux) = setup(AnchorOpeningPolicy::Precompute);
    update(&params, &instance, &mut proof, &mut aux, 1);
    let (release_tx, release_rx) = mpsc::sync_channel(1);
    let mut worker = BackgroundRebuilder::new(params.clone(), move || { release_rx.recv_timeout(Duration::from_secs(15)).unwrap(); Ok(LocalOracle::new()) }).unwrap();
    assert!(worker.start(&aux).unwrap());
    update(&params, &instance, &mut proof, &mut aux, 2);
    assert_eq!(proof.anchor_generation, 2);
    let before_proof = proof.clone(); let before_aux = aux.clone(); release_tx.send(()).unwrap();
    assert_eq!(poll(&mut worker, &mut proof, &mut aux).unwrap(), SwapOutcome::Discarded { source_generation: 1 });
    assert_eq!(proof, before_proof); assert_eq!(aux, before_aux);
}

#[test]
fn worker_failure_recreates_backend_and_drop_joins_its_owner() {
    let (params, instance, mut proof, mut aux) = setup(AnchorOpeningPolicy::Dense);
    update(&params, &instance, &mut proof, &mut aux, 1);
    let fail = Arc::new(AtomicBool::new(true)); let created = Arc::new(AtomicUsize::new(0));
    let dropped = Arc::new(AtomicUsize::new(0));
    let factory_fail = fail.clone(); let factory_created = created.clone(); let factory_dropped = dropped.clone();
    let mut worker = BackgroundRebuilder::new(params.clone(), move || {
        factory_created.fetch_add(1, Ordering::SeqCst);
        let mut backend = LocalOracle::new(); backend.fail = factory_fail.load(Ordering::SeqCst);
        backend.dropped = Some(factory_dropped.clone()); Ok(backend)
    }).unwrap();
    assert!(worker.start(&aux).unwrap()); let before = aux.clone(); let before_proof = proof.clone();
    assert!(poll(&mut worker, &mut proof, &mut aux).is_err());
    assert_eq!(aux, before); assert_eq!(proof, before_proof); assert_eq!(dropped.load(Ordering::SeqCst), 1);
    fail.store(false, Ordering::SeqCst); assert!(worker.start(&aux).unwrap());
    assert!(matches!(poll(&mut worker, &mut proof, &mut aux).unwrap(), SwapOutcome::Swapped { .. }));
    for _ in 0..2 { update(&params, &instance, &mut proof, &mut aux, 1);
        assert!(worker.start(&aux).unwrap()); assert!(matches!(poll(&mut worker, &mut proof, &mut aux).unwrap(), SwapOutcome::Swapped { .. })); }
    assert_eq!(created.load(Ordering::SeqCst), 2, "healthy worker backend must be reused");
    drop(worker); assert_eq!(dropped.load(Ordering::SeqCst), 2, "Drop must join before returning");
}

#[test]
fn failed_rebase_and_stale_proof_leave_live_state_intact() {
    let (params, instance, mut proof, mut aux) = setup(AnchorOpeningPolicy::Dense);
    update(&params, &instance, &mut proof, &mut aux, 1);
    let mut worker = BackgroundRebuilder::new(params.clone(), || Ok(LocalOracle::new())).unwrap();
    assert!(worker.start(&aux).unwrap()); update(&params, &instance, &mut proof, &mut aux, 1);
    let before = aux.clone(); let before_proof = proof.clone(); let mut broken = LocalOracle::new(); broken.fail = true;
    let deadline = Instant::now() + Duration::from_secs(15);
    loop { match worker.try_swap(&mut proof, &mut aux, Some(&broken)) {
        Ok(SwapOutcome::Building { .. }) => { assert!(Instant::now() < deadline); thread::sleep(Duration::from_millis(1)); }
        Err(_) => break, other => panic!("expected failed rebase, got {other:?}") } }
    assert_eq!(aux, before); assert_eq!(proof, before_proof);
    assert!(worker.start(&aux).unwrap()); let mut stale = proof.clone(); stale.generation = 0;
    let original_stale = stale.clone(); assert!(poll(&mut worker, &mut stale, &mut aux).is_err());
    assert_eq!(aux, before); assert_eq!(stale, original_stale);
}

#[test]
fn prefetch_is_due_only_after_progress_and_worker_panic_is_explicit() {
    let (params, instance, mut proof, mut aux) = setup(AnchorOpeningPolicy::Dense);
    let mut worker = BackgroundRebuilder::new(params.clone(), || Ok(LocalOracle::new())).unwrap();
    assert!(!worker.start_if_due(&aux).unwrap()); update(&params, &instance, &mut proof, &mut aux, 1);
    assert!(worker.start_if_due(&aux).unwrap()); poll(&mut worker, &mut proof, &mut aux).unwrap();
    update(&params, &instance, &mut proof, &mut aux, 1);
    let mut panicking = BackgroundRebuilder::new(params, || -> Result<LocalOracle> { panic!("injected worker panic") }).unwrap();
    assert!(panicking.start(&aux).unwrap()); let before = aux.clone(); let before_proof = proof.clone();
    assert!(poll(&mut panicking, &mut proof, &mut aux).is_err()); assert_eq!(aux, before); assert_eq!(proof, before_proof);
}

#[test]
fn corrupted_private_cache_is_rejected_before_publication() {
    let (params, instance, mut proof, mut aux) = setup(AnchorOpeningPolicy::Precompute);
    update(&params, &instance, &mut proof, &mut aux, 1);
    let mut worker = BackgroundRebuilder::new(params, || {
        let mut backend = LocalOracle::new(); backend.corrupt_cache = true; Ok(backend)
    }).unwrap();
    assert!(worker.start(&aux).unwrap()); let before = aux.clone(); let before_proof = proof.clone();
    let error = poll(&mut worker, &mut proof, &mut aux).unwrap_err();
    assert!(error.to_string().contains("cache differs from CPU"), "{error:#}");
    assert_eq!(aux, before); assert_eq!(proof, before_proof);
}

#[test]
fn candidate_far_from_current_witness_is_discarded_even_without_foreground_rebuild() {
    let (params, instance, mut proof, mut aux) = setup_at(64, AnchorOpeningPolicy::Dense);
    update(&params, &instance, &mut proof, &mut aux, 1);
    let original_value = aux.anchor_witness().wires[4 * 64 + 1];
    let (release_tx, release_rx) = mpsc::sync_channel(1);
    let mut worker = BackgroundRebuilder::new(params.clone(), move || {
        release_rx.recv_timeout(Duration::from_secs(15)).unwrap(); Ok(LocalOracle::new())
    }).unwrap();
    assert!(worker.start(&aux).unwrap());
    let old = aux.current_witness().clone();
    let revert = Update { indices: vec![4 * 64 + 1, 5 * 64 + 1],
        values: vec![original_value, old.wires[3 * 64 + 1] * original_value] };
    proof = params.update_proof(&instance, &instance, &old, &revert, &proof, &mut aux).unwrap();
    for gate in [2, 3, 4] { update(&params, &instance, &mut proof, &mut aux, gate); }
    assert_eq!(proof.anchor_generation, 0);
    let before = aux.clone(); let before_proof = proof.clone(); release_tx.send(()).unwrap();
    assert_eq!(poll(&mut worker, &mut proof, &mut aux).unwrap(), SwapOutcome::Discarded { source_generation: 1 });
    assert_eq!(aux, before); assert_eq!(proof, before_proof);
}

#[test]
fn drop_finishes_pending_job_and_factory_failure_can_be_retried() {
    let (params, instance, mut proof, mut aux) = setup(AnchorOpeningPolicy::Dense);
    update(&params, &instance, &mut proof, &mut aux, 1);
    let fail = Arc::new(AtomicBool::new(true)); let factory_fail = fail.clone();
    let dropped = Arc::new(AtomicUsize::new(0)); let factory_dropped = dropped.clone();
    let mut worker = BackgroundRebuilder::new(params, move || {
        ensure!(!factory_fail.load(Ordering::SeqCst), "injected factory error");
        let mut backend = LocalOracle::new(); backend.dropped = Some(factory_dropped.clone()); Ok(backend)
    }).unwrap();
    assert!(worker.start(&aux).unwrap()); let before = aux.clone();
    assert!(poll(&mut worker, &mut proof, &mut aux).is_err()); assert_eq!(aux, before);
    fail.store(false, Ordering::SeqCst); assert!(worker.start(&aux).unwrap());
    drop(worker); assert_eq!(dropped.load(Ordering::SeqCst), 1);
}

#[test]
fn reverting_to_older_snapshot_keeps_zero_delta_for_generation_gap() {
    let (params, instance, mut proof, mut aux) = setup(AnchorOpeningPolicy::Precompute);
    update(&params, &instance, &mut proof, &mut aux, 1);
    let snapshot = aux.current_witness().clone();
    let (release_tx, release_rx) = mpsc::sync_channel(1);
    let mut worker = BackgroundRebuilder::new(params.clone(), move || {
        release_rx.recv_timeout(Duration::from_secs(15)).unwrap(); Ok(LocalOracle::new())
    }).unwrap();
    assert!(worker.start(&aux).unwrap());
    update(&params, &instance, &mut proof, &mut aux, 1);
    let old = aux.current_witness().clone();
    let revert = Update { indices: vec![65, 81], values: vec![snapshot.wires[65], snapshot.wires[81]] };
    proof = params.update_proof(&instance, &instance, &old, &revert, &proof, &mut aux).unwrap();
    assert_eq!(*aux.current_witness(), snapshot);
    assert_eq!(proof.generation, 3);
    release_tx.send(()).unwrap();
    assert_eq!(poll(&mut worker, &mut proof, &mut aux).unwrap(),
        SwapOutcome::Swapped { anchor_generation: 1, current_generation: 3 });
    assert!(proof.delta.is_some(), "generation gap requires a zero delta proof after witness reversion");
    assert!(params.verifying_key().verify_reference(&instance, &proof).unwrap());
    assert_eq!(*aux.current_witness(), snapshot);
}
