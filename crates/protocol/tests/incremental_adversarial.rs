//! Caller-paid adversarial oracles: snapshots and dense reference rebuilding
//! happen outside the update whose bounded work is being checked.
use anyhow::{bail, Result};
use ark_bn254::{Fr, G1Affine, G2Affine};
use ark_ff::One;
use dynaverse_protocol::{
    dynaverse::{DynaverseParams, IncrementalSession},
    incremental_rebuild::{
        operators::{BufferId, BufferKind, CpuRebuildBackend, FieldBinaryOp, RebuildBackend, RebuildStats},
        IncrementalConfig, RebuildProgress, UpdateReceipt,
    },
    toy_circuit::ToyCircuit,
    BatchMsmRequest, DynaverseProof, Instance, OperatorBackend, Update, Witness,
};
use rand::{rngs::StdRng, SeedableRng};
use std::{cell::Cell, collections::BTreeMap, rc::Rc};

type History = BTreeMap<u64, (Instance, Witness)>;

fn fixture(n: usize, seed: u64) -> (DynaverseParams, Instance, Witness) {
    let circuit = ToyCircuit::new(n).unwrap();
    let (instance, witness) = circuit.valid_instance();
    let params = DynaverseParams::setup(circuit, &mut StdRng::seed_from_u64(seed)).unwrap();
    (params, instance, witness)
}

fn empty() -> Update { Update { indices: Vec::new(), values: Vec::new() } }

fn multiplier_update(witness: &Witness, n: usize, gate: usize) -> Update {
    let next = witness.wires[4 * n + gate] + Fr::one();
    let mut update = Update { indices: vec![4 * n + gate], values: vec![next] };
    if gate != 0 {
        update.indices.push(5 * n + gate);
        update.values.push(witness.wires[3 * n + gate] * next);
    }
    update
}

// Gate zero has zero sum, so one multiplier edit is valid. Every other gate
// contributes exactly two actual changes, independently of the session's k.
fn exact_changes(witness: &Witness, n: usize, k: usize) -> Update {
    let mut update = empty();
    if k % 2 == 1 {
        let one = multiplier_update(witness, n, 0);
        update.indices.extend(one.indices);
        update.values.extend(one.values);
    }
    for gate in 1..=k / 2 {
        assert!(gate < n, "fixture cannot provide {k} distinct valid changes");
        let two = multiplier_update(witness, n, gate);
        update.indices.extend(two.indices);
        update.values.extend(two.values);
    }
    assert_eq!(update.indices.len(), k);
    update
}

fn public_preserving_sum(witness: &Witness, instance: &Instance, n: usize) -> (Update, Update) {
    let left = witness.wires[0] + Fr::one();
    let right = witness.wires[n] - Fr::one();
    (
        Update { indices: vec![0, n], values: vec![left, right] },
        Update { indices: vec![0], values: vec![instance.public_inputs[0] + Fr::one()] },
    )
}

fn history(instance: &Instance, witness: &Witness) -> History {
    BTreeMap::from([(0, (instance.clone(), witness.clone()))])
}

fn assert_committed<B: RebuildBackend>(
    params: &DynaverseParams, session: &IncrementalSession<'_, B>, history: &History, expected_k: usize,
) {
    let vk = params.verifying_key();
    assert!(vk.verify(session.current_instance(), session.proof()).unwrap());
    assert!(vk.verify_reference(session.current_instance(), session.proof()).unwrap());
    let receipt = session.last_receipt();
    assert_eq!(receipt.changed, expected_k);
    assert_eq!(receipt.allowance, session.budget().q * expected_k as u128);
    assert!(receipt.rebuild_units <= receipt.allowance);
    assert_eq!(receipt.work.units().unwrap(), receipt.rebuild_units);
    assert!(session.anchor_distance() <= 2 * session.window());
    assert_eq!(receipt.large_batch, expected_k > session.horizon());
    if let Some(progress) = session.rebuild_progress() {
        assert!(progress.paid_changes < session.horizon(), "overdue job {progress:?}");
        assert!(progress.command < progress.commands);
        if receipt.restarted { assert_eq!(progress.paid_changes, 0); }
    }
    if receipt.swapped {
        let generation = session.proof().anchor_generation;
        let (frozen_instance, frozen_witness) = history.get(&generation)
            .unwrap_or_else(|| panic!("unknown frozen generation {generation}"));
        let reference = params.prove_anchor(frozen_instance, frozen_witness).unwrap();
        assert_eq!(session.proof().anchor, reference.anchor,
            "installed anchor differs from frozen generation {generation}");
    }
}

fn apply<B: RebuildBackend>(
    params: &DynaverseParams, session: &mut IncrementalSession<'_, B>, history: &mut History,
    wires: Update, public: Update, expected_k: usize,
) {
    let before = session.proof().generation;
    let mut expected_witness = session.current_witness().clone();
    let mut expected_instance = session.current_instance().clone();
    // Direct ordered assignment is an independent last-wins normalization oracle.
    for (&i, &value) in wires.indices.iter().zip(&wires.values) { expected_witness.wires[i] = value; }
    for (&i, &value) in public.indices.iter().zip(&public.values) { expected_instance.public_inputs[i] = value; }
    let proof = session.update_with_public_inputs(&wires, &public).unwrap();
    assert_eq!(&proof, session.proof());
    assert_eq!(session.current_witness(), &expected_witness);
    assert_eq!(session.current_instance(), &expected_instance);
    assert_eq!(proof.generation, before + u64::from(expected_k != 0));
    history.insert(proof.generation, (expected_instance, expected_witness));
    assert_committed(params, session, history, expected_k);
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VisibleState {
    proof: DynaverseProof,
    witness: Witness,
    instance: Instance,
    progress: Option<RebuildProgress>,
    receipt: UpdateReceipt,
    distance: usize,
}

fn visible<B: RebuildBackend>(session: &IncrementalSession<'_, B>) -> VisibleState {
    VisibleState {
        proof: session.proof().clone(), witness: session.current_witness().clone(),
        instance: session.current_instance().clone(), progress: session.rebuild_progress(),
        receipt: *session.last_receipt(), distance: session.anchor_distance(),
    }
}

#[test]
fn tiny_and_normal_sizes_preserve_hot_rewrite_and_frozen_anchor_invariants() {
    for n in [1, 2, 16, 64] {
        let (params, instance, witness) = fixture(n, 6200 + n as u64);
        for optimized_window in [false, true] {
            let config = IncrementalConfig { optimized_window, ..IncrementalConfig::default() };
            let mut session = params.start_incremental_session(&instance, &witness, config, CpuRebuildBackend::default()).unwrap();
            let mut states = history(&instance, &witness);
            let mut swaps = 0;
            if n > 1 {
                for gate in 1..=session.horizon().min(n - 1) {
                    let update = multiplier_update(session.current_witness(), n, gate);
                    apply(&params, &mut session, &mut states, update, empty(), 2);
                    swaps += usize::from(session.last_receipt().swapped);
                }
            }
            for _ in 0..2 * session.horizon() + 3 {
                let update = multiplier_update(session.current_witness(), n, 0);
                apply(&params, &mut session, &mut states, update, empty(), 1);
                swaps += usize::from(session.last_receipt().swapped);
            }
            assert!(swaps > 0, "n={n}, optimized={optimized_window}");
        }
    }
}

#[test]
fn exact_horizon_batches_and_large_batches_are_explicitly_paid() {
    for n in [1, 2, 16, 64] {
        let (params, instance, witness) = fixture(n, 6300 + n as u64);
        let mut session = params.start_incremental_session(&instance, &witness,
            IncrementalConfig::default(), CpuRebuildBackend::default()).unwrap();
        let mut states = history(&instance, &witness);
        for k in [session.horizon() - 1, session.horizon()] {
            let update = exact_changes(session.current_witness(), n, k);
            apply(&params, &mut session, &mut states, update, empty(), k);
        }
        if n == 1 {
            let (wires, public) = public_preserving_sum(session.current_witness(), session.current_instance(), n);
            apply(&params, &mut session, &mut states, wires, public, 3);
        } else {
            let k = session.horizon() + 1;
            let update = exact_changes(session.current_witness(), n, k);
            apply(&params, &mut session, &mut states, update, empty(), k);
        }
        assert!(session.last_receipt().large_batch);
        assert!(session.last_receipt().swapped);
        assert!(session.rebuild_progress().is_none());
        assert_eq!(session.anchor_distance(), 0);
    }
}

#[test]
fn duplicate_noops_reverts_and_public_namespaces_count_actual_changes() {
    let n = 64;
    let (params, instance, witness) = fixture(n, 6401);
    let mut session = params.start_incremental_session(&instance, &witness,
        IncrementalConfig::default(), CpuRebuildBackend::default()).unwrap();
    let mut states = history(&instance, &witness);
    let hot = 4 * n;
    let original = session.current_witness().wires[hot];
    let before = visible(&session);
    let duplicates = Update { indices: vec![hot, hot, hot], values: vec![original + Fr::from(2u64), original + Fr::one(), original] };
    apply(&params, &mut session, &mut states, duplicates, empty(), 0);
    assert_eq!(session.rebuild_progress(), before.progress);
    assert_eq!(session.proof(), &before.proof);
    let duplicates = Update { indices: vec![hot, hot, hot], values: vec![original, original + Fr::from(2u64), original + Fr::one()] };
    apply(&params, &mut session, &mut states, duplicates, empty(), 1);
    apply(&params, &mut session, &mut states,
        Update { indices: vec![hot], values: vec![original] }, empty(), 1);
    assert_eq!(session.anchor_distance(), 0);
    let (wire, mut public) = public_preserving_sum(session.current_witness(), session.current_instance(), n);
    public.indices.insert(0, 0);
    public.values.insert(0, instance.public_inputs[0] + Fr::from(19u64));
    apply(&params, &mut session, &mut states, wire, public, 3);
    let progress = session.rebuild_progress();
    let proof = session.proof().clone();
    let noop = Update { indices: vec![0, 0], values: vec![instance.public_inputs[0], session.current_instance().public_inputs[0]] };
    apply(&params, &mut session, &mut states, empty(), noop, 0);
    assert_eq!(session.rebuild_progress(), progress);
    assert_eq!(session.proof(), &proof);
}

#[test]
fn invalid_calls_preserve_all_published_state_and_rebuild_credit() {
    let n = 16;
    let (params, instance, witness) = fixture(n, 6501);
    let mut session = params.start_incremental_session(&instance, &witness,
        IncrementalConfig::default(), CpuRebuildBackend::default()).unwrap();
    let mut states = history(&instance, &witness);
    let first = multiplier_update(session.current_witness(), n, 0);
    apply(&params, &mut session, &mut states, first, empty(), 1);
    let snapshot = visible(&session);
    let cases = [
        (Update { indices: vec![6 * n], values: vec![Fr::one()] }, empty()),
        (Update { indices: vec![4 * n], values: vec![] }, empty()),
        (Update { indices: vec![1], values: vec![Fr::from(99u64)] }, empty()),
        (empty(), Update { indices: vec![n], values: vec![Fr::one()] }),
        (empty(), Update { indices: vec![0], values: vec![Fr::from(99u64)] }),
    ];
    for (wires, public) in cases {
        assert!(session.update_with_public_inputs(&wires, &public).is_err());
        assert_eq!(visible(&session), snapshot);
        assert!(!session.is_faulted());
    }
    let next = multiplier_update(session.current_witness(), n, 1);
    apply(&params, &mut session, &mut states, next, empty(), 2);
}

#[test]
fn unsupported_resources_and_layout_are_rejected_at_initialization() {
    let (params, instance, witness) = fixture(16, 6601);
    let mut configs = Vec::new();
    configs.push(IncrementalConfig { memory_limit_bytes: 1, ..IncrementalConfig::default() });
    for window in [0, 1, 5, usize::MAX] {
        configs.push(IncrementalConfig { window: Some(window), ..IncrementalConfig::default() });
    }
    for tile_len in [0, 1, usize::MAX] {
        configs.push(IncrementalConfig { tile_len: Some(tile_len), ..IncrementalConfig::default() });
    }
    for config in configs {
        assert!(params.start_incremental_session(&instance, &witness, config,
            CpuRebuildBackend::default()).is_err());
    }
}

#[test]
fn different_windows_and_tile_schedules_publish_equivalent_live_statements() {
    let n = 64;
    let (params, instance, witness) = fixture(n, 6701);
    let mut slow = params.start_incremental_session(&instance, &witness,
        IncrementalConfig::default(), CpuRebuildBackend::default()).unwrap();
    let mut fast = params.start_incremental_session(&instance, &witness,
        IncrementalConfig { optimized_window: true, tile_len: Some(32), ..IncrementalConfig::default() },
        CpuRebuildBackend::default()).unwrap();
    assert_ne!(slow.window(), fast.window());
    let mut slow_history = history(&instance, &witness);
    let mut fast_history = history(&instance, &witness);
    let mut different_anchor = false;
    for gate in [1, 2, 3, 0, 1, 0] {
        let update = multiplier_update(slow.current_witness(), n, gate);
        let k = if gate == 0 { 1 } else { 2 };
        apply(&params, &mut slow, &mut slow_history, update.clone(), empty(), k);
        apply(&params, &mut fast, &mut fast_history, update, empty(), k);
        assert_eq!(slow.current_witness(), fast.current_witness());
        assert_eq!(slow.current_instance(), fast.current_instance());
        different_anchor |= slow.proof().anchor_generation != fast.proof().anchor_generation;
    }
    assert!(different_anchor);
}

#[derive(Default)]
struct FaultControl {
    fail_at: Cell<Option<usize>>,
    resident_calls: Cell<usize>,
    foreground_calls: Cell<usize>,
    rollback_fatal: Cell<bool>,
    corrupt_cache: Cell<bool>,
}

impl FaultControl {
    fn reset(&self) {
        self.fail_at.set(None);
        self.resident_calls.set(0);
        self.foreground_calls.set(0);
        self.rollback_fatal.set(false);
    }

    fn arm(&self, fail_at: usize, fatal_rollback: bool) {
        self.reset();
        self.fail_at.set(Some(fail_at));
        self.rollback_fatal.set(fatal_rollback);
    }
}

struct FaultBackend {
    cpu: CpuRebuildBackend,
    control: Rc<FaultControl>,
}

impl FaultBackend {
    fn before_resident_operation(&self) -> Result<()> {
        let call = self.control.resident_calls.get();
        self.control.resident_calls.set(call + 1);
        if self.control.fail_at.get() == Some(call) {
            self.control.fail_at.set(None);
            bail!("injected resident operation failure at {call}");
        }
        Ok(())
    }

    fn record_foreground(&self) {
        self.control.foreground_calls.set(self.control.foreground_calls.get() + 1);
    }
}

impl OperatorBackend for FaultBackend {
    fn msm(&self, points: &[G1Affine], scalars: &[Fr]) -> Result<G1Affine> {
        self.record_foreground();
        self.cpu.msm(points, scalars)
    }

    fn msm_batch(&self, request: &BatchMsmRequest<'_>) -> Result<Vec<G1Affine>> {
        self.record_foreground();
        self.cpu.msm_batch(request)
    }

    fn ntt(&self, values: &mut [Fr], inverse: bool) -> Result<()> {
        self.record_foreground();
        self.cpu.ntt(values, inverse)
    }
}

// The macro only forwards typed trait arguments. Each resident call performs
// real CPU arithmetic unless this test's external controller injects a failure.
macro_rules! resident_forward {
    ($(fn $name:ident($($arg:ident: $ty:ty),*) -> $result:ty;)*) => {
        $(fn $name(&mut self, $($arg: $ty),*) -> $result {
            self.before_resident_operation()?;
            self.cpu.$name($($arg),*)
        })*
    };
}

impl RebuildBackend for FaultBackend {
    fn alloc(&mut self, kind: BufferKind, len: usize, writable: bool) -> Result<BufferId> {
        self.cpu.alloc(kind, len, writable)
    }
    fn seal(&mut self) { self.cpu.seal(); }
    fn buffer_len(&self, id: BufferId) -> Result<usize> { self.cpu.buffer_len(id) }
    fn buffer_kind(&self, id: BufferId) -> Result<BufferKind> { self.cpu.buffer_kind(id) }
    fn check_transaction_capacity(&self, required: u64) -> Result<()> {
        self.cpu.check_transaction_capacity(required)
    }
    fn begin_transaction(&mut self) -> Result<()> { self.cpu.begin_transaction() }
    fn commit_transaction(&mut self) { self.cpu.commit_transaction(); }
    fn rollback_transaction(&mut self) -> Result<()> {
        if self.control.rollback_fatal.get() { bail!("injected fatal rollback failure"); }
        self.cpu.rollback_transaction()
    }
    fn stats(&self) -> RebuildStats { self.cpu.stats() }

    fn read_points(&mut self,id:BufferId,offset:usize,values:&mut [G1Affine])->Result<()> {
        self.before_resident_operation()?;
        self.cpu.read_points(id,offset,values)?;
        if self.control.corrupt_cache.get() && !values.is_empty() {
            use ark_ec::{CurveGroup,Group};
            values[0]=(ark_bn254::G1Projective::from(values[0])
                +ark_bn254::G1Projective::generator()).into_affine();
        }
        Ok(())
    }

    resident_forward! {
        fn write_fields(id: BufferId, offset: usize, values: &[Fr]) -> Result<()>;
        fn read_fields(id: BufferId, offset: usize, values: &mut [Fr]) -> Result<()>;
        fn write_points(id: BufferId, offset: usize, values: &[G1Affine]) -> Result<()>;
        fn copy_buffer(src: BufferId, src_offset: usize, dst: BufferId, dst_offset: usize, len: usize) -> Result<()>;
        fn reverse_copy(kind: BufferKind, src: BufferId, src_offset: usize, dst: BufferId, dst_offset: usize, len: usize) -> Result<()>;
        fn transpose(kind: BufferKind, src: BufferId, dst: BufferId, rows: usize, cols: usize, start: usize, len: usize, twiddle: Option<Fr>) -> Result<()>;
        fn transform_rows(kind: BufferKind, src: BufferId, dst: BufferId, row_len: usize, first_row: usize, row_count: usize, inverse: bool) -> Result<()>;
        fn field_zip(op: FieldBinaryOp, left: BufferId, left_offset: usize, right: BufferId, right_offset: usize, dst: BufferId, dst_offset: usize, len: usize) -> Result<()>;
        fn scale_points(points: BufferId, point_offset: usize, scalars: BufferId, scalar_offset: usize, dst: BufferId, dst_offset: usize, len: usize) -> Result<()>;
        fn msm_g1(points: BufferId, point_offset: usize, scalars: BufferId, scalar_offset: usize, len: usize) -> Result<G1Affine>;
        fn msm_g2(points: &[G2Affine], scalars: BufferId, scalar_offset: usize) -> Result<G2Affine>;
    }
}

#[test]
fn initialization_rejects_corrupt_cache_and_bypasses_legacy_host_operators() {
    let (params,instance,witness)=fixture(16,6901);
    let control=Rc::new(FaultControl::default());
    let backend=FaultBackend {cpu:CpuRebuildBackend::default(),control:control.clone()};
    let session=params.start_incremental_session(&instance,&witness,
        IncrementalConfig::default(),backend).unwrap();
    assert_eq!(control.foreground_calls.get(),0,"initial proof called legacy host operators");
    assert!(params.verifying_key().verify(&instance,session.proof()).unwrap());
    drop(session);
    control.corrupt_cache.set(true);
    let backend=FaultBackend {cpu:CpuRebuildBackend::default(),control};
    let error=params.start_incremental_session(&instance,&witness,
        IncrementalConfig::default(),backend).err().unwrap();
    assert!(error.to_string().contains("opening cache check failed"));
}

fn pending_session<'p>(
    params: &'p DynaverseParams, instance: &Instance, witness: &Witness,
) -> (IncrementalSession<'p, FaultBackend>, History, Rc<FaultControl>) {
    let control = Rc::new(FaultControl::default());
    let backend = FaultBackend { cpu: CpuRebuildBackend::default(), control: control.clone() };
    let mut session = params.start_incremental_session(instance, witness,
        IncrementalConfig { window: Some(8), ..IncrementalConfig::default() }, backend).unwrap();
    let mut states = history(instance, witness);
    for gate in [1, 2] {
        let update = multiplier_update(session.current_witness(), 64, gate);
        apply(params, &mut session, &mut states, update, empty(), 2);
    }
    assert!(session.rebuild_progress().is_some(), "fault fixture needs an already active rebuild");
    control.reset();
    (session, states, control)
}

#[test]
fn first_middle_and_last_rebuild_failures_roll_back_and_retry_the_same_checkpoint() {
    let n = 64;
    let (params, instance, witness) = fixture(n, 6801);
    // Count this exact call's real resident operations before selecting failure
    // positions. The calibration is caller-paid and uses an independent owner.
    let (mut baseline, mut baseline_states, baseline_control) = pending_session(&params, &instance, &witness);
    let update = multiplier_update(baseline.current_witness(), n, 0);
    apply(&params, &mut baseline, &mut baseline_states, update, empty(), 1);
    let operations = baseline_control.resident_calls.get();
    assert!(operations >= 3);
    for position in [0, operations / 2, operations - 1] {
        let (mut session, mut states, control) = pending_session(&params, &instance, &witness);
        let checkpoint = visible(&session);
        let update = multiplier_update(session.current_witness(), n, 0);
        control.arm(position, false);
        let error = session.update(&update).unwrap_err();
        assert!(error.to_string().contains("injected resident operation failure"), "{error:#}");
        assert!(control.foreground_calls.get() > 0, "foreground proof must precede injected rebuild failure");
        assert_eq!(control.resident_calls.get(), position + 1);
        assert_eq!(visible(&session), checkpoint, "failed call at resident operation {position}");
        assert!(!session.is_faulted());
        control.reset();
        apply(&params, &mut session, &mut states, update, empty(), 1);
        assert_eq!(visible(&session), visible(&baseline), "retry differs after failure position {position}");
        // Continue through installation, checking that the candidate/cache
        // checkpoint restored by rollback remains valid for future calls.
        for _ in 0..session.horizon() {
            let update = multiplier_update(session.current_witness(), n, 0);
            apply(&params, &mut session, &mut states, update, empty(), 1);
        }
    }
}

#[test]
fn fatal_rollback_poisoning_retains_logical_state_and_rejects_further_updates() {
    let n = 64;
    let (params, instance, witness) = fixture(n, 6901);
    let (mut session, _states, control) = pending_session(&params, &instance, &witness);
    let checkpoint = visible(&session);
    let update = multiplier_update(session.current_witness(), n, 0);
    control.arm(2, true);
    let error = session.update(&update).unwrap_err();
    assert!(format!("{error:#}").contains("session faulted"));
    assert!(session.is_faulted());
    assert_eq!(visible(&session), checkpoint);
    let calls_after_failure = control.resident_calls.get();
    assert!(session.update(&update).is_err());
    assert!(session.update(&empty()).is_err());
    assert_eq!(control.resident_calls.get(), calls_after_failure);
    assert_eq!(visible(&session), checkpoint);
    let vk = params.verifying_key();
    assert!(vk.verify(session.current_instance(), session.proof()).unwrap());
    assert!(vk.verify_reference(session.current_instance(), session.proof()).unwrap());
}

#[test]
fn pending_noop_spends_no_credit_and_public_edits_preserve_frozen_snapshot_identity() {
    let n = 64;
    let (params, instance, witness) = fixture(n, 7001);
    let (mut session, mut states, control) = pending_session(&params, &instance, &witness);
    let frozen = session.rebuild_progress().unwrap();
    let value = session.current_witness().wires[4 * n + 1];
    let duplicates = Update { indices: vec![4 * n + 1, 4 * n + 1], values: vec![value + Fr::one(), value] };
    let public = Update { indices: vec![0], values: vec![session.current_instance().public_inputs[0]] };
    apply(&params, &mut session, &mut states, duplicates, public, 0);
    assert_eq!(session.rebuild_progress(), Some(frozen));
    assert_eq!(control.resident_calls.get(), 0);
    assert_eq!(control.foreground_calls.get(), 0);
    assert_eq!(session.last_receipt().blocks, 0);

    let (wires, public) = public_preserving_sum(session.current_witness(), session.current_instance(), n);
    apply(&params, &mut session, &mut states, wires, public, 3);
    assert!(session.last_receipt().swapped);
    assert!(!session.last_receipt().large_batch);
    assert_eq!(session.proof().anchor_generation, frozen.source_generation);
    assert!(session.proof().generation > session.proof().anchor_generation);
    assert_ne!(session.current_instance(), &states[&frozen.source_generation].0);
}

#[test]
fn paid_large_batch_replaces_an_unfinished_candidate_with_its_final_legal_state() {
    let n = 64;
    let (params, instance, witness) = fixture(n, 7101);
    let (mut session, mut states, _control) = pending_session(&params, &instance, &witness);
    let abandoned_generation = session.rebuild_progress().unwrap().source_generation;
    let k = session.horizon() + 1;
    let update = exact_changes(session.current_witness(), n, k);
    apply(&params, &mut session, &mut states, update, empty(), k);
    let receipt = session.last_receipt();
    assert!(receipt.large_batch && receipt.started && receipt.swapped);
    assert!(!receipt.restarted);
    assert!(session.proof().anchor_generation > abandoned_generation);
    assert_eq!(session.proof().anchor_generation, session.proof().generation);
    assert_eq!(session.anchor_distance(), 0);
    assert!(session.rebuild_progress().is_none());
}

#[test]
fn switching_with_horizon_overlay_immediately_restarts_without_reusing_credit() {
    let n = 64;
    let (params, instance, witness) = fixture(n, 7201);
    let (mut session, mut states, control) = pending_session(&params, &instance, &witness);
    let old_source = session.rebuild_progress().unwrap().source_generation;
    let k = session.horizon();
    let update = exact_changes(session.current_witness(), n, k);
    apply(&params, &mut session, &mut states, update, empty(), k);
    assert!(session.last_receipt().swapped);
    assert!(session.last_receipt().restarted);
    assert!(!session.last_receipt().large_batch);
    assert_eq!(session.proof().anchor_generation, old_source);
    let restarted = session.rebuild_progress().unwrap();
    assert_eq!(restarted.source_generation, session.proof().generation);
    assert_eq!(restarted.paid_changes, 0);
    assert_eq!(restarted.command, 0);
    assert_eq!(session.anchor_distance(), session.horizon());
    control.reset();
    apply(&params, &mut session, &mut states, empty(), empty(), 0);
    assert_eq!(session.rebuild_progress(), Some(restarted));
    assert_eq!(control.resident_calls.get(), 0);
    for _ in 0..session.horizon() {
        let update = multiplier_update(session.current_witness(), n, 0);
        apply(&params, &mut session, &mut states, update, empty(), 1);
    }
    assert_eq!(session.proof().anchor_generation, restarted.source_generation);
    assert!(session.rebuild_progress().is_none());
}
