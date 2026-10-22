#![cfg(feature = "icicle")]
use dynaverse_gpu::IcicleBackend;
use anyhow::{ensure, Result};
use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::CurveGroup;
use dynaverse_gpu::BatchStats;
use dynaverse_protocol::background::{BackgroundRebuilder, SwapOutcome};
use dynaverse_protocol::dynaverse::DynaverseParams;
use dynaverse_protocol::toy_circuit::ToyCircuit;
use dynaverse_protocol::trace::CpuTrace;
use dynaverse_protocol::{AnchorOpeningPolicy, BatchMsmRequest, BatchOpenMode,
    DynaverseProof, Instance, OperatorBackend, ProverAux, Update, VerificationMode};
use rand::{rngs::StdRng, SeedableRng};
use std::{fs::File, io::Write, sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}},
    thread, time::{Duration, Instant}};

#[derive(Clone, Copy, Default)]
struct Observation { batch: BatchStats, created: usize, dropped: usize }
struct WorkerGpu { inner: IcicleBackend, observation: Arc<Mutex<Observation>>, fail: Arc<AtomicBool> }
impl WorkerGpu {
    fn check(&self) -> Result<()> {
        ensure!(!self.fail.swap(false, Ordering::SeqCst), "injected background GPU worker failure"); Ok(())
    }
}
impl OperatorBackend for WorkerGpu {
    fn msm(&self, points: &[G1Affine], scalars: &[Fr]) -> Result<G1Affine> {
        self.check()?; self.inner.msm(points, scalars)
    }
    fn msm_batch(&self, request: &BatchMsmRequest<'_>) -> Result<Vec<G1Affine>> {
        self.check()?; let result = self.inner.msm_batch(request)?;
        self.observation.lock().unwrap().batch = self.inner.batch_stats()?; Ok(result)
    }
    fn ntt(&self, values: &mut [Fr], inverse: bool) -> Result<()> {
        self.check()?; self.inner.ntt(values, inverse)
    }
}
impl Drop for WorkerGpu { fn drop(&mut self) { self.observation.lock().unwrap().dropped += 1; } }

fn make_worker(params: Arc<DynaverseParams>, observation: Arc<Mutex<Observation>>, fail: Arc<AtomicBool>) -> BackgroundRebuilder {
    BackgroundRebuilder::new(params, move || {
        let inner = IcicleBackend::with_large_bucket_factor(10)?;
        observation.lock().unwrap().created += 1;
        Ok(WorkerGpu { inner, observation: observation.clone(), fail: fail.clone() })
    }).unwrap()
}
fn update(params: &DynaverseParams, instance: &Instance, proof: &mut DynaverseProof,
    aux: &mut ProverAux, gpu: &IcicleBackend, step: usize) {
    let n = instance.public_inputs.len(); let old = aux.current_witness().clone();
    // Long hotspots permit actual background swaps; spreading forces genuine
    // foreground rebuilds. Periodic reversions test true distance, not edit count.
    let mut gates = Vec::new();
    if step % 100 == 0 {
        // Four fresh gates change eight wires at n=64, forcing the original
        // synchronous rebuild even when background prefetch keeps up perfectly.
        for gate in 1..n {
            if old.wires[4 * n + gate] == aux.anchor_witness().wires[4 * n + gate]
                && old.wires[5 * n + gate] == aux.anchor_witness().wires[5 * n + gate] {
                gates.push(gate); if gates.len() == 4 { break; }
            }
        }
        assert_eq!(gates.len(), 4);
    } else { gates.push(if step % 32 < 16 { 1 } else { 1 + step % (n - 1) }); }
    let mut indices = Vec::new(); let mut values = Vec::new();
    for gate in gates {
        let mut value = if step % 13 == 0 { aux.anchor_witness().wires[4 * n + gate] }
            else { old.wires[4 * n + gate] + Fr::from(1u64) };
        // Every soak step changes state; a requested reversion of an already
        // unchanged gate becomes a fresh edit rather than inflating the count.
        if value == old.wires[4 * n + gate] { value += Fr::from(1u64); }
        indices.extend_from_slice(&[4 * n + gate, 5 * n + gate]);
        values.extend_from_slice(&[value, old.wires[3 * n + gate] * value]);
    }
    let edit = Update { indices, values };
    *proof = params.update_proof_traced_with_backend(instance, instance, &old, &edit,
        proof, aux, VerificationMode::Product, BatchOpenMode::Prepared, gpu, &mut CpuTrace::new()).unwrap();
}
fn check_current_commitments(params: &DynaverseParams, instance: &Instance, proof: &DynaverseProof, aux: &ProverAux) {
    let cpu = params.prove_anchor(instance, aux.current_witness()).unwrap();
    let mut w = G1Projective::from(proof.anchor.w_commitment);
    let mut h = G1Projective::from(proof.anchor.h_commitment);
    let anchor = proof.anchor.wires;
    let mut wires = [anchor.s1, anchor.s2, anchor.s3, anchor.s4, anchor.s5, anchor.s6];
    if let Some(delta) = &proof.delta {
        w += delta.changed_w_commitment; h += delta.changed_h_commitment;
        let changed = delta.changed_wires;
        let change = [changed.s1, changed.s2, changed.s3, changed.s4, changed.s5, changed.s6];
        for i in 0..6 { wires[i] = (G1Projective::from(wires[i]) + change[i]).into_affine(); }
    }
    let expected = cpu.anchor.wires;
    assert_eq!(w.into_affine(), cpu.anchor.w_commitment);
    assert_eq!(h.into_affine(), cpu.anchor.h_commitment);
    assert_eq!(wires, [expected.s1, expected.s2, expected.s3, expected.s4, expected.s5, expected.s6]);
}

#[test]
fn real_gpu_worker_overlaps_updates_swaps_and_releases_both_owners() {
    let circuit = ToyCircuit::new(16).unwrap(); let (instance, witness) = circuit.valid_instance();
    let mut rng = StdRng::seed_from_u64(20606);
    let params = Arc::new(DynaverseParams::setup(circuit, &mut rng).unwrap());
    let gpu = IcicleBackend::with_large_bucket_factor(10).unwrap();
    let (mut proof, mut aux) = params.prove_with_aux_traced_with_backend(&instance, &witness,
        AnchorOpeningPolicy::Precompute, &gpu, &mut CpuTrace::new()).unwrap();
    let observation = Arc::new(Mutex::new(Observation::default()));
    let mut worker = make_worker(params.clone(), observation.clone(), Arc::new(AtomicBool::new(false)));
    update(&params, &instance, &mut proof, &mut aux, &gpu, 1);
    assert!(worker.start(&aux).unwrap());
    update(&params, &instance, &mut proof, &mut aux, &gpu, 2);
    let deadline = Instant::now() + Duration::from_secs(20);
    loop { match worker.try_swap(&mut proof, &mut aux, Some(&gpu)).unwrap() {
        SwapOutcome::Building { .. } => { assert!(Instant::now() < deadline); thread::sleep(Duration::from_millis(1)); }
        SwapOutcome::Swapped { current_generation: 2, .. } => break,
        other => panic!("expected actual GPU swap, got {other:?}") } }
    assert!(params.verifying_key().verify_reference(&instance, &proof).unwrap());
    check_current_commitments(&params, &instance, &proof, &aux);
    let stats = observation.lock().unwrap().batch; assert_eq!(stats.max_batch_size, 6);
    assert_eq!(stats.buffer_grows, 1); drop(worker);
    assert_eq!(observation.lock().unwrap().dropped, 1);
    assert!(gpu.memory_info().unwrap().total_bytes > 0);
}

#[test]
#[ignore = "explicit 1,000-update actual GPU background stability experiment"]
fn thousand_gpu_updates_with_background_rebuild_and_recovery() {
    let path = std::env::var("DVGPU_SOAK_CSV").expect("set explicit output path");
    let mut file = File::create(path).unwrap();
    writeln!(file, "policy,step,generation,anchor_generation,foreground_rebuild,swap,worker_error,update_us,swap_us,front_capacity,front_grows,worker_capacity,worker_grows,worker_owners,gpu_free_bytes,gpu_total_bytes,cpu_verified,reference_commitment_match").unwrap();
    for policy in [AnchorOpeningPolicy::Dense, AnchorOpeningPolicy::Precompute] {
        if let Ok(selected) = std::env::var("DVGPU_SOAK_POLICY") {
            assert!(selected == "dense" || selected == "precompute");
            if (selected == "dense") != (policy == AnchorOpeningPolicy::Dense) { continue; }
        }
        let circuit = ToyCircuit::new(64).unwrap(); let (instance, witness) = circuit.valid_instance();
        let mut rng = StdRng::seed_from_u64(20607);
        let params = Arc::new(DynaverseParams::setup(circuit, &mut rng).unwrap());
        let gpu = IcicleBackend::with_large_bucket_factor(10).unwrap();
        let (mut proof, mut aux) = params.prove_with_aux_traced_with_backend(&instance, &witness,
            policy, &gpu, &mut CpuTrace::new()).unwrap();
        let observation = Arc::new(Mutex::new(Observation::default())); let fail = Arc::new(AtomicBool::new(false));
        let mut worker = make_worker(params.clone(), observation.clone(), fail.clone());
        let (mut swaps, mut rebuilds, mut discards, mut errors) = (0usize, 0usize, 0usize, 0usize);
        let began = Instant::now();
        for step in 1..=500 {
            let old_anchor = proof.anchor_generation; let started = Instant::now();
            update(&params, &instance, &mut proof, &mut aux, &gpu, step);
            let update_us = started.elapsed().as_secs_f64() * 1e6;
            let rebuilt = old_anchor != proof.anchor_generation; if rebuilt { rebuilds += 1; }
            let before_proof = proof.clone(); let before_aux = aux.clone(); let started = Instant::now();
            let mut errored = false;
            let outcome = match worker.try_swap(&mut proof, &mut aux, Some(&gpu)) {
                Ok(SwapOutcome::Swapped { .. }) => { swaps += 1; "swapped" }
                Ok(SwapOutcome::Discarded { .. }) => { discards += 1; "discarded" }
                Ok(SwapOutcome::Building { .. }) => "building",
                Ok(SwapOutcome::Idle) => "idle",
                Err(error) => { assert!(format!("{error:#}").contains("injected background GPU worker failure"),
                    "unexpected background failure at {policy:?} step={step}: {error:#}");
                    assert_eq!(proof, before_proof); assert_eq!(aux, before_aux);
                    errors += 1; errored = true; "error" }
            };
            let swap_us = started.elapsed().as_secs_f64() * 1e6;
            if step == 125 { fail.store(true, Ordering::SeqCst); }
            worker.start_if_due(&aux).unwrap();
            assert!(params.verifying_key().verify_reference(&instance, &proof).unwrap());
            assert_eq!(aux.scheduler().generation, proof.generation);
            assert_eq!(proof.generation, step as u64, "every soak step must change state");
            let check = step % 100 == 0;
            if check { check_current_commitments(&params, &instance, &proof, &aux); }
            let front = gpu.batch_stats().unwrap(); let background = *observation.lock().unwrap();
            assert!(front.input_capacity <= 384 && background.batch.input_capacity <= 384);
            assert!(front.buffer_grows <= 1 && background.batch.buffer_grows <= 1);
            let memory = if step % 50 == 0 { Some(gpu.memory_info().unwrap()) } else { None };
            let (free, total) = memory.map(|m| (m.free_bytes.to_string(), m.total_bytes.to_string())).unwrap_or_default();
            writeln!(file, "{policy:?},{step},{},{},{rebuilt},{outcome},{errored},{update_us:.3},{swap_us:.3},{},{},{},{},{},{free},{total},true,{}",
                proof.generation, proof.anchor_generation, front.input_capacity, front.buffer_grows,
                background.batch.input_capacity, background.batch.buffer_grows, background.created,
                if check { "true" } else { "" }).unwrap();
            if step % 50 == 0 { file.flush().unwrap(); eprintln!("SOAK_PROGRESS policy={policy:?} step={step} swaps={swaps} rebuilds={rebuilds} errors={errors}"); }
        }
        // Consume the one outstanding result, then join; no detached owner.
        let deadline = Instant::now() + Duration::from_secs(20);
        while matches!(worker.try_swap(&mut proof, &mut aux, Some(&gpu)).unwrap(), SwapOutcome::Building { .. }) {
            assert!(Instant::now() < deadline); thread::sleep(Duration::from_millis(1));
        }
        assert!(params.verifying_key().verify_reference(&instance, &proof).unwrap());
        check_current_commitments(&params, &instance, &proof, &aux); drop(worker);
        let observed = *observation.lock().unwrap();
        assert!(swaps > 0 && rebuilds > 0 && errors == 1);
        assert_eq!(observed.created, observed.dropped);
        eprintln!("SOAK_POLICY_OK policy={policy:?} updates=500 swaps={swaps} rebuilds={rebuilds} discards={discards} errors={errors} owners={} elapsed_s={:.3}", observed.created, began.elapsed().as_secs_f64());
    }
    file.flush().unwrap();
    if std::env::var("DVGPU_SOAK_POLICY").is_ok() { eprintln!("SOAK_SELECTED_POLICY_500_PUBLIC_CPU_VERIFIED_OK"); }
    else { eprintln!("SOAK_1000_PUBLIC_CPU_VERIFIED_OK"); }
}

#[test]
fn owner_memory_diagnostic_reads_real_device_capacity() {
    let owner = IcicleBackend::new().unwrap();
    let info = owner.memory_info().unwrap();
    assert!(info.total_bytes > 0, "must query actual CUDA memory availability");
    assert!(info.free_bytes <= info.total_bytes);
}
