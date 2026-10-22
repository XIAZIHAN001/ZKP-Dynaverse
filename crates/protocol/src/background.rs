//! Bounded background anchor reconstruction and foreground installation.
use crate::{dynaverse::DynaverseParams, trace::CpuTrace, AnchorOpeningPolicy,
    DynaverseProof, Instance, OperatorBackend, ProverAux, Witness};
use anyhow::{anyhow, Context, Result};
use std::sync::{mpsc::{self, Receiver, SyncSender, TryRecvError}, Arc};
use std::thread::{self, JoinHandle};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwapOutcome {
    Idle,
    Building { source_generation: u64 },
    Swapped { anchor_generation: u64, current_generation: u64 },
    Discarded { source_generation: u64 },
}

struct Snapshot {
    generation: u64,
    instance: Instance,
    witness: Witness,
    policy: AnchorOpeningPolicy,
}
struct Candidate { generation: u64, aux: ProverAux }

/// One outstanding snapshot and one thread-owned, reusable operator backend.
/// A background snapshot may become obsolete: polling rebases or discards it.
pub struct BackgroundRebuilder {
    params: Arc<DynaverseParams>,
    jobs: Option<SyncSender<Snapshot>>,
    results: Receiver<Result<Candidate>>,
    worker: Option<JoinHandle<()>>,
    pending: Option<u64>,
}
impl BackgroundRebuilder {
    /// The factory crosses threads; B itself never does. This permits CUDA
    /// owners that are !Send/!Sync to be created and destroyed on the worker.
    pub fn new<F, B>(params: Arc<DynaverseParams>, mut factory: F) -> Result<Self>
    where F: FnMut() -> Result<B> + Send + 'static, B: OperatorBackend + 'static {
        let (jobs, receiver) = mpsc::sync_channel::<Snapshot>(1);
        let (sender, results) = mpsc::sync_channel(1);
        let worker_params = params.clone();
        let worker = thread::Builder::new().name("dynaverse-anchor".into()).spawn(move || {
            // B is introduced here, after the Send closure reaches its thread.
            let mut backend: Option<B> = None;
            while let Ok(snapshot) = receiver.recv() {
                let result = (|| -> Result<Candidate> {
                    if backend.is_none() { backend = Some(factory().context("background backend creation")?); }
                    let (_, aux) = worker_params.prove_with_aux_traced_with_backend(
                        &snapshot.instance, &snapshot.witness, snapshot.policy,
                        backend.as_ref().unwrap(), &mut CpuTrace::disabled())
                        .context("background anchor construction")?;
                    worker_params.validate_background_candidate(&aux)?;
                    Ok(Candidate { generation: snapshot.generation, aux })
                })();
                // Never continue using an owner after any operator/proof error.
                if result.is_err() { backend = None; }
                if sender.send(result).is_err() { break; }
            }
        }).context("spawn background anchor worker")?;
        Ok(Self { params, jobs: Some(jobs), results, worker: Some(worker), pending: None })
    }

    /// Returns false while a job is outstanding or the current anchor already
    /// represents this generation. Caller state is cloned only for an accepted job.
    pub fn start(&mut self, aux: &ProverAux) -> Result<bool> {
        if self.pending.is_some() { return Ok(false); }
        self.params.check_background_state(&aux.current_proof, aux)?;
        if aux.current_proof.anchor_generation == aux.scheduler.generation { return Ok(false); }
        let snapshot = Snapshot { generation: aux.scheduler.generation,
            instance: aux.current_instance.clone(), witness: aux.current_witness.clone(),
            policy: aux.opening_policy };
        let generation = snapshot.generation;
        self.jobs.as_ref().ok_or_else(|| anyhow!("background worker closed"))?
            .try_send(snapshot).map_err(|e| anyhow!("background snapshot submission failed: {e}"))?;
        self.pending = Some(generation);
        Ok(true)
    }

    /// Prefetch at half the actual delta bound, or half the update interval for
    /// repeated hotspot edits. Synchronous foreground rebuild remains available.
    pub fn start_if_due(&mut self, aux: &ProverAux) -> Result<bool> {
        if self.pending.is_some() || !self.params.background_rebuild_due(aux)? { return Ok(false); }
        self.start(aux)
    }

    /// No state changes occur before replacement proof/cache preparation and
    /// public verification succeed. A failure consumes this result, permitting retry.
    pub fn try_swap(&mut self, proof: &mut DynaverseProof, aux: &mut ProverAux,
        backend: Option<&dyn OperatorBackend>) -> Result<SwapOutcome> {
        let Some(generation) = self.pending else { return Ok(SwapOutcome::Idle); };
        let candidate = match self.results.try_recv() {
            Err(TryRecvError::Empty) => return Ok(SwapOutcome::Building { source_generation: generation }),
            Err(TryRecvError::Disconnected) => {
                self.pending = None;
                return Err(anyhow!("background worker disconnected or panicked"));
            }
            Ok(candidate) => candidate,
        };
        self.pending = None;
        let candidate = candidate?;
        if candidate.generation != generation { return Err(anyhow!("background generation mismatch")); }
        if self.params.install_background_candidate(candidate.generation, candidate.aux, proof, aux, backend)? {
            Ok(SwapOutcome::Swapped { anchor_generation: proof.anchor_generation,
                current_generation: proof.generation })
        } else { Ok(SwapOutcome::Discarded { source_generation: generation }) }
    }
}

impl Drop for BackgroundRebuilder {
    fn drop(&mut self) {
        // One outstanding result fits in the bounded result channel, so the
        // worker cannot block sending while Drop waits. Close jobs, then join;
        // the worker backend and its CUDA resources die before Drop returns.
        self.jobs.take();
        if let Some(worker) = self.worker.take() { let _ = worker.join(); }
    }
}
