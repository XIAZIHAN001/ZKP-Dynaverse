//! Fixed-snapshot global rebuilding with call-level publication. Large
//! buffers are allocated once and exchanged, never cloned at a switch.
use super::*;
use crate::incremental_rebuild::{
    BlockTimings, IncrementalConfig, RebuildProgress, ResourcePlan, UpdateReceipt,UpdateTimings,RebuildPhase,
    SparseBackend,SparseDispatchPolicy,SparseDispatchStats,
    operators::{BufferId,BufferKind,FieldBinaryOp,FourStepPlan,RebuildBackend},
    work::{CachedWindow,RebuildBudget,WorkBound},
};
use std::collections::{BTreeMap,BTreeSet};
use std::cell::Cell;
use std::sync::Arc;
use std::time::Instant;
mod program;
mod machine;
mod linear;
mod global_g2;
mod preparation;
use program::{Program,POINTS};
use machine::rollback_host;
use super::sparse_linear::FixedLinear;
use super::sparse_geometry::SparseGeometry;

struct AnchorStorage {witness:Witness,cache:KzgPrecomputation}
struct AnchorEpoch {storage:AnchorStorage,proof:AnchorProof,generation:u64}

#[derive(Clone)]
struct Frozen {
    anchor:Arc<AnchorEpoch>,
    delta:BTreeMap<usize,Fr>,
    generation:u64,
    public_h:G1Projective,
    wire_mask:u8,
    fixed:Option<FixedLinear>,
}
impl Frozen {
    fn value(&self,index:usize)->Fr {
        self.anchor.storage.witness.wires[index]
            + self.delta.get(&index).copied().unwrap_or_else(Fr::zero)
    }
}

#[derive(Clone)]
struct Machine {
    source:Option<Frozen>,
    overlay:BTreeMap<usize,Fr>,
    cursor:usize,
    cache_offset:usize,
    transform:Option<FourStepPlan>,
    points:[G1Projective;POINTS],
    s5_g2:G2Projective,
    paid:usize,
}
impl Default for Machine {
    fn default()->Self {
        Self {source:None,overlay:BTreeMap::new(),cursor:0,cache_offset:0,transform:None,
            points:std::array::from_fn(|_|G1Projective::zero()),
            s5_g2:G2Projective::zero(),paid:0}
    }
}

enum HostUndo {
    Witness(usize,Vec<Fr>),
    Cache(usize,usize,Vec<G1Affine>),
}

struct Prepared {
    proof:DynaverseProof,
    delta:BTreeMap<usize,Fr>,
    public_h:G1Projective,
    install:bool,
    receipt:UpdateReceipt,
    live_fixed:Option<FixedLinear>,
    geometry:Option<Arc<SparseGeometry>>,
}

/// Owns its backend on the caller's thread. CUDA handles never cross threads.
/// Getters borrow live snapshots; exporting/cloning them is caller-paid O(n).
pub struct IncrementalSession<'p,B:RebuildBackend> {
    params:&'p DynaverseParams,
    backend:B,
    program:Program,
    active:Arc<AnchorEpoch>,
    candidate:AnchorStorage,
    current_witness:Witness,
    current_instance:Instance,
    current_proof:DynaverseProof,
    delta:BTreeMap<usize,Fr>,
    public_h:G1Projective,
    machine:Machine,
    last:UpdateReceipt,
    timings:BlockTimings,
    faulted:bool,
    sparse_policy:SparseDispatchPolicy,
    sparse_stats:Cell<SparseDispatchStats>,
    trace_updates:bool,
    update_timings:UpdateTimings,
    live_fixed:Option<FixedLinear>,
    geometry:Option<Arc<SparseGeometry>>,
    geometry_reuse:bool,
}

impl DynaverseParams {
    pub fn start_incremental_session<B:RebuildBackend>(&self,instance:&Instance,witness:&Witness,
        config:IncrementalConfig,mut backend:B)->Result<IncrementalSession<'_,B>> {
        self.circuit.validate(instance,witness)?;
        if config.sparse==SparseDispatchPolicy::ForceGpu {
            let caps=backend.gpu_capabilities();
            ensure!(caps.g1_msm && caps.g2_msm && caps.fr_ntt,
                "ForceGpu requires native Fr NTT and G1/G2 MSM capabilities");
        }
        let mut program=Program::new(self,&config,&mut backend)?;
        let n=self.circuit.n();
        let public_h=G1Projective::msm_unchecked(&self.verifying.public_h_bases,&instance.public_inputs);
        let mut storage=AnchorStorage {witness:witness.clone(),
            cache:KzgPrecomputation {q4:vec![G1Affine::identity();n],
                q5:vec![G1Affine::identity();n],q6:vec![G1Affine::identity();n]}};
        let anchor=Machine::build_initial(self,&program,&mut storage,&mut backend,public_h)?;
        let proof=DynaverseProof {anchor,delta:None,anchor_generation:0,generation:0};
        ensure!(self.verifying.verify(instance,&proof)?,"initial incremental proof failed");
        program.resource.resident_bytes=backend.stats().bytes_allocated;
        ensure!(program.resource.resident_bytes<=program.resource.upper_bytes,
            "initial foreground storage exceeds resource envelope");
        let candidate=AnchorStorage {witness:Witness {wires:vec![Fr::zero();6*n]},
            cache:KzgPrecomputation {q4:vec![G1Affine::identity();n],
                q5:vec![G1Affine::identity();n],q6:vec![G1Affine::identity();n]}};
        let active=Arc::new(AnchorEpoch {
            storage,
            proof:proof.anchor.clone(),generation:0,
        });
        let live_fixed=if config.foreground_linear_reuse {Some(FixedLinear::from_anchor(&active.proof)?)} else {None};
        Ok(IncrementalSession {params:self,backend,program,active,candidate,
            current_witness:witness.clone(),current_instance:instance.clone(),
            current_proof:proof,delta:BTreeMap::new(),public_h,machine:Machine::default(),
            last:UpdateReceipt::default(),timings:BlockTimings::default(),faulted:false,
            sparse_policy:config.sparse,sparse_stats:Cell::new(SparseDispatchStats::default()),
            trace_updates:config.trace_updates,update_timings:UpdateTimings::default(),live_fixed,
            geometry:None,geometry_reuse:config.support_geometry_reuse})
    }
}

impl<B:RebuildBackend> IncrementalSession<'_,B> {
    pub fn proof(&self)->&DynaverseProof {&self.current_proof}
    pub fn current_witness(&self)->&Witness {&self.current_witness}
    pub fn current_instance(&self)->&Instance {&self.current_instance}
    pub fn anchor_distance(&self)->usize {self.delta.len()}
    pub fn window(&self)->usize {self.program.window.window}
    pub fn horizon(&self)->usize {self.program.window.horizon}
    pub fn budget(&self)->RebuildBudget {self.program.budget}
    pub fn resource_plan(&self)->ResourcePlan {self.program.resource}
    pub fn cache_preparation_stats(&self)->Option<crate::incremental_rebuild::CachePreparationStats> {
        self.program.cache_preparation.as_ref().map(preparation::PublicPreparation::stats)
    }
    pub fn backend(&self)->&B {&self.backend}
    pub fn last_receipt(&self)->&UpdateReceipt {&self.last}
    pub fn block_timings(&self)->&BlockTimings {&self.timings}
    pub fn update_timings(&self)->&UpdateTimings {&self.update_timings}
    pub fn is_faulted(&self)->bool {self.faulted}
    pub fn sparse_dispatch_stats(&self)->SparseDispatchStats {self.sparse_stats.get()}
    pub fn rebuild_progress(&self)->Option<RebuildProgress> {
        self.machine.source.as_ref().map(|source|RebuildProgress {
            source_generation:source.generation,paid_changes:self.machine.paid,
            command:self.machine.cursor,commands:self.program.commands.len()})
    }

    pub fn update(&mut self,update:&Update)->Result<DynaverseProof> {
        self.update_with_public_inputs(update,&Update {indices:Vec::new(),values:Vec::new()})
    }

    pub fn update_with_public_inputs(&mut self,update:&Update,public:&Update)->Result<DynaverseProof> {
        let mut trace=UpdateTimings::new(self.trace_updates);
        let total_started=trace.start();
        let normalization_started=trace.start();
        ensure!(!self.faulted,"incremental backend is faulted; explicitly recreate the session");
        let n=self.params.circuit.n();
        let raw_edits=update.indices.len().checked_add(public.indices.len())
            .ok_or_else(||anyhow::anyhow!("raw edit count overflow"))?;
        let normalized=update.clone().canonicalize(6*n)?;
        let normalized_public=public.clone().canonicalize(n)?;
        let mut edits=BTreeMap::new();
        let mut public_edits=BTreeMap::new();
        let mut dirty=BTreeSet::new();
        let mut changed_wire_mask=0u8;
        for (&i,&value) in normalized.indices.iter().zip(&normalized.values) {
            if self.current_witness.wires[i]!=value {
                edits.insert(i,value);dirty.insert(i%n);changed_wire_mask|=1<<(i/n);
            }
        }
        for (&i,&value) in normalized_public.indices.iter().zip(&normalized_public.values) {
            if self.current_instance.public_inputs[i]!=value {
                public_edits.insert(i,value);dirty.insert(i);
            }
        }
        let k=edits.len().checked_add(public_edits.len())
            .ok_or_else(||anyhow::anyhow!("change count overflow"))?;
        trace.normalization_ns=UpdateTimings::elapsed(normalization_started);
        if k==0 {
            self.last=UpdateReceipt {raw_edits,delta_entries:self.delta.len(),
                frozen_entries:self.machine.source.as_ref().map_or(0,|s|s.delta.len()),
                frozen_wire_mask:self.machine.source.as_ref().map_or(0,|s|s.wire_mask),
                overlay_entries:self.machine.overlay.len(),..UpdateReceipt::default()};
            trace.total_ns=UpdateTimings::elapsed(total_started);
            self.update_timings=trace;
            return Ok(self.current_proof.clone());
        }
        let constraints_started=trace.start();
        let wire=|i:usize|edits.get(&i).copied().unwrap_or(self.current_witness.wires[i]);
        let input=|i:usize|public_edits.get(&i).copied().unwrap_or(self.current_instance.public_inputs[i]);
        crate::workcount::record_scan(normalized.indices.len()+normalized_public.indices.len());
        crate::workcount::record_gates(dirty.len());
        for &i in &dirty {
            ensure!(wire(i)==input(i),"public copy constraint fails at {i}");
            ensure!(wire(2*n+i)==wire(3*n+i),"wire copy constraint fails at {i}");
            ensure!(wire(i)+wire(n+i)==wire(2*n+i),"addition gate fails at {i}");
            ensure!(wire(3*n+i)*wire(4*n+i)==wire(5*n+i),"multiplication gate fails at {i}");
        }
        trace.constraints_ns=UpdateTimings::elapsed(constraints_started);
        let preparation_started=trace.start();
        let generation=self.current_proof.generation.checked_add(1)
            .ok_or_else(||anyhow::anyhow!("proof generation overflow"))?;
        let checkpoint=self.machine.clone();
        let mut undo=Vec::new();
        let mut call_timings=self.timings;
        if let Err(error)=self.backend.begin_transaction() {
            self.faulted=true;
            return Err(error.context("begin incremental transaction"));
        }
        let prepared=self.prepare(&edits,&public_edits,k,generation,&mut undo,&mut call_timings,&mut trace);
        let mut prepared=match prepared {
            Ok(prepared)=>prepared,
            Err(error)=>{
                rollback_host(&mut self.candidate,undo);
                self.machine=checkpoint;
                if let Err(rollback_error)=self.backend.rollback_transaction() {
                    self.faulted=true;
                    return Err(error.context(format!("backend rollback failed: {rollback_error}; session faulted")));
                }
                return Err(error);
            }
        };
        prepared.receipt.raw_edits=raw_edits;
        prepared.receipt.affected_gates=dirty.len();
        prepared.receipt.changed_wire_mask=changed_wire_mask;
        prepared.receipt.delta_entries=prepared.delta.len();
        prepared.receipt.frozen_entries=self.machine.source.as_ref().map_or(0,|s|s.delta.len());
        prepared.receipt.frozen_wire_mask=self.machine.source.as_ref().map_or(0,|s|s.wire_mask);
        prepared.receipt.overlay_entries=self.machine.overlay.len();
        trace.preparation_ns=UpdateTimings::elapsed(preparation_started);
        let publication_started=trace.start();
        // Every CUDA operation, candidate check and rebase has completed.
        // No fallible backend operation is allowed after this point.
        self.backend.commit_transaction();
        drop(checkpoint);
        if prepared.install {
            let source_generation=self.machine.source.as_ref().unwrap().generation;
            let empty=AnchorStorage {witness:Witness {wires:Vec::new()},
                cache:KzgPrecomputation {q4:Vec::new(),q5:Vec::new(),q6:Vec::new()}};
            let storage=std::mem::replace(&mut self.candidate,empty);
            let next=Arc::new(AnchorEpoch {storage,proof:prepared.proof.anchor.clone(),
                generation:source_generation});
            let old=std::mem::replace(&mut self.active,next);
            self.machine=Machine::default();
            // Only the session and its frozen job can own this private Arc.
            // The job and checkpoint were released above; retain all arrays
            // by recycling the old epoch as the next candidate.
            let previous=match Arc::try_unwrap(old) {
                Ok(previous)=>previous,
                Err(_)=>unreachable!("private anchor unexpectedly retained"),
            };
            self.candidate=previous.storage;
        }
        for (&i,&value) in &edits {self.current_witness.wires[i]=value;}
        for (&i,&value) in &public_edits {self.current_instance.public_inputs[i]=value;}
        self.delta=prepared.delta;
        self.public_h=prepared.public_h;
        self.current_proof=prepared.proof;
        self.live_fixed=prepared.live_fixed;
        self.geometry=prepared.geometry;
        self.last=prepared.receipt;
        self.timings=call_timings;
        if prepared.install && self.delta.len()>=self.horizon() {
            let fixed=if self.program.capture_limit.is_some() {self.live_fixed.clone()} else {None};
            self.freeze(self.delta.clone(),generation,self.public_h,fixed);
            self.last.restarted=true;
        }
        trace.publication_ns=UpdateTimings::elapsed(publication_started);
        trace.total_ns=UpdateTimings::elapsed(total_started);
        self.update_timings=trace;
        Ok(self.current_proof.clone())
    }

    fn freeze(&mut self,delta:BTreeMap<usize,Fr>,generation:u64,public_h:G1Projective,
        live:Option<FixedLinear>) {
        crate::workcount::record_scan(delta.len());
        let n=self.params.circuit.n();
        let wire_mask=delta.keys().fold(0u8,|mask,&index|mask|(1<<(index/n)));
        let fixed=if self.program.capture_limit.is_some() {live} else {None};
        self.machine=Machine {source:Some(Frozen {anchor:self.active.clone(),delta,
            generation,public_h,wire_mask,fixed}),..Machine::default()};
    }

    fn prepare(&mut self,edits:&BTreeMap<usize,Fr>,public_edits:&BTreeMap<usize,Fr>,
        k:usize,generation:u64,undo:&mut Vec<HostUndo>,timings:&mut BlockTimings,
        trace:&mut UpdateTimings)->Result<Prepared> {
        let mut delta=self.delta.clone();
        crate::workcount::record_scan(self.delta.len());
        for (&i,&value) in edits {
            let difference=value-self.active.storage.witness.wires[i];
            if difference.is_zero() {delta.remove(&i);} else {delta.insert(i,difference);}
        }
        let mut public_h=self.public_h;
        if !public_edits.is_empty() {
            let differences:Vec<(usize,Fr)>=public_edits.iter()
                .map(|(&i,&value)|(i,value-self.current_instance.public_inputs[i])).collect();
            let selected=SparseBackend {backend:&self.backend,policy:self.sparse_policy,stats:&self.sparse_stats};
            public_h+=crate::kzg::gather_msm_batch(
                &[(&self.params.verifying.public_h_bases,&differences)],&selected)?[0];
        }
        let large=k>self.horizon();
        let mut receipt=UpdateReceipt {changed:k,allowance:self.program.budget.allowance(k)?,
            large_batch:large,..UpdateReceipt::default()};
        let mut geometry=self.geometry.clone();
        let mut live_fixed=None;
        if let Some(live)=&self.live_fixed {
            if !large || self.program.capture_limit.is_some_and(|limit|edits.len()<=limit) {
                let started=trace.start();
                let eta:Vec<(usize,Fr)>=edits.iter().map(|(&i,&v)|(i,v-self.current_witness.wires[i])).collect();
                let selected=SparseBackend {backend:&self.backend,policy:self.sparse_policy,stats:&self.sparse_stats};
                live_fixed=Some(live.add(&FixedLinear::compute(self.params,&eta,&selected)?));
                receipt.foreground_linear_reused=true;receipt.foreground_linear_inputs=eta.len();
                receipt.foreground_linear_work=linear::work_bound(eta.len())?;
                trace.live_linear_ns+=UpdateTimings::elapsed(started);
            }
        }
        let mut proof=if large {None} else {
            let started=trace.start();
            let value=self.sparse_proof(&self.active.storage,&self.active.proof,
                self.active.generation,&delta,generation,public_h,live_fixed.as_ref(),&mut geometry,&mut receipt,&mut trace.delta_detail)?;
            trace.delta_proof_ns+=UpdateTimings::elapsed(started);
            Some(value)
        };
        if large {
            let fixed=if self.program.capture_limit.is_some() {live_fixed.clone()} else {None};
            self.freeze(delta.clone(),generation,public_h,fixed);
            receipt.started=true;
        } else if self.machine.source.is_none() && delta.len()>=self.horizon() {
            let fixed=if self.program.capture_limit.is_some() {live_fixed.clone()} else {None};
            self.freeze(delta.clone(),generation,public_h,fixed);
            receipt.started=true;
        } else if let Some(source)=&self.machine.source {
            for (&i,&value) in edits {
                let difference=value-source.value(i);
                if difference.is_zero() {self.machine.overlay.remove(&i);}
                else {self.machine.overlay.insert(i,difference);}
            }
        }
        let mut install=false;
        if self.machine.source.is_some() {
            self.machine.paid=self.machine.paid.checked_add(k)
                .ok_or_else(||anyhow::anyhow!("rebuild credit overflow"))?;
            self.machine.advance(self.params,&self.program,&mut self.candidate,&mut self.backend,
                receipt.allowance,undo,&mut receipt,timings,trace,&self.sparse_stats)?;
            let ready=self.machine.cursor==self.program.commands.len();
            ensure!(ready || self.machine.paid<self.horizon(),
                "rebuild missed its certified deadline; no synchronous catch-up is permitted");
            if ready {
                let anchor=self.machine.anchor();
                let source=self.machine.source.as_ref().unwrap();
                delta=self.machine.overlay.clone();
                let started=trace.start();
                proof=Some(self.sparse_proof(&self.candidate,&anchor,source.generation,
                    &delta,generation,public_h,live_fixed.as_ref(),&mut geometry,&mut receipt,&mut trace.rebase_detail)?);
                trace.rebase_ns+=UpdateTimings::elapsed(started);
                install=true;
                receipt.swapped=true;
            }
        }
        ensure!(receipt.rebuild_units<=receipt.allowance,"rebuild exceeded per-call budget");
        let proof=proof.ok_or_else(||anyhow::anyhow!("large batch did not finish its paid rebuild"))?;
        if self.live_fixed.is_some() && large && live_fixed.is_none() {
            // Paid large rebuild is the new current witness; seed from its
            // verified fixed fields without an oversized sparse gather.
            ensure!(proof.anchor_generation==generation,"large live reset requires current anchor");
            let started=trace.start();
            live_fixed=Some(FixedLinear::from_anchor(&proof.anchor)?);
            receipt.foreground_linear_work=crate::incremental_rebuild::work::WorkBound {
                host_elements:128,copy_bytes:56*96*2,field_ops:128,..Default::default()};
            trace.live_linear_ns+=UpdateTimings::elapsed(started);
        }
        if self.program.capture_limit.is_some() {
            // Covers immutable snapshot/checkpoint copies outside advance;
            // the machine's actual copy is charged separately to rebuild kQ.
            receipt.foreground_linear_work=receipt.foreground_linear_work
                .checked_add(program::capture_copy_bound())?;
        }
        Ok(Prepared {proof,delta,public_h,install,receipt,live_fixed,geometry})
    }

    fn sparse_proof(&self,storage:&AnchorStorage,anchor:&AnchorProof,anchor_generation:u64,
        delta:&BTreeMap<usize,Fr>,generation:u64,public_h:G1Projective,
        live_fixed:Option<&FixedLinear>,geometry:&mut Option<Arc<SparseGeometry>>,
        receipt:&mut UpdateReceipt,trace:&mut crate::incremental_rebuild::SparseProofTimings)->Result<DynaverseProof> {
        let selected=SparseBackend {backend:&self.backend,policy:self.sparse_policy,stats:&self.sparse_stats};
        let changed=if generation==anchor_generation {
            ensure!(delta.is_empty(),"new anchor has a nonempty same-generation delta");
            *geometry=None;
            None
        } else {
            let entries:Vec<(usize,Fr)>=delta.iter().map(|(&i,&value)|(i,value)).collect();
            let fixed=live_fixed.map(|live|live.relative_to(anchor)).transpose()?;
            let (proof,next,hit)=self.params.prove_delta_sparse_traced(&storage.witness,&entries,&anchor.wires,
                &storage.cache,&selected,Some(&selected),fixed.as_ref(),geometry.as_ref(),anchor_generation,self.geometry_reuse,trace)?;
            if self.geometry_reuse {
                receipt.support_geometry_reused|=hit;
                receipt.support_geometry_builds+=usize::from(!hit);
            }
            *geometry=next;
            Some(proof)
        };
        let proof=DynaverseProof {anchor:anchor.clone(),delta:changed,anchor_generation,generation};
        let validation_started=trace.start();
        let vk=&self.params.verifying;
        ensure!(vk.verify_anchor_core(&proof.anchor)?,"incremental anchor check failed");
        let delta_h=if let Some(delta)=&proof.delta {
            ensure!(vk.verify_delta_core(&proof.anchor,delta)?,"incremental sparse delta check failed");
            G1Projective::from(delta.changed_h_commitment)
        } else {G1Projective::zero()};
        ensure!(G1Projective::from(proof.anchor.h_commitment)+delta_h+public_h==G1Projective::zero(),
            "incremental public input binding failed");
        trace.finish(6,validation_started);
        Ok(proof)
    }
}
