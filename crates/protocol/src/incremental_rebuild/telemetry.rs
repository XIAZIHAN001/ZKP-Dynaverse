//! Optional host-wall timing. Rebuild and proof scopes nest inside preparation.
use std::time::{Duration,Instant};

/// Exclusive host-wall intervals inside one sparse proof; never CUDA events.
#[derive(Clone,Copy,Debug,Default,PartialEq,Eq)]
pub struct SparseProofTimings {
    pub ns:[u128;7],
    pub calls:[u64;7],
    pub(crate) enabled:bool,
}
impl SparseProofTimings {
    pub const NAMES:[&'static str;7]=["fixed_linear","geometry","interpolation_cauchy",
        "local_arithmetic","g1_commit_opening","g2_commit","proof_validation"];
    pub fn recorded_ns(&self)->u128 {self.ns.iter().sum()}
    pub(crate) fn new(enabled:bool)->Self {Self {enabled,..Self::default()}}
    pub(crate) fn start(&self)->Option<Instant> {if self.enabled {Some(Instant::now())} else {None}}
    pub(crate) fn finish(&mut self,phase:usize,started:Option<Instant>) {
        if let Some(started)=started {self.ns[phase]+=started.elapsed().as_nanos();self.calls[phase]+=1;}
    }
}

/// Exclusive host-wall details inside rebuild[cache_check]. Counts are logical
/// arkworks calls (including identities), not GPU instructions.
#[derive(Clone,Copy,Debug,Default,PartialEq,Eq)]
pub struct CacheCheckTimings {
    pub ns:[u128;8],
    pub equations:u64,
    pub pairing_terms:u64,
    pub miller_calls:u64,
    pub final_exponentiations:u64,
    pub g2_prepare_builds:u64,
    pub g2_prepare_hits:u64,
    /// Peak retained table/checker entries; excludes consumed temporary clones.
    /// This is not a count of all live preparations or an allocation/RSS peak.
    pub prepared_peak_entries:usize,
    pub divisors_formed:u64,
    /// Affine divisor lookups outside the independently prepared prefix.
    pub divisor_cache_hits:u64,
    /// Successful backend read submissions; checks remain independent per point.
    pub read_batches:u64,
    pub read_points:u64,
    pub max_read_roots:usize,
    pub(crate) enabled:bool,
}
impl CacheCheckTimings {
    pub const NAMES:[&'static str;8]=["read_points","g1_difference","g2_divisor",
        "g2_prepare","miller","final_exponentiation","undo_store","unchanged_copy"];
    pub fn recorded_ns(&self)->u128 {self.ns.iter().sum()}
    pub(crate) fn start(&self)->Option<Instant> {if self.enabled {Some(Instant::now())} else {None}}
    pub(crate) fn finish(&mut self,phase:usize,started:Option<Instant>) {
        if let Some(started)=started {self.ns[phase]+=started.elapsed().as_nanos();}
    }
}

/// Exclusive host intervals nested inside rebuild[linear_msm].
#[derive(Clone,Copy,Debug,Default,PartialEq,Eq)]
pub struct LinearTimings {
    pub ns:[u128;4],
    pub calls:[u64;4],
    pub(crate) enabled:bool,
}
impl LinearTimings {
    pub const NAMES:[&'static str;4]=["fixed_g1","global_s5_g2","frozen_copy","quotient_g1"];
    pub fn recorded_ns(&self)->u128 {self.ns.iter().sum()}
    pub(crate) fn start(&self)->Option<Instant> {if self.enabled {Some(Instant::now())} else {None}}
    pub(crate) fn finish(&mut self,phase:usize,started:Option<Instant>) {
        if let Some(started)=started {self.record(phase,started.elapsed());}
    }
    pub(crate) fn record(&mut self,phase:usize,elapsed:Duration) {
        if self.enabled {self.ns[phase]+=elapsed.as_nanos();self.calls[phase]+=1;}
    }
}

#[derive(Clone,Copy,Debug,Default,PartialEq,Eq)]
pub struct UpdateTimings {
    pub enabled:bool,
    pub total_ns:u128,
    pub normalization_ns:u128,
    pub constraints_ns:u128,
    pub preparation_ns:u128,
    pub publication_ns:u128,
    pub delta_proof_ns:u128,
    pub rebase_ns:u128,
    /// Nested in preparation, disjoint from delta/rebase/rebuild intervals.
    pub live_linear_ns:u128,
    pub delta_detail:SparseProofTimings,
    pub rebase_detail:SparseProofTimings,
    pub cache_detail:CacheCheckTimings,
    pub linear_detail:LinearTimings,
    pub rebuild_ns:[u128;8],
    pub rebuild_calls:[u64;8],
}

impl UpdateTimings {
    pub const REBUILD_NAMES:[&'static str;8]=[
        "materialize","linear_msm","fr_transform","g1_transform",
        "polynomial","memory","cache_check","finish"];
    pub(crate) fn new(enabled:bool)->Self {Self {enabled,delta_detail:SparseProofTimings::new(enabled),
        rebase_detail:SparseProofTimings::new(enabled),
        cache_detail:CacheCheckTimings {enabled,..Default::default()},
        linear_detail:LinearTimings {enabled,..Default::default()},..Self::default()}}
    pub(crate) fn start(&self)->Option<Instant> {
        if self.enabled {Some(Instant::now())} else {None}
    }
    pub(crate) fn elapsed(start:Option<Instant>)->u128 {
        start.map(|value|value.elapsed().as_nanos()).unwrap_or(0)
    }
    pub(crate) fn record(&mut self,phase:RebuildPhase,duration:Duration) {
        if self.enabled {
            self.rebuild_ns[phase as usize]+=duration.as_nanos();
            self.rebuild_calls[phase as usize]+=1;
        }
    }
}

#[derive(Clone,Copy)]
pub(crate) enum RebuildPhase {
    Materialize,Linear,FrTransform,G1Transform,Polynomial,Memory,Cache,Finish,
}
