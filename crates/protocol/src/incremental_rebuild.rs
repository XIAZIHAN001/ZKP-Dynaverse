//! Bounded, serial anchor reconstruction. The original synchronous and
//! threaded APIs remain available as independent correctness references.
pub mod operators;
pub mod work;
mod telemetry;
pub use telemetry::{UpdateTimings,SparseProofTimings,CacheCheckTimings,LinearTimings};
pub(crate) use telemetry::RebuildPhase;
mod sparse_dispatch;
pub use sparse_dispatch::{SparseDispatchPolicy,SparseDispatchStats};
pub(crate) use sparse_dispatch::SparseBackend;

/// Initialization-only policy; the normalized window and budget are cached.
#[derive(Clone, Debug)]
pub struct IncrementalConfig {
    pub window: Option<usize>,
    pub optimized_window: bool,
    pub tile_len: Option<usize>,
    pub memory_limit_bytes: u64,
    pub sparse: SparseDispatchPolicy,
    /// Optional host-wall diagnostics; nested rebuild categories are not additive to outer wall.
    pub trace_updates: bool,
    /// Opt-in frozen sparse correction; dense initialization/fallback remain available.
    pub linear_rebuild_reuse: bool,
    /// Reuse old verified opening caches only for frozen unchanged wires.
    pub unchanged_cache_reuse: bool,
    /// Maintain absolute fixed-linear G1 fields using actual wire edits.
    pub foreground_linear_reuse: bool,
    /// Opt-in immutable 56-G1 freeze capture. Requires foreground reuse;
    /// large calls maintain live fields only up to this actual wire-eta cap.
    /// Valid limits are 1..=4096; None preserves the previous schedule.
    pub freeze_linear_capture_limit: Option<usize>,
    /// Independent global s5 G2 = verified anchor + frozen delta_s5.
    /// None preserves dense/legacy R1 behavior; Some1..4096 bounds s5 entries.
    pub frozen_g2_reuse_limit: Option<usize>,
    pub support_geometry_reuse: bool,
    /// Existing resident row batching; bounded to 1..=16 at initialization.
    pub transform_row_batch: usize,
    /// Opt-in complete resident transform packet, bounded by transform length.
    /// None retains the tiled cursor; valid Some limits are1..=4096.
    pub transform_packet_max_len: Option<usize>,
    /// Exact per-equation CPU cache validation. Reference remains the default.
    pub rebuild_cache_mode:crate::VerificationMode,
    /// Opt-in bounded fixed-root public G2 preparation; Product mode only.
    pub rebuild_prepared_cache_roots:Option<usize>,
    /// Opt-in cache read coalescing, Some4..=64 roots; None keeps old tiles.
    /// Independent pairing checkers remain bounded to four roots.
    pub rebuild_cache_readback_roots:Option<usize>,
}

impl Default for IncrementalConfig {
    fn default() -> Self {
        Self { window: None, optimized_window: false, tile_len: None,
            memory_limit_bytes: 1 << 30, sparse:SparseDispatchPolicy::default(),
            trace_updates:false,linear_rebuild_reuse:false,unchanged_cache_reuse:false,
            foreground_linear_reuse:false,freeze_linear_capture_limit:None,
            frozen_g2_reuse_limit:None,
            support_geometry_reuse:false,transform_row_batch:1,transform_packet_max_len:None,
            rebuild_cache_mode:crate::VerificationMode::Reference,rebuild_prepared_cache_roots:None,
            rebuild_cache_readback_roots:None }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UpdateReceipt {
    pub raw_edits: usize,
    pub affected_gates: usize,
    /// Distance from the published anchor after this successful call.
    pub delta_entries: usize,
    /// Frozen job processed by this call, sampled before anchor publication.
    /// On swap/restart this describes the old job, not rebuild_progress().
    /// A no-op instead reports the currently pending job.
    pub frozen_entries: usize,
    /// Overlay of the processed job before publication; on swap it becomes
    /// delta_entries, even if publication immediately freezes a new job.
    pub overlay_entries: usize,
    pub changed_wire_mask: u8,
    /// Wire mask of the same pre-publication job as frozen_entries.
    pub frozen_wire_mask: u8,
    pub linear_reused: bool,
    pub linear_skipped_g1_points: usize,
    pub frozen_linear_captured: bool,
    pub frozen_linear_skipped_g1_points: usize,
    /// This call copied/corrected the frozen global G2, not a local delta field.
    pub global_g2_reused:bool,
    pub global_g2_inputs:usize,
    pub global_g2_skipped_points:usize,
    /// Complete FFT packets executed in this call; initialization excluded.
    pub transform_packet_calls:u64,
    /// Sum of complete FFT lengths, not the twice-length row-transform count.
    pub transform_packet_elements:u64,
    /// Cache-wire bits 0..2 mean s4/s5/s6 copied in this call (possibly partial).
    pub cache_reused_wire_mask: u8,
    /// Number of verified old cache points copied into the candidate this call.
    pub cache_reused_points: usize,
    pub foreground_linear_reused: bool,
    /// Actual wire eta inputs, excluding public-input namespace edits.
    pub foreground_linear_inputs: usize,
    /// Separate foreground ledger; rebuild_units/kQ cover rebuild work only.
    pub foreground_linear_work: work::WorkBound,
    pub support_geometry_reused: bool,
    pub support_geometry_builds: usize,
    pub changed: usize,
    pub allowance: u128,
    pub rebuild_units: u128,
    pub work: work::WorkBound,
    pub blocks: usize,
    pub started: bool,
    pub swapped: bool,
    pub restarted: bool,
    pub large_batch: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RebuildProgress {
    pub source_generation: u64,
    pub paid_changes: usize,
    pub command: usize,
    pub commands: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourcePlan {
    pub upper_bytes: u64,
    pub resident_bytes: u64,
    pub workspace_reserve: u64,
    pub transaction_bytes: u64,
    pub tile_len: usize,
    /// Additional host envelope for immutable captured bundles and rollback copies.
    pub frozen_linear_bytes:u64,
    /// Bounded temporary host vectors/encodings for the global G2 alternative.
    pub global_g2_host_bytes:u64,
    /// Existing native transient quota envelope, covered by workspace_reserve;
    /// this is not a new retained allocation or a measured allocator peak.
    pub global_g2_workspace_bytes:u64,
    /// Host-only fixed-root preparation, including bounded transient clones.
    pub cache_preparation_host_bytes:u64,
    /// Additional bounded host read/decode/undo staging, not device allocation.
    pub cache_readback_host_bytes:u64,
}

/// Initialization cost and requested retained allocation; excludes allocator
/// metadata/RSS and is not device memory. No acceptance result is retained.
#[derive(Clone,Copy,Debug,Default,PartialEq,Eq)]
pub struct CachePreparationStats {
    pub roots:usize,
    pub prepared_entries:usize,
    pub retained_bytes:u64,
    pub host_upper_bytes:u64,
    pub precompute_ns:u128,
}

/// Fixed-size latency telemetry. Quantiles are bucket upper bounds, not exact
/// percentiles; benchmark tools may keep exact samples outside the hot session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockTimings {
    pub count: u64,
    pub total_ns: u128,
    pub max_ns: u64,
    bins: [u64;64],
}

impl Default for BlockTimings {
    fn default()->Self { Self {count:0,total_ns:0,max_ns:0,bins:[0;64]} }
}

impl BlockTimings {
    pub(crate) fn record(&mut self,duration:std::time::Duration) {
        let ns=duration.as_nanos().min(u64::MAX as u128) as u64;
        let bin=if ns==0 {0} else {63-ns.leading_zeros() as usize};
        self.count=self.count.saturating_add(1);
        self.total_ns=self.total_ns.saturating_add(ns as u128);
        self.max_ns=self.max_ns.max(ns);
        self.bins[bin]=self.bins[bin].saturating_add(1);
    }
    pub fn percentile_upper_ns(&self,percent:u8)->Option<u64> {
        if self.count==0 || percent==0 || percent>100 {return None;}
        let target=((self.count as u128)*(percent as u128)).div_ceil(100);
        let mut seen=0u128;
        for i in 0..64 {
            seen+=self.bins[i] as u128;
            if seen>=target {
                return Some(if i==63 {u64::MAX} else {(1u64 << (i+1))-1});
            }
        }
        None
    }
}
