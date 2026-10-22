//! Fixed-layout diagnostic output; no GPU algorithm or persistent resources.

pub const TIMELINE_PHASES: [&str; 9] = ["allocation_preparation", "point_h2d", "scalar_h2d",
    "kernel_submission", "synchronization", "output_d2h", "remaining_overhead", "metadata_d2h", "free"];
pub const TIMELINE_KERNELS: [&str; 10] = ["split_scalars", "sort", "encode", "scan", "initialize",
    "reserved", "accumulate", "big_triangle", "final_accumulate", "other"];

/// repr(C) and fixed u64 arrays match DvgpuMsmTimeline. Wall scopes partition
/// inclusive; events measure stream intervals and may overlap across streams.
/// Event and wall durations must never be added as one cost decomposition.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MsmTimeline {
    pub wall_ns: [u64; 9],
    pub event_ns: [u64; 9],
    pub calls: [u64; 9],
    pub event_calls: [u64; 9],
    pub bytes: [u64; 9],
    pub kernel_ns: [u64; 10],
    pub kernel_calls: [u64; 10],
    pub instrumentation_inclusive_ns: u64,
    pub instrumentation_total_ns: u64,
    pub c: u64,
    pub bitsize: u64,
    pub windows: u64,
    pub buckets: u64,
}
