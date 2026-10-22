//! CSV output for diagnostics only. Native events and wall clocks stay separate.
use crate::msm_inventory::MsmCall;
use crate::msm_profile::CallProfile;
use anyhow::Result;
use dynaverse_gpu::{TIMELINE_PHASES, TIMELINE_KERNELS};
use std::io::Write;

pub const HEADER: &str = "commit,source_hash,source_dirty,n,policy,workload,verifier,batch_open_mode,kzg_backend,operation,step,call,length,nonzero_scalars,max_scalar_bits,dispatch,domain,phase,sample,elapsed_us,calls,bytes,cpu_match,nvtx_label,warmup,samples,c,bitsize,windows,buckets,icicle_large_bucket_factor";

// Borrow immutable case metadata while serializing one successful sequence.
pub struct Metadata<'a> {
    pub n: usize,
    pub policy: &'a str,
    pub workload: &'a str,
    pub verifier: &'a str,
    pub batch: &'a str,
    pub backend: &'a str,
    pub sample: usize,
    pub warmup: usize,
    pub samples: usize,
    pub large_bucket_factor: u32,
}

fn row(writer: &mut dyn Write, case: &Metadata<'_>, call: &MsmCall, profile: &CallProfile,
    domain: &str, phase: &str, ns: u64, count: u64, bytes: u64) -> Result<()> {
    let timeline = profile.timeline.unwrap_or_default();
    let factor = if profile.dispatch == "cpu" { 0 } else { case.large_bucket_factor };
    writeln!(writer, "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.3},{},{},true,{},{},{},{},{},{},{},{}",
        env!("DYNAVERSE_BUILD_COMMIT"), env!("DYNAVERSE_SOURCE_HASH"), env!("DYNAVERSE_SOURCE_DIRTY"),
        case.n, case.policy, case.workload, case.verifier, case.batch, case.backend,
        call.operation, call.step, call.call, call.length, call.nonzero_scalars, call.max_scalar_bits,
        profile.dispatch, domain, phase, case.sample, ns as f64 / 1000.0, count, bytes, profile.label,
        case.warmup, case.samples, timeline.c, timeline.bitsize, timeline.windows, timeline.buckets, factor)?;
    Ok(())
}

pub fn write_call(writer: &mut dyn Write, case: &Metadata<'_>, call: &MsmCall, profile: &CallProfile) -> Result<()> {
    let spans = profile.spans;
    for (phase, elapsed) in [("total", spans.total), ("validation", spans.validation), ("encoding", spans.encoding),
        ("native", spans.native), ("decode", spans.decode), ("native_conversion", spans.native_conversion),
        ("icicle_inclusive", spans.icicle_inclusive), ("native_output", spans.native_output)] {
        if profile.dispatch == "cpu" && phase != "total" { continue; }
        row(writer, case, call, profile, "wall", phase, elapsed.as_nanos() as u64, 1, 0)?;
    }
    if let Some(timeline) = profile.timeline {
        for i in 0..TIMELINE_PHASES.len() {
            let phase = TIMELINE_PHASES[i];
            row(writer, case, call, profile, "wall", phase, timeline.wall_ns[i], timeline.calls[i], timeline.bytes[i])?;
            if timeline.event_calls[i] > 0 {
                let phase = if phase == "kernel_submission" { "kernel_execution" } else { phase };
                row(writer, case, call, profile, "gpu_event", phase, timeline.event_ns[i], timeline.event_calls[i], timeline.bytes[i])?;
            }
        }
        for i in 0..TIMELINE_KERNELS.len() {
            if timeline.kernel_calls[i] == 0 { continue; }
            let phase = format!("kernel_{}", TIMELINE_KERNELS[i]);
            row(writer, case, call, profile, "gpu_event", &phase, timeline.kernel_ns[i], timeline.kernel_calls[i], 0)?;
        }
        row(writer, case, call, profile, "wall", "instrumentation_inclusive", timeline.instrumentation_inclusive_ns, 1, 0)?;
        row(writer, case, call, profile, "wall", "instrumentation_total", timeline.instrumentation_total_ns, 1, 0)?;
        // This residual includes context/stream setup, event readout, and other
        // work outside inclusive/conversion/output. It nests inside native.
        let residual = spans.native.saturating_sub(spans.native_conversion + spans.icicle_inclusive + spans.native_output);
        row(writer, case, call, profile, "wall", "native_remaining", residual.as_nanos() as u64, 1, 0)?;
    }
    Ok(())
}
