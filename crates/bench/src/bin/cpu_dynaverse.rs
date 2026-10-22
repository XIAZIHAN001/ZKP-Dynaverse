//! Full CPU proof/update sequences, including cache preparation and rebuilds.
// This binary's private diagnostics live one directory above bin/.
#[path = "../msm_inventory.rs"]
mod msm_inventory;
use msm_inventory::InventoryBackend;
#[cfg(feature = "icicle")]
#[path = "../msm_profile.rs"]
mod msm_profile;
#[cfg(feature = "icicle")]
#[path = "../msm_timeline.rs"]
mod msm_timeline;
#[cfg(feature = "icicle")]
use dynaverse_bench::operator_calibration;
use anyhow::{ensure, Result};
use ark_bn254::Fr;
use clap::{Parser, ValueEnum};
use dynaverse_protocol::dynaverse::DynaverseParams;
use dynaverse_protocol::toy_circuit::ToyCircuit;
use dynaverse_protocol::trace::{BatchOpenPhase, CpuPhase, CpuTrace, PairingCounts, VerificationPhase, VerificationTrace};
use dynaverse_protocol::{AnchorOpeningPolicy, BatchOpenMode, Instance, OperatorBackend, Update, VerificationMode, Witness};
use rand::{rngs::StdRng, SeedableRng};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

// ValueEnum restricts CLI input to the named variants; it does not enter the
// protocol crate or change the protocol's policy type.
#[derive(Clone, Copy, ValueEnum)]
enum Policy { Dense, Precompute }

impl Policy {
    fn name(self) -> &'static str {
        match self { Self::Dense => "dense", Self::Precompute => "precompute" }
    }

    fn protocol(self) -> AnchorOpeningPolicy {
        match self {
            Self::Dense => AnchorOpeningPolicy::Dense,
            Self::Precompute => AnchorOpeningPolicy::Precompute,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum Workload { Hotspot, Spreading }

impl Workload {
    fn name(self) -> &'static str {
        match self { Self::Hotspot => "hotspot", Self::Spreading => "spreading" }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum Verifier { Reference, Product }

impl Verifier {
    fn name(self) -> &'static str {
        match self { Self::Reference => "reference", Self::Product => "product" }
    }

    fn protocol(self) -> VerificationMode {
        match self { Self::Reference => VerificationMode::Reference, Self::Product => VerificationMode::Product }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum BatchMode { Reference, Prepared }

impl BatchMode {
    fn name(self) -> &'static str {
        match self { Self::Reference => "reference", Self::Prepared => "prepared" }
    }

    fn protocol(self) -> BatchOpenMode {
        match self { Self::Reference => BatchOpenMode::Reference, Self::Prepared => BatchOpenMode::Prepared }
    }
}

#[derive(Clone, Copy)]
struct RunMode { verifier: Verifier, batch: BatchMode, kzg: KzgBackend }

#[derive(Clone, Copy, ValueEnum)]
enum KzgBackend { Cpu, Icicle, Hybrid }

impl KzgBackend {
    fn name(self) -> &'static str {
        match self { Self::Cpu => "cpu", Self::Icicle => "icicle", Self::Hybrid => "hybrid" }
    }

    fn csv_name(self, gpu_min_points: usize, calibrated: bool) -> String {
        match self {
            Self::Hybrid if calibrated => "hybrid_calibrated".to_owned(),
            Self::Hybrid => format!("hybrid_{gpu_min_points}"),
            _ => self.name().to_owned(),
        }
    }
}

// Main owns both adapters for all sequences. These trait references only
// borrow them, and copying this struct does not copy a native resource.
#[derive(Clone, Copy)]
struct BackendRefs<'a> {
    gpu: Option<&'a dyn OperatorBackend>,
    hybrid: Option<&'a dyn OperatorBackend>,
}

#[derive(Parser)]
struct Args {
    #[arg(long)]
    n: usize,
    #[arg(long, default_value_t = 8)]
    updates: usize,
    #[arg(long, value_enum, value_delimiter = ',', default_value = "dense,precompute")]
    policies: Vec<Policy>,
    #[arg(long, value_enum, value_delimiter = ',', default_value = "hotspot,spreading")]
    workloads: Vec<Workload>,
    #[arg(long, value_enum, value_delimiter = ',', default_value = "product")]
    verifiers: Vec<Verifier>,
    #[arg(long, value_enum, value_delimiter = ',', default_value = "prepared")]
    batch_open_modes: Vec<BatchMode>,
    #[arg(long, value_enum, value_delimiter = ',', default_value = "cpu")]
    kzg_backends: Vec<KzgBackend>,
    /// Point-count minimum for the explicitly selected hybrid mode.
    #[arg(long, default_value_t = 512)]
    gpu_min_points: usize,
    /// Explicit profile from bench_operators; hardware/source/factor must match.
    #[arg(long)]
    dispatch_calibration: Option<PathBuf>,
    /// Per-owner ICICLE threshold multiplier (0 disables large buckets).
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u32).range(0..=100))]
    icicle_large_bucket_factor: u32,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 3)]
    samples: usize,
    #[arg(long)]
    output: PathBuf,
    /// Raw complete-sequence trials; excludes warmups/setup/diagnostics.
    #[arg(long)]
    sequence_samples_output: Option<PathBuf>,
    /// One diagnostic CPU sequence per matrix case, after all timed samples.
    #[arg(long)]
    msm_output: Option<PathBuf>,
    /// Separate profiled protocol sequences, with untimed per-MSM CPU checks.
    #[arg(long)]
    msm_profile_output: Option<PathBuf>,
    /// Separate native CUDA-event/NVTX diagnosis of real protocol calls.
    #[arg(long)]
    msm_timeline_output: Option<PathBuf>,
}

fn reject_diagnostic_aliases(args: &Args) -> Result<()> {
    let paths = [Some(&args.output), args.sequence_samples_output.as_ref(), args.msm_output.as_ref(), args.msm_profile_output.as_ref(), args.msm_timeline_output.as_ref(), args.dispatch_calibration.as_ref()];
    for i in 0..paths.len() {
        for j in i + 1..paths.len() {
            if let (Some(first), Some(second)) = (paths[i], paths[j]) {
                msm_inventory::reject_output_alias(first, second)?;
            }
        }
    }
    Ok(())
}

fn prepare_output_paths(args: &Args) -> Result<()> {
    // An absent intermediate directory can hide aliases such as missing/../x.
    // Resolve every output parent before checking aliases or truncating files.
    for path in [Some(&args.output), args.sequence_samples_output.as_ref(), args.msm_output.as_ref(), args.msm_profile_output.as_ref(), args.msm_timeline_output.as_ref()] {
        if let Some(path) = path {
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() { std::fs::create_dir_all(parent)?; }
            }
        }
    }
    reject_diagnostic_aliases(args)
}

// Operation labels refer to fixed literals; phase labels own their text because
// verification subphase names also include generated suffixes.
struct Measurement {
    operation: &'static str,
    step: usize,
    k: usize,
    rebuilds: usize,
    phase: String,
    duration: Duration,
    counts: PairingCounts,
}

#[allow(clippy::too_many_arguments)]
fn append_operation(
    rows: &mut Vec<Measurement>, operation: &'static str,
    step: usize, k: usize, rebuilds: usize, total: Duration, trace: &CpuTrace,
) -> Result<()> {
    ensure!(trace.recorded_total() <= total, "trace intervals overlap at {operation}/{step}");
    rows.push(Measurement { operation, step, k, rebuilds, phase: "total".to_owned(), duration: total,
        counts: trace.verification().counts() });
    for phase in CpuPhase::ALL {
        let counts = if matches!(phase, CpuPhase::PairingVerify) { trace.verification().counts() }
            else { PairingCounts::default() };
        rows.push(Measurement { operation, step, k, rebuilds, phase: phase.name().to_owned(), duration: trace.elapsed(phase), counts });
    }
    rows.push(Measurement { operation, step, k, rebuilds, phase: "other".to_owned(), duration: total - trace.recorded_total(), counts: PairingCounts::default() });
    append_verification_phases(rows, operation, step, k, rebuilds, trace.verification());
    ensure!(trace.batch_open().recorded_total() <= trace.elapsed(CpuPhase::BatchOpen),
        "batch opening trace intervals overlap");
    for phase in BatchOpenPhase::ALL {
        rows.push(Measurement { operation, step, k, rebuilds, phase: phase.name().to_owned(),
            duration: trace.batch_open().elapsed(phase), counts: PairingCounts::default() });
    }
    Ok(())
}

fn append_verification_phases(
    rows: &mut Vec<Measurement>, operation: &'static str, step: usize, k: usize,
    rebuilds: usize, trace: &VerificationTrace,
) {
    for phase in VerificationPhase::ALL {
        let measured = trace.phase(phase);
        rows.push(Measurement { operation, step, k, rebuilds, phase: phase.name().to_owned(),
            duration: measured.elapsed, counts: measured.counts });
        rows.push(Measurement { operation, step, k, rebuilds, phase: format!("{}_miller", phase.name()),
            duration: measured.miller_time, counts: PairingCounts::default() });
        rows.push(Measurement { operation, step, k, rebuilds, phase: format!("{}_final_exp", phase.name()),
            duration: measured.final_exponentiation_time, counts: PairingCounts::default() });
    }
}

fn append_verify(
    rows: &mut Vec<Measurement>, step: usize, k: usize, rebuilds: usize,
    total: Duration, trace: &VerificationTrace,
) -> Result<()> {
    ensure!(trace.recorded_total() <= total, "verification trace intervals overlap");
    rows.push(Measurement { operation: "verify", step, k, rebuilds, phase: "total".to_owned(), duration: total, counts: trace.counts() });
    append_verification_phases(rows, "verify", step, k, rebuilds, trace);
    rows.push(Measurement { operation: "verify", step, k, rebuilds, phase: "verify_other".to_owned(),
        duration: total - trace.recorded_total(), counts: PairingCounts::default() });
    Ok(())
}

fn make_update(n: usize, step: usize, workload: Workload, witness: &Witness) -> Update {
    match workload {
        Workload::Hotspot => {
            // s4[0] is zero, so changing s5[0] leaves the product relation valid.
            let index = 4 * n;
            Update { indices: vec![index], values: vec![witness.wires[index] + Fr::from(1u64)] }
        }
        Workload::Spreading => {
            let gate = 1 + (step - 1) % (n - 1);
            let multiplier = 4 * n + gate;
            let product = 5 * n + gate;
            let new_multiplier = witness.wires[multiplier] + Fr::from(1u64);
            let new_product = witness.wires[3 * n + gate] * new_multiplier;
            Update { indices: vec![multiplier, product], values: vec![new_multiplier, new_product] }
        }
    }
}

fn run_sequence(
    params: &DynaverseParams, instance: &Instance, witness: &Witness,
    n: usize, updates: usize, policy: Policy, workload: Workload, mode: RunMode,
    backends: BackendRefs<'_>,
    inventory: Option<&InventoryBackend<'_>>,
) -> Result<Vec<Measurement>> {
    // GPU modes borrow their adapters. CPU mode goes through the
    // original KZG API, preserving its actual arkworks performance baseline.
    let backend = if let Some(inventory) = inventory {
        Some(inventory as &dyn OperatorBackend)
    } else { match mode.kzg {
        KzgBackend::Cpu => None,
        KzgBackend::Icicle => Some(backends.gpu.ok_or_else(|| anyhow::anyhow!("ICICLE backend unavailable"))?),
        KzgBackend::Hybrid => Some(backends.hybrid.ok_or_else(|| anyhow::anyhow!("hybrid backend unavailable"))?),
    } };
    let vk = params.verifying_key();
    let mut rows = Vec::new();
    let sequence_started = Instant::now();
    let mut trace = CpuTrace::new();
    if let Some(inventory) = inventory { inventory.begin("prove", 0); }
    let started = Instant::now();
    let (mut proof, mut aux) = match backend {
        Some(backend) => params.prove_with_aux_traced_with_backend(instance, witness, policy.protocol(), backend, &mut trace)?,
        None => params.prove_with_aux_traced(instance, witness, policy.protocol(), &mut trace)?,
    };
    append_operation(&mut rows, "prove", 0, 0, 0, started.elapsed(), &trace)?;
    let mut verification = VerificationTrace::new();
    let started = Instant::now();
    ensure!(vk.verify_traced(instance, &proof, mode.verifier.protocol(), &mut verification)?, "initial proof was rejected");
    append_verify(&mut rows, 0, 0, 0, started.elapsed(), &verification)?;
    let mut total_rebuilds = 0;

    for step in 1..=updates {
        // This owned snapshot permits the following exclusive mutable borrow
        // of aux; an &Witness borrowed from aux would remain tied to it.
        let old_witness = aux.current_witness().clone();
        let update = make_update(n, step, workload, &old_witness);
        let k = update.k();
        let old_anchor_generation = proof.anchor_generation;
        trace = CpuTrace::new();
        if let Some(inventory) = inventory { inventory.begin("update", step); }
        let started = Instant::now();
        proof = match backend {
            Some(backend) => params.update_proof_traced_with_backend(instance, instance, &old_witness,
                &update, &proof, &mut aux, mode.verifier.protocol(), mode.batch.protocol(), backend, &mut trace)?,
            None => params.update_proof_traced_with_modes(instance, instance, &old_witness,
                &update, &proof, &mut aux, mode.verifier.protocol(), mode.batch.protocol(), &mut trace)?,
        };
        let update_elapsed = started.elapsed();
        let mut rebuilt = 0;
        if proof.anchor_generation != old_anchor_generation {
            rebuilt = 1;
            total_rebuilds += 1;
        }
        append_operation(&mut rows, "update", step, k, rebuilt, update_elapsed, &trace)?;
        verification = VerificationTrace::new();
        let started = Instant::now();
        ensure!(vk.verify_traced(instance, &proof, mode.verifier.protocol(), &mut verification)?, "updated proof was rejected at step {step}");
        append_verify(&mut rows, step, k, rebuilt, started.elapsed(), &verification)?;
    }
    rows.push(Measurement { operation: "sequence", step: updates, k: 0,
        rebuilds: total_rebuilds, phase: "total".to_owned(), duration: sequence_started.elapsed(), counts: PairingCounts::default() });
    Ok(rows)
}

fn cpu_model() -> String {
    if let Ok(contents) = std::fs::read_to_string("/proc/cpuinfo") {
        for line in contents.lines() {
            if line.starts_with("model name") {
                if let Some((_, value)) = line.split_once(':') {
                    return value.trim().replace(',', " ");
                }
            }
        }
    }
    std::env::consts::ARCH.to_owned()
}

fn percentiles(mut durations: Vec<f64>) -> (f64, f64) {
    // The caller moves its sample vector here so sorting needs no extra clone.
    durations.sort_by(f64::total_cmp);
    let count = durations.len();
    let mid = count / 2;
    let median = if count % 2 == 0 {
        (durations[mid - 1] + durations[mid]) / 2.0
    } else { durations[mid] };
    let p95 = durations[((count as f64 * 0.95).ceil() as usize) - 1];
    (median, p95)
}

#[allow(clippy::too_many_arguments)]
fn write_row(
    writer: &mut BufWriter<File>, cpu: &str, n: usize, policy: &str, workload: &str, verifier: &str,
    batch_mode: &str, kzg_backend: &str, row: &Measurement, samples: usize, median: f64, p95: f64,
    large_bucket_factor: u32,
) -> Result<()> {
    let factor = if kzg_backend == "cpu" || kzg_backend == "shared" { 0 } else { large_bucket_factor };
    writeln!(writer, "{},{},{},{},{},{},{},{},{},{},{},{:.3},{:.3},{},{},{},{},{},{},{},{},{},{}",
        env!("DYNAVERSE_BUILD_COMMIT"), cpu, n, policy, workload,
        row.operation, row.step, row.k, row.rebuilds, row.phase, samples, median, p95,
        env!("DYNAVERSE_SOURCE_HASH"), env!("DYNAVERSE_SOURCE_DIRTY"), verifier, row.counts.pairing_terms,
        row.counts.miller_loops, row.counts.final_exponentiations, row.counts.equations, batch_mode, kzg_backend, factor)?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.n >= 2 && args.n.is_power_of_two(), "n must be a power of two >= 2");
    ensure!(args.updates > 0, "updates must be positive");
    ensure!(args.samples > 0, "samples must be positive");
    ensure!(!args.policies.is_empty() && !args.workloads.is_empty() && !args.verifiers.is_empty()
        && !args.batch_open_modes.is_empty() && !args.kzg_backends.is_empty(), "matrix must not be empty");
    reject_diagnostic_aliases(&args)?;
    let want_hybrid = args.kzg_backends.iter().any(|backend| matches!(backend, KzgBackend::Hybrid));
    ensure!(args.dispatch_calibration.is_none() || want_hybrid, "dispatch calibration requires hybrid mode");
    if want_hybrid {
        ensure!(args.gpu_min_points > 0 && args.gpu_min_points <= 1 << 20,
            "hybrid GPU minimum must be within 1..=2^20");
    }
    let want_gpu = args.kzg_backends.iter().any(|backend| matches!(backend, KzgBackend::Icicle | KzgBackend::Hybrid));
    ensure!(args.icicle_large_bucket_factor == 0 || want_gpu, "large bucket factor requires an ICICLE or hybrid mode");
    ensure!(args.msm_profile_output.is_none() || want_gpu, "MSM profiling requires an ICICLE or hybrid mode");
    ensure!(args.msm_timeline_output.is_none() || want_gpu, "MSM timeline requires an ICICLE or hybrid mode");
    let init_started = Instant::now();
    #[cfg(feature = "icicle")]
    let gpu = if want_gpu { Some(dynaverse_gpu::IcicleBackend::with_large_bucket_factor(args.icicle_large_bucket_factor)?) } else { None };
    #[cfg(not(feature = "icicle"))]
    let gpu: Option<dynaverse_gpu::CpuBackend> = {
        ensure!(!want_gpu, "ICICLE mode requires cargo --features icicle");
        None
    };
    let gpu_init = init_started.elapsed();
    #[cfg(feature = "icicle")]
    let hybrid = if want_hybrid {
        let owner = gpu.as_ref().ok_or_else(|| anyhow::anyhow!("GPU owner unavailable"))?;
        Some(match &args.dispatch_calibration {
            Some(path) => dynaverse_gpu::HybridBackend::with_thresholds(owner,
                operator_calibration::load(path, owner, env!("DYNAVERSE_SOURCE_HASH"))?)?,
            None => dynaverse_gpu::HybridBackend::new(owner, args.gpu_min_points)?,
        })
    } else { None };
    #[cfg(not(feature = "icicle"))]
    let hybrid: Option<dynaverse_gpu::CpuBackend> = None;
    prepare_output_paths(&args)?;
    // The concrete backend is owned until all sequences finish. The trait
    // object below is a short-lived borrow and cannot outlive that owner.
    let gpu_ref = gpu.as_ref().map(|backend| backend as &dyn OperatorBackend);
    let hybrid_ref = hybrid.as_ref().map(|backend| backend as &dyn OperatorBackend);
    let backends = BackendRefs { gpu: gpu_ref, hybrid: hybrid_ref };
    let mut modes = Vec::new();
    for &verifier in &args.verifiers {
        for &batch in &args.batch_open_modes {
            for &kzg in &args.kzg_backends { modes.push(RunMode { verifier, batch, kzg }); }
        }
    }
    let circuit = ToyCircuit::new(args.n)?;
    let (instance, witness) = circuit.valid_instance();
    let mut rng = StdRng::seed_from_u64(2812);
    let started = Instant::now();
    let mut setup_trace = CpuTrace::new();
    let params = DynaverseParams::setup_traced(circuit, &mut rng, &mut setup_trace)?;
    let setup_elapsed = started.elapsed();
    let cpu = cpu_model();
    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() { std::fs::create_dir_all(parent)?; }
    }
    reject_diagnostic_aliases(&args)?;
    let mut writer = BufWriter::new(File::create(&args.output)?);
    // Creating the timing file may resolve a formerly dangling symlink.
    reject_diagnostic_aliases(&args)?;
    writeln!(writer, "commit,cpu,n,policy,workload,operation,step,k,rebuilds,phase,samples,median_us,p95_us,source_hash,source_dirty,verifier,pairing_terms,miller_loops,final_exponentiations,equations,batch_open_mode,kzg_backend,icicle_large_bucket_factor")?;
    let mut raw_writer = if let Some(path) = &args.sequence_samples_output {
        let mut raw = BufWriter::new(File::create(path)?);
        reject_diagnostic_aliases(&args)?;
        writeln!(raw, "commit,source_hash,source_dirty,n,policy,workload,verifier,batch_open_mode,kzg_backend,icicle_large_bucket_factor,sample,elapsed_us,updates,warmup,samples")?;
        Some(raw)
    } else { None };
    let mut setup_rows = Vec::new();
    append_operation(&mut setup_rows, "setup", 0, 0, 0, setup_elapsed, &setup_trace)?;
    for row in &setup_rows {
        let micros = row.duration.as_secs_f64() * 1e6;
        write_row(&mut writer, &cpu, args.n, "shared", "shared", "shared", "shared", "shared", row, 1, micros, micros, args.icicle_large_bucket_factor)?;
    }
    if want_gpu {
        let row = Measurement { operation: "backend_init", step: 0, k: 0, rebuilds: 0,
            phase: "total".to_owned(), duration: gpu_init, counts: PairingCounts::default() };
        let micros = gpu_init.as_secs_f64() * 1e6;
        write_row(&mut writer, &cpu, args.n, "shared", "shared", "shared", "shared", "icicle", &row, 1, micros, micros, args.icicle_large_bucket_factor)?;
    }

    // Borrow the policies so the optional diagnostic pass can reuse them.
    for &policy in &args.policies {
        for workload in &args.workloads {
            // Report the first execution of each matrix case separately from
            // warm samples. Only the process's first ICICLE case is globally
            // cold; later cases can reuse CUDA module loading and memory pools.
            for &mode in &modes {
                let kzg_label = mode.kzg.csv_name(args.gpu_min_points, args.dispatch_calibration.is_some());
                let first = run_sequence(&params, &instance, &witness, args.n,
                    args.updates, policy, *workload, mode, backends, None)?;
                let total = first.last().ok_or_else(|| anyhow::anyhow!("missing first sequence"))?;
                let row = Measurement { operation: "first_sequence", step: total.step,
                    k: total.k, rebuilds: total.rebuilds, phase: "total".to_owned(),
                    duration: total.duration, counts: total.counts };
                let micros = row.duration.as_secs_f64() * 1e6;
                write_row(&mut writer, &cpu, args.n, policy.name(), workload.name(),
                    mode.verifier.name(), mode.batch.name(), &kzg_label, &row, 1, micros, micros, args.icicle_large_bucket_factor)?;
            }
            for warmup in 0..args.warmup {
                for offset in 0..modes.len() {
                    let mode = modes[(warmup + offset) % modes.len()];
                    run_sequence(&params, &instance, &witness, args.n, args.updates, policy, *workload, mode, backends, None)?;
                }
            }
            // One bucket per mode pair; each trial contains one complete sequence
            // of measurement rows. Rotate execution order to balance order bias.
            let mut trials = Vec::with_capacity(modes.len());
            for _ in &modes { trials.push(Vec::with_capacity(args.samples)); }
            for sample in 0..args.samples {
                for offset in 0..modes.len() {
                    let index = (sample + offset) % modes.len();
                    trials[index].push(run_sequence(&params, &instance, &witness,
                        args.n, args.updates, policy, *workload, modes[index], backends, None)?);
                }
            }
            for mode_index in 0..modes.len() {
                let mode = modes[mode_index];
                let kzg_label = mode.kzg.csv_name(args.gpu_min_points, args.dispatch_calibration.is_some());
                let mode_trials = &trials[mode_index];
                if let Some(raw) = &mut raw_writer {
                    let factor = if matches!(mode.kzg, KzgBackend::Cpu) { 0 } else { args.icicle_large_bucket_factor };
                    for (sample, trial) in mode_trials.iter().enumerate() {
                        let total = trial.last().ok_or_else(|| anyhow::anyhow!("missing sequence total"))?;
                        ensure!(total.operation == "sequence" && total.phase == "total", "invalid sequence sample");
                        writeln!(raw, "{},{},{},{},{},{},{},{},{},{},{},{:.3},{},{},{}",
                            env!("DYNAVERSE_BUILD_COMMIT"), env!("DYNAVERSE_SOURCE_HASH"), env!("DYNAVERSE_SOURCE_DIRTY"),
                            args.n, policy.name(), workload.name(), mode.verifier.name(), mode.batch.name(), kzg_label,
                            factor, sample, total.duration.as_secs_f64() * 1e6, args.updates, args.warmup, args.samples)?;
                    }
                    raw.flush()?;
                }
                for index in 0..mode_trials[0].len() {
                    let reference = &mode_trials[0][index];
                    let mut durations = Vec::with_capacity(args.samples);
                    for trial in mode_trials {
                        ensure!(trial.len() == mode_trials[0].len(), "trial row count changed");
                        let row = &trial[index];
                        ensure!(row.operation == reference.operation && row.phase == reference.phase
                            && row.step == reference.step && row.k == reference.k
                            && row.rebuilds == reference.rebuilds && row.counts == reference.counts,
                            "trial metadata changed");
                        durations.push(row.duration.as_secs_f64() * 1e6);
                    }
                    let (median, p95) = percentiles(durations);
                    write_row(&mut writer, &cpu, args.n, policy.name(), workload.name(), mode.verifier.name(),
                        mode.batch.name(), &kzg_label, reference, args.samples, median, p95, args.icicle_large_bucket_factor)?;
                }
            }
            writer.flush()?;
            eprintln!("SEQUENCE_OK n={} policy={} workload={} modes={} samples={}",
                args.n, policy.name(), workload.name(), modes.len(), args.samples);
        }
    }
    if let Some(path) = &args.msm_output {
        reject_diagnostic_aliases(&args)?;
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() { std::fs::create_dir_all(parent)?; }
        }
        reject_diagnostic_aliases(&args)?;
        let mut diagnostic = BufWriter::new(File::create(path)?);
        reject_diagnostic_aliases(&args)?;
        writeln!(diagnostic, "commit,source_hash,source_dirty,n,policy,workload,verifier,batch_open_mode,operation,step,call,length,nonzero_scalars,max_scalar_bits")?;
        for &policy in &args.policies {
            for &workload in &args.workloads {
                for &verifier in &args.verifiers {
                    for &batch in &args.batch_open_modes {
                        let mode = RunMode { verifier, batch, kzg: KzgBackend::Cpu };
                        let observer = InventoryBackend::new(&dynaverse_gpu::ArkworksBackend);
                        run_sequence(&params, &instance, &witness, args.n, args.updates,
                            policy, workload, mode, backends, Some(&observer))?;
                        let calls = observer.take_calls();
                        for call in &calls {
                            writeln!(diagnostic, "{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                                env!("DYNAVERSE_BUILD_COMMIT"), env!("DYNAVERSE_SOURCE_HASH"), env!("DYNAVERSE_SOURCE_DIRTY"),
                                args.n, policy.name(), workload.name(), verifier.name(), batch.name(),
                                call.operation, call.step, call.call, call.length, call.nonzero_scalars, call.max_scalar_bits)?;
                        }
                        eprintln!("MSM_INVENTORY_OK n={} policy={} workload={} calls={}", args.n, policy.name(), workload.name(), calls.len());
                    }
                }
            }
        }
        diagnostic.flush()?;
    }
    #[cfg(feature = "icicle")]
    if let Some(path) = &args.msm_profile_output {
        reject_diagnostic_aliases(&args)?;
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() { std::fs::create_dir_all(parent)?; }
        }
        reject_diagnostic_aliases(&args)?;
        let mut diagnostic = BufWriter::new(File::create(path)?);
        writeln!(diagnostic, "commit,source_hash,source_dirty,n,policy,workload,verifier,batch_open_mode,kzg_backend,operation,step,call,length,nonzero_scalars,max_scalar_bits,dispatch,phase,sample,elapsed_us,warmup,samples,cpu_match,icicle_large_bucket_factor")?;
        let gpu_owner = gpu.as_ref().ok_or_else(|| anyhow::anyhow!("GPU owner unavailable"))?;
        for &policy in &args.policies {
            for &workload in &args.workloads {
                for &mode in &modes {
                    if matches!(mode.kzg, KzgBackend::Cpu) { continue; }
                    let selected_hybrid = if matches!(mode.kzg, KzgBackend::Hybrid) { hybrid.as_ref() } else { None };
                    let label = mode.kzg.csv_name(args.gpu_min_points, args.dispatch_calibration.is_some());
                    for trial in 0..args.warmup + args.samples {
                        let profiler = msm_profile::ProfileBackend::new(gpu_owner, selected_hybrid);
                        let observer = InventoryBackend::new(&profiler);
                        run_sequence(&params, &instance, &witness, args.n, args.updates,
                            policy, workload, mode, backends, Some(&observer))?;
                        if trial < args.warmup { continue; }
                        let calls = observer.take_calls();
                        let profiles = profiler.take_profiles();
                        ensure!(calls.len() == profiles.len(), "diagnostic call/profile count mismatch");
                        for i in 0..calls.len() {
                            let call = &calls[i];
                            let profile = &profiles[i];
                            let spans = profile.spans;
                            for (phase, elapsed) in [("total", spans.total), ("validation", spans.validation),
                                ("encoding", spans.encoding), ("native", spans.native), ("decode", spans.decode),
                                ("native_conversion", spans.native_conversion), ("icicle_inclusive", spans.icicle_inclusive),
                                ("native_output", spans.native_output)] {
                                if profile.dispatch == "cpu" && phase != "total" { continue; }
                                let factor = if profile.dispatch == "cpu" { 0 } else { args.icicle_large_bucket_factor };
                                writeln!(diagnostic, "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.3},{},{},true,{}",
                                    env!("DYNAVERSE_BUILD_COMMIT"), env!("DYNAVERSE_SOURCE_HASH"), env!("DYNAVERSE_SOURCE_DIRTY"),
                                    args.n, policy.name(), workload.name(), mode.verifier.name(), mode.batch.name(), label,
                                    call.operation, call.step, call.call, call.length, call.nonzero_scalars, call.max_scalar_bits,
                                    profile.dispatch, phase, trial - args.warmup, elapsed.as_secs_f64() * 1e6, args.warmup, args.samples, factor)?;
                            }
                        }
                    }
                    diagnostic.flush()?;
                    eprintln!("MSM_PROFILE_OK n={} policy={} workload={} backend={} samples={}",
                        args.n, policy.name(), workload.name(), label, args.samples);
                }
            }
        }
    }
    #[cfg(feature = "icicle")]
    if let Some(path) = &args.msm_timeline_output {
        reject_diagnostic_aliases(&args)?;
        let mut diagnostic = BufWriter::new(File::create(path)?);
        writeln!(diagnostic, "{}", msm_timeline::HEADER)?;
        let gpu_owner = gpu.as_ref().ok_or_else(|| anyhow::anyhow!("GPU owner unavailable"))?;
        for &policy in &args.policies {
            for &workload in &args.workloads {
                for &mode in &modes {
                    if matches!(mode.kzg, KzgBackend::Cpu) { continue; }
                    let selected_hybrid = if matches!(mode.kzg, KzgBackend::Hybrid) { hybrid.as_ref() } else { None };
                    let backend_label = mode.kzg.csv_name(args.gpu_min_points, args.dispatch_calibration.is_some());
                    for trial in 0..args.warmup + args.samples {
                        let sample_label = if trial < args.warmup { format!("warmup{trial}") } else { (trial - args.warmup).to_string() };
                        let run_label = format!("dvgpu/n={}/policy={}/workload={}/verifier={}/batch={}/backend={}/sample={}",
                            args.n, policy.name(), workload.name(), mode.verifier.name(), mode.batch.name(), backend_label, sample_label);
                        let profiler = msm_profile::ProfileBackend::new_timeline(gpu_owner, selected_hybrid, run_label);
                        // Both traits borrow the same observer. Notification
                        // occurs before the first call in each operation.
                        let observer = InventoryBackend::with_context(&profiler, &profiler);
                        run_sequence(&params, &instance, &witness, args.n, args.updates,
                            policy, workload, mode, backends, Some(&observer))?;
                        if trial < args.warmup { continue; }
                        let calls = observer.take_calls();
                        let profiles = profiler.take_profiles();
                        ensure!(calls.len() == profiles.len(), "timeline call/profile count mismatch");
                        let metadata = msm_timeline::Metadata { n: args.n, policy: policy.name(), workload: workload.name(),
                            verifier: mode.verifier.name(), batch: mode.batch.name(), backend: &backend_label,
                            sample: trial - args.warmup, warmup: args.warmup, samples: args.samples,
                            large_bucket_factor: args.icicle_large_bucket_factor };
                        for i in 0..calls.len() { msm_timeline::write_call(&mut diagnostic, &metadata, &calls[i], &profiles[i])?; }
                    }
                    diagnostic.flush()?;
                    eprintln!("MSM_TIMELINE_OK n={} policy={} workload={} backend={} samples={}",
                        args.n, policy.name(), workload.name(), backend_label, args.samples);
                }
            }
        }
    }
    Ok(())
}
