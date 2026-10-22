//! Reproducible CPU Dynamo phase timing with setup excluded from prove time.

use anyhow::{bail, ensure, Context, Result};
use ark_bn254::Fr;
use clap::Parser;
use dynaverse_gpu::CpuBackend;
use dynaverse_protocol::dynamo::{
    derive_h_sparse, dynamo_prove_sparse, dynamo_verify, DynamoParams, DynamoPermutation,
    SparseVector,
};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use std::hint::black_box;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    m: usize,
    #[arg(long, default_value = "0..12")]
    k_powers: String,
    #[arg(long, default_value_t = 10)]
    warmup: usize,
    #[arg(long, default_value_t = 30)]
    samples: usize,
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.m > 0 && args.m.is_power_of_two(), "m must be a nonzero power of two");
    ensure!(args.samples > 0, "samples must be positive");
    let ks = parse_k_powers(&args.k_powers, args.m)?;
    let commit = env!("DYNAVERSE_BUILD_COMMIT");
    let cpu = cpu_model();
    let mut rng = StdRng::seed_from_u64(7);
    let permutation = DynamoPermutation::try_new((0..args.m).rev().collect())?;
    let setup_start = Instant::now();
    let params = DynamoParams::setup(args.m, permutation.clone(), &mut rng)?;
    let setup_us = setup_start.elapsed().as_secs_f64() * 1e6;

    if let Some(parent) = args.output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(&args.output)
        .with_context(|| format!("cannot create {}", args.output.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    writeln!(writer, "commit,cpu,m,k,phase,samples,median_us,p95_us,source_hash,source_dirty")?;
    write_row(&mut writer, commit, &cpu, args.m, 0, "setup", 1, setup_us, setup_us)?;

    let backend = CpuBackend;
    for k in ks {
        let z = deterministic_sparse_vector(args.m, k)?;
        let (derive_median, derive_p95) = measure(args.warmup, args.samples, || {
            derive_h_sparse(&z, &permutation)
        })?;
        write_row(&mut writer, commit, &cpu, args.m, k, "derive_h", args.samples, derive_median, derive_p95)?;
        let h = derive_h_sparse(&z, &permutation)?;
        let mut proof_rng = StdRng::seed_from_u64(17 + k as u64);
        let (prove_median, prove_p95) = measure(args.warmup, args.samples, || {
            dynamo_prove_sparse(&params, &z, &h, &backend, &mut proof_rng)
        })?;
        write_row(&mut writer, commit, &cpu, args.m, k, "prove_sparse", args.samples, prove_median, prove_p95)?;
        let proof = dynamo_prove_sparse(&params, &z, &h, &backend, &mut proof_rng)?;
        ensure!(dynamo_verify(&params, &proof)?, "generated proof failed verification at k={k}");
        let (verify_median, verify_p95) = measure(args.warmup, args.samples, || {
            let accepted = dynamo_verify(&params, &proof)?;
            ensure!(accepted, "proof failed during verify timing");
            Ok(accepted)
        })?;
        write_row(&mut writer, commit, &cpu, args.m, k, "verify", args.samples, verify_median, verify_p95)?;
        writer.flush()?;
    }
    Ok(())
}

fn parse_k_powers(spec: &str, m: usize) -> Result<Vec<usize>> {
    let (start, end) = spec.split_once("..").context("k-powers must be START..END")?;
    let start: u32 = start.parse().context("invalid k-powers start")?;
    let end: u32 = end.parse().context("invalid k-powers end")?;
    ensure!(start <= end, "k-powers start exceeds end");
    let mut ks = Vec::new();
    for power in start..=end {
        let k = 1usize.checked_shl(power).context("k power exceeds usize width")?;
        ensure!(k <= m, "k={k} exceeds m={m}");
        ks.push(k);
    }
    Ok(ks)
}

fn deterministic_sparse_vector(m: usize, k: usize) -> Result<SparseVector> {
    let mut indices: Vec<usize> = (0..m).collect();
    let mut rng = StdRng::seed_from_u64(2026);
    indices.shuffle(&mut rng);
    indices.truncate(k);
    indices.sort_unstable();
    let entries = indices
        .into_iter()
        .enumerate()
        .map(|(order, index)| (index, Fr::from((order + 1) as u64)))
        .collect();
    SparseVector::try_new(m, entries)
}

fn measure<T>(warmup: usize, samples: usize, mut operation: impl FnMut() -> Result<T>) -> Result<(f64, f64)> {
    for _ in 0..warmup {
        black_box(operation()?);
    }
    let mut durations = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = Instant::now();
        black_box(operation()?);
        durations.push(start.elapsed().as_secs_f64() * 1e6);
    }
    durations.sort_by(f64::total_cmp);
    if durations.is_empty() {
        bail!("no timing samples");
    }
    let mid = durations.len() / 2;
    let median = if durations.len() % 2 == 0 {
        (durations[mid - 1] + durations[mid]) / 2.0
    } else {
        durations[mid]
    };
    let p95 = durations[((durations.len() as f64 * 0.95).ceil() as usize).saturating_sub(1)];
    Ok((median, p95))
}

#[allow(clippy::too_many_arguments)]
fn write_row(
    writer: &mut impl Write,
    commit: &str,
    cpu: &str,
    m: usize,
    k: usize,
    phase: &str,
    samples: usize,
    median_us: f64,
    p95_us: f64,
) -> Result<()> {
    writeln!(
        writer,
        "{commit},{cpu},{m},{k},{phase},{samples},{median_us:.3},{p95_us:.3},{},{}",
        env!("DYNAVERSE_SOURCE_HASH"),
        env!("DYNAVERSE_SOURCE_DIRTY")
    )?;
    Ok(())
}

fn cpu_model() -> String {
    let model = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|contents| {
            contents
                .lines()
                .find(|line| line.starts_with("model name"))
                .and_then(|line| line.split_once(':'))
                .map(|(_, value)| value.trim().to_owned())
        })
        .unwrap_or_else(|| std::env::consts::ARCH.to_owned());
    model.replace(',', " ")
}
