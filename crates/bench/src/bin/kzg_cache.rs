//! KZG opening microbenchmark; setup and precomputation are not update time.
use anyhow::{ensure, Result};
use ark_bn254::Fr;
use ark_ff::PrimeField;
use clap::Parser;
use dynaverse_protocol::{domain::radix2_domain, kzg::{combine_single_openings, KzgParams}};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use std::{hint::black_box, io::Write, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    n: usize,
    #[arg(long, value_delimiter = ',', default_value = "1,4,16")]
    ks: Vec<usize>,
    #[arg(long, default_value_t = 3)]
    warmup: usize,
    #[arg(long, default_value_t = 10)]
    samples: usize,
    #[arg(long)]
    output: PathBuf,
}

fn measure<T>(warmup: usize, samples: usize, mut operation: impl FnMut() -> Result<T>) -> Result<(f64, f64)> {
    for _ in 0..warmup { black_box(operation()?); }
    let mut durations = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = Instant::now();
        black_box(operation()?);
        durations.push(start.elapsed().as_secs_f64() * 1e6);
    }
    durations.sort_by(f64::total_cmp);
    let mid = durations.len() / 2;
    let median = if durations.len() % 2 == 0 {
        (durations[mid - 1] + durations[mid]) / 2.0
    } else { durations[mid] };
    let p95 = durations[((samples as f64 * 0.95).ceil() as usize) - 1];
    Ok((median, p95))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let domain = radix2_domain(args.n)?;
    ensure!(args.samples > 0, "samples must be positive");
    ensure!(!args.ks.is_empty() && args.ks.iter().all(|&k| k > 0 && k <= args.n), "k must be in 1..=n");
    let mut rng = StdRng::seed_from_u64(2805);
    let params = KzgParams::setup(args.n, &mut rng)?;
    let coefficients: Vec<Fr> = (0..args.n).map(|_| {
        let mut bytes = [0u8; 32];
        rng.fill_bytes(&mut bytes);
        Fr::from_le_bytes_mod_order(&bytes)
    }).collect();
    let commitment = params.commit(&coefficients)?;
    let precompute = measure(args.warmup, args.samples,
        || params.precompute_single_openings(&coefficients, &domain))?;
    let cache = params.precompute_single_openings(&coefficients, &domain)?;
    let cpu = std::fs::read_to_string("/proc/cpuinfo").ok().and_then(|contents| {
        contents.lines().find(|line| line.starts_with("model name"))
            .and_then(|line| line.split_once(':')).map(|(_, value)| value.trim().to_owned())
    }).unwrap_or_else(|| std::env::consts::ARCH.to_owned()).replace(',', " ");
    if let Some(parent) = args.output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = std::io::BufWriter::new(std::fs::File::create(&args.output)?);
    writeln!(writer, "commit,cpu,n,k,phase,samples,median_us,p95_us,source_hash,source_dirty")?;
    let mut row = |k: usize, phase: &str, timing: (f64, f64)| -> Result<()> {
        writeln!(writer, "{},{},{},{},{},{},{:.3},{:.3},{},{}",
            env!("DYNAVERSE_BUILD_COMMIT"), cpu, args.n, k, phase, args.samples, timing.0, timing.1,
            env!("DYNAVERSE_SOURCE_HASH"), env!("DYNAVERSE_SOURCE_DIRTY"))?;
        Ok(())
    };
    row(0, "precompute_fft", precompute)?;
    for k in args.ks {
        // 37 is odd, so this is injective modulo every radix-2 n.
        let indices: Vec<usize> = (0..k).map(|i| (i * 37) % args.n).collect();
        let dense = params.open_batch(&coefficients, &domain, &indices)?;
        ensure!(params.verify_batch(&commitment, &domain, &indices, &dense)?, "dense proof was rejected");
        ensure!(dense.witness == combine_single_openings(&domain, &indices, &cache)?, "cache differs from dense oracle");
        row(k, "dense_batch", measure(args.warmup, args.samples,
            || params.open_batch(&coefficients, &domain, &indices))?)?;
        row(k, "cached_combine", measure(args.warmup, args.samples,
            || combine_single_openings(&domain, &indices, &cache))?)?;
    }
    writer.flush()?;
    Ok(())
}
