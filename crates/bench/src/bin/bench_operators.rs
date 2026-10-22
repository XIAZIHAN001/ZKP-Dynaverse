//! Paired inclusive CPU/GPU operator measurements, independent of SRS setup.
use anyhow::Result;
#[cfg(not(feature = "icicle"))]
fn main() -> Result<()> { anyhow::bail!("build bench_operators with --features icicle") }

#[cfg(feature = "icicle")]
mod measurement {
    use dynaverse_bench::operator_calibration::{sustained_minimum, Summary, SPARSE_MSM_VARIANTS};
    use anyhow::{ensure, Result};
    use ark_bn254::{Fr, G1Affine, G1Projective};
    use ark_ec::{CurveGroup, Group};
    use ark_ff::{UniformRand, Zero};
    use clap::Parser;
    use dynaverse_gpu::{ArkworksBackend, CpuBackend, HybridThresholds, IcicleBackend, MAX_MSM_SIZE};
    use dynaverse_protocol::OperatorBackend;
    use rand::{rngs::StdRng, SeedableRng};
    use std::fs::File;
    use std::io::{BufWriter, Write};
    use std::path::PathBuf;
    use std::time::Instant;

    #[derive(Parser)]
    struct Args {
        #[arg(long, value_delimiter = ',', default_value = "16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576")]
        lengths: Vec<usize>,
        #[arg(long, default_value_t = 2)] warmup: usize,
        #[arg(long, default_value_t = 5)] samples: usize,
        #[arg(long, default_value_t = 10)] large_bucket_factor: u32,
        #[arg(long)] output: PathBuf,
        #[arg(long)] profile_output: PathBuf,
    }

    fn quantile(values: &[f64], probability: f64) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        sorted[((sorted.len() as f64 * probability).ceil() as usize - 1).min(sorted.len() - 1)]
    }

    fn record(writer: &mut BufWriter<File>, gpu: &IcicleBackend, device: &str,
        operation: &str, length: usize, gpu_route: bool, phase: &str, sample: usize,
        order: usize, micros: f64, args: &Args) -> Result<()> {
        writeln!(writer, "{},{},{},{},{},{},{},{},{},{},{},{:.3},{},{},true",
            env!("DYNAVERSE_BUILD_COMMIT"), env!("DYNAVERSE_SOURCE_HASH"), env!("DYNAVERSE_SOURCE_DIRTY"),
            device, gpu.large_bucket_factor(), operation, length,
            if gpu_route { "icicle" } else { "cpu" }, phase, sample, order, micros, args.warmup, args.samples)?;
        Ok(())
    }

    fn paired(writer: &mut BufWriter<File>, gpu: &IcicleBackend, device: &str, operation: &'static str,
        length: usize, args: &Args, mut call: impl FnMut(bool) -> Result<f64>) -> Result<Summary> {
        for (order, route) in [false, true].into_iter().enumerate() {
            let micros = call(route)?;
            record(writer, gpu, device, operation, length, route, "first_call", 0, order, micros, args)?;
        }
        let mut cpu = Vec::new();
        let mut cuda = Vec::new();
        for trial in 0..args.warmup + args.samples {
            for order in 0..2 {
                let route = (trial + order) % 2 == 1;
                let micros = call(route)?;
                let phase = if trial < args.warmup { "warmup" } else { "sample" };
                let sample = if trial < args.warmup { trial } else { trial - args.warmup };
                record(writer, gpu, device, operation, length, route, phase, sample, order, micros, args)?;
                if trial >= args.warmup {
                    if route { cuda.push(micros); } else { cpu.push(micros); }
                }
            }
        }
        writer.flush()?;
        eprintln!("OPERATOR_EXACT_OK operation={operation} length={length} samples={}", args.samples);
        Ok(Summary { operation, length, cpu_median_us: quantile(&cpu, 0.5),
            gpu_median_us: quantile(&cuda, 0.5), cpu_p95_us: quantile(&cpu, 0.95), gpu_p95_us: quantile(&cuda, 0.95) })
    }

    pub fn run() -> Result<()> {
        let mut args = Args::parse();
        ensure!(!args.lengths.is_empty() && args.samples >= 5, "calibration needs sizes and >=5 samples");
        ensure!(args.large_bucket_factor <= 100, "invalid bucket factor");
        args.lengths.sort_unstable();
        args.lengths.dedup();
        let maximum = *args.lengths.last().unwrap();
        ensure!(args.lengths[0] > 0 && maximum <= MAX_MSM_SIZE, "operator sizes exceed 1..=2^20");
        ensure!(args.output != args.profile_output, "raw output and profile must differ");
        if let (Ok(a), Ok(b)) = (std::fs::metadata(&args.output), std::fs::metadata(&args.profile_output)) {
            use std::os::unix::fs::MetadataExt;
            ensure!(a.dev() != b.dev() || a.ino() != b.ino(), "raw/profile outputs alias");
        }
        let gpu = IcicleBackend::with_large_bucket_factor(args.large_bucket_factor)?;
        let device = gpu.device_identity()?;
        ensure!(!device.contains(',') && !device.contains('\n') && !device.contains('\r'),
            "device name is not supported in raw CSV");
        for path in [&args.output, &args.profile_output] {
            if let Some(parent) = path.parent() { if !parent.as_os_str().is_empty() { std::fs::create_dir_all(parent)?; } }
        }
        // Recheck after resolving missing parents, before either truncation.
        if let (Ok(a), Ok(b)) = (std::fs::metadata(&args.output), std::fs::metadata(&args.profile_output)) {
            use std::os::unix::fs::MetadataExt;
            ensure!(a.dev() != b.dev() || a.ino() != b.ino(), "raw/profile outputs alias");
        }
        let resolved = |path: &PathBuf| -> Result<PathBuf> {
            let parent = path.parent().filter(|parent| !parent.as_os_str().is_empty())
                .unwrap_or_else(|| std::path::Path::new("."));
            Ok(parent.canonicalize()?.join(path.file_name().ok_or_else(|| anyhow::anyhow!("missing output filename"))?))
        };
        ensure!(resolved(&args.output)? != resolved(&args.profile_output)?, "raw/profile outputs alias");
        // Construct distinct points cheaply by a group walk; batch normalize.
        // This data is outside timing, independent of protocol SRS and secrets.
        let generator = G1Projective::generator();
        let mut current = generator;
        let mut projective = Vec::with_capacity(maximum);
        for _ in 0..maximum { projective.push(current); current += generator; }
        let points: Vec<G1Affine> = G1Projective::normalize_batch(&projective);
        drop(projective);
        let mut writer = BufWriter::new(File::create(&args.output)?);
        if let (Ok(a), Ok(b)) = (std::fs::metadata(&args.output), std::fs::metadata(&args.profile_output)) {
            use std::os::unix::fs::MetadataExt;
            ensure!(a.dev() != b.dev() || a.ino() != b.ino(), "raw/profile outputs alias");
        }
        writeln!(writer, "commit,source_hash,source_dirty,device,large_bucket_factor,operation,length,backend,phase,sample,order,elapsed_us,warmup,samples,cpu_match")?;
        let mut rng = StdRng::seed_from_u64(2102026);
        let mut summaries = Vec::new();
        for &length in &args.lengths {
            for operation in ["msm_dense", SPARSE_MSM_VARIANTS[0], SPARSE_MSM_VARIANTS[1],
                SPARSE_MSM_VARIANTS[2], "msm_binary"] {
                let mut scalars = vec![Fr::zero(); length];
                for i in 0..length {
                    scalars[i] = match operation {
                        "msm_dense" => Fr::rand(&mut rng),
                        "msm_sparse_single" if i == 0 => Fr::rand(&mut rng),
                        "msm_sparse_64" if i % 64 == 0 => Fr::rand(&mut rng),
                        "msm_sparse_16" if i % 16 == 0 => Fr::rand(&mut rng),
                        "msm_binary" => Fr::from((i % 2) as u64),
                        _ => Fr::zero(),
                    };
                }
                let expected = ArkworksBackend.msm(&points[..length], &scalars)?;
                summaries.push(paired(&mut writer, &gpu, &device, operation, length, &args, |route| {
                    let started = Instant::now();
                    let actual = if route { gpu.msm(&points[..length], &scalars)? }
                        else { ArkworksBackend.msm(&points[..length], &scalars)? };
                    let micros = started.elapsed().as_secs_f64() * 1e6;
                    ensure!(actual == expected, "MSM differs from independent CPU result");
                    Ok(micros)
                })?);
            }
            if length.is_power_of_two() {
                let input: Vec<Fr> = (0..length).map(|_| Fr::rand(&mut rng)).collect();
                for inverse in [false, true] {
                    let mut expected = input.clone();
                    CpuBackend.ntt(&mut expected, inverse)?;
                    let operation = if inverse { "ntt_inverse" } else { "ntt_forward" };
                    summaries.push(paired(&mut writer, &gpu, &device, operation, length, &args, |route| {
                        let mut actual = input.clone(); // identical preparation outside timing
                        let started = Instant::now();
                        if route { gpu.ntt(&mut actual, inverse)?; } else { CpuBackend.ntt(&mut actual, inverse)?; }
                        let micros = started.elapsed().as_secs_f64() * 1e6;
                        ensure!(actual == expected, "NTT differs from independent CPU result");
                        Ok(micros)
                    })?);
                }
            }
        }
        let thresholds = HybridThresholds { dense_msm_min: sustained_minimum(&summaries, "msm_dense"),
            sparse_msm_min: sustained_minimum(&summaries, "msm_sparse"),
            forward_ntt_min: sustained_minimum(&summaries, "ntt_forward"),
            inverse_ntt_min: sustained_minimum(&summaries, "ntt_inverse"), max_measured_size: maximum };
        thresholds.validate()?;
        let mut rows = Vec::new();
        for row in &summaries {
            rows.push(serde_json::json!({"operation":row.operation,"length":row.length,
                "cpu_median_us":row.cpu_median_us,"gpu_median_us":row.gpu_median_us,
                "cpu_p95_us":row.cpu_p95_us,"gpu_p95_us":row.gpu_p95_us}));
        }
        let profile = serde_json::json!({"schema":1,"curve":"bn254","icicle":"2.8.0",
            "commit":env!("DYNAVERSE_BUILD_COMMIT"),"source_hash":env!("DYNAVERSE_SOURCE_HASH"),
            "source_dirty":env!("DYNAVERSE_SOURCE_DIRTY"),"device":device,"large_bucket_factor":args.large_bucket_factor,
            "samples":args.samples,"warmup":args.warmup,
            "selection":">=3 distinct sustained scales; every later variant median <=0.90 CPU and p95 <= CPU",
            "input_policy":{"seed":2102026,"sparse_variants":SPARSE_MSM_VARIANTS,
                "sparse_definition":"one nonzero, every 64th, every 16th; random full-width Fr",
                "binary_definition":"alternating 0/1; always routed to CPU"},
            "thresholds":{"dense_msm_min":thresholds.dense_msm_min,"sparse_msm_min":thresholds.sparse_msm_min,
                "forward_ntt_min":thresholds.forward_ntt_min,"inverse_ntt_min":thresholds.inverse_ntt_min,
                "max_measured_size":maximum},"summaries":rows});
        // Reject a dangling output symlink that resolved once raw was created.
        if let (Ok(a), Ok(b)) = (std::fs::metadata(&args.output), std::fs::metadata(&args.profile_output)) {
            use std::os::unix::fs::MetadataExt;
            ensure!(a.dev() != b.dev() || a.ino() != b.ino(), "raw/profile outputs alias");
        }
        std::fs::write(&args.profile_output, serde_json::to_string_pretty(&profile)? + "\n")?;
        eprintln!("CALIBRATION_COMPLETE thresholds={thresholds:?}");
        Ok(())
    }
}
#[cfg(feature = "icicle")]
fn main() -> Result<()> { measurement::run() }
