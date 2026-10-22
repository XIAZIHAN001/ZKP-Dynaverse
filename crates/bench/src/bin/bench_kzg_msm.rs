//! KZG G1 MSM CPU/ICICLE comparison, including the complete Rust/native path.
use anyhow::Result;

#[cfg(not(feature = "icicle"))]
fn main() -> Result<()> { anyhow::bail!("build bench_kzg_msm with --features icicle") }

#[cfg(feature = "icicle")]
mod measurement {
    use anyhow::{ensure, Result};
    use ark_bn254::Fr;
    use ark_bn254::G1Affine;
    use ark_ff::UniformRand;
    use clap::Parser;
    use dynaverse_gpu::{IcicleBackend, MsmProfile};
    use dynaverse_protocol::{BatchMsmRequest, OperatorBackend};
    use dynaverse_protocol::kzg::{KzgCommitment, KzgParams};
    use rand::{rngs::StdRng, SeedableRng};
    use std::fs::File;
    use std::cell::Cell;
    use std::io::{BufWriter, Write};
    use std::path::{Path, PathBuf};
    use std::os::unix::fs::MetadataExt;
    use std::time::{Duration, Instant};

    #[derive(Parser)]
    struct Args {
        #[arg(long, value_delimiter = ',', default_value = "1,2,17,256,1024,8192")]
        lengths: Vec<usize>,
        #[arg(long, default_value_t = 10)]
        warmup: usize,
        #[arg(long, default_value_t = 30)]
        samples: usize,
        #[arg(long)]
        output: PathBuf,
        /// Separate diagnostic calls after the ordinary benchmark matrix.
        #[arg(long)]
        profile_output: Option<PathBuf>,
    }

    // This short-lived adapter borrows the GPU owner. Cell records the result
    // through OperatorBackend's shared receiver without changing the protocol
    // trait or exposing private SRS powers. It never leaves this caller thread.
    struct ProfileBackend<'a> { gpu: &'a IcicleBackend, last: Cell<Option<MsmProfile>> }
    impl OperatorBackend for ProfileBackend<'_> {
        fn msm(&self, points: &[G1Affine], scalars: &[Fr]) -> Result<G1Affine> {
            let (point, profile) = self.gpu.msm_profiled(points, scalars)?;
            self.last.set(Some(profile));
            Ok(point)
        }
        fn msm_batch(&self, request: &BatchMsmRequest<'_>) -> Result<Vec<G1Affine>> {
            self.gpu.msm_batch(request)
        }
        fn ntt(&self, values: &mut [Fr], inverse: bool) -> Result<()> { self.gpu.ntt(values, inverse) }
    }

    fn profile_call(params: &KzgParams, coefficients: &[Fr], gpu: &IcicleBackend,
        expected: &KzgCommitment) -> Result<MsmProfile> {
        let backend = ProfileBackend { gpu, last: Cell::new(None) };
        let actual = params.commit_with_backend(coefficients, &backend)?;
        ensure!(&actual == expected, "profiled commitment differs from CPU");
        if coefficients.is_empty() { return Ok(MsmProfile::default()); }
        backend.last.take().ok_or_else(|| anyhow::anyhow!("missing MSM diagnostics"))
    }

    fn measure(params: &KzgParams, coefficients: &[Fr], gpu: &IcicleBackend,
        use_gpu: bool, expected: &KzgCommitment) -> Result<Duration> {
        let started = Instant::now();
        let commitment = if use_gpu { params.commit_with_backend(coefficients, gpu)? }
            else { params.commit(coefficients)? };
        let elapsed = started.elapsed();
        ensure!(&commitment == expected, "CPU/GPU commitment differs");
        Ok(elapsed)
    }

    fn row(writer: &mut BufWriter<File>, length: usize, backend: &str, phase: &str,
        sample: usize, elapsed: Duration, args: &Args) -> Result<()> {
        writeln!(writer, "{},{},{},{},{},{},{},{:.3},{},{}",
            env!("DYNAVERSE_BUILD_COMMIT"), env!("DYNAVERSE_SOURCE_HASH"),
            env!("DYNAVERSE_SOURCE_DIRTY"), length, backend, phase, sample,
            elapsed.as_secs_f64() * 1e6, args.warmup, args.samples)?;
        Ok(())
    }

    fn reject_output_alias(output: &Path, profile: &Path) -> Result<()> {
        // Existing files must be checked before either is truncated. Check
        // again after creating the ordinary file to catch dangling symlinks.
        if let Ok(original) = std::fs::metadata(output) {
            if let Ok(other) = std::fs::metadata(profile) {
                ensure!(other.dev() != original.dev() || other.ino() != original.ino(),
                    "profile output must differ from ordinary output");
            }
        }
        Ok(())
    }

    pub fn run() -> Result<()> {
        let args = Args::parse();
        ensure!(!args.lengths.is_empty() && args.samples > 0, "empty benchmark matrix");
        let maximum = *args.lengths.iter().max().unwrap();
        ensure!(maximum > 0 && maximum <= dynaverse_gpu::MAX_MSM_SIZE, "length must be within 1..=2^20");
        if let Some(path) = &args.profile_output {
            ensure!(path != &args.output, "profile output must differ from ordinary output");
            reject_output_alias(&args.output, path)?;
        }
        if let Some(parent) = args.output.parent() {
            if !parent.as_os_str().is_empty() { std::fs::create_dir_all(parent)?; }
        }
        let output_file = File::create(&args.output)?;
        let mut profile_writer = None;
        if let Some(path) = &args.profile_output {
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() { std::fs::create_dir_all(parent)?; }
            }
            reject_output_alias(&args.output, path)?;
            let mut diagnostic = BufWriter::new(File::create(path)?);
            writeln!(diagnostic, "commit,source_hash,source_dirty,length,backend,phase,sample,elapsed_us,warmup,samples")?;
            profile_writer = Some(diagnostic);
        }
        let mut writer = BufWriter::new(output_file);
        writeln!(writer, "commit,source_hash,source_dirty,length,backend,phase,sample,elapsed_us,warmup,samples")?;
        let started = Instant::now();
        let gpu = IcicleBackend::new()?;
        row(&mut writer, 0, "icicle", "backend_init", 0, started.elapsed(), &args)?;
        let mut rng = StdRng::seed_from_u64(2907);
        let started = Instant::now();
        let params = KzgParams::setup(maximum - 1, &mut rng)?;
        row(&mut writer, maximum, "shared", "srs_setup", 0, started.elapsed(), &args)?;
        let mut profiled_cases = Vec::new();
        for &length in &args.lengths {
            let mut coefficients = Vec::with_capacity(length);
            for _ in 0..length { coefficients.push(Fr::rand(&mut rng)); }
            let expected = params.commit(&coefficients)?;
            for use_gpu in [false, true] {
                let elapsed = measure(&params, &coefficients, &gpu, use_gpu, &expected)?;
                row(&mut writer, length, if use_gpu { "icicle" } else { "cpu" },
                    "first_call", 0, elapsed, &args)?;
            }
            for trial in 0..args.warmup + args.samples {
                for offset in 0..2 {
                    let use_gpu = (trial + offset) % 2 == 1;
                    let elapsed = measure(&params, &coefficients, &gpu, use_gpu, &expected)?;
                    if trial >= args.warmup {
                        row(&mut writer, length, if use_gpu { "icicle" } else { "cpu" },
                            "sample", trial - args.warmup, elapsed, &args)?;
                    }
                }
            }
            writer.flush()?;
            eprintln!("KZG_MSM_OK length={length} samples={}", args.samples);
            if profile_writer.is_some() { profiled_cases.push((coefficients, expected)); }
        }
        if let Some(mut diagnostic) = profile_writer.take() {
            for (coefficients, expected) in profiled_cases {
                for trial in 0..args.warmup + args.samples {
                    let profile = profile_call(&params, &coefficients, &gpu, &expected)?;
                    if trial < args.warmup { continue; }
                    let sample = trial - args.warmup;
                    let spans = [("total", profile.total), ("validation", profile.validation),
                        ("encoding", profile.encoding), ("native", profile.native), ("decode", profile.decode),
                        ("native_conversion", profile.native_conversion), ("icicle_inclusive", profile.icicle_inclusive),
                        ("native_output", profile.native_output)];
                    for (phase, elapsed) in spans {
                        row(&mut diagnostic, coefficients.len(), "icicle", phase, sample, elapsed, &args)?;
                    }
                }
                diagnostic.flush()?;
                eprintln!("KZG_PROFILE_OK length={} samples={}", coefficients.len(), args.samples);
            }
        }
        Ok(())
    }
}

#[cfg(feature = "icicle")]
fn main() -> Result<()> { measurement::run() }
