//! CPU -> GPU -> CPU roundtrip test against the real ICICLE MSM (day 2 closure).
//!
//! The Rust side writes canonical-LE scalar/point vectors to a temp file,
//! invokes `roundtrip_msm` (built from icicle-smoke/roundtrip_msm.cu against
//! ICICLE v2.8.0 sm_120 libs), decodes the GPU MSM result, and compares it
//! with the arkworks CPU reference. Skips (not fails) when the harness
//! binary or GPU environment is unavailable, so `cargo test` stays usable
//! on machines without the GPU stack.

use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::Zero;
use dynaverse_gpu::codec::{decode_g1, encode_fr, encode_g1, G1_BYTES};

/// Locate the CUDA roundtrip harness binary.
///
/// `DYNAVERSE_ROUNDTRIP_BIN` wins when set. Otherwise we search the workspace
/// `target/` dir, which is where the nvcc build drops it. Note that `cargo
/// test` runs integration tests with the cwd set to the *package* directory
/// (`crates/gpu`), not the workspace root, so a bare relative
/// `target/roundtrip_msm` would silently miss and turn every test into a SKIP.
/// Resolve against `CARGO_MANIFEST_DIR` instead of relying on cwd.
fn harness() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("DYNAVERSE_ROUNDTRIP_BIN") {
        let p = std::path::PathBuf::from(p);
        return p.exists().then_some(p);
    }
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // CARGO_MANIFEST_DIR is crates/gpu; workspace root is two levels up.
    let workspace = manifest.parent().and_then(|p| p.parent())?;
    // Canonical paths from  / .
    // target/roundtrip_msm is the legacy path from earlier manual nvcc builds.
    [
        workspace.join("build").join("sm120").join("roundtrip_msm"),
        workspace.join("build").join("sm90").join("roundtrip_msm"),
        workspace.join("target").join("roundtrip_msm"),
    ]
    .into_iter()
    .find(|p| p.exists())
}

/// Deterministic non-trivial scalars: cheap LCG over Fr, starting from a seed.
fn lcg_scalars(n: usize, seed: u64) -> Vec<Fr> {
    let mut state = Fr::from(seed);
    let two = Fr::from(2u64);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        state = state * two + Fr::from((i as u64 % 97) + 1);
        out.push(state);
    }
    out
}

fn points_from_scalars(scalars: &[Fr]) -> Vec<G1Affine> {
    let g = G1Projective::from(G1Affine::generator());
    scalars
        .iter()
        .map(|s| (g * s).into_affine())
        .collect()
}

fn write_input(path: &std::path::Path, scalars: &[Fr], points: &[G1Affine]) -> std::io::Result<()> {
    use std::io::Write;
    assert_eq!(scalars.len(), points.len());
    let mut f = std::fs::File::create(path)?;
    f.write_all(&(scalars.len() as u32).to_le_bytes())?;
    for s in scalars {
        f.write_all(&encode_fr(s))?;
    }
    for p in points {
        f.write_all(&encode_g1(p))?;
    }
    Ok(())
}

fn run_roundtrip(
    name: &str,
    scalars: &[Fr],
    points: &[G1Affine],
    expected: G1Affine,
) {
    let Some(bin) = harness() else {
        eprintln!("SKIP {name}: roundtrip harness not found (set DYNAVERSE_ROUNDTRIP_BIN)");
        return;
    };
    let dir = std::env::temp_dir();
    let inp = dir.join(format!("dynaverse_rt_{name}_in.bin"));
    let outp = dir.join(format!("dynaverse_rt_{name}_out.bin"));
    write_input(&inp, scalars, points).expect("write input");
    let status = std::process::Command::new(&bin)
        .arg(&inp)
        .arg(&outp)
        .status()
        .expect("run roundtrip harness");
    let _ = std::fs::remove_file(&inp);
    assert!(status.success(), "{name}: harness failed with {status}");
    let bytes = std::fs::read(&outp).expect("read output");
    let _ = std::fs::remove_file(&outp);
    assert_eq!(bytes.len(), 4 + G1_BYTES, "{name}: bad output size");
    assert_eq!(&bytes[..4], &1u32.to_le_bytes(), "{name}: bad result count");
    let got = decode_g1(&bytes[4..]).expect("decode gpu result");
    assert_eq!(got, expected, "{name}: GPU MSM result != arkworks CPU reference");
}

fn cpu_msm(scalars: &[Fr], points: &[G1Affine]) -> G1Affine {
    let mut acc = G1Projective::zero();
    for (s, p) in scalars.iter().zip(points.iter()) {
        acc += G1Projective::from(*p) * s;
    }
    acc.into_affine()
}

#[test]
fn roundtrip_gpu_msm_matches_arkworks_reference() {
    let scalars = lcg_scalars(17, 0x2026_0923);
    let points = points_from_scalars(&scalars);
    run_roundtrip("mixed", &scalars, &points, cpu_msm(&scalars, &points));
}

#[test]
fn roundtrip_gpu_msm_large_vector() {
    let scalars = lcg_scalars(1024, 0xABCDEF01);
    let points = points_from_scalars(&scalars);
    run_roundtrip("large", &scalars, &points, cpu_msm(&scalars, &points));
}

#[test]
fn roundtrip_gpu_msm_zero_scalar_gives_identity_or_offset() {
    // one zero scalar among non-zero ones: MSM == MSM without that term
    let mut scalars = lcg_scalars(8, 7);
    scalars[3] = Fr::from(0u64);
    let points = points_from_scalars(&scalars);
    run_roundtrip("zeroscalar", &scalars, &points, cpu_msm(&scalars, &points));
}

#[test]
fn roundtrip_gpu_msm_all_zero_scalars_is_infinity() {
    let scalars = vec![Fr::from(0u64); 4];
    let points = points_from_scalars(&lcg_scalars(4, 11));
    run_roundtrip("allzero", &scalars, &points, G1Affine::identity());
}

#[test]
fn roundtrip_gpu_msm_infinity_point_input() {
    // a zero-affine (infinity) point among real ones must contribute nothing
    let mut scalars = lcg_scalars(5, 99);
    let mut points = points_from_scalars(&scalars);
    scalars[2] = Fr::from(12345u64);
    points[2] = G1Affine::identity(); // encoded as all-zero 64B
    let expected = {
        let mut acc = G1Projective::zero();
        for (s, p) in scalars.iter().zip(points.iter()) {
            // identity contributes nothing; skipping it keeps the oracle's
            // intent explicit (arkworks 0.4.2 spells this `is_zero`, via Zero)
            if !p.is_zero() {
                acc += G1Projective::from(*p) * s;
            }
        }
        acc.into_affine()
    };
    run_roundtrip("inftypoint", &scalars, &points, expected);
}
