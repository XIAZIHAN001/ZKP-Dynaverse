//! Explicit local calibration policy; used by both calibration and sequence CLI.
use anyhow::{ensure, Result};
use dynaverse_gpu::{HybridThresholds, IcicleBackend};
use serde_json::Value;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Summary {
    pub operation: &'static str,
    pub length: usize,
    pub cpu_median_us: f64,
    pub gpu_median_us: f64,
    pub cpu_p95_us: f64,
    pub gpu_p95_us: f64,
}

pub const SPARSE_MSM_VARIANTS: [&str; 3] = ["msm_sparse_single", "msm_sparse_64", "msm_sparse_16"];

pub fn sustained_minimum(rows: &[Summary], operation: &str) -> Option<usize> {
    let operations: &[&str] = if operation == "msm_sparse" {
        &SPARSE_MSM_VARIANTS
    } else { std::slice::from_ref(&operation) };
    let mut lengths = Vec::new();
    for row in rows { if operations.contains(&row.operation) { lengths.push(row.length); } }
    lengths.sort_unstable();
    lengths.dedup();
    for start in 0..lengths.len() {
        if lengths.len() - start < 3 { break; }
        let mut wins = true;
        // A sparse class covers very different CPU costs. Every representative
        // must win at every later size; three variants at one size are one scale.
        for &length in &lengths[start..] {
            for &variant in operations {
                let mut found = false;
                for row in rows {
                    if row.length != length || row.operation != variant { continue; }
                    found = true;
                    if row.gpu_median_us > row.cpu_median_us * 0.90 || row.gpu_p95_us > row.cpu_p95_us {
                        wins = false;
                    }
                }
                if !found { wins = false; }
            }
        }
        if wins { return Some(lengths[start]); }
    }
    None
}

fn optional_minimum(value: &Value, key: &str) -> Result<Option<usize>> {
    let field = value.get(key).ok_or_else(|| anyhow::anyhow!("missing calibration field {key}"))?;
    if field.is_null() { return Ok(None); }
    let minimum = field.as_u64().ok_or_else(|| anyhow::anyhow!("invalid calibration field {key}"))?;
    Ok(Some(usize::try_from(minimum)?))
}

pub fn validate_profile(value: &Value, source: &str, device: &str, factor: u32) -> Result<HybridThresholds> {
    ensure!(value["schema"] == 1 && value["curve"] == "bn254" && value["icicle"] == "2.8.0",
        "unsupported calibration schema/curve/ICICLE");
    ensure!(value["source_hash"].as_str() == Some(source), "calibration source identity differs");
    ensure!(value["device"].as_str() == Some(device), "calibration hardware differs");
    ensure!(value["large_bucket_factor"].as_u64() == Some(factor as u64), "calibration bucket factor differs");
    ensure!(value["samples"].as_u64().unwrap_or(0) >= 5, "calibration requires at least five samples");
    let fields = &value["thresholds"];
    let maximum = fields["max_measured_size"].as_u64()
        .ok_or_else(|| anyhow::anyhow!("missing calibration maximum"))?;
    let thresholds = HybridThresholds {
        dense_msm_min: optional_minimum(fields, "dense_msm_min")?,
        sparse_msm_min: optional_minimum(fields, "sparse_msm_min")?,
        forward_ntt_min: optional_minimum(fields, "forward_ntt_min")?,
        inverse_ntt_min: optional_minimum(fields, "inverse_ntt_min")?,
        max_measured_size: usize::try_from(maximum)?,
    };
    thresholds.validate()?;
    Ok(thresholds)
}

pub fn load(path: &Path, gpu: &IcicleBackend, source: &str) -> Result<HybridThresholds> {
    ensure!(std::fs::metadata(path)?.len() <= 65536, "calibration profile is too large");
    let value: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    validate_profile(&value, source, &gpu.device_identity()?, gpu.large_bucket_factor())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sustained_policy_requires_three_wins_and_rejects_later_regression() {
        let make = |length, ratio| Summary { operation: "ntt_inverse", length,
            cpu_median_us: 100.0, gpu_median_us: ratio * 100.0,
            cpu_p95_us: 120.0, gpu_p95_us: ratio * 120.0 };
        let mut rows = vec![make(16, 1.2), make(32, 0.8), make(64, 0.7), make(128, 0.6)];
        assert_eq!(sustained_minimum(&rows, "ntt_inverse"), Some(32));
        assert_eq!(sustained_minimum(&rows, "msm_dense"), None);
        rows.push(make(256, 1.1));
        assert_eq!(sustained_minimum(&rows, "ntt_inverse"), None);
        rows[1].gpu_p95_us = 200.0;
        rows.pop();
        assert_eq!(sustained_minimum(&rows, "ntt_inverse"), None);
    }

    #[test]
    fn repeated_rows_at_one_length_do_not_count_as_three_scales() {
        let row = Summary { operation: "msm_dense", length: 64,
            cpu_median_us: 100.0, gpu_median_us: 80.0,
            cpu_p95_us: 120.0, gpu_p95_us: 90.0 };
        assert_eq!(sustained_minimum(&[row.clone(), row.clone(), row], "msm_dense"), None);
    }

    #[test]
    fn sparse_threshold_requires_every_variant_at_every_later_scale() {
        let mut rows = Vec::new();
        for length in [16, 32, 64, 128] {
            for operation in ["msm_sparse_single", "msm_sparse_64", "msm_sparse_16"] {
                rows.push(Summary { operation, length, cpu_median_us: 100.0,
                    gpu_median_us: 80.0, cpu_p95_us: 120.0, gpu_p95_us: 90.0 });
            }
        }
        assert_eq!(sustained_minimum(&rows, "msm_sparse"), Some(16));
        // A single cheap CPU variant defeats an otherwise winning class.
        rows[0].gpu_median_us = 110.0;
        assert_eq!(sustained_minimum(&rows, "msm_sparse"), Some(32));
        rows.last_mut().unwrap().gpu_p95_us = 130.0;
        assert_eq!(sustained_minimum(&rows, "msm_sparse"), None);
        rows.last_mut().unwrap().gpu_p95_us = 90.0;
        rows.remove(7); // missing 1/64 data at 64: no supported suffix of three scales
        assert_eq!(sustained_minimum(&rows, "msm_sparse"), None);
    }

    #[test]
    fn profile_requires_matching_identity_and_explicit_valid_thresholds() {
        let value = serde_json::json!({"schema":1,"curve":"bn254","icicle":"2.8.0",
            "source_hash":"source-a","device":"gpu-a/sm_120","large_bucket_factor":10,"samples":5,
            "thresholds":{"dense_msm_min":512,"sparse_msm_min":null,
                "forward_ntt_min":1024,"inverse_ntt_min":2048,"max_measured_size":4096}});
        let valid = validate_profile(&value, "source-a", "gpu-a/sm_120", 10).unwrap();
        assert_eq!(valid.sparse_msm_min, None);
        assert!(validate_profile(&value, "source-b", "gpu-a/sm_120", 10).is_err());
        assert!(validate_profile(&value, "source-a", "gpu-b/sm_120", 10).is_err());
        assert!(validate_profile(&value, "source-a", "gpu-a/sm_120", 0).is_err());
        for (key, bad) in [("dense_msm_min", Value::from(0)),
            ("sparse_msm_min", Value::from(8192)), ("forward_ntt_min", Value::from("invalid"))] {
            let mut invalid = value.clone();
            invalid["thresholds"][key] = bad;
            assert!(validate_profile(&invalid, "source-a", "gpu-a/sm_120", 10).is_err());
        }
    }
}
