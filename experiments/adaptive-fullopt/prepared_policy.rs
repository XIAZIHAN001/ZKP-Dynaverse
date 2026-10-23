//! One-variable experiment; the published R5 strategy remains unchanged.
use anyhow::{ensure, Result};
use dynaverse_protocol::{VerificationMode,
    incremental_rebuild::{IncrementalConfig, SparseDispatchPolicy}};
use crate::strategy::Strategy;

pub const CAPS: [usize; 4] = [64, 256, 1024, 4096];

pub fn config(n: usize, cap: usize) -> Result<IncrementalConfig> {
    let policy = Strategy::for_size(n).map_err(anyhow::Error::msg)?;
    ensure!(CAPS.contains(&cap), "prepared cap must be 64, 256, 1024 or 4096");
    Ok(IncrementalConfig {
        window: Some(policy.window), trace_updates: true,
        memory_limit_bytes: policy.limit_bytes, sparse: SparseDispatchPolicy::ForceGpu,
        linear_rebuild_reuse: true, unchanged_cache_reuse: true,
        foreground_linear_reuse: true, freeze_linear_capture_limit: Some(128),
        frozen_g2_reuse_limit: Some(128), support_geometry_reuse: true,
        transform_row_batch: 16, transform_packet_max_len: policy.packet_cap,
        rebuild_cache_mode: VerificationMode::Product,
        rebuild_prepared_cache_roots: Some(n.min(cap)),
        rebuild_cache_readback_roots: Some(policy.cache_roots),
        ..IncrementalConfig::default()
    })
}

pub fn scan_caps(n: usize) -> Result<Vec<usize>> {
    Strategy::for_size(n).map_err(anyhow::Error::msg)?;
    let mut caps = Vec::new();
    let mut previous = 0;
    for cap in CAPS {
        let actual = n.min(cap);
        if actual != previous { caps.push(cap); previous = actual; }
    }
    Ok(caps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_capacity_does_not_expand_readback_or_change_paid_work_policy() {
        for n in [256, 1024, 65536] {
            for cap in CAPS {
                let c = config(n, cap).unwrap();
                assert_eq!(c.rebuild_prepared_cache_roots, Some(n.min(cap)));
                assert_eq!(c.rebuild_cache_readback_roots, Some(64));
                assert_eq!(Strategy::for_size(n).unwrap().cache_roots, 64);
                assert_eq!(c.window, Some(if n == 256 {6} else if n == 1024 {8} else {16}));
                assert_eq!(c.transform_row_batch, 16);
                assert_eq!(c.transform_packet_max_len, if n == 65536 {None} else {Some(4096)});
                assert_eq!(c.memory_limit_bytes, 2 << 30);
                assert_eq!(c.rebuild_cache_mode, VerificationMode::Product);
                assert!(c.trace_updates && c.unchanged_cache_reuse && c.frozen_g2_reuse_limit == Some(128));
            }
        }
    }

    #[test]
    fn scan_deduplicates_actual_roots_and_rejects_unapproved_inputs() {
        assert_eq!(scan_caps(256).unwrap(), vec![64, 256]);
        assert_eq!(scan_caps(1024).unwrap(), vec![64, 256, 1024]);
        assert_eq!(scan_caps(65536).unwrap(), vec![64, 256, 1024, 4096]);
        for cap in [0, 32, 65, 4097, usize::MAX] { assert!(config(1024, cap).is_err()); }
        for n in [0, 255, 131072] { assert!(scan_caps(n).is_err()); }
    }
}
