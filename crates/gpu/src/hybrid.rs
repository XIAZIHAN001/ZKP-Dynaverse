//! Explicit MSM selection based on local measurements, without GPU-error fallback.
use crate::{ArkworksBackend, CpuBackend, IcicleBackend, MsmProfile, MsmTimeline, MAX_MSM_SIZE};
use anyhow::{ensure, Result};
use ark_bn254::{Fr, G1Affine};
use ark_ff::{One, Zero};
use dynaverse_protocol::{BatchMsmRequest, OperatorBackend, SparseMsmRequest};
use std::ffi::CStr;

/// Explicit measured policy. None disables that GPU route; each minimum is
/// bounded by the largest size covered by this hardware's calibration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HybridThresholds {
    pub dense_msm_min: Option<usize>,
    pub sparse_msm_min: Option<usize>,
    pub forward_ntt_min: Option<usize>,
    pub inverse_ntt_min: Option<usize>,
    pub max_measured_size: usize,
}

impl HybridThresholds {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.max_measured_size > 0 && self.max_measured_size <= MAX_MSM_SIZE,
            "calibration maximum must be within 1..=2^20");
        for minimum in [self.dense_msm_min, self.sparse_msm_min,
            self.forward_ntt_min, self.inverse_ntt_min].into_iter().flatten() {
            ensure!(minimum > 0 && minimum <= self.max_measured_size,
                "calibration minimum exceeds measured bounds");
        }
        Ok(())
    }
}

// The lifetime ties the borrowed GPU owner to this adapter. The adapter does
// not acquire, clone or store any native pointer; its calls are synchronous.
pub struct HybridBackend<'a> {
    gpu: &'a IcicleBackend,
    gpu_min_points: usize,
    thresholds: Option<HybridThresholds>,
}

impl<'a> HybridBackend<'a> {
    pub fn new(gpu: &'a IcicleBackend, gpu_min_points: usize) -> Result<Self> {
        ensure!(gpu_min_points > 0 && gpu_min_points <= MAX_MSM_SIZE,
            "hybrid GPU minimum must be within 1..=2^20");
        Ok(Self { gpu, gpu_min_points, thresholds: None })
    }

    pub fn with_thresholds(gpu: &'a IcicleBackend, thresholds: HybridThresholds) -> Result<Self> {
        thresholds.validate()?;
        Ok(Self { gpu, gpu_min_points: thresholds.dense_msm_min.unwrap_or(MAX_MSM_SIZE),
            thresholds: Some(thresholds) })
    }

    /// Diagnostics use the same selection predicate. None means the CPU
    /// adapter ran; GPU timings retain IcicleBackend's existing boundaries.
    /// The ordinary trait entry remains free of diagnostic clocks.
    pub fn msm_profiled(&self, points: &[G1Affine], scalars: &[Fr]) -> Result<(G1Affine, Option<MsmProfile>)> {
        ensure!(points.len() == scalars.len(), "MSM length mismatch");
        ensure!(points.len() <= MAX_MSM_SIZE, "MSM exceeds baseline size bound (2^20)");
        if self.use_gpu(scalars) {
            let (point, profile) = self.gpu.msm_profiled(points, scalars)?;
            Ok((point, Some(profile)))
        } else {
            Ok((ArkworksBackend.msm(points, scalars)?, None))
        }
    }

    /// The same predicate selects the dedicated GPU diagnostic entry. CPU
    /// calls acquire no native events and return no invented GPU timings.
    pub fn msm_timeline(&self, points: &[G1Affine], scalars: &[Fr], label: &CStr)
        -> Result<(G1Affine, Option<(MsmProfile, MsmTimeline)>)> {
        ensure!(points.len() == scalars.len(), "MSM length mismatch");
        ensure!(points.len() <= MAX_MSM_SIZE, "MSM exceeds baseline size bound (2^20)");
        if self.use_gpu(scalars) {
            let (point, profile, timeline) = self.gpu.msm_timeline(points, scalars, label)?;
            Ok((point, Some((profile, timeline))))
        } else {
            Ok((ArkworksBackend.msm(points, scalars)?, None))
        }
    }

    fn use_gpu(&self, scalars: &[Fr]) -> bool {
        self.use_gpu_values(scalars.len(), scalars.iter())
    }

    // Read scalar values from either layout without making a temporary Fr
    // vector. The same measured/binary selection applies to both interfaces.
    fn use_gpu_values<'s>(&self, length: usize, scalars: impl Iterator<Item = &'s Fr>) -> bool {
        if length < self.gpu_min_points && self.thresholds.is_none() { return false; }
        // Real protocol calls include long all-zero/all-one vectors. Route
        // these through the same CPU library; do not skip point validation
        // or implement their group arithmetic in this selection layer.
        let mut nonzero = 0;
        let mut binary = true;
        for scalar in scalars {
            if !scalar.is_zero() { nonzero += 1; }
            if !scalar.is_zero() && !scalar.is_one() { binary = false; }
        }
        if binary { return false; }
        if let Some(thresholds) = self.thresholds {
            if length > thresholds.max_measured_size { return false; }
            let minimum = if nonzero <= length / 16 {
                thresholds.sparse_msm_min
            } else { thresholds.dense_msm_min };
            return minimum.map(|minimum| length >= minimum).unwrap_or(false);
        }
        true
    }
}

impl OperatorBackend for HybridBackend<'_> {
    fn msm(&self, points: &[G1Affine], scalars: &[Fr]) -> Result<G1Affine> {
        ensure!(points.len() == scalars.len(), "MSM length mismatch");
        ensure!(points.len() <= MAX_MSM_SIZE, "MSM exceeds baseline size bound (2^20)");
        if self.use_gpu(scalars) {
            self.gpu.msm(points, scalars)
        } else {
            ArkworksBackend.msm(points, scalars)
        }
    }

    fn msm_batch(&self, request: &BatchMsmRequest<'_>) -> Result<Vec<G1Affine>> {
        let count = request.point_sets.len();
        ensure!(request.offsets.len() == count && request.lengths.len() == count, "batch shape mismatch");
        ensure!(count <= MAX_MSM_SIZE, "too many batch sets (maximum 2^20)");
        for i in 0..count {
            let end = request.offsets[i].checked_add(request.lengths[i])
                .ok_or_else(|| anyhow::anyhow!("scalar range overflow"))?;
            ensure!(end <= request.shared_scalars.len(), "scalar range out of bounds");
            ensure!(request.lengths[i] <= request.point_sets[i].len(), "point range out of bounds");
            ensure!(request.lengths[i] <= MAX_MSM_SIZE, "MSM exceeds baseline size bound (2^20)");
        }
        let mut results = vec![G1Affine::identity(); count];
        let mut gpu_indices = Vec::new();
        let mut gpu_sets = Vec::new();
        let mut gpu_offsets = Vec::new();
        let mut gpu_lengths = Vec::new();
        for i in 0..count {
            let offset = request.offsets[i];
            let length = request.lengths[i];
            let scalars = &request.shared_scalars[offset..offset + length];
            if self.use_gpu(scalars) {
                gpu_indices.push(i);
                gpu_sets.push(request.point_sets[i]);
                gpu_offsets.push(offset);
                gpu_lengths.push(length);
            } else {
                results[i] = ArkworksBackend.msm(&request.point_sets[i][..length], scalars)?;
            }
        }
        // CPU sets are checked/computed first; the GPU wrapper validates all
        // remaining sets before dispatch and groups them into real batches.
        if !gpu_indices.is_empty() {
            let gpu_results = self.gpu.msm_batch(&BatchMsmRequest { point_sets: &gpu_sets,
                shared_scalars: request.shared_scalars, offsets: &gpu_offsets, lengths: &gpu_lengths })?;
            for (slot, &index) in gpu_indices.iter().enumerate() { results[index] = gpu_results[slot]; }
        }
        Ok(results)
    }

    fn msm_sparse_batch(&self, requests: &[SparseMsmRequest<'_>]) -> Result<Vec<G1Affine>> {
        ensure!(requests.len() <= MAX_MSM_SIZE, "too many batch sets (maximum 2^20)");
        for request in requests {
            ensure!(request.entries.len() <= MAX_MSM_SIZE, "MSM exceeds baseline size bound (2^20)");
            request.validate()?;
        }
        let mut results = vec![G1Affine::identity(); requests.len()];
        let mut gpu_requests = Vec::new();
        let mut gpu_indices = Vec::new();
        gpu_requests.try_reserve_exact(requests.len())?;
        gpu_indices.try_reserve_exact(requests.len())?;
        for (index, request) in requests.iter().enumerate() {
            if self.use_gpu_values(request.entries.len(), request.entries.iter().map(|entry| &entry.1)) {
                gpu_requests.push(*request);
                gpu_indices.push(index);
            } else {
                results[index] = ArkworksBackend.msm_sparse_batch(std::slice::from_ref(request))?[0];
            }
        }
        if !gpu_requests.is_empty() {
            let points = self.gpu.msm_sparse_batch(&gpu_requests)?;
            ensure!(points.len() == gpu_requests.len(), "backend returned wrong sparse commitment count");
            for (slot, &index) in gpu_indices.iter().enumerate() { results[index] = points[slot]; }
        }
        Ok(results)
    }

    fn ntt(&self, values: &mut [Fr], inverse: bool) -> Result<()> {
        if let Some(thresholds) = self.thresholds {
            let minimum = if inverse { thresholds.inverse_ntt_min } else { thresholds.forward_ntt_min };
            if values.len() <= thresholds.max_measured_size
                && minimum.map(|minimum| values.len() >= minimum).unwrap_or(false) {
                return self.gpu.ntt(values, inverse);
            }
        }
        CpuBackend.ntt(values, inverse)
    }
}
