//! Bench-only profiling of real protocol MSMs, with untimed CPU checks.
use anyhow::{ensure, Result};
use ark_bn254::{Fr, G1Affine};
use dynaverse_gpu::{ArkworksBackend, HybridBackend, IcicleBackend, MsmProfile, MsmTimeline};
use dynaverse_protocol::{BatchMsmRequest, OperatorBackend};
use std::cell::{Cell, RefCell};
use std::ffi::CString;
use std::time::Instant;
use crate::msm_inventory::MsmContext;

pub struct CallProfile {
    pub dispatch: &'static str,
    pub spans: MsmProfile,
    pub timeline: Option<MsmTimeline>,
    pub label: String,
}

// Both adapters are borrowed from main. The hybrid's own borrow keeps the
// GPU owner alive as well. RefCell records through the shared trait receiver
// on this caller thread; no input points/scalars are retained.
pub struct ProfileBackend<'a> {
    gpu: &'a IcicleBackend,
    hybrid: Option<&'a HybridBackend<'a>>,
    profiles: RefCell<Vec<CallProfile>>,
    run_label: Option<String>,
    operation: Cell<&'static str>,
    step: Cell<usize>,
    next_call: Cell<usize>,
}

impl<'a> ProfileBackend<'a> {
    pub fn new(gpu: &'a IcicleBackend, hybrid: Option<&'a HybridBackend<'a>>) -> Self {
        Self { gpu, hybrid, profiles: RefCell::new(Vec::new()), run_label: None,
            operation: Cell::new("prove"), step: Cell::new(0), next_call: Cell::new(0) }
    }

    pub fn new_timeline(gpu: &'a IcicleBackend, hybrid: Option<&'a HybridBackend<'a>>, run_label: String) -> Self {
        let mut backend = Self::new(gpu, hybrid);
        backend.run_label = Some(run_label);
        backend
    }

    pub fn take_profiles(&self) -> Vec<CallProfile> { self.profiles.take() }
}

impl MsmContext for ProfileBackend<'_> {
    fn begin(&self, operation: &'static str, step: usize) {
        self.operation.set(operation);
        self.step.set(step);
        self.next_call.set(0);
    }
}

impl OperatorBackend for ProfileBackend<'_> {
    fn msm(&self, points: &[G1Affine], scalars: &[Fr]) -> Result<G1Affine> {
        let mut label = String::new();
        if let Some(run_label) = &self.run_label {
            label = format!("{run_label}/operation={}/step={}/call={}/size={}",
                self.operation.get(), self.step.get(), self.next_call.get(), points.len());
        }
        let native_label = CString::new(label.as_str())?;
        let started = Instant::now();
        let mut timeline = None;
        let (point, gpu_profile) = if self.run_label.is_some() {
            let (point, diagnostics) = match self.hybrid {
                Some(hybrid) => hybrid.msm_timeline(points, scalars, &native_label)?,
                None => {
                    let (point, profile, timeline) = self.gpu.msm_timeline(points, scalars, &native_label)?;
                    (point, Some((profile, timeline)))
                }
            };
            match diagnostics {
                Some((profile, gpu_timeline)) => { timeline = Some(gpu_timeline); (point, Some(profile)) }
                None => (point, None),
            }
        } else { match self.hybrid {
            Some(hybrid) => hybrid.msm_profiled(points, scalars)?,
            None => {
                let (point, profile) = self.gpu.msm_profiled(points, scalars)?;
                (point, Some(profile))
            }
        }
        };
        let elapsed = started.elapsed();
        let profile = match gpu_profile {
            Some(spans) => CallProfile { dispatch: "icicle", spans, timeline, label },
            None => CallProfile { dispatch: "cpu", spans: MsmProfile { total: elapsed, ..MsmProfile::default() }, timeline: None, label },
        };
        // The original CPU library checks every diagnostic MSM, after the
        // recorded span has ended. This changes diagnostic call cadence; it
        // is deliberately excluded from ordinary sequence measurements.
        let expected = ArkworksBackend.msm(points, scalars)?;
        ensure!(point == expected, "diagnostic MSM differs from CPU reference");
        self.profiles.borrow_mut().push(profile);
        self.next_call.set(self.next_call.get() + 1);
        Ok(point)
    }

    fn msm_batch(&self, request: &BatchMsmRequest<'_>) -> Result<Vec<G1Affine>> {
        let count = request.point_sets.len();
        ensure!(request.offsets.len() == count && request.lengths.len() == count, "batch shape mismatch");
        // Validate the whole shape before the first GPU call. Each checked
        // slice is then dispatched through msm so it has its own record.
        for i in 0..count {
            let end = request.offsets[i].checked_add(request.lengths[i])
                .ok_or_else(|| anyhow::anyhow!("scalar range overflow"))?;
            ensure!(end <= request.shared_scalars.len(), "scalar range out of bounds");
            ensure!(request.lengths[i] <= request.point_sets[i].len(), "point range out of bounds");
        }
        let mut results = Vec::with_capacity(count);
        for i in 0..count {
            let offset = request.offsets[i];
            let length = request.lengths[i];
            results.push(self.msm(&request.point_sets[i][..length], &request.shared_scalars[offset..offset + length])?);
        }
        Ok(results)
    }

    fn ntt(&self, values: &mut [Fr], inverse: bool) -> Result<()> {
        match self.hybrid {
            Some(hybrid) => hybrid.ntt(values, inverse),
            None => self.gpu.ntt(values, inverse),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::G1Projective;
    use ark_ec::{CurveGroup, Group};

    #[test]
    fn profile_batch_consumes_checked_slices_and_records_actual_routes() {
        let gpu = IcicleBackend::new().unwrap();
        let hybrid = HybridBackend::new(&gpu, 2).unwrap();
        let backend = ProfileBackend::new(&gpu, Some(&hybrid));
        let generator = G1Projective::generator();
        let points = [generator.into_affine(); 3];
        let sets: [&[G1Affine]; 2] = [&points, &points];
        let scalars = [Fr::from(9u64), Fr::from(2u64), Fr::from(3u64)];
        assert!(backend.msm_batch(&BatchMsmRequest { point_sets: &sets, shared_scalars: &scalars,
            offsets: &[1, usize::MAX], lengths: &[2, 1] }).is_err());
        assert!(backend.take_profiles().is_empty());
        assert_eq!(backend.msm_batch(&BatchMsmRequest { point_sets: &sets, shared_scalars: &scalars,
            offsets: &[1, 0], lengths: &[2, 1] }).unwrap(),
            vec![(generator * Fr::from(5u64)).into_affine(), (generator * Fr::from(9u64)).into_affine()]);
        let profiles = backend.take_profiles();
        assert_eq!(profiles.len(), 2);
        assert_eq!(profiles[0].dispatch, "icicle");
        assert_eq!(profiles[1].dispatch, "cpu");
        assert!(profiles[0].spans.icicle_inclusive.as_nanos() > 0);
        assert_eq!(profiles[1].spans.icicle_inclusive.as_nanos(), 0);
        assert!(backend.take_profiles().is_empty());
    }
}
