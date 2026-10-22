//! GPU operator backends.
//!
//! `CpuBackend` is the simple correctness oracle used by tests.
//! `ArkworksBackend` adapts the original KZG CPU MSM library call. The optional
//! `icicle` feature enables synchronous ICICLE G1 MSM and explicit hybrid
//! selection and Fr NTT; GPU execution errors propagate.

pub mod codec;
mod arkworks;
pub use arkworks::ArkworksBackend;
mod fixed_base_cpu;
pub use fixed_base_cpu::FixedBaseCpuBackend;
pub mod g2_validation;
#[cfg(feature = "icicle")]
pub mod bn254_tower;
#[cfg(feature = "icicle")]
pub mod bn254_lines;
#[cfg(feature = "icicle")]
pub mod bn254_miller;
pub use g2_validation::{G2ValidationMode,G2ValidationStats,G2ValidationHostStats,G2ValidationTraffic};

#[cfg(feature = "icicle")]
mod icicle;
#[cfg(feature = "icicle")]
pub use icicle::{BatchStats, MemoryInfo, IcicleBackend, MsmProfile, MAX_MSM_SIZE, MAX_MSM_BATCH_SIZE, MAX_NTT_SIZE};
#[cfg(feature = "icicle")]
mod resident;
#[cfg(feature = "icicle")]
mod setup;
#[cfg(feature = "icicle")]
pub use setup::{SetupMode, SetupStats, SetupHostStats};
#[cfg(feature = "icicle")]
pub use resident::{ResidentDiagnostics, ResidentCallProfile, ResidentNativeProfile};
#[cfg(feature = "icicle")]
mod timeline;
#[cfg(feature = "icicle")]
pub use timeline::{MsmTimeline, TIMELINE_PHASES, TIMELINE_KERNELS};
#[cfg(feature = "icicle")]
mod hybrid;
#[cfg(feature = "icicle")]
pub use hybrid::{HybridBackend, HybridThresholds};

pub use codec::{decode_fr, encode_fr};

use anyhow::Result;
use ark_bn254::{Fr, G1Affine};
use dynaverse_protocol::{BatchMsmRequest, OperatorBackend};

/// Error type for codec/backend boundary. Display-only; carries a message.
/// anyhow::Error already wraps any std::error::Error via its blanket impl,
/// so no manual From impl is needed.
#[derive(Debug)]
pub struct CpuBackendError(&'static str);
impl std::fmt::Display for CpuBackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "gpu backend error: {}", self.0)
    }
}
impl std::error::Error for CpuBackendError {}

/// Shorthand constructor for codec errors.
pub(crate) fn codec_error(msg: &'static str) -> CpuBackendError {
    CpuBackendError(msg)
}

/// Plain-CPU implementation via arkworks projective arithmetic.
pub struct CpuBackend;

impl OperatorBackend for CpuBackend {
    fn msm(&self, points: &[G1Affine], scalars: &[Fr]) -> Result<G1Affine> {
        if points.len() != scalars.len() {
            anyhow::bail!("msm length mismatch: {} points vs {} scalars", points.len(), scalars.len());
        }
        Ok(straight_msm(points, scalars))
    }

    fn msm_batch(&self, request: &BatchMsmRequest<'_>) -> Result<Vec<G1Affine>> {
        let mut out = Vec::with_capacity(request.point_sets.len());
        for (set_i, points) in request.point_sets.iter().enumerate() {
            let off = *request
                .offsets
                .get(set_i)
                .ok_or_else(|| anyhow::anyhow!("missing offset for set {set_i}"))?;
            let len = *request
                .lengths
                .get(set_i)
                .ok_or_else(|| anyhow::anyhow!("missing length for set {set_i}"))?;
            if off + len > request.shared_scalars.len() {
                anyhow::bail!("scalar range [{off},{}) out of bounds {}", off + len, request.shared_scalars.len());
            }
            if len > points.len() {
                anyhow::bail!("set {set_i}: length {len} exceeds points {}", points.len());
            }
            out.push(straight_msm(points, &request.shared_scalars[off..off + len]));
        }
        Ok(out)
    }

    fn ntt(&self, values: &mut [Fr], inverse: bool) -> Result<()> {
        if !values.len().is_power_of_two() {
            anyhow::bail!("ntt size {} is not a power of two", values.len());
        }
        if values.is_empty() {
            return Ok(());
        }
        use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
        let domain = Radix2EvaluationDomain::<Fr>::new(values.len())
            .ok_or_else(|| anyhow::anyhow!("cannot build radix-2 domain of size {}", values.len()))?;
        let mut coeffs = values.to_vec();
        if inverse {
            domain.ifft_in_place(&mut coeffs);
        } else {
            domain.fft_in_place(&mut coeffs);
        }
        values.copy_from_slice(&coeffs);
        Ok(())
    }
}

/// Straightforward left-to-right MSM; correctness reference only (slow on
/// purpose - Pippenger-level CPU speed is not a project goal).
fn straight_msm(points: &[G1Affine], scalars: &[Fr]) -> G1Affine {
    use ark_ec::{CurveGroup, Group};
    use ark_ff::{BigInteger, PrimeField};
    use ark_ec::AffineRepr;
    let mut acc = <G1Affine as AffineRepr>::Group::default();
    for (p, s) in points.iter().zip(scalars.iter()) {
        // s * p via biginteger bit decomposition
        let bits = s.into_bigint().to_bits_le();
        let mut cur = p.into_group();
        for bit in bits {
            if bit {
                acc += cur;
            }
            cur = cur.double();
        }
    }
    acc.into_affine()
}
