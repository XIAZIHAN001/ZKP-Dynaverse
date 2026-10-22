//! Wire-format codec for the CPU<->GPU boundary (day 2, frozen contract).
//!
//! - Scalars: 32 bytes little-endian **canonical integer** (value < modulus,
//!   NOT Montgomery form). arkworks `into_bigint()` yields canonical limbs;
//!   ICICLE's `MSMConfig.are_scalars_montgomery_form = false` consumes this
//!   representation directly on the host.
//! - G1Affine: 64 bytes = x || y, both 32-byte LE canonical integers.
//!   Infinity = all-zero 64 bytes (matches arkworks' uncompressed infinity
//!   encoding and ICICLE's zero-affine convention when Montgomery flags are
//!   off).
//! - Non-canonical / off-curve inputs are rejected, never silently reduced.
//!
//! IMPORTANT: if a consumer needs Montgomery-form inputs (e.g. data already
//! resident on device in ICICLE's native form), the conversion must happen
//! via `bn254_scalar_convert_montgomery` / `bn254_affine_convert_montgomery`
//! with the corresponding MSMConfig flags set to true. The Rust-side frozen
//! contract stays canonical LE.

use crate::{CpuBackendError, codec_error};
use ark_bn254::{Fr, G1Affine};
use ark_ec::AffineRepr;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, Validate};

pub const FR_BYTES: usize = 32;
pub const G1_BYTES: usize = 64;

/// Encode a scalar into 32 LE bytes of its canonical integer form
/// (value < modulus). This is arkworks' standard serialization and is what
/// crosses the CPU<->GPU boundary; Montgomery conversion (if needed) is the
/// GPU side's explicit responsibility.
pub fn encode_fr(f: &Fr) -> [u8; FR_BYTES] {
    let mut out = [0u8; FR_BYTES];
    for (i, limb) in f.into_bigint().0.iter().enumerate() {
        out[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    out
}

/// Decode 32 LE canonical bytes into a scalar, rejecting values >= modulus
/// and wrong lengths. arkworks' validated deserializer performs exactly
/// this check, so delegate to it.
pub fn decode_fr(bytes: &[u8]) -> Result<Fr, CpuBackendError> {
    if bytes.len() != FR_BYTES {
        return Err(codec_error("scalar must be exactly 32 bytes"));
    }
    Fr::deserialize_with_mode(&mut &bytes[..], ark_serialize::Compress::No, Validate::Yes)
        .map_err(|_| codec_error("non-canonical or invalid scalar bytes"))
}

/// Encode a G1 point into 64 bytes (x || y canonical LE), or all-zeros for
/// infinity.
pub fn encode_g1(p: &G1Affine) -> [u8; G1_BYTES] {
    let mut out = [0u8; G1_BYTES];
    if let Some((x, y)) = p.xy() {
        let xb = x.into_bigint();
        let yb = y.into_bigint();
        for i in 0..4 {
            out[i * 8..(i + 1) * 8].copy_from_slice(&xb.0[i].to_le_bytes());
            out[32 + i * 8..32 + (i + 1) * 8].copy_from_slice(&yb.0[i].to_le_bytes());
        }
    }
    out
}

/// Decode 64 bytes into a G1Affine. All-zero is infinity; otherwise the
/// point must deserialize (on-curve, canonical field elements).
pub fn decode_g1(bytes: &[u8]) -> Result<G1Affine, CpuBackendError> {
    if bytes.len() != G1_BYTES {
        return Err(codec_error("G1 point must be exactly 64 bytes"));
    }
    if bytes.iter().all(|&b| b == 0) {
        return Ok(G1Affine::identity());
    }
    G1Affine::deserialize_with_mode(&mut &bytes[..], ark_serialize::Compress::No, Validate::Yes)
        .map_err(|_| codec_error("invalid or off-curve G1 bytes"))
}

// This shortcut is specific to BN254 G1. It must never be generalized to
// G2 or another curve by changing the point type without revisiting the proof.
#[cfg(any(feature = "icicle", test))]
const _: () = {
    use ark_ec::CurveConfig;
    assert!(ark_bn254::g1::Config::COFACTOR.len() == 1);
    assert!(ark_bn254::g1::Config::COFACTOR[0] == 1);
};

/// Internal readback alternative, preserving `decode_g1` as an independent
/// full-validation oracle. In pinned arkworks, Validate::No still performs
/// canonical field decoding and validates the serialization flags; the only
/// point check replaced here is on-curve + [r]P=0 with on-curve, since h=1.
/// Keep the legacy all-zero wire identity and flagged-infinity acceptance.
#[cfg(any(feature = "icicle", test))]
pub(crate) fn decode_g1_cofactor_one(bytes: &[u8]) -> Result<G1Affine, CpuBackendError> {
    if bytes.len() != G1_BYTES {
        return Err(codec_error("G1 point must be exactly 64 bytes"));
    }
    if bytes.iter().all(|&b| b == 0) {
        return Ok(G1Affine::identity());
    }
    let point = G1Affine::deserialize_with_mode(&mut &bytes[..], ark_serialize::Compress::No, Validate::No)
        .map_err(|_| codec_error("invalid or off-curve G1 bytes"))?;
    if !point.is_on_curve() {
        return Err(codec_error("invalid or off-curve G1 bytes"));
    }
    Ok(point)
}

#[cfg(test)]
#[path = "codec/g1_readback_tests.rs"]
mod g1_readback_tests;
