//! One immutable affected-root/anchor preparation, never value-dependent delta data.
use super::*;
use crate::domain::SubproductTree;
use std::sync::Arc;

pub(super) struct SparseGeometry {
    pub anchor_generation:u64,
    pub affected:Vec<usize>,
    pub points:Vec<Fr>,
    pub tree:Arc<SubproductTree>,
    pub inverses:Vec<Fr>,
    pub cauchy_weights:Vec<Fr>,
    pub affected_g1:G1Affine,
    pub affected_g2:G2Affine,
    pub unaffected_g1:G1Affine,
    pub unaffected_g2:G2Affine,
    pub anchor_local:[Vec<Fr>;3],
    pub anchor_local_commitments:MultiplicationWireCommitments,
    pub anchor_openings:MultiplicationWireCommitments,
}
