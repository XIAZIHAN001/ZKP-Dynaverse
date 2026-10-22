//! Dynaverse protocol reference implementation on BN254 (CPU).
//!
//! Module layout follows the three-week plan; stable interfaces are frozen
//! after day 3 and may only be extended, never renamed.

pub mod domain;
pub mod consistency;
pub mod dynamo;
pub mod dynaverse;
pub mod background;
pub mod incremental_rebuild;
pub mod kzg;
pub mod srs;
mod setup;
pub mod toy_circuit;
pub mod trace;
mod verification;
pub mod types;
pub mod workcount;

pub use types::*;
