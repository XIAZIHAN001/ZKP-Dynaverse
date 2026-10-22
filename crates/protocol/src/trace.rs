//! Optional, non-overlapping CPU phase timings for the reference prover.
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
pub enum CpuPhase {
    Validation,
    StateCheck,
    Interpolation,
    KzgCommit,
    DynamoMsm,
    Consistency,
    BatchOpen,
    AnchorOpenings,
    PairingVerify,
    CachePrecompute,
    AnchorRebuild,
    PublicDivisionPrecompute,
    SetupTrapdoors, SetupDynamo, SetupKzg, SetupConsistency, SetupSparse, SetupPublic,
}

impl CpuPhase {
    pub const ALL: [Self; 18] = [
        Self::Validation, Self::StateCheck, Self::Interpolation, Self::KzgCommit,
        Self::DynamoMsm, Self::Consistency, Self::BatchOpen, Self::AnchorOpenings,
        Self::PairingVerify, Self::CachePrecompute, Self::AnchorRebuild, Self::PublicDivisionPrecompute,
        Self::SetupTrapdoors, Self::SetupDynamo, Self::SetupKzg,
        Self::SetupConsistency, Self::SetupSparse, Self::SetupPublic,
    ];

    pub fn name(self) -> &'static str {
        // These names are string literals, so they live for the entire program.
        match self {
            Self::Validation => "validation",
            Self::StateCheck => "state_check",
            Self::Interpolation => "interpolation",
            Self::KzgCommit => "kzg_commit",
            Self::DynamoMsm => "dynamo_msm",
            Self::Consistency => "consistency",
            Self::BatchOpen => "batch_open",
            Self::AnchorOpenings => "anchor_openings",
            Self::PairingVerify => "pairing_verify",
            Self::CachePrecompute => "cache_precompute",
            Self::AnchorRebuild => "anchor_rebuild",
            Self::PublicDivisionPrecompute => "public_division_precompute",
            Self::SetupTrapdoors => "setup_trapdoors",
            Self::SetupDynamo => "setup_dynamo",
            Self::SetupKzg => "setup_kzg",
            Self::SetupConsistency => "setup_consistency",
            Self::SetupSparse => "setup_sparse",
            Self::SetupPublic => "setup_public",
        }
    }
}

#[derive(Clone, Debug)]
pub struct CpuTrace {
    enabled: bool,
    durations: [Duration; 18],
    verification: VerificationTrace,
    batch_open: BatchOpenTrace,
}

impl CpuTrace {
    pub fn new() -> Self {
        Self { enabled: true, durations: [Duration::ZERO; 18], verification: VerificationTrace::new(), batch_open: BatchOpenTrace::default() }
    }

    pub(crate) fn disabled() -> Self {
        Self { enabled: false, durations: [Duration::ZERO; 18], verification: VerificationTrace::disabled(), batch_open: BatchOpenTrace::default() }
    }

    pub fn elapsed(&self, phase: CpuPhase) -> Duration {
        // Fieldless enum variants have consecutive indices, starting at zero.
        self.durations[phase as usize]
    }

    pub fn recorded_total(&self) -> Duration {
        let mut total = Duration::ZERO;
        for duration in &self.durations {
            total += *duration;
        }
        total
    }

    /// Nested verification timings are a breakdown of PairingVerify and are
    /// deliberately excluded from recorded_total to avoid counting them twice.
    pub fn verification(&self) -> &VerificationTrace { &self.verification }

    /// Exclusive details inside BatchOpen, excluded from recorded_total.
    pub fn batch_open(&self) -> &BatchOpenTrace { &self.batch_open }

    pub(crate) fn finish_batch_open(&mut self, phase: BatchOpenPhase, started: Option<Instant>) {
        if let Some(started) = started {
            self.batch_open.durations[phase as usize] += started.elapsed();
        }
    }

    pub(crate) fn verification_mut(&mut self) -> &mut VerificationTrace { &mut self.verification }

    pub(crate) fn start(&self) -> Option<Instant> {
        if self.enabled { Some(Instant::now()) } else { None }
    }

    pub(crate) fn finish(&mut self, phase: CpuPhase, started: Option<Instant>) {
        if let Some(started) = started {
            self.durations[phase as usize] += started.elapsed();
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchOpenPhase {
    PublicVanishing,
    PublicQuotient,
    PublicCommit,
    Factors,
    DivisorCommit,
    LocalInterpolation,
    LocalArithmetic,
    LocalCommit,
    OpeningQuotients,
    OpeningCommit,
}

impl BatchOpenPhase {
    pub const ALL: [Self; 10] = [Self::PublicVanishing, Self::PublicQuotient,
        Self::PublicCommit, Self::Factors, Self::DivisorCommit,
        Self::LocalInterpolation, Self::LocalArithmetic, Self::LocalCommit,
        Self::OpeningQuotients, Self::OpeningCommit];

    pub fn name(self) -> &'static str {
        // Fixed labels borrow string literals with program-wide lifetime.
        match self {
            Self::PublicVanishing => "batch_open_public_vanishing",
            Self::PublicQuotient => "batch_open_public_quotient",
            Self::PublicCommit => "batch_open_public_commit",
            Self::Factors => "batch_open_factors",
            Self::DivisorCommit => "batch_open_divisor_commit",
            Self::LocalInterpolation => "batch_open_local_interpolation",
            Self::LocalArithmetic => "batch_open_local_arithmetic",
            Self::LocalCommit => "batch_open_local_commit",
            Self::OpeningQuotients => "batch_open_opening_quotients",
            Self::OpeningCommit => "batch_open_opening_commit",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct BatchOpenTrace {
    durations: [Duration; 10],
}

impl BatchOpenTrace {
    pub fn elapsed(&self, phase: BatchOpenPhase) -> Duration {
        // Fieldless enum variants index the fixed duration array.
        self.durations[phase as usize]
    }

    pub fn recorded_total(&self) -> Duration {
        let mut total = Duration::ZERO;
        for duration in &self.durations { total += *duration; }
        total
    }
}

#[derive(Clone, Copy, Debug)]
pub enum VerificationPhase {
    Structure,
    AnchorDynamo,
    AnchorConsistency,
    AnchorRelations,
    DeltaDynamo,
    DeltaConsistency,
    DeltaDivisors,
    DeltaMultiplication,
    DeltaOpenings,
    PublicInputs,
}

impl VerificationPhase {
    pub const ALL: [Self; 10] = [
        Self::Structure, Self::AnchorDynamo, Self::AnchorConsistency,
        Self::AnchorRelations, Self::DeltaDynamo, Self::DeltaConsistency,
        Self::DeltaDivisors, Self::DeltaMultiplication, Self::DeltaOpenings,
        Self::PublicInputs,
    ];

    pub fn name(self) -> &'static str {
        // All labels are fixed string literals with program-wide lifetime.
        match self {
            Self::Structure => "verify_structure",
            Self::AnchorDynamo => "verify_anchor_dynamo",
            Self::AnchorConsistency => "verify_anchor_consistency",
            Self::AnchorRelations => "verify_anchor_relations",
            Self::DeltaDynamo => "verify_delta_dynamo",
            Self::DeltaConsistency => "verify_delta_consistency",
            Self::DeltaDivisors => "verify_delta_divisors",
            Self::DeltaMultiplication => "verify_delta_multiplication",
            Self::DeltaOpenings => "verify_delta_openings",
            Self::PublicInputs => "verify_public_inputs",
        }
    }
}

/// Counts API calls and supplied terms, including identity points. These are
/// not counts of the nonzero terms executed inside arkworks' Miller loop.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PairingCounts {
    pub equations: usize,
    pub pairing_terms: usize,
    pub miller_loops: usize,
    pub final_exponentiations: usize,
}

impl PairingCounts {
    pub(crate) fn add(&mut self, other: Self) {
        self.equations += other.equations;
        self.pairing_terms += other.pairing_terms;
        self.miller_loops += other.miller_loops;
        self.final_exponentiations += other.final_exponentiations;
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct VerificationMeasurement {
    pub elapsed: Duration,
    pub miller_time: Duration,
    pub final_exponentiation_time: Duration,
    pub counts: PairingCounts,
    /// Disjoint from miller_time. Includes lookup and cloning on a hit.
    pub g2_prepare_time: Duration,
    pub g2_prepare_builds: usize,
    pub g2_prepare_hits: usize,
    /// Peak retained checker entries, excluding temporary consumed clones.
    pub prepared_peak_entries: usize,
}

#[derive(Clone, Debug)]
pub struct VerificationTrace {
    pub(crate) enabled: bool,
    phases: [VerificationMeasurement; 10],
}

impl VerificationTrace {
    pub fn new() -> Self {
        Self { enabled: true, phases: [VerificationMeasurement::default(); 10] }
    }

    pub(crate) fn disabled() -> Self {
        Self { enabled: false, phases: [VerificationMeasurement::default(); 10] }
    }

    pub fn phase(&self, phase: VerificationPhase) -> VerificationMeasurement {
        // Fieldless variants are consecutive array indices, starting at zero.
        self.phases[phase as usize]
    }

    pub fn recorded_total(&self) -> Duration {
        let mut total = Duration::ZERO;
        for phase in &self.phases { total += phase.elapsed; }
        total
    }

    pub fn counts(&self) -> PairingCounts {
        let mut total = PairingCounts::default();
        for phase in &self.phases { total.add(phase.counts); }
        total
    }

    pub(crate) fn start(&self) -> Option<Instant> {
        if self.enabled { Some(Instant::now()) } else { None }
    }

    pub(crate) fn measurement_mut(&mut self, phase: VerificationPhase) -> &mut VerificationMeasurement {
        &mut self.phases[phase as usize]
    }
}
