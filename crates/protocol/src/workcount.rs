//! Thread-local operation-size counters for complexity checks.
//!
//! Wall-clock time at one size cannot show an asymptotic bound. These counters
//! record the operand length of every polynomial transform, MSM, polynomial
//! product/division and dense witness scan made on the calling thread, so a
//! test can check whether a code path still touches length-n data.

use std::cell::Cell;

// Counting happens at protocol call sites: KZG commits (also before dispatch
// to a caller-supplied backend), the private CPU reference backend used by
// Dynamo/consistency, prover interpolation and verifier MSMs. Work inside an
// external backend is counted once at its call site, not inside the backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkCount {
    /// Sum of NTT/IFFT lengths (field and group transforms).
    pub ntt_elements: u64,
    /// Sum of G1 MSM lengths, including KZG commitments.
    pub msm_points: u64,
    /// Sum of G2 MSM lengths.
    pub g2_msm_points: u64,
    /// Sum of operand lengths in polynomial products and divisions.
    pub poly_elements: u64,
    /// Elements read by dense witness scans, clones and comparisons.
    pub scan_elements: u64,
    /// Circuit gates checked by full or local validation.
    pub gate_checks: u64,
    /// Largest single operand recorded above.
    pub max_operand: u64,
}

impl WorkCount {
    /// Sum of all additive counters (not `max_operand`).
    pub fn total(&self) -> u64 {
        self.ntt_elements + self.msm_points + self.g2_msm_points + self.poly_elements
            + self.scan_elements + self.gate_checks
    }
}

thread_local! {
    // Cell gives interior mutability without a RefCell borrow flag; WorkCount
    // is Copy, so get/set move the whole small struct by value.
    static COUNTS: Cell<WorkCount> = Cell::new(WorkCount::default());
}

/// Add `amount` to one counter and raise `max_operand` to `operand`.
/// `field` is a plain function pointer that picks which counter to bump.
fn record(amount: usize, operand: usize, field: fn(&mut WorkCount) -> &mut u64) {
    COUNTS.with(|counts| {
        let mut current = counts.get();
        *field(&mut current) += amount as u64;
        let operand = operand as u64;
        if operand > current.max_operand { current.max_operand = operand; }
        counts.set(current);
    });
}

pub(crate) fn record_ntt(len: usize) { record(len, len, |c| &mut c.ntt_elements); }
pub(crate) fn record_msm(len: usize) { record(len, len, |c| &mut c.msm_points); }
pub(crate) fn record_g2_msm(len: usize) { record(len, len, |c| &mut c.g2_msm_points); }
/// `work` is the arithmetic estimate; `operand` is the largest input length.
pub(crate) fn record_poly(work: usize, operand: usize) {
    record(work, operand, |c| &mut c.poly_elements);
}
pub(crate) fn record_scan(len: usize) { record(len, len, |c| &mut c.scan_elements); }
/// Gate checks do not raise `max_operand`: each gate is a constant-size check.
pub(crate) fn record_gates(count: usize) { record(count, 0, |c| &mut c.gate_checks); }

/// Run `f` and return the work it recorded on this thread. Nested calls are
/// allowed; the outer measurement still includes the inner work.
pub fn measure<T>(f: impl FnOnce() -> T) -> (T, WorkCount) {
    let before = COUNTS.with(|counts| counts.get());
    // Reset only the maximum so this measurement reports its own largest operand.
    COUNTS.with(|counts| {
        let mut reset = before;
        reset.max_operand = 0;
        counts.set(reset);
    });
    let value = f();
    let after = COUNTS.with(|counts| counts.get());
    let work = WorkCount {
        ntt_elements: after.ntt_elements - before.ntt_elements,
        msm_points: after.msm_points - before.msm_points,
        g2_msm_points: after.g2_msm_points - before.g2_msm_points,
        poly_elements: after.poly_elements - before.poly_elements,
        scan_elements: after.scan_elements - before.scan_elements,
        gate_checks: after.gate_checks - before.gate_checks,
        max_operand: after.max_operand,
    };
    COUNTS.with(|counts| {
        let mut restored = after;
        restored.max_operand = before.max_operand.max(after.max_operand);
        counts.set(restored);
    });
    (value, work)
}
