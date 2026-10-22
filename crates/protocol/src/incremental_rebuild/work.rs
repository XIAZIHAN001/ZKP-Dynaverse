//! Checked deterministic work budgets for bounded incremental rebuilding.
//!
//! Units are fixed normalized bookkeeping costs, never nanoseconds or weights
//! fitted to benchmarks. A caller must first expand its actual algorithm into
//! the primitive upper bounds below, including copying and transaction work.

use anyhow::{ensure, Context, Result};

// Fixed BN254 projective addition/doubling formulas take fewer than 64 base
// field operations in G1 and 256 after expanding the Fq2 formulas in G2.
// A 254-bit binary scalar multiplication uses at most 508 additions/doublings;
// charging 1024 gives room for the fixed conversion/normalization work too.
// Other algorithms must still supply an upper bound on their actual primitive
// counts; an MSM point is not interchangeable with one group addition.
const FIELD_WEIGHT: u128 = 1;
const G1_ADD_WEIGHT: u128 = 64;
const G1_MUL_WEIGHT: u128 = 1024 * G1_ADD_WEIGHT;
const G2_ADD_WEIGHT: u128 = 256;
const G2_MUL_WEIGHT: u128 = 1024 * G2_ADD_WEIGHT;
const COPY_BYTE_WEIGHT: u128 = 1;
// These charge the bookkeeping of a bounded element visit or submission.
// Arithmetic, encoding bytes and other work within it are counted separately.
const HOST_ELEMENT_WEIGHT: u128 = 256;
const LAUNCH_WEIGHT: u128 = 1024;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkBound {
    pub field_ops: u64,
    pub g1_adds: u64,
    pub g1_muls: u64,
    pub g2_adds: u64,
    pub g2_muls: u64,
    pub copy_bytes: u64,
    pub host_elements: u64,
    pub launches: u64,
}

impl WorkBound {
    /// One common currency for a dependency chain of differently typed work.
    /// The loop has exactly eight entries, independent of input/domain size.
    pub fn units(&self) -> Result<u128> {
        let entries = [
            (self.field_ops, FIELD_WEIGHT),
            (self.g1_adds, G1_ADD_WEIGHT),
            (self.g1_muls, G1_MUL_WEIGHT),
            (self.g2_adds, G2_ADD_WEIGHT),
            (self.g2_muls, G2_MUL_WEIGHT),
            (self.copy_bytes, COPY_BYTE_WEIGHT),
            (self.host_elements, HOST_ELEMENT_WEIGHT),
            (self.launches, LAUNCH_WEIGHT),
        ];
        let mut total = 0u128;
        for (count, weight) in entries {
            let contribution = u128::from(count).checked_mul(weight)
                .context("work-unit multiplication overflow")?;
            total = total.checked_add(contribution).context("work-unit addition overflow")?;
        }
        Ok(total)
    }

    /// Add primitive upper bounds without wrapping any individual counter.
    pub fn checked_add(self, rhs: Self) -> Result<Self> {
        Ok(Self {
            field_ops: self.field_ops.checked_add(rhs.field_ops).context("field work overflow")?,
            g1_adds: self.g1_adds.checked_add(rhs.g1_adds).context("G1 addition work overflow")?,
            g1_muls: self.g1_muls.checked_add(rhs.g1_muls).context("G1 multiplication work overflow")?,
            g2_adds: self.g2_adds.checked_add(rhs.g2_adds).context("G2 addition work overflow")?,
            g2_muls: self.g2_muls.checked_add(rhs.g2_muls).context("G2 multiplication work overflow")?,
            copy_bytes: self.copy_bytes.checked_add(rhs.copy_bytes).context("copy-byte work overflow")?,
            host_elements: self.host_elements.checked_add(rhs.host_elements).context("host-element work overflow")?,
            launches: self.launches.checked_add(rhs.launches).context("submission work overflow")?,
        })
    }

    /// Scale a repeated primitive bound, checking the stored u64 counters.
    pub fn checked_mul(self, factor: u64) -> Result<Self> {
        Ok(Self {
            field_ops: self.field_ops.checked_mul(factor).context("field work overflow")?,
            g1_adds: self.g1_adds.checked_mul(factor).context("G1 addition work overflow")?,
            g1_muls: self.g1_muls.checked_mul(factor).context("G1 multiplication work overflow")?,
            g2_adds: self.g2_adds.checked_mul(factor).context("G2 addition work overflow")?,
            g2_muls: self.g2_muls.checked_mul(factor).context("G2 multiplication work overflow")?,
            copy_bytes: self.copy_bytes.checked_mul(factor).context("copy-byte work overflow")?,
            host_elements: self.host_elements.checked_mul(factor).context("host-element work overflow")?,
            launches: self.launches.checked_mul(factor).context("submission work overflow")?,
        })
    }
}

/// Initialization-time certificate for the greedy, unified work ledger.
///
/// If every next chunk costs at most Cmax, every unfinished call leaves less
/// than Cmax of its kQ allowance unused. Over the first calls obtaining at
/// least H changes, the spent capacity is at least H(Q-Cmax), covering W.
/// No unused per-type or per-call credit needs to carry over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RebuildBudget {
    pub q: u128,
    pub horizon: usize,
    pub total: u128,
    pub max_chunk: u128,
    pub large: u128,
}

impl RebuildBudget {
    pub fn new(total: WorkBound, max_chunk: WorkBound, large: WorkBound, h: usize) -> Result<Self> {
        ensure!(h > 0, "rebuild horizon must be positive");
        let total = total.units()?;
        let max_chunk = max_chunk.units()?;
        let large = large.units()?;
        let changes = u128::try_from(h).context("rebuild horizon does not fit u128")?;
        let large_changes = changes.checked_add(1).context("large-batch horizon overflow")?;
        let small_q = ceil_div(total, changes)?.checked_add(max_chunk)
            .and_then(|value| value.checked_add(1)).context("rebuild quota overflow")?;
        let q = small_q.max(ceil_div(large, large_changes)?);

        // Check the proof inequalities too: storing a wrapped certificate
        // would make later per-call accounting appear valid without a bound.
        let usable = q.checked_sub(max_chunk).context("quota below maximum chunk")?;
        let small_capacity = changes.checked_mul(usable).context("deadline capacity overflow")?;
        let large_capacity = large_changes.checked_mul(q).context("large-batch capacity overflow")?;
        ensure!(q > max_chunk, "rebuild quota must exceed maximum chunk");
        ensure!(small_capacity >= total, "rebuild deadline capacity is insufficient");
        ensure!(large_capacity >= large, "large-batch capacity is insufficient");
        Ok(Self { q, horizon: h, total, max_chunk, large })
    }

    /// k is the number of accepted actual changes, including paid large batches.
    /// A no-op has zero allowance; this method does not mutate or bank credit.
    pub fn allowance(&self, k: usize) -> Result<u128> {
        let changes = u128::try_from(k).context("change count does not fit u128")?;
        self.q.checked_mul(changes).context("per-call rebuild allowance overflow")
    }
}

/// Cached even window B=2H. All roots and logarithm estimates are computed only
/// during construction, never while awarding an update's work allowance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CachedWindow {
    pub window: usize,
    pub horizon: usize,
}

impl CachedWindow {
    pub fn new(n: usize, requested: Option<usize>, optimized: bool) -> Result<Self> {
        ensure!(n > 0, "circuit size must be positive");
        let limit = ceil_sqrt(n)?.max(2);
        let request = match requested {
            Some(value) => value,
            None if optimized => {
                // For n>0, bit_length(n) equals ceil(log2(n+1)); unlike
                // explicitly constructing n+1, this works at usize::MAX.
                let logarithm = usize::try_from(usize::BITS - n.leading_zeros())
                    .context("window logarithm does not fit usize")?;
                let reduced = usize::try_from(ceil_div(n as u128, logarithm as u128)?)
                    .context("optimized window input does not fit usize")?;
                ceil_sqrt(reduced)?.max(2)
            }
            None => limit,
        };
        ensure!(request >= 2 && request <= limit, "requested rebuild window is outside supported range");
        let horizon = (request / 2).checked_add(request % 2).context("rebuild horizon overflow")?;
        let window = horizon.checked_mul(2).context("normalized rebuild window overflow")?;
        Ok(Self { window, horizon })
    }
}

fn ceil_div(value: u128, divisor: u128) -> Result<u128> {
    ensure!(divisor > 0, "work-bound divisor must be positive");
    let remainder = u128::from(value % divisor != 0);
    (value / divisor).checked_add(remainder).context("rounded work-bound division overflow")
}

/// Integer binary search avoids both a squared midpoint and value+1, so even
/// usize::MAX is supported. Only initialization calls this logarithmic loop.
fn ceil_sqrt(value: usize) -> Result<usize> {
    if value == 0 { return Ok(0); }
    let mut lower = 1;
    let mut upper = value;
    while lower < upper {
        let middle = lower + (upper - lower) / 2;
        let quotient = value / middle;
        let ceiling = quotient.checked_add(usize::from(value % middle != 0))
            .context("integer square-root division overflow")?;
        if middle >= ceiling {
            upper = middle;
        } else {
            lower = middle.checked_add(1).context("integer square-root bound overflow")?;
        }
    }
    Ok(lower)
}

#[cfg(test)]
#[path = "work_tests.rs"]
mod tests;
