use super::{CachedWindow, RebuildBudget, WorkBound};

fn fields(count: u64) -> WorkBound {
    WorkBound { field_ops: count, ..WorkBound::default() }
}

fn primitive(kind: usize, count: u64) -> WorkBound {
    let mut work = WorkBound::default();
    match kind {
        0 => work.field_ops = count,
        1 => work.g1_adds = count,
        2 => work.g1_muls = count,
        3 => work.g2_adds = count,
        4 => work.g2_muls = count,
        5 => work.copy_bytes = count,
        6 => work.host_elements = count,
        7 => work.launches = count,
        _ => unreachable!("test primitive index"),
    }
    work
}

#[test]
fn weighted_group_work_cannot_be_billed_as_field_work_or_additions() {
    let scalar = fields(1).units().unwrap();
    let g1_add = primitive(1, 1).units().unwrap();
    let g1_mul = primitive(2, 1).units().unwrap();
    let g2_add = primitive(3, 1).units().unwrap();
    let g2_mul = primitive(4, 1).units().unwrap();
    assert!(g1_add > scalar);
    assert!(g2_add > g1_add);
    // A full-width scalar multiplication needs hundreds of group operations.
    assert!(g1_mul >= 508 * g1_add);
    assert!(g2_mul >= 508 * g2_add);
    assert!(g2_mul > g1_mul);

    let mut all = WorkBound::default();
    let mut separate = 0;
    for kind in 0..8 {
        let work = primitive(kind, 3);
        all = all.checked_add(work).unwrap();
        separate += work.units().unwrap();
    }
    assert_eq!(all.units().unwrap(), separate);
    assert_eq!(WorkBound::default().units().unwrap(), 0);
}

#[test]
fn vector_arithmetic_rejects_each_counter_overflow_without_mutation() {
    for kind in 0..8 {
        let full = primitive(kind, u64::MAX);
        assert!(full.checked_add(primitive(kind, 1)).is_err(), "counter {kind}");
        assert!(full.checked_mul(2).is_err(), "counter {kind}");
        assert_eq!(full.checked_mul(0).unwrap(), WorkBound::default());
        assert_eq!(full.checked_mul(1).unwrap(), full);
        assert_eq!(full.checked_add(WorkBound::default()).unwrap(), full);
        assert!(full.units().is_ok());
    }
    let original = primitive(4, 13).checked_add(fields(7)).unwrap();
    let scaled = original.checked_mul(3).unwrap();
    assert_eq!(scaled.field_ops, 21);
    assert_eq!(scaled.g2_muls, 39);
    assert_eq!(scaled.units().unwrap(), 3 * original.units().unwrap());
}

#[test]
fn budget_reserves_fragmentation_and_the_paid_large_batch() {
    // Small-batch requirement gives 14 + 10 + 1 = 25; large gives 120/4 = 30.
    let budget = RebuildBudget::new(fields(40), fields(10), fields(120), 3).unwrap();
    assert_eq!(budget.q, 30);
    assert_eq!(budget.horizon, 3);
    assert_eq!(budget.total, 40);
    assert_eq!(budget.max_chunk, 10);
    assert_eq!(budget.large, 120);
    assert_eq!(budget.allowance(0).unwrap(), 0);
    assert_eq!(budget.allowance(2).unwrap(), 60);
    assert_eq!(budget.allowance(4).unwrap(), 120);
    assert!(budget.q > budget.max_chunk);
    assert!(3 * (budget.q - budget.max_chunk) >= budget.total);
    assert!(4 * budget.q >= budget.large);

    let small = RebuildBudget::new(fields(41), fields(10), fields(0), 4).unwrap();
    assert_eq!(small.q, 22); // ceil(41/4) + 10 + 1.
    let empty = RebuildBudget::new(fields(0), fields(0), fields(0), 1).unwrap();
    assert_eq!(empty.q, 1); // Strict Q > Cmax also holds for an empty job.
}

#[test]
fn invalid_horizon_and_budget_products_return_errors() {
    assert!(RebuildBudget::new(fields(1), fields(1), fields(1), 0).is_err());
    let huge = WorkBound { g2_muls: u64::MAX, ..WorkBound::default() };
    let valid = RebuildBudget::new(huge, huge, huge, 1).unwrap();
    assert!(valid.allowance(usize::MAX).is_err());
    assert!(RebuildBudget::new(huge, huge, huge, usize::MAX).is_err());
}

// This scheduler fixture executes a real dependency chain of costed chunks. It
// spends only the current call's allowance: unused credit never carries over.
fn advance(chunks: &[WorkBound], cursor: &mut usize, allowance: u128) -> u128 {
    let mut spent = 0;
    while *cursor < chunks.len() {
        let next = chunks[*cursor].units().unwrap();
        if next > allowance - spent { break; }
        spent += next;
        *cursor += 1;
    }
    spent
}

fn check_all_call_partitions(
    chunks: &[WorkBound], budget: RebuildBudget, cursor: usize, accepted: usize,
) {
    if accepted >= budget.horizon {
        assert_eq!(cursor, chunks.len(), "deadline at {accepted} changes, {budget:?}");
        return;
    }
    for k in 1..=budget.horizon {
        let mut next = cursor;
        let allowance = budget.allowance(k).unwrap();
        let spent = advance(chunks, &mut next, allowance);
        assert!(spent <= allowance);
        // The last batch may overshoot H by up to H-1 actual changes.
        check_all_call_partitions(chunks, budget, next, accepted + k);
    }
}

#[test]
fn unified_budget_completes_dependent_phases_for_every_small_batch_partition() {
    let alphabet = [fields(1), fields(19), primitive(1, 1), primitive(4, 1)];
    // Exhaust all 4^4 ordered chains, including sharply different phase costs.
    for encoded in 0..256usize {
        let mut digits = encoded;
        let mut chunks = Vec::new();
        let mut total = WorkBound::default();
        let mut largest = WorkBound::default();
        for _ in 0..4 {
            let chunk = alphabet[digits % alphabet.len()];
            digits /= alphabet.len();
            total = total.checked_add(chunk).unwrap();
            if chunk.units().unwrap() > largest.units().unwrap() { largest = chunk; }
            chunks.push(chunk);
        }
        for horizon in 1..=5 {
            let budget = RebuildBudget::new(total, largest, total, horizon).unwrap();
            check_all_call_partitions(&chunks, budget, 0, 0);
        }
    }
}

#[test]
fn fragmentation_and_a_threshold_overshoot_do_not_borrow_future_credit() {
    let chunks = vec![fields(7); 13];
    let budget = RebuildBudget::new(fields(91), fields(7), fields(182), 4).unwrap();
    let mut cursor = 0;
    for (k, expected_cursor) in [(1, 5), (1, 10), (3, 13)] {
        let allowance = budget.allowance(k).unwrap();
        let spent = advance(&chunks, &mut cursor, allowance);
        assert!(spent <= allowance);
        assert_eq!(cursor, expected_cursor);
    }
    assert_eq!(cursor, chunks.len());
}

#[test]
fn independent_type_totals_can_miss_a_dependency_deadline() {
    // Both type budgets satisfy 2*quota == total. Call one cannot use its G1
    // quota before both field chunks finish; that lost quota breaks call two.
    let chunks = [fields(1), fields(1), primitive(1, 1), primitive(1, 1)];
    let mut cursor = 0;
    for _ in 0..2 {
        let mut remaining_field = 1;
        let mut remaining_g1 = 1;
        while cursor < chunks.len() {
            let work = chunks[cursor];
            if work.field_ops > remaining_field || work.g1_adds > remaining_g1 { break; }
            remaining_field -= work.field_ops;
            remaining_g1 -= work.g1_adds;
            cursor += 1;
        }
    }
    assert_eq!(cursor, 3, "independent quotas leave the last G1 chunk unfinished");

    let total = fields(2).checked_add(primitive(1, 2)).unwrap();
    let budget = RebuildBudget::new(total, primitive(1, 1), total, 2).unwrap();
    check_all_call_partitions(&chunks, budget, 0, 0);
}

#[test]
fn cached_windows_normalize_odd_requests_and_enforce_the_supported_range() {
    let cases = [
        (1, 2, 2), (4, 2, 2), (5, 3, 4), (9, 3, 4), (10, 4, 4),
        (16, 4, 4), (17, 5, 6), (64, 8, 8), (65, 9, 10),
        (255, 16, 16), (256, 16, 16), (257, 17, 18),
    ];
    for (n, cap, normalized) in cases {
        let default = CachedWindow::new(n, None, false).unwrap();
        assert_eq!(default.window, normalized, "n={n}");
        assert_eq!(default.horizon, normalized / 2);
        assert!(CachedWindow::new(n, Some(cap + 1), false).is_err());
        for request in 2..=cap {
            let actual = CachedWindow::new(n, Some(request), false).unwrap();
            assert_eq!(actual.window % 2, 0);
            assert!(actual.window >= request && actual.window <= request + 1);
            assert_eq!(actual.horizon * 2, actual.window);
        }
    }
    assert!(CachedWindow::new(0, None, false).is_err());
    assert!(CachedWindow::new(64, Some(0), false).is_err());
    assert!(CachedWindow::new(64, Some(1), false).is_err());
    assert_eq!(CachedWindow::new(64, Some(3), true).unwrap().window, 4);
}

#[test]
fn optimized_window_uses_integer_rounding_and_stays_valid_near_usize_max() {
    for (n, expected) in [(1, 2), (5, 2), (64, 4), (256, 6), (1024, 10)] {
        assert_eq!(CachedWindow::new(n, None, true).unwrap().window, expected);
    }
    // usize::MAX + 1 and a naive sqrt midpoint/square would overflow here.
    let root = 1usize << (usize::BITS / 2);
    let below_square = (root - 1) * (root - 1);
    for n in [below_square - 1, below_square, below_square + 1, usize::MAX] {
        assert_eq!(CachedWindow::new(n, None, false).unwrap().window, root);
        let optimized = CachedWindow::new(n, None, true).unwrap();
        assert!(optimized.window >= 2 && optimized.window <= root);
        assert_eq!(optimized.horizon * 2, optimized.window);
        assert!(CachedWindow::new(n, Some(usize::MAX), false).is_err());
    }
}
