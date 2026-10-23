# Full G2 Divisor Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add an opt-in bounded public affine-divisor table shared by initialization and updates, without weakening independent cache equations.

**Architecture:** A private owner/SRS/domain-bound table constructs at most64 roots at a time. Program admission charges its host envelope before allocations; existing prepared64 takes precedence during updates. Existing exact initial checks consume the same public table and initialization reports build/use counts separately.

**Tech Stack:** Rust1.89.0, pinned arkworks BN254, existing CUDA/ICICLE backend; no new dependency or native changes.

**Spec:** `docs/superpowers/specs/2026-10-23-full-divisor-cache-design.md` (approved by user).

## Global Constraints

- Independent worktree based on518bfed; preserve main, PR1 and1157/1158 seals.
- Default off; Product-only when enabled; n<=65536 only for enabled path.
- Fixed64-root construction chunks; root plus affine point, no prepared coefficients or acceptance cache.
- Charge actual Rust/capacity host storage, transient construction, initialization and lookup work.
- Retain per-owner2GiB, original undo, full paid rebuild and independent equations.
- Local gates before a single H20065536 submission; no automation or automatic resubmission.

## Review Focus

- Invalid mode/domain/budget must reject before backend or table allocation (Task1).
- Same-SRS cloned/foreign owner and root reorder must not reuse a table (Task1).
- Prepared-prefix/full-table boundary must not skip or duplicate equations (Task2).
- Failure after an already-written first cache block must fully roll back and retry (Task2).
- Initial build cost and hits cannot be omitted or confused with update hits (Tasks2/3).

### Task 1: Public table and resource admission

**Files:** create `crates/protocol/src/dynaverse/rebuilding/divisors.rs`; modify
`incremental_rebuild.rs`, `dynaverse/rebuilding.rs`, `dynaverse/rebuilding/program.rs`.

**Interfaces:** `IncrementalConfig::rebuild_divisor_cache:bool=false`;
`PublicDivisors::{host_bound(n,enabled,mode)->Result<u64>,new(params,host_upper)->Result<Self>,validate_owner(params)->Result<()>,get(index,root)->Result<G2Affine>,stats()->CacheDivisorStats}`.
`ResourcePlan::cache_divisor_host_bytes:u64`; `CacheDivisorStats` exposes roots,
construction_divisors, retained_bytes, host_upper_bytes, precompute_ns.

- [x] Add behavior tests in `divisors.rs`: `public_divisors_are_exact_and_bound_to_owner_srs_root`, `divisor_table_admission_precedes_allocations`.
- [x] Watch expected failure before implementing table behavior (scaffold may delegate to old calculation without caching).
- [x] Implement checked bound from concrete Rust sizes/capacity plus64-root temporary envelope; construct chunks, validate identity and root on lookup.
- [x] Add preflight to Program before any allocation and allocate opt-in table; add host accounting/stats getter.
- [x] Run `cargo test --offline --locked --release -p dynaverse-protocol --lib divisor -- --test-threads=1`; expect exact/negative tests pass.
- [x] Commit table and admission with the approved design/plan.

### Task 2: Initialization, update reuse and safety

**Files:** modify `dynaverse/rebuilding/machine.rs`, `dynaverse/rebuilding.rs`,
`incremental_rebuild/telemetry.rs`; create focused private test module for new checks.

**Interfaces:** `CacheCheckTimings::divisor_cache_hits:u64`;
`InitialCacheStats {equations,divisors_formed,divisor_cache_hits}` via session getter.
`Machine::build_initial` returns `(AnchorProof,InitialCacheStats)`; its only caller stores stats.
Keep test-only serial wrappers to preserve old exact-check tests.

- [x] Add test requiring full-table update fallback hits, zero formed divisors and unchanged equations/Miller/final exponents; run RED on unintegrated table.
- [x] Pass table into initial and update validators; prepared path stays first, full-table fallback uses affine points while checks remain4-root independent.
- [x] Charge extra lookup/copy bookkeeping in Program cache-command upper; do not subtract old conservative certificate.
- [x] Add independent exact/constant/partial/end ranges, corrupted commitment/value/opening and cancelling-errors tests through full-table path.
- [x] Add late-block corrupt-read rollback/retry test comparing proof/witness/receipt/timings/resident state and clean reference endpoint.
- [x] Assert initial construction=n, checker formed=0/hits=n/equations=3n when enabled; off remains formed=n/hits=0.
- [x] Run focused tests plus complete protocol suite; fix every regression and commit.

### Task 3: Paired full-flow measurement

**Files:** add separate external probe/analyzer/tests under `experiments/full-divisor-cache/`;
reuse immutable R5 size/calibration code by path, without editing original three probes.

**Interfaces:** raw JSON events with route off/on, initial table/build/check counts,
resource/budget,4 updates(2 warmups+2 measured), paired equality, final CPU dense,
existing nested timing categories and new real hit counts.

- [x] Add raw-evidence contract tests for missing pair, false correctness, hidden precompute, wrong construction/hits, reduced equations, quota, resource and late pending endpoint; demonstrate RED then implement analyzer.
- [x] Run n256/n1024 CPU and GPU with deterministic R5 inputs, prepared64/readback64/row16/window fixed, counterbalanced order.
- [x] Require same proof/witness, Reference each call, final CPU dense, all paid rebuilds complete, each own quota and reported conservative-budget delta.
- [x] Record update sample ratio and initialization/whole-route phase sums separately. Reject non-gain or resource violation before H200.
- [x] Commit probe, sanitized raw fixtures/results and updated core snapshot with explicit change provenance.

### Task 4: Full regression, independent review and H200

- [x] Run `cargo test --offline --locked --workspace --release -- --test-threads=1` and full default GPU package suite; no source edits while a bound run is active.
- [x] Request fresh independent branch review, repair important findings, rerun only affected gates.
- [x] If local gates/gain pass, create a new immutable H200 campaign with new Rust source identity and unchanged native library hashes; do not reuse old product identity for modified source.
- [x] Verify support/source/dependencies, admission, shell and binary SM90, submit once n65536 off/on with five-second telemetry.
- [x] On failure collect evidence once and report; on success require terminal0:0 plus local/remote raw audits and CPU oracle, then seal.
- [x] Publish sanitized bilingual evidence on a new draft PR; leave main and previous PR unchanged.
