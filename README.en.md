# ZKP-Dynaverse

This experimental branch adds an opt-in full-domain public G2 divisor cache. The local n=1024 GPU two-update sample is5.844→4.878s (1.198×), with construction inside separately reported initialization and complete independent validation/work accounting preserved. The CPU workspace passed285 tests and the default GPU suite154; independent review passed. H200 job1159 was submitted once and is not yet accepted. See the [bilingual experiment report](docs/FULL_DIVISOR_CACHE.md).

Prepared-cache tuning is isolated in an experimental branch; the R5 strategy remains unchanged. Local two-update sample ratios against the fully optimized 64-root control are 1.149× at n=256 and 1.348× at n=1024. The H200 n=65536 scan passed acceptance: 64→4096 roots changes the two-update sum from 380.454 s to 373.280 s (1.019×); the large-domain gain is modest. See the [bilingual experiment notes](docs/PREPARED_CACHE_TUNING.md); these ratios are not end-to-end speedups.

English | [中文](README.md)

A Dynaverse incremental-proof research prototype built with Rust, arkworks BN254, and CUDA/ICICLE. It retains a CPU reference and calls GPU operators through a thin Rust interface to study sparse updates, staged rebuilding, and cache reuse.

The current protocol is an **unmasked correctness prototype**. It does not provide complete zero-knowledge guarantees and has not received a production security audit.

## Implementation

- Rust protocol layer: Dynaverse, Dynamo, KZG, consistency, SRS, and public verification.
- CPU/GPU operator and proof comparisons, with the public Reference verification path retained.
- GPU operators: G1/G2 MSM, Fr NTT, resident transforms, fixed-base setup, and G2 validation.
- Incremental options: indexed sparse MSM, foreground linear updates, frozen linear/G2 reuse, support geometry, and unchanged-cache reuse.
- Rebuilding options: row batching, bounded transform packets, Product cache checks, prepared roots, and coalesced readback.
- An external size-based experiment selects combinations and explicitly enables bounded G1 decoding and optional device affine readback.

Protocols and data structures remain in Rust. Performance-sensitive operators reuse ICICLE or CUDA C++. [Repository constraints](AGENTS.md) describe the language boundary and CPU-reference requirements.

## Validation status

The initial public snapshot's adaptive bundle completed paired n=256/1024 experiments on an RTX 5070 Ti Laptop GPU (sm_120). Each size runs four large updates: two warmups followed by two measured updates. Every update checks Reference verification, proof/witness equality, and its work allowance.

| n | Window | Baseline: two updates, s | Optimized bundle: two updates, s | Baseline / bundle |
|---:|---:|---:|---:|---:|
| 256 | 6 | 16.156810 | 1.711010 | 9.443 |
| 1024 | 8 | 37.424205 | 5.249226 | 7.129 |

Both routes use the same window and inputs. Timing excludes readback calibration and external Reference verification. Ratios describe the combined configuration, not individual options. These are small local samples, **not H200 speedups, stable tail-latency estimates, or end-to-end speedups**.

At n=256, the full bundle recorded 11 geometry-reuse hits; the separate geometry audit recorded five hits, and the separate linear audit passed. See [validation boundaries](docs/VALIDATION.md), [raw events](experiments/adaptive-fullopt/fixtures/n256.log), and [initial structured results](docs/results/local-2026-10-22.json).

Supplemental tests on 2026-10-22 passed all default GPU gates, 13 Python regressions, and a new n=1024 comparison, with a baseline/bundle ratio of 7.523 for the new sample. **H200 job 1157 completed the n=256/65536 matrix with terminal state COMPLETED/0:0; raw-event acceptance and integrity checks passed.**

| H200 n | Window | Baseline: two updates, s | Bundle: two updates, s | Baseline / bundle |
|---:|---:|---:|---:|---:|
| 256 | 6 | 17.553499 | 2.116730 | 8.293 |
| 65536 | 16 | 1254.203179 | 379.355222 | 3.306 |

Each H200 size also has only two timed updates, excluding calibration and external Reference verification. These ratios do not establish end-to-end speedup or statistical significance. Across 821 process samples, no OOM occurred; observed host VmHWM was about 1.30 GiB and the largest GPU sample about 2.35 GiB, not certified resource peaks. The [bilingual supplemental report](docs/SUPPLEMENTAL_TESTS.md) and [H200 raw events](experiments/adaptive-fullopt/fixtures/h200-20261022.log) record actual hits, provenance, and limits.

## Size-based strategy

[`Strategy::for_size`](experiments/adaptive-fullopt/strategy.rs) freezes the experiment configuration before session creation. Protocol defaults are unchanged.

| n | Window / H | Transforms | Prepared/readback roots | Undo arena |
|---|---|---|---|---|
| 64–512 | 6 / 3 | Packets for lengths ≤4096 | 64 | 64 MiB |
| 1024–4096 | 8 / 4 | Packets where eligible, otherwise tiled | 64 | 64 MiB |
| 8192–65536 | 16 / 8 | Tiled, row batch 16 | 64 | Next power of two above 1472n, minimum 64 MiB |

n=65536 uses a 128 MiB undo arena. The 2 GiB limit is **per backend owner**, not a process-wide memory bound.

The optimized bundle enables bounded G1 decoding. Device affine readback remains disabled below n=1024. Larger sizes first compare 64-point host/device readback on a disposable owner and select the device path only when its median is at least 3% faster. Calibration checks every output against an independent CPU result and is excluded from update timing. These are experimental thresholds, not proven optima for every GPU or update shape.

## CPU quick start

Install Git and Rust 1.89.0. `rust-toolchain.toml` selects the pinned toolchain. CPU tests do not require CUDA.

```bash
git clone https://github.com/XIAZIHAN001/ZKP-Dynaverse.git
cd ZKP-Dynaverse
cargo test --locked --workspace --release -- --test-threads=1
```

Example CPU benchmark:

```bash
mkdir -p results
cargo run --locked -p dynaverse-bench --bin cpu_dynaverse --release -- \
  --n 256 --updates 8 --policies dense,precompute \
  --workloads hotspot,spreading --warmup 1 --samples 3 \
  --output results/cpu_dynaverse_n256.csv
```

The benchmark records the Git commit and build-source identity. Build from a Git checkout rather than a source ZIP without `.git` metadata.

## GPU build and experiments

GPU execution requires Linux/WSL2, an NVIDIA driver, CUDA, CMake, and ICICLE v2.8.0 libraries built for the target architecture. Local runs use sm_120; H200 uses sm_90. Build their artifacts separately.

See the [English build guide](docs/BUILD.md) / [中文构建指南](docs/BUILD.zh-CN.md). After building the native bridge and setting library paths, run inside the development container:

```bash
# Complete small-size bundle
cargo run --locked --release \
  --manifest-path experiments/adaptive-fullopt/Cargo.toml -- fullopt-small

# Complete medium-size bundle
cargo run --locked --release \
  --manifest-path experiments/adaptive-fullopt/Cargo.toml -- fullopt-medium

# Full n=256/65536 matrix: requires sufficient memory and runtime
cargo run --locked --release \
  --manifest-path experiments/adaptive-fullopt/Cargo.toml -- fullopt
```

The probe emits `ROW {json}` events. It prints the success marker only after the entire selected matrix passes. The analyzer checks paired samples, actual hits, resource plans, and calibration decisions. Run acceptance regressions independently:

```bash
python3 -B -m unittest discover \
  -s experiments/adaptive-fullopt -p 'test_*.py' -v
cargo test --locked --release \
  --manifest-path experiments/adaptive-fullopt/Cargo.toml --bin fullopt_probe
```

## Layout

```text
crates/protocol/              Rust protocols and CPU reference
crates/gpu/                   GPU wrapper, codecs, and correctness tests
crates/bench/                 CPU/GPU benchmark entry points
native/                       CUDA C++ bridge and resident operators
experiments/adaptive-fullopt/ Size policy, calibration, probe, acceptance rules
env/                          Development image and pinned versions
docs/                         Build instructions, validation boundaries, roadmap
```

This public snapshot preserves core Rust/CUDA sources and omits machine-specific SSH/Slurm deployment scripts, execution ledgers, build caches, and server logs. [The source snapshot manifest](docs/SOURCE_SNAPSHOT.json) records its origin.

## Next steps

n=65536 H200 acceptance passed. Next, collect independent repeats and additional update shapes to evaluate window/readback choices, then account for actual update support and aggregate memory use. See the [roadmap](docs/ROADMAP.md).

Pinned dependencies are recorded in [`env/versions.lock`](env/versions.lock). See [third-party references](THIRD_PARTY_NOTICES.md). No repository license has been selected yet.
