# Building and reproducing

The separate full-divisor probe defaults to driverless CPU; add `--features gpu` for GPU. See [commands and evidence](FULL_DIVISOR_CACHE.md).

[中文](BUILD.zh-CN.md) | [Project overview](../README.en.md)

## CPU only

Use a Git checkout and Rust 1.89.0. The workspace benchmark embeds Git provenance at build time.

```bash
cargo test --locked --workspace --release -- --test-threads=1
```

The ordinary CPU suite does not enable the optional `icicle` feature. Explicit profiling and scale experiments are ignored by default and must be requested separately.

## Development container

The following commands run on a Linux host or inside WSL2. Configure Docker and NVIDIA Container Toolkit on the host first. The NVIDIA driver is supplied by the host; do not install a second Linux GPU driver inside a WSL/container environment.

Build the pinned development image from the repository root:

```bash
docker build -t dynaverse-dev:cuda12.8.1-sm120 \
  --build-arg CUDA_ARCH=120 env/
```

Fetch the exact ICICLE source outside the checkout. The paths below are conventional example locations; choose writable locations and mount them at the same container paths if needed.

```bash
sudo mkdir -p /opt/src /opt/icicle
sudo chown "$(id -u):$(id -g)" /opt/src /opt/icicle
git clone https://github.com/ingonyama-zk/icicle.git /opt/src/icicle-src
git -C /opt/src/icicle-src checkout aacec3f72f14f53dd96d835698d2c87cde3a68b8

docker run --rm --gpus all --user 0:0 \
  -v /opt/src/icicle-src:/src \
  -v /opt/icicle:/out \
  -v "$PWD/scripts:/scripts:ro" \
  dynaverse-dev:cuda12.8.1-sm120 bash /scripts/build_icicle.sh 120
```

Start a development shell, preserving Cargo caches separately from the source tree:

```bash
docker run --rm -it --gpus all \
  -v "$PWD:/workspace" \
  -v /opt/src/icicle-src:/opt/src/icicle-src:ro \
  -v /opt/icicle:/opt/icicle:ro \
  -v dynaverse-cargo-registry:/usr/local/cargo/registry \
  -v dynaverse-cargo-git:/usr/local/cargo/git \
  -e RUSTUP_TOOLCHAIN=1.89.0 \
  -w /workspace \
  dynaverse-dev:cuda12.8.1-sm120 bash
```

Inside that shell:

```bash
set -a; source env/local.env; set +a
export CUDA_HOME=/usr/local/cuda
export DYNAVERSE_NATIVE_LIB_DIR=/workspace/build/native-sm120
export ICICLE_LIB_DIR=/opt/icicle/arch120

bash scripts/check_env.sh
bash scripts/build_native_icicle.sh local
cargo fetch --locked
cargo test --offline --locked --workspace --release -- --test-threads=1
```

For Windows bind mounts, use a Linux-filesystem CMake cache when necessary:

```bash
DYNAVERSE_NATIVE_BUILD_DIR=/tmp/dynaverse-native-sm120 \
  bash scripts/build_native_icicle.sh local
```

The script publishes the resulting native archive under `build/native-sm120`. ICICLE and native libraries are architecture-specific and are not committed.

## GPU correctness and the adaptive experiment

With the same environment variables and native build:

```bash
cargo test --locked --release -p dynaverse-gpu --features icicle \
  --test icicle_backend -- --test-threads=1

cargo fetch --locked --manifest-path experiments/adaptive-fullopt/Cargo.toml
cargo run --offline --locked --release \
  --manifest-path experiments/adaptive-fullopt/Cargo.toml -- fullopt-small
```

Other probe modes:

| Mode | Scope |
|---|---|
| `fullopt-small` | n=256 plus feature and independent reuse audits |
| `fullopt-medium` | n=1024 paired large updates |
| `fullopt` | n=256 and n=65536 complete matrix |
| `readback-check` | n=1024 readback calibration against the CPU oracle |

Run the policy tests and JSON acceptance regressions:

```bash
cargo test --locked --release \
  --manifest-path experiments/adaptive-fullopt/Cargo.toml --bin fullopt_probe
python3 -B -m unittest discover \
  -s experiments/adaptive-fullopt -p 'test_*.py' -v
```

For the CPU scheduling regression, the old window deliberately fails its geometry-hit assertion; the corrected window passes:

```bash
cargo run --locked --release \
  --manifest-path experiments/adaptive-fullopt/cpu-check/Cargo.toml -- 4
# Expected: nonzero exit, zero geometry hits.
cargo run --locked --release \
  --manifest-path experiments/adaptive-fullopt/cpu-check/Cargo.toml -- 6
# Expected: GEOMETRY_OK hits=5.
```

The retained fixture events can be analyzed without a GPU:

```bash
cd experiments/adaptive-fullopt
python3 -B -c 'from pathlib import Path; from analysis import analyze_log; print(analyze_log(Path("fixtures/n256.log").read_text(), (256,))["matrix"][0]["observed"])'
```

## H200

Build ICICLE with architecture `90`, use `env/h200.env`, set `ICICLE_LIB_DIR=/opt/icicle/arch90`, and build the bridge with `bash scripts/build_native_icicle.sh h200`. Set `DYNAVERSE_NATIVE_LIB_DIR=/workspace/build/native-sm90` before building Rust GPU targets.

Schedule the `fullopt` matrix through your own cluster allocation. This public repository intentionally excludes account-specific SSH aliases, Slurm paths, and submission credentials. H200 job 1157 passed the n=256/65536 matrix for the published probe and core sources; see the [acceptance report](SUPPLEMENTAL_TESTS.md). Reproduction on another build or hardware still requires actual execution and raw-event validation.

H200 n=65536 off/on reproduction passed in job1159; see [phase report](H200_1159_PHASE_REPORT.md).
