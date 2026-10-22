#!/usr/bin/env bash
# Build CUDA smoke test for H200 (sm_90). COMPILE-ONLY by default;
# running requires a real H200 - do not mark sm_90 run-validated until it runs there.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source env/h200.env; set +a
mkdir -p build/sm90
# Use a Linux build cache so Windows bind mounts do not reject configure_file.
build_dir="${TMPDIR:-/tmp}/dynaverse-cuda-sm90"
cmake -S "$PWD/cuda-smoke" -B "$build_dir" -G Ninja \
    -DCMAKE_CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCHITECTURES" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir"
cp "$build_dir/vector_add" build/sm90/vector_add
echo "sm_90 build OK (compile-only validation). Running requires a real H200."
