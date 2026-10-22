#!/usr/bin/env bash
# Build CUDA smoke test for the LOCAL GPU (sm_120, RTX 5070 Ti Laptop).
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source env/local.env; set +a
mkdir -p build/sm120
# CMake configure_file changes permissions, which Windows bind mounts may reject.
# Keep its cache on the container's Linux filesystem and publish only the binary.
build_dir="${TMPDIR:-/tmp}/dynaverse-cuda-sm120"
cmake -S "$PWD/cuda-smoke" -B "$build_dir" -G Ninja \
    -DCMAKE_CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCHITECTURES" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir"
cp "$build_dir/vector_add" build/sm120/vector_add
echo "sm_120 build OK. Run: ./build/sm120/vector_add"
