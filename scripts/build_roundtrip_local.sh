#!/usr/bin/env bash
# Compile the ICICLE MSM roundtrip harness for the local GPU (sm_120).
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source env/local.env; set +a
mkdir -p build/sm120
cmake -S icicle-smoke -B build/sm120 -G Ninja \
    -DCMAKE_CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCHITECTURES" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build/sm120 --target roundtrip_msm
echo "roundtrip_msm sm_120 build OK -> build/sm120/roundtrip_msm"
