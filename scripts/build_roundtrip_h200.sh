#!/usr/bin/env bash
# Compile the ICICLE MSM roundtrip harness for H200 (sm_90); run on H200 only.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source env/h200.env; set +a
mkdir -p build/sm90
cmake -S icicle-smoke -B build/sm90 -G Ninja \
    -DCMAKE_CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCHITECTURES" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build/sm90 --target roundtrip_msm
echo "roundtrip_msm sm_90 compile-only OK -> build/sm90/roundtrip_msm"
