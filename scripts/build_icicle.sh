#!/bin/bash
# Build ICICLE v2.8.0 BN254 (MSM+NTT) for a given CUDA_ARCH and copy libs to /out.
# Usage: build_icicle.sh <arch> <out-dir-label>
set -e
ARCH=${1:-120}
cmake -S /src/icicle -B /tmp/b -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=$ARCH -DCURVE=bn254 -DMSM=ON -DNTT=ON -DBUILD_TESTS=OFF > /tmp/cfg.log 2>&1
cmake --build /tmp/b -j24 > /tmp/mk.log 2>&1
echo "BUILD_OK arch=$ARCH"
ls /tmp/b/lib/
mkdir -p /out/arch${ARCH}
cp /tmp/b/lib/*.a /out/arch${ARCH}/
echo "COPIED to /out/arch${ARCH}"
