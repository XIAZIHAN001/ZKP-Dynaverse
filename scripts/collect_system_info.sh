#!/usr/bin/env bash
# Collect full system info into results/environment.txt (append-safe).
set -u
OUT="results/environment.txt"
mkdir -p results
{
    echo "=== $(date -Is) ==="
    echo "--- OS ---"; uname -a; cat /etc/os-release | head -2
    echo "--- GPU ---"; nvidia-smi || echo "nvidia-smi failed"
    echo "--- nvcc ---"; nvcc --version 2>/dev/null || echo "nvcc not found"
    echo "--- rust ---"; rustc --version 2>/dev/null; cargo --version 2>/dev/null
    echo "--- cmake ---"; cmake --version 2>/dev/null | head -1
    echo "--- docker ---"; docker --version 2>/dev/null
    echo "--- container test ---"
    docker run --rm --gpus all nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97 \
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo "container GPU test failed"
} >> "$OUT"
echo "Written to $OUT"
