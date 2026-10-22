#!/usr/bin/env bash
# Environment check for dynaverse-gpu. Exit non-zero on any failure, listing failures.
set -u

FAILURES=()
pass() { echo "[PASS] $1"; }
fail() { echo "[FAIL] $1"; FAILURES+=("$1"); }

# 1. nvcc
if command -v nvcc >/dev/null 2>&1; then
    NVCC_VER=$(nvcc --version | grep -o 'release [0-9.]*' | head -1)
    pass "nvcc found: $NVCC_VER"
else
    fail "nvcc not found in PATH"
fi

# 2. nvidia-smi
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    pass "GPU visible: $GPU_NAME"
else
    fail "nvidia-smi not available or GPU not visible"
fi

# 3. rustc / cargo
if command -v rustc >/dev/null 2>&1; then
    pass "rustc $(rustc --version)"
else
    fail "rustc not found"
fi
if command -v cargo >/dev/null 2>&1; then
    pass "cargo $(cargo --version)"
else
    fail "cargo not found"
fi

# 4. cmake
if command -v cmake >/dev/null 2>&1; then
    pass "cmake $(cmake --version | head -1)"
else
    fail "cmake not found"
fi

# 5. env file
ENV_FILE="${CUDA_ENV_FILE:-env/local.env}"
if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    set -a; source "$ENV_FILE"; set +a
    pass "loaded $ENV_FILE (CUDA_ARCH=$CUDA_ARCH, CARGO_TARGET_DIR=$CARGO_TARGET_DIR)"
else
    fail "env file not found: $ENV_FILE"
fi

# 6. results dir writable
if mkdir -p results 2>/dev/null && touch results/.write_test 2>/dev/null; then
    rm -f results/.write_test
    pass "results/ writable"
else
    fail "results/ not writable"
fi

# 7. ICICLE commit pinned in versions.lock
if grep -q "^commit = TBD" env/versions.lock 2>/dev/null; then
    echo "[WARN] ICICLE commit still unpinned in env/versions.lock"
else
    pass "ICICLE commit pinned in env/versions.lock"
fi

echo
if [ "${#FAILURES[@]}" -gt 0 ]; then
    echo "check_env: ${#FAILURES[@]} failure(s)."
    exit 1
fi
echo "check_env: all checks passed."
