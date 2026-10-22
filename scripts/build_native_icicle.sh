#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
profile=${1:-local}
case "$profile" in
  # Windows checkouts can use CRLF; remove CR before sourcing Bash assignments.
  local) source <(tr -d '\r' < env/local.env) ;;
  h200) source <(tr -d '\r' < env/h200.env) ;;
  *) echo "usage: build_native_icicle.sh [local|h200]" >&2; exit 2 ;;
esac
# Docker builds over a Windows mount can reject configure_file permission
# copying. Set this to a Linux filesystem directory in that environment.
build_dir=${DYNAVERSE_NATIVE_BUILD_DIR:-"$PWD/build/native-sm${CUDA_ARCH}"}
cmake -S native -B "$build_dir" -G Ninja \
  -DCMAKE_CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCHITECTURES" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" --target dynaverse_icicle
output_dir="$PWD/build/native-sm${CUDA_ARCH}"
if [[ "$build_dir" != "$output_dir" ]]; then
  mkdir -p "$output_dir"
  cp "$build_dir/libdynaverse_icicle.a" "$output_dir/"
fi
echo "Use DYNAVERSE_NATIVE_LIB_DIR=$PWD/build/native-sm${CUDA_ARCH} ICICLE_LIB_DIR=/opt/icicle/arch${CUDA_ARCH} cargo ... --features icicle"
