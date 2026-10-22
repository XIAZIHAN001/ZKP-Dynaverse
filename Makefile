# Environment commands (load with: source env/local.env, then make <target>)

include env/local.env
export CUDA_ARCH CMAKE_CUDA_ARCHITECTURES CARGO_TARGET_DIR

.PHONY: check build-local run-local build-h200 roundtrip-local roundtrip-h200 clean-local

check:
	./scripts/check_env.sh

build-local:
	./scripts/build_local.sh

run-local: build-local
	./build/sm120/vector_add

build-h200:
	./scripts/build_h200.sh

roundtrip-local:
	./scripts/build_roundtrip_local.sh

roundtrip-h200:
	./scripts/build_roundtrip_h200.sh

clean-local:
	rm -rf build/sm120 target/sm120
