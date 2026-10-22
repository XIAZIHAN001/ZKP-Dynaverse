# Third-party references / 第三方依赖

Dependencies are fetched from their upstream projects; this repository does not bundle their prebuilt libraries.

依赖从各自上游获取，本仓库不附带其预编译库。

- [arkworks](https://github.com/arkworks-rs): BN254 arithmetic, polynomial operations, and the CPU reference stack. Exact resolved versions are in `Cargo.lock`.
- [ICICLE](https://github.com/ingonyama-zk/icicle/tree/aacec3f72f14f53dd96d835698d2c87cde3a68b8): pinned v2.8.0 CUDA operators. `native/CMakeLists.txt` verifies the pinned MSM source before generating instrumented host-side integration code.
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit): toolkit and development-container dependencies, pinned in `env/Dockerfile` and `env/versions.lock`.

Consult the respective upstream projects for their attribution and license files.
