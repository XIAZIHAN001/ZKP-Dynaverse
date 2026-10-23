# 构建与复现

独立 full-divisor 探针默认 CPU 构建无需 CUDA driver；GPU 需显式 `--features gpu`。见[复现与证据](FULL_DIVISOR_CACHE.md)。

[English](BUILD.md) | [项目说明](../README.md)

## CPU路径

使用Git checkout和Rust 1.89.0。基准程序在构建时记录Git来源标识。

```bash
cargo test --locked --workspace --release -- --test-threads=1
```

普通CPU测试不启用可选的`icicle` feature。需要显式执行的profiling和规模实验默认标记为ignored。

## 开发容器

以下命令在Linux主机或WSL2内执行。先在主机配置Docker和NVIDIA Container Toolkit；容器使用主机提供的驱动，WSL/容器内不另装Linux显卡驱动。

在仓库根目录构建固定版本镜像：

```bash
docker build -t dynaverse-dev:cuda12.8.1-sm120 \
  --build-arg CUDA_ARCH=120 env/
```

把ICICLE固定版本源码放在仓库外。下面是常规示例路径，也可以选择有写权限的位置并映射到相同容器路径。

```bash
sudo mkdir -p /opt/src /opt/icicle
sudo chown "$(id -u):$(id -g)" /opt/src /opt/icicle
git clone https://github.com/ingonyama-zk/icicle.git /opt/src/icicle-src
git -C /opt/src/icicle-src checkout aacec3f72f14f53dd96d835698d2c87cde3a68b8

docker run --rm --gpus all --user 0:0 \
  -v /opt/src/icicle-src:/src \
  -v /opt/icicle:/out \
  -v "$PWD/scripts:/scripts:ro" \
  dynaverse-dev:cuda12.8.1-sm120 bash /scripts/build_icicle.sh 120
```

进入开发容器，Cargo缓存独立保存：

```bash
docker run --rm -it --gpus all \
  -v "$PWD:/workspace" \
  -v /opt/src/icicle-src:/opt/src/icicle-src:ro \
  -v /opt/icicle:/opt/icicle:ro \
  -v dynaverse-cargo-registry:/usr/local/cargo/registry \
  -v dynaverse-cargo-git:/usr/local/cargo/git \
  -e RUSTUP_TOOLCHAIN=1.89.0 \
  -w /workspace \
  dynaverse-dev:cuda12.8.1-sm120 bash
```

在容器内执行：

```bash
set -a; source env/local.env; set +a
export CUDA_HOME=/usr/local/cuda
export DYNAVERSE_NATIVE_LIB_DIR=/workspace/build/native-sm120
export ICICLE_LIB_DIR=/opt/icicle/arch120

bash scripts/check_env.sh
bash scripts/build_native_icicle.sh local
cargo fetch --locked
cargo test --offline --locked --workspace --release -- --test-threads=1
```

Windows bind mount若遇到CMake权限问题，把CMake cache放在Linux文件系统：

```bash
DYNAVERSE_NATIVE_BUILD_DIR=/tmp/dynaverse-native-sm120 \
  bash scripts/build_native_icicle.sh local
```

脚本会把native归档复制到`build/native-sm120`。ICICLE和native库按架构分别构建，不提交到仓库。

## GPU正确性与自适应实验

在相同环境变量和native构建下执行：

```bash
cargo test --locked --release -p dynaverse-gpu --features icicle \
  --test icicle_backend -- --test-threads=1

cargo fetch --locked --manifest-path experiments/adaptive-fullopt/Cargo.toml
cargo run --offline --locked --release \
  --manifest-path experiments/adaptive-fullopt/Cargo.toml -- fullopt-small
```

探针模式：

| 模式 | 范围 |
|---|---|
| `fullopt-small` | n=256，包含功能序列和独立复用审计 |
| `fullopt-medium` | n=1024，成对大批update |
| `fullopt` | n=256/65536完整矩阵 |
| `readback-check` | n=1024读回校准，与CPU逐点对照 |

策略测试和JSON验收规则回归：

```bash
cargo test --locked --release \
  --manifest-path experiments/adaptive-fullopt/Cargo.toml --bin fullopt_probe
python3 -B -m unittest discover \
  -s experiments/adaptive-fullopt -p 'test_*.py' -v
```

CPU调度回归中，旧窗口是故意保留的失败复现；修正窗口应通过：

```bash
cargo run --locked --release \
  --manifest-path experiments/adaptive-fullopt/cpu-check/Cargo.toml -- 4
# 预期：非零退出，geometry命中为0。
cargo run --locked --release \
  --manifest-path experiments/adaptive-fullopt/cpu-check/Cargo.toml -- 6
# 预期：GEOMETRY_OK hits=5。
```

无需GPU即可分析已保留的真实事件：

```bash
cd experiments/adaptive-fullopt
python3 -B -c 'from pathlib import Path; from analysis import analyze_log; print(analyze_log(Path("fixtures/n256.log").read_text(), (256,))["matrix"][0]["observed"])'
```

## H200

ICICLE使用架构`90`构建，加载`env/h200.env`，设置`ICICLE_LIB_DIR=/opt/icicle/arch90`，通过`bash scripts/build_native_icicle.sh h200`构建桥接库。构建Rust GPU目标前，设置`DYNAVERSE_NATIVE_LIB_DIR=/workspace/build/native-sm90`。

请在自己的集群资源分配中调度`fullopt`矩阵。公开仓库不包含账户专用的SSH别名、Slurm目录和提交凭据。H200作业1157已对公开探针及对应核心源码完成n=256/65536验收，见[验收报告](SUPPLEMENTAL_TESTS.md)。在其他构建或硬件上复现仍需实际运行并核对原始事件。
