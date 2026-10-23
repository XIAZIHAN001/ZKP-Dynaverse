# ZKP-Dynaverse

本实验分支新增默认关闭的轻量全域 G2 divisor 缓存。n=1024 GPU 两笔 update 本地样本为5.844→4.878秒（1.198×）；初始化构造成本单独记录，完整独立验证与工作计费保留。CPU工作区285项、GPU默认测试154项通过，独立审查通过；H200作业1159已单次提交，尚未验收。详见[双语实验说明](docs/FULL_DIVISOR_CACHE.md)。

[English](README.en.md) | 中文

基于Rust、arkworks BN254与CUDA/ICICLE的Dynaverse增量证明研究原型。项目保留CPU reference，并通过薄Rust接口调用GPU算子，研究稀疏更新、分阶段重建和缓存复用的正确性与性能。

当前协议是**未加掩码的正确性研究原型**，不提供完整零知识保证，也不是经过生产安全审计的证明系统。

## 当前实现

- Rust协议层：Dynaverse、Dynamo、KZG、consistency、SRS及公开验证。
- CPU reference与GPU逐算子/逐证明对照；公开Reference验证路径保留。
- GPU算子：G1/G2 MSM、Fr NTT、resident变换、固定基setup与G2验证。
- 增量优化：稀疏索引MSM、foreground linear、冻结线性/G2复用、support geometry、未变缓存复用。
- 重建优化：row batch、受限transform packet、Product缓存验证、prepared roots和读回合并。
- 外部实验入口按规模选择组合，有界G1解码与设备affine读回均通过显式配置启用。

协议及数据结构保持Rust；性能敏感算子复用ICICLE或由CUDA C++实现。[架构约束](AGENTS.md)规定了Rust/CUDA边界及CPU reference要求。

## 验证状态

初始公开快照的自适应组合在RTX 5070 Ti Laptop（sm_120）上完成n=256/1024对照。每档4次大批update，前2次预热、后2次计时；每步检查Reference验证、proof/witness一致性及work allowance。

| n | 窗口 | baseline两笔update合计/s | 优化组合两笔update合计/s | baseline/优化组合 |
|---:|---:|---:|---:|---:|
| 256 | 6 | 16.156810 | 1.711010 | 9.443 |
| 1024 | 8 | 37.424205 | 5.249226 | 7.129 |

baseline与优化组合使用相同窗口和输入。时间排除读回校准及外部Reference验证；比值描述整个组合，不能归因于单个开关。这是少量本地样本，**不能外推为H200收益、稳定尾延迟或端到端加速**。

n=256完整组合的geometry复用命中11次，独立geometry审计命中5次，独立linear审计通过。详见[验证边界](docs/VALIDATION.md)、[原始事件](experiments/adaptive-fullopt/fixtures/n256.log)和[初始结构化结果](docs/results/local-2026-10-22.json)。

2026-10-22追加测试已通过全部默认GPU门禁、13项Python回归及n=1024新对照；新样本baseline/优化组合比值为7.523。**H200作业1157已完成n=256/65536矩阵，终态COMPLETED/0:0，原始事件和完整性验收通过**。

| H200 n | 窗口 | baseline两笔update合计/s | 优化组合两笔update合计/s | baseline/优化组合 |
|---:|---:|---:|---:|---:|
| 256 | 6 | 17.553499 | 2.116730 | 8.293 |
| 65536 | 16 | 1254.203179 | 379.355222 | 3.306 |

H200每档同样只有两笔计时update，排除校准及外部Reference；这些比值不代表端到端加速或统计显著性。821次进程采样未发现OOM，主机VmHWM约1.30GiB、显存采样最大约2.35GiB，不能当作认证峰值。[双语追加测试说明](docs/SUPPLEMENTAL_TESTS.md)和[H200原始事件](experiments/adaptive-fullopt/fixtures/h200-20261022.log)记录了真实命中、测试来源与限制。

## 按规模选择策略

独立实验分支正在扫描prepared缓存容量，保留本页R5策略不变。相对64根全优化对照，本地n=256扩大到256根的两笔update样本比值为1.149，n=1024扩大到1024根为1.348；H200 n=65536扫描已验收通过，64→4096根的两笔update合计380.454→373.280秒，样本比值1.019；大规模收益有限。详见[双语实验说明](docs/PREPARED_CACHE_TUNING.md)，这些比值不能称作端到端加速。

策略位于[`Strategy::for_size`](experiments/adaptive-fullopt/strategy.rs)，在创建session前固定，不改变协议默认值。

| n | window/H | 变换 | prepared/readback roots | undo |
|---|---|---|---|---|
| 64–512 | 6/3 | 长度≤4096使用packet | 64 | 64MiB |
| 1024–4096 | 8/4 | packet与分块结合 | 64 | 64MiB |
| 8192–65536 | 16/8 | 分块，row batch16 | 64 | 按1472n向上取2次幂，最低64MiB |

65536对应128MiB undo；每owner限额为2GiB。它不是整个进程的内存上界。

优化组合启用有界G1解码。设备affine读回在n<1024时保持关闭；更大规模先用独立owner比较64点host/device读回，仅当device中位数至少快3%才启用。该校准会与独立CPU结果逐点比较，且不计入update时间。阈值仍是实验策略，尚未证明在每种硬件和更新形状下最优。

## 快速开始：CPU

需要Git及固定的Rust 1.89.0。仓库的`rust-toolchain.toml`会选择该版本；CPU路径不需要CUDA。

```bash
git clone https://github.com/XIAZIHAN001/ZKP-Dynaverse.git
cd ZKP-Dynaverse
cargo test --locked --workspace --release -- --test-threads=1
```

CPU基准示例：

```bash
mkdir -p results
cargo run --locked -p dynaverse-bench --bin cpu_dynaverse --release -- \
  --n 256 --updates 8 --policies dense,precompute \
  --workloads hotspot,spreading --warmup 1 --samples 3 \
  --output results/cpu_dynaverse_n256.csv
```

benchmark会把Git commit及源码标识写入输出，因此请从Git checkout构建，而不是从不含`.git`的源码ZIP构建。

## GPU构建与实验

GPU路径支持Linux/WSL2，需要NVIDIA驱动、CUDA、CMake和按GPU架构构建的ICICLE v2.8.0。local为sm_120，H200为sm_90；两种架构的编译产物分别构建。

完整步骤见[中文构建指南](docs/BUILD.zh-CN.md) / [English build guide](docs/BUILD.md)。构建native bridge并设置库路径后，在开发容器内运行：

```bash
# 小规模完整组合
cargo run --locked --release \
  --manifest-path experiments/adaptive-fullopt/Cargo.toml -- fullopt-small

# 中规模完整组合
cargo run --locked --release \
  --manifest-path experiments/adaptive-fullopt/Cargo.toml -- fullopt-medium

# n=256/65536完整矩阵：需要足够显存和运行时间
cargo run --locked --release \
  --manifest-path experiments/adaptive-fullopt/Cargo.toml -- fullopt
```

事件以`ROW {json}`输出；成功标记仅在整个选定矩阵通过后打印。分析器检查配对样本、真实命中、资源计划和校准选择，接受规则的负向回归可独立运行：

```bash
python3 -B -m unittest discover \
  -s experiments/adaptive-fullopt -p 'test_*.py' -v
cargo test --locked --release \
  --manifest-path experiments/adaptive-fullopt/Cargo.toml --bin fullopt_probe
```

## 目录

```text
crates/protocol/              Rust协议与CPU reference
crates/gpu/                   GPU wrapper、codec与正确性测试
crates/bench/                 CPU/GPU基准入口
native/                       CUDA C++桥接与resident算子
experiments/adaptive-fullopt/ 按规模配置、校准、探针及接受规则
env/                          开发镜像与固定版本
docs/                         构建、验证边界与后续方向
```

公开快照保留核心Rust/CUDA源码，省略机器专用SSH/Slurm部署脚本、作业账本、构建缓存和服务器日志。[源码快照清单](docs/SOURCE_SNAPSHOT.json)记录了发布来源。

## 后续方向

65536规模H200验收已通过；下一步增加独立重复样本和更新形状，检验窗口与读回策略，再评估按实际更新支持集和总内存占用调整策略。见[路线图](docs/ROADMAP.md)。

依赖固定版本见[`env/versions.lock`](env/versions.lock)，第三方项目见[依赖说明](THIRD_PARTY_NOTICES.md)。本仓库暂未指定许可证。
