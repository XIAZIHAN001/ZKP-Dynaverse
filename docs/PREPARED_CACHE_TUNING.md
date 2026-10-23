# Prepared 缓存容量实验 / Prepared-cache capacity experiment

## 中文

本实验以公开快照 `08b1ee6` 的**全优化组合、64 根 prepared 缓存**为对照。只改变 prepared 根容量，候选为64、256、1024、4096，实际根数为 `min(n, cap)`；实际容量相同的档位只跑一次。协议核心、native 算法和三个 R5 Rust 探针文件保持逐字节一致；原入口和默认策略继续使用64根。

固定项包括按规模选择的window、readback64、独立owner的64点affine校准、row batch16、Product逐方程校验及既有复用开关。每条路径执行4次完整update，前2次预热、后2次实测；两种更新族各改变16条wire，沿用R5固定随机种子。每种更新族在预热和实测间交换路径顺序，两笔实测也交换首尾。

### 本地结果（sm_120）

| n | 实际 prepared 根数 | 两笔实测 update 合计 / s | 64根对照 / 本档 | 请求保留的prepared内存 / MiB |
|---:|---:|---:|---:|---:|
|256|64|1.826540|1.000|1.536|
|256|256|1.589572|1.149|6.073|
|1024|64|5.210872|1.000|1.536|
|1024|256|4.791173|1.088|6.073|
|1024|1024|3.865116|1.348|24.219|

这是本次两笔实测update的比值，不代表统计显著性、稳定尾延迟或端到端加速。预计算仍计入session初始化，不能作为免费工作扣除。完整原始事件保留了初始化、预计算、外部Reference、最终CPU稠密锚点和分项耗时；不能把内层cache/linear计时再次加到外层rebuild中。

扩大前缀实际减少了重复divisor构造与G2 prepare。每次update仍完成 `2*n` 个独立Product方程、相同数目的Miller调用和最终幂，未合并不同方程。完整前缀的divisor与现场prepare构建数为0；前缀外每4根的fallback校验器仍需准备一个generator。

每条路径检查完整付费重建完成（swapped、无pending）、工作配额、逐步proof/witness一致、独立Reference和最终CPU稠密锚点，确认已有复用机制命中且ForceGpu未回落到CPU sparse路径。新探针与R5入口共10项Rust测试、26项Python回归通过；Python反例覆盖缺行、错误终态、错误证明标志、嵌套计时、配额及资源篡改。首轮新探针把fallback generator漏算进prepare次数，修正了测试断言；产品代码未修改。独立审查发现的日志字段覆盖及验收缺口已修复并补反例。

### H200及资源边界

H200 n=65536四档扫描已完成并通过原始事件、完整重建、Reference/proof/witness、最终CPU稠密锚点、配额和资源验收，终态COMPLETED/0:0。两笔实测update合计如下：

| prepared根数 | 两笔update / s | 64根对照 / 本档 |
|---:|---:|---:|
|64|380.453942|1.000000|
|256|380.092028|1.000952|
|1024|378.743913|1.004515|
|4096|373.279908|1.019219|

4096根每两笔样本共节省7.174034秒，时间减少约1.886%；额外prepared请求保留内存约95.269MiB，预计算从0.026768秒增至1.777054秒，仍计入初始化。最大前缀只覆盖6.25%根，未减少Miller/最终幂次数，因此大规模收益有限。两笔样本不足以断言统计显著性；R5默认64根保持不变，更大容量作为可选实验配置。

2GiB为每owner配额，四条比较路径同时保留多个owner，进程总RSS或显存可以超过2GiB。最大档的资源计划上界为1,613,360,896字节/owner，其中prepared host envelope为271,060,992字节；undo128MiB覆盖92MiB事务界。这些是计划界，不是RSS或显存测量值。远端取得820次五秒进程采样，VmHWM观测最大2,175,139,840字节（约2.03GiB），显存采样最大4,972,347,392字节（约4.63GiB），采样错误0、未发现OOM。进程合计包含四个owner，不等于单owner用量；这些观测不是认证峰值。未新增完整零知识或生产安全保证。

## English

This experiment compares against the **fully optimized 64-root configuration** at public snapshot `08b1ee6`. Only the prepared-prefix capacity changes: 64, 256, 1024, 4096, bounded by `min(n, cap)` and deduplicated by actual capacity. Core/native code and the three R5 Rust probe files remain byte-identical. The original entry point and default strategy retain 64 roots.

The size-selected window, 64-root readback, separate-owner 64-point affine calibration, row batch 16, independent Product equations and existing reuse options stay fixed. Each route completes four updates: two warmups and two measured samples. Both 16-wire edit families retain R5 seeds. Route order reverses across families and between warmup and measurement for each family.

The local sm_120 table above reports the sum of the **two measured updates**. At n=256, 64→256 roots changes 1.826540 s to 1.589572 s (1.149×); at n=1024, 64→1024 roots changes 5.210872 s to 3.865116 s (1.348×). These are sample ratios, not statistical significance, tail guarantees or measured end-to-end speedups. Precomputation remains part of initialization. Raw events include initialization, precompute, external Reference checks, final CPU dense anchor and timing details. Cache/linear details nest inside rebuild scopes and must not be added twice.

Actual divisor/G2 preparation work decreases while every update retains 2*n independent Product equations, Miller calls and final exponentiations. Each route must complete its paid rebuild, stay within work/resource limits, match proof/witness after each step, pass Reference, and match the final CPU dense anchor. Existing optimization hits and GPU sparse dispatch are checked. Ten Rust tests and 26 Python regressions pass; repaired harness assertions and independent-review findings have regression coverage. Product code is unchanged.

The H200 n=65536 four-capacity scan passed terminal, raw-event, complete rebuild, Reference/proof/witness, final CPU dense-anchor, quota and resource acceptance. The table above reports two measured update sums: 380.453942 s at 64 roots versus 373.279908 s at 4096 roots (1.019219×), a 7.174034 s sample saving or 1.886% less time. The larger table retains about 95.269 MiB extra requested memory and increases precomputation from 0.026768 s to 1.777054 s, charged to initialization. Two samples do not establish statistical significance. R5 keeps its 64-root default; larger prefixes remain optional experiments. A 4096-root prefix covers only 6.25% of that domain, so local gains cannot establish H200 gains. The 2 GiB admission is per owner, not aggregate process memory; the comparison retains several owners. At 4096 roots, the planned per-owner upper bound is 1,613,360,896 bytes, including 271,060,992 bytes for the prepared host envelope. The 128 MiB undo arena covers the 92 MiB transaction bound. Planned bounds are not measured RSS/VRAM peaks. The 820 five-second process samples report an observed VmHWM maximum of 2,175,139,840 bytes (about 2.03 GiB) and GPU sampled maximum of 4,972,347,392 bytes (about 4.63 GiB), with no sampling errors or observed OOM. These aggregate four-owner observations are not certified peaks. The prototype provides no complete zero-knowledge or production-security guarantee.

## 复现 / Reproduction

Use the pinned Rust/CUDA/native environment described in [BUILD](BUILD.md):

```bash
cargo test --locked --release --manifest-path experiments/adaptive-fullopt/Cargo.toml --bins
python -B -m unittest discover -s experiments/adaptive-fullopt -p 'test_*.py' -v
cargo run --locked --release --manifest-path experiments/adaptive-fullopt/Cargo.toml \
  --bin prepared_cache_probe -- 1024 scan
python -B experiments/adaptive-fullopt/prepared_analysis.py results.log
```

Save the probe stdout as `results.log` before running the analyzer. Use `256 scan` for the small matrix and `65536 scan` for the admitted H200 matrix. Build time, calibration and independent external oracles are outside the two-update timing. The scan does not change protocol defaults.

[Structured results](results/prepared-cache-tuning-2026-10-23.json) · [n256 raw events](../experiments/adaptive-fullopt/fixtures/prepared-n256.log) · [n1024 raw events](../experiments/adaptive-fullopt/fixtures/prepared-n1024.log)

[H200原始事件 / H200 raw events](../experiments/adaptive-fullopt/fixtures/prepared-h200-20261023.log)
