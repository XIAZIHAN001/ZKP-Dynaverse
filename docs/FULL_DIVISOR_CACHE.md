# 轻量全域 G2 divisor 缓存 / Full-domain public G2 divisor cache

本实验分支在已验收的 prepared 扫描分支上，新增默认关闭的 `IncrementalConfig.rebuild_divisor_cache`。只保存公开根与 affine divisor；仅支持 Product 模式及 n≤65536。初始化按64根分块构造全表，初始检查复用该表；update 仍优先使用原 prepared64，其他根从全表读取。每根独立方程、Miller、final exponentiation、Reference 和最终 CPU dense oracle 均保留。

This experimental branch adds opt-in `IncrementalConfig.rebuild_divisor_cache` on top of the accepted prepared-cache branch. It retains public roots and affine divisors only, for Product mode and n≤65536. Initialization constructs chunks of64 and reuses the table in initial checks. Updates keep prepared64 first, then use public affine lookups. Independent equations, Miller loops, final exponentiations, Reference checks and the final CPU dense oracle remain intact.

## 本地样本 / Local samples

CPU 与 RTX5070 Ti Laptop GPU 使用相同确定性输入与 R5 策略，仅改变全域 divisor 开关。每路4笔大批更新，前2笔预热、后2笔计时，顺序交错；每步要求完整付费重建完成、proof/witness 相等及各自工作配额通过。初始化包含缓存构造和初始检查，外部 Reference 和 dense oracle 单独计时。

CPU and RTX5070 Ti Laptop GPU use identical deterministic inputs and R5 policy, changing only this cache flag. Each route has two warmups and two measured large updates in counterbalanced order. Every call must complete its paid rebuild, match proof/witness, and satisfy its own work allowance. Initialization includes construction and initial checks; external verification and the dense oracle have separate clocks.

| Backend | n | Off: two updates/s | On: two updates/s | Off/on | On initialization/s | On precompute/s | Phase-sum ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| cpu | 256 | 2.230537 | 2.057728 | 1.083980 | 1.656060 | 0.054452 | 1.039288 |
| cpu | 1024 | 9.503292 | 8.535913 | 1.113330 | 7.122545 | 0.217190 | 1.043142 |
| gpu | 256 | 2.170937 | 1.993899 | 1.088790 | 2.970167 | 0.053232 | 1.092676 |
| gpu | 1024 | 5.843537 | 4.878391 | 1.197841 | 6.424695 | 0.211220 | 1.103086 |

Off/on 只描述本次两笔 update 样本。Phase-sum 是共享 setup + 各路初始化 + 全部4笔 update 的算术和，排除外部验证、校准与程序其他开销，**不是独立测得的端到端墙钟**。不能宣称统计显著性或稳定尾延迟。H200 n=65536 已单次提交作业1159，最近调度快照为RUNNING；本机结果不能代替其验收。

Off/on describes only the two measured update samples. Phase-sum adds shared setup, route initialization and all four updates; it excludes external verification, calibration and other process overhead. It is **not an independently measured end-to-end wall time**. No statistical significance or stable tail latency is established. H200 n=65536 was submitted once as job1159; the latest scheduler snapshot isRUNNING. Local results cannot substitute for H200 acceptance.

## 资源和正确性 / Resources and correctness

Rust1.89 x86_64 下，每项实际类型大小168字节，表头368字节。n=65536 计划常驻11,010,416字节（约10.50MiB）；含有界构造暂存的新增主机 envelope 为11,145,584字节（约10.63MiB），完整单 owner 上界1,358,200,944字节。以上是类型/容量与计划核算，不是 RSS/显存峰值；2GiB 是每 owner 配额。H200 对照会同时保留两个 owner，undo 为每 owner128MiB。

With pinned Rust1.89 x86_64, each entry occupies168 bytes and the header368. At n=65536, planned retained storage is11,010,416 bytes (about10.50MiB); the additional host envelope including bounded construction scratch is11,145,584 bytes (about10.63MiB). The full per-owner bound is1,358,200,944 bytes. These are type/capacity and planning figures, not measured RSS/VRAM peaks. The2GiB quota applies per owner; the H200 comparison retains two owners, each with128MiB undo.

实际计数要求构造 n 个 divisor，初始检查 formed=0/hits=n/equations=3n；每笔完整 update formed=0/hits=2(n−64)，equations=Miller=final exponentiations=2n，terms=4n。关闭时回退构造2(n−64)个 divisor。旧算术 work 证书不下调，额外查表/拷贝计费。表绑定 owner/SRS/domain，不能缓存验证通过结果。

Acceptance requires construction=n, initial formed=0/hits=n/equations=3n; each complete update has formed=0/hits=2(n−64), equations=Miller=final exponentiations=2n and terms=4n. The off route forms2(n−64) fallback divisors. Existing arithmetic certificates stay conservative; extra lookups/copies are charged. The table binds to owner/SRS/domain and retains no validation decisions.

协议测试覆盖边界、常量、多项式/承诺/opening 破坏、抵消错误、迟发缓存错误的整 Session 回滚及精确重试，以及跨 pending epoch 的复用。原三份 R5 Rust 探针和全部 native/GPU wrapper 保持原字节；修改后的协议核心在 `SOURCE_SNAPSHOT.json` 中单独追踪，不能沿用旧产品身份。

Protocol tests cover boundaries, constants, malformed values/commitments/openings, cancelling errors, late cache failure with whole-session rollback and exact retry, and reuse across pending epochs. The three original R5 Rust probes and all native/GPU wrapper bytes remain unchanged. Changed protocol sources have explicit provenance in `SOURCE_SNAPSHOT.json` and must receive a new product identity.

## 复现 / Reproduction

本探针为独立 Cargo workspace，复用原 R5 策略与校准代码。CPU 默认构建不链接 CUDA driver；GPU 显式开启 feature。离线运行需事先准备锁定依赖；首次在线复现可去掉 `--offline`。

The probe is a separate Cargo workspace reusing original R5 policy/calibration. CPU defaults do not link the CUDA driver; GPU builds explicitly enable the feature. Offline execution needs prefetched locked dependencies; omit `--offline` for first online preparation.

```bash
python -m unittest discover -s experiments/full-divisor-cache -p 'test_*.py'
cargo run --quiet --offline --locked --release --manifest-path experiments/full-divisor-cache/Cargo.toml -- cpu 1024
# After the native/GPU build described in BUILD.md:
cargo run --quiet --offline --locked --release --manifest-path experiments/full-divisor-cache/Cargo.toml --features gpu -- gpu 1024
```

原始事件位于 `experiments/full-divisor-cache/fixtures/`，仅省略构建横幅与机器路径。派生矩阵见 [JSON](results/full-divisor-cache-2026-10-23.json)，可用 `analysis.analyze` 从原始事件重新计算。CPU 工作区285项通过/13项忽略，GPU默认测试154项通过/58项忽略，均0失败；独立审查无Critical/Important，文档命令已修正。H200结果待完成验收后追加，未完成项不得视为通过。当前协议仍未实现完整零知识保证。

Raw events are in `experiments/full-divisor-cache/fixtures/`, omitting only build banners and machine paths. The [JSON](results/full-divisor-cache-2026-10-23.json) is recomputable with `analysis.analyze`. The CPU workspace passed285 tests/13 ignored; the default GPU suite passed154/58 ignored, with zero failures. Independent review found no Critical/Important issues; documentation commands were corrected. H200 outcomes await acceptance; unfinished gates are not passes. The protocol does not establish complete zero knowledge.
