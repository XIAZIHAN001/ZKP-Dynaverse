# 追加测试 / Supplemental tests

状态截至北京时间2026-10-22 22:27；完整时间戳、命令与哈希见[结构化结果](results/supplemental-2026-10-22.json)。

Status as of 2026-10-22 22:27 Asia/Shanghai. Exact timestamps, commands, and hashes are in the [structured results](results/supplemental-2026-10-22.json).

## 已完成 / Completed

测试代码提交为`f55243b5f88b3449da44e485766e3f376c12e836`。本次验证直接使用公开checkout；198个核心文件与源码清单一致。GPU运行链接已有的对应架构native库，库哈希在结果中记录，本次没有重新构建native库。

The tested code commit is `f55243b5f88b3449da44e485766e3f376c12e836`. Tests used the public checkout; all 198 core files matched the source manifest. GPU runs linked the existing architecture-specific native library, whose hash is recorded. This supplement did not rebuild that library.

| 检查 / Gate | 结果 / Result |
|---|---|
| `dynaverse-gpu --features icicle`全部默认测试 / All default tests | PASS，528.28 s |
| 公开Python验收回归 / Public Python acceptance regressions | 13 passed |
| n=1024完整组合 / Complete adaptive bundle | PASS |
| 同一提交的CPU工作区 / CPU workspace on the same commit | 之前已PASS，本次未重复 / Previously PASS; not rerun here |

默认GPU门禁涵盖曲线/域运算、CPU/GPU证明对照、缓存、回滚、资源拒绝和索引MSM等。`ignored`故障注入、profiling及规模专项仍需显式运行，不能计为本次已完成。

The default GPU gate covers curve/field operations, CPU/GPU proof comparisons, caches, rollback, quota rejection, and indexed MSM. Ignored failure-injection, profiling, and scale experiments require explicit execution and are not included in this pass.

## n=1024新样本 / New n=1024 sample

硬件为RTX 5070 Ti Laptop（sm_120）。两路使用相同窗口8和输入，四次成对更新，前两次预热，后两次计时。每步检查Reference、proof/witness一致性和work allowance。

The RTX 5070 Ti Laptop GPU (sm_120) ran four paired updates with the same window of eight and identical inputs: two warmups and two measured updates. Every step checked Reference verification, proof/witness equality, and work allowance.

| n | baseline两次合计 / Baseline total, s | 优化两次合计 / Bundle total, s | baseline / bundle |
|---:|---:|---:|---:|
| 1024 | 37.846863 | 5.031059 | 7.523 |

实际命中frozen capture、foreground linear、global G2和cache reuse；48次packet调用，GPU sparse G1累计1600点，CPU sparse回退0点。计数包含预热。geometry和独立linear复用不属于本档负载的覆盖结论。

The run recorded frozen capture, foreground linear, global G2, and cache reuse; 48 packet calls; 1,600 GPU sparse G1 points; and zero CPU sparse fallback points. Counts include warmups. This size does not establish geometry reuse or independent linear-reuse coverage.

该比值只描述两次计时update，不含校准和外部Reference，不代表端到端加速、统计显著性或H200收益。[原始事件](../experiments/adaptive-fullopt/fixtures/n1024-supplement-20261022.log)保留实际JSON事件和最终成功标记。

The ratio describes only two timed updates and excludes calibration and external Reference verification. It does not establish end-to-end speedup, statistical significance, or H200 gains. [Raw events](../experiments/adaptive-fullopt/fixtures/n1024-supplement-20261022.log) retain the actual JSON events and final success marker.

## H200进行中 / H200 in progress

作业1157正在执行n=256/65536完整矩阵，尚未取得最终验收。n=256序列和独立审计已完成；n=65536已完成setup和读回校准，设备affine读回被选择，初始会话仍在构造。三个公开探针Rust文件与远端已封包输入逐字节一致。

Job 1157 is running the complete n=256/65536 matrix; final acceptance is pending. The n=256 sequence and independent audits completed. n=65536 completed setup and readback calibration, selected device affine readback, and is constructing its initial sessions. The three public Rust probe files exactly match the sealed remote inputs.

截至上述快照，未观察到OOM或校验失败；观测到的主机VmHWM约1.27 GiB，进程显存采样最大值约2.35 GiB。采样周期五秒，显存最大值是下界；这些值不是整个运行的认证峰值。2 GiB仍为单owner配额。

No OOM or verification failure was observed as of this snapshot. Observed host VmHWM was about 1.27 GiB; the largest sampled process GPU allocation was about 2.35 GiB. GPU observations are five-second samples and lower bounds, not certified run-wide peaks. The 2 GiB quota remains per owner.

只有终态COMPLETED/0:0、原始事件验收及完整性复核均通过后，才能把H200矩阵标为PASS。

The H200 matrix can be marked PASS only after a COMPLETED/0:0 terminal state, raw-event acceptance, and integrity verification.
