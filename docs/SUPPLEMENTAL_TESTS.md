# 追加测试 / Supplemental tests

测试日期为2026-10-22（北京时间）；完整时间戳、命令与哈希见[结构化结果](results/supplemental-2026-10-22.json)。

Test date: 2026-10-22 Asia/Shanghai. Exact timestamps, commands, and hashes are in the [structured results](results/supplemental-2026-10-22.json).

## 已完成 / Completed

公开测试代码提交为`f55243b5f88b3449da44e485766e3f376c12e836`。本地追加测试直接使用公开checkout，198个核心文件与源码清单一致；H200执行封包副本，已核对核心产品身份及三个探针文件与公开源对应。GPU运行链接已有的对应架构native库，库哈希在结果中记录，本次没有重新构建native库。

The public tested code commit is `f55243b5f88b3449da44e485766e3f376c12e836`. Local supplemental tests used the public checkout; all 198 core files matched the source manifest. H200 executed a sealed deployment, with its core product identity and three probe files checked against the public source. GPU runs linked the existing architecture-specific native library, whose hash is recorded. This supplement did not rebuild that library.

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

## H200验收通过 / H200 acceptance passed

作业1157于北京时间2026-10-22 23:21:05结束，耗时1小时9分5秒，终态COMPLETED/0:0。n=256/65536完整矩阵及唯一最终成功标记通过；32份远端证据、237个源码文件、支持输入、原生库和探针二进制均完成完整性核对。产品210文件保持封存身份，三个公开Rust探针文件与远端输入相同。

Job 1157 completed at 23:21:05 Asia/Shanghai on 2026-10-22 after 1:09:05, with terminal state COMPLETED/0:0. The complete n=256/65536 matrix and unique final success marker passed. Integrity checks verified 32 remote evidence files, 237 source files, support inputs, native libraries, and the probe binary. The 210 product files retained their sealed identity; the three public Rust probe files match the remote inputs.

每档四次成对更新，前两次预热、后两次计时。下表为后两次update的合计，排除读回校准和外部Reference验证；两路窗口及输入相同。

Each size used four paired updates: two warmups and two timed updates. The table totals the two timed updates, excluding readback calibration and external Reference verification. Both routes used identical windows and inputs.

| H200 n | Window | Baseline total, s | Bundle total, s | Baseline / bundle |
|---:|---:|---:|---:|---:|
| 256 | 6 | 17.553499 | 2.116730 | 8.293 |
| 65536 | 16 | 1254.203179 | 379.355222 | 3.306 |

| n | Frozen capture | Foreground linear | Global G2 | Cache reuse | Geometry hits | Packet calls | GPU sparse G1 points | CPU sparse fallback |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 4 | 16 | 4 | 4 | 11 | 48 | 2341 | 0 |
| 65536 | 4 | 4 | 4 | 4 | 0 | 0 | 1600 | 0 |

计数包含预热及n=256额外功能序列。n=256独立geometry审计命中5次，独立linear审计通过。n=65536采用分块row batch16，packet为0是规模策略的预期结果；该大批负载也不宣称geometry复用。设备affine读回在n=256关闭，在n=65536经真实64点校准后启用；两档均使用有界G1解码。

Counts include warmups and the extra n=256 feature sequence. The separate n=256 geometry audit recorded five hits, and the independent linear audit passed. n=65536 used tiled transforms with row batch 16; zero packet calls are expected by policy. Its large-update workload does not establish geometry reuse. Device affine readback remained off at n=256 and was selected after actual 64-point calibration at n=65536. Both sizes used bounded G1 decoding.

本次没有OOM或校验失败。821次五秒进程采样中，主机VmHWM为1,396,609,024字节（约1.30 GiB），显存采样最大值2,524,971,008字节（约2.35 GiB），采样器无错误。显存采样最大值是下界，VmHWM为观测到的内核RSS高水位；它们不是全进程所有资源的认证峰值。2 GiB限额属于单owner。n=65536事务需求模型为92 MiB，undo配置128 MiB，资源计划及实际运行均通过。

No OOM or verification failure occurred. Across 821 five-second process samples, observed host VmHWM was 1,396,609,024 bytes (about 1.30 GiB), and the largest GPU sample was 2,524,971,008 bytes (about 2.35 GiB), with no sampler errors. GPU sample maxima are lower bounds; VmHWM is the observed kernel RSS high-water mark. Neither certifies aggregate resource peaks. The 2 GiB quota is per owner. At n=65536, the transaction model required 92 MiB and the configured undo arena was 128 MiB; resource checks and execution passed.

每档只有两笔计时样本，不能据此宣称统计显著性、稳定尾延迟、每个开关的单项贡献或端到端加速。完整默认GPU测试是本地独立门禁，H200本批验收限定于上述矩阵，不包括所有ignored实验。

Each size has only two timed samples. These do not establish statistical significance, stable tail latency, individual option contributions, or end-to-end speedup. The complete default GPU suite was a separate local gate; H200 acceptance covers this matrix, not every ignored experiment.

[H200原始事件 / H200 raw events](../experiments/adaptive-fullopt/fixtures/h200-20261022.log) | [结构化结果 / Structured results](results/supplemental-2026-10-22.json)
