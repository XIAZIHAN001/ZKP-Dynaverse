# Validation boundaries / 验证边界

## Evidence included / 已包含的证据

The public fixtures contain unchanged JSON events and success markers from real local n=256/1024 runs on an RTX 5070 Ti Laptop GPU. Build banners and machine-specific command wrappers are omitted. `docs/results/local-2026-10-22.json` contains the derived matrices.

公开fixture保留了本机RTX 5070 Ti Laptop上n=256/1024实跑产生的原始JSON事件与成功标记，省略构建横幅和机器专用命令包装。结构化矩阵见`docs/results/local-2026-10-22.json`。

Each size executes four paired large updates: two warmups and two measured updates. The n=256 run additionally executes twelve paired same-support updates and separate geometry/linear audits. Every update checks Reference verification, proof/witness equality, and work allowance.

每档运行四次成对大批更新，两次预热、两次计时。n=256另有十二次成对同support更新，以及geometry/linear独立审计。每步均检查公开Reference、proof/witness一致性与work allowance。

## Acceptance / 接受规则

- Missing or duplicate paired samples, false correctness flags, mismatched receipt counts, CPU sparse fallback, and insufficient undo are rejected.
- n=256 requires actual geometry reuse in the full bundle and successful independent geometry/linear audits. Frozen capture can supersede linear rebuild reuse inside the combined route.
- Packet execution is required only for eligible size policies. n=65536 selects the tiled route.
- Readback calibration verifies CPU/GPU equality, the actual bounded decoder, and 96-byte versus 64-byte transfer boundaries.
- `validate_performance` recomputes results from raw events and rejects a changed or stale derived matrix.

上述规则分别检查配对完整性、真实执行计数、功能覆盖、资源计划、读回路径及原始日志和派生报告的一致性。配置为true本身不算执行命中。

## What is not established / 尚未证明

- The adaptive n=256/65536 H200 matrix is running as job 1157; final acceptance is pending. See the dated [supplemental report](SUPPLEMENTAL_TESTS.md).
- Two measured updates do not establish stable percentiles or statistical significance.
- Same-window baseline/full comparisons do not isolate individual options or prove the window thresholds optimal.
- Primitive readback calibration does not establish whole-update gains by itself.
- Per-owner resource estimates and sampled maxima are not process-wide certified peaks.
- The current protocol is unmasked; these experiments do not establish complete zero knowledge or production security.

本轮尚未完成65536规模H200验收，也不据本机小样本宣称尾延迟、单项贡献、全局最优窗口、整个进程的硬内存上界或完整零知识保证。

The public checkout additionally passed all default GPU tests and a fresh n=1024 comparison; 13 Python acceptance regressions passed again. This does not include ignored experiments. Supplemental evidence is separate from the initial fixtures.

公开checkout另已通过全部默认GPU测试和一次新的n=1024对照，13项Python验收回归再次通过；ignored专项不在其中。追加证据与初始fixture分别保存。

## Reproduction gates / 复现门禁

```bash
cargo test --locked --workspace --release -- --test-threads=1
python3 -B -m unittest discover -s experiments/adaptive-fullopt -p 'test_*.py' -v
```

GPU build and execution commands are in [BUILD.md](BUILD.md) / [BUILD.zh-CN.md](BUILD.zh-CN.md). Ignored profiling, fault-injection, and scale tests are separate experiments; a default suite pass does not mean those experiments ran.

GPU构建与运行见上述指南。被标记ignored的profiling、故障注入和规模测试需要单独执行，不能把普通测试通过写成它们也已完成。
