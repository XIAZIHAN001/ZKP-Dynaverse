# Next steps / 后续方向

The full-domain affine-divisor cache is now implemented as a default-off experiment. Local n=256/1024 CPU/GPU pairs passed; full regression/review and H200 n=65536 remain pending. 轻量全域 divisor 缓存已实现，本地配对通过，完整门禁、审查与 H200 待完成。见 [report](FULL_DIVISOR_CACHE.md).

The current snapshot is suitable for sharing the implementation and reproducing local experiments. Further optimization remains possible.

当前快照适合分享实现和复现本机实验，仍保留以下改进方向。

1. **Extend H200 evidence.** The adaptive n=256/65536 matrix passed in job 1157; see the [report](SUPPLEMENTAL_TESTS.md). Add independent repetitions. Admit additional scales only after checking setup domains and resource plans.
   **扩充H200证据。** 作业1157已通过n=256/65536矩阵，见上述报告。继续增加独立重复；新规模先核对setup域与资源计划，再运行。
2. **More representative timing.** Add repeated runs and update shapes, including hotspot, changing support, and both sides of the rebuild threshold. Compare alternative windows on the same hardware.
   **更有代表性的计时。** 增加重复次数和更新形状，覆盖热点、support变化及重建阈值两侧，在同一硬件比较窗口。
3. **Shape-aware strategy selection.** Extend the external policy with actual changed-wire counts, support stability, and cache eligibility. Verify whole-update cost rather than selecting solely by primitive timings.
   **按更新形状选择策略。** 引入实际变更数、support稳定性和缓存适用性，用完整update代价校准，避免只看原语计时。
4. **Aggregate resource accounting.** Account for multiple owners, CUDA contexts, journal metadata, and host-side validation alongside device quotas.
   **总资源核算。** 在单owner配额之外，核算多个owner、CUDA上下文、journal元数据与主机验证内存。
5. **Continuous CPU validation.** Add CI for the default CPU suite and acceptance regressions, while keeping GPU/H200 jobs explicit and tied to real hardware.
   **持续验证。** 增加CPU与验收规则CI，GPU/H200实验继续显式调度并绑定真实硬件。

These are follow-up experiments, not performance promises or a claim that every existing option should be enabled for every input.

这些是后续实验方向，不代表性能承诺，也不意味着应对所有输入无条件开启全部选项。
