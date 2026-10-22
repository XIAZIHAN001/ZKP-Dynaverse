# Next steps / 后续方向

The current snapshot is suitable for sharing the implementation and reproducing local experiments. Further optimization remains possible.

当前快照适合分享实现和复现本机实验，仍保留以下改进方向。

1. **H200 scale acceptance.** Run the adaptive n=65536 matrix, retain terminal status and per-process telemetry, and check correctness before interpreting speedups.
   **H200规模验收。** 完成65536矩阵，保存终态和进程遥测，先确认正确性再解释收益。
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
