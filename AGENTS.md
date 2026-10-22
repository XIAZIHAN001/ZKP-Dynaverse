# 语言与架构约束

以下约束适用于本仓库后续开发与代码审查。

1. 不要将现有 Rust 协议代码为了语言统一而重写为 C++。`dynaverse`、`dynamo`、`kzg`、`consistency`、`srs`、`types` 等协议与数据结构层继续保持 Rust。
2. Rust 侧的 `OperatorBackend` 及 GPU backend 仅作为薄封装层，负责参数检查、生命周期/资源管理、数据转换和调用底层算子，避免在 Rust wrapper 中实现复杂的 GPU 算法。
3. 如果 ICICLE 已有相应 GPU 实现，优先复用 ICICLE，不重复实现。
4. 如果后续必须自行实现或深度优化 GPU kernel，优先使用 CUDA C++，尤其是 MSM、NTT、batch MSM、内存布局、warp/shared-memory 优化等性能敏感部分。
5. Rust 与 CUDA C++ 之间保持清晰、最小化的接口边界；不要无必要引入复杂 FFI 层。
6. 新增 Rust 代码优先使用可读的显式写法。避免为了简短而堆叠多层 iterator、closure、泛型链和复杂生命周期。如果简单 `for` 循环更容易维护，就优先使用简单写法。
7. 对新增的非显然 Rust 语法写注释，特别是 ownership、borrowing、lifetime、iterator chain、trait dispatch 和 `unsafe`。
8. CPU reference backend 必须保留，用于与 GPU backend 做逐算子正确性对照。
9. 不得仅为了代码风格或语言偏好重构已经稳定工作的模块。优化应集中在明确的性能瓶颈。
10. 每次新增 GPU 实现时说明：上层调用位置、Rust/CUDA 边界、输入输出数据布局、CPU reference 对应实现，以及验证 GPU 结果正确性的方法。

## 新增 GPU 实现的说明模板

- **上层调用位置：** 模块、函数及协议阶段。
- **Rust/CUDA 边界：** Rust wrapper 的职责、底层 ICICLE 或 CUDA C++ 入口，以及必要的资源所有权与生命周期约定。
- **输入输出数据布局：** 域元素/曲线点的编码、数组或批次布局、索引与长度，以及主机/设备内存和传输方式。
- **CPU reference：** 对应的 CPU 实现位置。
- **正确性验证：** CPU/GPU 对照方法、覆盖的输入规模与边界情况，以及实际执行结果。
