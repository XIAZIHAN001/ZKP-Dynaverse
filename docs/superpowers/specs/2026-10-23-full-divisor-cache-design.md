# 轻量全域 G2 divisor 缓存：设计草案

状态：用户已选定方向，设计待确认；产品代码尚未修改。

## 目标与依据

1158的n=65536两笔64根对照update共380.454秒，divisor构造86.394秒，
缓存验证362.784秒。只缓存4096根prepared系数的样本收益为1.019×。
本轮缓存全部根对应的公开G2 divisor，减少重复标量乘；不减少独立校验方程。
约23%是该阶段在旧样本中的占比，不是本轮承诺收益。

## 方向选择

- CPU并行校验：覆盖面大，但增加线程调度、栈内存和并行计时/故障处理。
- 轻量全域divisor缓存：用户选择。保留更少的公开数据，以初始化成本换取重复update收益。
- 完整GPU pairing：现有native仍含检查探针，需要另行完成算术链，不进入本轮。

## 行为与边界

1. 在新的独立worktree/分支，以已验收提交518bfed为基线；发布main、PR1及
   1157/1158封存目录不改动。Rust协议层继续Rust，native/CUDA不改。
2. 新增默认关闭的初始化配置开关。开启只允许Product缓存验证，域大小不超过65536。
   关闭时保留旧行为和旧规模支持，不能因新上限拒绝旧的更大域。
3. 构造与owner、SRS两项G2幂、域大小和根顺序绑定的私有只读表；存root和
   affine divisor，不存每根prepared line coefficients，不存任何校验通过标志。
4. 固定以最多64根的小块构造/归一化，避免同时保留全域projective临时数组。
   根点由当前SRS与域计算；外来owner、SRS或根顺序不允许复用。
5. 初始化的精确缓存校验也读取该表，避免先构造全表、再在初始化校验里重复构造。
   全表预计算计入session初始化；原prepared64表保持原构造流程，两条路径都有它。
6. 后续update优先保留原prepared64路径；前缀外取全域affine divisor，继续逐点
   Product校验。Miller、最终幂、proof/witness、Reference和CPU reference均保留。
7. 不改变GPU owner调度、readback64、affine校准64、row16、window或更新输入。
   不新增并行、GPU pairing、零知识或更大电路规模支持。

## 内存、工作预算和失败

- canonical G2点数据在65536根时为8MiB；Rust对象含标志、对齐和root元数据，
  不能把8MiB称为实际host占用。用size_of、Vec capacity和checked算术报告
  requested retained bytes；资源上界同时覆盖表、64根构造临时量和容器开销。
- 资源预检先于新表/后端分配；加入ResourcePlan明确字段并计入upper_bytes。
  每owner2GiB与undo128MiB规则不变，不用它们代表整个进程RSS/显存上界。
- 继续使用保守的重建预算，不靠减少逻辑校验数量获得收益；新lookup/复制显式计费，
  若W/Cmax/Q变化必须展示，两路各自满足配额。比较相同完整重建、方程数和发布端点。
- 只读表不保留witness或opening结果。任何校验失败走现有整次update回滚；
  proof、witness、receipt、timing和owner事务状态恢复；错误数据不得写成已校验缓存。
- 日志区分一次性构造数、初始化校验现场构造数、update现场构造数及缓存命中。
  不能把表的构造隐藏在setup外，也不能把嵌套计时重复求和。

## 验收与停止条件

1. 定向测试先失败后实现：开关/模式/域/预算边界，owner/SRS/root绑定；表值与
   原计算逐点一致；错误commitment/value/opening、非单位点损坏、可相消错误仍拒绝。
2. 对新路径注入在已处理首个缓存块之后的后续故障，核对完整回滚与精确retry。
3. 本地CPU/GPU n=256/1024，以prepared64全优化为两路共同基线，只切换全域表。
   每路两次预热、两次计时，交换顺序；每步Reference及proof/witness一致，最终CPU dense。
4. 按读回/根数模型检查真实命中、2n独立Product方程、Miller/最终幂次数不变；
   update的现场divisor构造应为0，初始化与一次性预计算分项完整。
5. 跑相关完整CPU/GPU回归及独立审查。只有本地正确性、预算与收益通过才单次提交
   H200 n=65536配对验证；若本地无收益/回归或资源不合规，保留负结果并停止升级。
6. H200失败收集证据，不自动重复提交。只有终态、原始配对、CPU oracle和资源审计
   都通过才更新实验文档。只报告两笔样本比值，另列含初始化的流程阶段和内存代价。

## 留给实现者的常规选择

文件拆分、新字段/计数器名称、测试夹具布局及独立分支命名由实现者按现有风格决定。
新核心修改需要新的SOURCE_SNAPSHOT与差异说明；旧发布快照和已封包源身份不能改写。
本轮不自动合并main，不恢复定时跟进。
