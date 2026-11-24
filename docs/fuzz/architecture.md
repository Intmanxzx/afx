# Fuzz 工作负载与执行架构

Fuzz 是 AFX 的自动泛化工作负载，生成逻辑和阶段队列位于 `autofusex/workloads/fuzz/`。公共客户端、执行设施、中心和 viewer 的职责见[平台架构](../architecture.md)。

## 组件与执行流

| 组件 | 职责 |
|---|---|
| `workloads/fuzz/generation`、`nnsmith` | E/F seed、采样、NNSmith/Z3 构图与 ATen 目录 |
| `workloads/fuzz/scheduling` | campaign、seed ledger、阶段依赖、worker/pool 与队列 supervisor |
| `workloads/fuzz/torch_backend.py` | NNSmith 图的 Torch 后端适配 |
| `execution/scheduling`、`backends` | 公共预算、设备放置、设备租约与后端能力检查 |
| `execution/storage`、`reporting` | 执行事实、结果查询、统计与派生索引 |

`afx start` 由客户端 daemon 提交 fuzz 任务；`afx fuzz run/replay` 提供直接调试入口。生成 Case 后，CPU 参考与目标后端执行同一图和输入，保存比较结果、阶段事实及制品。带 token 的 daemon 上传启用同步的结果；直接调试可使用 `afx sync`。

队列 supervisor 属于 fuzz 调度器，与负责客户端安装和监护的 `client/supervisor` 是不同组件。本地结果通过 `afx view` 只读展示，页面资源来自 `server/static`。

## 生成与重放

当前使用 E 加 26 位 base36 的静态 seed、F 加 26 位 base36 的动态 seed，统一走 NNSmith + ATen 生成流程。seed 编码采样规格、构图 method、动态样本数及随机熵；固定代码、源码指纹和依赖下重建图与输入，不承诺任意平台/版本逐字节一致。历史记录可读不等于历史 seed 可执行。

- 目录共 368 个 overload，涵盖逐元素/比较、归约和 view/shape，见[清单](aten_op_list.md)。矩阵乘、卷积、原地及 out 变体未纳入。
- CaseSpec 保存实际 ATen 目标、算子状态、属性、输入及全部输出。CPU 与目标后端使用同图、同输入字节。
- 动态样本 2–8 组，默认 3；同图固定权重，按模型约束变化尺寸。共享 CPU 阶段验证全部样本，不偷偷回退静态图。
- --ops / --exclude-op 在生成后过滤，不改变同一 seed 的图或输入。生成失败、CPU 无法执行、确认无 Autofuse 候选的用例跳过，不进入有效 attempts。
- Autofuse 覆盖依据运行时编译候选事件与调度汇总；进入 Autofuse 后的 lowering、schedule、codegen 或精度失败仍保留。
- NNSmith fork 由 [pyproject.toml](../../pyproject.toml) 固定，执行时校验源码和依赖，缺失或不匹配直接失败，没有备用手写图生成器。

[limits.py](../../autofusex/workloads/fuzz/limits.py) 统一维护预算：当前最多 64 个计算节点、8 组 Shape、每组 256 个张量，CaseSpec 512 KiB、执行 context 768 KiB。生成时检查序列化大小；上传单请求上限 2 MiB，按实际编码大小分批。

## 执行与资源

直接 fuzz 默认使用 CPU 参考与 AF-MLIR；托管 fuzz_task 默认配置包含 CPU、AF-Legacy 和 AF-MLIR，平台或个人配置可覆盖。另支持 eager-npu、triton-npu。AF-MLIR 检查实际 MLIR bridge 与完整 compile_adapter，不能静默回退 legacy。

默认 staged 引擎按 Case → Backend Session → Stage Task 建依赖图。每个 Shape 的 CPU 参考供多个目标后端共享；动态 session 保持同一编译实例、设备和 Shape 顺序。首次调用可能包含惰性编译、autotune、执行和同步，仍是合并阶段，不能当作 kernel 时间。直接 fuzz 保留 combined 路径，托管 task schema 只接受 staged。

共享进程池受 CPU、内存、缓存与设备预算限制。物理 NPU 通过 UUID 锁独占，CPU 与不同卡可并行。worker 按后端、设备和编译参数隔离并有界复用，默认最多 16 个任务，失败或资源增长会触发回收。

自动资源策略在启动时按有效 CPU、RAM 和设备数计算，编译并发起点上限为 8，workers × compile_threads 不超过有效 CPU；这是启发式，不是在线最优调参。托管任务使用平台拉取的设备和资源配置，all 在所选执行环境中展开为可见设备；手动资源策略使用配置中的并发值。每次启动冻结本次任务的配置，不读取 trust 或旧 client policy 的设备、资源上限；运行时保留资源可用性检查和设备锁。

精度以 CPU 为准，浮点容差由 [compare.py](../../autofusex/execution/compare.py) 按 dtype 决定，整数/布尔精确比较；NaN 不通过，同符号 Inf 可通过。CPU 不与自身比较。精度失败直接保留，不自动追加 eager-npu 复核；人工 replay 产生独立记录。

## 数据与统计

| 数据 | 内容与边界 |
|---|---|
| manager.sqlite、environments/、runs/ | 私有操作、配置、环境 manifest、托管任务与控制状态 |
| tasks/tasks.sqlite | 直接 fuzz 队列、claim、恢复；托管队列在 runs/RUN_ID/tasks.sqlite |
| results/results.sqlite | 直接 fuzz 的 attempts、shapes、contexts；托管结果在 runs/RUN_ID/results.sqlite，不默认保存 tensor 和编译产物 |
| 派生索引与公共统计侧库 | 可重建的当前用例、历史、维度和统计贡献 |
| runtime、stage/lifecycle 事实 | 心跳、运行状态、资源和时序；缺失指标保持未采集 |

每次执行使用独立 attempt ID，不以 seed 覆盖结果。聚合层通过 round_id 与 expected_backends 汇合同次执行；当前结果按 **seed + shape_index** 聚合，选择最后首次入库的完整轮次，再应用用户、版本、状态等筛选。要求的后端到齐且包含 AF-MLIR 才成为完整轮次，不拼接不同轮次的后端。重复写入不改变顺序。新版本仅接受当前协议；结果库兼容性以存储 schema 校验为准；不要把客户端结果库的格式限制套用到支持有序迁移的中心数据库。

当前质量和性能以当前完整轮次的 AF-MLIR 为准；CPU、多后端、重试不重复扩大用例分母。历史全部保留。算子类别和 dtype 是多标签统计，每个 Shape 在各标签内计一次；最大维度取输入/输出最大轴长，张量规模取最大单张量元素数。

结构化读取模型按变更 seed 更新，源记录、派生更新与 ACK 保持事务一致；模型过期时回退精确查询。公共缓存保留最后成功快照及 stale 状态，后台增量刷新、失败退避；忙时可返回 503/Retry-After，旧缓存不能作为任务刚完成的证据。

shape_processing 是 Shape 处理墙钟时间；端到端、排队、工具吞吐与 kernel 性能分别统计。无法独立测量的编译/kernel 指标留空，不用首次调用或阶段和代替，也不为历史记录补造事件。

## 上传与问题处理

客户端的 `platform_transport.py` 与平台同步模块负责上传、心跳和复测适配。中心负责入库、归类、问题领取及验证/回归，见[中心服务架构](../server/architecture.md)。日常配置见[客户端使用说明](../client/configuration.md)，临时运行见[进阶命令](../client/advanced.md)。
