# Fuzz 整体架构

平台由中心服务、客户端管理进程和执行引擎组成。中心管理身份、设备、操作和结果，NPU 客户端执行任务。Agent 与网页共用管理配置和操作类型。代码集中在 [autofusex/fuzz](../../autofusex/fuzz/)，NNSmith fork 是独立执行依赖。

## 组件与数据流

~~~mermaid
flowchart TD
    Agent[Agent / 本地 CLI] --> Manager[客户端管理 daemon]
    Web[网页 / HTTP 调用方] --> Central[中心服务]
    Manager <-->|轮询操作 / 回执 / 状态| Central
    Manager --> Env[环境与工具版本管理]
    Manager --> Run[托管任务 worker]
    CLI[afx fuzz run / replay] --> Engine[阶段调度器]
    Run --> Engine
    Engine --> Gen[NNSmith + ATen 构图]
    Gen --> CPU[共享 CPU 参考]
    Gen --> NPU[目标后端 session]
    CPU --> Compare[精度比较 / 可选性能测量]
    NPU --> Compare
    Compare --> DB[本地结果与执行事实]
    DB --> Sync[异步同步 / outbox / ACK]
    Sync --> Central
    Central --> Read[结果查询 / 统计 / 问题与复测]
    Read --> Web
~~~

| 模块 | 职责 |
|---|---|
| management/ | 本地 Unix socket、操作日志、设备绑定、环境、任务、配置与升级；管理进程不加载模型运行时 |
| generation/、nnsmith/ | seed、NNSmith/Z3 构图、ATen 规则、CaseSpec 与确定性输入 |
| scheduling/、backends/ | 阶段依赖、进程池、设备租约、后端能力和执行 |
| storage/、reporting/ | 执行结果、观测事实、可重建查询模型与统计 |
| sync/ | 上传、重试和确认进度 |
| central/、management/central.py | 身份、HTTP 接口、远程操作、复测、问题归类及公共缓存 |
| view/ | 中心与本地查看器共用的网页资源 |

中心是 Python HTTP 应用加 SQLite，监听 loopback，由 TLS 代理对外提供 HTTPS；无需 Torch、NNSmith、NPU 或 Node 服务。离线 view 使用同一网页和查询层，只读打开结果快照。

## 控制与任务生命周期

本地 Agent 通过 /api/agent 的六个接口使用用户 Bearer token 控制自己的设备，全程使用中心任务 ID；本地 CLI 经私有 manager.sock 请求 daemon。远程操作由客户端主动轮询中心领取。两种入口共用操作白名单、配置校验和执行器。远程领取与回执受租约代次和归属约束，本地持久操作日志通过 request key 去重。

**Operation 和 Run 是两种对象**：提交、暂停、安装各有操作状态；任务提交操作成功后得到 run_id，再查询 Run 的执行状态。配置和环境解析为实际快照及指纹，结果保留执行时配置。连续任务可在安全边界切换配置，更换环境须停止后重新提交。

环境安装到独立目录；已有 CANN 可以复用，Python 环境仍由安装流程创建。ready 表示版本/导入验证通过，manifest 记录 npu_verified=false，不等于真实 NPU 验证完成。本地 trust 限定设备、资源、下载来源、制品摘要和构建命令，远程不能修改它。工具升级使用签名制品、稳定启动器、空闲切换及健康检查。

同一 home 只允许一个活动托管 campaign，并检查直接 fuzz 调度器所有权。管理进程意外退出后不凭旧 PID 盲目重放安装或终止进程，需要处理 needs_attention。后台进程主要支持 POSIX。

## 生成与重放

当前使用 E 加 26 位 base36 的静态 seed、F 加 26 位 base36 的动态 seed，统一走 NNSmith + ATen 生成流程。seed 编码采样规格、构图 method、动态样本数及随机熵；固定代码、源码指纹和依赖下重建图与输入，不承诺任意平台/版本逐字节一致。历史记录可读不等于历史 seed 可执行。

- 目录共 368 个 overload，涵盖逐元素/比较、归约和 view/shape，见[清单](aten_op_list.md)。矩阵乘、卷积、原地及 out 变体未纳入。
- CaseSpec 保存实际 ATen 目标、算子状态、属性、输入及全部输出。CPU 与目标后端使用同图、同输入字节。
- 动态样本 2–8 组，默认 3；同图固定权重，按模型约束变化尺寸。共享 CPU 阶段验证全部样本，不偷偷回退静态图。
- --ops / --exclude-op 在生成后过滤，不改变同一 seed 的图或输入。生成失败、CPU 无法执行、确认无 Autofuse 候选的用例跳过，不进入有效 attempts。
- Autofuse 覆盖依据运行时编译候选事件与调度汇总；进入 Autofuse 后的 lowering、schedule、codegen 或精度失败仍保留。
- NNSmith fork 由 [pyproject.toml](../../pyproject.toml) 固定，执行时校验源码和依赖，缺失或不匹配直接失败，没有备用手写图生成器。

[limits.py](../../autofusex/fuzz/limits.py) 统一维护预算：当前最多 64 个计算节点、8 组 Shape、每组 256 个张量，CaseSpec 512 KiB、执行 context 768 KiB。生成时检查序列化大小；上传单请求上限 2 MiB，按实际编码大小分批。

## 执行与资源

默认后端为 CPU 参考与 AF-MLIR，另外支持 eager-npu、triton-npu、af-legacy。AF-MLIR 检查实际 MLIR bridge 与完整 compile_adapter，不能静默回退 legacy。

默认 staged 引擎按 Case → Backend Session → Stage Task 建依赖图。每个 Shape 的 CPU 参考供多个目标后端共享；动态 session 保持同一编译实例、设备和 Shape 顺序。首次调用可能包含惰性编译、autotune、执行和同步，仍是合并阶段，不能当作 kernel 时间。直接 fuzz 保留 combined 路径，托管 task schema 只接受 staged。

共享进程池受 CPU、内存、缓存与设备预算限制。物理 NPU 通过 UUID 锁独占，CPU 与不同卡可并行。worker 按后端、设备和编译参数隔离并有界复用，默认最多 16 个任务，失败或资源增长会触发回收。

直接 fuzz run 的自动资源策略在启动时按有效 CPU、RAM 和设备数计算，编译并发起点上限为 8，workers × compile_threads 不超过有效 CPU；这是启发式，不是在线最优调参。托管任务还受 trust 和 client policy 上限约束，默认策略仅允许 npu:0、1 worker、1 编译线程，不能套用直接 run 的多卡默认值。

精度以 CPU 为准，浮点容差由 [compare.py](../../autofusex/compare.py) 按 dtype 决定，整数/布尔精确比较；NaN 不通过，同符号 Inf 可通过。CPU 不与自身比较。精度失败直接保留，不自动追加 eager-npu 复核；人工 replay 产生独立记录。

## 数据与统计

| 数据 | 内容与边界 |
|---|---|
| manager.sqlite、environments/、runs/ | 私有操作、配置、环境 manifest、托管任务与控制状态 |
| tasks/tasks.sqlite | 直接 fuzz 队列、claim、恢复；托管队列在 runs/RUN_ID/tasks.sqlite |
| results/results.sqlite | 直接 fuzz 的 attempts、shapes、contexts 与同步进度；托管结果在 runs/RUN_ID/results.sqlite，不默认保存 tensor 和编译产物 |
| 中心结果库 | 执行结果、身份、来源、接收回执、管理与复测数据 |
| 派生索引与公共统计侧库 | 可重建的当前用例、历史、维度和统计贡献 |
| runtime、stage/lifecycle 事实 | 心跳、运行状态、资源和时序；缺失指标保持未采集 |

每次执行使用独立 attempt ID，不以 seed 覆盖结果。中心通过 round_id 与 expected_backends 汇合同次执行；当前结果按 **seed + shape_index** 全站聚合，选择最后首次入库的完整轮次，再应用用户、版本、状态等筛选。要求的后端到齐且包含 AF-MLIR 才成为完整轮次，不拼接不同轮次的后端。重复上传不改变顺序；旧记录缺乏完整标识时按兼容规则展示。

当前质量和性能以当前完整轮次的 AF-MLIR 为准；CPU、多后端、重试不重复扩大用例分母。历史全部保留。算子类别和 dtype 是多标签统计，每个 Shape 在各标签内计一次；最大维度取输入/输出最大轴长，张量规模取最大单张量元素数。

结构化读取模型按变更 seed 更新，源记录、派生更新与 ACK 保持事务一致；模型过期时回退精确查询。公共缓存保留最后成功快照及 stale 状态，后台增量刷新、失败退避；忙时可返回 503/Retry-After，旧缓存不能作为任务刚完成的证据。

shape_processing 是 Shape 处理墙钟时间；端到端、排队、工具吞吐与 kernel 性能分别统计。无法独立测量的编译/kernel 指标留空，不用首次调用或阶段和代替，也不为历史记录补造事件。

## 同步、复测与权限

管理员发放用户 token，中心存 hash；托管设备绑定后使用可撤销设备凭据。上传器独立于模型执行，outbox、revision 与 ACK 持久化。断网重试不阻塞测试，重复内容幂等，不同内容冲突拒绝。

网页用户可选择公开用例在自己的客户端复测，按 seed、执行配置和目标客户端合并任务，不因多个 Shape 重复执行整图。复测包含 CPU 与 AF-MLIR，沿用资源池；领取、续租、取消和完成受租约代次约束。合法迟到结果保留，但不能完成其他代次的任务。

问题归类与认领由 central/issue_groups.py 管理，与执行事实分离。外部 Agent 的接口边界见 [Agent 与 API](agent_api.md)。公开结果快照不含身份和 ACK，完整灾备使用私有中心备份，见[运维说明](operations.md)。

## Agent 修复工作流

中心将问题组冻结为修复任务，保存领取人、Agent 实例、租约、事件和关联用例。Agent 通过 HTTP 领取与续租，通过 npussh 在隔离环境调试、构建、安装候选 RUN 包，再通知中心按原始用例规格复测。任务所有权由事务和唯一约束保证，过期接管换 lease_id；网页操作不能绕过租约。

复测沿用 retest 客户端，批次与修复任务在同一事务内创建。中心检查本轮合法租约结果、完整目标 Shape、精度与请求的性能指标后，才允许提交背景/方案/备注。总结与修复任务一对一，关联到复测结果的 summary_id；失败继续迭代，原始结果不覆盖。API 与安装边界见 [Agent 与 API](agent_api.md)。
