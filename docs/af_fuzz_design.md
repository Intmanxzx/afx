# Fuzz 设计

本文件描述当前 `autofusedebugger/fuzz` 及已确认的演进设计；第 9 节为分期方案，9.10 保留第一切片历史，9.11 记录当前阶段调度、资源池及观测实现与验证边界。本文取代旧重构方案、开发计划、D0调查、命令草案和旧NNSmith fuzz文档。历史调查/测试日志可从Git历史查阅。命令见[使用说明](af_fuzz_usage.md)。

## 1. 模块与依赖

同一wheel、同一版本线提供NPU客户端、中心服务和个人电脑查看。Fuzz不依赖旧debugger；普通import不初始化NPU或修改依赖。NNSmith是广泛随机图生成的必需组件，不能用有限模板替代。

```
cli / scheduling -> generation -> contracts + tensors
                 -> execution -> backend adapter
                 -> storage -> sync -> central
                            -> reporting -> view / central HTTP
```

| 模块 | 职责及扩展点 |
|---|---|
| contracts、tensors | frozen CaseSpec/CaseBundle，规范字节、TensorSpec、执行与评估配置 |
| generation | 统一生成入口；C 静态协议与 D 多 Shape 协议，固定 NNSmith 源码/Z3及算子域，无模板回退 |
| backends | 延迟加载Torch，适配check/prepare/inputs/invoke/synchronize/compare；新增后端不能改变case |
| execution | 执行保留共用 Shape 循环，支持 C 静态与 D 多 Shape；正确性和可选性能阶段；失败阶段、误差、计时 |
| scheduling | 持久任务、批次、worker监督、租约、超时/恢复、预算 |
| storage | 白名单规范化、事务保存、schema版本、快照、导入和查询 |
| sync、central | 本地outbox与ACK、token身份、幂等接收、私有备份 |
| reporting | 持久派生索引、shape级汇总、错误摘要、安全WHERE条件 |
| view/static | 同一套预构建网页，无Node运行依赖；公开网站与离线view复用 |

架构测试禁止生成/契约反向依赖执行与存储，禁止fuzz引用旧debugger，NNSmith由生成适配层延迟加载。扩展必须补协议golden、adapter能力、存储白名单/迁移及查询测试，不增加第二套执行引擎。

## 2. NNSmith生成与seed

run/replay统一使用NNSmith `graph_gen.model_gen`，通过symbolic-cinit与独立Z3 context生成随机拓扑、输入规格和算子属性，再转成不可变CaseBundle。手写v1/v2模板已删除；开发态不保留旧A/B的解析、展示或执行兼容；旧数字编码也拒绝，不重新解释为新图，旧记录需清理后重新采集。

当前C协议：Elementwise（Add/Mul/Neg/ReLU）、Reduce（Sum/Mean/Min/Max）、独立Transpose，固定8个计算节点，fp32、单静态shape、维度1–8、单tensor最多4096元素。生成未达到节点预算时拒绝，不接受随机器速度变化的部分图。算子选择不按设备能力过滤，后端不支持则失败。

C/D + 9位 base36 为新批次的 10 字符 seed；C 保持单静态 Shape，D 在同一图和固定权重上构造 3 个合法不同 Shape，不能构造时显式失败，不退回静态。图由NNSmith生成，数据字节由字段寻址SHA256和规范小端编码生成，避免Torch全局RNG影响输入。workload指纹覆盖完整规格与字节，缓存不参与确定性保证。

依赖要求：源码base commit `bc0af42c7d5fc4fd201efb76e5313f6298c2d573`的已补丁隔离副本，Python源码树SHA256 `a1a524f235f8b6299c397fd12c383d4b15a80f027230ece2d7d18259f92c872e`，Z3 4.12.2。实际import来源必须匹配，缺失或不匹配直接报错，无模板fallback。此门禁只是依赖固定，不等于已证明跨设备/跨时间全域确定性；跨平台验证仍待完成。

图、输入/输出规格作为受限轻量case_spec进入结果和同步协议，云端与离线view直接索引采集数据，不重新生成图、不依赖NNSmith。它不是重放所需的缓存。普通import不加载NNSmith；真正生成时才检查依赖。

## 3. 执行、精度与性能

正常批测及默认 replay 执行 CPU 和 AF-MLIR。CPU 是 eager 精度基准。**已实施：移除自动 eager-npu 复核**；与 CPU 不一致即记录精度失败，保存误差证据后进入数据记录，不再追加设备任务。需要时由用户独立运行 `autofusedbg fuzz replay <seed> --backend eager-npu`，人工分析，不能自动覆盖原始失败。

实现状态：执行代码不再调用自动复核，CPU 不一致即失败；显式 eager-npu replay 是新的独立 attempt。历史记录中的 `secondary_used`、`secondary_pass`、`basis` 保留原始语义。编译/执行异常不能靠复核变成 Pass。

公开后端：`cpu`、`eager-npu`、`triton-npu`、`af-legacy`、`af-mlir`。单后端replay用于定点诊断。AF-MLIR要求实际加载可用MLIR bridge，严格模式不能静默回退legacy。

执行阶段：生成 → capability → prepare → 每shape输入绑定 → first_call_combined → 精度校验 → 可选性能测量。

- prepare只创建执行器，不代表编译结束。
- first_call_combined是工具阶段，含惰性编译/调用/同步，不是错误原因或kernel指标。
- telemetry v2 新增 shape_processing：输入绑定开始到要求的评估结束的墙钟跨度（包括失败）；v1 历史阶段和不能当作处理耗时或端到端延迟。
- execution_total包含能力检查、准备、全部shape；generation单独记录，两者与子阶段不能重复累加。
- 仅独立可测的compile/kernel_mean进入性能分析。CPU不测kernel；没有可靠采集时为空，不能用wall-time或首次调用冒充。
- 严格设备测量必须持同卡独占租约；使用NPU的autotune不能算纯CPU编译。

默认 scheduling=staged：Case/Backend Session/Stage Task 公共 DAG；队列仍按 case/backend 保留 claim/attempt 兼容恢复，执行拆为生成、CPU参考、backend合并调用、CPU比较、记录阶段。combined 保留为回滚路径。资源默认值是保守自适应，不宣称最优并发。纯CPU编译与NPU执行拆分、跨进程产物交接尚未验证，不开放伪split路径。

## 4. SQLite与统计单位

长期只存轻量结果，不默认归档tensor、编译产物、原始profiler或长日志。

| 数据 | 用途 |
|---|---|
| contexts | 内容寻址去重的环境/执行/评估配置；含CANN、torch_npu、Autofuse commit等版本 |
| attempts、shapes | 不可变执行记录及逐shape状态、指标、短错误；attempt UUID用于幂等，不以seed覆盖结果 |
| case_facts、attempt_facts、stage_metrics、case_dimensions | 可重建的SQL索引、shape规格、OP/类别、版本、分阶段指标 |
| tasks库 | 调度状态、任务租约、批次；不混入公开结果库 |
| outbox / revision / ACK | 本地上传进度；事务与结果一致 |
| 中心users / sources / owners / receipts / tombstones | token hash、归属、去重、恢复/防复活元数据 |

一条网站用例 = 相同来源/批次/配置/版本下一个图的一个shape；CPU对照不重复计数。100个seed各3shape是300条用例，不是200条后端attempt或600条CPU+NPU明细。每后端最新attempt决定当前状态，历史仍可查看。无shape的早期整体失败保留S—，不伪造shape执行。

综合状态优先AF-MLIR；功能错误/精度不一致/未知分开。性能错误独立，不替换精度判定。UI使用Pass、功能Failed、精度Failed、性能Failed、Unkown。图例默认隐藏功能Failed与Unkown，只影响图形，不删除结果或改变顶部全量计数。

## 5. 错误与查询

AF-MLIR 列表摘要使用已保存的运行日志，悬浮展示完整已保存内容，与详情一致。生成源码/包装异常不能淹没实际 codegen 错误。详情直接展示 ASCIR 文本与去重后的报错日志，不再折叠；运行级 stderr 没有 Shape 归属时，不把它声称为当前 Shape 根因。旧日志截断后无法恢复。精度误差保存max/mean abs、max rel、mismatch/nonfinite数量及二级标准，不保存张量。

列表/图表/汇总共用shape投影，搜索Pass不能因CPU成功匹配目标后端失败用例。SQL入口只接受白名单WHERE比较、LIKE、AND/OR、括号；绑定参数，限制长度/嵌套/时间/返回条数，不执行任意SQL/PRAGMA/ATTACH。用户字符串使用textContent，不拼接HTML。

## 6. 身份与后台同步

管理员离线登记唯一用户名、生成/轮换随机token；云端仅存hash。客户端setup只提交token，调用identity解析用户名。展示名不能鉴权，客户端自报姓名不能覆盖云端归属。没有在线注册或密钥申请。

token保存在客户端0600私有文件，仅用于上传；不进入结果、网页或普通日志。上传器是独立后台进程，默认30秒，配置范围1–86400秒。断网保留outbox；相同source/revision/内容重传幂等，不同内容冲突拒绝；ACK丢失可重试，不阻塞测试。服务器恢复必须保留与结果对应的receipt/ACK，不可随意回退后接受超前ACK。

POSIX后台启动已实现；容器/主机重启后需由systemd或entrypoint拉起上传器。尚不承诺原生Windows批测daemon。

## 7. 服务、离线查看、迁移

中心：单Python应用+SQLite，绑定127.0.0.1:8090，由TLS代理提供公网80/443；无Redis/PostgreSQL。Compose固定基础镜像版本，或同wheel+systemd；前端随wheel，无Node服务。默认客户端地址https://autofuse.online。

离线view：个人电脑主动启动临时loopback服务，随机capability路径、Host/Origin检查、CSP、只读查询；不上传、不执行模型。不在NPU环境常驻查看API。仅打开可信快照，不将SQLite视为恶意文件沙箱。

公开snapshot通过一致性事务复制白名单结果表，不含凭据/上传状态；不能直接复制正在WAL写入的主文件。中心admin backup保留完整身份、ACK和结果，必须私密存储。升级/跨机迁移：先私有备份和停服务，再复制完整备份到新路径、安装固定版本、校验、启动；失败恢复旧wheel和配套私有备份，不能用公开snapshot替代中心灾备。

## 8. 发布与当前边界

发布版本以pyproject及包版本一致为准；Tag必须精确等于`v<version>`，构建同一个wheel用于GitCode Release和PyPI；网站资源必须入wheel。具体流水线、凭据和演练命令见使用说明。PyPI不允许重用同版本文件。

当前验证见本轮远端日志。历史模板结果不能作为NNSmith能力证据。已实现的阶段调度与资源池见 9.11；未验证边界为编译型后端跨 Case 持久复用、独立编译产物交接及数小时长期稳定性。其他未完成：更广算子和dtype、跨平台全域seed确定性、准确编译/kernel指标、自动重建已补丁依赖与真实发布联调。旧NNSmith代码和patches保留用于依赖准备及迁移，公开执行只有当前统一流程。

CLI：`--fg`、`--perf`；run/replay默认不保存结果，`--save`才写结果及outbox。setup成功后记住数据目录。C 协议保留单 Shape 静态语义；D 协议使用同一 NNSmith 图的 3 个合法不同 Shape。随机 campaign 按 entropy 奇偶选择 C/D；静态模式不展示 S0，动态样本标签与 seed 分离。

## 9. 阶段级调度、资源池与工具性能统计（已确认，分期实施）

### 9.1 目标与边界

从“一个 backend 一个完整 worker”演进为“公共阶段任务 + 资源池 + 依赖调度”，提高 CPU 和 NPU 使用率，降低重复生成、重复初始化和无效等待。统一执行引擎供 run/replay 使用，不另建一套后端流程。

统一的是阶段契约，不是假设所有 backend 都能独立编译。目前 `torch.compile` 首次调用可能包含惰性编译、autotune、执行和同步；没有验证可拆分前，必须标记为合并阶段，不能把 `prepare` 当编译完成，不能把需要设备的 autotune 当纯 CPU 工作。

性能目标分别验证：降低单 Shape 处理延迟、提高整机成功吞吐、降低单位 Shape 的工具开销。并发吞吐增加不等于单 Shape 或 Kernel 变快，不提前承诺提升倍数。

### 9.2 公共任务模型

| 层次 | 内容和生命周期 |
|---|---|
| Case | seed、不可变图和输入数据、Shape 样本、共享 CPU 参考结果；一个 Case 只生成一次 |
| Backend Session | 一个 backend 运行 Case 的上下文；维护编译实例、设备亲和性、缓存与有序 Shape 序列 |
| Stage Task | 最小调度单元；包含依赖、资源需求、超时、输入输出引用及恢复策略 |

每个 Stage Task 至少包含：

- 标识：stage ID、case ID、session ID、attempt ID、可选 shape index。
- 执行契约：阶段类型、依赖、输入/输出引用、backend 能力要求、可否重试、超时、取消策略。
- 资源需求：CPU token、主机内存预算、可选 NPU UUID/兼容设备集合、显存及临时缓存预算。
- 状态：等待依赖、就绪、等待资源、运行、成功、失败、跳过、取消。
- 观测：各状态时间事件、实际 worker/设备、失败原因、冷/热初始化与缓存状态。

同一次恢复/重试必须保留 attempt 和阶段执行次数，使用 claim token 防止旧 worker 提交过期结果；不能把失败静默改成重试成功。各阶段输出应有校验信息和明确的所有权、引用计数及清理时机。

### 9.3 公共执行流程与后端能力

```text
生成 Case ──┬── CPU 参考计算 ──────────────────────────┐
            │                                          │
            ├── Backend 能力检查 / 准备                 │
            │        ↓                                 │
            │   编译 → 硬件执行                         │
            │   或首次调用（编译与执行合并）             │
            │        ↓                                 │
            │   同步 / 输出转存 ────────────────────────┤
            │                                          ↓
            │                                      精度比较
            │                                          ↓
            └── 可选性能测量（受控设备窗口）────────→ 结果汇合
                                                       ↓
                                                    数据记录
                                                       ↓
                                                    异步上报
```

- CPU 参考计算与目标 backend 准备、编译、执行可以并行；精度比较等待两边输出。
- 同一 Case 的参考结果可供多个 backend 共享，以 workload、Shape、参考实现及版本为键，不能跨不兼容运行环境复用。
- **没有自动 eager-npu 复核阶段。** 精度不一致直接记录失败；人工复核是独立任务、独立记录，不属于自动任务 DAG，也不改写原始结果。显式指定 eager-npu 作为测试 backend 仍然允许。
- 性能比较等待对应测量结果，并检查设备、版本、预热和测量配置可比性；是否在精度失败后继续测量由明确评估策略决定，不能静默改变流程。
- CPU 对照不重复计入目标 backend 吞吐。若展示 CPU 自身正确性，不得用同一次输出与自身比较伪造验证。
- 动态 S0/S1/S2 保持同一个编译实例和规定执行顺序，不为了吞吐拆到不同卡；各样本的 CPU 参考计算可以独立准备。

Backend adapter 声明以下能力，调度器只消费声明，不硬编码某个 backend 的特殊执行流程：

| 能力 | 调度影响 |
|---|---|
| 纯 CPU 编译是否可用 | 允许在 CPU 池提前编译，否则保留合并阶段 |
| 编译是否需要设备/autotune | 申请 CPU 与 NPU 的联合资源 |
| 编译产物是否可转移 | 决定是否可跨进程、跨同型号设备交接；能力未经验证时禁止 |
| session 是否可复用 | 决定 worker 生命周期、任务间清理及隔离策略 |
| Shape 顺序及设备亲和性 | 决定是否必须在同一个 session/设备运行 |
| profiler 是否可用 | 决定是否提供独立 Kernel 指标；不可用时为空 |

### 9.4 CPU / NPU 资源池

**CPU 池：**按 CPU token、内存和线程预算准入，而不是仅按 worker 个数。图生成、CPU 参考计算、编译、比较分别申请资源。进程内 Torch/OpenMP/BLAS/编译线程数必须计入预算，避免嵌套并行导致线程过量。

采用隔离进程池。NNSmith 的 RNG/Z3/算子设置、Torch 编译配置和原生日志捕获涉及进程级状态，不能简单用共享线程池并发执行。记录与上传使用独立轻量执行通道，不长期占用计算槽位。

**NPU 池：**以物理 UUID 为唯一资源标识，逻辑编号只用于映射。默认由空闲兼容设备领取任务，避免在生成时固定轮转到繁忙设备；用户指定设备和已绑定 session 的约束仍然生效。

设备租约仅覆盖真正需要设备的阶段：设备相关编译/autotune、执行、同步及输出转存。输出已安全转移且不再引用设备内存后，释放执行槽，CPU 比较与记录可以继续。必须区分：

- NPU 执行槽占用：同卡受控执行互斥。
- session 显存驻留：即使未执行也占显存，必须有容量上限和回收策略。
- 物理设备租约：不能在仍有异步工作或子进程使用设备时提前释放；现有同用户锁不能保证外部/跨用户任务不干扰。

多资源联合准入必须原子化，或者采用固定顺序并在获取失败时释放已取得资源；禁止持有 CPU token 无限等待 NPU，反之亦然。性能测量需要受控独占窗口，不能把高吞吐运行状态当成严格设备性能测试环境。

**持久 worker：**复用导入的 Torch 和设备初始化；按 backend、运行时配置和设备分组，af-legacy/af-mlir 等进程级配置必须隔离。每次任务后清理 case 状态和日志；设备异常、RSS/显存持续增长、达到任务数量上限时回收。崩溃或超时的 worker 直接替换，不能污染后续任务。

**临时产物：**生成结果、参考输出和转存输出使用有界本地引用传递，校验内容和版本；不默认落长期库、不上传张量、不反序列化来自远端的可执行对象。取消或失败时也必须回收。

### 9.5 调度策略和背压

1. 按 CPU-ready、NPU-ready、compare-ready、record-ready、upload-ready 分类管理就绪任务；只调度依赖已满足的阶段。
2. 按每类资源的可运行任务数、设备兼容性和内存水位补充任务，不能仅看 pending 总数。修复某张卡积压使另一张卡和 CPU 无任务的问题。
3. 优先处理可释放 NPU 输出缓冲或大量主机内存的转存/比较阶段；保证空闲 NPU 有兼容就绪任务。加入等待时间优先级，避免长任务饥饿。
4. CPU 参考与目标执行独立推进；编译可拆分时才预编译，不允许虚构 split 阶段。
5. 调度主循环只做依赖推进、准入、派发、超时/取消判断。日志收尾、产物清理、失败记录放入有界收尾通道，不能在主循环重新运行图生成器。租约只有在进程组退出、设备安全后才释放。
6. 对已生成未执行 Case、等待比较的输出字节数、驻留 session、待记录队列分别设上限。上游依据下游水位背压，不能用无限预取换取瞬时使用率。
7. SQLite 保留短写事务，outbox 与结果保存保持原子性；上传继续独立运行，同一来源的 revision/ACK 保持顺序和幂等，不为并发破坏数据一致性。
8. 有限批次完成条件是所有要求的结果已持久化；上报完成另行显示。取消请求停止补充和派发，取消依赖任务并清理活动任务，不无限等待未产生的输出。

### 9.6 单 Shape 计时、吞吐和统计口径

当前列表阶段和主要覆盖输入绑定、首次调用和精度校验，不含完整生成、运行时初始化、排队、记录成本。新指标必须单独版本化，不能把历史阶段和改名成完整端到端耗时。

| 指标 | 定义 | 用途 |
|---|---|---|
| Shape 处理耗时（s） | Shape 处理分支实际开始，到要求的精度/性能评估完成；按墙钟跨度计，包含范围内的等待 | 单样本处理延迟 |
| Shape 端到端延迟（s） | 所属 Case 被有限调度窗口接纳，到该 Shape 结果持久化完成 | 用户真实等待时间，含共享生成/准备、排队及记录 |
| 排队/等待耗时（s） | 依赖等待与资源等待分别记录；总等待按时间区间并集统计 | 定位依赖及资源瓶颈 |
| 成功 Shape/min | 窗口内成功且满足所选评估要求的目标 Shape 数 / 窗口分钟数 | 有效测试吞吐 |
| 已完成 Shape/min（含失败） | 窗口内已持久化最终判定的目标 Shape 数 / 窗口分钟数 | 工具处理吞吐，必须同时展示状态构成 |
| 摊销秒/Shape | 测量窗口秒数 / 窗口内已完成目标 Shape 数；成功口径另列 | 整机处理成本，不是单 Shape 延迟 |
| Kernel 延迟（μs/ms） | profiler 采集的设备执行指标 | 后端 Kernel 性能，与工具耗时分离 |

“已完成结果”表示这次 Shape 尝试已经结束，并保存最终判定，**不等于成功**：包括通过、精度失败、功能错误/不支持和性能错误。超时仅在已明确归属 Shape 且保存失败结果时计入；尚未开始或未完成评估的取消、跳过项不计入已完成 Shape，单独展示。生成失败或其他尚无 Shape 归属的错误计入 Case 级错误，不能虚构 Shape 来增加吞吐。页面不用“终结结果”一词。

统计约束：

- CPU 对照不与目标 backend 重复计数；多个目标 backend 分开统计，跨 backend 总量明确标为“Shape-backend 尝试”。
- 吞吐按 attempt 计并单列重试数；覆盖率按唯一 Case/Shape 计，两者不混淆。显式人工 eager-npu 复核不计入原批次的自动测试吞吐。
- 成功/失败、静态/动态、冷/热初始化、缓存命中状态分别分组；不能靠快速失败制造“提速”。
- 连续 campaign 从每个 Case 真正进入有界调度窗口开始计时，不从 campaign 启动时间计算所有 Shape 延迟。
- 共享生成、初始化和编译开销在 Case/session 级记录，不平均分摊后冒充 Shape 实测；若展示摊销值必须明确标注。并行分支总耗时按跨度或区间并集计算，不能直接相加。
- P50/P95 按同口径样本计算，同时展示样本数。无完成样本时吞吐为 0、摊销耗时为空；缺失耗时为空，不视为 0。超时耗时是到超时为止的观测值，不是正常完成延迟。
- 有限基准批次窗口从首个 Case 接纳到最后一个目标结果持久化结束，包含填充/排空；稳态滚动窗口另列。多个主机按各自采样窗口/设备秒归一化，不能随意把持续时间相加当全局墙钟。

### 9.7 时间事件和数据模型

记录 `admitted → ready → dispatched → started → finished → persisted → uploaded_ack`，并保存每个 stage 的依赖就绪时间、资源申请/取得/释放时间、worker/session ID、实际设备 UUID、attempt/重试序号和失败阶段。

- 本机持续时间使用同一时钟域的单调时钟，携带主机及启动会话标识；重启前后、不同机器之间不得直接相减单调时间。墙钟仅用于展示和关联，跨主机延迟需标明时钟误差或采用客户端本地观测。
- 结果正文及哈希仍不可变。持久化完成和上传 ACK 在结果产生后才可知，作为独立追加的生命周期事件保存，不反写已哈希结果；事件使用稳定 ID、幂等同步和独立版本。
- 明确 `persisted` 观测点为结果事务成功返回；生命周期事件自身落库有单独恢复/去重规则。不能为统计引入重复结果或虚假的精确时间。
- 引入版本化 stage facts、资源租约区间和调度窗口摘要；原始事件有数量/保留期限上限，长期保留可查询聚合。迁移可重建索引，但不能给历史记录补造时间。
- 记录/上报失败不覆盖执行判定；页面区分“执行已完成”“结果已保存”“已上报”。上报 ACK 延迟不计入本地端到端执行延迟。

### 9.8 展示设计：统一使用“使用率”

**总览增加工具执行效率：**成功 Shape/min、已完成 Shape/min（含失败）、处理耗时与端到端延迟 P50/P95、样本数、CPU 资源池使用率、各 NPU 执行槽使用率。

资源指标必须说明分母：

- CPU 资源池使用率 = 已分配 CPU token-seconds / 可用 token-seconds，是调度容量使用情况，不冒充操作系统实测 CPU 使用率。
- NPU 执行槽使用率 = 执行槽占用时间区间并集 / 可用设备时间，可能包含设备编译/autotune，不等于 AICore 使用率。
- 实测 CPU 使用率、AICore 使用率和显存使用率若可采集则独立展示，使用可用时段及有效采样覆盖率；未采集则为空。

**列表：**Seed、独立的小字体动态 Shape 标签、Backend、状态、处理耗时（s）、端到端延迟（s）；静态不显示 S0。历史不完整指标标“未采集”或保留明确的旧口径标签。

**详情：**保留当前重放命令、DAG、ASCIR、日志和环境信息；在计时区域提供 CPU/NPU/比较/记录/上报分泳道时间线，显示共享阶段、等待区间和失败点，可区分实际执行与排队。

**性能分析：**拆为“工具吞吐与调度效率”和“编译及 Kernel 性能”。按 backend、设备/硬件、软件版本、静态/动态、冷/热、缓存状态及成功/失败过滤；状态筛选后的吞吐分母仍是相同测量窗口，不能只截取成功任务时间。

### 9.9 分期实施与验收

1. **公共契约和观测基线：**落地阶段模型及版本化计时；移除自动 eager-npu 复核及原有自动改判逻辑，更新精度测试和使用说明；修复资源偏斜和同步失败收尾。保留当前合并执行能力，先验证结果语义和准确计时。
2. **共享产物与持久运行时：**一次生成、多 backend 复用 CPU 参考结果；CPU/NPU 进程池、有限 session 复用、输出转存后释放设备、异步比较及记录。验证污染隔离、内存上限、worker 回收和停止/恢复。
3. **能力证明后的阶段拆分：**只对验证通过的 backend 启用编译/设备执行解耦和产物交接；其余保持合并阶段，不能为了调度粒度虚构指标。

在 `npussh mlir` 两张可见 NPU 上，以固定 seed 集、相同资源预算对比：单 worker、当前四 worker、新阶段调度、再加运行时复用。分别测冷启动和稳态，记录成功/失败分布、P50/P95、吞吐、CPU/NPU 使用率、队列等待、RSS/显存及缓存峰值。增加单卡慢任务、同卡积压、设备异常、超时、取消、写库失败、上传断网和恢复场景。

此前 4 任务小样本的 30.90 s / 14.49 s 只能说明跨任务并发已有效，其中 AF-MLIR 为失败路径，不能据此承诺新架构加速比例。新的对照必须报告自动复核取消所改变的工作量，性能基准固定相同复核策略；不能把少做工作与调度收益混为一谈。

验收要求：历史 C/D seed 及输入内容不变、CPU 精度判定不因调度改变、不同卡确有重叠且同卡约束有效、没有未归属错误被伪装成功、资源不越过配置上限、取消/恢复不重复提交、长期内存稳定。通过后再部署 `oracle1` 与 NPU 客户端，保留版本化产物、私有备份和回滚路径。


### 9.10 第一交付切片：合并执行调度与观测基线（2026-11-10）

**已实现**

- 删除自动 eager-npu 复核及改判；保留历史诊断字段，不改历史哈希。
- campaign 候选设备集合保存在私有队列扩展表，派发时领取空闲兼容逻辑卡；显式 enqueue/replay 设备仍固定。实际设备写入 claimed execution，物理 UUID 锁仍由 backend 执行；尚不是按显存/UUID 容量原子准入的完整资源池。
- refill 按 CPU/NPU 两类待处理需求补充，有独立预取上限与既有 max_active 总上限；不是完整 Stage DAG-ready 调度。
- 有界收尾线程负责进程组终止、日志收尾、临时目录清理、保存及任务确认；收尾未结束不释放 worker/设备名额。其他活动任务的超时检查不等待慢收尾。worker 失败使用 worker-failed/1，无 Shape，不重新生成图。
- telemetry v2 记录 Shape 实际处理墙钟跨度，v1 可读；列表明确处理耗时与端到端列，旧数据缺失显示“未采集”。保留 DAG、日志、环境。
- 本地 lifecycle_v1 以 (attempt_id,event) 幂等追加，每 attempt 最多七个事件。入队时 admitted/ready（当前合并任务无上游依赖）、派发 dispatched、worker started/finished、结果事务返回 persisted、ACK 事务返回 uploaded_ack。Linux 按主机+boot 域计算单调时间，其他平台不跨进程相减；重试重新接纳，历史缺失不补造。结果哈希不含生命周期事件。提交与事件间崩溃会缺事件，仍承认已保存结果。
- 同一个 attempt 的 Shape 同事务持久化，当前本地端到端跨度相同；跨时钟域为空。工具效率先提供按 backend 的处理耗时样本数/均值/最大值，不从相加阶段伪造窗口吞吐。

**明确未完成（不能作为完整第 9.9 第一阶段验收）**

Case/Backend Session/Stage Task 公共执行契约、依赖 DAG、一次生成与共享 CPU 参考、CPU token/内存/显存联合池、输出转存后提前释放设备、持久 worker/session、独立比较及记录进程、租约区间、P50/P95 与完整窗口吞吐、原始事件保留期限及聚合、分泳道时间线、硬件/软件/冷暖/重试维度过滤均待后续切片。ExecutionConfig.scheduling 仍严格限定 combined；不能把本切片包装成 split 或新阶段调度基准。收尾结束前保留全部并发名额，属于保守安全策略，尚未优化资源提前释放。

生命周期当前是**本地扩展**，未加入 protocol 1 上传或公开 snapshot；中心网站不能从客户端 v2 payload 得到端到端/ACK 时间，显示未采集。不可用中心接收时间替代客户端持久化时间。版本化事件同步与恢复仍待实施。受控性能测量策略和严格设备独占限制不变。

**测试与基准**

远端完整 Fuzz 回归：284 passed、4 skipped（旧 Node）；本地 DAG 4 项通过。新增测试覆盖空闲兼容卡领取、CPU 补充不受 NPU 积压影响、慢收尾不阻塞其他超时、worker 失败不生成图、生命周期幂等/哈希不变/跨时钟域、处理跨度不重复相加。现有取消、进程崩溃、写库故障、断网恢复测试继续运行；未宣称长稳态 RSS/显存回收已验收。

真实双卡 eager-npu 成功路径，固定 C16/D16 各重复两次、8 Shape，compile_threads=1，每任务仍为冷 Python worker：旧单 worker 28.22 s，旧四 worker 13.64 s，本切片四 worker 14.46 s。三组均 8/8 Shape 成功。卡间重叠、同卡不重叠；这是一次小样本，**未证明本切片相比已有四 worker 加速**。eager-npu 不走原 af-mlir 自动复核，因此这组工作量不受取消复核影响。未测持久运行时/稳态，不给出其收益；af-mlir 成功编译路径尚无证据。

最终源代码回归再次得到 284 passed、4 skipped；随后补充独立生命周期写失败测试，阶段基线测试共 7 passed。本地轻量验证 18 passed、3 skipped（无 Torch；macOS 上 Linux 管理进程测试不运行），另新增写失败与 DAG 合集 9 passed、2 skipped。macOS 的 flock/SQLite 组合不作为受支持的客户端管理运行环境；实际管理执行在 Linux 验证。新入口 D16 eager-npu 3/3 Shape 通过；AF-MLIR C16 仍为后端能力失败，未把它计入成功编译提速证据。


### 9.11 阶段 DAG、资源池及独立事实同步（2026-11-10）

本节覆盖 9.10 所列核心待实施代码；9.10 是历史切片状态，不再代表当前默认行为。

**公共执行链路**

- `stages.py` 定义 Case、BackendSession、StageTask、Requirements。`staged.py` 是所有 backend 共用的依赖协调器，run/replay 都接入它；`execution.py` 仍是唯一 backend Shape 调用循环，未复制一套 AF 专用引擎。
- Case 只生成一次图和规范输入字节，写入有界本地产物并校验 SHA256。CPU 参考和各 backend 合并阶段从相同输入字节加载；不再次运行生成器。一个 Case 只运行一次 CPU 参考阶段，参考结果供所有目标 backend 使用；CPU 自身正确性沿用独立 eager 参考调用，不拿同一输出自比。
- 图生成结束即可同时派发 CPU 参考和 backend 工作。目标分支不等待参考；CPU compare 等待两边。prepare 仍不等于编译完成，first_call_combined 没有虚构编译/执行拆分。D S0/S1/S2 在同一 prepared callable、同一设备和进程中顺序执行。
- 输出 `.cpu().clone()`、设备同步、数据化 NPZ 发布完成后结束 NPU 阶段；比较及保存移入 CPU worker。所有产物仅在私有窗口目录，禁止 pickle/object-array，校验长度/版本/内容哈希，每份输出上限 64 MiB；Case、总缓存、worker 数有界。取消/异常退出删除私有产物。

**资源和进程**

- CPU token、内存、缓存、兼容物理 UUID 与设备预算在 coordinator 中一次准入；申请失败不占其他资源。空闲兼容物理卡取任务，不按生成顺序预绑定。显式 replay 指定设备仍固定；run 的设备列表是候选池。
- CPU capacity = workers × compile_threads；Torch/BLAS/Inductor 线程受控。主机 RSS 采样包括闲置 worker，缓存采样允许并发回收目录；超限终止活动进程组。NPU 从运行时探测实际 UUID/空闲显存，支持每卡 device-memory-bytes 预算，运行时峰值超限失败。**这些是预约/采样约束，不是 cgroup 硬隔离，采样间可短暂超量；外部未遵守 UUID 锁的程序仍可争抢硬件。**
- worker 使用继承私有 socket 的独立进程组，协议只传 JSON 和本地数据引用。按 backend/device 隔离复用；每个 session 清理图、输出、Dynamo 状态及设备缓存。默认最多 16 阶段，RSS 增长、设备驻留增长、异常/超时触发回收；max-tasks=1 禁用复用。
- 已验证 CPU/eager-npu 的运行时复用。编译型 backend 的跨 Case 持久复用默认关闭，每个 session 结束回收进程：当前 AF-MLIR 样本无法成功编译，不能凭 eager 成功宣称 AF 编译状态污染隔离已验证。能力契约明确 independent_compile/transferable_compilation=False；没有可验证独立产物的 backend 不开放编译拆分。
- 调度循环不等待子进程 IO/退出/日志；有限 IO executor 执行这些步骤，事实落库也使用有界收尾通道。记录失败保持任务 interrupted、显式恢复；提交后 ACK 丢失按已有 immutable attempt 对账，不盲目重跑。

**事实、同步和统计**

- 结果/哈希不变；lifecycle_v1、stage_facts_v1、stage_windows_v1 是独立追加事实。stage 记录依赖、ready/dispatched/started/finished、worker/冷暖、资源租约取得/释放、RSS；记录阶段保存 retry_index。admitted 是进入有限 Case 窗口的时间，不冒充任意早期队列时间。
- `api/facts` 使用独立版本/revision/ACK；先上传并 ACK 结果，再上传事实。中心检查用户、source、attempt、Case/window 所有权及内容冲突。重复包幂等，丢 ACK 可重传，旧 result protocol 1 不变。生命周期已进入公开 snapshot/restore/import，但不包含 fact outbox/ACK/凭据。
- 有限窗口首 Case admitted 到最后结果 persisted；连续 campaign 每约 100 attempt 排空形成可比有限窗口。按 backend、硬件/软件环境、static/dynamic、cold/warm、状态聚合，保留共同墙钟分母，不把 CPU 对照重复计入目标吞吐。记录成功/已完成 Shape/min、摊销秒/Shape、唯一 Shape、显式重试数、处理/端到端 P50/P95、样本数、token 使用率、各物理执行槽使用率。
- 生成失败、无 Shape 的 worker 错误不计 Shape；取消与记录失败计窗口独立数量。同一 attempt 全部 Shape 同事务提交，端到端结束点相同；处理开始点各自实测。缺失事件及跨时钟域不补造。profiler Kernel 保持独立，未采集仍为空。
- 原始阶段/生命周期保留最近 256 个已完成窗口（窗口 ACK 前不裁剪），保留长期窗口聚合；离线 outbox 不静默丢弃，磁盘保留水位阻止继续接纳。当前查询分页展示最近 100 个窗口，已归档窗口的细粒度筛选/时间线不伪造。
- 页面提供效率维度筛选、处理/端到端列、CPU/NPU/比较/记录分泳道时间线、依赖/资源等待与失败点；生命周期显示上传 ACK。保留 DAG/ASCIR/日志/环境。资源池使用率不是实测 CPU/AICore/显存使用率，后者未接采样仍显示未采集。

**真实验证与性能结论**

固定 C16/D16 各两次、每 Case CPU+eager-npu，同样 compile_threads=1：combined 单 worker 42.73 s、combined 四 worker 15.52 s、staged 四 worker 禁用复用 26.75 s、staged 四 worker 复用 24.12 s；目标 8 Shape 全通过。这是首次批次的冷启动/填充排空对照，**阶段化在该小图样本更慢，不宣称提速**。少做自动复核不影响该 eager-npu 对照。复用相对不复用有改善，不等于 Kernel 改善。

另跑 12 Case/24 attempt 的有限复用样本，36.55 s、全部成功，72 阶段中 47 次热调用（目标合并阶段 5 次热调用），采集到单 worker RSS 峰值约 1.92 GiB。运行结束两卡无测试进程；HBM 与基线接近。这不是数小时长期稳定性证明。AF-MLIR 同四 Case 的失败路径 28.21 s，CPU 全通过、AF 4 attempt 失败，8 Shape 已完成但成功吞吐为 0；不能当成功编译性能证据。

首轮完整回归 297 passed / 4 skipped（远端 Node 旧），新增故障用例另行纳入最终回归。覆盖依赖 join、原子多资源、不足显存拒绝、空闲卡领取、同卡互斥、取消/超时/崩溃、写库失败、独立事实所有权/哈希不变、HTTP 断网与丢 ACK 恢复、安全产物、快照、并发缓存回收。真实双卡冒烟与隔离中心结果+事实同步通过；不把模拟故障当作实际设备故障证明。


最终回归：完整 Fuzz 300 passed / 4 skipped；新事实 epoch 重建/HTTP 丢 ACK 等最后变动另跑专项。macOS 本地轻量测试通过，Linux-only/NPU 测试以远端为准；不在 macOS 运行管理进程的 flock/SQLite 路径。

### 9.12 最终审计与发布补充（2026-11-10）

- CPU 参考阶段不再调用第二次 eager oracle；每个 Shape 执行一次并共享输出。CPU 参考自身标为 `not_checked`，不声称独立正确性验证；目标 backend 正常与参考比较。新增 1/3 Shape 调用次数测试，历史记录不重写。
- 有限 Case 窗口接纳按兼容设备积压优先级选择；跨窗口取消/写库失败计数不累计重复。生命周期只发布观测到的事件，不以 admitted 伪造未启动阶段的时间。
- 修复取消先于 I/O worker 启动的竞态；补充 NPZ 解压前总大小检查。资源约束仍是预约和采样，而非硬隔离。
- 修正 CPU 参考重复执行后重跑固定 C16/D16 各两次：combined 单 worker **41.23 s**、combined 四 worker **14.86 s**、staged 四 worker max-tasks=1 **28.22 s**、staged 四 worker max-tasks=16 **24.94 s**。目标 eager-npu 的 8 Shape 全通过。阶段化依然慢于已有四 worker 路径；复用减少部分启动开销，但不证明成功编译或 Kernel 性能提高。该组基准保留在远端 `stages-1110/audit-*`，最终取消/解包保护另经全量回归。
- AF-MLIR 失败路径 **35.29 s**，4 个 backend attempt 失败；不能当作成功编译性能。仍需后端成功样本才能验证编译型跨 Case 复用，当前保持每 session 回收。数小时压力测试、独立编译交接与 profiler 实测未完成，不冒充已实现能力。
- 最终发布环境为 oracle1 `/home/ubuntu/af-fuzz/venv-stages-audit-1110`，私有备份 `/home/ubuntu/af-fuzz/backups/stages-audit-1110/`。客户端为 mlir `/mnt/workspace/af-fuzz/stages-audit-1110/installed`；测试使用隔离数据库，不上传测试记录到生产。

最终源码回归：mlir **305 passed / 4 skipped**（旧 Node）；日志 `/mnt/workspace/af-fuzz/stages-audit-1110/complete-tests.log`。本地页面/DAG **29 passed**；本地 CLI 源码门禁用例因未安装固定 NNSmith 失败，该用例以完整远端结果为准，不宣称 macOS 全量通过。隔离中心验证 **89 条事实** 的结果先 ACK、事实独立 ACK、幂等重传与时间线。独立安装入口在 `/tmp` 下重放 D16 eager-npu：3 Shape 通过，避免源码工作目录遮蔽安装产物。

oracle1 已切换到新环境，服务 active；公开 app.js SHA256 `fd8108a52b347efd79a2a68bc8139a6793a5e4927fa2ac4579e480cb20f45975` 与本地一致。生产 users=1、sources=5、receipts=2409、owners=200、attempts=200 保持，integrity_check=ok。公开页保留历史数据的“未采集”，没有上传测试结果伪造线上吞吐。round64 独立上传器已从旧 round64 源码切换至新安装，私有库备份在 mlir `stages-audit-1110/private-sync-backup/`，启动后 state=ok、pending=0、原结果 ACK=200 保持；没有改动 token 或身份配置。

### 9.13 随机长跑与输入轴分布审计

本次 256 对固定随机 seed：C 256 成功生成，D 188 成功、68 无三组合法 Shape；不含 Transpose 的 C/D 分别 129/85。C S0 输入轴直方图（1..8）为 `[744,130,76,70,66,83,83,70]`，1 占 56.3%；排除 Transpose 后为 `[364,55,26,29,32,33,33,32]`，1 占 60.3%。D S0/S1/S2 分别为 `[589,119,74,70,64,78,80,67]`、`[589,145,131,41,51,61,67,56]`、`[589,8,103,206,51,61,67,56]`。因此最终输入轴**不是独立均匀分布**；它符合当前小尺寸/广播约束/确定性变体实现，但不符合广范围覆盖的假设。

初始 concrete placeholder 使用 1..8 均匀抽样，元素预算超限时改为 1 并打乱轴；后续符号求解和广播产生更多单例轴，不能用初始抽样公式推断最终分布。D 按升序尝试 `{2,3,4,max(2,extent-1),extent+1}`，取前两个合法变体，使 S1/S2 偏向 2/3/4。审计保存 816 组分层直方图与每个输入的实际 Shape。未扩大历史协议；新增 campaign 过滤只影响候选是否执行。

部署清理按用户本次明确要求：新版验证后移除 oracle1 旧环境、旧发布包、旧备份和旧结果，保留当前身份/token/来源/ACK，以及防止旧结果恢复上传的删除标记；不再承诺已删除环境可原地回滚。自动备份脚本改用新环境，未来仍可备份新数据。

可复核分层计数归档在 `docs/benchmarks/shape-distribution-20261110.json`，完整逐输入样本在 mlir `/mnt/workspace/af-fuzz/continuous-1110/shape-distribution.json`。双卡预算对照与日志位于同目录 `tune-4/`、`tune-8/`；对照期间同时运行测试，有 CPU 干扰，不能当严格性能排名，只支持“不盲目增加 worker”的保守选择。

最终空闲环境全量回归：**307 passed / 5 skipped**（旧 Node），日志 `continuous-1110/final-idle-tests.log`。并跑校准时两个短超时测试失败，停止基准后完整回归通过；未通过放宽生产超时或删除断言隐藏失败。本地页面/DAG 单独 **30 passed**。细分每个输入/rank/轴后多数分组样本很少，不据此声称每个轴都通过统计拟合；唯一达到 20 样本的无 Transpose C 分组 input0/rank1/axis0 共 29 个值，89.7% 为 1，进一步说明约束引入明显偏置。
