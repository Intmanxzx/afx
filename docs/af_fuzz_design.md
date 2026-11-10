# Fuzz 设计

本文件描述当前 `autofusedebugger/fuzz`，取代旧重构方案、开发计划、D0调查、命令草案和旧NNSmith fuzz文档。历史调查/测试日志可从Git历史查阅。命令见[使用说明](af_fuzz_usage.md)。

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
| generation/nnsmith_v3 | 唯一生成入口，固定NNSmith源码/Z3及算子域，无模板回退 |
| backends | 延迟加载Torch，适配check/prepare/inputs/invoke/synchronize/compare；新增后端不能改变case |
| execution | 执行保留共用shape循环，当前NNSmith协议仅静态；正确性和可选性能阶段；失败阶段、误差、计时 |
| scheduling | 持久任务、批次、worker监督、租约、超时/恢复、预算 |
| storage | 白名单规范化、事务保存、schema版本、快照、导入和查询 |
| sync、central | 本地outbox与ACK、token身份、幂等接收、私有备份 |
| reporting | 持久派生索引、shape级汇总、错误摘要、安全WHERE条件 |
| view/static | 同一套预构建网页，无Node运行依赖；公开网站与离线view复用 |

架构测试禁止生成/契约反向依赖执行与存储，禁止fuzz引用旧debugger，NNSmith由生成适配层延迟加载。扩展必须补协议golden、adapter能力、存储白名单/迁移及查询测试，不增加第二套执行引擎。

## 2. NNSmith生成与seed

run/replay统一使用NNSmith `graph_gen.model_gen`，通过symbolic-cinit与独立Z3 context生成随机拓扑、输入规格和算子属性，再转成不可变CaseBundle。手写v1/v2模板已删除；开发态不保留旧A/B的解析、展示或执行兼容；旧数字编码也拒绝，不重新解释为新图，旧记录需清理后重新采集。

当前C协议：Elementwise（Add/Mul/Neg/ReLU）、Reduce（Sum/Mean/Min/Max）、独立Transpose，固定8个计算节点，fp32、单静态shape、维度1–8、单tensor最多4096元素。生成未达到节点预算时拒绝，不接受随机器速度变化的部分图。算子选择不按设备能力过滤，后端不支持则失败。

C + 9位base36为新批次的10字符seed。图由NNSmith生成，数据字节由字段寻址SHA256和规范小端编码生成，避免Torch全局RNG影响输入。workload指纹覆盖完整规格与字节，缓存不参与确定性保证。

依赖要求：源码base commit `bc0af42c7d5fc4fd201efb76e5313f6298c2d573`的已补丁隔离副本，Python源码树SHA256 `a1a524f235f8b6299c397fd12c383d4b15a80f027230ece2d7d18259f92c872e`，Z3 4.12.2。实际import来源必须匹配，缺失或不匹配直接报错，无模板fallback。此门禁只是依赖固定，不等于已证明跨设备/跨时间全域确定性；跨平台验证仍待完成。

图、输入/输出规格作为受限轻量case_spec进入结果和同步协议，云端与离线view直接索引采集数据，不重新生成图、不依赖NNSmith。它不是重放所需的缓存。普通import不加载NNSmith；真正生成时才检查依赖。

## 3. 执行、精度与性能

正常批测及默认replay执行CPU和AF-MLIR。CPU是eager精度基准；只有AF-MLIR与CPU精度不一致时额外执行eager-npu复核。如果AF-MLIR与eager-npu一致，可判精度Pass，同时保存CPU分歧、误差及判定basis。编译/执行异常不能靠复核变成Pass。

公开后端：`cpu`、`eager-npu`、`triton-npu`、`af-legacy`、`af-mlir`。单后端replay用于定点诊断。AF-MLIR要求实际加载可用MLIR bridge，严格模式不能静默回退legacy。

执行阶段：生成 → capability → prepare → 每shape输入绑定 → first_call_combined → 精度校验 → 可选性能测量。

- prepare只创建执行器，不代表编译结束。
- first_call_combined是工具阶段，含惰性编译/调用/同步，不是错误原因或kernel指标。
- 当前shape总耗时为已采集非嵌套阶段和；失败调用可从telemetry取计时。
- execution_total包含能力检查、准备、全部shape；generation单独记录，两者与子阶段不能重复累加。
- 仅独立可测的compile/kernel_mean进入性能分析。CPU不测kernel；没有可靠采集时为空，不能用wall-time或首次调用冒充。
- 严格设备测量必须持同卡独占租约；使用NPU的autotune不能算纯CPU编译。

调度粒度为case/backend，任务持久化、worker隔离、超时终止进程组、重试保留attempt。目前监督器仍串行，资源默认值是保守自适应，不宣称最优并发。纯CPU编译与NPU执行拆分、跨进程产物交接尚未验证，不开放伪split路径。

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

综合状态优先AF-MLIR；功能错误/精度不一致/未知分开。性能错误独立，不替换精度判定。UI使用Pass、功能Failed、精度Failed、性能Failed、Unkown。图例默认隐藏功能Failed只影响图形，不删除结果或改变顶部全量计数。

## 5. 错误与查询

当前shape核心错误优先展示；日志摘要与详情共用提取器。生成源码/包装异常不能淹没实际codegen错误。运行级stderr没有shape归属时单独折叠、去重，不把它声称为当前shape根因；旧日志截断后无法恢复。精度误差保存max/mean abs、max rel、mismatch/nonfinite数量及二级标准，不保存张量。

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

当前验证见本轮远端日志。历史模板结果不能作为NNSmith能力证据。未完成：动态shape协议、更广算子和dtype、跨平台全域seed确定性、准确编译/kernel指标、自动重建已补丁依赖与真实发布联调。旧NNSmith代码和patches保留用于依赖准备及迁移，公开执行只有当前统一流程。

CLI：`--fg`、`--perf`；run/replay默认不保存结果，`--save`才写结果及outbox。setup成功后记住数据目录。当前C协议只有一个shape，记录标记static；不把单shape标记dynamic。
