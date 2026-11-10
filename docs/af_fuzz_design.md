# Fuzz 设计

本文件描述当前 `autofusedebugger/fuzz`，取代旧重构方案、开发计划、D0调查、命令草案和旧NNSmith fuzz文档。历史调查/测试日志可从Git历史查阅。命令见[使用说明](af_fuzz_usage.md)。

## 1. 模块与依赖

同一wheel、同一版本线提供NPU客户端、中心服务和个人电脑查看。Fuzz不依赖旧debugger或NNSmith，不在import时初始化NPU或修改依赖。

```
cli / scheduling -> generation -> contracts + tensors
                 -> execution -> backend adapter
                 -> storage -> sync -> central
                            -> reporting -> view / central HTTP
```

| 模块 | 职责及扩展点 |
|---|---|
| contracts、tensors | frozen CaseSpec/CaseBundle，规范字节、TensorSpec、执行与评估配置 |
| generation/v1、v2 | 固定生成协议；registry按seed版本分发，改规则必须新增版本 |
| backends | 延迟加载Torch，适配check/prepare/inputs/invoke/synchronize/compare；新增后端不能改变case |
| execution | 静态/动态共用shape循环；正确性和可选性能阶段；失败阶段、误差、计时 |
| scheduling | 持久任务、批次、worker监督、租约、超时/恢复、预算 |
| storage | 白名单规范化、事务保存、schema版本、快照、导入和查询 |
| sync、central | 本地outbox与ACK、token身份、幂等接收、私有备份 |
| reporting | 持久派生索引、shape级汇总、错误摘要、安全WHERE条件 |
| view/static | 同一套预构建网页，无Node运行依赖；公开网站与离线view复用 |

架构测试禁止生成/契约反向依赖执行与存储，禁止fuzz引用旧debugger/NNSmith。扩展必须补协议golden、adapter能力、存储白名单/迁移及查询测试，不增加第二套执行引擎。

## 2. Seed、图与数据

仅凭一个seed可在无历史文件/数据库/缓存的设备上重建图、广播轴、全部shape、dtype/layout、权重和输入字节。缓存只用于性能，不参与确定性承诺。设备不支持时记录unsupported，不改变用例或重抽参数。

- 内部数值：`(version - 1) * 2^128 + entropy`；规范JSON以十进制字符串保存，避免JS整数精度损失。
- 短表示：A(v1)/B(v2) + base36 entropy，补齐至少9位。新批次entropy限定在`36^9`范围，产生10字符seed；旧大数不截断，避免碰撞。未知版本拒绝执行。
- 随机字段由固定命名空间、seed、字段名、重试号的SHA256及无偏拒绝采样得到，不使用设备能力、时间、Python hash或库全局RNG。
- v1固定Add、ReLU/Neg/Mul、可选Sum；v2额外包含独立Transpose类别。当前生成域是有界小图，不是任意NNSmith算子全集。
- 三组确定输入shape、同一图和权重构成不可变CaseBundle；存储为tuple[bytes]，采用规范小端浮点编码。数据值是有界dyadic数。
- workload指纹覆盖规范spec及全部storage内容；oracle单独版本化。执行后端/设备/测量配置不进入case内容指纹。
- `--static-only`选择基础样本并使用静态编译；默认动态编译、依次运行全部样本。两者不修改原bundle。
- v1/v2实现及golden保留；以后任何影响图或字节的规则变化都必须新增版本，不能重定义已有seed。

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

已验证：v1/v2跨aarch64/x86_64黄金指纹、Linux NPU真实执行、幂等同步/快照、公开站点和离线浏览器；近期100seed=300shape真实数据用于UI验收（27Pass/273功能Failed，不代表性能验收）。

未验证/未完成：通用算子覆盖；当前CANN下Triton兼容；准确独立编译/kernel采集；纯CPU编译流水线/产物移交；Windows/macOS原生执行验收；GitCode远端真实发布与PyPI权限联调。不得用文档或mock通过冒充这些能力。

本次整理验收（2026-11-10，全部在 mlir 隔离目录执行）：原版本相关测试36通过；整理后fuzz/发布脚本及非fuzz HTML/CLI回归312通过；无Torch构建环境271通过、20项执行依赖测试明确跳过；当前shape页面浏览器验收通过。已删除旧NNSmith执行引擎、旧报告渲染器及只适用旧页面/旧引擎的测试探针，保留当前shape页面探针与协议golden。发布脚本使用mock验证创建、上传、hash幂等和凭据脱敏；不代表真实GitCode/PyPI发布已验证。

打包验收：固定 `SOURCE_DATE_EPOCH` 连续构建两次wheel的SHA256一致，`twine check`和隔离Git fixture的Tag/版本dry-run通过。无模型依赖的独立wheel环境已验证CLI；HTTP/静态资源检查由 `tools/fuzz_wheel_smoke.py` 自动执行，发布流水线包含此门禁。构建依赖固定直接版本，尚未提供全部传递依赖的hash锁，不能据此声称跨时间构建完全可复现。
