# 模块归因与升级方案

本改动为候选代码，未升级生产服务或客户端，未写入现网历史记录。

## Shape 局部异常与调用栈补充（分类器 v11，待部署）

服务端新增经确认的八类规则：Inductor remove_noop_ops 的 int_noop/decimals 参数错误、same_meta 的 SymInt.size 错误、AOT functional graph 断言、torch_npu 索引表达式校验失败、保留实际 Torch lowering/ir 栈的 pow/SymInt 签名错误、validate_lib 产物权限错误、autofused Kernel 初始化或执行错误、Autofuse AscendC 主机或设备编译异常。分别使用既有 inductor、environment、fused_kernel、af_common 分类，不新增细分标签。仅改变服务端分类器和存量补正工具；客户端、协议、采集版本、数据库 schema 和页面分类无需更新。

每个失败 Shape 必须同时包含具体异常和实际 traceback，并证明相同模块；已知模块冲突、缺失 Shape 异常、只有共享日志或生成源码时不触发。保留 JSON 字符串形式 diagnostics 的兼容。2026-11-25 17:13:59 只读快照的 9,895 条 unknown_retest 历史执行中，新规则命中 6,940 条（当前执行头 6,934 条），与逐条审核范围一致；这是离线预览，尚未写入现网。

历史记录不会因部署分类器自动刷新。独立一致快照或已停止写入的数据可用 tools/classification_local_failure.py DATA --batch local-failure-v11 --offline --preview manifest.json 预览；将 --preview 改为 --apply 可应用清单，使用 --rollback 可回滚同批。

预览只读打开 core/content，不初始化 Store 或执行迁移。应用仅处理 unknown_retest 的失败记录，逐条核对 UUID、旧分类和当前证据，在既有 classification_changes 中保存可回滚审计并重建统计。原始 detail/log、reported 字段、执行头和问题状态不改写；不执行通用全量刷新，不创建复测任务。

## 分类契约

| status | stage | 展示 |
|---|---|---|
| gen_fail | inductor | 构图失败 |
| ascgraph_fail | af_fronted | AscGraph构图失败 |
| compile_fail | schedule | Schedule 失败 |
| compile_fail | legacy_codegen | Legacy Codegen 失败 |
| compile_fail | mlir | MLIR 失败 |
| compile_fail | triton_ascend | TritonAscend 编译失败 |
| compile_fail | af_common | AF其他失败 |
| runtime_fail | aclnn | aclNN单算子执行失败 |
| runtime_fail | fused_kernel | 融合算子执行失败 |

未知模块保留 unknown。精度未检查不算精度通过。总览 results 为互斥计数，合计等于 total；图例切换只影响图形，不改变精度通过率分母。

客户端观察安装环境 _asc_codegen 的 compile 调用，保存代码对象到源行的映射，从真实异常链提取失败行，在 1600 字符截断之前保留 AFX_MODULE_V1 标记。不修改 Torch 安装源码。每次 codegen_kernel_def 单独捕获原生日志，避免不同融合图混用 MLIR 结果。MLIR kFailed 或 ERROR MLIR codegen: 为失败证据；kSucceeded 不算失败；kNotSupported 不证明 Legacy 失败。只有实际配置 mlir_codegen_config=none 且 codegen 调用失败才直接归 Legacy。其他情况未知。

目前未实现原生崩溃前持久化逐接口事件；进程段错误/超时缺少异常链时无法可靠定位。Triton 编译模块及融合 kernel 的自动精确识别尚需真实样本验证，不以测试后端名称推断模块。

## 环境验证

2026-11-24：mlir，加载 ~/env.sh 和 torch210_daily，检查两张 NPU；独立目录 /mnt/workspace/af-classification-probe-1124。真实 NPU 加法结果与 CPU 一致，AF 前端真实非法参数 TypeError 经外层包装后归为 ascgraph_fail/af_fronted。强制 MLIR 的加法运行看到 MLIR Codegen Result：kSucceeded。Schedule/Codegen 边界注入与真实模块错误应分开统计。现有 fuzz daemon/task 未停止。

afx：/srv/afx-classification-test-1124 独立代码、venv、临时测试数据库；生产位于 /srv/afx-platform。现网审计工具只读打开 core/content，不初始化 Store、不执行迁移、不修改身份或上传游标。

## 发布顺序

1. 固定候选版本与制品哈希，跑完客户端 NPU、服务端入库/查询和页面测试。发布前还需补齐真实 MLIR 失败验证，不能用注入替代。
2. 在另一台机器保存私有一致备份：SQLite backup 先 core 后只追加的 content，保存 secret.key、部署版本/配置；校验 integrity_check、哈希和恢复演练。禁止直接复制运行中的 SQLite 主文件。
3. 先升级服务端。新版本兼容旧 build_fail；新客户端的 gen_fail/ascgraph_fail 不能先发给旧服务端。服务端迁移不自动批量改历史结果。
4. 灰度升级一个客户端：等待现有任务安全结束，保留身份、token、运行数据、上传游标；新建小批任务验证新状态、去重重传及统计守恒，再扩大升级。
5. 历史重分类先在私有一致快照上运行 tools/classification_audit.py，输出版本、exec_id 示例、前后分类分布。当前脚本是预览工具，没有 apply 功能。
6. 实施历史迁移前另行实现带审计的批处理：保存 exec_id、旧/新 status/stage/owner/classifier_version，保留原始 detail/log、exec_uuid、content_hash；使用原始上传事实重新归因，不能以已经重分类的值循环推断。重复运行必须幂等，版本未变化的不重复处理。
7. 短暂暂停写入后应用已批准的迁移清单，重建 stats，校验每个 backend/arch/dim 的类别和等于 total，核验 execution_heads、case 数与执行数不变。问题聚类与标题不能悄悄重建：先列出受影响 issue，保留领取/修复/复测状态，并制定显式映射后再更新。
8. 恢复上传并抽查 API、总览与用例详情，验证新旧客户端并存、重复上传仍幂等、增量统计与全量重建一致。

## 存量策略与回滚

有明确原始证据才细分；旧 graph_check 不自动当成 af_fronted，因为校验可能发生在 Schedule 内。只有 first_call_combined 或被截断日志时保持未知。需要精确定位的存量用例通过新客户端复测产生新执行，不伪造旧执行证据。

回滚优先回滚客户端，并保留兼容新状态的服务端。若必须回旧服务端，先停止新客户端上传，并恢复一致的数据库与匹配版本；不能只换旧二进制让其处理新枚举。历史重分类可通过审计清单恢复分类并重建 stats，不能删除新产生的执行。发布前需完成 apply/rollback 演练。

## 后续验证记录

真实空 FusedGraph 经 schedule 报错，标记为 compile_fail/schedule；BitwiseAnd、Sin、Cos 等生成图在 infer_dtypes 报错，准确标记 ascgraph_fail/af_fronted。Legacy 明确配置 none 的 Div/Pow/FloorDiv/Maximum 编译通过。多种浮点算子通过 MLIR，采集到 kSucceeded，但未获得可重现的 MLIR Codegen 失败样本。

现网 seed F0SQR0HF7D3QEYXA8AE8QFGMQ8N 的历史错误是 arith.index_cast 到 ui8；重放先因测试命令覆盖 PYTHONPATH 缺失 compile_adapter 被拒，保留原环境 PYTHONPATH 后遭设备租约拒绝。未停止现有任务、未绕过租约。该重放不计为后端失败验证。

afx 最终针对上传/查询/HTTP 分类的测试为 66 passed、1 skipped（环境没有 Node，JS 测试在本地执行）。本地真实 Chrome 验证桌面/移动端总览，无页面异常；切换 aclNN 图例后精度通过率保持 14.3%。截图在 artifacts/module-classification。一次只读审计扫描 147778 条记录，31202 条候选变化；这是特定时点预览，生产仍在上传，执行正式迁移前必须基于一致快照重新评估。

## 上传协议与本次停机升级（替代前述灰度步骤）

请求 protocol_version=2 表示传输协议；缺省为 1。执行 capture_version=2 由执行时写入本地 execution context，上传旧记录时缺省仍为 1；不能用客户端包版本推断。classifier_version 独立表示服务端派生归因版本。GET /api/client/protocol 返回支持集合；新客户端在上传前检查，不支持或旧服务器无接口时保留上传队列，不伪装降级。未知请求版本返回 426，非法类型拒绝；未知 capture 逐项拒绝。请求幂等键加 p2 前缀，执行 UUID 和内容哈希继续检测重复与冲突。

迁移 18 保存 upload_protocol_version、capture_version、reported_status、reported_stage 和完整原始 item 的 content 引用。历史默认两个版本均为 1，reported 字段为空，表示原始声明未保存，不能把旧服务端分类冒充原始声明。原始 item 内容按既有压缩/摘要机制保存，可能增加 content.db 大小；备份及保留策略应包含这些内容。

本次允许停全部客户端，按以下顺序执行：

1. 停止领取新任务，让在途任务结束并完成旧队列上传；记录每台客户端运行、上传与版本状态，再停止 daemon/worker。保留客户端数据目录和身份。
2. 停止服务端写入，制作私有一致备份并验证。保留新旧 wheel、配置及 secret；复制快照进行升级演练。
3. 部署新服务端制品，在离线数据上完成 schema 升级。使用 tools/classification_refresh.py DATA --batch BATCH --offline 预览；加 --apply 应用，加 --rollback 回滚同批。全批在事务中完成，重建统计；二次刷新跳过同 classifier_version。工具不会修改 capture_version、UUID、content_hash 或原始证据。
4. 核对预览、执行头与分类统计。classification_changes 保存逐条旧/新值；回滚前检查当前值仍等于本批写入值，避免覆盖后续修改。问题聚类/标题仍需单独评估，刷新工具不会自动重写问题领取和修复状态。
5. 启动新服务端，核对协议接口、API 和页面。升级全部客户端，确认新包及执行环境，再启动 daemon。未上传旧结果继续以 capture_version=1 补传，新执行为 2。
6. 抽查新上报、重复重试、精度与模块统计。失败时先停止客户端，按审计回滚分类或恢复一致备份与匹配服务端版本；禁止旧服务端直接接收协议 2。

这些步骤是操作方案，当前未停止生产客户端、未升级生产服务、未应用现网刷新。

## 2026-11-25 mlir 上传协议实测

隔离目录 /mnt/workspace/afx-protocol-live-1125，加载 ~/env.sh、torch210_daily。两张 NPU 健康且有现有 worker；NPU 测试通过 DeviceLease 等待获得 npu:0 租约，没有停止已有任务。

真实 NNSmith seed E0F89WAF2DT1E0LH3Y63FIZT6AY 分别执行 CPU 和 AF-MLIR；NPU 日志为 AF_MLIR_STRICT gate=none shortfall=no graph=autofused_amin_c392804f41aafbb6ff0b72850e77bdd5。本次 correctness=False，验证执行和上传，不声称精度验证。执行结果 ok 后通过 ResultRepository → Mapper → upload_new → 实际 HTTP → 隔离 PlatformServer 入库。capture_version=2 在本地保存并上传；移除 capture_version 的同结果副本模拟旧采集，补传后仍为 1，两者均采用协议 2。模拟仅支持协议 1 的服务端时，游标保持 0、rejected 表为空；随后上传成功，重复扫描为空。

mlir Python 原有 SQLite 3.31.1 不支持服务端既有 trigram tokenizer；单独编译 SQLite 3.50.4，仅通过测试进程 LD_PRELOAD 使用，没有替换系统或客户端依赖。测试用例对旧 Node 10 使用 assert.strict 兼容入口；协议与归因测试 18 passed。隔离服务仅绑定 loopback，结束后关闭。CPU/NPU 报告和复现脚本保存在 artifacts/module-classification/protocol-1125。

### 存量证据不足的待复测标记

旧采集版本（capture_version < 2）的失败记录，如果当前 shape 的异常和实际失败调用栈不能证明模块，使用 fail_stage=unknown_retest，页面显示“未知·待复测”，位于“其他”组且默认隐藏。保留 function_status、原始日志、UUID、capture_version 和分类算法版本；该标记是离线审计决定，不是客户端新故障类型。精度未检查、超时、不支持和新采集记录不进入该批。

已证明的模块异常、运行前环境校验、平台输出签名校验等不标记；共享日志和生成代码中的接口名称不能单独证明当前 shape 的失败模块。无法确定 Codegen 分支的 marker 仍需要复测。

停止写入并备份后，用 `PYTHONPATH=. python tools/classification_retest.py DATA --batch BATCH --offline --preview manifest.json` 生成逐条修改清单和同名 CSV（仅最新执行头，含 case_key/backend/npu_arch/exec_uuid/reason）。复核清单后，使用 `--apply manifest.json` 应用；使用 `--rollback` 回滚同批。应用会检查清单是否过期，并记录 classification_changes。脚本不会创建复测任务。

页面搜索 `stage:unknown_retest`，可加 `backend:af-mlir` 或 `arch:...`，通过已有批量复测入口重新运行。新执行正常参与 execution_heads 选择后，总览不再使用旧执行的待复测标记；历史记录仍保留审计信息。若升级后客户端重新上传的仍是旧采集记录，不能视为已完成新版本复测。

历史 strict gate 若带有 graph 名称，且与当前 shape 的“Codegen generate kernel failed”中的图名一致，可以作为模块证据；其他图的 gate 不参与判定。已知的 Inductor 特征异常和安装权限错误也不归入待复测。

2026-11-25 已应用批次 historical-unknown-retest-20261125：标记 23,277 条历史执行，其中当前执行头 23,263 条（AF-MLIR 11,892、AF-legacy 11,289、TritonNPU 82）。原始事实、用例、执行头和问题状态校验保持一致；副本完成应用/回滚/重新应用演练。线上服务恢复后，公开矩阵计数、筛选接口和静态文件哈希已核对。备份、清单、CSV 和校验报告保存在服务端 releases/0.5.0/unknown-retest-20261125 下，未创建复测任务。

### aclNN shape 局部证据补正（分类器 v3）

CANN plog 摘要可能只有 OpName:[aclnnMedian_5]，实际 Python 异常保存在 contexts.execution.diagnostics.shapes 或 shape_results 中。服务端先读取失败 shape 的局部异常；只有每个失败 shape 都明确指向 aclNN，且没有相冲突的模块 marker，才归为 runtime_fail/aclnn。共享日志不能单独触发该补正规则；不完整、多模块失败不强行合并。分类器版本提升为 3，上传协议版本和采集版本保持独立。

reclassification.aclnn_plan 只预览满足该规则、尚未归 aclNN 的失败记录；apply_aclnn 校验原值和局部证据并保存逐条修改审计。该补正不执行通用全量刷新，不改变原始日志和待复测集合。

2026-11-25 已应用 aclnn-shape-local-20261125：补正 4,762 条当前执行（AF-MLIR 2,123、AF-legacy 2,603、TritonNPU 36）。用例 E0SQR0WLFHGE5JG59LC7LXZFXXM 的两条 AF 执行均已归 runtime_fail/aclnn；公开接口复核通过。待复测 23,263 个当前结果保持不变。相关测试 56 passed，副本回滚演练、数据库完整性、统计和原始事实校验通过。服务端备份与审计清单位于 releases/0.5.0/aclnn-local-20261125。

### AF前端来源证据（分类器 v6）

仅使用 pyascir 来源证据识别 AF前端实际构图错误，归 ascgraph_fail/af_fronted；HintGraph 关键词不参与判定。CANN ERROR 的来源不限于 pyascir_types.cpp，也识别 pyascir.cpp、pyascir_common_utils.cpp 等 pyascir 文件；根据错误载荷与每个失败 shape 的终端异常匹配，避免会话共享日志误归。局部终端异常明确指向 pyascir 也属于前端；仅生成代码中出现 ascir.HintGraph 或文件名不算证据，冲突的 Schedule/Codegen/aclNN marker 优先保留。明确来源证据可解除 unknown_retest 标记；历史待复测生成器也排除已证明的前端错误。

2026-11-25 检索升级前完整快照的全部 177,621 条执行：20,047 条执行的 CANN ERROR 涉及 pyascir，当前执行头 20,035 条；来源均为 pyascir_types.cpp，函数包含 DoInference、InferDtype、InferDataType。未发现 HintGraph 实际错误行。前端候选清单 20,026 条（当前执行头 20,014 条），包含 F0SQR0WLFHGE5JG59LC7LOVP5SV 的两条 AF 执行。该候选批次随后已随 v5 上线，实际应用结果见下文。

### NNSmith 输出签名契约与命名（分类器 v5）

终端 ValueError: materialized Torch graph differs from declared NNSmith output signature 归 gen_fail/nnsmith，页面显示“构图失败 · NNSmith”。该归类表示用例声明与实际生成/执行输出的契约不一致，不单凭该消息断言 NNSmith 库自身存在缺陷。读取每个失败 shape 的终端异常；共享日志、生成代码中的字符串或混合模块异常不作为整条执行的 NNSmith 归因。按 nnsmith_plan/apply_nnsmith 预览和应用专用批次，保留原始证据及回滚审计。总览新增 NNSmith 独立分类，避免 gen_fail 被统一显示为 Inductor。

页面 TritonAscend 统一显示 TritonNPU；保留既有 triton_ascend 分类码，避免改动数据协议。分类器 v5 包含此前尚未上线的 AF前端来源规则，客户端协议和 capture_version 不变。

分类器 v6 移除 HintGraph 关键词匹配，仅保留 pyascir 来源和终端异常证据；新增 HintGraph 独立错误不触发 AF前端规则的回归验证。此次升级不改变上传协议、capture_version 或客户端代码；已正确分类的历史记录保留原审计版本，不重复刷新。

### TritonNPU 编译归因（分类器 v7）

仅对 triton-npu 的失败执行判定 compile_fail/triton_ascend，显示“编译失败 · TritonNPU”。每个失败 shape 都必须具有以下证据之一：终端 triton.compiler.errors.CompilationError/MLIRCompilationError；dtype_guard.py 调用栈加实际 operator 不支持 dtype 及 supported dtypes on Atlas 消息；Triton 原生 ERROR 块或编译调用栈加 BiShengHIR/BiShengIR 流水线失败、BiShengLIR 编译失败、二进制重定位失败或 external hivmc 编译失败。异常头被截断时允许使用 ERROR 块的 END 边界，但不能仅凭编译器路径、MLIR/hivm 关键词、生成代码或共享日志归因。

局部模块 marker、aclNN、NNSmith 输出契约及明确环境故障冲突时不使用该规则；混合或缺失 shape 证据保留未知。明确证据先于 unknown_retest 保留逻辑；历史 triage 也排除已证明的 TritonNPU 编译失败。triton_plan/apply_triton 仅补正 TritonNPU 中 unknown/unknown_retest/inductor 或空阶段的失败记录，逐条核对原值和证据并记录可回滚审计。协议、capture_version、客户端及原始日志保持不变。

2026-11-25 完整读取线上 561 条 TritonNPU 失败执行（均为当前执行头）：244 条各失败 shape 有明确编译异常，另 31 条由原生编译错误补足，合计 275 条候选，原分类均为 runtime_fail/unknown。类型支持日志涉及 97 条，其中 96 条已被上述候选覆盖；另 1 条仍有未证明的 shape，不强行归因。异常只有共享日志、含路径的栈尾或外层 InductorError/SubprocException 时不充分。

2026-11-25 已部署 v7，应用 triton-compile-20261125-v7 批次 275 条（全部为当前执行头）。副本完成应用、回滚、重新应用；生产逐条清单、数据库完整性、原始事实及问题状态哈希、互斥统计校验通过。公开接口确认上述两个示例均为 compile_fail/triton_ascend，运行未知从 277 降至 2；TritonNPU 待复测仍为 82，全后端合计 3,249。平台全量测试 171 passed，最后的应用范围校验修改另经 19 项专项测试通过。备份、manifest、演练和生产报告位于 releases/0.5.0/triton-attribution-v7-20261125。

### 原生运行错误与 AICPU（分类器 v8）

新增 runtime_fail/aicpu，页面显示“运行失败 · AICPU”，运行失败组默认不选中，搜索支持 stage:aicpu。aclNN 单算子证据优先于 AICPU 概括证据。每个失败 shape 明确指向 AICPU 执行失败时可归因；没有 shape 结果、error_stage=worker 的失败，使用采集器已按执行 PID/时间窗截取的 plog_errors，再核对唯一失败 NPU worker 的 PID。OP ERROR 中非零 errno 且 OpName 为 aclnn 算子，归 aclnn；否则 RUNTIME ERROR 明确报告 aicpu execute failed、AICPU execution exception 或 AI CPU kernel execution failed 等，归 aicpu。仅出现 AICPU 任务类型、同步 API 名、错误码、INFO 或其他 PID 日志均不足以归因。

### 错误日志与 ASCIR 展示

问题发现摘要从已保存的 Shape 异常、运行异常、Triton/AF 原生日志、CANN plog 和上报摘要提取具体错误，跳过外层 Inductor 包装；详情统一为“错误日志”，按来源保留完整已采集文本。读取时派生，不回写原始日志或分类，不需要刷新存量数据。旧客户端继续兼容。

ASCIR 不再使用 diagnostic_view.shape_log（这是异常文本）。客户端在已有 Autofuse compile 观察点采集实际生成的 ASCIR 构图源码，按 Shape 独立保存于 execution.ascir_sources，单段 64 KiB、总计 128 KiB、最多 16 段；排除调度注释和 schedule/codegen 调用，不执行源码、不启用 dump。该可选字段沿既有 detail/context 上传，协议与 capture_version=2 的模块证据语义不变。现有服务端能保存可选 detail，新服务端识别并展示；要采集新图需要升级执行客户端。旧日志仅在完整构图段保留时提取，截断文本不伪造图。缓存命中或未进入 AF 时可能没有采集，不强制清缓存。此源码表示 Schedule 之前的构图程序，不是 Schedule 后的图快照。

不依赖 507018、rtDeviceSynchronizeWithTimeout 或某条完整异常模板。明确的执行错误先于旧 timeout/unknown_retest 标签；成功、不支持和已有冲突模块证据不覆盖。native_runtime_plan/apply_native_runtime 逐条预览、复核和审计，只补正未明确归因的失败，保留原始 status/stage 上报事实及日志。旧分类器 v2 的两个示例没有 reported_status/raw_upload，无法反推最初误判发生在哪一步；现有客户端复现不会仅因 WithTimeout API 名归 timeout，服务端旧 timeout 保留和历史补正排除 timeout 才是当前未修正的直接原因。

2026-11-25 已部署分类器 v8，并应用 native-runtime-20261125-v8：8 条旧 timeout/unknown 当前执行中，2 条改为 runtime_fail/aclnn、6 条改为 runtime_fail/aicpu。完整未归因失败扫描没有新增候选；83 条超时剩余 75 条。F0SQR0WLFHGE5JG59LC7MFU8PDR 的两条 AF 执行归 aclnn，F0SQR0WLFHGE5JG59LC7LWBNEGX 的两条 AF 执行归 aicpu。副本应用/回滚/重新应用、生产数据库完整性、原始事实与问题状态哈希、互斥总览计数通过。平台测试 187 passed；公开接口、stage:aicpu 搜索、静态文件哈希和服务状态均核对，AICPU 默认隐藏。备份、manifest 和审计报告位于 releases/0.5.0/native-runtime-v8-20261125。

2026-11-25 已部署 v5，并分别应用 frontend-20261125-v5（20,026 条、当前执行头 20,014）与 nnsmith-20261125-v5（395 条）批次。副本应用/回滚/重新应用、数据库完整性与原始事实校验通过；线上服务恢复后目标用例、总览互斥计数、筛选和静态文件均已核对。此前 AF前端 v4 待部署规则已随本次上线。当前待复测结果剩余 3,249 条（AF-MLIR 1,837、AF-legacy 1,330、TritonNPU 82），已重新导出 remaining-retest.csv。备份和审计位于 releases/0.5.0/frontend-nnsmith-20261125；相关测试 62 passed。

## AFIR 原生诊断归因（分类器 v12）

MLIR 源码 dialect/AFIR/IR/AFIR.h 声明稳定诊断分类；LoopGenerate.cpp 使用 emitError 输出 InvalidSchedule: 'afir.<op>' <原因>，AFIR verifier 使用 emitOpError 输出 'afir.<op>' op <原因>。例如用例 E0SQR0WLFHGE5JG59LC7N9891FQ 的 af-legacy 执行 d84eed55-bfbc-4a49-8fe6-68c40df25f60 在 worker 异常中记录 InvalidSchedule: 'afir.load' result escapes its schedule scope，实际属于 compile_fail/mlir。

新增规则仅识别完整原生 error 行（支持 loc 和文件行列位置），要求每个失败 shape 都有证据；没有 shape 结果时，可读取终端异常及 error_stage=worker 的执行诊断。共享日志、IR 文本、源码引用、只有 afir.xxx 或 InvalidSchedule 关键词不足以归因；明确的其他模块、运行环境或运行错误证据冲突时保留原有判定。规则适用于两条 AF 后端，后端配置名称不能替代实际报错来源。

classification_mlir.py 支持只读清单预览、离线应用和回滚，只修正原阶段为空、unknown、unknown_retest 的 AF 失败记录；应用时复核范围、原值和诊断。保持执行 UUID、原始上报、客户端协议、capture_version、用例和问题状态，重建派生统计。

2026-11-25：当前线上主机 afx 的 /srv/afx-platform 部署分类器 v12，应用 afir-diagnostics-20261125-v12 批次 15 条，全部为当前执行头（af-legacy 7、af-mlir 8）。指定执行已由 runtime_fail/unknown_retest 改为 compile_fail/mlir，公开执行与总览接口复核通过。平台测试 356 passed；私有完整备份、停写最终快照、副本应用/回滚/重新应用演练通过；生产三库 integrity_check、原始事实哈希及 191,249 个执行头互斥统计校验通过。备份、manifest 和报告保存在 /srv/afx-platform/private/afir-v12-20261125，原 v11 部署目录保留。
