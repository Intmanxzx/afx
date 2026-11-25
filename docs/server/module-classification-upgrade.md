# 模块归因与升级方案

本改动为候选代码，未升级生产服务或客户端，未写入现网历史记录。

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
