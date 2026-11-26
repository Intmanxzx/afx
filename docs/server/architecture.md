# 中心服务架构

中心服务位于 `autofusex/server/`，通过 FastAPI + Uvicorn 提供 HTTP API 和 Web。NPU 客户端负责执行；中心保存事实、派发任务、管理配置、统计结果和组织问题修复。整体边界见[平台架构](../architecture.md)。

## 数据与模块

| 数据库 | 内容 |
|---|---|
| `core.db` | 用户与 token 摘要、配置、用例与执行头、客户端、问题、复测、知识及固定源码任务 |
| `content.db` | 日志、图规格等去重内容 |
| `stats.db` | 可重建的统计聚合 |

`schema.py` 管理有序 schema 迁移；`store.py` 管理连接、写锁和内容存储；`ingest.py` 校验执行上传并推进执行头；`aggregates.py` 维护聚合，`queries.py` 提供列表、详情与总览。已有数据目录升级时保留身份和历史事实，不以空库替换，备份与恢复见[部署说明](../../deploy/platform/README.md)。

| 业务模块 | 职责 |
|---|---|
| `auth.py`、`accounts.py` | 用户、申请审核、token、角色与审计 |
| `ops.py`、`client_access.py` | 心跳、在线状态、升级与启停请求、维护白名单 |
| `task_config.py` | 管理员默认配置与个人配置 |
| `source_jobs.py` | 固定源码登记、任务派发、结果及报告 |
| `issues.py`、`retests.py`、`regression.py` | 问题归类、领取、验证、回归与状态推进 |
| `knowledge.py`、`profile.py` | 修复知识检索与能力画像 |
| `server.py`、`api_models.py` | 路由、鉴权、请求模型、限流和响应 |

## HTTP 与权限

公开读取允许匿名访问；个人信息、固定源码任务及写操作要求登录，管理员操作检查 admin 角色。新账号角色为 developer 或 admin。token 明文只在签发时展示，服务端使用私有密钥计算 HMAC 摘要，不存明文。申请与重置规则见[用户管理](../user_registration.md)。

字段、响应结构和路由以自动生成的 [OpenAPI](../openapi.md) 为准。业务层继续检查所有权、版本、租约与幂等性；并非所有写接口都使用同一种 request_key 机制。参数校验错误返回 400。

- `/api/client/*`：心跳、执行上传、复测领取及完成回执。
- `/api/config/*`、`/api/ops/*`：个人/默认配置与运行概览。
- `/api/cases`、`/api/executions/*`、`/api/overview/*`：执行查询、当前结果及统计。
- `/api/issues/*`、`/api/regression/*`、`/api/knowledge/*`：问题修复、回归集与知识。
- `/api/source-cases`、`/api/source-jobs*`、`/api/source-results`：固定源码、整网派发与结果。

维护白名单见[接入控制](../client_access.md)，在线规则见[客户端身份](../client/identity.md)。

## 固定源码与整网任务

`source_cases` 保存不可变源码包，`source_jobs` 保存任务与租约，`source_artifacts` 保存报告。用例身份为 `I:` 或 `M:` 加内容摘要，并带 revision。中心校验入口、路径、内容摘要与用户所有权。

本地 `case run --submit` 登记源码并提交任务；客户端通过 `/api/source-jobs/claim` 领取，写入本地持久化操作并执行，再提交 `/complete` 和制品。已有任务占用时排队；同机器可恢复原领取，租约不自动跨机器转移。`/api/source-results` 接收本地已完成任务的结果。

中心源码包上限 512 KiB，报告限 HTML、JSON、TXT、CSV、SVG 等允许类型，单文件上限 1 MiB。制品按摘要校验，原始大 tensor 留在执行机器。“导入与整网”页面提供列表、版本重跑和报告下载。

此流程与小图问题验证/回归队列分别存储；固定源码任务完成不意味着已参与小图统计或自动问题闭环。当前不提供独立 `/api/import/baize` 或 `/api/models/jobs` 接口。

## 执行事实与问题闭环

规则：

- **执行头**：只有 `fuzz` / `retest` / `import` 批次会成为当前结果。`verify` / `regress` 批次跑的是候选修复版本，只作为历史保留，不改变总览。迟到的旧执行（`created_at` 更早）只进历史。
- **上报幂等**：同一用户同一 `request_key` 必须是逐字节相同的请求体，相同则返回首次响应，不同则返回 409。单条记录按 `exec_uuid` 去重：内容相同返回 duplicate，内容不同返回 conflict。校验失败的单条记录返回 rejected，不影响同批其他记录。
- **聚合一致性**：core 与 stats 各记一个序号，每次写入同时推进。启动时两者不一致（例如两个文件提交之间崩溃）就从 core 全量重建 stats。修改持平阈值也会触发重建。
- **两两对比**：每对 backend 只存一份（字典序 x<y），反向查询时翻转结论与直方图。只有两边都功能通过、精度不失败、且有 kernel 耗时的用例才参与对比。
- **管道共享阶段**：接口按 backend 分别返回各阶段计数，并给出 `shared_stages` 配置；页面在同族 backend 计数一致时合并显示。按用例求同族交集需要单独的聚合，暂未实现。
- **用例特征**：服务端自动补充 `source`、`dynamic`、`size_bucket` 三个离散特征和 `node_count`、`max_numel`、`max_dim`、`shape_count` 四个数值特征。可分组统计的维度为 `op_category`、`dtype`、`size_bucket`、`dynamic`、`source`。
- **聚类粒度**：功能签名使用首错误日志标准化结果及 function_status/fail_stage，主算子类别不参与签名。精度和性能各自在自己的维度内按主算子类别聚类；主类别按 Reduce > Transpose > Concat > Split > Gather > Scatter > View > Reshape > Broadcast > Elementwise 取最高者。
- **何时归类**：只有成为当前结果（fuzz/retest/import）的失败执行会归入问题。verify/regress 的失败执行只记录签名，不归入问题。
- **重新打开**：已解决的问题只在新失败满足两个条件时重新打开：执行时间晚于解决时间；所在 backend 版本是修复版本，或不在解决前已知失败的版本里。修复合入前，旧版本持续失败是预期内的，不会反复打开问题。
- **回归失败不另建问题**：回归跑的是尚未合入的候选包，用它的失败建新问题会污染问题池。失败用例记在该轮回归结果中（`issue_cases.result=fail`），问题退回修复中。`caused_by` 字段保留，等候选包合入后再考虑使用。
- **租约**：问题租约默认 5 天，在下一次写请求时检查过期。过期后，处于修复中的问题回到未领取；处于验证中、回归中或归档中的问题保持原状态，只清空持有人，其他 Agent 可以按 issue_id 接管，旧租约的写请求会被拒绝。
- **轮次评估**：一轮的全部任务都结束后才评估。有 fail 时退回修复中（没有有效租约则回到未领取）；没有 fail 但有 invalid 时同样退回；全部 pass 时，验证轮自动进入回归轮，回归轮进入归档中。回归集为空时直接进入归档中。
- **知识**：回归通过后由租约持有者提交，生成 `verified / published` 条目，自动附带证据（问题、各轮结果、样例用例、修复版本）。`version_range` 默认为"≥ 修复版本"。领取问题时返回同 backend 的相关知识 Top-5。

客户端复测协议：

1. `POST /api/client/retests/claim`，带 `machine_id`、`npu_arch` 和已安装包 `installed: {backend: {version, sha256}}`。只会领到 arch、backend、版本、sha256 全部匹配的任务（每次最多 20 个）。返回内容包括建议的 `run_uuid`、`run_kind`（verify/regress）和需要执行的 backend 列表；历史非 NNSmith 用例可附带构图规格；固定源码任务使用 source API。
2. 执行后按普通方式上报（`run.kind` 为 verify 或 regress）。这些结果不会成为当前结果，也不进入总览统计。
3. `POST /api/client/retests/{task_id}/complete`，带 `lease_id`、`exec_uuids` 和执行时的 `installed`。服务端核对执行记录归属与用例，再判定任务结果：
   - **pass**：目标 backend 功能通过且精度不失败。性能问题还要求 kernel 耗时与锚点的比值 < `issue.perf_slowdown`。
   - **fail**：未达到上述条件。
   - **invalid**：上报的已安装包与待测版本不符，或缺少结果。

   invalid 不算修复失败；一轮里没有 fail、只有 invalid 时，问题退回修复中，需要重新提交验证。任务租约默认 60 分钟，超时重新排队，连续超时 3 次判为 invalid。

### 问题用例重新归类

普通运行的当前结果与已确认的复测结果使用同一个归类函数。仍属于原问题记 Fail；通过记 Pass；问题签名或维度变化时在原问题记 Skip，并在对应 F/A/P 问题登记 Fail。Skip 保留目标问题和历史记录，不使用表示旧图的 stale 标记。后续验证和领取复现集排除 Skip，用例总数保留历史涉及范围。

复测结果必须匹配本任务运行和当前用例版本，安装包校验通过后才归类。性能归类使用同次复测的基准耗时；缺少基准数据不清除已有性能问题。后台运行不会覆盖正在修复的问题用例结论。Skip 不计为原问题失败。


## 展示与查询

中心 Web 与本地 viewer 共用 `server/static`，按能力加载不同数据适配。中心页面包含总览、问题发现、修复闭环、知识库、能力画像、导入与整网、平台运维及用户管理。

用例查询使用游标和受限表达式，字段与运算符白名单校验后参数化执行，不接受任意 SQL。总览使用统计聚合，原始事实保留供重建。两两性能对比使用符合条件的 kernel 耗时，不能用首次编译执行墙钟时间代替。

服务端测试位于 [tests/platform](../../tests/platform/)，容量工具为 [bench_platform.py](../../tools/platform/bench_platform.py)。合成查询结果不代表真实 NPU 吞吐或 HTTP 并发能力。
