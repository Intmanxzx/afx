# Agent 与 API

本地 Agent 优先使用中心 API；机器内维护可以使用管理 CLI。管理 CLI 有统一 JSON envelope；中心 HTTP 接口返回各路由自己的 JSON，两者不是同一响应格式。

## 本地 Agent 使用中心 API

本地无需安装 afx。NPU 环境运行客户端，中心运行服务；Agent 直接访问 HTTPS，所有请求带 Authorization: Bearer 用户TOKEN，无需 Cookie 或 CSRF。复用管理员发放的用户 token，可操作该用户自己的设备；设备 token 不能调用这些接口。token 轮换或用户停用立即使旧凭据失效。

统一响应为 schema_version=1、ok 和 data；失败返回 error.code / error.message。所有任务都保存并上传结果，服务端会将 results 固定为 save=true、upload=true。

| 接口 | 输入 | 返回 |
|---|---|---|
| GET /api/agent/devices | 无 | 设备 id、online、seen、可用 environments |
| POST /api/agent/tasks | device_id、spec、request_key | 中心任务 id |
| GET /api/agent/tasks/ID | 无 | state、terminal、stale、results |
| GET /api/agent/tasks/ID/results | 可选 status、limit、after、attempt | 保存的 attempt 或单次详情 |
| POST /api/agent/tasks/ID/stop | request_key | 停止请求或取消状态 |
| POST /api/agent/tasks/ID/replay | seeds 数组、request_key | 新任务 id |

最小提交 body：

~~~json
{"device_id":"设备ID","request_key":"campaign-001","spec":{"schema_version":1,"kind":"fuzz_task","environment":"环境ID","workload":{"mode":"generate"},"termination":{"mode":"finite","max_cases":100}}}
~~~

设备/环境 ID 来自 devices；已有客户端尚未报告环境时先升级客户端。spec 与本地任务 YAML 使用同一 schema。后续始终使用返回的中心任务 id，不需要 operation ID 或本地 run ID。

轮询任务，直到 terminal=true；stale=true 表示尚无运行报告或报告超过 60 秒，不能推断远端当前仍正常。再等待 results.complete=true，表示中心接收的 attempt 数量与客户端终态报告一致；它不代表模型通过。结果状态 ok/fail/error/unsupported，分页继续使用 next_cursor，单次详情用 attempt 参数。空结果且 complete=false 不能视为成功，complete=true 且 expected=0 表示确实未保存有效 attempt（例如全部跳过）。

同一次写请求重试必须保持 request_key 和 body 不变，新任务用新 key。提交/重放/停止返回 HTTP 202，表示接收请求；停止后仍应轮询任务。提交已派发但尚无运行 ID 时，stop 返回 409，可用原 key 重试。HTTP 400 为输入错误、401 为凭据或归属错误、404 为资源不存在、409 为冲突、503 为暂不可用（按 Retry-After 退避）。

重放沿用原提交的环境、后端和评估规格，新的本地策略仍会校验资源；原任务须先释放设备所有权。结果查询按用户、设备和运行批次隔离，直接读取中心已入库数据。它与网站“最新完整 Shape”聚合口径不同。

## 本地管理契约

入口定义在 [management/cli.py](../../autofusex/fuzz/management/cli.py)。普通请求输出：

~~~json
{"schema_version":1,"ok":true,"data":{}}
{"schema_version":1,"ok":false,"error":{"code":"ValueError","message":"..."}}
~~~

--help、--version 和前台 daemon 生命周期入口不作为状态协议。--json 是兼容参数，普通管理命令已默认 JSON；afx fuzz 不承诺相同 envelope。

| 退出码 | 当前含义 |
|---|---|
| 0 | 请求成功或操作已接受；不代表 Run 完成 |
| 1 | 返回操作状态为 failed、cancelled、stopped、interrupted 或 needs_attention；envelope 仍可为 ok:true |
| 2 | 参数、校验或运行异常；等待超时也返回此码，原操作仍继续 |
| 3 | OSError / 连接类异常；可能是 daemon 不可用，也可能是文件错误 |

使用 data.id 追踪异步操作；终态为 succeeded / failed / cancelled / needs_attention，成功产物在 data.result。失败不能仅判断 ok，还要检查退出码与 data.state。

~~~sh
afx task submit task.yaml --request-key campaign-001
afx operation wait OP_ID --timeout 300
afx task wait RUN_ID
afx task results RUN_ID
afx operation logs OP_ID --after 0
~~~

OP_ID 来自 submit 的 data.id；RUN_ID 来自成功 operation 的 data.result.run_id。operation 成功表示提交工作完成，Run 仍可处于 preparing/running。超时后继续查询原 ID，不换 key 重新提交。

--request-key 对提交为 operation 的动作生效：同一 key、同一 payload 返回原操作；同一 key、不同 payload 冲突。daemon 启停、设备绑定和本地 trust 等直接动作虽接受公共参数，不能据此声称具有同样去重语义。

## 常用命令

| 命令（前缀 afx） | 返回或用途 |
|---|---|
| daemon start / status / stop | 本地进程状态；stop 要求无活动工作 |
| device enroll --token-stdin / device pair | 注册或浏览器配对；device status / revoke 查询或撤销 |
| environment list / environment status ENV_ID | 查询已登记环境 |
| environment plan FILE | 返回 operation；wait 后从 result.plan_id 取计划 ID |
| environment apply PLAN_ID | 返回 operation；wait 后取得环境 manifest |
| task submit FILE [--wait] | 返回 operation；成功 result.run_id 指向任务 |
| task validate FILE [--wait] | 校验规格并解析执行环境，返回 operation |
| task wait RUN_ID [--timeout SEC] | 等待 Run 完成或需处理；超时不取消 |
| task results RUN_ID | 本地保存结果，支持 --status、--seed、--limit、--after、--attempt |
| task replay SEED... --from-run RUN_ID [--wait] | 按原环境/后端/评估设置提交独立重放 |
| task list / task inspect RUN_ID | 查询任务与实际配置 |
| task pause / resume / stop / force-stop RUN_ID | 提交控制操作；查询 Run 确认实际状态 |
| task reconfigure RUN_ID FILE [--wait] | 连续任务在安全边界切换配置 |
| operation inspect / wait / logs / cancel OP_ID | 操作状态、等待、事件及取消请求 |
| config validate FILE / config effective | 校验客户端策略或查询当前策略 |
| config apply FILE [--wait] | 对后续任务应用策略 |
| build submit FILE / build inspect OP_ID | 使用本地受信任 profile 构建制品 |
| tool stage FILE / tool activate RELEASE_ID | 校验签名版本并在空闲时激活 |

配置 schema 在 [management/config.py](../../autofusex/fuzz/management/config.py)，支持 client_policy、environment、fuzz_task。未知字段、YAML 重复键/别名、超限内容会拒绝。任务默认有限生成 100 例、保存但不上传；资源还受 trust 和 policy 限制。FILE 参数统一支持 - 读取 stdin。

## 中心只读 API

路由定义在 [central/server.py](../../autofusex/fuzz/central/server.py)。以下接口不需要登录：

| GET 路由 | 用途及主要参数 |
|---|---|
| /api/cases | 当前完整 Shape 用例；seed、status、user_id、q、limit、after |
| /api/case-history | 用例历史；必填 seed、shape，可用 after、limit、snapshot |
| /api/attempts/ATTEMPT_ID | 执行详情；可指定 shape |
| /api/summary、/api/statistics | 当前用例汇总与分布；seed、status、user_id、q |
| /api/results | attempt 级记录；不要当作用例分母 |
| /api/compare?left=ID&right=ID | 两次 attempt 的可比性与差异 |
| /api/runtime | 运行快照；user_id、machine_id、model、backend、version、minutes |

读取返回的游标继续翻页；公共统计可能是带 stale 状态的缓存。503 可带 Retry-After，应退避重试。q 只支持受限 WHERE 表达式，不执行任意 SQL。

## 远程控制 API

远程操作白名单包含 `client.update`。payload 为 {"version":"0.1.4","package":"autofusex"}，可选 `index_url` 仅用于受控 PyPI 镜像。version 必须是精确版本，daemon 不执行无版本的升级，也不自行探测最新版本。

用户控制接口使用网页登录会话：POST /api/session 提交 {"token":"..."}，需要与站点 Host 一致的 HTTPS Origin；保留返回的会话 Cookie 和 CSRF token。后续写请求带 Origin、Cookie、X-CSRF-Token 与 Content-Type: application/json。管理 POST 必须有 Content-Length，body 最大 64 KiB。设备 Bearer 凭据不能替代用户控制凭据。

| 路由 | 方法 / body | 作用 |
|---|---|---|
| /api/manage/devices | GET | 当前用户设备 |
| /api/manage/schemas | GET | 配置 schema |
| /api/manage/validate | POST {"config": SPEC} | 纯配置校验，不解析本机执行环境 |
| /api/manage/submit | POST device_id、kind、payload、request_key | 异步远程操作 |
| /api/manage/operations | POST {"device_id": ID} | 设备操作与回执 |
| /api/manage/config | POST device_id、config、base_revision | 按修订号更新设备策略 |
| /api/retest/clients | GET | 当前用户复测客户端 |
| /api/retest/preview、/api/retest/batches | POST | 预览/提交复测；选择结构见 retests.py |
| /api/issues | GET，可选 dimension | 问题分组 |
| /api/issues/claim、/api/issues/release | POST {"issue_key": KEY} | 认领/释放问题 |
| /api/issues/retest | POST issue_key、client_id，可选 repetitions、request_key | 问题组复测 |

kind 使用本地同一操作白名单，例如 task.submit 的 payload 是任务规格，task.action 是 {"run_id": ID, "action":"stop"}。服务端拒绝跨用户设备操作，本地 trust 仍限制最终权限。详见 [management/central.py](../../autofusex/fuzz/management/central.py)、[central/control_api.py](../../autofusex/fuzz/central/control_api.py) 和 [retests.py](../../autofusex/fuzz/central/retests.py)。

设备使用 Bearer 调用 /api/manage/poll、ack、revoke；注册配对由 enroll/pair/exchange 完成。上传与复测领取有各自协议，交给客户端实现，外部 Agent 无需重写租约或 ACK 逻辑。

## 当前自动化边界

以下是当前实现限制，不能当作已经支持的接口：

- Agent API 目前复用用户 token，尚无独立细分权限的 API key、OpenAPI 或 CLI schema 导出入口。网页管理 API 继续使用 session/CSRF。
- task submit/replay --wait 等待提交；task wait 单独等待执行。停止/暂停操作成功表示控制请求已写入。completed 只表示调度完成，不表示所有模型通过，需读取 results。
- 托管 task schema 尚未覆盖直接 fuzz 的 --ops、method、dynamic-variants、dtype 等全部生成参数；需要时使用 afx fuzz，并遵守任务所有权限制。
- CLI 错误分类与中心错误体尚未统一。task results 查询本地已保存 attempt，available=false 表示结果库尚未生成，不能当作全部通过；未保存的执行无法补取结果。结果不等同于中心最新完整 Shape 投影。

后续统一托管与直接 fuzz 的生成参数，补充细分权限凭据及可发现的 API schema。

## 问题修复闭环：Agent 领取，SSH 调试，中心复测归档

这是修复工作入口，区别于前面的通用测试任务。所有路由以 /api/agent 为前缀，使用同一用户 Bearer token。Agent 在本地使用 npussh 调试和构建，不需要中心执行任意 shell。

| 方法 / 路由 | 输入 | 用途 |
|---|---|---|
| POST /repairs/claim | agent_id、request_key，可选 issue_key | 原子领取一个问题；无任务返回 data=null |
| GET /repairs/ID | 无 | 快照、状态、复测结果、总结与最近事件 |
| POST /repairs/ID/renew | lease_id、request_key | 延长 30 分钟租约 |
| POST /repairs/ID/release | lease_id、request_key | 交还处理权，保留问题与历史 |
| GET /repairs/clients | 无 | 可用于中心复测的当前用户客户端 |
| POST /repairs/ID/retest | lease_id、request_key、client_id、artifact | 对冻结的全部关联用例提交复测 |
| POST /repairs/ID/summary | lease_id、request_key、background、solution，可选 notes | 通过后提交经验并解决问题 |
| GET /experience | 无 | 最近 100 条当前用户归档经验 |

领取返回修复 id 和 lease_id。agent_id 区分同一用户的多个 Agent 实例；同一实例同时只持有一个活动问题。租约到期或 release 后其他 Agent 可接管同一修复记录，lease_id 换代，旧持有者的写入被拒绝。每次新操作用新 request_key，网络重试沿用原 key/body；空领取的重试仍返回空，下一次轮询用新 key。

状态为 claimed → retesting → fixing（失败继续修复）或 verified（已通过）→ resolved（总结归档）。复测状态在读取或下一次操作时根据中心事实更新。领取快照不随网站当前行变化而改写；重新认领保留历史复测。不同目标快照再次出现可产生新修复记录，相同已解决快照不重复派发。网页认领/释放/复测不能绕过 Agent 租约。

复测复用原有 /api/retest 客户端协议，client_id 是复测客户端 ID，不是 /devices 返回的托管设备 ID。需要已注册且在线、使用新 RUN 包环境的专用复测客户端。中心自动按 seed/配置合并多 Shape，所有批次创建与修复状态关联在同一个事务内；已存在在途复测时返回 409。前一轮完成后可提交下一轮。

artifact 格式：

~~~json
{"sha256":"64位小写十六进制摘要","version":"候选包版本","installed_at":"2026-11-15T12:00:00Z"}
~~~

该字段记录 Agent 的包安装证据，中心不会仅凭它证明实际加载了对应二进制。Agent 必须在远端核实包路径、动态库和版本。中心只以本轮合法租约绑定的结果作通过依据：所有任务完成且被接收、AF-MLIR 状态 ok、全部目标 Shape 存在并精度 pass；要求性能评估时还需性能成功。取消、中断、缺结果、缺 Shape、旧结果或失败均不能归档。

经验总结必须由当前租约持有者提交，且最新复测通过。每个修复记录只能有一份总结；相同请求返回同一份记录，新 key 重复归档返回 409。总结关联复测结果的 summary_id，原始执行事实不改写。可通过修复详情及 experience 查询背景、方案、备注和复测证据。

本地 skill：autofuse-repair，覆盖领取/续租 → npussh 调试 → 仓库原生构建 → 隔离安装 RUN 包 → 中心复测 → 经验归档。中心不提供任意构建命令或任意安装命令的远程执行接口。
