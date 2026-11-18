# 平台使用说明与 Agent API

两种角色：**NPU 客户端**执行用例并上报数据（自动泛化测试、导入用例）；**Agent** 领取失败问题、调试修复、提交复测归档。客户端默认交给 daemon 托管，中心统一下发任务、管理环境；Agent 不需要装 afx，直接调用中心 HTTP API。

## 快速开始

### 一、NPU 客户端：执行用例并上报数据（自动泛化测试、导入用例）

由中心统一下发任务、管理环境、支持远程一键更新，完整步骤见[部署说明](../../deploy/fuzz/README.md)和[运维说明：托管客户端首次准备](operations.md#托管客户端首次准备)，摘要：

~~~sh
pip download autofusex --no-deps -d /packages   # install_client.sh 要本地 wheel 文件，不接受包名
bash deploy/fuzz/scripts/install_client.sh \
  --wheel /packages/autofusex-*.whl --cann-root "$HOME/Ascend/cann" --home "$HOME/af-fuzz" \
  --torch-wheel /packages/torch.whl --torch-npu-wheel /packages/torch_npu.whl

afx daemon start
afx management trust trust.json                 # 本机 trust：允许的设备/资源、CANN 路径、wheel 摘要
afx device enroll --token-stdin                  # 或 afx device pair 后网页批准
afx environment plan environment.yaml --request-key env-plan-001
afx operation wait PLAN_OPERATION_ID
afx environment apply PLAN_ID --request-key env-apply-001
~~~

环境 ready 后提交任务，见下文"方式二"。

不需要托管、只是本机临时调试时，可以跳过上面整套流程，直接：

~~~sh
python -m pip install 'autofusex[execution]'
# 再按架构装匹配的 Torch 2.10.0+cpu、torch_npu 和 CANN toolkit（无通用 pip 包，手工装或用上面的部署脚本）
afx fuzz maintenance doctor        # 校验环境
afx fuzz setup --token TOKEN       # 要上传结果到中心才需要，TOKEN 由管理员发放
afx fuzz run --count 100 --save
afx fuzz status
~~~

完整参数见下文"方式三"。

### 二、Agent：处理问题（修复闭环）

Agent 不需要装 afx，直接用中心发放的用户 Bearer token 调 `/api/agent/repairs/*`：领取 → 本机 npussh 调试/构建 → 安装候选包 → 提交复测 → 通过后提交总结归档。完整路由、状态机见下文"问题修复闭环"一节；也可以直接用封装好的本地 skill：

~~~sh
cp -r skills/afx-repair ~/.codex/skills/
~~~

之后对话里用 `$afx-repair` 调用，自动走"领取/续租 → npussh 调试 → 仓库原生构建 → 隔离安装 RUN 包 → 中心复测 → 经验归档"全流程。

## 安装与准备

autofusex 已发布到 PyPI，直接 pip 安装即可，无需本地 wheel 文件；不指定版本号默认安装最新发布版本：

~~~sh
python -m pip install autofusex
~~~

需要固定到某个历史版本时加 `==VERSION`。执行端（NPU 客户端）还需匹配的 Torch、torch_npu、CANN 和固定版本的 `nnsmith-autofusex`（随 `[execution]` extras 一并从 PyPI 安装），按[部署说明](../../deploy/fuzz/README.md)用部署脚本安装；中心和本地查看器只需上面的基础安装。源码安装执行依赖：

~~~sh
python -m pip install -e '.[execution]'
~~~

数据目录用 AUTOFUSEX_HOME 或命令末尾的 `--data-dir DIR` 指定。管理 CLI 需要 daemon 已启动、执行环境已登记为 ready、设备已获本地授权；首次准备见[运维说明](operations.md)。

## 方式一：中心 API（外部 Agent，推荐）

无需安装 afx。Agent 直接访问中心 HTTPS，所有请求带 `Authorization: Bearer 用户TOKEN`（复用管理员发放的用户 token，只能操作该用户自己的设备；设备 token 不能调用这些接口）。统一响应 `{"schema_version":1,"ok":…,"data"|"error":…}`；服务端固定 `results.save=true、upload=true`。

| 接口 | 输入 | 返回 |
|---|---|---|
| GET /api/agent/devices | 无 | 设备 id、online、seen、可用 environments |
| POST /api/agent/tasks | device_id、spec、request_key | 中心任务 id |
| GET /api/agent/tasks/ID | 无 | state、terminal、stale、results |
| GET /api/agent/tasks/ID/results | 可选 status、limit、after、attempt | 保存的 attempt 或单次详情 |
| POST /api/agent/tasks/ID/stop | request_key | 停止请求或取消状态 |
| POST /api/agent/tasks/ID/replay | seeds 数组、request_key | 新任务 id |

最小提交 body（spec 与本地任务 YAML 是同一 schema）：

~~~json
{"device_id":"设备ID","request_key":"campaign-001","spec":{"schema_version":1,"kind":"fuzz_task","environment":"环境ID","workload":{"mode":"generate"},"termination":{"mode":"finite","max_cases":100}}}
~~~

device_id/环境 ID 来自 devices；提交后用返回的中心任务 id 轮询，直到 `terminal=true`（`stale=true` 表示超过 60 秒无运行报告，不能推断远端仍正常）。再等 `results.complete=true`（表示中心接收数量与客户端终态一致，不代表模型通过）；结果状态为 ok/fail/error/unsupported，分页用 `next_cursor`，单次详情用 `attempt` 参数。重放沿用原提交的环境/后端/评估规格，原任务须先释放设备所有权。

写请求重试必须保持 `request_key` 和 body 不变，新任务用新 key；提交/重放/停止返回 HTTP 202，表示接收请求，停止后仍应轮询任务。400 为输入错误、401 为凭据或归属错误、404 为资源不存在、409 为冲突（提交已派发但尚无运行 ID 时 stop 返回 409，可用原 key 重试）、503 为暂不可用（按 Retry-After 退避）。

## 方式二：本地管理 CLI（afx，NPU 客户端机器）

入口定义在 [management/cli.py](../../autofusex/fuzz/daemon/management/cli.py)。普通请求输出统一 envelope（`--help`/`--version`/daemon 生命周期命令不计入此协议）：

~~~json
{"schema_version":1,"ok":true,"data":{}}
{"schema_version":1,"ok":false,"error":{"code":"ValueError","message":"..."}}
~~~

退出码：0 成功或已接受（不代表 Run 完成）；1 操作终态为 failed/cancelled/stopped/interrupted/needs_attention（envelope 仍可为 ok:true）；2 参数/校验错误或等待超时（原操作仍继续）；3 OSError/连接异常（daemon 不可用或文件错误）。用 `data.id` 追踪异步操作，终态 succeeded/failed/cancelled/needs_attention，成功产物在 `data.result`。

**Operation 和 Run 是两种对象**：提交、暂停、安装各有操作状态；任务提交操作成功后得到 run_id，再查询 Run 的执行状态。

先查询可用环境，填入 task.yaml 的 environment：

~~~sh
afx daemon start
afx environment list
~~~

~~~yaml
schema_version: 1
kind: fuzz_task
environment: npu-env
workload:
  mode: generate
termination:
  mode: finite
  max_cases: 100
results:
  save: true
  upload: true
~~~

~~~sh
afx task submit task.yaml --request-key campaign-001 --wait
afx task wait RUN_ID
afx task results RUN_ID
afx task stop RUN_ID --wait --request-key stop-campaign-001
afx task replay SEED --from-run RUN_ID --wait --request-key replay-001
~~~

`submit --wait` 等待的是提交操作，`task wait` 单独等待执行；超时不取消任务，可用同一 ID 继续等待，重试同一操作沿用 request key，新操作用新 key。`upload: true` 要求设备已绑定，仅本地保存可设为 false；省略 execution 时使用 CPU + AF-MLIR、npu:0 和 staged 调度。重放沿用原环境/后端/资源/评估配置，原任务须先释放所有权，不覆盖旧记录。FILE 参数统一支持 `-` 读取 stdin；可先 `afx task validate task.yaml --wait` 校验环境与规格。

常用命令：

| 命令（前缀 afx） | 返回或用途 |
|---|---|
| daemon start / status / stop | 本地进程状态；stop 要求无活动工作 |
| device enroll --token-stdin / device pair | 注册或浏览器配对；device status / revoke 查询或撤销 |
| environment plan FILE / apply PLAN_ID | 返回 operation；wait 后从 result.plan_id / 环境 manifest 取值 |
| task submit/validate FILE [--wait] | 提交或校验规格，返回 operation |
| task wait RUN_ID [--timeout SEC] / results RUN_ID | 等待执行；查询结果（--status、--seed、--limit、--after、--attempt） |
| task replay SEED... --from-run RUN_ID [--wait] | 按原环境/后端/评估设置提交独立重放 |
| task list/inspect/pause/resume/stop/force-stop/reconfigure RUN_ID | 查询与控制 Run；reconfigure 在安全边界切换配置 |
| operation inspect/wait/logs/cancel OP_ID | 操作状态、等待、事件、取消 |
| config validate/apply FILE / config effective | 校验或应用客户端策略 |
| build submit FILE / tool stage FILE / tool activate RELEASE_ID | 本地受信构建与签名制品升级 |

配置 schema 见 [management/config.py](../../autofusex/fuzz/daemon/management/config.py)，支持 client_policy/environment/fuzz_task；未知字段、YAML 重复键/别名、超限内容会拒绝。任务默认有限生成 100 例、保存但不上传，资源还受 trust 和 policy 限制。

## 方式三：临时运行与重放（afx fuzz，本机直接调试）

在已准备好的执行环境内运行，不经过 daemon 的任务队列：

~~~sh
afx fuzz run --count 100 --save
afx fuzz status
afx fuzz stop
afx fuzz replay SEED --save
~~~

将 SEED 替换为结果中的当前 E/F seed。不传 count 持续生成，`--fg` 前台运行，`--save` 才归档结果；已配置同步时保存结果进入上传队列。首次上传配置用 `afx fuzz setup --token TOKEN`（TOKEN 由管理员发放，与托管设备绑定是不同入口）。

| 目的 | 命令 |
|---|---|
| 单后端重放 | afx fuzz replay SEED --backend af-mlir --save |
| 人工 eager-npu 复核 | afx fuzz replay SEED --backend eager-npu --save |
| 过滤算子 | afx fuzz run --ops add,ge,sum,reshape --save |
| 双卡指定资源 | afx fuzz run --device npu:0 --device npu:1 --workers 4 --compile-threads 1 --save |
| 查看同步 | afx fuzz maintenance sync status |
| 检查环境 | afx fuzz maintenance doctor |

run --backend 增加被测后端，replay --backend 只执行指定后端；不要与同一 home 的活动托管任务同时运行。

## 查询结果

中心提供[网页](https://autofuse.online)和只读 API：

~~~sh
curl --fail-with-body 'https://autofuse.online/api/cases?limit=10'
curl --fail-with-body 'https://autofuse.online/api/summary'
~~~

本地结果可导出快照后查看，不能替代包含身份和 ACK 的中心备份：

~~~sh
afx fuzz maintenance snapshot /path/results.sqlite ./results.snapshot.sqlite
afx fuzz view ./results.snapshot.sqlite --no-browser
~~~

### 中心只读 API

路由定义在 [central/server.py](../../autofusex/fuzz/server/central/server.py)，以下接口不需要登录：

| GET 路由 | 用途及主要参数 |
|---|---|
| /api/cases | 当前完整 Shape 用例；seed、status、user_id、q、limit、after |
| /api/case-history | 用例历史；必填 seed、shape，可用 after、limit、snapshot |
| /api/attempts/ATTEMPT_ID | 执行详情；可指定 shape |
| /api/summary、/api/statistics | 当前用例汇总与分布；seed、status、user_id、q |
| /api/results | attempt 级记录；不要当作用例分母 |
| /api/compare?left=ID&right=ID | 两次 attempt 的可比性与差异 |
| /api/runtime | 运行快照；user_id、machine_id、model、backend、version、minutes |

游标继续翻页；公共统计可能是带 stale 状态的缓存，503 可带 Retry-After，应退避重试。`q` 只支持受限 WHERE 表达式，不执行任意 SQL。更多参数查对应命令的 --help。

## 远程控制 API

白名单包含 `client.update`：payload `{"version":"0.1.4","package":"autofusex"}`，可选 `index_url`（仅用于受控 PyPI 镜像）；version 必须精确，daemon 不执行无版本的升级，也不自行探测最新版本。

网页用户控制走登录会话：`POST /api/session {"token":"..."}`，需与站点 Host 一致的 HTTPS Origin，保留返回的会话 Cookie 和 CSRF token；后续写请求带 Origin/Cookie/X-CSRF-Token/Content-Type: application/json，body 上限 64 KiB。设备 Bearer 凭据不能替代用户控制凭据。

| 路由 | 方法 / body | 作用 |
|---|---|---|
| /api/manage/devices、/api/manage/schemas | GET | 当前用户设备、配置 schema |
| /api/manage/validate | POST {"config": SPEC} | 纯配置校验，不解析本机执行环境 |
| /api/manage/submit | POST device_id、kind、payload、request_key | 异步远程操作（kind 复用本地操作白名单，例如 task.submit 的 payload 是任务规格，task.action 是 {"run_id": ID, "action":"stop"}） |
| /api/manage/operations | POST {"device_id": ID} | 设备操作与回执 |
| /api/manage/config | POST device_id、config、base_revision | 按修订号更新设备策略 |
| /api/retest/clients、/preview、/batches | GET / POST | 当前用户复测客户端；预览/提交复测（选择结构见 retests.py） |
| /api/issues、/claim、/release、/retest | GET / POST | 问题分组；认领/释放；问题组复测 |

服务端拒绝跨用户设备操作，本地 trust 仍限制最终权限。设备使用 Bearer 调用 `/api/manage/poll`、`ack`、`revoke`；注册配对由 enroll/pair/exchange 完成。上传与复测领取协议交给客户端实现，外部 Agent 无需重写租约或 ACK 逻辑。详见 [management/central.py](../../autofusex/fuzz/daemon/management/central.py)、[central/control_api.py](../../autofusex/fuzz/server/central/control_api.py)、[retests.py](../../autofusex/fuzz/server/central/retests.py)。

## 问题修复闭环：Agent 领取 → SSH 调试 → 中心复测归档

区别于前面的通用测试任务，这是修复工作入口。路由前缀统一为 /api/agent，使用同一用户 Bearer token；Agent 本地用 npussh 调试和构建，中心不提供任意 shell 执行接口。

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

状态机：`claimed` → `retesting` → `fixing`（失败继续修复）或 `verified`（已通过）→ `resolved`（总结归档）。领取返回修复 id 和 lease_id；`agent_id` 区分同一用户的多个 Agent 实例，同一实例同时只持有一个活动问题。租约到期或 release 后其他 Agent 可接管同一记录，lease_id 换代，旧持有者写入被拒绝；每次新操作用新 request_key，网络重试沿用原 key/body。

复测复用原有 /api/retest 客户端协议，`client_id` 是复测客户端 ID，不是 /devices 返回的托管设备 ID，需已注册在线、使用新 RUN 包环境的专用复测客户端；中心按 seed/配置自动合并多 Shape，已存在在途复测时返回 409，前一轮完成后才能提交下一轮。

artifact 格式：

~~~json
{"sha256":"64位小写十六进制摘要","version":"候选包版本","installed_at":"2026-11-15T12:00:00Z"}
~~~

该字段只是 Agent 的包安装证据，中心不凭它证明实际加载了对应二进制，Agent 必须在远端核实包路径、动态库和版本。归档依据：本轮合法租约绑定、所有任务完成且被接收、AF-MLIR 状态 ok、全部目标 Shape 存在并精度 pass（需性能评估时还需性能成功）；取消、中断、缺结果、缺 Shape、旧结果或失败均不能归档。

经验总结必须由当前租约持有者提交，且最新复测通过；每个修复记录只能有一份总结，相同请求返回同一份记录，新 key 重复归档返回 409，总结关联复测结果的 summary_id，原始执行事实不改写。可通过修复详情及 experience 查询背景、方案、备注和复测证据。

本地 skill：[afx-repair](../../skills/afx-repair/SKILL.md)。将仓库中的 `skills/afx-repair` 目录复制到 `~/.codex/skills/` 后，可在对话中用 `$afx-repair` 调用，覆盖领取/续租 → npussh 调试 → 仓库原生构建 → 隔离安装 RUN 包 → 中心复测 → 经验归档。

## 当前自动化边界

以下是当前实现限制，不能当作已经支持的接口：

- Agent API 目前复用用户 token，尚无独立细分权限的 API key、OpenAPI 或 CLI schema 导出入口；网页管理 API 继续使用 session/CSRF。
- `task submit/replay --wait` 等待提交；`task wait` 单独等待执行。停止/暂停操作成功表示控制请求已写入；completed 只表示调度完成，不表示所有模型通过，需读取 results。
- 托管 task schema 尚未覆盖直接 fuzz 的 --ops、method、dynamic-variants、dtype 等全部生成参数；需要时使用 afx fuzz，并遵守任务所有权限制。
- CLI 错误分类与中心错误体尚未统一。`task results` 查询本地已保存 attempt，`available=false` 表示结果库尚未生成，不能当作全部通过；未保存的执行无法补取结果，结果不等同于中心最新完整 Shape 投影。

后续计划：统一托管与直接 fuzz 的生成参数，补充细分权限凭据及可发现的 API schema。
