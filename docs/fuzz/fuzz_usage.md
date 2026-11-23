# 平台使用说明

**NPU 客户端**执行用例并上报数据（自动泛化测试、导入用例）。客户端默认交给 daemon 托管；下文"托管"工作流（下发任务、管理环境、一键更新）都是通过本机 `afx` CLI 与本地 daemon 交互，不需要远程托管时，跳过它直接用"方式二"本机运行即可。问题闭环、知识库等 Agent 工作流由独立的平台服务（`autofusex/server/`）提供，见[平台业务与数据设计](platform_v3_design.md)；本文档只覆盖 fuzz 执行引擎本身。

## 快速开始

### 一、NPU 客户端：执行用例并上报数据（自动泛化测试、导入用例）

在已加载 PyTorch、torch_npu、CANN 和 Autofuse 运行环境的 NPU 机器上：

~~~sh
python -m pip install 'autofusex[execution]'
afx init
afx start
afx status
# 需要停止时
afx stop
~~~

初始化只需一次。接入平台时将 `afx init` 换成 `afx init --token YOUR_TOKEN`。`start` 自动启动 daemon、登记当前环境并启动任务；带 token 时自动同步配置与心跳，按 `results.upload` 上传结果，无需另起上传进程。详见[客户端配置与重放](client_configuration.md)。下文“方式一”是手工管理环境和任务的进阶接口。

不需要托管、只是本机临时调试时，可以跳过上面整套流程，直接：

~~~sh
python -m pip install 'autofusex[execution]'
# 再按架构装匹配的 Torch 2.10.0+cpu、torch_npu 和 CANN toolkit（无通用 pip 包，手工装）
afx fuzz maintenance doctor        # 校验环境
afx fuzz run --count 100 --save
afx fuzz status
~~~

完整参数见下文"方式二"。要把结果上传到云端（上传、心跳、复测），独立运行 `afx sync run`，用法见 [deploy/platform/README.md](../../deploy/platform/README.md#客户端npu机器)。

## 安装与准备

autofusex 已发布到 PyPI，直接 pip 安装即可，无需本地 wheel 文件；不指定版本号默认安装最新发布版本：

~~~sh
python -m pip install autofusex
~~~

需要固定到某个历史版本时加 `==VERSION`。执行端（NPU 客户端）还需匹配的 Torch、torch_npu、CANN 和固定版本的 `nnsmith-autofusex`（随 `[execution]` extras 一并从 PyPI 安装）；中心和本地查看器只需上面的基础安装。源码安装执行依赖：

~~~sh
python -m pip install -e '.[execution]'
~~~

数据目录用 AUTOFUSEX_HOME 或命令末尾的 `--data-dir DIR` 指定。管理 CLI 需要 daemon 已启动、执行环境已登记为 ready、设备已获本地授权；首次准备见[运维说明](operations.md)。

## 方式一：本地管理 CLI（afx，NPU 客户端机器）

入口定义在 [management/cli.py](../../autofusex/client/cli.py)。普通请求输出统一 envelope（`--help`/`--version`/daemon 生命周期命令不计入此协议）：

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

`submit --wait` 等待的是提交操作，`task wait` 单独等待执行；超时不取消任务，可用同一 ID 继续等待，重试同一操作沿用 request key，新操作用新 key。`upload: true` 需要带 token 的客户端同步上传，仅本地保存可设为 false；省略 execution 时使用 CPU、AF-Legacy 和 AF-MLIR，全部可见 NPU 和 staged 调度。重放沿用原环境/后端/资源/评估配置，原任务须先释放所有权，不覆盖旧记录。FILE 参数统一支持 `-` 读取 stdin；可先 `afx task validate task.yaml --wait` 校验环境与规格。

托管 Run 的上传由带 token 客户端的 daemon 自动完成；本地模式须设置 `results.upload: false`。独立上传进程用于下面的 `afx fuzz` 直接调试流程。

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
| config show/edit/reset/validate/export | 查看、修改、重置、校验或导出任务配置 |
| build submit FILE / tool stage FILE / tool activate RELEASE_ID | 本地受信构建与签名制品升级 |

配置 schema 见 [fuzz 配置协议](../../autofusex/workloads/fuzz/configuration.py)，支持 client_policy/environment/fuzz_task；未知字段、YAML 重复键/别名、超限内容会拒绝。任务默认持续生成、保存并启用上传（本地模式须关闭上传），资源还受 trust 和 policy 限制。

## 方式二：临时运行与重放（afx fuzz，本机直接调试）

在已准备好的执行环境内运行，不经过 daemon 的任务队列：

~~~sh
afx fuzz run --count 100 --save
afx fuzz status
afx fuzz stop
afx fuzz replay SEED --save
~~~

将 SEED 替换为结果中的当前 E/F seed。不传 count 持续生成，`--fg` 前台运行，`--save` 才归档结果到本地库；上传到云端用独立的 `afx sync run`（见上文"快速开始"）。

| 目的 | 命令 |
|---|---|
| 单后端重放 | afx fuzz replay SEED --backend af-mlir --save |
| 人工 eager-npu 复核 | afx fuzz replay SEED --backend eager-npu --save |
| 过滤算子 | afx fuzz run --ops add,ge,sum,reshape --save |
| 双卡指定资源 | afx fuzz run --device npu:0 --device npu:1 --workers 4 --compile-threads 1 --save |
| 检查环境 | afx fuzz maintenance doctor |

run --backend 增加被测后端，replay --backend 只执行指定后端；不要与同一 home 的活动托管任务同时运行。

## 查询结果

公网结果网页：https://autofuse.online 。页面和 HTTP API 由 `autofusex/server/` 提供，路由与鉴权见[平台业务与数据设计](platform_v3_design.md)，不在本文档重复。

本地结果可导出快照后离线查看：

~~~sh
afx fuzz maintenance snapshot /path/results.sqlite ./results.snapshot.sqlite
afx fuzz view ./results.snapshot.sqlite --no-browser
~~~

## 设备管理

设备管理（配对、环境、任务下发、`client.update` 远程升级）是本机 `afx` CLI 与本地 daemon 之间的协议，数据模型见 [management/central.py](../../autofusex/client/central.py)。目前没有暴露对应的远程 HTTP 接口，所有管理操作都在本机通过上文"快速开始"列出的 `afx` 命令完成。

`client.update` 的 payload 是 `{"version":"0.1.4","package":"autofusex"}`，可选 `index_url`（仅用于受控 PyPI 镜像）；version 必须精确，daemon 不执行无版本的升级，也不自行探测最新版本。

问题领取、知识库、Agent 修复闭环由平台服务的 `/api/issues/*` 等接口提供，见[平台业务与数据设计](platform_v3_design.md)第 5 节；本地 skill [afx-repair](../../skills/afx-repair/SKILL.md) 把那份文档当作当前 API 参考。

## 当前自动化边界

以下是当前实现限制，不能当作已经支持的接口：

- `task submit/replay --wait` 等待提交；`task wait` 单独等待执行。停止/暂停操作成功表示控制请求已写入；completed 只表示调度完成，不表示所有模型通过，需读取 results。
- 托管 task schema 尚未覆盖直接 fuzz 的 --ops、method、dynamic-variants、dtype 等全部生成参数；需要时使用 afx fuzz，并遵守任务所有权限制。
- `task results` 查询本地已保存 attempt，`available=false` 表示结果库尚未生成，不能当作全部通过；未保存的执行无法补取结果。

后续计划：统一托管与直接 fuzz 的生成参数。
