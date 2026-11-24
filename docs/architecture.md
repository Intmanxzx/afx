# AFX 平台架构

AFX 面向 Autofuse 开发与验证，包含中心服务、NPU 客户端、公共执行设施，以及自动泛化、固定用例和整网分析三类工作负载。本文描述当前代码的职责与实现边界；使用入口见[文档索引](README.md)。

## 模块与依赖

| 目录 | 职责 |
|---|---|
| `autofusex/contracts/` | 公共执行配置、结果、IPC、序列化、固定源码包与环境策略协议 |
| `autofusex/server/` | HTTP API、身份、配置、执行记录、统计、问题、复测、知识库及中心任务 |
| `autofusex/client/` | 本机 CLI、daemon、环境与任务管理、资源准入、心跳、同步及升级 |
| `autofusex/client/supervisor/` | 独立监督进程、安装切换、就绪检查与失败回退 |
| `autofusex/execution/` | 公共执行环境、资源预算、设备租约、子进程、比较、存储及查询 |
| `autofusex/workloads/fuzz/` | E/F seed、NNSmith、ATen 构图、campaign、阶段队列与后端适配 |
| `autofusex/workloads/imported/` | 固定源码登记、不可变版本、用例 CLI 与任务配置 |
| `autofusex/workloads/model/` | 模型和脚本执行、框架适配、图采集、精度分析与报告 |
| `autofusex/viewer/` | 本地只读 HTTP 服务与报告适配 |
| `autofusex/server/static/` | 中心 Web 与本地 viewer 共用的前端资源 |

客户端调用 workload，workload 使用 execution；公共数据通过 contracts 交换。中心通过 HTTP 与客户端通信，任务配置使用公共协议和 workload 校验器。contracts 不依赖 Torch、NNSmith、数据库或服务端业务，中心与本地查看器无需加载模型执行框架。

NNSmith 生成逻辑和 seed 调度归 fuzz。execution 的历史存储、查询和报告仍有显式 fuzz 数据适配；固定源码和整网任务使用自己的结果协议，不伪造 seed。模型执行也会调用客户端的制品与用例登记设施，目前并非所有模块都满足完全单向的层级依赖。

## 进程与单安装客户端

| 进程 | 职责 |
|---|---|
| supervisor | 监护 daemon，执行安装、重启、就绪判断与回退 |
| daemon | 同步平台配置、心跳、领取任务、维护操作日志、启动和停止 worker、上传结果 |
| worker | 在登记的执行环境中运行 workload，保存执行事实和制品 |

客户端从调用 `afx` 的 Python 安装启动。supervisor 的固定代码快照位于数据目录的 `supervisor/v2`，使用协议 3；普通客户端升级不覆盖该快照。快照仅包含监督和状态协议所需代码，通过子进程启动同一 Python 环境中的 daemon。

worker 使用已登记的 Torch/CANN 解释器，通过 `PYTHONPATH` 优先加载当前安装的 AFX。`afx stop` 表示主动停止业务；`afx restart` 重新加载客户端和执行环境。详细更新流程见[客户端升级](client/upgrade.md)。

## 用例与任务

| 工作负载 | 身份与规格 | 执行路径 |
|---|---|---|
| fuzz | E/F seed、CaseSpec、生成协议与输入 | NNSmith → 阶段调度 → CPU 参考与目标后端 → 本地结果库 |
| imported | `I:` 内容摘要、revision、不可变源码包和入口 | 固定源码登记 → 客户端任务 → 模型 runner |
| model | `M:` 内容摘要、revision、模型脚本或工厂入口 | 整网任务 → 模型 runner → 精度结果与报告 |

来源、运行类型和执行目的分别记录。Baize 是来源标记；当前固定源码任务使用模型 runner，TensorFlow/JAX adapter 仍未实现完整执行。不能把来源标签视为已支持该框架。

`client/configuration.py` 分派 `fuzz_task`、`imported_task` 和 `model_task` 校验。三类任务共用环境解析、资源准入、任务锁与持久化 operation。固定源码任务由 `execution/source_task.py` 管理进程组、停止、超时和内存约束；当前配置只接受 `af-legacy` 后端，不具备 fuzz 的多后端阶段调度能力。

`afx case import` 只登记源码和计算版本，不执行代码。`case run` 默认本地运行，`--submit` 登记源码并提交中心任务，`--upload` 上传本地结果。中心任务按用户隔离，领取与回执可重复恢复；已有任务占用时排队，租约不自动跨机器抢占。源码变化产生新的 revision，历史记录保持原版本。

固定源码任务通过 `source_cases/source_jobs/source_artifacts` 保存，由中心“导入与整网”页面展示。其任务重跑与报告流独立于小图统计及问题修复的验证/回归队列，不能把统一入口理解为所有业务语义已经合并。

## 数据与身份

中心的 `core.db` 保存身份、配置、用例、执行头、任务及问题状态，`content.db` 保存内容，`stats.db` 保存派生统计。客户端的数据目录保存安装身份、配置、管理状态、执行环境及 `runs/RUN_ID`。数据库和升级操作保留已有身份、token、配置、执行事实与上传进度。部署和恢复使用[一致性备份流程](../deploy/platform/README.md)。

完整 seed 决定生成规格和输入；实际硬件与软件版本归执行记录。CaseSpec 内冻结的 materializer 名称及 `graph_json.torch` 历史 wire 标记属于身份协议，不随 Python 包路径或当前 Torch 版本改写。已有摘要保持原值；重建要求匹配生成算法与依赖，不承诺任意依赖版本都生成相同内容。

本地结果分类使用 `execution/reporting/failure_origin.py`，上传转换在 `client/platform_transport.py`，中心按 `server/failure_attribution.py` 重新判定并持久化。当前仍存在不同分类实现，共用页面资源不代表分类规则完全一致。

## Web 与本地查看

中心和 viewer 使用同一份 `server/static`，由能力信息选择中心或本地模块；查询与渲染仍保留各自适配。viewer 只读打开本地结果，不连接中心数据库、不启动中心业务，也不要求 Torch、NNSmith 或 NPU。只读限制由接口执行。

模型 HTML 报告由 `viewer/report.py` 生成，属于独立报告产物。模型采集和比较边界见[模型分析架构](model_architecture.md)。

## 命令与接口

公开命令为 `afx`，同时支持 `python -m afx` 和 `python -m autofusex`。

| 命令组 | 用途 |
|---|---|
| `init/start/stop/restart/status/config/replay` | 日常客户端控制、配置和按本机历史重放 |
| `env/task/operation/build/update/release` | 环境、任务、异步操作与版本管理 |
| `fuzz` | 临时泛化、重放、队列状态与专项诊断 |
| `case/model` | 固定用例、整网运行与模型归一化 |
| `results/view/analyze/debug/doctor` | 结果维护、查看、精度分析和诊断 |
| `server/sync/retest` | 中心服务管理、独立结果同步与复测执行 |

顶层 `replay` 从本机历史恢复配置并排队；`fuzz replay` 接受任意 seed 和显式 backend，默认先通过 daemon 将任务以 priority=100 加入当前活动 fuzz run 的队列，使用该 run 的执行环境和资源预算。它优先于普通任务，但不抢占正在执行的 stage。没有活动 run 时独立执行；显式 `--database` 或 `--scheduling combined` 使用独立执行路径。内部 daemon 启动使用 `python -m autofusex.client`，不属于日常公开命令。HTTP 参数与鉴权以实际生成的 [OpenAPI](openapi.md) 为准。
