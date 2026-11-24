# CLI 命令汇总

本文是 AFX 的统一命令索引，按当前仓库的 `afx` 实际路由与参数定义整理，覆盖全部公开顶层命令和子命令。工作流细节见[客户端进阶命令](client/advanced.md)、[客户端配置](client/configuration.md)、[模型分析](usage.md)和[服务端部署](../deploy/platform/README.md)。

## 入口与约定

安装后的统一入口是 `afx`，源码调试可用 `python -m afx`。以下表格中的命令均可追加 `--help` 查看该层参数；`<>` 表示需要替换的值，`[]` 表示可选项，不要原样输入括号。

```sh
afx --help
afx --version
afx task --help
afx task submit --help
```

客户端数据目录由 `AUTOFUSEX_HOME` 或对应命令的 `--data-dir DIR` 指定。它不是所有命令通用的根参数：日常客户端命令和 `case` 支持放在命令末尾；任务管理参数放在具体动作后；服务端使用 `afx server --data DIR ...`。

## 顶层命令导航

| 命令 | 用途 |
|---|---|
| `afx init` | 初始化本地或平台客户端 |
| `afx start` / `stop` / `restart` / `status` | 客户端启动、停止、重启和状态 |
| `afx config` | 查看、修改、校验、重置或导出配置 |
| `afx daemon` | 单独启动、停止或查询后台管理服务 |
| `afx env` | 执行环境检查、准备与安装 |
| `afx replay` | 使用客户端历史配置重放 |
| `afx task` | 提交和控制托管任务 |
| `afx operation` | 查询和控制异步操作 |
| `afx build` | 构建 CANN RUN 包 |
| `afx update` | 更新到指定客户端版本 |
| `afx release` | 管理签名工具版本 |
| `afx case` | 导入和运行固定用例或整网 |
| `afx fuzz` | 随机生成、重放与调度诊断 |
| `afx model` | 运行和归一化模型脚本 |
| `afx analyze` | 精度比较、manifest 和报告 |
| `afx debug` | 捕获外部命令和 CAModel 调试 |
| `afx results` | 查询、比较和维护结果数据库 |
| `afx view` | 打开本地只读查看器 |
| `afx doctor` | 检查执行能力和环境 |
| `afx server` | 运行和管理中心服务 |
| `afx sync` | 独立结果同步 |
| `afx retest` | 独立平台复测 |

## 日常客户端与配置

| 命令 | 用途与参数 |
|---|---|
| `afx init` | 初始化本地客户端；`--local` 显式选择本地模式 |
| `afx init --token <TOKEN>` | 接入平台；`--endpoint URL` 默认 `https://autofuse.online`；`--device npu:0` 可重复指定设备 |
| `afx start [--dump]` | 启动 daemon 并开始配置的任务；`--dump` 开启本次编译调试产物 |
| `afx stop` | 停止当前客户端会话 |
| `afx restart [--dump]` | 重启客户端，重新加载环境和软件包 |
| `afx status` | 查看客户端、任务和环境状态 |
| `afx config [show]` | 查看当前配置，省略动作默认 `show` |
| `afx config edit <FILE>` | 从 JSON/YAML 文件更新配置 |
| `afx config validate <FILE>` | 校验配置 |
| `afx config reset` | 重置为默认配置 |
| `afx config export` | 输出配置内容，可用 shell 重定向保存 |
| `afx replay <SEED> [SEED ...] [--dump]` | 按客户端历史配置重放一个或多个 seed |
| `afx env list` | 列出本地托管环境 |
| `afx env inspect` | 检测 Python、架构和已安装环境 |
| `afx env status [ENVIRONMENT_ID]` | 查看指定环境；省略时为 `local` |
| `afx env prepare <FILE>` | 根据环境规格创建安装计划；`-` 从 stdin 读取 |
| `afx env apply <PLAN_ID>` | 执行安装计划 |

以上命令支持 `--data-dir DIR`。平台模式的 `config` 通过中心读取和更新配置，本地模式使用本地配置。

```sh
afx init --device npu:0 --device npu:1
afx config validate task.yaml
afx config edit task.yaml
afx start
afx status
```

## 托管任务与异步操作

### 后台管理服务

| 命令 | 用途 |
|---|---|
| `afx daemon start` | 启动 supervisor 和 daemon，等待通信接口就绪；不开始测试会话；支持 `--foreground` |
| `afx daemon status` | 查看后台服务、连接、操作和任务状态 |
| `afx daemon stop` | 停止后台管理服务；有活动任务或操作时拒绝停止 |

以上命令支持 `--data-dir DIR`、`--json`、`--timeout SECONDS` 和 `--request-key KEY`。完全停机时，先执行 `afx stop` 并等待任务退出，再执行 `afx daemon stop`。`run` 和 `supervise` 保留为内部管理入口，不在公开 `afx daemon` 中提供。

### 任务与操作参数

本节 `task`、`operation`、`build`、`release` 的每个动作以及 `update` 均支持：

| 参数 | 含义 |
|---|---|
| `--data-dir DIR` | 客户端数据目录 |
| `--json` | 输出稳定 JSON envelope，默认即使用该协议 |
| `--timeout SECONDS` | 等待超时，默认 300 秒 |
| `--request-key KEY` | 幂等键；仅同一请求内容重试时复用 |

规格 `FILE` 可为 YAML/JSON，`-` 表示 stdin。异步提交返回 Operation ID；提交操作成功后才获得 Run ID。`task submit --wait` 和 `task replay --wait` 等待提交操作，等待任务实际结束应使用 `task wait`。超时不会取消原操作或任务。

| 命令 | 用途与参数 |
|---|---|
| `afx task validate <FILE> [--wait]` | 校验任务、解析环境并返回有效配置快照 |
| `afx task submit <FILE> [--wait]` | 提交任务；可加 `--workers N`、`--dump` |
| `afx task list` | 列出本地任务 Run |
| `afx task inspect <RUN_ID>` | 查看任务和不可变配置快照 |
| `afx task status <RUN_ID>` | `task inspect` 的别名 |
| `afx task wait <RUN_ID>` | 等待任务结束、中断或需要处理 |
| `afx task results <RUN_ID>` | 读取保存结果；支持 `--limit N`、`--after N`、`--status {ok,fail,error,unsupported}`、`--seed SEED`、`--attempt ID` |
| `afx task replay <SEED> [SEED ...] --from-run <RUN_ID>` | 沿用指定历史 Run 的环境及执行/评估设置；可加 `--dump`、`--wait` |
| `afx task pause <RUN_ID> [--wait]` | 在安全边界暂停新生成和派发 |
| `afx task resume <RUN_ID> [--wait]` | 恢复暂停任务 |
| `afx task stop <RUN_ID> [--wait]` | 正常停止任务；`--graceful` 为显式别名 |
| `afx task force-stop <RUN_ID> [--wait]` | 强制停止该任务拥有的进程 |
| `afx task reconfigure <RUN_ID> <FILE> [--wait]` | 在下一安全 epoch 更新持续任务配置 |
| `afx operation inspect <OPERATION_ID>` | 查询异步操作状态 |
| `afx operation wait <OPERATION_ID>` | 等待操作终态 |
| `afx operation logs <OPERATION_ID> [--after N]` | 读取结构化操作事件 |
| `afx operation cancel <OPERATION_ID>` | 取消排队操作或请求取消运行中操作 |
| `afx build submit <FILE> [--wait]` | 提交 CANN RUN 构建规格 |
| `afx build inspect <OPERATION_ID>` | 查看构建操作 |
| `afx update <VERSION> [--index-url URL] [--wait]` | 通过 daemon 安装指定的准确版本 |
| `afx release stage <FILE> [--wait]` | 暂存签名 release manifest |
| `afx release list` | 列出已暂存版本 |
| `afx release inspect <RELEASE_ID>` | 查看一个已暂存版本 |
| `afx release activate <RELEASE_ID> [--wait]` | 通过稳定 launcher 激活版本 |

管理协议退出码：`0` 成功或已接受；`1` 操作或任务失败、停止、取消、中断或需要处理；`2` 参数/校验错误或等待超时；`3` daemon 不可用、文件或连接异常。`ok:true` 不代表任务已执行成功，应继续查看操作和 Run 状态。

## 固定用例与整网

| 命令 | 用途与参数 |
|---|---|
| `afx case import <PATH>` | 复制 Python 脚本或源码包到本地注册表；支持 `--script PATH`、`--entry NAME`、`--name NAME`、`--source {manual,baize_torch,baize_tf}`、`--workload {imported,model}` |
| `afx case list` | 列出已导入用例 |
| `afx case inspect <CASE_KEY>` | 查看用例及版本信息 |
| `afx case run <CASE_KEY>` | 提交运行；`--environment NAME` 默认 `local`；`--device DEVICE` 可重复；`--submit` 提交中心；`--upload` 上传结果 |

`case run` 还支持 `--with-cpu`、`--fullgraph`、`--dynamic`、`--rtol FLOAT`、`--atol FLOAT`、可重复的 `--script-arg ARG`。全部 `case` 命令支持 `--data-dir DIR`。

## Fuzz 生成、重放与诊断

| 命令 | 用途与参数 |
|---|---|
| `afx fuzz run [SEED ...]` | 无 seed 时随机生成；有 seed 时调度指定用例；默认后台运行 |
| `afx fuzz run --continue` | 按保存的随机 campaign 原配置继续；不接受 seed 或其他生成、后端、设备、资源覆盖参数 |
| `afx fuzz status [TASK_ID]` | 查看任务或分页状态；支持 `--limit N`、`--after N` |
| `afx fuzz stop [TASK_ID]` | 停止指定任务；省略 ID 时请求停止随机 campaign |
| `afx fuzz resume <TASK_ID>` | 恢复一个调度任务 |
| `afx fuzz replay <SEED>` | 直接重放自包含短 seed；默认 CPU 和 AF-MLIR |
| `afx fuzz seed <TOKEN> [--show-spec]` | 解析 seed 并预览冻结规格 |
| `afx fuzz diagnose <EXPECTED_JSON> <ACTUAL_JSON>` | 比较两份生成 trace，报告首个分歧 |

`fuzz run` 参数：

| 参数 | 含义 |
|---|---|
| `--count N` | 随机用例数；省略时持续生成；不能同时传 seed |
| `--fg` | 在当前终端等待和监督 |
| `--tensor-scale PROFILE` | `small`、`medium`、`large`、`16k`、`32k`、`64k`、`128k`、`default` |
| `--dim-range MIN MAX` | 维度范围 |
| `--dim-dist-mode MODE` | `uniform`、`half_normal`、`log_normal` |
| `--dtype-mode MODE` | `float32`、`float16`、`mixed`、`all` |
| `--value-dist-mode MODE` | 新配置使用 `log_scale`、`llm`、`recommendation`；解析器仍保留历史 `uniform` |
| `--max-elem-per-tensor N` / `--max-nodes N` | 张量元素上限、图节点上限 |
| `--generation-timeout SECONDS` | 生成求解超时，编码进 seed |
| `--dynamic-variants N` | 动态形状变体数，2–8 |
| `--method METHOD` | `symbolic-cinit`、`symbolic`、`concolic` |
| `--ops TARGETS` | 逗号分隔算子名、显示名或类别；在完整范围生成后过滤；要求 staged 调度 |
| `--exclude-op transpose` | 排除包含 transpose 的随机候选；仅随机 staged campaign |
| `--backend BACKEND` | 可重复增加 `cpu`、`eager-npu`、`triton-npu`、`af-legacy`、`af-mlir`；CPU 与 AF-MLIR 始终包含 |
| `--device npu:N` | 可重复，默认全部可见 NPU |
| `--auto-resources` / `--fixed-resources` | 自动计算资源或使用固定资源；互斥 |
| `--workers N` | CPU/NPU 合计最大并发 worker 数 |
| `--device-memory-bytes N` | 每张物理 NPU 的内存预算 |
| `--worker-max-tasks N` | 阶段运行时回收间隔，默认 16；1 禁用复用 |
| `--scheduling {staged,combined}` | 默认 `staged` |
| `--static-only` / `--perf` | 仅基础形状样本 / 请求精确性能指标 |
| `--save` / `--dump` | 保存结果 / 开启编译调试产物，两者独立且默认关闭 |
| `--timeout SECONDS` | 调度超时参数，默认 120 秒 |

生成采样参数仅用于新随机用例，指定 seed 时使用 seed 内的配置。`fuzz replay` 支持 `--scheduling {staged,combined}`、单个 `--backend BACKEND`、`--device DEVICE`、`--static-only`、`--save`、`--dump`、`--perf`、`--warmup-steps N`（默认 3）、`--measured-steps N`（默认 10）。

```sh
afx fuzz run --count 100 --save
afx fuzz status
afx fuzz run --continue
afx fuzz replay <SEED> --device npu:0 --dump --save
```

## 模型运行与脚本归一化

| 命令 | 用途与必填参数 |
|---|---|
| `afx python3 <SCRIPT.py> [ARGS ...]` | 直接捕获模型脚本并生成分析产物 |
| `afx bash <LAUNCH.sh>` / `afx ./launch.sh` | 静态解析 shell 启动脚本中的环境和 Python 入口 |
| `afx model run --script <PATH> --out-dir <DIR> [-- ARGS ...]` | 显式指定模型运行和分析选项 |
| `afx model normalize --script <PATH> [-- ARGS ...]` | 通过配置的 Agent 生成脚本归一化 JSON |

`model run` 常用选项：

| 参数 | 含义 |
|---|---|
| `--framework NAME` / `--backend NAME` | 框架和后端；默认 `auto` / `inductor_autofuse` |
| `--device DEVICE` | 默认 `npu:0` |
| `--capture-call N` / `--module-pattern REGEX` | 选择捕获调用和模块 |
| `--entry NAME` / `--name NAME` | case factory 入口 / 报告模型名 |
| `--fullgraph` / `--allow-graph-breaks` | 全图编译 / 允许 graph break；互斥 |
| `--dynamic` / `--rewrite-device-hints` | 动态编译 / 兼容设备属性改写 |
| `--with-cpu` | 增加 CPU replay，生成三方比较；默认比较 NPU eager 与 Autofuse |
| `--env-file PATH` / `--env KEY=VALUE` | 读取 export 环境 / 追加环境变量，`--env` 可重复 |
| `--cann-root PATH` / `--html PATH` | CANN 根目录 / HTML 输出路径 |
| `--rtol FLOAT` / `--atol FLOAT` | 覆盖精度阈值 |
| `--save-elementwise-data` | 保存逐元素数据 |
| `--agent-normalize` / `--agent-recover` | 执行前归一化 / 失败后生成恢复建议 |
| `--agent-output PATH` | 归一化 JSON 路径 |

`model run` 与 `model normalize` 均支持 Agent 连接参数：`--agent-endpoint URL`、`--agent-transport {auto,http,cannbot}`、`--agent-command PATH`、`--agent-model NAME`、`--agent-timeout SECONDS`。`normalize` 另支持 `--framework NAME`、`--out PATH`。实际模型执行目前实现 PyTorch + Inductor Autofuse，TensorFlow/JAX adapter 尚未实现执行。shell 启动脚本只做静态解析；使用前需自行加载 CANN 等外部环境，细节见[模型分析](usage.md)。

## 精度分析与报告

| 命令 | 用途与必填参数 |
|---|---|
| `afx analyze manifest --side <SIDE> --graph <JSON> --dump-dir <DIR> --out <PATH>` | 从图与 npy dump 构建 manifest；可加 `--model-name NAME` |
| `afx analyze compare --cpu-manifest <PATH> --npu-manifest <PATH>` | 两方 manifest 比较 |
| `afx analyze compare-three --cpu-manifest <PATH> --npu-eager-manifest <PATH> --npu-autofuse-manifest <PATH>` | CPU、NPU eager 与 Autofuse 三方比较 |
| `afx analyze compare-npy --golden <PATH> --actual <PATH>` | 两个 npy 文件直接比较；可加 `--golden-format FORMAT`、`--actual-format FORMAT` |
| `afx analyze report --report <JSON> --out <HTML>` | 从已有报告 JSON 生成 HTML；可加 `--title TITLE` |

三种比较命令均支持 `--out PATH`、`--csv PATH`、`--html PATH`、`--rtol FLOAT`、`--atol FLOAT`、`--save-elementwise-data`。`compare` 和 `compare-three` 另支持 `--autofuse-dump DIR`。manifest 的 SIDE 为 `cpu`、`npu`、`npu_eager`、`npu_autofuse`、`inductor_prefusion_cpu` 或 `inductor_prefusion_npu_eager`。

## 外部命令与 CAModel 调试

| 命令 | 用途与参数 |
|---|---|
| `afx debug run --out-dir <DIR> --command <COMMAND>` | 运行外部 shell 命令并捕获日志；支持 `--mode {normal,cpu}`、可重复 `--env KEY=VALUE` |
| `afx debug camodel info --cann-root <DIR>` | 检查 CANN simulator/CAModel 目录 |
| `afx debug camodel run --cann-root <DIR> --out-dir <DIR> --command <COMMAND>` | 在模拟器环境执行命令；支持 `--soc NAME`、可重复 `--env KEY=VALUE` |

`--command` 应以 shell 引号包裹为单个参数，支持 `{out_dir}` 占位符。

## 结果数据库与本地查看器

| 命令 | 用途与参数 |
|---|---|
| `afx results list <DATABASE>` | 分页读取结果；支持 `--limit N`（默认 50）、`--after N`、`--seed SEED`、`--attempt ID` |
| `afx results compare <DATABASE> <LEFT> <RIGHT>` | 检查两个 attempt 的可比性并比较结果 |
| `afx results snapshot <SOURCE> <DESTINATION>` | 导出一致性 SQLite 快照 |
| `afx results restore <SOURCE> <DESTINATION>` | 恢复到新数据库 |
| `afx results merge <SOURCE> <DESTINATION>` | 原子合并到现有数据库 |
| `afx results check <DATABASE>` | 只读审计结果数据库 |
| `afx results rebuild-index <DATABASE> [--kind {results,cases,all}]` | 重建报告索引，默认 `all` |
| `afx view <DATABASE> [--port N] [--no-browser]` | 启动本地只读查看器；默认自动选择端口并打开浏览器 |
| `afx doctor [--workload fuzz] [--data-dir DIR]` | 输出资源、依赖和配置诊断；不执行设备运行探测 |

## 中心服务管理

服务端数据目录参数放在动作前：`afx server --data <DIR> <COMMAND>`。以下表格省略 `--data DIR`，执行时必须补上。

| 命令 | 用途与参数 |
|---|---|
| `afx server serve` | 启动 HTTP API；`--host` 默认 `127.0.0.1`，`--port` 默认 `8100` |
| `afx server user add <USER_ID>` | 创建用户；`--role {admin,developer}` 默认 `developer`；可加 `--name NAME` |
| `afx server user disable <USER_ID>` | 禁用用户 |
| `afx server user enable <USER_ID>` | 启用用户 |
| `afx server token issue <USER_ID> [--expires-days N]` | 发放用户 token |
| `afx server token list [--user USER_ID]` | 列出 token |
| `afx server token revoke <PREFIX>` | 撤销指定前缀的 token |
| `afx server rebuild-stats` | 从 core.db 重建统计 |
| `afx server backup --out <DIR> [--core-only] [--tmp DIR]` | 创建一致性备份归档；`--core-only` 跳过 content.db |
| `afx server verify-backup <ARCHIVE> [--tmp DIR]` | 验证归档哈希及 SQLite 完整性 |
| `afx server restore <ARCHIVE> [--content-from ARCHIVE]` | 恢复到空数据目录；可从完整备份补充 content.db |
| `afx server client-access show` | 查看客户端准入策略 |
| `afx server client-access apply <FILE>` | 启动前原子应用客户端准入策略 |

```sh
afx server --data /srv/afx serve --host 127.0.0.1 --port 8100
afx server --data /srv/afx backup --out /srv/backups
```

## 独立同步与平台复测

日常带 token 的客户端由 daemon 自动处理同步。独立 Fuzz 调试需要单独同步时，使用以下入口。

| 命令 | 用途 |
|---|---|
| `afx sync setup` | 初始化独立同步 |
| `afx sync run` | 持续同步 |
| `afx sync once` | 单次同步 |
| `afx sync status` | 查看同步状态 |
| `afx retest run` | 持续运行平台复测循环 |
| `afx retest once` | 单次平台复测 |

上述每条命令都必须提供 `--endpoint URL --token-file PATH --results DATABASE`，可选 `--state PATH`、可重复 `--device npu:N`、`--interval SECONDS`（默认 30）。

```sh
afx sync run --endpoint https://autofuse.online \
  --token-file /path/to/token --results /path/to/results.sqlite
```

## 维护本索引

新增、删除或重命名公开命令时同步更新本文。命令路由以 [autofusex/cli.py](../autofusex/cli.py) 为准；各组参数分别定义在客户端、Fuzz、固定用例、结果和服务端 CLI 模块。本文使用公开的 `env`、`release`、`update` 命令名；内部管理模块中的 `environment`、`tool`、`client` 不属于统一入口的公开顶层命令。
