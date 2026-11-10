# Fuzz 使用说明

同一个 `autofusedebugger` wheel 提供客户端、云端和本地查看器。架构及能力边界见[设计文档](af_fuzz_design.md)。

## 安装

使用独立虚拟环境，不替换已有 Torch/CANN：

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install /path/to/autofusedebugger-<版本>-py3-none-any.whl
```

发布到 PyPI 后可用 `python -m pip install autofusedebugger==<版本>`。Windows 激活命令为 `.venv\Scripts\activate`。云端和本地查看仅使用标准库，可安装 wheel 时加 `--no-deps`，无需 Torch、NNSmith、CANN、NPU 或 Node。客户端必须自行准备匹配的 Torch、torch_npu、CANN 和所选后端；安装本包不会配置或升级它们。

## 一、客户端：执行、采集、上报

在已有 NPU Python 环境中安装 wheel；先加载所在机器的环境脚本。管理员私下发放 token，用户名由云端识别，不需要客户端填写。

```sh
autofusedbg fuzz setup --token '管理员发放的TOKEN' --sync-interval 30
autofusedbg fuzz run --save          # 无限后台测试，保存并自动上报
autofusedbg fuzz status
autofusedbg fuzz stop                # 停止当前批次
```

`--sync-interval` 单位为秒，默认 30。独立同步进程上传轻量记录，网络失败不阻塞测试，outbox/ACK 落盘后可续传。命令行可能被 shell 历史及进程列表记录，客户端凭据文件权限为仅当前用户可读。

```sh
autofusedbg fuzz run --count 100 --save                  # 有限批次
autofusedbg fuzz run --count 10 --fg      # 前台观察
autofusedbg fuzz run --backend triton-npu         # 在 CPU、AF-MLIR 外增加后端
autofusedbg fuzz replay B000000016               # 只靠 seed 重建并运行
autofusedbg fuzz replay B000000016 --backend af-mlir  # 单后端诊断
autofusedbg fuzz replay B000000016 --static-only  # 仅基础 shape，静态编译
autofusedbg fuzz replay B000000016 --save      # 保存结果并加入自动上传队列
```

- 默认 CPU 为对照、AF-MLIR 为被测后端；AF-MLIR 对 CPU 精度不一致时用 eager-npu 复核，记录判定依据。
- 后端名称：`cpu`、`eager-npu`、`triton-npu`、`af-legacy`、`af-mlir`。`run --backend` 增加被测后端，`replay --backend` 则只执行指定后端。
- 默认运行动态 shape 样本；同图不同 shape 在网站中分别计数，CPU 对照不重复计数。
- `--perf` 请求性能采集；目前无法可靠拆分的编译/kernel 指标留空，不能把首次调用或总耗时当 kernel 性能。
- 使用--save时，功能失败保存短错误、阶段和已采集上下文；精度失败保存误差统计。长日志、tensor、编译产物不默认上传。
- 停止测试不会删除数据或凭据，也不会停止独立上传进程。

数据根目录可用 `AUTOFUSEDBG_HOME` 指定。默认 Linux 为 `${XDG_DATA_HOME:-~/.local/share}/autofusedbg`，macOS 为 `~/Library/Application Support/autofusedbg`，Windows 为 `%LOCALAPPDATA%/autofusedbg`。结果在 `results/results.sqlite`，任务在 `tasks/tasks.sqlite`。后台执行/同步进程目前以 POSIX 环境为主；Windows 支持的是本地查看，不承诺 NPU 后台运行。

```sh
autofusedbg fuzz maintenance doctor
autofusedbg fuzz maintenance sync status
autofusedbg fuzz maintenance sync once
autofusedbg fuzz maintenance sync start
```

setup成功后会记住数据目录，不必每次指定。优先级：显式--data-dir、AUTOFUSEDBG_HOME、已记住目录、系统默认目录。默认上报站点为 `https://autofuse.online`；私有部署使用维护级 `sync configure --endpoint` 配置。开机后须通过客户端入口或进程管理器重新启动同步，不把一个后台进程当作系统开机服务。

## 二、云端：身份、汇总、网站

服务端不运行模型。单应用加 SQLite；HTTP 服务只监听 loopback，由 TLS 代理对外提供 HTTPS。

```sh
autofusedbg fuzz admin user /srv/fuzz/results.sqlite niuyuhu
# 输出一次 token，私下发给该用户；展示名不参与鉴权

autofusedbg fuzz admin serve /srv/fuzz/results.sqlite --port 8090
# 代理 https://autofuse.online -> 127.0.0.1:8090

autofusedbg fuzz admin rotate-token /srv/fuzz/results.sqlite niuyuhu
# 旧 token 立即失效，客户端用新 token 重新 setup
```

没有在线注册。管理员在服务器终端执行命令；服务端保存 token hash，不保存可还原的 token。网站提供总览、功能&精度、性能分析及按 shape 的运行详情，支持用户过滤、列选择和受限 SQL WHERE 查询。网站结果默认公开，不能上传秘密日志。

Linux 部署可使用 [Compose、Caddy、systemd 模板](../deploy/fuzz/README.md)。固定 wheel 版本；先验证再切换，不直接执行不受控的升级命令。

### 备份、恢复、升级、迁移

```sh
# 在线一致性灾备：包含结果、用户身份、接收回执和 ACK，必须私有保存
autofusedbg fuzz admin backup /srv/fuzz/results.sqlite /backup/fuzz-20261110.sqlite
autofusedbg fuzz maintenance check-db /backup/fuzz-20261110.sqlite
```

恢复或迁移：停止目标服务，将私有备份复制到**新的目标目录**作为 `results.sqlite`，使用原版本 wheel 启动；核对用户、结果计数、客户端上传与重复上传不增记录后，再切域名/流量。不要覆盖运行中的 SQLite，也不要只复制活跃数据库主文件遗漏 WAL。异地保存备份，限制访问权限；公开快照不能替代此灾备。

升级：先私有备份，用新虚拟环境安装目标固定版本，先在备份副本上验证启动及查询，再停旧服务、切新环境。若回滚需要恢复原版本及备份数据库，不要让旧代码打开已迁移的新 schema。恢复到较旧服务端备份后，客户端 ACK 可能超前；确认数据后执行 `maintenance sync reset-epoch --data-dir DIR`，重新建立来源流并重传，已有 attempt 幂等去重。

## 三、本地查看：SQLite 快照

在数据源导出结果快照，再复制到个人电脑：

```sh
autofusedbg fuzz maintenance snapshot /path/results.sqlite ./results.snapshot.sqlite
autofusedbg fuzz view ./results.snapshot.sqlite
# 不自动打开浏览器：
autofusedbg fuzz view ./results.snapshot.sqlite --no-browser --port 8091
```

与公网使用同一套网页，默认只监听本机、只读查看，关闭命令即停止服务。仅打开可信 SQLite 文件。结果快照不含中心身份和 ACK，但仍含用例错误和环境元数据，分享前检查内容。

## 维护命令速查

| 命令（前缀 `autofusedbg fuzz maintenance`） | 用途 |
|---|---|
| `seed SEED --show-spec` | 不执行模型，查看固定生成规格 |
| `diagnose EXPECTED ACTUAL` | 比较两份生成轨迹，定位首个分歧 |
| `compare DATABASE LEFT RIGHT` | 对比两个 attempt，检查可比性 |
| `results DATABASE` | 查询轻量结果 |
| `check-db DATABASE` | 检查数据库 |
| `snapshot SOURCE DEST` | 导出轻量一致性快照 |
| `restore SOURCE DEST` | 从结果快照恢复到新库；不用于中心身份灾备 |
| `import SOURCE DEST` | 合并结果，保留幂等语义 |
| `index-results DATABASE` | 重建/补齐持久统计索引，操作前备份 |
| `resume TASK` | 恢复特定任务 |
| `upgrade-queue QUEUE BACKUP` | 备份后升级旧任务库 |
| `sync -h` | 配置、启动、检查或修复同步 |

各命令完整参数见 `-h`；普通入口 `autofusedbg fuzz -h` 按上述场景分组。

## 四、维护者：构建和发布

流程：**推送版本 Tag → 测试、构建 wheel → 创建 GitCode Release 并上传同一 wheel → 发布 PyPI**。流水线见 [release.yml](../.gitcode/workflows/release.yml)，脚本见 [release_wheel.py](../tools/release_wheel.py)。不是点击网页“发布 Release”触发；GitCode 的 `release.published` 触发器尚无已确认支持依据。

发布前需管理员：

1. 确认 PyPI 项目名所有权和项目开源授权。本仓库尚无 LICENSE，脚本不代替作者选择许可证。
2. 在 GitCode 项目配置 Secrets：`GITCODE_RELEASE_TOKEN`（本项目 Release 写权限）、`PYPI_API_TOKEN`（该 PyPI 项目范围）。不要放到代码、日志或客户端 token 中。
3. 保护 `v*` Tag 的创建权限；只允许可信维护者发布。确认流水线 runner 可以访问 PyPI、GitCode API。
4. 同步修改 `pyproject.toml` 和 `autofusedebugger/__init__.py` 中版本，以及部署模板版本，提交后创建同名 `v<版本>` Tag。

```sh
# 在隔离构建环境中；本项目开发测试仍按约定在 mlir 进行
python -m pip install -r tools/release-requirements.txt
# x86_64 发布 runner 使用隔离 CPU Torch；NPU 开发环境保留既有 Torch
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pytest -q tests/test_fuzz*.py tests/test_release_wheel.py
python -m build --wheel --no-isolation
python -m twine check dist/*.whl
python tools/release_wheel.py --tag v0.1.2 --repository niuyuhu/Autofuse-Debugger
# 默认只校验及展示计划，不上传；真实发布仅由可信 Tag 流水线加 --publish
```

Tag 必须对应当前提交，版本必须一致，目录必须无 tracked 修改；构建 wheel 必须含网页资源且包含NNSmith patches。发布不覆盖已有文件：同名同 SHA256 可跳过，不同内容报错；GitCode 成功而 PyPI 失败时重跑相同产物可以续发，**不能重建不同内容覆盖同一版本**。PyPI 版本不可覆盖，版本已占用时必须升版本。

本轮只进行脚本 mock 测试、构建及安装验证，不使用真实发布凭据，不创建 Tag/Release、不上传 PyPI。首次真实流水线及 GitCode 项目权限需维护者发布时验证。

官方依据：[工作流结构](https://docs.gitcode.com/docs/help/home/org_project/pipeline/writing-pipelines/workflow-file-location-structure)、[Tag 触发器](https://docs.gitcode.com/docs/help/home/org_project/pipeline/writing-pipelines/configure-triggers)、[创建 Release](https://docs.gitcode.com/docs/apis/post-api-v-5-repos-owner-repo-releases)、[附件上传 URL（PUT）](https://docs.gitcode.com/docs/apis/get-api-v-5-repos-owner-repo-releases-tag-upload-url)。

### 执行默认值与 NNSmith 状态

- `run` 和 `replay` 默认不保存结果、不产生新的上传记录；显式 `--save` 才保存并进入自动上传队列。已有待上传记录不受新任务选项影响。
- 后台任务的 seed、状态等调度元数据仍持久化，用于 status/stop/恢复；它们不等于结果归档。
- `run` 不传 `--count` 且未指定 seed 列表时无限生成，直到 `fuzz stop`；指定 seed 列表则只执行该列表。
- 前台使用 `--fg`，请求性能采集使用 `--perf`。
- NNSmith 集成和全部 patches 已恢复，wheel 包含补丁；不在导入时打补丁，也不修改共享 NNSmith。当前 A/B 协议仍使用有限生成器，不能代表 NNSmith 泛化能力。广泛随机图接入统一执行流程尚未完成，不应将该版本当作完成此能力的正式发布版。
