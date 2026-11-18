# Fuzz 运维说明

日常任务见[使用说明](fuzz_usage.md)，安装客户端/服务端见[部署脚本说明](../../deploy/fuzz/README.md)。以下记录当前约定，不记录某次线上部署状态。

## 托管客户端首次准备

1. 通过 pip 从 PyPI 安装基础包（`python -m pip install autofusex`，默认取最新版本；固定版本加 `==VERSION`），启动 afx daemon start。
2. 准备本机 trust：允许的设备/资源、CANN 路径、兼容 Python/包组合、下载域名与 wheel 摘要。执行 afx management trust trust.json。trust 保存在 home 下私有文件，仅本地可修改。
3. 需要远程管理或上传时，使用 afx device enroll --token-stdin 从私有凭据来源读取 token，或执行 afx device pair 后在网页批准。
4. 准备 environment.yaml，提交计划并执行；成功后取环境 ID 填到任务规格。

~~~sh
afx environment plan environment.yaml --request-key env-plan-001
afx operation wait PLAN_OPERATION_ID
afx environment apply PLAN_ID --request-key env-apply-001
afx operation wait APPLY_OPERATION_ID
afx environment list
~~~

PLAN_ID 来自计划操作的 data.result.plan_id，两个 OPERATION_ID 均来自提交响应 data.id。环境规格示意（版本、路径、兼容配置名必须与本机 trust 一致）：

~~~yaml
schema_version: 1
kind: environment
name: npu-env
python: "3.11"
compatibility_profile: local-npu
cann:
  source: existing
  path: /path/to/cann
packages:
  torch: "实际版本"
  torch_npu: "实际版本"
  nnsmith: "实际版本"
~~~

trust 的 compatibility 配置还需提供匹配架构、Python 可执行文件、完整 wheel 清单及摘要、必要环境变量；existing_roots 限制已有 CANN 路径，resource_limits 限制设备与并发。读取字段见 [environments.py](../../autofusex/fuzz/daemon/management/environments.py) 和 [artifacts.py](../../autofusex/fuzz/daemon/management/artifacts.py)。目前没有自动生成本机 trust/兼容配置的一步入口，只复制上述 YAML 不能完成新机器初始化。

多卡或更高并发需同时配置 trust 和 afx config apply policy.yaml --wait 的资源上限。环境 ready 只证明版本/导入检查通过；真实 NPU 路径须在目标机器验证。开机自启由 systemd 或进程管理器负责。

## 中心身份与服务

~~~sh
afx fuzz admin user /srv/fuzz/results.sqlite USERNAME
afx fuzz admin serve /srv/fuzz/results.sqlite --port 8090
afx fuzz admin rotate-token /srv/fuzz/results.sqlite USERNAME
~~~

创建/轮换只显示一次原始 token，私下交付，中心保存 hash。服务绑定本机，由 HTTPS 代理对外访问。轮换后旧用户 token 失效，使用它的客户端须更新配置；托管设备凭据另有撤销入口。

数据目录解析优先级：显式 --data-dir、AUTOFUSEX_HOME、记住的 setup 目录、系统默认值。Linux 默认 XDG_DATA_HOME 下的 autofusex（未设置时使用 ~/.local/share/autofusex），macOS 默认 ~/Library/Application Support/autofusex，Windows 默认 LOCALAPPDATA 下的 autofusex。Windows 不承诺 NPU 后台任务支持。

## 备份、恢复与升级

~~~sh
afx fuzz admin backup /srv/fuzz/results.sqlite /private/backup/results.sqlite
afx fuzz maintenance check-db /private/backup/results.sqlite
~~~

中心备份包含身份、来源、回执与结果，私有保存并异地备份。公开 maintenance snapshot 只导出允许分享的结果和观测事实，不替代中心灾备；不要直接复制正在 WAL 写入的数据库主文件。

升级先做一致性备份，在新虚拟环境 `pip install autofusex==VERSION` 固定版本，用私有数据库副本验证启动/查询后切换服务，保留旧环境、有效服务配置和身份。旧 Python 进程不会自动加载新模块，客户端切换后须重启上传器和相关进程。

恢复时停止目标服务，将备份恢复到新目录，以匹配版本打开；核对身份、结果、上传和重复上传去重后再切流量。回滚不能让旧代码打开不兼容的新 schema，也不能用旧库直接覆盖上线后的新增记录。恢复较旧中心备份导致客户端 ACK 超前时，核对数据后执行 afx fuzz maintenance sync reset-epoch --data-dir DIR 重传。

数据库恢复或替换后，不能沿用另一份库的旧统计侧文件。派生索引可重建，维护前备份，再按需要执行：

~~~sh
afx fuzz maintenance index-results /srv/fuzz/results.sqlite
afx fuzz maintenance index-cases /srv/fuzz/results.sqlite
~~~

## 维护期写入限制

AF_FUZZ_WRITE_ALLOWLIST 指向私有 JSON，服务启动时读取；不设置则不加限制，配置错误拒绝启动。示意：

~~~json
{"enabled":true,"users":["用户UUID"],"clients":[{"user_id":"用户UUID","client_id":"客户端UUID"}]}
~~~

替换为真实 UUID。普通上传/复测使用对应 source_id/client_id，托管管理请求使用设备身份。只读查询仍可用，受限写入返回 503、Retry-After: 60 与 maintenance_write_blocked，ACK 不推进。配置不会停止远端执行或冻结租约；调整后重启服务，客户端保留积压结果重试。

## 破坏性升级（清空重装）

当新版本改变结果 schema 或 seed 协议、旧库不能原地打开时，客户端与服务端必须一起清空重装，不做原地迁移：

1. 停止所有客户端管理守护进程、fuzz/retest supervisor、worker 和 sync daemon；停止服务端及自动备份任务，核对没有遗留执行进程。
2. 服务停止后保留一份私有、完整的一致性备份，用于回退到旧版本；新程序不导入其中的结果。
3. 删除服务端整个持久化数据目录（主库、WAL/SHM、runtime/public-statistics 侧库、设备注册、会话、回执、复测、问题领取、修复记录、上传暂存和缓存）；Docker 应重新创建专用数据卷，不只删主库。
4. 每个客户端清空实际 data home（results、tasks、manager.sqlite、同步/outbox/ACK、运行产物、日志、配对凭据、缓存、active-home 指针）。重装环境后重新设置本机 trust 和配置；不要恢复旧 manager.sqlite 或 config/source ID。
5. 安装同一新版本，从空目录启动中心，建立用户，重新 enroll/pair 每个客户端；检查空首页、新 token 登录、新设备心跳、少量新结果上传及重复上传 ACK。
6. 验证通过后再启动持续任务，任何旧客户端都不能保留运行或上传。

可选保留用户 token：明文由用户自己保存，中心只存 SHA256，所以停服后只需把旧中心 `users` 表的 `id、display_name、key_hash、active` 四列私下复制到新空中心同名表即可保留原 token；不要复制 `sources`、设备 token、`sessions`、owners、回执或任何结果表。导出文件按凭据材料保管（权限 0600，不进日志）。新客户端用原用户 token 重新注册，获得新设备凭据和同步身份；若旧 `users` 表不符合这四列契约，直接建新用户重新发 token。这是部署时的一次性数据操作，不是运行时兼容层。

## 构建与发布

源码真源是 GitCode 的 xingzhixiong/Autofuse-Debugger，GitHub 的 [Intmanxzx/afx](https://github.com/Intmanxzx/afx) 是只读镜像，也是唯一的发布出口。版本在 pyproject.toml、autofusex/__init__.py 及部署模板中保持一致，Tag 使用 v 加实际版本号。

~~~sh
python -m pip install -r tools/release-requirements.txt
python -m build --wheel --no-isolation
python -m twine check dist/*.whl
python tools/release_wheel.py --host github --tag vVERSION --repository Intmanxzx/afx
~~~

替换 vVERSION。脚本默认校验和展示计划，真实发布由可信 Tag 流水线使用 --publish。GitHub Release 和 PyPI 使用同一 wheel，包含网页资源；NNSmith fork 通过 execution 依赖安装。发布前完成相关回归、执行环境验证及包安装检查，配置 Release/PyPI 凭据并保护 Tag 权限。同版本不得重建不同内容覆盖发布，需要修改时升版本。

镜像由 [mirror.yml](../../.github/workflows/mirror.yml) 按每 30 分钟执行一次，也可手动 workflow_dispatch。镜像对分支只做快进推送：GitHub 与 GitCode 分叉时任务失败并报错，不会覆盖只存在于 GitHub 一侧的提交。

在 GitHub 仓库 Settings → Secrets and variables → Actions 中配置 Repository secrets：

- `MIRROR_TOKEN`：GitHub fine-grained personal access token，仅授权 `Intmanxzx/afx`，Repository permissions 为 Contents: Read and write、Workflows: Read and write。内置 `GITHUB_TOKEN` 无法推送工作流文件的变更，因此不能用于完整镜像。令牌到期后需更新此 Secret。
- `PYPI_API_TOKEN`：正式 PyPI（pypi.org）的上传令牌。首次创建项目时需要可创建项目的账号范围令牌，发布后可改为仅授权 `autofusex` 的项目范围令牌。令牌只保存在 GitHub Secrets 中。

[release.yml](../../.github/workflows/release.yml) 由镜像或直接推送的 `vX.Y.Z` Tag 触发，也可手动指定已有 Tag 重跑。`MIRROR_TOKEN` 的推送会触发工作流，无需额外 dispatch。发布流程检出该 Tag、确认它匹配默认分支声明的版本且提交属于默认分支历史、干净构建 wheel、在只含 wheel 的独立虚拟环境里跑安装冒烟测试、校验 Tag/元数据/目录结构，最后写入 GitHub Release 与 PyPI。历史版本 Tag 会跳过发布。GitHub Release 用内置的 `GITHUB_TOKEN`，无需额外配置。

首次启用时，先把包含这两个工作流的默认分支通过 SSH 同步到 GitHub，并确认 GitHub 默认分支为 `master`。后续在 GitCode 推送版本 Tag，镜像同步后会自动发布；手动同步时也应先推送默认分支，再推送 Tag。以当前版本为例：

~~~sh
git tag -a v0.2.1 -m 'Release autofusex 0.2.1'
git push origin master refs/tags/v0.2.1
~~~

发布成功后使用 `python -m pip install autofusex==0.2.1` 安装，运行 `afx --help` 验证。PyPI 同版本文件不能覆盖；上传失败可从 GitHub Actions 重跑对应发布任务，代码或依赖需要改变时升版本。

流水线只运行 `tests/test_release_wheel.py`。`tests/fuzz` 仍有若干用例引用 `c6feaac`（refactor fuzz into client daemon server modules）重构前的模块路径（如 `autofusex.fuzz.resources`、`autofusex.fuzz.runtime`、`autofusex.fuzz.failure_details`），修复前不能作为发布门禁。

GitCode 侧已移除自动发布工作流，避免同一 Tag 同时从两处向 PyPI 上传。
