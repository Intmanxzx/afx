# Fuzz 运维说明

日常任务见[使用说明](fuzz_usage.md)，服务端部署见 [deploy/platform/README.md](../../deploy/platform/README.md)。以下记录当前约定，不记录某次线上部署状态。

## 普通客户端首次准备

在已加载 NPU 运行环境的 shell 中执行：

~~~sh
python -m pip install 'autofusex[execution]'
afx init
afx start
~~~

接入平台时，首次初始化改用 `afx init --token YOUR_TOKEN`。`start` 自动启动 daemon、登记当前 Python 环境并准备基础资源授权。配置、设备选择和重放见[客户端配置与重放](client_configuration.md)。`init` 不安装 Torch/CANN。已有客户端无需重新初始化；更换运行环境或安装包后使用 `afx restart`。

## 托管客户端首次准备

本节是管理员手工安装受信环境的进阶流程；普通客户端使用上面的 `init/start` 即可。

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

trust 的 compatibility 配置还需提供匹配架构、Python 可执行文件、完整 wheel 清单及摘要、必要环境变量；existing_roots 限制已有 CANN 路径，resource_limits 限制设备与并发。读取字段见 [environments.py](../../autofusex/fuzz/daemon/management/environments.py) 和 [artifacts.py](../../autofusex/fuzz/daemon/management/artifacts.py)。普通 init/start 流程会准备当前环境的基础资源授权；本节的受信包安装仍需完整 compatibility 配置，不能只复制上述 YAML。

多卡或更高并发需同时配置 trust 和 afx config apply policy.yaml --wait 的资源上限。环境 ready 只证明版本/导入检查通过；真实 NPU 路径须在目标机器验证。开机自启由 systemd 或进程管理器负责。

## 结果数据库维护

数据库恢复或替换后，不能沿用另一份库的旧统计侧文件。派生索引可重建，维护前备份，再按需要执行：

~~~sh
afx fuzz maintenance index-results /path/to/results.sqlite
afx fuzz maintenance index-cases /path/to/results.sqlite
~~~

~~~sh
afx fuzz maintenance check-db /path/to/results.sqlite
~~~

公开 maintenance snapshot 只导出允许分享的结果和观测事实，不是完整灾备；不要直接复制正在 WAL 写入的数据库主文件。

升级先做一致性备份，在新虚拟环境 `pip install autofusex==VERSION` 固定版本，用数据库副本验证启动/查询后切换服务，保留旧环境和有效配置。旧 Python 进程不会自动加载新模块，客户端切换后须重启相关进程。

平台服务端（`autofusex/platform/`）的身份、token、备份与恢复，见 [deploy/platform/README.md](../../deploy/platform/README.md)。

## 构建与发布

源码真源是 GitCode 的 xingzhixiong/Autofuse-Debugger，GitHub 的 [Intmanxzx/afx](https://github.com/Intmanxzx/afx) 是只读镜像，也是唯一的发布出口。版本在 pyproject.toml、autofusex/__init__.py 及部署模板中保持一致，Tag 使用 v 加实际版本号。

~~~sh
python -m pip install -r tools/release-requirements.txt
python -m build --wheel --no-isolation
python -m twine check dist/*.whl
python tools/release_wheel.py --host github --tag vVERSION --repository Intmanxzx/afx
~~~

替换 vVERSION。脚本默认校验和展示计划，真实发布由可信 Tag 流水线使用 --publish。GitHub Release 和 PyPI 使用同一 wheel，包含网页资源；NNSmith fork 通过 execution 依赖安装。发布前完成相关回归、执行环境验证及包安装检查，配置 Release/PyPI 凭据并保护 Tag 权限。同版本不得重建不同内容覆盖发布，需要修改时升版本。

镜像使用 GitCode 原生 Push 镜像，在 GitCode 提交后由平台触发同步，不再依赖 GitHub Actions 的定时调度。配置入口为 GitCode 源仓库「项目设置 → 仓库镜像 → Push → 添加镜像」：

| 配置项 | 值 |
| --- | --- |
| 目标项目地址 | `https://github.com/Intmanxzx/afx.git` |
| 账号 | `Intmanxzx` |
| 个人令牌 | 授权 GitHub 仓库写入的个人访问令牌 |

GitHub fine-grained personal access token 仅授权 `Intmanxzx/afx`，Repository permissions 为 Contents: Read and write、Workflows: Read and write，以同步代码及 `.github/workflows/` 的变更。令牌原值保存在 GitCode 镜像设置中，不能只填写 `MIRROR_TOKEN` 这个名称；令牌到期后在该设置中更新。GitHub Actions 不再使用 `MIRROR_TOKEN` Secret，原生同步验证成功后可删除旧 Secret。

只配置 GitCode → GitHub 的 Push 方向。GitCode 官方说明原生 Push 镜像会覆盖目标仓库代码，因此日常开发和合并应在 GitCode 进行，GitHub 只接收镜像更新。镜像范围是 Git 分支、标签和提交，不包括 Issue、PR、评论或 Release 附件。平台说明见 [仓库镜像](https://docs.gitcode.com/docs/help/home/org_project/project_manage/project_settings/repository_mirroring/)。

在 GitHub 仓库 Settings → Secrets and variables → Actions 中保留 Repository secret `PYPI_API_TOKEN`：正式 PyPI（pypi.org）的上传令牌。首次创建项目时需要可创建项目的账号范围令牌，发布后可改为仅授权 `autofusex` 的项目范围令牌。

[release.yml](../../.github/workflows/release.yml) 由镜像推送的 `vX.Y.Z` Tag 触发，也可手动指定已有 Tag 重跑。原生镜像使用个人访问令牌推送，会触发 GitHub 工作流，无需额外 dispatch。发布流程检出该 Tag、确认它匹配默认分支声明的版本且提交属于默认分支历史、干净构建 wheel、在只含 wheel 的独立虚拟环境里跑安装冒烟测试、校验 Tag/元数据/目录结构，最后写入 GitHub Release 与 PyPI。历史版本 Tag 会跳过发布。GitHub Release 用内置的 `GITHUB_TOKEN`，无需额外配置。

首次配置后的验证：

1. 确认 GitHub 默认分支为 `master`，向 GitCode 推送一个文档更新等普通提交，触发原生 Push 镜像。
2. 在 GitCode 镜像页面检查同步状态，并用 `git ls-remote origin refs/heads/master` 和 `git ls-remote https://github.com/Intmanxzx/afx.git refs/heads/master` 核对两端提交 SHA 一致。验证期间仅向 GitCode 推送。
3. 确认 GitHub 默认分支已移除 `.github/workflows/mirror.yml`，保留 `.github/workflows/release.yml`。若镜像失败，先查看 GitCode 页面中的错误信息，检查令牌有效期、目标地址及写入权限。

发布新版本时，先将版本修改提交到 GitCode 并等待 `master` 同步完成，再推送新版本 Tag；如 Tag 已同步而发布门禁跳过，可在 GitHub Actions 手动运行发布工作流。以新版本 `0.2.1` 为例（版本文件须先更新为 `0.2.1`）：

~~~sh
git push origin master
# 等待 GitHub master 同步完成后再创建和推送 Tag。
git tag -a v0.2.1 -m 'Release autofusex 0.2.1'
git push origin refs/tags/v0.2.1
~~~

发布成功后使用 `python -m pip install autofusex==0.2.1` 安装，运行 `afx --help` 验证。PyPI 同版本文件不能覆盖；上传失败可从 GitHub Actions 重跑对应发布任务，代码或依赖需要改变时升版本。

流水线只运行 `tests/test_release_wheel.py`。`tests/fuzz` 仍有若干用例引用 `c6feaac`（refactor fuzz into client daemon server modules）重构前的模块路径（如 `autofusex.fuzz.resources`、`autofusex.fuzz.runtime`、`autofusex.fuzz.failure_details`），修复前不能作为发布门禁。

GitCode 侧已移除自动发布工作流，避免同一 Tag 同时从两处向 PyPI 上传。
