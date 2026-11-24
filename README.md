# AFX 平台

AFX（AutofuseX）是面向 Autofuse 开发与验证的平台，集成自动化测试、模型精度分析、结果管理、客户端管理，以及问题复测与修复闭环。NPU 客户端负责执行，平台汇总结果、管理配置并跟踪问题。

平台入口：[autofuse.online](https://autofuse.online) · [文档](docs/README.md)

## 1. 环境准备

### 1.1 安装CANN Toolkit 和用例必要（Torch&TorchNPU等）环境

安装方式请参考: [Autofuse 指导文档](https://gitcode.com/cann/graph-autofusion/blob/master/autofuse/mlir/README.md)

### 1.2 执行环境初始化命令

```shell
# CANN软件包环境初始化命令，默认安装目录：~/Ascend，可根据上述指导文档自行修改
source ~/Ascend/ascend-toolkit/set_env.sh

# Python&Torch 独立环境，通常安装在下列目录，其中 torch210_daily 是默认使用的 torch 版本，可根据上述指导文档自行修改
source /mnt/workspace/env/venv/torch210_daily/bin/activate
```

## 2. 快速开始

> 先执行上述两条环境初始化命令，确保处于 `torchXXX` python 环境，通常命令行左侧带有 `(torchXXX) ` 前缀。

### 2.1 安装并初始化客户端

```bash
# 安装工具
python -m pip install 'autofusex[execution]'
# 初始化工具，其中 XXX 在网页申请获批后领取
afx init --token XXX
```

### 2.2 启动客户端

```bash
afx start
```
客户端启动后，默认目录：
* 工作目录（客户端配置等）：`~/.local/share/autofusex`
* 执行实例目录（含实例日志、任务参数配置、用例数据等）：`~/.local/share/autofusex/runs`

> 工作说明：
>
> 1. 默认连接 [autofuse.online](https://autofuse.online)，可在初始化时用 `--endpoint URL` 指定其他平台。
> 2. 带 --token 时，自动拉取服务端配置并上报结果，用户可使用同一个`token`登录 [autofuse.online](https://autofuse.online) 的`平台运维` 页面，管理自己的任务参数配置。
> 3. 不带 --token 时，在本机运行。

### 2.3 其他命令

```bash
afx status                 # 查看客户端状态
afx stop                   # 停止当前任务
afx start                  # 按当前配置启动任务
afx restart                # 若更新了CANN包等，可热加载配置
afx start --dump           # 本次任务开启编译调试 dump（默认关闭）
afx replay SEED --dump     # 重放指定 seed 并开启编译调试 dump
```

更多详细介绍，参考：[客户端使用说明](docs/client/configuration.md)。

## 3. 平台能力

- 自动化验证：生成测试用例，执行多后端对比并保存结果。
- 模型分析：运行已有模型，进行张量精度比较、图定位和报告展示。
- 客户端管理：查看设备与运行环境、同步任务配置、管理执行端版本。
- 问题闭环：汇总问题，组织复测、回归验证与修复知识。
- 已有用例分析：支持导入测试用例和整网用例、功能/精度/性能问题可视化分析定位，可参考：[模型分析使用说明](docs/usage.md)。

## 4. 文档汇总

| 文档 | 内容 |
|---|---|
| [申请加入与用户管理](docs/user_registration.md) | 申请、审核、领取 token 和账号管理 |
| [客户端使用说明](docs/client/configuration.md) | 初始化、设备选择、配置、重放与更新 |
| [客户端升级](docs/client/upgrade.md) | 单安装更新、回退与旧安装接入 |
| [进阶命令](docs/client/advanced.md) | 手工管理任务、环境及临时调试 |
| [模型分析](docs/usage.md) | 模型运行、精度定位和报告 |
| [功能示例](docs/feature_gallery.md) | 精度比较、图展示和生命周期分析 |
| [Agent 配置](docs/agent_normalization.md) | 归一化与失败诊断 |
| [平台架构](docs/architecture.md) | 模块职责、进程、工作负载、数据与接口边界 |
| [中心服务架构](docs/server/architecture.md) | 问题闭环、知识库及接口 |
| [OpenAPI 接口](docs/openapi.md) | 下载 `/openapi.json` 与自动生成方式 |
| [服务端部署与备份](deploy/platform/README.md) | 安装、身份发放、备份与恢复 |
| [运维说明](docs/operations.md) | 环境管理、数据维护、构建与发布 |
| [执行引擎架构](docs/fuzz/architecture.md) | 用例生成、调度、执行与存储 |
| [ATen 算子清单](docs/fuzz/aten_op_list.md) | 生成器算子目录 |
| [精度映射](docs/accuracy_mapping.md) | CPU、NPU eager 与 Autofuse 的节点对应关系 |
| [Manifest 格式](docs/manifest_schema.md) | 采集字段和状态 |
| [Inductor / AscendC 流程](docs/inductor_ascendc_pipeline.md) | 编译执行链路 |
| [模型分析架构](docs/model_architecture.md) | 脚本运行、采集、比较与报告 |
