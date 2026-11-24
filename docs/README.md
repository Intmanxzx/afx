# AFX 文档

本目录维护当前架构、接口和使用约定。部署流水、阶段验收、实验结果和设计演进通过 Git 历史追溯。

## 平台架构

| 文档 | 内容 |
|---|---|
| [平台架构](architecture.md) | 模块职责、进程、三类工作负载与实现边界 |
| [中心服务](server/architecture.md) | 数据库、任务派发、执行事实、问题闭环与 Web |
| [Fuzz 工作负载](fuzz/architecture.md) | 生成、阶段调度、资源、比较与结果统计 |
| [模型分析](model_architecture.md) | 模型执行、图采集、精度定位和报告 |

## 客户端与运维

| 文档 | 内容 |
|---|---|
| [客户端配置](client/configuration.md) | 初始化、配置、重放、固定用例与整网任务 |
| [进阶命令](client/advanced.md) | 环境和任务管理、临时调试及异步操作 |
| [安装与升级](client/upgrade.md) | 单安装更新、回退及旧安装接入 |
| [客户端身份](client/identity.md) | 身份归属、版本事实与在线状态 |
| [平台运维与发布](operations.md) | 环境、数据库维护、镜像、构建及发布 |
| [服务端部署与备份](../deploy/platform/README.md) | 安装、身份发放、一致性备份和恢复 |
| [申请与用户管理](user_registration.md) | 申请、审核、token 和账号管理 |
| [接入控制](client_access.md) | 客户端维护白名单 |

## 接口与分析参考

| 文档 | 内容 |
|---|---|
| [OpenAPI](openapi.md) | 从实际服务生成 HTTP 接口文档 |
| [模型使用](usage.md) | 模型命令、执行阶段和产物 |
| [脚本归一化](agent_normalization.md) | Agent 配置和失败诊断 |
| [功能展示](feature_gallery.md) | 精度、图和生命周期报告示例 |
| [ATen 算子目录](fuzz/aten_op_list.md) | 当前生成器 overload 清单 |
| [精度映射](accuracy_mapping.md) | CPU、NPU eager 与融合输出的对应规则 |
| [Manifest](manifest_schema.md) | 采集字段与状态 |
| [Inductor / AscendC](inductor_ascendc_pipeline.md) | 编译和执行链路 |
