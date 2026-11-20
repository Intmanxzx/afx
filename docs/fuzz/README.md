# Fuzz 文档

Fuzz 平台负责随机图生成、CPU/NPU 验证、设备与任务管理，以及结果汇总。本地控制入口是 `afx` 管理命令；结果上报、问题闭环等云端功能由独立的平台服务（`autofusex/platform/`）提供。

| 文档 | 内容 |
|---|---|
| [使用说明](fuzz_usage.md) | 安装、提交任务、查询状态、重放、JSON 契约、鉴权和当前限制 |
| [整体架构](fuzz_design.md) | fuzz 测试引擎：daemon/client、seed 协议、执行与统计语义 |
| [平台业务与数据设计](platform_v3_design.md) | 一站式开发平台：问题闭环、知识库、数据库与 HTTP API |
| [运维说明](operations.md) | 环境准备、结果数据库维护与发布 |
| [ATen 算子清单](aten_op_list.md) | 当前生成目录，按精确 overload 列出 |

这里只维护当前实现及使用约定。参数全集以命令 `--help` 和配置 schema 为准；开发计划、实验与部署过程通过 Git 历史追溯。
