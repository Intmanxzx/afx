# 客户端与执行引擎文档

AFX 平台的客户端与执行引擎负责用例生成、CPU/NPU 验证、设备与任务管理。本地控制入口是 `afx` 管理命令，平台服务负责汇总结果与问题闭环。本目录保留现有代码路径命名，平台完整文档见[根目录文档索引](../../README.md#文档)。

| 文档 | 内容 |
|---|---|
| [客户端配置与重放](client_configuration.md) | 从 `afx init` + `afx start` 开始，平台接入、设备选择和更新 |
| [进阶命令](fuzz_usage.md) | 手工管理环境和任务、临时调试、JSON 契约和当前限制 |
| [整体架构](fuzz_design.md) | fuzz 测试引擎：daemon/client、seed 协议、执行与统计语义 |
| [平台业务与数据设计](platform_v3_design.md) | 一站式开发平台：问题闭环、知识库、数据库与 HTTP API |
| [运维说明](operations.md) | 环境准备、结果数据库维护与发布 |
| [ATen 算子清单](aten_op_list.md) | 当前生成目录，按精确 overload 列出 |

这里只维护当前实现及使用约定。参数全集以命令 `--help` 和配置 schema 为准；开发计划、实验与部署过程通过 Git 历史追溯。
