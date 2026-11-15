# Fuzz 文档

Fuzz 平台负责随机图生成、CPU/NPU 验证、设备与任务管理，以及结果汇总。Agent 的本地控制入口是 `afx` 管理命令，跨机器查询使用中心 HTTP API。

| 文档 | 内容 |
|---|---|
| [使用说明](fuzz_usage.md) | 安装、提交任务、查询状态、重放 |
| [整体架构](fuzz_design.md) | 控制、执行、数据流及统计语义 |
| [Agent 与 API](agent_api.md) | JSON 契约、异步操作、鉴权和当前限制 |
| [运维说明](operations.md) | 环境准备、身份、部署、备份与发布 |
| [ATen 算子清单](aten_op_list.md) | 当前生成目录，按精确 overload 列出 |

这里只维护当前实现及使用约定。参数全集以命令 `--help` 和配置 schema 为准；开发计划、实验与部署过程通过 Git 历史追溯。
