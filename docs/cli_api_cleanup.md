# CLI 与 API 清理记录

总体方案见 [统一平台重构设计](platform_refactor_design.md)，本文记录第一阶段实际清理范围。

本轮统一用户配置入口，移除无业务实现的公开 HTTP 路由及废弃控制入口。尚未进行目录重构。

| 接口 | 处理 | 调用依据 |
|---|---|---|
| config show/edit/reset/validate/export | 统一到 modern_cli；校验与编辑均接受任务配置 | CLI 与平台个人配置共用任务 schema |
| config effective/apply，旧 policy validate/export | 移除用户入口 | effective_policy 仍由内部执行策略使用，不作为第二套用户配置 |
| management trust/status | 移除 | trust 写入仅由此 CLI 调用；status 与已有入口重复 |
| 本地 IPC trust/effective | 移除 | 已删除对应用户调用方；自动初始化和内部策略读取不依赖这两个 action |
| POST /api/import/baize | 移除路由 | 仅返回 501，无导入实现 |
| POST/GET /api/models/jobs，GET /api/models/jobs/{id} | 移除路由 | 仅返回 501；模型页面只显示规划提示，无 API 请求 |
| /api/config/me、default、operators | 保留 | Web 配置、客户端同步和 CLI 使用 |
| 心跳、执行上传、复测领取与回执 | 保留 | 客户端和问题闭环使用 |
| 本地 submit/query、升级排空/恢复、shutdown | 保留内部职责 | daemon/supervisor/任务执行依赖 |
| 内部 config.apply 与 effective_policy | 保留 | runtime 远程配置同步及执行资源检查仍使用 |
| trust.json 读取及自动初始化 | 保留 | 任务资源检查、环境安装、构建与签名升级依赖 |
| device 配对及旧 central 管理协议 | 本轮保留 | runtime 仍含轮询及已持久化 credential 消费逻辑，需整套客户端迁移时处理，不能单删入口 |

config show 返回配置和来源；export 输出纯 JSON，支持直接再次 edit/validate。托管模式继续使用 /api/config/me；本地 reset 保证 results.upload=false。旧命令不保留别名。

“无需用户直接使用”不等于“内部无人调用”。内部控制协议应在目录重构时明确归属，不应因未出现在日常帮助中而删除。
