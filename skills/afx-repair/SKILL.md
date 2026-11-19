---
name: afx-repair
description: 通过 AutofuseX 平台 API 领取一类 Autofuse 问题，使用 npussh 调试、构建并安装 graph-autofusion 候选包，提交验证与回归，回归通过后上传知识总结。用于用户授权的 Autofuse 问题修复闭环。
---

# Autofuse 问题修复（平台 v3）

平台地址与接口见 docs/fuzz/platform_v3_design.md 第 5、8 节。所有请求带 `Authorization: Bearer <agent token>`，写请求带 `request_key`：网络重试沿用原 key 和原 body，新操作用新 key。不要把 token 写入代码、日志、总结或命令输出。

## 1. 领取一类问题

- `GET /api/issues?dimension=function|accuracy|performance&status=open` 浏览；`POST /api/issues/claim`，body `{request_key, agent_id, issue_id?, dimension?, backend?}`。无可领取时 data 为 null。
- 返回内容包括 `lease_id`、`repro.cases`（用例 ID、芯片、复现命令、日志摘要）和 `related_knowledge`。先读相关知识：`GET /api/knowledge/{id}`、`GET /api/knowledge/search?q=…&op=…&backend=…`。
- 问题租约默认 5 天：长时间修复过程中应定期调用 `POST /api/issues/{id}/renew {request_key, lease_id}`。验证和回归期间也要续租。返回 409 `lease_lost` 表示已过期或被他人接管，此时停止对该问题的一切写操作。
- 一个 agent_id 同时只能持有一个问题；无法继续时调用 `POST /api/issues/{id}/release`。

## 2. 调试与构建

- `npussh -T mlir '…'`（不分配 tty，stdin 重定向自 /dev/null）。先加载远端环境，使用隔离目录；不要停止正在运行的 fuzz 与平台客户端（`/mnt/workspace/afx-platform`）。
- 用 repro 中的 `afx fuzz replay SEED --backend af-mlir` 复现，固定原输入和精度语义；不要换 seed 或放宽精度掩盖问题。一次修复覆盖这一类问题，而不只是其中一个用例。
- 按目标仓库的 AGENTS.md 和原生构建流程构建 graph-autofusion RUN 包；记录提交、构建命令和产物 SHA256。

## 3. 安装候选包并提交验证

- 在复测客户端所用的运行环境中安装候选包，保留原环境，不覆盖在用 CANN。核对实际导入的 `autofuse` 包路径，以及 `pyautofuse.so` 的 SHA256。
- 平台按两项匹配复测任务：`share/info/graph_autofusion/version.info` 中的 Version（即 `fix_version`），以及 `pyautofuse.so` 的 SHA256（即 `fix_sha256`）。可在客户端机器上用 `afx platform client status …` 查看 `installed` 字段确认。
- `POST /api/issues/{id}/verify {request_key, lease_id, fix_version, fix_sha256, fix_ref?}`：状态变为验证中。只有安装了完全一致包的在线客户端会领取任务，且只在问题出现过的芯片上执行。
- 可选：`POST /api/issues/{id}/regression-cases {request_key, lease_id, case_keys, reason}` 指定必选回归用例（必须在进入回归之前提交）。`GET /api/regression/preview?issue_id=` 可预览回归集。
- 轮询 `GET /api/issues/{id}`（rounds、events）和 `GET /api/issues/{id}/rounds/{n}/tasks`：
  - 有失败：问题回到修复中。阅读失败任务对应的执行日志（`GET /api/executions/{exec_uuid}`），修正后用新 key 重新 verify。
  - 只有 invalid（包不匹配或结果缺失）：检查客户端安装情况后重新 verify。
  - 全部通过：平台自动挑选回归集并进入回归中；回归通过后进入归档中。

## 4. 知识总结

- 只有在状态为归档中、且由租约持有者提交时才会被接受：`POST /api/issues/{id}/knowledge {request_key, lease_id, knowledge}`。
- knowledge 字段：
  - 必填：`kind`（bugfix_function / bugfix_accuracy / perf_tuning / fusion_pattern / pitfall …）、`title`、`symptom`、`root_cause`、`solution`。
  - 可选：`keywords`、`ops`、`dtypes`、`backends`、`graph_pattern`、`applicability`、`version_range`、`fix_ref`。
  写清触发条件、根因和可复用的经验，方便后续 Agent 检索。不要写入凭据或敏感日志。
- 返回 `status=resolved` 和 `knowledge_id` 才算闭环完成；之后的代码合入由人工完成。没有通过验证时，不得编造总结或声称已解决。
