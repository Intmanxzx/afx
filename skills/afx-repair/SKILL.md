---
name: afx-repair
description: 通过 Autofuse 中心 API 领取问题，使用 npussh 调试、构建与安装 CANN RUN 候选包，提交复测并归档修复经验。用于用户授权的 Autofuse 问题修复闭环。
---

# Autofuse 问题修复

本地 Agent 使用 HTTP API 领取问题，通过 npussh 调试；不要求本地安装 afx。中心维护唯一问题状态与经验记录。API 约定见当前工作区 docs/fuzz/agent_api.md；本仓库通常位于 /Volumes/GM9/code/Autofuse-Debugger，代码位置变化时使用实际工作区。API 尚未部署或缺少凭据时明确指出，不能把本地记录当作中心已认领。

## 领取和保活

- 使用用户提供的中心 HTTPS 地址和私有用户 token；不要把 token 写入代码、总结或命令输出。设备 token 不是 Agent 凭据。
- POST /api/agent/repairs/claim，传稳定的 agent_id（本次 Agent 实例唯一）和 request_key，可选 issue_key。保存返回的 id、lease_id、expires、snapshot。
- 同一次请求重试复用 key 和原 body；新操作用新 key。没有任务时返回 data=null，下一次轮询用新 key。
- 租约 30 分钟，至少每 10 分钟调用 repairs/ID/renew。长编译用后台进程并持续续租。409 表示已过期或被接管，停止该问题的写操作，不提交旧结果。
- 必须以中心领取成功为准。已有其他 Agent 领取时不要自行并行修复。无法继续时 release；记录检查点后交还租约。

## 调试与构建

- 从 snapshot.targets 和 examples 获取原始 attempt、seed、Shape，使用公开 /api/attempts/ID 读取完整诊断。检查旧 seed 是否仍可重放。
- 使用 npussh mlir，先加载 $HOME/env.sh 和项目实际运行时；检查两张 NPU 的占用及既有任务。使用隔离 home、日志和代码目录，不停止其他任务。
- 在远端使用 afx fuzz replay SEED 等实际帮助支持的命令读取详细日志，固定原输入、后端和评估语义。不要以更换 seed 或放宽精度掩盖失败。
- 修改和构建遵循目标仓库 AGENTS.md 与现有构建流程。编译出 RUN 包无需集成进 AutofuseX。记录代码提交/补丁、构建命令和产物 SHA256，不能拿旧包冒充新构建。

## 安装 RUN 候选包

用户授权修复与候选包验证时，可以在隔离目录安装候选包。替换正式环境、停止正式任务或切换生产服务超出隔离验证范围时，先确认已有授权是否覆盖。

1. 检查包的实际 --help/厂商安装说明、架构与当前 Python/CANN 配套要求；不要猜测所有 RUN 包支持相同参数。
2. 校验 SHA256，在新候选目录使用该包已确认的非交互、独立安装前缀参数。保留原环境，禁止直接覆盖在用 CANN 目录；不自动 sudo。
3. 加载候选环境脚本，检查 Python 实际导入的 autofuse、compile_adapter、pyautofuse 和 MLIR codegen 动态库来源与版本，避免旧 PYTHONPATH 遮蔽新包。仅修改当前隔离入口。
4. 执行原 seed 的 CPU/AF-MLIR 对照，检查目标 NPU；记录实际包版本、SHA256 和安装时间。安装成功不等于问题修复。
5. 将专用复测客户端绑定到这个候选运行环境并确认心跳、客户端 ID；API repairs/clients 列出当前用户的复测客户端。重启的是专用隔离客户端，不能覆盖正式客户端身份、ACK 或结果。

## 中心复测与总结

- POST repairs/ID/retest，带 lease_id、request_key、client_id、artifact（sha256、version、带时区的 installed_at）。制品字段是 Agent 提供的追踪证据，不是中心独立证明加载了该二进制；必须完成上述实际路径检查。
- 中心复测整个领取快照，不用任务调度 completed 代替测试通过。GET repairs/ID 等待 retest.terminal=true；持续续租。失败时读取 batch_ids/attempt_ids 和详细错误，回到调试/构建/安装，再用新 key 发起复测；在途复测不可重复提交。
- retest.passed=true 后 POST repairs/ID/summary，带 lease_id、request_key、background、solution、notes。背景包含失败现象/触发条件，方案包含根因和修复，备注记录提交、候选包摘要、环境、验证范围及限制。不要上传凭据、私有路径中的秘密或完整敏感日志。
- 中心返回 state=resolved 和 summary.id 才算闭环完成；同一请求重试返回同一总结。查询 GET /api/agent/experience 获取已有经验。没有通过验证时不得编造总结或声称已关闭问题。
