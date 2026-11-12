# 验证期身份白名单操作

白名单启用时只读网页保持开放，未允许客户端写入返回 503。当前验证期白名单已关闭，所有有效身份可正常写入。配置不保存 Token。

## 配置

在服务器私有配置目录创建 JSON 文件，例如 `/home/ubuntu/af-fuzz/write-allowlist.json`：

```json
{
  "enabled": true,
  "users": ["替换为验证账号的用户UUID"],
  "clients": [
    {"user_id": "替换为验证账号的用户UUID", "client_id": "替换为验证客户端source_id的UUID"}
  ]
}
```

占位符不能直接使用，必须填真实 UUID。配置路径由环境变量 `AF_FUZZ_WRITE_ALLOWLIST` 指定，例如 systemd 服务环境项：

```ini
Environment=AF_FUZZ_WRITE_ALLOWLIST=/home/ubuntu/af-fuzz/write-allowlist.json
```

建议文件权限 0600，服务运行用户可读。与已有 systemd override 合并，不覆盖其他配置。启动时读取；缺失、错误、超限的文件会让服务拒绝启动。`enabled: true` 且两个列表为空表示拒绝全部业务写入。

账号 ID 来自鉴权后的用户信息；client_id 是目标客户端结果库 `sync_identity.source_id`，也用于注册与复测请求，不是 machine_id。使用隔离验证目录时必须填写该目录自己的 source_id。白名单允许注册尚未登记的指定来源。现有运行心跳仍使用 config.json 的 source_id（安装身份）；如果与结果库 source_id 不同，恢复同一客户端时须同时允许这个已确认归属的身份，不要重写身份或 ACK。

## 范围

- `/api/upload`、`/api/facts`、`/api/heartbeat`：验证用户与 source_id。
- `/api/retest/register|claim|renew|complete`：验证用户与 client_id。
- 网页复测 preview/create：验证用户与目标 client_id；取消：验证用户，仍检查任务归属。
- 登录、退出、健康检查、公开结果/历史查询、自己的任务查询正常开放。

被拒绝时返回 `503` + `Retry-After: 60`，错误码 `maintenance_write_blocked`。原始 Token 不失效，结果和事实 ACK 不推进。现有客户端按自身间隔重试，不会因 503 清空待上传队列。白名单不提供独立于用户 Token 的设备认证，也不能区分复制了同一 Token 和 source_id 的进程。

## 启停

1. 停止自己的正式客户端任务，确认租约和正在执行的任务，备份正式数据库。
2. 填写白名单并在受控维护窗口重启服务端；用允许/不允许的两个身份分别验证。
3. 启动隔离验证客户端，执行复测验收。
4. 分阶段恢复时，只把获准的正式 source_id 加入 clients，保持 enabled=true，重启服务；先同步积压，再恢复原 campaign。全部放开须另行批准，届时才将 enabled 改为 false 或移除环境变量。
5. 确认其他客户端恢复上传；其积压数据可以成为最新结果，这是既定入库顺序策略。

配置修改不会热更新。白名单也不会冻结租约或停止远端执行。不要通过删除用户、旋转 Token、重置 ACK 或返回伪成功来阻断上传。

2026-11-12 已按用户批准完成 10 分钟观察并关闭验证期白名单（enabled=false），重启服务生效；所有有效身份恢复上传、心跳与复测控制请求。正常 Token 鉴权、任务归属和资源保护不变。关闭前配置及观察日志保存在服务器 private/retest-validation-1112/。非白名单随机 source 的无效上传探针返回 400 而非维护 503，未写入测试数据；尚未观察到其他客户端主动续传。
