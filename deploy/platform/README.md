# 平台服务端备份

备份放在另一台机器上；本机只在系统盘上临时生成归档，上传后立即删除。

| 内容 | 是否备份 | 说明 |
|---|---|---|
| core.db | 每 6 小时 + 每天 | 用户、用例、执行记录、问题、知识，最重要 |
| content.db | 每天 | 日志与用例构图描述，只追加 |
| secret.key | 每次 | token 校验密钥，丢失后所有 token 失效 |
| stats.db | 不备份 | 启动时自动从 core.db 重建 |

快照使用 SQLite 在线备份接口，服务运行中也能得到一致副本；core 先于 content 快照，保证 core 引用的内容都在。每个归档上传前都会做 SHA-256 和 `integrity_check` 校验，未通过不上传。

## 安装

1. 远端机器建目录（如 `/srv/backups/afx-platform`），为本机生成专用 SSH 密钥并加入远端 `authorized_keys`。
2. 复制 `backup.sh` 到 `/opt/afx-platform/deploy/`，按 `backup.env.example` 写 `/etc/afx-platform/backup.env`（0600）。
3. 安装 `afx-platform-backup@.service` 和两个 timer，`User=` 改为运行服务端的用户：

```bash
sudo systemctl enable --now afx-platform-backup-full.timer afx-platform-backup-core.timer
```

4. 手动跑一次确认：

```bash
sudo systemctl start afx-platform-backup@full.service && journalctl -u afx-platform-backup@full.service -n 20
```

远端保留：core 归档 7 天、完整归档 14 天（`BACKUP_KEEP_*` 可调）。按 300 万执行记录估算，完整归档约 2～3 GB，远端预留 50 GB 左右即可。

## 恢复

只能恢复到空目录，恢复前先停服务：

```bash
afx platform --data /data/afx-platform.new restore afx-platform-20261117T033000Z.tar
```

用较新的 core 归档恢复时，用最近一次完整归档补 content：

```bash
afx platform --data /data/afx-platform.new restore afx-platform-20261117T153000Z-core.tar --content-from afx-platform-20261117T033000Z.tar
```

两次快照之间新增的日志无法找回，执行详情会标记 `log_missing`，不会伪造内容。stats.db 在服务启动时自动重建。

## 定期演练

每月在远端执行一次，确认备份能用：

```bash
afx platform --data /tmp/unused verify-backup /srv/backups/afx-platform/afx-platform-最新.tar
```

## 服务端安装与升级

```bash
bash deploy/platform/install_server.sh --wheel autofusex-0.2.0-py3-none-any.whl --root /home/ubuntu/afx-platform --port 8100
```

- 每个 wheel 安装到独立 venv（`ROOT/deployments/platform-<摘要>`）。脚本先用新代码打开（并迁移）数据目录做启动检查，再写 systemd 单元 `afx-platform`（只读系统、仅 `ROOT/data` 可写、内存上限 600M），重启服务，最后检查 `/healthz`。
- `ROOT/bin/afx` 指向当前版本。用户与 token 离线发放：

```bash
ROOT/bin/afx platform --data ROOT/data user add alice --role developer
ROOT/bin/afx platform --data ROOT/data token issue alice --kind client   # client | agent | web
ROOT/bin/afx platform --data ROOT/data token list
ROOT/bin/afx platform --data ROOT/data token revoke afx_XXXXXXXX
```

- Caddy 等 TLS 代理把站点反代到 `127.0.0.1:8100`。服务会在响应前读完请求体，以防代理复用连接时残留的请求体污染下一个请求。

## 客户端（NPU 机器）

客户端只读取 `afx fuzz run --save` / `afx fuzz replay --save` 写出的本地结果库，上传进度保存在旁边独立的 `platform-sync.sqlite` 中，不与旧版同步状态共用。

```bash
afx fuzz run --save --backend af-mlir --backend af-legacy --device npu:0 --device npu:1   # 不带 --count 即为持续运行
afx platform client run --endpoint https://autofuse.online --token-file client.token \
  --results $AUTOFUSEX_HOME/results/results.sqlite --retest --device npu:1
```

- `run`：持续上传，每 60 秒发送一次心跳（携带计数），开启 `--retest` 时还会领取复测任务（问题验证、回归和手动复测），通过 `afx fuzz replay` 执行后上传并回执结果。
- `once`：上传一次并发送心跳；`status`：查看上传游标，以及可用于验证的已安装包（graph-autofusion Version 与 `pyautofuse.so` 的 SHA256）。
- 管理员在“平台运维”页点击一键更新后，客户端在下一次心跳时 `pip install autofusex==版本` 并重启自身进程；正在进行的 fuzz 任务需另行重启。
- 已知限制：当前执行引擎不支持严格的 NPU kernel 计时（replay `--perf` 会报 `strict NPU measurement requires a device lease; not implemented yet`），因此 Kernel 对比与能力画像暂时只能使用其他来源的 kernel 数据。
