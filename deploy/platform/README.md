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
