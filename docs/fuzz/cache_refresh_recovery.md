# 2026-11-11 公共缓存降载

针对 oracle1 CPU steal 高、统计超时及上传 SQLITE_BUSY，先收敛后台重算压力；本改动不解决宿主 CPU 配额/争用，也不宣称解决全部数据库写锁冲突。

- 默认后台检查间隔由 15 秒改为 30 秒，server 与 cache 共用同一默认值。
- 改为每轮工作结束后等待完整 30 秒，慢查询不再只等待 1 秒立即重跑。
- 连续失败后等待 30、60、120、240、300 秒，之后保持 300 秒；成功后恢复正常间隔。
- 保留最后一次成功快照和 stale 标识；失败不推进 data_version；数据库不变时不重算。
- 页面仍每 15 秒读取公共缓存，不因读取而触发重算。首次预热响应 Retry-After 默认 30 秒。
- 停止服务可以中断等待，不需要等满 300 秒。

## 部署记录

仅更新当前线上 `venv-failure-classification-1110` 内 `central/cache.py` 与 `central/server.py`，修改前 SHA256 与仓库 HEAD 一致。没有部署工作区其他未提交改动；只重启 af-fuzz.service，没有重启主机。

私有一致性备份：`/home/ubuntu/af-fuzz/private/cache-rest-1111-1794387500/`，包含结果库、runtime 库、原两个源码文件及有效 service 配置。备份含身份数据，禁止公开或入库。

回滚：先检查是否已有更新部署；若仍为本次发布，恢复备份的两个源码文件到原环境并重启服务。不要用旧数据库覆盖新上传数据。

验证：缓存/同步/viewer 45 项测试通过；线上默认间隔 30 秒、退避上限 300 秒；healthz、summary、statistics 均 HTTP 200。额外 hosting 测试 3 passed / 1 skipped / 6 failed，失败涉及调度 ledger 数据库锁和本机缺少 nnsmith，不宣称全套测试通过。
