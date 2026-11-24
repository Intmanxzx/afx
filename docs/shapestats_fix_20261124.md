# 用户运行概览 Shape 统计修复

2026-11-24 北京时间 11:22:31–11:22:35，在 ssh afx 上切换 afx-platform.service。

原用户 Shape 汇总依赖全局最近 100 条 efficiency_windows，未过滤已删除客户端和 24 小时范围。zhixiong 当前客户端没有窗口，已删除客户端在 11 月 21 日留下的 5 个窗口却被按用户合并，因此页面显示成功 54、完成 135、失败 81。原无窗口回退还将执行条数当作 Shape 数。

现在以 execution_shapes 明细统计最近 24 小时非 CPU 执行，含复测，排除已删除客户端。功能通过且精度未失败计成功，其余 Shape 计失败，完成等于成功加失败。无 Shape 明细的失败执行单独计 Case 错误。窗口保留作诊断，不再决定用户总数。页面增加口径说明。

部署目录：/srv/afx-platform/deployments/platform-shapestats-82711f01061b74c8。从原 platform-f647f264cabc4c06 独立复制，仅替换 server/ops.py 和 server/static/page-ops.js；不发布 PyPI 或升级客户端。

私有停机备份：/srv/afx-platform/backups/pre-shapestats-20261124T032231Z，包含完整 data、原 service 和 launcher。三个备份数据库 integrity_check 均为 ok；secret.key 和 client-access.json 在切换后逐字节一致。原部署目录保留。

验证：tests/platform 共 105 项通过；page-ops.js 语法检查通过。候选查询线上只读数据库耗时 0.14 秒。部署后本机和公网 healthz、summary、efficiency 和更新的静态资源均通过，浏览器确认新数字和口径说明。五台客户端心跳恢复，服务 active、NRestarts=0。

11:23 页面快照：zhixiong 执行 5204，完成 Shape 6343、成功 2568、失败 3775、Case 错误 1731。稍后同一只读数据库事务中将新函数结果与独立按 Shape 状态分组核对：执行 5210，完成 6355、成功 2571、失败 3784，逐项一致；数据随运行增长。

回滚：停止 afx-platform.service，将该备份的 afx-platform.service 恢复至 /etc/systemd/system/，afx 恢复至 /srv/afx-platform/bin/，daemon-reload 后启动。此次没有数据库迁移，回滚代码应保留当前 data，避免丢弃后续上传记录。

归档说明：上述内容记录 11:22 的部署版本。整理本地改动时，最新 upstream/master（ae426e1）已包含 Shape 统计修复和对应回归测试，且后续将 Case 错误调整为按唯一用例统计：同一用例在统计窗口内没有任何 Shape 明细、并且至少一次执行失败时计 1 个错误用例。归档时保留 master 的后续实现和页面说明。
