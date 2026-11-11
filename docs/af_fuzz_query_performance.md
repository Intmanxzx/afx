# 数据查看性能优化（2026-11-10）

## 定位与修改

在 oracle1 约 235MB、9890 attempts / 13391 shapes 的在线数据库上分析查询计划，并通过 SQLite backup 创建私有一致性快照。没有清空结果或修改身份。

- `attempt_facts(group_id,attempt_id)` 覆盖索引避免分组扫描读取宽 execution/diagnostics 字段。
- 用窄键 `max(rowid)` 选择每组最新 backend，再读取其上下文；不改变重试覆盖规则。
- 无筛选列表按 rowid 窗口选择候选组，完整保留每组 backend/history。候选不足时扩展窗口；游标及 Shape 顺序保持不变。
- 文本/SQL 搜索采用一次性匹配集合；Pass/Failed 等状态直接筛聚合状态，避免构造日志搜索投影。
- 图表在同一只读事务中复用 connection-local latest/cases/attempts 临时表，避免重复聚合和重复解析日志 JSON。没有引入持久结果缓存。
- 列表独立于汇总返回后渲染；汇总失败显示错误，不将已有列表当成空结果。

## 验证

同一私有快照，原 `venv-query-fix-1110` 与优化代码对照：20 条列表、汇总、Failed 列表、完整统计 JSON 排序序列化后的 SHA256 全部一致。该次串行测量如下（秒；线上负载与缓存会引起明显波动，不是 SLA）：

| 查询 | 原实现 wall / CPU | 优化 wall / CPU |
|---|---:|---:|
| 20 条列表 | 2.212 / 1.068 | 0.097 / 0.096 |
| 汇总 | 1.586 / 1.469 | 0.716 / 0.715 |
| Failed 列表 | 2.408 / 1.777 | 0.900 / 0.820 |
| 完整统计（latest 复用阶段） | 12.906 / 5.342 | 3.790 / 2.810 |

最终版进一步将 telemetry joins 改为窄 selected_attempts 表，完整统计 JSON 校验值仍一致；部署后 loopback 串行验证：列表 0.202s、汇总 0.889s、统计 3.210s、Failed 0.900s、无匹配的 transpose 全文搜索 2.601s，全部 HTTP 200。全文子串搜索仍需扫描，尚未实现大规模全文索引；不应把当前结果外推到百万记录。

查询、Shape、同步、存储、统计与 viewer 相关测试：104 passed / 2 skipped。新增多窗口分页、重试、部分 Shape、无 Shape 错误、极大游标与完整 rollup 对照。另一次包含 hosting 的测试出现 6 个调度 SQLite 锁失败，非本次查询改动范围；没有宣称全套测试通过。

## 部署与回滚

分析过程中另一项采样工作发布了 `venv-sampling-1110`；最终以它为基底创建 `/home/ubuntu/af-fuzz/venv-query-performance-1110`，只叠加 reporting/cases.py、reporting/index.py、reporting/statistics.py、storage/queries.py、view/static/app.js。采样实现保持不变。

服务 override：`/etc/systemd/system/af-fuzz.service.d/query-performance-1110.conf`。私有一致性备份、原服务配置位于 `/home/ubuntu/af-fuzz/private/query-performance-1110/`，不得发布或入库。上线后核对备份内用户、来源、attempt ID 均未丢失，线上继续接收新结果。

回滚前核对是否已有更新部署。若仍为本版本，将此 override 移出 `.conf` 后缀范围，执行 `systemctl daemon-reload` 和 `systemctl restart af-fuzz.service`，回到主服务的 sampling 环境。不要恢复旧数据库覆盖新上传；新增索引可保留。

最终校验：生产库 `integrity_check=ok`；上线前备份中的 1 个用户、5 个来源及 10042 个 attempt ID 均保留（检查时线上已增至 10133 条 attempts）。线上 app.js 与本地 SHA256 一致。浏览器实测记录页加载、下一页成功。
