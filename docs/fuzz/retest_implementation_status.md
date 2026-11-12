# 测试记录聚合与复测实现状态

日期：2026-11-12。本文件按当前实现覆盖更新，不保留旧阶段的相互矛盾结论。

## 实现范围

- Token 登录和受保护控制接口；未登录不显示复测/选择操作。登录弹窗去除多余说明，取消/登录同行排列；桌面导航靠左、更新时间与登录靠右；服务端标题为“Autofuse一站式工作平台”，配套绿色融合节点 SVG 图标。
- 验证期身份写入白名单：启动时加载本机配置，按鉴权用户 + source/client ID 限制全部客户端写入；网页写操作按账号限制，预览/提交还限制目标客户端。未允许者返回可重试 503，登录/退出/查询正常。操作说明见 `write_allowlist.md`。
- 单条、批量及跨页选择；同配置多 Shape 合并为一次完整 CPU + AF-MLIR 执行。
- 客户端默认注册与领取；每次最多 8 项，持久租约、恢复、取消、ACK 后完成关联。
- 使用现有 staged 调度器和物理设备锁；不抢占用户任务。
- 全站 seed + shape_index 最新完整轮次；不按用户或后端拆分外层聚合。
- 先选当前再筛选，AF-MLIR 主质量指标；普通上传与复测采用相同规则。
- 新执行 round_id 区分多后端轮次；历史兼容边界详见设计文档。
- 持久单调首次接收序号和时间；重传不改顺序，旧记录不伪造接收时间。
- 当前行折叠、历史背景、三角展开、按需分页；普通来源空、复测来源“复测”。
- 可空 summary_id 以及详情展示占位；未实现总结库/agent 总结接口。
- 增量统计贡献粒度改为 seed；当前轮次变化会重新计算该 seed 的所有 Shape。
- 快照/恢复保留首次接收信息；拒绝不受信任的结果表触发器。

## 结构化数据库优化（已受限部署）

代码已实现结构化当前/历史索引、预计算失败展示、常用字段展开、受影响 Seed 事务内维护、脏数据安全回退、显式 `maintenance index-cases` 迁移。原始 JSON/日志及身份不变。

- 初轮兼容回归 mlir **132 passed**；开启结构化模型后的原有回归 **130 passed**（`model-enabled-final.log`）。
- Docker 开启模型回归 **108 passed、1 skipped**（Node 项），独立模型/导入/恢复/清理组 **32 passed**；对应 mlir **32 passed**。
- mlir 真实双卡复测通过：npu:0 / npu:1 各 3 Shape Pass、任务 completed，报告 `/mnt/workspace/af-fuzz/retest-1112/model-npu-e2e/report.json`。
- Docker HTTPS Chromium 在启用结构化模型后验证登录、3 Shape 合并任务、上传自动刷新、历史展开、退出成功；无 JS 错误，日志 `/tmp/af-model-ui.log`。
- 生产历史副本 25,988 attempts / 20,673 当前 Shape：新旧当前行逐行一致、完整统计相等、integrity_check 通过；旧当前聚合 193 秒、旧统计 345 秒，回填 371 秒。直接读取结构化列表 10 条 0.91 秒、200 条 3.29 秒、summary 1.60 秒。移除统计对宽 attempt_facts 的无效关联后，统计约 19.21 秒；单 Seed 刷新约 0.257 秒。这些是同机观测值，受资源争抢影响，不是 SLA。
- 最后统计改动 mlir **22 passed**；Docker **21 passed、1 Node 跳过**。缓存退出/模型补充两端 **14 passed**。
- 正式库迁移前私有一致备份已完成，目录 `/home/ubuntu/af-fuzz/private/model-deploy-1112/`，保存 25,992 attempts、39,170 shapes、5 users、7 sources、42,758 receipts；正式迁移约 99.68 秒，原始表数量及所有状态计数不变，外键检查通过，白名单未解除。正式库直接查询 summary 0.50 秒、page10 0.81 秒、Fail 列表 0.69 秒（非公共响应缓存）。
- 迁移后真实 HTTPS → mlir → 上传 → 当前索引更新闭环完成：批次 `48d998ba-7158-4eb1-8731-892e4895429a`，attempt `d5ef2859-5189-45a8-912c-64689bce36fc`；3 Shape 全 Pass，历史数均为 3，dirty 队列为 0。报告 `/mnt/workspace/af-fuzz/retest-1112/production-validation/model-completed-report.json`。独立验证进程完成后停止；正式客户端现已按用户批准恢复，见当前受限部署状态。
- 正式服务浏览器历史验收通过（通过 SSH 隧道访问 oracle1，公网 Chromium 导航重试遇到网络超时）：3 个当前 Shape，可展开 2 条历史，无 JS 错误；截图 `/tmp/af-model-production-history.png`。mlir 到公网 HTTPS 的提交、查询、上报闭环通过。
- **未解决的容量问题**：副本统计 19 秒不能作为生产 SLA；生产完整统计一次在 90 秒截止时间中断，单连接 32 MiB 缓存和内存临时表诊断测试约 96 秒。该 PRAGMA 仅用于诊断，未作为默认配置部署。结构化读取/增量写入已交付，但全量统计扫描阶段指标等仍需继续优化，不解除白名单。

## 验证记录

### 界面改版验收

- Docker Chromium 验证 1440/1024/390 三种宽度：标题、图标、更新时间与登录顺序、弹窗按钮同行、无页面横向溢出通过，无 JavaScript 错误。日志 `/tmp/af-branding-ui.log`。
- mlir 品牌路由与会话回归 **2 passed**，使用 Node 20；日志 `/mnt/workspace/af-fuzz/retest-1112/branding-final.log`。
- 仅本次界面相关文件已更新至 oracle1 现有受限发布目录；线上 Chromium 检查标题、弹窗、顶栏顺序、favicon 和 healthz 通过。回滚文件保存在 `/home/ubuntu/af-fuzz/private/branding-1112/`。该轮仅验收界面；后续生产查询、复测闭环和正式客户端恢复状态见下文。

### Docker

隔离容器 `af-retest-final`（验收后已删除），Python 3.11，临时数据库和测试身份。未使用生产数据或凭据。

- HTTPS Chromium：登录、未登录隐藏按钮、3 条 Shape 选择合并 1 项任务、结果自动刷新、历史展开/折叠、来源标记、详情、退出登录、24 轮记录加载更多及当前 Fail 筛选下展开过去 Pass 通过；无 JavaScript 错误。
- 日志 `/tmp/ui-global-v4.log`；截图和验收日志已导出到 `/tmp/af-retest-delivery/`。
- 2,000 轮历史规模验证：汇总约 0.023 秒、首页约 0.025 秒、统计约 0.045 秒；10 次分页读取全部 1,999 条历史，约 4.242 秒，无重复/缺失。
- 规模日志 `/tmp/scale-verified.log`；这些是隔离数据测试，不是生产全量容量承诺。
- Docker 验收组 **137 passed**；聚合/历史/增量统计组 **40 passed**；最终迁移/传输补充组 **29 passed**；上传刷新/缓存组 **19 passed**。测试组有重叠，不将数量相加。日志分别为 `/tmp/docker-acceptance-final.log`、`/tmp/final-query-tests.log`、`/tmp/docker-delta.log`、`/tmp/refresh-final.log`。

- 白名单专项与控制面/消费端/同步回归：Docker **41 passed**，日志 `/tmp/af-whitelist-docker.log`。覆盖全部写路由拦截、同用户不同客户端、冒用允许 source_id、无效 Token、ACK 保留、重启解禁续传、消费端 503 不取消本地租约以及允许身份完成复测。

### mlir

隔离源码 `/mnt/workspace/af-fuzz/retest-1112/src`；加载 `$HOME/env.sh` 和项目运行时、新版 SQLite 动态库。

- 真实 AF-MLIR 动态 D16：npu:0 / npu:1 均完成，每张卡 3 个 Shape 全部 Pass。
- 验证跨用户复测、批量 Shape 合并、CPU 参考配对、当前结果更新以及历史数量。
- 相同 workload：`f89eb706419a8d536e4256b25e9cd8b0c8998dd371bee7aae630028c564599ec`。
- 证据 `/mnt/workspace/af-fuzz/retest-1112/e2e-afmlir-v2/report.json`。
- 第二次真实 AF-MLIR 双卡复验通过；最新迁移/传输/同步/缓存补充组 **48 passed**（`final-delta.log`）。
- 扩展软件回归 **218 passed，2 个 JavaScript 测试因系统 Node 10 不支持现代语法失败**（`global-acceptance-final.log`）；使用隔离 Node 20 重跑这两项及会话测试，**3 passed**（`node20-final.log`）。最终所选回归项全部验证通过，未替换系统 Node。
- 曾尝试将临时数据库放入 `/dev/shm` 提速，但该挂载仅 64 MB，因空间耗尽导致该次验证无效；已清理本任务临时目录，恢复空闲空间。以上有效结论均来自常规磁盘回归及 Node 20 补测，不使用该次失败运行作为通过证据。

- 白名单同组回归：mlir **41 passed**，日志 `/mnt/workspace/af-fuzz/retest-1112/write-allowlist-tests.log`。该组隔离回归期间未启用生产白名单、未停止现有客户端；后续受限部署状态见下文。

## 当前受限部署状态

用户已批准恢复自己的正式 mlir 客户端；现已恢复原 campaign 和上报，服务端继续以白名单运行，不是全量放开服务。

- oracle1 备份目录：`/home/ubuntu/af-fuzz/private/retest-validation-1112/`；备份原库 25,988 attempts、5 users、6 sources、42,754 receipts，数据库完整性通过。
- 发布目录：`/home/ubuntu/af-fuzz/releases/retest-validation-1112/`，override：`zz-retest-validation-1112.conf`。原 venv 和配置保留。
- 白名单允许 NiuYH 用户、独立验证客户端及正式客户端。正式上传/复测 source_id 为 `7ddb931b-1b06-4763-a80c-b31aff42a585`；现有心跳使用 config.json 中的安装身份 `314d5de1-2a64-40c1-99cc-8a7edf1e2307`，已同时放行。其他身份继续受限。HTTPS 注册/登录 200，非白名单来源上传 503 已实测。Token 不存入文档。
- mlir 正式目录 `/mnt/workspace/af-fuzz/niuyh-home` 已恢复原 campaign，未重新初始化 payload、next_device、任务或 ACK。恢复前一致备份位于 `/mnt/workspace/af-fuzz/retest-1112/formal-resume/`。沿用持久预算：4 workers、每编译 8 threads，双卡 npu:0 / npu:1，加载 `~/autofuse/build`。验收时 completed 从 7,222 增至 7,260，4 running、6 pending；两张物理卡 12/13 均观察到工作进程。上传 ACK 从 7,221 增至 7,260，pending=0；同步、复测轮询、运行心跳均 ok。恢复日志 `/mnt/workspace/af-fuzz/retest-1112/resume-formal.log`。
- 独立客户端：`/mnt/workspace/af-fuzz/retest-1112/production-validation/client`；CPU + AF-MLIR D16 基线完成，3 Shape Pass。真实 HTTPS 控制面批量提交 3 个 Shape，合并为 1 项任务并在第一代租约完成；CPU/AF-MLIR 结果均已上报。独立验证消费进程在完成后停止，不与正式客户端重复运行。
- 用户新构建位于 `~/autofuse/build`，解析为 `/mnt/workspace/gitCode/niuyuhu/graph-autofusion/build`。实际加载 codegen 文件与该目录一致，SHA256 为 `71352eeaaa5509975caab307fc1755cccd736542bcede5afa8c1834d15cf9508`。Python 辅助包仍来自完整 package-closure 运行时。
- 上线检查发现生产规模查询慢，并多次观察到 87%–96% CPU steal。进一步移除中间结果中的宽执行日志、复用同轮 Shape 成员读取，将统计双次全站聚合合并为单次，精确短 Seed 搜索前推至入口；全站重查询采用单并发预算，避免挤占上传/续租。主机独立全量统计测得约 165 秒，后台回退预算调整为 300 秒，交互查询预算不变。最终重启后约 90 秒缓存就绪；线上 summary/statistics/cases 三接口均 200、cache=ready，HTTPS 响应分别约 0.45/0.52/0.24 秒。浏览器首页正确显示总数 20,676、Pass 13,551，无 JS 错误（截图 `/tmp/af-production-ready.png`）。仍需持续验证全量服务容量，不据此恢复一般上传。
- 线上目标列表/历史 API 与真实 Chromium 验收通过：3 个 Shape 均 Pass，当前标记“复测”，history_count 均为 2，默认折叠并能展开原记录，无 JavaScript 错误。批次 `7aaf1009-f083-4c3a-870e-faf95db00a42`，新 AF-MLIR attempt `45c3772d-ecab-4691-88b6-74fb78ce5da5`；报告 `/mnt/workspace/af-fuzz/retest-1112/production-validation/completed-report.json`，截图 `/tmp/af-production-retest-history.png`。summary_id 三个均为空，预留正常。
- 性能改动回归：mlir 控制面/缓存/聚合 **55 passed**（`query-gate-tests.log`），精确 Seed/缓存 **15 passed**（`query-seed-tests.log`），预热预算 **8 passed**（`query-bootstrap-tests.log`）；Docker 对应统计组 **37 passed**、控制面/缓存组 **25 passed**、精确 Seed/缓存 **15 passed**、预热 **8 passed**。组间有重叠，不合计。覆盖/Shape/失败归因补充回归：mlir Node 20 **74 passed**（`query-coverage-final.log`），Docker **73 passed、1 个 Node 测试跳过**（`/tmp/af-query-coverage-final.log`）；该 Node 项已在 mlir 通过。补充修正旧测试对缓存版本 4 的硬编码，当前版本为 6。
- API 以白名单模式运行，正式客户端已恢复；service active、healthz 正常，继续上传后结构化 dirty 队列为 0。已完成上述真实复测闭环，但不等同于全量服务容量验收。

## 交付边界

功能代码已提交为 `9f2d526` 并推送 `origin/refactor`。当前仅恢复用户自己的 mlir 客户端；全量解除写入白名单仍须用户另行确认。

历史数据没有显式 round_id 时无法无损推断所有乱序缺失场景；新客户端通过显式 round_id 消除该歧义。总结服务与主后端切换不属于本期。
