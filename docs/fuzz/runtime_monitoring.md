# 多用户实时运行观测

## 页面与身份

导航为总览、功能&精度、工具效率、Kernel性能；旧 `#performance` 链接兼容工具效率。
服务器管理员登记用户并签发 token。客户端 identity、结果、阶段事实和心跳请求均通过 Bearer token 鉴权；服务端根据 token 确定 user-id 和用户名，不接收心跳自报用户归属。公开只读页面不需要 token。

总览仅展示结果统计和用例分布，不展示实时运行面板，也不轮询心跳。实时运行面板仅保留在工具效率页，默认全部用户，支持用户、机器、卡型号、后端、版本及 1/5/15/60 分钟范围筛选。每个用户/机器一行，每页 10 行，展开看物理卡和执行器阶段。同一机器/卡跨用户和进程在全局资源计数中去重。后端、型号、版本目前是**运行会话筛选**，吞吐是匹配会话整体产出，不是细分到单卡或单后端的基准。

机器显示名采用 `用户名-npu序号`（如 `NiuYH-npu0`、`NiuYH-npu1`），序号代表机器，不是机器内的卡号。中心 runtime 侧库按用户与机器 ID 持久保存编号，从 0 开始；重启、筛选、主机名或来源变更不会重排。存量机器升级时按机器 ID 排序补号，新机器按登记顺序追加。列表与机器筛选使用显示名，展开详情仍可核对原主机名及完整机器 ID；筛选参数仍使用原机器 ID。同一物理机器跨用户时各自编号，全局机器筛选合并别名且仅保留一个 ID 选项；资源去重不变。备份 runtime 侧库可保留编号。

用例结果分类：NNSmith 生成、CPU 参考执行以及 Torch/Dynamo/Inductor/torch_npu 编译问题统一记为“构图Failed”（`gen_failed`）；只有明确 Autofuse 错误证据（例如 strict gate `shortfall=yes`、AF codegen 失败）才记为“功能Failed”（`function_failed`）。`InductorError` 包装、生成代码中的 AF import、`shortfall=no` 均不是 AF 错误证据。当前 Shape 的明确 Torch 错误优先于其他 Shape 的共享 AF 日志；不能确认归属、设备占用、超时等显示 Unknown，不计为功能失败。执行成功但数值不一致仍单独记为“精度Failed”。

功能Failed 下仅细分日志可确认的 `mlir_failed`（MLIR 路径失败）与 `unconfirmed`（路径未确认）。优先匹配 `MLIR Codegen Result：kFailed/kSucceeded/kNotSupported`（也支持英文冒号、DEBUG 前缀及枚举整数 1/2/0）。只有 `kFailed` 确认 MLIR 失败；出现新结果标记后不再使用旧规则覆盖成功或 NotSupported。无新标记时，暂时保留下述旧日志规则，待所有生产者更新后再清理。后端名、AF import、通用 `Codegen generate kernel failed` 不足以确认 MLIR 失败。`AF_MLIR_STRICT` 的失败 gate 可作证据，`shortfall=yes/no/unknown` 只说明能力不足与否；`gate=none` 为成功，`gate=cube-fusion` 为范围外，不归为 MLIR 失败。详情保留证据原文和 Shape/运行级作用域。共享日志仅在确认只有一个 Shape 或尚无 Shape 时参与路径判断，不能根据缺少其他 Shape 日志推断是单 Shape。图校验报错不因共享日志变成 MLIR 失败。列表、统计及 SQL 筛选 `WHERE failure_path = 'mlir_failed'` 使用同一规则；“未确认”不等于“非 MLIR”。

分类在读取存量及新增结果时使用同一函数，不改写原始 status、诊断、hash 或身份。升级分类版本会使可丢弃的增量统计侧库重算全部历史 group；不能继续使用旧规则的汇总缓存。

## 采集与统计

- 执行器每 5 秒原子发布本地小型快照，不在调度线程访问网络。
- 独立同步进程每 10 秒尝试发送心跳，不等待结果 ACK；所有网络传输禁止重定向。
- 心跳每次最多 32 个会话、每个会话 64 张卡。连续运行保留累积计数，序列重传不重复累计、不刷新执行器观测时间。
- 机器在线由上传器心跳判断；卡是否占用由执行器的新鲜快照判断。超过 30 秒延迟、60 秒失联；未知不当成空闲。占用是执行槽，不是实测 AICore 利用率。
- 完成量是**后端 Shape 执行数**，含重试；CPU 参考不计，不等于去重用例覆盖数。生成失败无 Shape，单独统计 Case 错误。完成/成功每分钟 = 所选窗口内连续新鲜心跳的计数增量 / 固定窗口分钟数。
- 跨重启创建新会话；首次样本只建立基线。断网或观察间隔超过 30 秒不把累计补传量计入实时吞吐，因此网络缺口可能低估实时产出，历史结果仍保持完整。
- 卡身份来自实际 NPU UUID；机器 ID 优先为 DMI UUID 的 SHA256，再尝试 machine-id，最后用本地持久 UUID。容器不暴露稳定宿主身份、克隆 machine-id 时仍有识别局限，不声称可以无条件识别物理宿主。
- 实时速率保留一天，运行快照保留七天，机器/卡登记保留。历史结果和 ACK 不修改。

## 当前边界

历史分位数留在工具效率的折叠诊断中，按结果搜索条件独立统计；不把各组 P50/P95 直接平均，也不伪称它们已按实时设备筛选重算。Kernel 页只显示真实采集的计时，未采集显示空态。

尚未采集实测 AICore/温度/功耗，尚未提供跨机器同用例配对 Kernel 加速比；心跳只覆盖 staged 执行器。离线查看器明确显示实时状态不可用。在线机器/执行卡历史趋势、细分单卡产出、全局阶段分位数与同步积压告警还需独立的数据口径与实现。

## 此次发布与回滚

中心环境 `/home/ubuntu/af-fuzz/venv-fleet-runtime-1110`，override `zzzz-fleet-runtime-1110.conf`，私有一致性备份 `backups/fleet-runtime-1110/`。移出新增 override 并重启可回到前一版本；不要用旧备份覆盖新生产结果。

NPU 安装 `/mnt/workspace/af-fuzz/fleet-runtime-1110/installed`；私有队列/结果备份和原入口位于同目录的 `private-backup/`。升级时暂停 campaign feeder，等待已接纳任务自然结束再重启原配置，不删除结果和 ACK。旧上传器已替换为带心跳的上传器。

实时观测使用独立的 `results.sqlite.runtime.sqlite` WAL 数据库，避免历史结果上传事务长时间阻塞心跳；权威用户认证仍读取原中心库，结果库不复制 token/hash 到观测库。此库不是结果/ACK 的一部分，生产备份如需保留运行历史，应额外做 SQLite 一致性备份。

联调同时发现已有 E/F 采样生成的 `nnsmith-sampled/5` 元数据未被 CaseSpec 解码器接受，导致写库失败；解码器已按 seed 版本验证对应 dialect 和冻结采样配置，新增回归测试。升级期间写库失败的任务以原队列的 resume/reconcile 流程恢复，不删除旧结果。

### MLIR 返回值日志采集

`MLIR Codegen Result` 已由 GELOGI 输出。采集器不按日志等级过滤，并给该标记保留有界空间，避免后续错误日志挤掉返回值。但采集器只能读取 worker 的 stdout/stderr，不能恢复被运行库日志级别过滤或只写到 slog 文件的 INFO 日志。需要采集新标记时，在启动 NPU worker 前配置 `ASCEND_GLOBAL_LOG_LEVEL=1 ASCEND_SLOG_PRINT_TO_STDOUT=1`；既有持久 worker 需重启才能可靠应用。这里不自动修改全局日志等级，也不扫描其他进程的日志文件。未采集到标记时沿用旧逻辑，不把缺失当成成功。

### 2026-11-11 机器显示名

仅更新 oracle1 当前 `venv-35fe47c` 中的 `central/runtime.py` 和 `view/static/runtime.js`，未携带工作区其他改动。新增 runtime 侧库映射表，服务重启时补齐存量机器编号；客户端协议和原机器 ID 不变。

部署前私有一致性备份位于 `/home/ubuntu/af-fuzz/private/machine-names-1111-1794446740/`，包含结果库、runtime 库、原文件和 service 配置。回滚须先检查后续部署，再恢复这两个源码文件并重启服务；保留新增映射表，不覆盖生产数据库。

验证：运行状态/前端/刷新测试 13 passed，sync/view/cache 测试 45 passed。额外 hosting 测试 3 passed / 1 skipped / 6 failed，失败涉及本机 ledger 锁和缺少 nnsmith，不宣称全量通过。线上 API 和浏览器均显示 `XiangWF-npu0`、`NiuYH-npu0`，机器选项无重复，公开 runtime.js 哈希与本地一致；service active，runtime integrity_check=ok。原用户、来源 ID、receipts、owners、attempt ID 无丢失；来源 ACK 随持续上传更新。
