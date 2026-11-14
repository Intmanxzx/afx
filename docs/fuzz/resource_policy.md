# 自适应资源策略 v3（编译并发上限 8）

## 从试验提炼出的规则，而不是某台机器的固定配置

1. 以 host CPU、affinity、各级 cgroup quota 的最小值作为有效 CPU，36 核配额不能按
   40 个可见 CPU 分配。保持现有准入：workers × compile_threads 不超过整数 CPU 上限。
2. worker 是共享池，不按每张卡固定分配两个。N 张卡初始目标为
   `N + max(2, ceil(N/2))`，为 CPU reference/生成/比较/保存保留共享名额；CPU 或 RAM
   不足时减少 worker，允许部分设备串行，不强行让全部设备同时运行。
3. 不为了用满 CPU 创建无收益的 worker。双卡对照里 4×9 和 6×6 吞吐近似，增加
   worker 没有可测收益；3 个 worker 曾导致设备 runtime 被替换。
4. 编译并发起点上限为 8，而非强制最小 8。`compile_threads=min(8, cores//workers,
   memory//(workers*2GiB)-1)`，至少 1。每槽基线加每编译槽约 2GiB 只是保守 RSS 启发式，
   不是硬隔离或任意图的内存保证。低内存先削减编译 fan-out；原有实际 RSS/缓存/设备
   内存与异常回收继续生效。
5. 八路上限是吞吐优先的起始策略：双卡 4×9 比 4×6 的小样本平均快约 2.93%，
   据此采用 8 作为更积极的默认值，**不是所有设备/后端已证明的最优值**。
   用户可显式覆盖；尚未实现按不同模型/后端在线试跑寻找全局最优。
6. 继续复用通过验证的 worker，按 backend/device/compile_threads 隔离；默认最多
   16 个任务，失败和资源增长回收。新图仍须编译，不能把重复 seed 命中算成新图提速。

### 内存、磁盘与未知环境

自动模式 RAM 预算取配置上限与检测 RAM 的一半中的较小值；缓存取配置上限、20GiB、
剩余磁盘 20% 中较小值。换到小容器后不会照搬旧机器的大预算。缺少可靠 CPU/RAM 信息时
保守降级，不伪造容量；运行准入仍可能因信息不足而拒绝。小于一核的 quota 最低只能启动
一个单线程进程，受 cgroup 节流，不保证无超额。内层编译器也可能有独立并行机制。

## 自动生效、兼容性与可解释性

- 新 `setup` 持久保存 `resource_policy=auto`，但不在 setup 时导入 Torch/占用 NPU。
- 新 home 的 run、auto 配置的 run：**每次启动**根据当前可见/选定设备数与环境重新计算。
- 旧配置没有 policy：保留手动语义，不静默覆盖用户历史预算。
- `run --auto-resources` 显式重算；`--fixed-resources` 使用保存预算；两者互斥。
- 显式 `--workers`/`--compile-threads` 优先，仍受预算准入；与显式 auto 同用时报错。
- auto 不回写历史 task，不修改正在运行任务的编译契约，也不自动暂停其他任务。
- 启动输出 `resource_plan` 包含 policy 版本、检测资源、预算修正、worker 目标、最终选择、
  原因和限制；保存的任务契约仍记录实际 workers/compile_threads。
- replay 不改为在线调参；重放语义保持原样。这是启动时启发式，不是持续在线自动调优。

## 代表性环境（内存充分）

| 有效 CPU / NPU | 自动 worker × 编译并发 |
| --- | --- |
| 24 核 / 1 卡 | 3 × 8 |
| 36 或 40 核 / 2 卡 | 4 × 8 |
| 16 核 / 2 卡 | 4 × 4 |
| 64 核 / 4 卡 | 6 × 8 |
| 64 核 / 8 卡 | 12 × 5 |
| 3 核 / 2 卡 | 3 × 1 |

低内存会进一步降低这些值。不同后端、图规模与设备的最优值可能不同，表格是起点而非承诺。

实验数据见 docs/benchmarks/ 下 MLIR reuse、compile-j10、adaptive-reuse、workers6 记录；
本轮验证位于 mlir `/mnt/workspace/af-fuzz/resource-policy-v2/`。旧实验文档中的默认配置是
当时状态，最新行为以本文及 owning source 为准。

## v2 历史验证与部署（以下不是 v3 当前状态）

4×6 两次 59.595 / 58.496 秒；6×6 两次 59.708 / 61.698 秒，四轮共 32 session、
48 shape 全部正确。两者均保住各卡 warm worker；结果支持少开无可测收益的进程，但
不支持把约 1–3 秒差异推广为通用性能优势。数据在
`docs/benchmarks/resource-policy-v2-20261112.json`。

本地资源/CLI/阶段相关测试 93 passed、8 skipped、1 deselected（固定 NNSmith 环境
依赖在 Linux 跑）；跨环境表格为模拟资源边界验证，不是这些硬件的实机性能证明。
覆盖自动/固定/旧配置/显式覆盖/重新 setup 不覆盖手动资源/换机预算收缩等行为。
部署只涉及 resources、config、onboarding 和 scheduling/cli 四个文件，不捎带其他任务修改。

最终 Linux 定向回归 128 passed、1 warning。oracle1 部署前 SQLite backup + integrity_check
（结果库、runtime 库）均 ok，服务 active、首页可访问；仅替换上述四文件。
mlir 已把 resource_policy 改为 auto，保留旧 6×6 数值作为 fixed 模式的回退值；
本次自动计算并运行的预算为 4×6，新 run id
`ed1fc163-2e67-4e2f-a5bf-6c64b6027fc3`，原随机采样策略/双卡/身份不变。
备份分别在服务器 `/home/ubuntu/af-fuzz/private/resource-policy-v2/` 和客户端 probe 目录。
回滚代码后还需恢复旧 resource_policy 或用保存配置启动；不回拷旧结果库。

## 实现补齐与重新验证（2026-11-12）

本次在已有未提交实现上补齐边界行为，不替换或重启正在运行的 NPU campaign：

- RAM 探测失败时，即使配置保存了旧机器的大内存预算，也只选择 `1 × 1`；
  `resource_plan.memory_budget_bytes` 保留 `null`，不把估算值伪装成检测容量。
  保存预算不被偷偷删除，后续准入仍会拒绝无法核实的内存上限。
- `resource_plan.budget_corrections` 列出配置预算到实际预算的修正；亚核 quota
  明确说明单进程仍受 cgroup 节流，降低编译并发时说明 CPU/RAM 限制。
- 连续 `run --fg` 在进入 launch 前将 `{"resource_plan": ...}` 输出到 stderr
  并立即 flush；stdout 仍保留原来的最终 JSON 契约。后台启动仍在返回 JSON 中带计划。
- 补充未知 RAM、亚核 quota、预算修正、setup 不导入 Torch/探测设备、前台启动输出、
  参数互斥测试；更新旧 hosting 测试以接受新 home 自动输出计划。

验证：本地资源/CLI/阶段测试 **103 passed、8 skipped**；mlir 隔离目录中资源、CLI、
阶段、预算、hosting、stage baseline 六组回归 **145 passed**，退出码 0。
Linux 测试隔离目录为 `/mnt/workspace/af-fuzz/resource-policy-implementation/`，
最终日志为 `regression.log`。本轮不是重新做 4×6 / 6×6 吞吐对照。
实际 mlir 资源仍为 host/affinity 40 核、cgroup quota 36 核，两卡规划结果 **4×6**。
原 campaign 保持运行，未修改它的配置、任务契约或编译并发；新代码仅在隔离测试副本验证。

oracle1 已重新核对有效 ExecStart（`venv-69a6245`，无 systemd override），
备份结果库和 runtime 库且 integrity_check 均为 ok 后，仅部署 resources、config、
onboarding、scheduling/cli 四文件并重启服务。私有备份位于
`/home/ubuntu/af-fuzz/private/resource-policy-implementation/`。
回滚只恢复其中 `code-before.tgz` 的代码并重启服务，不覆盖持续写入的生产数据库。
部署后四文件 SHA256 与本地一致，`af-fuzz.service` 为 active；线上首页含 app.js，
首页与 `/api/summary` 均返回 HTTP 200，汇总响应为有效 JSON。

### 4×6 / 4×9 直接复测补充

后续直接交错对照（每组 3 次）显示，本机固定 E/C16/D16/E 小样本中，4×9 平均
56.983 秒，4×6 平均 58.700 秒，4×9 耗时低 2.93%。因此上限 6 应理解为保守起点，
不能表述为“9 已无收益”或通用饱和点。本次没有据此更改默认值；详见
[直接对照记录](compile_6_vs_9.md)。

## v3：采用上限 8 并切换 mlir（2026-11-12）

用户选择更积极的启动预算，自动编译上限由 6 改为 8，`resource_plan.version=3`。
双卡 36 核/内存充分时为 4×8（预算乘积 32），不是固定预留四个操作系统 CPU，
内层线程池仍可能使用其他 CPU。CPU/RAM 准入、低资源降级和 worker 回收规则不变。
保存配置没有被改写，旧 fixed 模式预算继续保留；自动 run 使用新的上限。

mlir 先停止 feeder 补充任务、自然排空旧任务，再部署单个 resources.py，做隔离双卡
4×8 E/C16/D16/E 验证：**8 session、12 shape 全通过，墙钟 58.525 秒**。
这仅是一轮正确性/可运行性检查，不与之前不同时间的均值混算，不证明 8 的最优性。
随后以原采样/设备/evaluation 策略启动新 campaign，实际预算为 **4×8**，run id
`6030bd99-af7e-433f-a51f-54c89d014206`；已核实 running、队列被 supervisor 持有、
任务模板 compile_threads=8、保存配置逐字节不变。私有代码/配置备份、验证与日志位于
`/mnt/workspace/af-fuzz/resource-policy-v3/`。

oracle1 检查到最新有效环境为 `venv-kpi-filter-1112`（保留 kpi-filter-1112 override），
结果库/runtime 库 SQLite backup 与 integrity_check 均 ok 后，仅覆盖 resources.py，
没有覆盖 KPI/UI 代码。重启后服务 active，首页和 `/api/summary` HTTP 200，内容有效；
mlir、oracle1 和本地 resources.py SHA256 一致。
私有备份在 `/home/ubuntu/af-fuzz/private/resource-policy-v3/`。
回滚只恢复旧代码；mlir 需自然排空后按旧预算启动，不恢复旧数据库覆盖新结果。
回归：本地 **108 passed、8 skipped**；使用 mlir 实际安装路径的资源/CLI/阶段测试
**116 passed**（首次缺测试辅助模块，补齐测试搜索路径后完整重跑通过）。
