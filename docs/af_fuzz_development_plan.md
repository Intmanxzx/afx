# Fuzz 重构开发计划

日期：2026-11-09。总体方案已同意；本文件描述目标与验收，实际实施进度见第 6 节。

设计基线：`af_fuzz_refactor_plan.md`。原则：同seed无缓存重建；正确性/性能统一；轻量SQLite；公网近实时汇总；用户/密钥离线配置；同一pip包提供离线view；中心易部署、易迁移。

## 1. 交付方式

- 按里程碑、可评审的小变更实施，每阶段附测试结果、使用说明和已知限制。
- 按用户最新决定全新重构，不兼容旧 fuzz 入口/语义；数字 seed + replay 为唯一执行入口。历史模块仅供基线调查，非 fuzz 功能不受影响。
- 不先大规模移动文件，不改远端共享依赖，不默认重建Autofuse。远端验证使用隔离目录/环境。
- 已从 D0 开始编码及测试；各阶段完成情况、证据和遗留项见第 6 节。
- 不给未经基线验证的日期或吞吐提升承诺。以明确验收门槛排期，完成D0后评估关键技术工作的规模。

## 2. 开发里程碑

### D0：风险验证与行为基线

工作：

1. 盘点现有CLI、测试、结果消费者和依赖；记录本地/远端版本与可用设备。
2. 捕获seed广播轴漂移及input size问题的图属性、有序输入签名和首个分歧阶段。缺原故障证据时标记未定位，不把其他shape错误当作修复。
3. 建立黄金样本：静态、广播、规约、动态shape、不同dtype/布局；比较图/输入/权重字节，不仅比较seed。
4. 验证AscendC/Triton编译设备依赖、autotune和跨进程执行包，输出split/combined能力矩阵；补齐隔离双后端测试环境。
5. 记录生成、编译、执行、报告耗时和内存/磁盘基线；确定支持的Python、SQLite与OS矩阵。

产出：可复现脚本、基线记录、失败回归测试、能力矩阵。

验收：能解释已捕获的漂移来源；未决问题有证据边界。完全编译拆分未证明时不阻止轻量查看开发，但不宣称达到完整并行目标。

### D1：安装、配置与契约基础

工作：

- 引入轻量核心契约及版本：seed协议、CaseSpec、执行/评估配置、结果与schema。
- 消除包import时的依赖补丁和NPU环境副作用；重型依赖只在执行适配器加载。
- 统一`autofusedbg fuzz`；旧`af-fuzz`别名已按最新决定移除；配置解析与数据路径规范化。
- 实现setup/doctor框架：环境检测、持久化目录、资源预算、离线身份配置；中心未就绪时允许本地使用，在线校验在D5接入。
- 建立wheel发布、前端静态资源随包分发、三系统干净环境CI。查看模式不安装Torch/NNSmith/Node。

验收：安装和轻量CLI无NPU依赖；setup重复执行不更换来源ID、不丢配置；不自动替换用户Torch/CANN或修改共享NNSmith源码。

### D2：确定性生成与统一执行

工作：

- 实现v1稳定seed解码、候选目录、字段寻址随机决策、唯一shape/广播决策及规范化数值物化。
- 处理求解器多解、浮点refine搜索、动态候选过滤的环境依赖；不以保存文件替代确定性。
- 内存CaseBundle、有序输入绑定、布局/alias校验、workload指纹；能力不足返回unsupported。
- 统一静态/动态与正确性/性能的执行入口，后端只保留编译/执行差异。
- 统一错误、逐shape结果、诊断重跑；性能与正确性判定分离。

验收：无历史库/缓存，同seed跨新进程、受支持CPU架构、设备和线程配置重建一致；黄金样本旧协议不漂移；已定位input size回归通过；两个后端输入相同，性能模式与正确性模式共享同一执行实现。

范围控制：可先完成明确的v1因素子集，再逐步扩展；未支持因素显式拒绝，不降低一致性保证。v1通用协议验收前保持预览状态。

### D3：轻量结果库、快照与离线view

工作：

- SQLite版本迁移、增量事务、共享环境/配置去重、任务与attempt结果。
- 旧summary导入、幂等来源识别、未知字段兼容；不凭seed虚构原workload身份。
- `snapshot`一致性导出白名单结果，剔除同步/租约和秘密配置。
- `fuzz view <sqlite>`：回环只读服务、自动打开浏览器、分页/筛选/聚合/详情；与中心共享查询和前端。
- 默认不保存tensor、完整图、长日志、原始profiler；临时文件清理与编译缓存限额。
- UI支持复制完整seed、重放/诊断命令；不在查看过程中启动测试。

验收：Windows/macOS/Linux同一wheel离线查看；源数据库不改写；运行中WAL快照、损坏/旧schema有测试；十万级合成结果不全量送入浏览器；测量bytes/case和查询延迟。

首个可用交付：即使新生成器仍在验证，也可导入现有结果进行轻量跨批查询，不把旧数据标记为新协议可精确重放。

### D4：任务生命周期与资源感知流水线

工作：

- case/backend独立任务；CPU、内存、磁盘和设备预算；有界队列、背压。
- 所有run共享本机设备调度，性能采样独占设备；控制外层并发×编译线程总量。
- 支持真实split后端；不支持的明确记录combined阶段。
- `run/status/stop/resume`；默认后台批测，提供前台模式。SSH断开不终止已托管任务。
- 优雅停止、强制停止、超时进程组清理、崩溃恢复、设备故障隔离。
- 服务可自动恢复；测试中断后默认需显式resume，不悄悄占用设备。

验收：相同工作负载和设备下实测阶段重叠、吞吐及资源峰值；NPU计时无同工具任务竞争；kill/断连/磁盘不足无错误重复提交或残留任务；断点续跑不换case。

### D5：后台自动同步与中心网站

工作：

- 同步outbox与结果同事务提交、ACK/revision/epoch协议、重试与有界批次。
- 首次setup校验管理员离线预置的展示名/上传密钥；不开发在线注册/申请。
- 同步服务独立托管，无需逐run publish；CLI显示已同步、待传、待更新与受阻状态。
- 中心认证上传、单写导入、身份归属、批次幂等、旧revision防覆盖；服务端不接受任意源SQL。
- 共用D3前端：我的/团队结果、可分享筛选URL、同步延迟、环境分组与可比性说明。
- 浏览凭据与上传凭据分开；由管理员离线管理。

验收：断网批测不停，恢复补传；ACK丢失/重启/上传时新增结果/密钥轮换/中心旧备份恢复均不丢失已保留结果、不重复统计。健康网络下验证5–10秒页面可见目标；无正常运行SLA数据前不作保证。

### D6：简单部署、迁移演练与正式切换

工作：

- 版本固定Compose（应用+可选TLS代理），amd64/arm64镜像；无Docker时同pip包服务入口。
- 统一初始化、离线用户管理、健康检查、备份/恢复、升级说明；预构建前端，服务器无需NPU或Node构建环境。
- 从干净环境A部署，备份后迁到环境B；覆盖域名切换、旧服务停写、epoch重同步、凭据/身份保留。
- DB迁移前备份，升级需检查活动任务；明确回滚兼容边界。
- 全流程用户验收：setup一次 -> 后台run -> 中心近实时查看 -> seed重放 -> snapshot -> 另一OS离线view -> 升级/迁移继续同步。
- 更新README、使用指南、错误排查与旧入口弃用说明，最后删除被统一执行器替代的重复实现。

验收：管理员无需编辑源码或手工建表即可部署；迁移后历史结果、用户归属和同步正确；无Redis/PostgreSQL等额外依赖；全部关键测试通过后再切换默认执行入口。

## 3. 依赖与推进顺序

```text
D0 -> D1 -> D2 -------------------> D4
          \-> D3 -> D5 -----------------> D6
```

D3的schema/导入/展示可在D2验证期间推进；D5协议建立在D3的版本化数据契约之上。最终D6必须同时满足D2/D4/D5，不以UI完成代替seed或资源调度验收。

## 4. 评审与验证方式

每个可合并变更附：变更范围、迁移影响、单测命令/结果、远端验证证据、尚未满足的验收项。建议按以下边界切小提交：

1. 契约/轻量导入与CLI；
2. 单一生成因素确定性及黄金测试；
3. 后端统一执行；
4. 数据库/旧数据导入；
5. 查询API/view与快照；
6. 任务调度与恢复；
7. 同步协议和客户端；
8. 中心鉴权/查询；
9. 安装发布/部署与迁移。

不在此阶段创建PR、发布包、部署公网服务或修改现有远端依赖；实际外部交付按后续实施范围执行。

## 5. 待具备的验证资源

- 用户原广播轴漂移/input size故障的seed、配置与完整错误证据（若现有结果可找到则直接使用）。
- 稳定可用的双后端NPU环境，以及跨架构生成/三OS查看器测试环境。
- 中心服务器/域名/TLS与管理员凭据管理方式，供D5/D6联调；此前可用隔离测试服务完成协议测试，不阻塞本地开发。
- 计划支持的用户规模/结果量可先用合成负载压测，正式延迟和容量预算据实测确定。

下一步从D0开始：复查现有测试基线、增加seed分歧诊断和编译阶段探针，然后实施D1轻量导入与新CLI骨架。

## 6. 实施进度：2026-11-09（D0 + D1 首批）

状态：已开始实际实施；**D0/D1 均未整体验收完成**。证据与复现命令见
[`af_fuzz_d0_evidence.md`](af_fuzz_d0_evidence.md)。所有后续测试/安装在 `npussh mlir`
隔离根目录 `/mnt/workspace/af-fuzz/refactor-d0-Nh5vfp`；不修改共享 NNSmith/Torch/CANN。

### D0 已落地

- [x] CLI、旧结果/报告/checkpoint 消费者、依赖和远端设备/仓库基线盘点。
- [x] opt-in 生成阶段 trace、具体图属性/有序 tensor 签名与字节哈希、首个差异比较命令。
- [x] 捕获裸 NNSmith seed=42 的图 shape 漂移；独立 solver context 对照与回归。
- [x] 受限静态/动态 legacy trace、手工广播/规约/dtype/layout 诊断样本。
- [x] AscendC 首次调用惰性编译实测、设备 autotune 源码边界复查、combined 能力矩阵。
- [ ] 用户原始广播轴/input size 故障仍未定位；不要把此次裸 API 失败或动态合法性拒绝当作原问题已修复。
- [ ] 全生成器跨架构 golden、双后端可工作环境、端到端执行包异进程移交、完整吞吐/资源基线未完成。

### D1 已落地

- [x] 无 import-time 环境变量设置/自动依赖修改；显式执行初始化和只读依赖兼容检查。
- [x] stdlib 核心预览契约：版本化 seed token、CaseSpec/CaseBundle、执行/评估配置、轻量 shape result。
- [x] `autofusedbg fuzz setup/doctor/run/diagnose`，旧子命令及 console alias 转发同一 handler。
- [x] 平台数据路径、持久 source_id、原子 setup、并发锁、已有配置保留、CPU预算、离线管理员身份/密钥文件引用。
- [x] 远端 Linux aarch64 干净 venv 的无重依赖 wheel 安装及轻量命令验证。
- [ ] v1 物化尚未实现/注册；旧整数 seed 不升级伪装成自包含 seed。
- [ ] 资源完整检测/预算、在线配置校验（D5）、前端资源、三 OS CI 尚未实现；view 等保留命令显式报未实现。

### 测试证据

- 原始远端全量：**256 passed / 5 failed**（48.30s，`baseline.log`）。
- 最终远端全量：**279 passed / 同 5 failed**（27.91s，`final.log`）；新增 23 个通过项，无新增全量失败。
- 聚焦诊断/契约/seed context：**38 passed / 3 deselected**（17.95s，`focused-final.log`）。此聚焦命令排除了原裸候选顺序组三项；全量仍执行并保留 seed=42 失败，不隐藏基线问题。
- AscendC 广播 Add+ReLU combined 通过；Triton 隔离安装后因 CANN 头文件缺失枚举导致扩展编译失败。
- wheel 在未安装 Torch/NNSmith/numpy 的远端隔离 venv 中成功安装，doctor/setup/run-help/af-fuzz-help 可用；setup 重复运行 source_id 不变。
- 未运行 Windows/macOS 兼容矩阵、跨 CPU 架构、完整 workload 端到端 split/IPC 或严格性能测量。

下一批：继续缩小原 seed/input 故障证据、确定最小 v1 profile/黄金样本，补齐匹配的 Triton 构建；独立推进 D1 配置校验及 D3 轻量 SQLite/view，不先重写两套现有执行器。

## 7. 第二批实施：D0 重放调查 + D1 资源/配置

继续沿用远端隔离根目录，新增 `round2/` 源码副本，所有测试与 wheel 构建仍在远端。

### 已实现

- `doctor` 无重依赖读取 CPU affinity、cgroup v1/v2 挂载与实际成员路径，考虑可见父级 CPU quota/内存限制，并读磁盘剩余空间；未知项返回 null，不伪造无限资源。
- `setup` 增加 compile_threads / memory_budget_bytes / cache_budget_bytes；验证 UUID、配置结构与正整数预算，保留未知字段和来源 ID；UTF-8 JSON 损坏时拒绝改写。
- 管理员身份仍为离线配置：校验字段、密钥文件存在/非空/Unix 权限，仅读元数据、不读或打印密钥内容；明确展示名不参与鉴权。Windows/macOS 路径逻辑只做模拟单测，不冒充实际 OS 验证。
- 明确资源值是限制/诊断，CPU worker ceiling 不是吞吐建议；**旧 run 暂不强制执行 setup 预算**，D4 调度尚未接入。
- 重放探针能在前序 case 失败后继续检查目标 case，另存运行 metadata，不让前序历史/计时污染目标图的指纹。
- 新增有界矩阵 `tools/fuzz_replay_audit.py`，跨新进程改变 PYTHONHASHSEED/OMP/MKL 线程环境/前序调用；区分成功重放、分歧、相同生成错误、worker 错误，不将“同样失败”算作生成通过。
- 发现旧 IR 文本省略 Transpose 的交换轴，增加 `graph_extra_attrs` 诊断阶段和回归；因此新版 trace digest 与前一版不同，这是诊断覆盖变化，不是 workload 变化。

### 待完成边界

仍未完成：用户原始故障定位、跨架构/长期版本协议验收、v1 materializer、匹配的 Triton 双后端环境、编译跨进程移交，以及 D1 三 OS CI/前端打包。此批没有升级共享依赖、安装本地环境、修改旧执行调度算法或发布服务。

### 第二批最终验证

- 串行全量（矩阵审计结束后）：**299 passed / 原 5 failed / 31.53s**，日志 `round2-serial-final.log`。相对首批新增 20 个通过项；此前 1:9 分配断言失败在该串行全量中通过，仍保留偶发失败证据。
- 加入具体 Transpose 轴属性后，24 对 / 48 新进程重放矩阵全部一致；前序 8 次动态合法性拒绝不再阻断目标生成。
- 新版 wheel 在同一无执行依赖 venv 中安装验证，pip list 仅本包、pip、setuptools；setup/doctor 保持轻量。
- 本轮未在本地执行或安装，未新增本地临时产物；未提交/推送。完整证据见 D0 证据文档第 7 节。

## 8. 第三批：最小 v1 协议与统一执行纵向切片

新增协议说明 [`af_fuzz_seed_v1.md`](af_fuzz_seed_v1.md)。这批从基础设施推进到了可运行用例，但仍为 preview，D2 总体未验收。

### 已实现

- 两个不可变内置 profile：`preview-static-v1` / `preview-dynamic-v1`；自包含完整 seed 选择协议、因素、图和数据，不读设备/历史库/缓存。
- 固定 SHA256 字段寻址 rejection sampling、规范 little-endian float16/float32 存储、固定有界 dyadic 数值；不依赖库 RNG 或 solver。
- Add 广播 -> ReLU/Neg/Mul -> 可选 ReduceSum；固定广播轴/动态轴、contiguous 或转置布局；动态 case 三个具体 shape 与同一固定权重由 seed 直接决定。
- 完整输入/输出签名、不可变 bytes bundle、存储/view 边界检查、内容 workload ID，独立标量参考实现与单独 oracle_id。
- 一个静态/动态/正确性/可选测量共用循环，逐 shape 分离正确性和性能状态；输出树/数量/shape/dtype 校验；测量异常不擦除正确性。
- CPU eager/Inductor、AscendC/Triton 适配入口，后端不改变 case。编译创建明确为 prepare，首次调用保留 combined，不伪称 split。
- CLI `fuzz seed <token>`、`fuzz run --seed-token <token> --backend ...`；旧 `--seed` 仍为 legacy，不混淆身份。新协议功能只进入新共用流程，不继续扩展旧双执行器。
- 六个冻结 protocol golden（两 profile × 0/42/uint128_max），保存 spec+bytes 内容指纹及独立 oracle 指纹。改变生成规则须新增 profile，不能直接改旧向量。

### 验证及边界

- 新进程 hash seed/随机历史测试；标准库空环境生成和六个 golden 校验；128 seed 因素可达性检查。
- CPU eager 两 profile 各 32 seed 与独立参考精确一致；CPU Inductor dynamic seed42 三个 shape 正确性通过，并走同一可选 wall-time 测量阶段。
- AscendC NPU static seed42 及 dynamic seed42 三个 shape 正确性通过，使用旧 overlay 只读依赖+本轮隔离缓存。
- **NPU 性能模式明确 unsupported**，直到 D4 实现同卡 lease；CPU wall-time 不等于 kernel profiler 指标。
- 未验证：跨架构/跨版本长期稳定性、所有 NPU dtype/layout/因素、Triton 执行、compiled artifact IPC、可变状态/任意 alias、通用 IR、后台调度/轻量数据库。
- CLI 结果当前仅输出短 JSON，不保存 tensor/长日志；SQLite 持久化和 view 是接下来 D3 的工作。

第三批最终全量：**322 passed / 原 5 failed / 31.58s**（`round3-full-final.log`），相对第二批新增 23 个通过项；无新增全量失败。协议专项中间验证 22 passed，最后增加 backend/device mismatch 回归并纳入全量。最终 wheel 重建后轻量 seed 命令再次输出同一 workload ID（`round3-wheel-final.log`）。

## 9. 第四批：纯数字 seed + replay，全新语义

用户明确不需兼容旧设计/语义/功能。本批替换第三批预览接口，不再维持长 token 与两个 profile。

- `Seed(value)` 的数字区间解码：version=value//2^128+1；当前仅支持 version1。长 token、负数、前导零、未知版本明确拒绝。
- `replay 42` 默认执行完整三样本用例；`--static-only` 只执行基础样本并使用 static 编译，共用完整 bundle、权重和 workload ID。
- 移除旧 af-fuzz handler、参数树、console alias、legacy forwarding 以及无用 fuzz/runtime.py。非 fuzz 命令保留。
- 新协议黄金向量取代旧预览向量；数字种子 0/1/42/389609433/uint128_max 覆盖完整内容和 oracle 指纹。
- 旧 NNSmith 模块/测试暂留作为 D0 调查材料，新执行入口没有依赖它们。尚未完成历史文件整体清理；不把新旧执行器做互相转发。
- 更新 CLI/契约/测试/README 与协议文档，生成参数不可覆盖 seed 含义；所有资源和执行选项都不参与生成。

验证：远端 `round4/` 全量 **319 passed / 原 5 failed / 36.23s**；数量变化包含取消 static/dynamic profile 参数化重复项与删除旧初始化测试，不是跳过新增失败。新数字协议/static-only/单参数 replay/旧命令拒绝回归通过。

CPU Inductor `replay 42 --performance` 三个 shape 通过；AscendC 同 seed 全三 shape 与 static-only 均通过，并拥有同一个完整 workload ID。轻量 wheel 在无执行依赖环境通过数字 seed 生成和五个 golden；旧 console alias 不再分发。详见证据第 9 节。

仍未完成跨架构验收、NPU 性能独占租约、Triton 可用环境、后台批测/数据库/view，以及历史 NNSmith 目录清理；本批没有发布/提交或修改共享依赖。

## 10. 阶段提交与模块化（本轮）

用户要求每阶段测试后 commit 并继续下一阶段；本地仍只编辑/提交，执行测试仍在 npussh mlir。前述“不自行提交”由本次明确授权更新为阶段性提交、不推送。

首个 checkpoint：`c9039fa`，记录已验证数字 seed/replay、D0/D1/D2 预览及文档。

模块化：generation/v1 + registry、backends/base + lazy factory + Torch adapter、通用 tensors、独立 execution loop 和 `python -m autofusedebugger.fuzz`。协议字节不改变，numeric golden 不更新。架构边界与扩展规则见 `af_fuzz_module_architecture.md`。测试完成提交后进入 D3 SQLite 轻量结果/快照，不等待跨架构资源才开始独立的数据层工作。

模块化远端全量：**322 passed / 原 5 failed / 32.56s**，`round5-modules.log`；新增 3 个架构边界测试通过，五个数字 golden 不变。阶段完成后提交，再推进 D3。

## 11. 已进入 D3：轻量 SQLite 与一致性快照

模块化阶段已提交 `e20e44d` 后继续实现，而非停在计划。

- 独立 storage/schema、records、repository；无 Torch/NNSmith/生成器依赖。
- 默认 replay 保存轻量 attempt 与逐 shape 指标，支持 --database/--no-save；配置与环境白名单去重，不存图/tensor/raw profiler/长日志/环境全集。
- 事务写入、相同 attempt ID 幂等、冲突拒绝、新 replay 新 attempt；并发首次初始化原子发布完整 schema。
- results 分页/seed 过滤/详情，限制 1..200；snapshot read transaction 一致性导出白名单表，包含 WAL 已提交数据，排除未提交变化及额外 secrets 表，拒绝覆盖目标文件。
- 非本工具/损坏/未知 schema 数据库拒绝操作；schema1 尚无升级迁移，不宣称完成 D3 全验收。
- 新增存储测试：白名单、短错误、回滚、幂等、分页、WAL 快照、非覆盖、损坏拒绝、无重依赖查询和并发首次初始化。

远端独立 CLI 已实际完成 replay 42 -> results.sqlite -> snapshot -> 只读 results 查询。后续继续预构建静态 view 与共享查询界面，再做规模/bytes-case 验证；本轮不部署网站、不实现 SSH 查看助手。

D3 首批最终远端全量：**332 passed / 原 5 failed / 42.15s**（`round5-final.log`）；storage 专项 10 passed。wheel 在无 Torch/NNSmith/numpy 的 light-venv 中独立运行 results，读取 snapshot 成功（`round5-wheel.log` / `round5-light-query.json`）。本阶段测试通过后提交；下一阶段为离线 view 与分页/规模验收。原 5 项历史基线失败继续列示，不等于全库全绿。

## 12. D3 第二批：同包离线 view 与规模验证（2026-11-09）

- 已接通 `autofusedbg fuzz view ./snapshot.sqlite`，默认浏览器打开本机随机端口，支持 `--no-browser/--port`、Ctrl+C 退出；不部署 NPU 常驻服务。
- `storage/queries` 统一 CLI 与 HTTP 只读查询：seed/状态筛选、keyset 分页、汇总、逐 shape/共享上下文详情和单 seed 重放命令。最多 200 条/请求，页面每页 50 条；查询设 3 秒 SQLite progress budget，响应上限 2 MiB。
- `view/server` 与 `view/static` 分层；wheel 内含预构建资源，无 Node/模型运行时需求。固定 loopback、随机 capability 路径、Host/Origin 校验、CSP、无 CORS、无写路由。仅面向可信结果/快照，不宣称可安全打开任意恶意 SQLite。
- 真浏览器测试发现 `id=reset` 遮蔽原生 `form.reset()`，已修复并加浏览器回归；编辑但未提交的筛选不影响当前翻页。seed 全程作为字符串，错误/数据库文本只走 textContent。
- 十万 attempt + 三十万 shape 合成库：74,817,536 bytes（约 748 B/attempt）；首页/深页查询约 0.56/0.55 ms，状态汇总约 36.1 ms，分页响应约 12.2 KB。只是当前 ARM64 机器单次合成数据结果，不是生产吞吐承诺，也不是持久化 NPU 结果的 bytes/case 上界。
- 远端 light-venv 安装 wheel，从源码目录外启动 CLI；无 Torch/NNSmith/numpy，PATH 不含 Node，静态资产、results/summary API、SIGINT 正常退出均通过。
- 远端 Playwright 1.48 + Chromium 130 实测筛选、翻页、复制、详情、空结果/非法查询、重置、超大 seed 字符串、XSS 文本及 390px 无横向页面溢出。浏览器依赖只 apt download + dpkg-deb 解包到隔离目录，未 apt install、未改系统/共享依赖。新版 Playwright 不支持该 Ubuntu20.04 ARM64，故使用测试专用旧版，非产品依赖。

本阶段新增 25 项 view 测试 + 1 项架构边界测试；最终全量 **358 passed / 原 5 failed / 33.85s**，日志见证据第 11 节。历史 5 项失败继续保留，不以新增模块通过冒充整个仓库全绿。所有执行位于 npussh mlir；本地仅编辑/Git，临时检查截图已删除。

D3 仍待：Windows/macOS 实测、可信快照导入/恢复与 schema 升级迁移、异常聚类/指标配对比较。D4 任务持久化/恢复与设备独占 lease，D5 自动 outbox/revision/ACK + 网站，D6 部署迁移/备份演练均未完成。接下来先补 D3 恢复/升级边界与故障注入，再推进 D4；不可将本机 viewer 变成 NPU 常驻查看 API，也不把现有后端 prepare 当纯 CPU 编译完成。

## 13. D3 第三批：可信快照恢复/合并和升级边界

- 新增 `restore <snapshot> <new-db>`、`import <snapshot> <existing-db>`、`check-db <db>`。全程只依赖标准库，不导入执行层；不恢复 tensor/秘密/任务/同步状态。
- restore 流式写临时新库，检查完成后原子无覆盖发布；import 在一个事务内合并，原 attempt ID/时间/指纹不变，重复完全一致才跳过，同 ID 内容冲突整批回滚，不按 seed 去重。
- 逐记录重做 whitelist/指标/seed/UUID/时间/上下文及结果 hash 验证，不能只信任 result_hash；源完整性/引用检查、目标触发器拒绝。仅支持可信来源，不把自带 hash 当签名。
- schema1 目前没有更早的已发布 fuzz schema，也没有必需的 schema2 变更，因此不造空迁移、不把 user_version 改号当升级。check-db 显式报告当前 schema/无需升级，未知/旧号拒绝且不改文件。未来真实升级需逐版本转换与备份/恢复测试；尚不能宣称 schema 迁移已完成。
- 旧 summary 导入按用户最新“不兼容旧设计/功能”要求取消，不虚构其数字 seed/workload 身份。
- 故障注入涵盖写入后异常、发布竞争、子进程强制退出，验证现有库回滚/新目标不发布；突然断电/硬件故障不在这些测试覆盖内。SIGKILL 恢复临时文件可能残留，不会发布为最终目标；自动清理他人临时文件暂不实现。

第三批验收：远端 transfer 专项 **23 passed**；全量 **381 passed / 原 5 failed / 74.14s**，无新增失败；安装 wheel 后无模型依赖 restore 1 条、重复 import skipped=1、check-db integrity=ok 均通过。完成阶段提交后进入 D4 的操作状态账本；尚不开放未接执行/设备租约的 run 命令。

## 14. D4 第一批：持久化任务状态机（内部基础 API）

D3 恢复/导入阶段提交 `c69cd85` 后继续开发，而非只列下一阶段计划。

- 新增独立 `scheduling/ledger.py`：单 case/backend 的规范 seed + execution/evaluation 配置持久化，不保存 tensor、不执行 backend。
- 操作库有独立 application ID/schema，与长期结果库隔离；任务库不能误作结果库导出/查看。持久化全队列 active 上限，多个客户端不能通过新建对象改变容量；超额明确背压。
- enqueue 有 task ID 幂等/冲突检测，IMMEDIATE 事务单任务领取；claim token 与 attempt UUID 每次重新领取更新。
- pending 可取消；running 停止变为 stop_requested，不能直接 resume；worker 已确认终止/reap 后才可 interrupt，再由用户显式 resume；不按超时自动抢占、不因服务重启自动重跑。
- 旧 claim 无法完成新领取的任务，seed/执行配置不变。跨线程/多进程领取与子进程死亡恢复已纳入测试。

边界：内部 API 尚未接 supervisor/run/status/stop/resume CLI；不声称后台测试已可用。claim 是结果状态栅栏，不是 NPU 设备锁，也无法杀掉仍在运行的旧进程。complete 要求调用者先保存对应 attempt 结果，但本批不实现两库提交间的恢复协调。设备全局租约、进程组/超时清理、CPU/NPU 流水、资源完整预算、worker 心跳/恢复及后台进程托管是下一批，未通过之前不开放性能独占承诺。

D4 首批最终验证：任务账本专项 **10 passed / 0.97s**；远端全量 **391 passed / 原 5 failed / 35.75s**，无新增失败。安装 wheel 后在轻量 venv 走 enqueue/claim/interrupt/resume/complete 全流程，未加载 Torch/NNSmith/numpy。阶段测试完成提交，下一批是 supervisor 与统一执行器衔接、结果提交中断协调及设备全局租约，不能提前启用 NPU 性能模式。
