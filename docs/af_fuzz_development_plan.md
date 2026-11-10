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

## 15. D4 第二批：前台 CPU supervisor 与结果确认恢复

- `scheduling/worker` 仅调用既有 materialize → create_backend → execute → ResultRepository.save；没有第二套执行引擎。每个 worker 一个 case/backend，按 claim 的 attempt UUID 写入。
- `scheduling/supervisor` 前台串行启动独立进程/进程组，设有限 timeout，轮询 stop_requested；停止/超时/KeyboardInterrupt/启动失败执行清理。结果确实已持久化才确认 completed；失败的正确性结果同样是已完成 attempt，不误当需要重跑。
- 结果保存与任务确认的崩溃窗口：重启持有队列锁后按 attempt ID 检查规范记录 hash、seed/workload/oracle、执行/评估配置，匹配则确认而不重跑；无结果则 interrupted，必须显式 resume，resume 后新 attempt，不改变用例。
- 独立 POSIX flock 队列所有权，worker 继承锁 FD；父 supervisor 非正常退出但 worker 尚活着时，新 supervisor 拒绝接管。不能基于 PID/过期时间抢占。队列固定绑定结果库路径，避免错误目标库造成误判。
- 新 CLI：`run <seeds...> --foreground`（可重复 --backend cpu-eager/cpu-compile）、`status [task]`、`stop <task>`、`resume <task>`；支持 --data-dir/--queue，run/resume 可指定 --database。run 不带 seeds 时仅处理已排队 pending；resume 只改状态，不悄悄启动执行。run 必须明确 --foreground，尚未实现默认后台托管，不能静默降级。
- 任务状态分页最多200，CLI默认50；串行 worker，compile_threads <= CPU count，worker OMP/MKL/OpenBLAS threads 设为1。尚未接 setup 完整内存/磁盘预算；这不是已验证的最大吞吐并发方案。

边界：目前仅 POSIX CPU 托管执行，NPU 管理入口拒绝（直接 replay 既有功能不变，NPU performance 仍 unsupported）。进程组清理验证普通同组子进程，不保证清理自行 setsid/double-fork 的 helper；父进程与 worker 同时死亡而仍有不继承 FD 的 helper、断电、SSH 自动重启/托管尚未验收。不会把队列锁称为设备锁，不会承诺严格性能同卡独占。当前 run 为部分入队语义，达到容量/后续失败时保留已接受任务；通过 status 查看，取消或不带 seed 继续排队任务，不自动重交整批。

第二批全量最终 **407 passed / 原5 failed / 50.31s**，新增16项通过；真实 CLI 的 CPU eager/Inductor 两个 seed42 任务均ok且完整workload ID一致。安装wheel的无模型依赖 status 成功。首次全量旧并行分配测试另有时序断言失败，最终全量复验通过，过程保留在证据第14节，不隐去。下一批先完善失败原因/批次事务与后台监督，再实现同机设备身份/租约及NPU进程清理；这些未完成前保持NPU托管禁用和显式前台要求。

## 16. D4 第三批：原子批次、短中断原因与操作库升级

- enqueue_many 规范化整个批次后，在单个 IMMEDIATE 事务内检查幂等/冲突/容量，再写入；任何失败不留下半批任务。单批最多10000任务，active容量仍是持久化队列配置。
- run 使用原子批次入队，取代上一批部分入队语义。容量/晚到非法seed/ID冲突的失败不提交任何新任务。
- 操作库 schema2 新增短 reason，记录 timeout、user_stop、worker_exit、supervisor_interrupted、worker_lost_before_ack 等；不记录长stderr/日志，恢复重试清空旧reason。
- 明确命令 `upgrade-queue <queue> <new-backup>`：仅支持 task schema1→2，在队列POSIX锁和SQLite写事务内冻结写入、生成完整一致性备份，再ALTER+版本号事务提交。备份不覆盖、活跃队列拒绝、普通访问不隐式升级。长期结果schema仍为1，不受此操作迁移影响。
- 任务操作库备份是管理文件，可能包含运行状态，不能直接当结果快照公开分享。无新格式兼容别名；此迁移只为本次已提交的重构操作库升级。

第三批最终远端全量 **414 passed / 原5 failed / 52.19s**；阶段测试后提交，再进入后台CPU启动握手。

## 17. D4 第四批：有限后台 CPU 批测与启动握手

- run 默认 detached CPU supervisor，保留 --foreground。后台只处理当前队列直到无pending后退出，不安装常驻查看API/服务，不引入任务调度基础设施。
- `scheduling/hosting` 独立负责POSIX启动、所有权传递、有限socket握手及短状态文件；worker仍走原execution。父进程先持锁、验证结果路径绑定与现有任务，再原子入队；child准备好后需父进程start授权，drain校验/恢复完成再ACK running，父进程才返回。
- 启动失败/超时清理子supervisor，批次保留pending供显式处理，不虚构running；提交方在start授权前断开不会领取任务。ACK后的启动状态不是任务成功承诺；仍以task/result为准。
- 子supervisor start_new_session、stdin/out/err不关联终端、忽略SIGHUP；SIGTERM转入正常worker进程组清理路径。启动器退出不等于终止worker。有限状态文件包含run_id/state/pid/时间/短错误和结束摘要，不保存原始日志。
- status同时展示任务页及supervisor元数据/真实队列锁占用，PID仅展示不用来授权kill或判断存活。旧running元数据但锁已释放时显示exited_without_final_status，不误报后台存活。

限制：尚无机器重启后的自动服务恢复、系统级cgroup、NPU设备租约/跨用户协调或并行worker。操作库坏盘、ACK丢失后的不确定执行窗口、父/worker同时被杀后的脱离helper仍需人工核对；保留锁栅栏而非强抢。新run在队列忙时拒绝，不把已存在后台supervisor当成可永久追加任务的daemon。stop仍按task操作，不是整个队列shutdown；中断任务默认显式resume再run。

第四批验收：后台CPU Inductor seed42在提交SSH连接结束后由新连接确认completed/result ok且supervisor finished/unowned；安装wheel无模型依赖环境后台空队列成功启动并结束。`round11-full.log` **422 passed / 原5 failed / 112.09s**；升级事务回滚新增项独立通过，最终包含该项的全量见证据第16节。没有宣称机器重启自动恢复或NPU托管已完成。

第四批最终全量 **423 passed / 原5 failed / 46.68s**；阶段测试后提交。下一阶段仍是资源配置强制执行、同机NPU设备身份/独占租约与worker清理能力验证；自动服务重启/outbox同步未实现，不以后台CPU批次代替这些验收。

## 18. 剩余范围盘点与资源预算首批

按可独立验收的大工作包归组，开始本轮时还剩8包（不是8轮工期或完成率）：
1. 执行资源预算：CPU/编译线程、内存、临时磁盘/缓存。
2. NPU托管、设备身份/同卡租约、严格性能独占及异常清理。
3. 经能力验证的流水并行、完整后台恢复/故障隔离。
4. 跨架构确定性、Triton、原seed/input问题证据及因素验收。
5. 三OS安装查看、结果分析/配对比较与迁移边界收口。
6. 自动同步outbox/revision/ACK、断网续传及托管。
7. 中心网站、管理员离线身份接入/认证上传/团队查询。
8. 部署运维及全链路验收：Compose/同pip、备份恢复升级和跨环境迁移演练。

### 本轮：第1包首批（非硬隔离完成）

- `scheduling/budgets` 只依赖标准库和资源盘点；run读取setup，compile_threads默认配置值，CLI仅可在已配上限内下调，旧pending任务超出当前预算则拒绝，不改其规格。
- 按affinity/cgroup/host最小CPU容量准入，而不是仅os.cpu_count。当前仅1个worker，cpu_workers作为上限，不伪装并发已实现；OMP/MKL/OpenBLAS仍1线程。
- 内存配置启用Linux /proc进程组RSS采样，超额终止worker组并标记memory_budget_exceeded；非Linux显式拒绝该采样能力。RSS共享页可能重复计数，采样不是cgroup硬限制，不保证峰值不越界。
- 每attempt创建私有临时目录，TMPDIR/Inductor/Triton缓存均重定向其中；cache_budget_bytes用于此目录总用量采样（含元数据/块占用），超额终止并清理。不会删除用户共享缓存，代价是跨attempt不复用编译缓存。
- 开始前检查scratch与结果路径所在盘的空间，默认保留64MiB；运行中采样磁盘余量，低于reserve则终止。未控制任意后端私自写出目录的路径；真实磁盘填满、硬配额、cgroup管理不在本轮完成范围。
- 前后台相同策略，短白名单resources.json在父进程持锁时落盘，child启动后重读再检查；不把身份密钥/环境全集传入策略，不修改seed或CaseBundle。
- worker结束/超时/资源超限/采样异常后先进程组清理，再清理仅本任务的临时目录；parent SIGKILL仍可能留下临时目录，不能未经ownership核对自动删他人目录。

这是第1工作包的可用保护层，不把它标记为硬隔离/全局多队列总预算已完成；直接replay仍不走托管预算。NPU托管仍关闭。

资源首批验收：远端17项新增预算测试通过，全量440 passed/原5 failed；真实seed42 eager与Inductor在预算内执行均ok且无scratch残留。后续goal已获公网部署授权，保留这些约束，不把采样保护当硬隔离完成。

## 19. 公网可用性 goal（用户明确授权）

用户授权ssh oracle1、停止babynews并复用esay.app/80/443，要求默认30秒可配置自动同步、至少1000组真实Elementwise+Reduce+Transpose测试，Transpose单独类别。已创建active goal；不再受“不部署公网”旧限制，但仍不本地测试/安装，不修改共享依赖。

- babynews源码/服务/Caddy配置已备份于oracle1私有/var/backups/af-fuzz-cutover；新服务健康后停止disable旧service，保留数据。esay.app由原Caddy TLS转发127.0.0.1:8090的新服务。
- sync outbox与result save同事务写入，attempt不可变，revision递增，ACK和中心receipt持久化；HTTPS拒绝redirect转发凭据。默认30秒，每周期有界批次，失败不阻塞执行。admin-user离线生成唯一展示名和密钥，中心仅存hash，密钥私有文件传递不输出。
- v1冻结含义不变；新增数字区间v2明确Transpose图节点、独立scalar oracle及Elementwise/Reduce/Transpose分类，旧view类别移除Transpose。不是把转置输入layout冒充显式Transpose节点。
- oracle1(x86_64/Python3.12)与mlir(aarch64/Python3.11)已比对seed42/v2首例workload指纹一致；更大golden待验收。
- 初次NPU试跑因覆盖PYTHONPATH漏掉CANN Autofuse而error，恢复原PYTHONPATH后3个v2 seed×CPU/AscendC×3shape全部ok。错误probe不计入1000正式验收。
- NPU容器存在systemctl二进制但PID1不是systemd，因此不宣称systemd同步托管成功，改用独立带flock的POSIX daemon；自动机器重启恢复仍需容器启动器接入。

## 20. 中心上线后加固与设备锁边界

- 中心结果列表改为最新优先的keyset分页，离线列表保持原有升序；30秒刷新首页现在能看到新结果，而不是只有计数变化。
- 本地once/daemon通过独立上传锁串行化，避免手动诊断与daemon并发ACK竞争。
- 新增按torch_npu物理UUID映射的POSIX独占lease（逻辑index不作为身份）；同用户所有新版直接执行共享默认锁目录，可配置受保护共享目录用于站点跨用户协调。只约束遵守同协议的程序，不能宣称外部NPU任务不存在。
- execute finally释放lease；错误/prepare失败仍清理；跨进程竞争、不同UUID并行、崩溃释放测试通过。NPU1真实v2 seed42三shape ok。正在运行的旧1000验收进程没有新lease，不能据此开启NPU严格性能模式或全局调度承诺。
- 公网真实Chromium130筛选、复制重放命令/详情、390px移动布局通过（round14-public-browser.log），无JS错误。
- Docker Compose在oracle1备用8091验证全新库健康通过；私有backup恢复到容器验证进行中，不切换当前线上pip服务。

### 进一步可用性验证

最新托管入口允许AscendC/Triton正确性、显式--device npu:index，统一worker持物理UUID lease；NPU性能入口仍拒绝。round16-managed.json实际npu:1 seed v2+42 completed=1，结果ok且3shape；不是仅fake adapter通过。默认锁同OS用户，跨用户需管理员共享受保护锁目录，不控制外部非本工具进程。

中心Compose固定Python3.12.8镜像构建成功，image ff1966f94d92c73f5a30754cd88d857a3eb06016df70ee00fb81f50d19549c23；备用8091新库healthz通过，完整私有backup恢复后summary=37条ok，与备份相同；验证容器已down，线上pip服务不受影响。一次cp因data目录属10001被拒，改sudo复制并设置正确属主后通过，没有放宽整个目录权限。

setup新增--sync-endpoint/--sync-interval，使用管理员key-file完成同步配置并首次启动独立daemon；之后无需每批publish/start。容器机器重启启动挂钩仍需容器管理器接入，不擅改用户启动脚本。

## 21. 可用交付加固与验收进行中（2026-11-09）

当前公网入口为 https://esay.app，oracle1 的 babynews 已停止并禁用（原源码/配置保留备份），Caddy继续提供TLS，单应用+SQLite由systemd运行。上传器在NPU容器以独立POSIX daemon运行，默认30秒，可配置；不是systemd，容器重启后的自动拉起需entrypoint接入。

新增：物理UUID设备lease、托管NPU正确性任务、同seed比较API/离线view与公网UI、分享筛选URL、最新结果优先；管理员离线轮换key、认证identity校验、sync实时本地待传统计；恢复旧中心备份后显式reset-epoch全量幂等补传。普通run支持--count生成独立ETR v2种子，默认后台运行。

测试证据：round16全量469通过/5原有失败；round18聚焦42通过；round19全量发现额外WAL切换并发锁失败，新增只针对BUSY/LOCKED的有界重试后round20全量478通过/5原有失败（63.18s）。不把该新失败归咎于环境或隐藏为skip。

断网实测：本地结果157→173期间ACK维持157、状态retrying，网络恢复后自动补传；配置已恢复https://esay.app、30秒。1000例验收分为entropy [0,500)、[500,1000)，分别npu:0/1执行CPU-eager+AscendC，每seed三shape。旧进程停止前已保存的结果按seed/backend/workload且三shape完备条件跳过；完成量最终以SQLite核对，不能用日志行数或仅生成数替代。

### 未决边界（不是本次已完成功能）

- 真正CPU编译/NPU执行拆分、跨进程编译产物移交仍未证实；combined首次调用仍包含惰性编译。单queue内worker串行，不宣称流水性能收益。
- NPU严格性能/kernel-profiler仍显式拒绝；UUID租约只约束遵守协议程序，不保证外部任务不存在。
- 当前v2只覆盖Add、ReLU/Neg/Mul、Transpose、可选Sum的小型图协议，不覆盖类别内所有算子。旧NNSmith原始广播故障未因此宣称修复。
- Linux aarch64/x86_64有实际wheel/黄金样本证据；Windows/macOS本机验收未进行。当前版本package依赖仍有numpy，但view本身不导入，--no-deps干净venv可用。
- cgroup硬内存限制、全站跨用户预算、容器重启托管，以及未可工作的Triton后端仍是后续能力，不以“网站可用”替代验收。

### round21追加验收

- 480 passed/5原有失败；WAL并发初始化压力测试连续10轮通过。
- 两平台完整1000 seed生成fingerprint序列SHA256一致：`9147ea351acff3264469d3b2e5d65ae0cb9f8a6bc3e7bc8f6eac3dcb2e8cdb07`（aarch64 Python3.11.4、x86_64 Python3.12.3）。这是生成确定性证据，不能代替NPU执行计数。
- 固定1000种子的op分布：Add1000、Transpose1000、Neg369、Mul325、ReLU306；Sum656（axis0=312、axis1=344）；float32=509、float16=491，广播axis0=499、axis1=501。
- NPU0完成前500后，NPU1的独立冷缓存编译仍慢；在当前backend完成后SIGTERM优雅停NPU1（完成563个seed总范围），剩余通过resume迁回NPU0已热隔离缓存。缓存只用于执行加速，不参与case物化；最终审计重新逐seed物化检查指纹。
- Triton隔离overlay重新验证仍失败：CANN头文件缺`RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE`导致npu_utils.cpp编译错误。未改共享CANN/Torch，不把import triton成功视为后端可用。日志round21-triton.log。

## 22. 本轮可用交付验收完成（2026-11-09）

**完成的是本轮可用交付，不是D0–D6所有长期架构能力。** 未决边界见第21节，尤其真实编译拆分、NPU性能、Triton与原生三系统测试，不打勾冒充完成。

- 1000唯一seed（v2 entropy0..999），CPU-eager与AscendC各1000个成功case/backend，6000个成功shape校验；跨平台重新物化的1000指纹序列一致。
- 失败历史保留：2条AscendC首次调用的隔离cache权限错误，修正权限后同seed重放通过；验收数据库2002条attempt，不删除或覆盖失败。最终审计`acceptance-final-audit.json`：missing_passed_pairs=[]、passed=true。
- 公网汇总2004条attempt（2002条验收+正式托管流程的2条），6012 shape行；2002条ok、2条error。两来源ACK分别2002、2，自动同步追平。
- 正式端到端：一次setup+管理员身份校验→后台run→CPU/NPU独立任务→SQLite→30秒outbox→HTTPS中心→浏览器筛选/重放/对比，均实测。普通运行和验收都复用统一execute，无第二套执行引擎。
- 安装版runtime-venv在源码目录外npu:1 replay通过；初始system-site-packages未继承嵌套venv中的Torch，改仅在隔离venv添加只读shared-torch.pth后通过，无共享依赖修改。
- 当前交付说明：docs/af_fuzz_quickstart.md。公网https://esay.app；本地无安装/测试产物，阶段commit已创建、未push。

## 23. 短seed、后端名称与原HTML展示（2026-11-09）

- seed增加无损短文本编码：A/B为固定协议v1/v2，后续规范大写base36；例如v2 entropy42为B16。底层数值和所有既有golden/workload不改，历史数值输入可用。默认新随机熵小于36^9，显示最多10字符；任意历史128位熵不截断，可能长于10字符。
- 新公开后端名cpu、npu-eager、triton-npu、af-legacy、af-mlir。cpu为eagerCPU；npu-eager不调用torch.compile；af-legacy显式--mlir_codegen_config=none；af-mlir在缺乏运行时能力证明时fail closed，避免编译时关闭MLIR后静默legacy。旧ascendc结果未记录模式，只展示为历史未区分，不伪造改名。
- Web时间固定UTC+8，YYYY-MM-DD HH:mm:ss；浏览器本地时区不影响。结果详情增加逐shape表，原JSON保留折叠查看。
- 旧summary_html及性能图表实现移动为独立fuzz/reporting共享代码，旧入口仅薄转发，不维护两份renderer。/report接入旧总体/分类/dtype/shape图表、可排序记录表，并增加后端分组/筛选、短seed和详情链接。现有统计页面保留入口。报告上限5000条，超限要求过滤；分页API继续可用，不能悄悄截断汇总。
- 报告元数据按冻结seed计划重建；未存储kernel/profiler指标显示未采集，不推造性能。HTML使用精确script/style内容hash CSP，动态style属性仅样式允许，错误/后端文本进行HTML转义。

## 24. 网站整合修正（2026-11-09）

按用户纠正：不是把旧HTML作为外链页面，也不是仅更换标题。现在单一shell（侧栏总览/实验记录/后端）、单一style.css、同一字体配色卡片；从原renderer抽取图表算法为charts.js，报告数据通过/api/report提供，原/report URL也进入同一网站。没有iframe、target blank或独立报表皮肤。总览六个维度图表，实验记录分页筛选、逐shape详情与同seed比较，后端页说明五个执行路径。

seed显示规范补零到10字符，例如B000000016；默认生成熵保持36^9以内。短输入B16仍解析同一个用例。历史完整128bit熵不可无损塞进10字符，内部保留其长形式，不截断破坏确定性。现有1000验收数据均显示固定10位。

历史cpu-eager显示cpu，历史ascendc显示“AF · 模式未记录”，不能把缺失模式强行归到af-legacy或af-mlir。新任务仍是五个公开名字。所有时间北京时间。

round28-focused=42 passed；round29浏览器在本地远端服务和公网都通过navigation/charts/fixed10seed/backend_names/filter/replay/detail/mobile，无pageerror。round29全量483通过/6失败，其中新增1项仅旧首页标题断言；改为检查主导航后sync+seed报表16通过。此前round26最终全量484通过/5原有失败。未隐藏既有失败。

公网wheel SHA256 453cc58d599d18c962643c11e15b6103dc1934f275dc392c367b99873f6c5c96，venv-round29；切换前private/before-round29.sqlite备份。测试/安装均在远端；本地截图临时文件已清理。

## 25. 用例并排、来源/版本/阶段指标与网站布局（2026-11-09）

- reporting additive schema v1：case_facts / attempt_facts / case_dimensions / stage_metrics / report_devices。新保存、中心上传在同一事务建立索引；snapshot保留、restore重建、显式index-results幂等。核心结果hash不改。汇总走SQLite事实表，不再向首页传输5000条明细计算图表。
- 一个可比用例分组一行，后端横向，重试保留；来源/环境/批次/配置不同不误合并。历史未记录批次不能恢复真实批次，保留historical-unrecorded。列表明确标注静态/动态、正确性/性能区别。
- 新采集CANN、Torch、Torch NPU、实际选中Autofuse binary SHA256；仅匹配binary manifest才使用commit，不拿共享源码HEAD冒充。历史缺失不回填。阶段：generation/capability/prepare/input_binding/first_call_combined/correctness/warmup/measurement/execution_total；prepare不是编译完成，first call包含编译与执行。总时间不包含持久化/上传。
- round30远端fuzz 239通过；全量493通过/6失败（原5项+非fuzz spawn任务分配精确1:9断言失败，独立复跑仍失败，round29基线单跑通过，未声称已解决）。存储/同步重点55通过。真实CPU性能与NPU eager执行成功并上传3条。100k合成查询证据保留，但用户之后要求仅几十到百条验证，后续不再扩大数据。
- round31修复nowrap导致版本穿透相邻列：固定布局、状态/耗时分层、只显示当前页有结果后端、长ID不占默认视图。round32顶部导航，全宽工作区，删除指定同步/时区文案；版本拆成独立可选列，默认隐藏；所有列可显隐/上移下移，localStorage版本化持久化，至少保留一列。
- round32仅12个合成case/24个attempt验证，桌面1440/手机390，展开长binary与ID也无cell溢出；列选择/顺序刷新持久化。focused=49 passed。截图和日志都保留在mlir /mnt/workspace/af-fuzz/refactor-d0-Nh5vfp/round32-*；本地不安装、不生成测试库/截图。
- 用户明确授权清理旧数据：先备份oracle1 private/before-round32-cleanup.sqlite，再删除核对过的2004条缺CANN及Autofuse标识记录，保留3条完整新数据。保留users/sources/receipts/ACK；deleted_attempts保留ID/hash/owner墓碑，旧库恢复为新source再上传也不会复活同attempt。测试覆盖count guard、同source重传、新source重传、完整结果保留。旧源库仍作为历史存档存在，不重置自动同步进度。
- 公网round32 venv，wheel SHA256 ce0de3d59489fbc90d6f57f4ab36ce9ff0ef7ab70899976b4fca6e0845f04741，服务/Caddy active。
- 未完成边界：Kernel profiler数量/时间仍未采集；不能用wall-time替代。严格NPU性能仍需要能力接入；旧HTML所有交互筛选/排序尚未逐项验收，不能把新增性能表称为全部迁移完毕。Autofuse binary未附可信commit manifest时只显示binary指纹。旧库离线查看需先备份再index-results。
- round32公网Playwright：顶部导航、总计3条/2个分组、删除指定文案、默认隐藏CANN、勾选/移序/刷新保留、390px无页面横向溢出全部通过，无pageerror。直接urllib默认UA请求公网被403拒绝，浏览器正常；未绕过服务安全策略。

## 26. 公网域名切换（2026-11-09）

- 主站改为https://autofuse.online；DNS经Cloudflare代理。原Caddy仅匹配esay.app，新增站点并成功签发Let's Encrypt证书，源站与公网GET均200。
- esay.app/www.esay.app页面重定向到新站；旧/api/*与healthz保留直连响应，避免拒绝重定向的存量同步客户端中断。
- mlir已知acceptance-home、round21-e2e-home的sync.json原子切换到新域名，30秒周期不变，密钥/ACK不变。没有创建新测试数据。
- 远端Playwright：新站记录页2分组/3条结果、新summary 200、旧API不跳转200、旧页面跳新域名保留fragment均通过，无pageerror。Caddy备份/etc/caddy/Caddyfile.before-autofuse-online。
- 浏览器列偏好按origin隔离，旧esay.app localStorage不会自动迁移到新域名。

## 27. 精度/性能结果拆列与用户归属过滤（2026-11-09）

- 后端不再用一个“通过”概括所有阶段。每后端独立精度结果、性能测量、总耗时列；accuracy由shape正确性事实聚合，性能失败不抹除已通过的精度。性能仅标记已测量/未测量/测量失败/部分测量/未完成，没有性能达标阈值就不称性能通过。
- 性能列逐shape展示mean_wall_ns（同步wall-time/次），不拿generation+execution总耗时冒充性能。总耗时仍单列，历史部分计时明确标注。历史重试也分别展示精度/测量状态。
- 用户ID默认列、点击ID填入筛选；精确ID参数化SQL作用于分页、汇总、statistics。筛选进入URL可收藏/分享，不读token、不把公开筛选当鉴权。旧列偏好迁移展开后端列，保留版本选择并默认启用用户ID。
- 小批12case/24attempt验证：独立状态、ID点击/无结果/重置、列偏好、桌面和手机布局均通过；focused=50 passed。公网浏览器验证同样通过，目前仍3条真实记录，无新增大批测试。
- 公网round33，备份private/before-round33.sqlite；wheel SHA256 6ef5aae7042a5ffe0cef06154c8636777eb8da2b27e8e2de98af4e80edd7bc9b。测试/安装全部远端，本地无生成文件。

## 28. 精简记录与可信指标框架（2026-11-09）

- CPU仅默认提供总耗时；不展示CPU精度/kernel性能，执行引擎停止把CPU重复调用wall-time作为性能测量。总耗时为秒/1位小数，不附生成+执行文案。
- NPU精度无边框绿色Pass/红色Failed；未知值灰色Unkown（按用户指定拼写）。NPU性能仅接受显式kernel profiler指标，无值的列默认隐藏。旧wall-time保留于历史事实/详情，不转义为kernel性能。
- 统一execute新增可选compile_duration_ns()/measure_kernel()适配器钩子，telemetry/stage_metrics存compile/kernel_mean，查询和汇总接通；真实Torch适配器尚无可靠分离证据，因此不实现这些钩子，性能页留空。测试假适配器验证传输/索引，不代表NPU profiler已实现。
- 设备列删除；公开用户列取管理员发行的display_name，不再暴露UUID作为姓名。setup新增--user-name别名；姓名不参与鉴权，仍由已发token对应身份确定。目前线上既有身份名MLIR NPU保留，未擅自归给niuyuhu。
- 单搜索框：大小写不敏感子串匹配，report_search在写入/显式索引时保存轻量搜索文档，覆盖seed/用户名/版本/算子/状态/时间/指标；参数化SQL跨分页搜索，卡片与图表共用筛选。旧库需要index-results；公网3条已索引。
- 每页10/25/50/100/200条，浏览器保存；列设置浮动层不挤布局，外部点击/Escape关闭，全选/全不选/恢复默认，允许0列且刷新保留。默认隐藏全空列，手动显隐优先。删除顶部AUTOFUSE / FUZZ和底部wall-time文案。
- 图例按钮独立显隐pass/fail/gen_error，隐藏后重算可见数据数量比例；图例状态在30秒数据刷新间保留。无变化刷新不替换记录/图表DOM，不清空表格或重置页按钮；展开历史/列设置时推迟后台换表。
- 远端小批12case/24attempt浏览器验证：CPU/NPU列口径、秒格式、border=0、搜索、分页、列全不选持久化、弹层不移位、图例90.9%重算、刷新DOM身份/MutationObserver无变化、390px布局全部通过。fuzz回归244通过。第一次SSH在测试结束后断开，读取日志确认完成，未将断线当测试通过。
- 公网round34，备份private/before-round34.sqlite，wheel SHA256 315122c64b616254ba32e5019a7f6237d58619a86c9dd198c6ef3d655470932c。未跑大批NPU测试、未新增线上记录。
- 最终重点回归37通过；公网浏览器验证列/颜色/秒值/搜索/浮层/图例/手机布局通过，无pageerror。一次公网probe失败为SSH shell剥离JS字符串引号（Private field #rows语法错误），改为远端脚本文件重跑通过，非网站故障。截图只存远端round34-public-records.png。

## 29. CLI评审前的小修正（2026-11-09）

- setup姓名参数统一为--user，移除--user-name/--display-name选项；--user-id仍是现有内部身份参数，列入本次CLI评审，不擅自扩大改名范围。
- replay --performance帮助文案去掉已废止的CPU wall-time承诺，明确仅请求后端支持的精确指标。
- npussh mlir round35：foundation 18通过，实际--help核对23个顶级子命令及setup参数。本轮源码修改/提交，未重新打包部署CLI，不需要为参数改名重启公网网站。

## 30. CLI收敛与token初始化（2026-11-09）

- 顶层23命令收敛到setup/run/stop/status/replay/view + admin/maintenance。低频诊断、同步、数据维护归maintenance；用户创建/轮换/网站/私有备份归admin。轮换按展示名查内部UUID。
- setup仅公开--user/--token/--sync-interval，默认autofuse.online；在线校验token和管理员登记姓名，不接受不匹配姓名，UUID自动获取。凭据内部0600保存用于无人值守上传，不输出到配置摘要或结果。token argv存在shell历史/进程可见风险，文档明确说明。
- 自适应默认使用有效CPU/cgroup/亲和性/内存/磁盘，保守限额，不宣称最优；当前串行cpu_workers=1。replay/run未配置时采用同一启发式，公共帮助隐藏底层路径/编译线程。
- stop默认对整个队列请求停止/取消，不再强迫用户提供单任务ID。新增eager-npu公开名，旧记录npu-eager展示映射，不修改既有结果hash。
- 命令与实施边界见docs/af_fuzz_cli.md。默认无限run、强制CPU+af-mlir、二级精度判断、误差/核心日志、100条新采集尚未完成；本阶段未清除线上3条数据、未部署新版CLI（部署时服务命令需改admin serve）。
- 远端round36重新检查：CANN9.3.0、Torch2.10.0+cpu、torch_npu2.10.0.post7.dev20261015，共享autofuse HEAD c1b46272，无NPU进程。已安装AFMLIR能力仍未证明，保持fail closed。共享依赖不改。SSH曾断连/认证失败，后续重连可用。
- 第一轮5项CLI新测试通过；迁移旧测试后60项重点通过。全量中3项失败为隔离传输遗漏tools和测试旧backend展示名，已补齐tools并更新为eager-npu后重新验证。用户原有index.html搜索占位符改动保留且不纳入本阶段commit。
- round36最终完整fuzz回归：250 passed，1条原有pygraphviz缺失warning；172.23秒。测试全部npussh mlir，无本地安装/运行。

## 31. 九项反馈续修（2026-11-10，round38，进行中）

- 普通 run 强制 CPU + AF-MLIR，默认无限；--count 惰性逐 case 投放，stop 原子关闭 feeder 并取消/终止当前任务，避免预分配无限队列。默认 replay 执行两者，显式 --backend 用于单后端故障诊断。
- 精度标杆使用实际 eager CPU 输出。仅 AF-MLIR 与 CPU 不一致时，重新绑定输入并独立执行 eager-npu；一致则最终 Pass，同时保存 CPU 分歧、判定标准和误差指标。不是忽略 CPU 分歧，也不把编译失败转成 eager 成功。
- SQLite 保存有界标量诊断：max/mean absolute、max relative、mismatch/elements/nonfinite、secondary 判定；网页详情展示。worker stdout/stderr 在内存有界收集，优先核心错误/MLIR严格判定，脱敏后最多1600字符，无长日志归档。进程硬崩溃/超时的结果诊断仍需完善，不宣称已覆盖所有 native 崩溃。
- 已重新检查 mlir：c1b46272 工作树干净，设备空闲；默认 CANN codegen 无 MLIR bridge。复制已有 dev-install 到隔离 mlir-runtime（未改共享源码/依赖）。加载库路径及 RunMlirCodegen 动态符号校验，强制 --mlir_codegen_config=mlir。冻结图无 Cube（严格模式唯一允许 fallback 的范围外路径）。这不是通用后端能力API。
- 真实 seed B000000016 产生 RunMlirCodegen IR，Elementwise 接管成功，Transpose 报 `Schedule buffer-form bridge does not support node ... Transpose` / `gate=lowering shortfall=yes`；不以 legacy 冒充 AF-MLIR。首次探测还发现导入顺序的 static TLS 错误，改为先导入 Torch，再加载 Autofuse。
- 远端首轮 42 项定向回归通过；round38-full3 257 passed；12 case/24 attempt 浏览器回归通过（布局、筛选、分页、刷新不跳、列偏好、图例、移动端）。后续补 CPU oracle、全量清理保护与 replay 默认矩阵后需重新全量验证。
- 第9项尚未验收：100组真实 AF-MLIR 批次和公网部署/清理/自动30秒同步将使用最终验证版本重跑。性能指标不以首次调用或墙钟时间冒充编译/kernel时间。

### round40 回归与部署证据

- 最终核心代码 `round40-tests.log`：260 passed（251.45s），唯一warning为隔离NNSmith未装可选pygraphviz。全部在mlir运行；light-venv缺wheel，已仅在隔离venv补齐，未改共享Python。
- `round40-ui.log`：12 case/24 attempt，额外验证核心错误、最大绝对误差渲染；浏览器执行仍在mlir，截图保留远端。
- round40 wheel SHA256 `a305b60d15a74f990bf3ea677b70dda2626f4e700f22a1b29abad406b3f5b493`；公网独立venv升级，服务命令切换admin serve。原3条记录先完整私有备份，再按expected_count=3清理，保留身份、ACK及防复活tombstone。
- 100组最终批次在round40-home进行，默认CPU与AF-MLIR、3个shape，真实严格MLIR并非全失败：早期已有2组AF-MLIR成功。失败核心包含Transpose lowering和exec_condition cache guard；不修改seed规避失败。
- 真实二级判定故障注入（不入库）额外暴露多卡问题：npu:1输入并不保证Inductor使用当前卡，产生npu:0输出与npu:1 eager结果比较异常。补显式set_device并在close恢复，需在隔离round41复验；不能把单卡0通过扩展为多卡通过。

### 二级标准与异常进程补测（round41/42）

- 修复set_device后，同一个真实AF-MLIR成功seed在npu:1再次编译运行，对CPU判定注入不一致（只用于测试，不归档）触发真实eager-npu复核：3个shape全部secondary_pass=true，最终ok；保留cpu_pass=false及basis=eager-npu。实际误差为0，不冒充自然精度失败。
- 公网浏览器已验证同case CPU/AF-MLIR并排、10字符seed、用户名、秒单位，以及详情实际`AF_MLIR_STRICT gate=lowering`核心日志。
- 补父进程有界日志管道：worker超时/异常退出时持久化error attempt并保留任务interrupted、允许显式重试；用户stop不伪造功能失败。无长日志文件，日志脱敏后最多1600字符。
- round42/round43各261项fuzz回归通过；追加CPU真实oracle/二级不一致仍Failed测试9项通过。round41定向39项通过。round43浏览器12组/24条记录通过。父进程异常日志测试含os._exit(7)、脱敏和SQLite持久化。
- 扩展回归round42：503 passed / 5 failed（排除已单测的架构/hosting重复项）。对上一commit 0dd0fa4在独立baseline目录重跑5个失败：4个复现（CANN GE fork、旧NNSmith seed42、两项prefusion Linear.graph），1个单跑通过；当前版本也将单独复查该项，不能未经验证把全部失败归为历史。
- 修正CLI收敛后部署模板遗漏：Dockerfile、systemd、backup.sh和操作文档统一admin/maintenance；公网定时备份也升级命令并实际触发验证。旧运维脚本不能长期依赖旧venv语义。

## 32. 九项反馈验收（2026-11-10）

1–3：普通setup只有--user/--token/--sync-interval，默认https://autofuse.online。4：预算自适应保守计算，串行worker=1，不宣称全局最优。5：公开eager-npu。6：CPU+AF-MLIR必选、实际CPU oracle、按需eager-npu复核。7：默认无限/stop，有限count惰性投放。8：核心日志及精度误差入SQLite、同步和Web详情；worker异常/超时有界日志。9：公网已清理旧结果并完成100组真实严格MLIR采集。

最终统计：100唯一seed，200条case/backend，600条shape；CPU 100成功；AF-MLIR 9成功、91功能/编译错误（不是91精度不一致）。shape pass327/error273。91错误均有严格MLIR核心日志。全部100 seed重新物化指纹一致、短文本10字符，批次200任务全部completed（完成任务不等于用例通过），所有结果自动上传。

- 原始批次归档审计：round43-final-audit.json，ACK200/pending0。后续最终浏览器发现1条case分成两组，原因是该CPU/MLIR运行间构建wheel生成package metadata，Debugger版本从null变为0.1.2。未伪造修改历史环境：使用安装版重新跑该seed的CPU/MLIR对，上传后备份并精确删除原2条，保留tombstone；中心最终仍100case/200attempt，个人本地保留202条（含原2条历史），ACK应202。
- 最终公网Playwright：100行，每行CPU/AF-MLIR并排，姓名MLIR NPU、CANN9.3.0/Autofuse c1b46272、10字符seed、核心日志、稳定刷新全部通过，JS错误为空。网站https://autofuse.online，线上venv-round43，wheel SHA256 `6a275b716760c1864bb15af8ea299ef7588db80898496441f54e8d8eeb6fe89f`。
- NPU隔离安装runtime-round43；`bin/autofusedbg-npu`更新指向该版本、隔离MLIR库和round40-home，旧包装有备份。安装版源码目录外CPU replay通过。新测试默认无限；此次验收批次有限100，结束后未擅自继续烧卡。
- 私有完整备份、定时backup服务实际触发Result=success、恢复副本integrity=ok；轻量一致性快照check-db通过。没有本地安装、测试、DB或截图产物；保留用户index.html搜索占位文字修改未提交。
- 非fuzz扩展回归的第5个失败在当前版本单跑也通过（0.11s）；4个基线可复现失败+1个全套时才失败，未在本次跨范围改旧调试器/NNSmith。

仍未声称完成的长期能力：准确独立compile/kernel指标、CPU/NPU流水拆分、多worker吞吐、Triton当前CANN兼容、任意图MLIR能力API、父进程被强杀时完整日志恢复。自然精度失败此次100组未出现；二级判定真实NPU故障注入和网页误差fixture均通过，不能把注入记录充当真实精度失败数据。

## 33. 网站用例口径与日志可发现性修复（2026-11-10，round44/45）

- 更正上轮验收遗漏：200是后端attempt数，不是测试case数。HTTP summary及质量/覆盖图统一使用与记录列表相同的case分组；CPU仅对照不重复计数，AF-MLIR决定当前case结论，重试取该后端最新结果，历史仍可查看。不同批次/来源不盲目按seed合并；CPU-only未完成为unknown，不当Pass。
- 原日志仅藏在点击状态/耗时后的详情下方，可发现性不足。新增默认可见的失败“日志”列，明确“查看日志/查看误差”，点击进入可分享/刷新保留的attempt详情；核心日志提前到详情顶部，缺diagnostics时回退已有错误摘要。
- 新增功能列：功能错误为功能Failed，精度显示Unkown；只有真实不一致才精度Failed。本批正确口径100case，9通过、91功能失败、0精度失败。后端对照与底层200明细不删除或改写。
- 新增100case/200attempt聚合、重试、CPU-only、筛选保留MLIR结论测试；浏览器使用真实点击“查看日志”（不再只用JS直接调用details）验证入口及顶部日志。
- round45远端全fuzz测试：265 passed，1个可选pygraphviz warning；12case/24attempt本地服务浏览器回归通过。
- 已备份并部署oracle1 venv-round45，wheel SHA256 `60257a48b0fe0fe1e1608c9304542d8dce1753f0dae9a1bc5deebc0e74dbb301`，未清理/改写200条明细。
- 公网真实点击验收round45-public.log：总数100、通过9、功能失败91、精度失败0；100行，91个“查看日志”按钮；功能Failed对应精度Unkown；点击日志后顶部显示AF_MLIR_STRICT，刷新详情仍保留；图表100用例、不计CPU，JS errors=[]。截图仅远端round45-records.png/round45-log.png。

## 34. 紧凑记录与独立 OP 列（round46）

- Seed仅保留10字符标识，原下方算子列表拆为独立默认“图结构 / OP”列，支持列显隐/排序/浏览器偏好；旧列顺序自动在Seed后插入新列，不重置其他偏好。
- 单元格上下padding收紧至10px，垂直居中；OP单行省略、悬停完整文本，日期和Seed不折行。不是把OP列表伪造为图拓扑。
- 所有验证在mlir：38项API/查询测试通过，12case浏览器回归通过；公网实际行高41px，旧偏好迁移、日志入口和移动端页面宽度验证通过，JS errors=[]。
- 网站部署venv-round46（先完整备份），wheel SHA256 a1a07750ec9d44045aa9eb36fb2c640d9160d572b457b279ae80a27b32cf8659；无数据变更。截图仅远端round46-public.png。

## 35. Shape级用例与结构化详情（round47–51）

- 用户明确修改计数语义：一个实际执行shape为一条用例。case分页、summary、质量/覆盖图共享shape级SQL投影；seed保持完整图的自包含标识，列表显示Seed / S0、S1、S2，不更改生成协议。CPU对照仍不重复计数；执行前整体失败无shape时保留S—记录，不伪造执行过的shape。
- 列表增加真实graph节点连线、全部输入（含权重）的shape/dtype、静态/动态；派生索引重新生成graph/weights/逐shape元数据，原attempt和同步hash不修改。维度/元素统计取当前shape，不复用整个图最大值。
- 日志列提取native核心错误，默认100字符，可在页面设置40–500并缓存，超出用...；详情查询接受shape，严格筛选shape结果/阶段/诊断。旧共享native日志明确标注整图共享，不声称能补回旧截断部分。
- 详情以当前shape结果、图/张量信息、结构化版本表、阶段耗时和共享开销组成，移除可见JSON原始记录。shape行耗时仅汇总当前shape计时，kernel/compile嵌套指标不重复相加；不把整图耗时复制给每条用例。
- 功能/精度/耗时/日志列按结果后端动态生成，新增fixture验证triton-npu自动出现；不代表本机Triton已可执行。
- 新采集器按完整日志行拼接pipe分块并保留错误后的上下文，修复error行被任意读块切碎问题；仍保留轻量日志限额。
- round50全fuzz 269 passed；新shape浏览器验证通过（3条shape、输入和权重、图、截断偏好、单shape详情刷新、triton列、移动端）。公网首次200行分页发现响应超2MiB，已移除列表重复携带的整图diagnostics/telemetry，完整内容只在详情请求。
- round51定向42 passed。公网最终Playwright：300shape用例、100seed、27Pass/273功能失败；200+100分页无重复/遗漏，输入权重/shape/dtype/动态图/真实连线可见，日志60字符截断偏好、单shape详情刷新通过，JS errors=[]。线上venv-round51，wheel SHA256 ae2bd65636fdd573827b8b0813a7dd3dfb81a813b88775b11eec4bc17e3e5417。
- 原始200后端attempt/600shape行不删除、不重跑，派生index在私有完整备份后重建。截图与构建文件仅远端，原用户index.html修改保留未提交。

## Round52：查询口径、Token-only 注册及紧凑展示（2026-11-10）

- 先核对线上只读计时：200 attempts/100 seeds 对应300 shape cases；27 Pass shape 的执行阶段和为0.000697181–15.702961311秒。示例 BD0MYWOXCJ 整图34.570727549秒，S0=15.553260811秒、S1=14.170879074秒、S2=0.000732161秒；首次调用分别15.365682565/14.170107352/0.000254964秒。上一版shape投影已不复用整图总计；本轮明确总耗时提示及详情总计包含关系，不把首次调用当kernel、也不伪造独立编译耗时。为何旧执行更快尚无同环境A/B证据。
- 查询统一按当前shape判定；`pass`不会因为CPU参考成功而选中AF-MLIR失败行。列表/总览/图表/性能筛选使用同一case投影；保留同shape CPU参考和多后端历史。模糊metadata搜索不再包含其他shape的输入规格。
- 新增安全SQL WHERE子集：白名单字段/比较/LIKE/AND/OR/括号、绑定参数、长度/嵌套限制、SQLite查询超时；不执行用户原始SQL。支持短seed精确查找。页面列出字段、值域和示例。
- 云端 `fuzz admin user DATABASE NAME` 创建唯一用户名并仅输出一次token，数据库只存hash；`admin rotate-token DATABASE NAME`同样不要求文件。客户端 `fuzz setup --token TOKEN` 通过identity API获取云侧姓名，不接受客户端用户名。后台周期仍默认30秒可配置。token内部0600保存，不能进入结果或普通日志。
- 删除无数据价值的静态后端菜单，保留总览后端分布及自动后端列。输入shape/dtype合并为x1[2,3,fp32], x2[2,1,fp32]；OP按图顺序紧凑展示；统一Pass绿、精度Failed黄、功能Failed红、Unknown灰。
- 基线round51定向11 passed。首次完整回归发现打包传输的macOS AppleDouble伪Python文件、旧setup断言、把所有seed错误写成同workload的旧fixture；已清除远端元数据文件，禁用tar xattrs，修正fixture真实shape状态与workload身份，未弱化产品语义。
- 浏览器隔离fixture：shape分页/输入格式/OP/动态triton列/日志截断/shape详情刷新/Pass与SQL筛选/移动端通过，JS错误为空。真实HTTP token-only身份解析及生成/轮换hash验证已补回归。最终全量测试与公网部署证据见下方验收记录。
- 最终验收：npussh mlir round52 全fuzz **275 passed**（188.81s，仅已有NNSmith可选pygraphviz告警）；最终隔离浏览器probe通过。公网浏览器：300shape/27Pass/273功能Failed、pass/Failed/SQL过滤与图表一致、短seed查询、15.6/14.2/0.0秒逐shape展示、紧凑输入与OP、日志详情通过，JS errors=[]。截图/日志只保存在远端round52-*。
- 部署：oracle1 `venv-round52`；wheel SHA256 `21e9f9b486d0bb5f84767f35c80c224556e79ed07304b94ec5df10c08bd5f203`；部署前一致性备份 `private/before-round52.sqlite`。重建200条派生索引以保留OP顺序，原始attempt/shape数据与用户token没有清理、没有重跑。管理员生成/轮换token及客户端只传token用隔离HTTP测试验证，未在生产新增测试账户。
- 未完成边界：没有旧版本/同环境冷暖缓存性能A/B，无法把首次调用15秒归因到某个编译子阶段；kernel/独立编译指标未伪造。SQL是WHERE安全子集，不开放任意数据库SQL。用户原有搜索placeholder修改保留于工作树，不纳入本轮commit。

## Round53：顶部统计及表格交互（2026-11-10）

- 四项按测试记录/功能失败/精度失败/精度通过排序，移除全部卡片底部提示；颜色按稳定ID绑定，避免位置变化导致红黄绿错位。
- 错误摘要长度label与输入框横排；change或Enter均立即重新渲染当前列表，不整页重载、不重置筛选/分页；输入限制40–500，超界钳制并写回输入框与localStorage。Enter上限、持久化加入远端浏览器验收。
- 所有表头居中，阶段耗时/性能表补用语义th而非普通td。
- 验证：npussh mlir 32项定向测试通过，隔离浏览器及公网浏览器全部通过（四卡文案/顺序/红黄绿、无提示、横排、Enter即时应用、500钳制、偏好持久化、记录和阶段表头居中，JS错误为空）。线上venv-round53；wheel SHA256 `43d8298b0ed349c648efe7147e8bc45dfdc8a27a1ed6e2e63b517704c3b0c2e4`，部署前备份private/before-round53.sqlite，无结果数据变更。未在本地测试或生成临时文件，原placeholder修改保留。

## Round54：图例与悬浮提示一致性（2026-11-10）

- 总览六张统计图的图例仅保留状态名称和色块，不重复数量/比例；保持点击显隐及重算比例能力。
- 统一STATUS_LABELS映射用于环图及柱图tooltip，内部gen_error不再泄漏到悬浮文案；当前该键实际聚合error/unsupported功能错误，不仅是图生成失败。
- Unknown计数为0时隐藏图例，有未知判定时仍显示；未知不能伪造成功能Failed。隔离fixture覆盖非零Unknown恢复展示。
- 验收：npussh mlir 定向9 passed；隔离浏览器通过（首次测试误发mouseenter，修为实际pointerenter事件后通过）。公网六图逐一验证纯名称色块图例、零Unknown隐藏、tooltip无gen_error、点击显隐正常，JS errors=[]。部署venv-round54，wheel SHA256 fcd2f6bf6d4299fcfad7e4dc61070b035894a58127e6c82bf2c976eac7a29c1a；备份private/before-round54.sqlite。无结果数据变更，本地无新增测试或构建产物。

## Round55：表头与内容对齐（2026-11-10）

- 根据用户截图，撤销表头居中，统一左对齐以匹配正文；表头/正文列边界和左内边距一致。未变更列宽、数据口径或查询。
- npussh mlir隔离浏览器通过；公网浏览器逐列检查边界差<1px、padding一致、全部表头左对齐、JS错误为空。纯样式改动未重复跑Python全量；所有浏览器测试和wheel构建在远端。部署venv-round55，wheel SHA256 26dd982f63a116cfc77bd7a560a2cd282451a10aad9c2ba9da811f4ca0359de9；保留上一服务配置用于回退，不修改结果库。本地无新增临时产物。

## Round56：状态名称与默认图例（2026-11-10）

- 顶部测试记录/功能Failed/精度Failed/Pass；图例同名，六图初始隐藏功能Failed，可点击恢复，不过滤顶部数字或测试记录。
- 测试记录及详情按功能/精度/性能错误分别显示功能Failed/精度Failed/性能Failed，正常Pass，未知按用户要求拼写Unkown；不把缺失性能数据伪造失败。
- npussh mlir定向9 passed；隔离浏览器覆盖三种失败显示、非零未知、图例开关；公网验证六图默认隐藏/恢复、卡片和记录名称通过，JS errors=[]。公网测试首次错误假设50条全失败，改为核对实际失败单元格文案后通过，未改变数据。
- 部署venv-round56，wheel SHA256 579cea99d57d6b205496398721e983d6752f30cb3efab7b99dcae684c7bb910b；无数据库变更，本地无测试/构建临时文件，原placeholder修改仍保留。

## Round57：功能精度与性能分析入口（2026-11-10）

- 菜单、页面标题/区块标题同步改为功能&精度、性能分析；保留records/performance路由，旧链接继续有效。
- 功能&精度默认隐藏kernel/总耗时列，保留自定义列可选及既有用户显式偏好；聚焦功能精度日志。性能分析仍只汇总compile/kernel_mean真实指标，未采集时明确空状态，不把first_call_combined或总耗时包装为编译/kernel性能。
- npussh mlir精确性能2项测试、隔离浏览器与公网浏览器通过（导航/标题/默认列/空状态/刷新，无JS错误）。部署venv-round57，wheel SHA256 a415393d9de97090f41727ef83d0dcbd6654e802ffa32451f52705b7374c6e7e；无结果库变更。本地未安装执行测试、未创建临时产物，原placeholder修改保留。
