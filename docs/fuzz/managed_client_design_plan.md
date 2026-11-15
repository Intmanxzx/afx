# 托管客户端、环境与任务控制：设计方案和开发计划

- 日期：2026-11-12（America/Phoenix）
- 状态：设计基线与实现计划；本分支实现了 P1–P3 的本地/服务端控制骨架、环境适配器、安全操作日志和 Docker 验证夹具；真实 CANN/NPU 适配仍需单独验收。
- 范围：AutofuseX 客户端、中心服务、Web 控制台、CLI、依赖制品与发布流程。
- 目标平台：首期完整执行与自托管能力针对 Linux NPU 开发机/容器；其他平台提供能力检测和明确的“不支持”提示，不承诺相同的 CANN 安装与 NPU 能力。

## 1. 目标与核心决策

用户仅需安装 AutofuseX wheel 并启动轻量守护进程，即可绑定设备。此后通过网页准备运行环境、修改参数、控制任务和更新工具；所有主要操作同时提供 CLI，供人工和 Agent 使用。

核心决策：

1. **管理与执行分离**：守护进程不依赖 Torch、torch_npu、NNSmith 或 CANN 才能启动。
2. **安装与任务分离**：环境安装、源码构建、任务执行都是有状态的操作，不是配置中的布尔开关。
3. **YAML 输入，JSON 协议**：用户配置使用受约束的 YAML；统一 schema 校验后，协议与有效配置采用规范化 JSON。
4. **三类配置**：客户端策略、环境规格、任务规格分离；秘密和实际运行状态不混入配置。
5. **期望与实际分离**：服务端发布不等于客户端生效，必须有修订号、应用状态与回执。
6. **CANN 三种来源**：现有环境、下载 RUN 包、本地源码构建 RUN 包；默认安装根目录为客户端运行用户的 `~/Ascend`。
7. **兼容组合而非单包最新版**：Python、架构、CANN、Torch、torch_npu、NNSmith 通过版本组合管理。
8. **安全边界切换**：配置按类别生效；工具与环境更新通过排空、切换和重启完成，不做 Python 进程内模块热替换。
9. **源码现场保护**：允许未提交代码参与开发验证，不自动 reset、clean、checkout 或覆盖用户文件。
10. **身份、结果和任务连续性**：升级不能丢失身份、结果、未上传数据或任务恢复信息。

### 1.1 非目标与边界

- 不提供通用远程 shell 或任意脚本执行平台。
- 不自动提权，不默认安装或升级 NPU 驱动、固件，不默认重启宿主机。
- 不承诺任意 CANN RUN 包、任意版本组合都能无权限安装；按包类型和已验证适配器支持。
- 不把 RSS/磁盘采样描述为硬资源隔离；硬隔离需要系统 cgroup 等独立能力。
- 不保证网络断开时网页指令实时生效；本地 CLI 必须仍能安全停止与诊断。
- 首期不自动跨设备迁移运行中的任务，不建设通用分布式构建集群。

## 2. 当前基础与差距

以下为设计时的源码检查结果，不代表线上部署状态。

| 当前模块 | 已有基础 | 本方案需要补充 |
| --- | --- | --- |
| `fuzz/config.py` | 本地配置、身份和资源设置 | 分层 schema、修订、受控覆盖和迁移 |
| `fuzz/onboarding.py` | token 验证、私密落盘、setup、启动同步 | 一次性配对、设备凭据、无执行依赖启动 |
| `fuzz/sync/client.py` | 上传守护进程、锁、重试 | 通用控制收件箱、操作回执；不能直接等同于管理服务 |
| `fuzz/runtime.py` | 约 10 秒心跳、运行快照 | 能力、实际配置修订、环境和工具版本状态 |
| `fuzz/central/control_api.py`、`sessions.py` | Web 会话、来源检查、复测控制 | 设备权限、配置/环境/操作管理 |
| `fuzz/central/retests.py`、`sync/retests.py` | 复测租约、幂等与结果关联 | 通用控制语义；保持原复测兼容 |
| `fuzz/scheduling/` | 队列、campaign、staged 调度、资源预算、后台执行 | 管理服务适配、控制动作、配置安全应用 |
| `scheduling/staged.py` | 启动时读取预算并创建 Pool | 不能靠修改配置文件热调整运行中的 Pool |
| 独立 AutofuseX NNSmith fork | 固定 commit、源码指纹和 Z3 版本 | 由 pip 安装，客户端不再现场打补丁 |
| `pyproject.toml` | wheel、CLI；基础依赖含 NumPy | 轻量引导依赖和运行依赖拆分、入口惰性导入审计 |

特别注意：当前 campaign 的 stop 会请求中断/取消任务，不能直接作为新设计的“安全排空停止”；现有后台 supervisor 也不是永久进程管理服务，需要新增语义而不是重命名接口。

## 3. 用户流程

### 3.1 新用户

1. 安装 wheel。
2. 执行 `afx daemon start`，在未绑定模式启动管理服务。
3. 打开绑定地址，在已登录网页确认设备；或通过 CLI 输入注册 token。
4. 服务端显示设备系统、Python、CPU/NPU、驱动、空间及已有环境。
5. 用户选择环境方案，填写 CANN 地址或选择现有环境/本地源码构建。
6. 预检展示安装计划、来源、路径、空间需求和风险，用户确认后执行。
7. 安装与验证完成后显示“就绪”。
8. 在网页选择任务模板、修改参数、启动任务，查看结果并安全停止。

所有命令和 API 名称在本文中均为拟议接口，落地前不应当作可用功能使用。

### 3.2 本地开发者

修改客户端机器上的源码 → 选择已授权构建方案 → 增量构建 RUN 包 → 安装开发环境 → 验证 → 执行指定 seed/回归/持续 fuzz → 对比不同 build_id 的结果。

允许分别执行“仅构建、仅安装、仅验证、构建并测试”，不强制完整流水线。

### 3.3 Agent 与批量部署

CLI 支持无交互 JSON 输出、标准输入/秘密文件注册、预检、提交长操作、查询/等待、取消、安全停止。非交互安装必须明确选择已授权方案；不能绕过本地权限和制品信任策略。

## 4. 总体架构

```text
Web 表单 / YAML 编辑器 / 远程 CLI
                │ HTTPS + 身份授权
                ▼
中心服务
  设备注册 ─ 配置修订 ─ 环境/任务模板 ─ 操作队列 ─ 审计
                ▲
                │ 客户端主动发起 HTTPS（轮询，后续可长轮询）
                ▼
轻量管理守护进程 ← 本地 CLI（受权限保护的 Unix socket）
  配对/凭据 ─ 控制收件箱 ─ 本地状态库 ─ 状态/日志回传
       ├─ 下载与制品校验
       ├─ 构建/安装/验证进程
       ├─ 调度器适配层 → 工作进程池 → CPU/NPU 任务
       └─ 稳定启动器/更新器 → 工具版本安全切换
                │
                ▼
独立运行环境：CANN + Python 环境 + Torch/torch_npu/NNSmith
```

### 4.1 进程职责

- **稳定启动器/更新器**：启动管理服务、确认新版本健康、执行受控回退；不依赖测试环境。
- **管理守护进程**：拥有控制状态机、安装与任务操作，不导入重型执行库。
- **安装/构建子进程**：耗时和易失败工作隔离，进度持久化；失败不导致设备离线。
- **调度器/工作进程**：在固定环境中执行任务，复用现有队列与所有权锁。
- **上传职责**：保留 outbox 的独立可靠性，与管理控制共享必要身份，但不让安装阻塞结果上传。

首期本地控制使用权限为 0600、目录为 0700 的 Unix socket，不暴露局域网 HTTP 端口。自启动优先支持 systemd user；容器或无 user systemd 的环境支持前台模式交给外部 supervisor，并明确区分“后台运行”和“开机自启动”。

### 4.2 本地目录

沿用当前 `data_home()` 选择的数据根目录，**不悄悄迁移已有用户数据**。建议新增：

```text
<data_home>/
  manager.sqlite             # 操作、配置应用、恢复信息
  credentials/               # 0700；凭据文件 0600
  config/                    # 有效配置与最后有效快照
  environments/<env-id>/     # 托管 Python 环境和 manifest
  artifacts/                 # 下载/构建制品、哈希和来源元数据
  builds/<build-id>/          # 构建记录、可选源码快照
  tools/<release-id>/         # 托管工具版本
  logs/                      # 有界、脱敏、可轮转日志
  results/                   # 保留现有结果存储
  runtime/                   # 保留/扩展现有运行快照
~/Ascend/                    # CANN 默认安装根目录；适配器选择子布局
```

CANN 子目录布局由包的安装能力决定，不假设所有包都支持版本并存或任意 install-path；不覆盖外部管理的既有安装。

## 5. 配置模型与 YAML 规范

### 5.1 格式选择

- 用户/Agent 文件与导入导出：YAML 1.2 风格的受限子集。
- HTTP 协议、数据库快照、有效配置：JSON。
- 结构约束：版本化 JSON Schema（建议 Draft 2020-12）及共享业务校验器。
- Web 表单来自 schema 和显式 UI 元数据；兼容性、资源检查等不能仅依赖前端。
- 采用轻量解析/校验依赖；守护进程不为读取配置而安装 NumPy/Torch。具体库在 P0 验证 YAML 语义后选定。

拒绝重复键、未知字段、自定义标签、复杂别名/merge、隐式命令执行和不受控环境变量插值。限制大小与深度；版本号必须是字符串；时间如 `120s`，容量如 `16GiB`，解析后转换为整数单位。普通导出不保证保留原注释排版；可保留原始 YAML 作编辑草稿，语义以规范化快照为准。

配置指纹基于补全默认值、解析单位、固定环境引用后的规范化 JSON；拒绝 NaN/Infinity。schema_version、配置 revision、工具版本、环境 ID 是不同概念。

### 5.2 客户端策略

```yaml
schema_version: 1
kind: client_policy
resources:
  allowed_devices: ["npu:0", "npu:1"]
  max_workers: 4
  max_compile_threads: 4
  memory_budget: "16GiB"
sync:
  interval: "30s"
updates:
  tool_policy: manual       # manual / when_idle
```

本地信任根、数据目录、服务端地址、密钥、允许源码目录、安装授权和设备最大权限属于受保护设置，普通远程配置不能覆盖。远程策略只能在本地授权范围内进一步限制或申请资源。

### 5.3 环境规格

```yaml
schema_version: 1
kind: environment
name: "npu-development"
python: "3.11"
compatibility_profile: "validated-profile-id"
cann:
  source: download
  install_root: "~/Ascend"
  packages:
    - kind: toolkit
      url: "https://example.com/cann-toolkit.run"
      sha256: "<可信来源提供的完整 SHA-256>"
    - kind: kernels
      url: "https://example.com/cann-kernels.run"
      sha256: "<可信来源提供的完整 SHA-256>"
packages:
  torch: "<验证过的精确版本>"
  torch_npu: "<匹配的精确版本>"
  nnsmith: "<项目维护的制品版本>"
```

这是结构示意，带占位符的示例不能直接通过生产验证。包列表随实际芯片和 CANN 发行方式变化，不要求每种环境都有同样的包。

其他 CANN 来源使用互斥分支：

```yaml
# 使用现有环境
cann:
  source: existing
  path: "/home/developer/Ascend/ascend-toolkit/<version>"
```

```yaml
# 本地源码构建
cann:
  source: build
  install_root: "~/Ascend"
  build:
    source_dir: "/home/developer/code/cann"
    profile: "local-dev"      # 本机已授权的构建方案
    mode: incremental         # incremental / snapshot
    jobs: 8
```

`~` 与路径解析均在目标客户端执行；只对明确的路径字段展开 `~`，不执行 shell 插值。对 URL 带敏感参数的情况使用 `url_secret_ref` 或受控秘密字段，导出只保留引用，不暴露原始 URL。

### 5.4 任务规格

```yaml
schema_version: 1
kind: fuzz_task
name: "AF-MLIR 快速回归"
environment: "npu-development"  # 启动时解析为不可变 env-id
workload:
  mode: generate
  generation:
    tensor_scale: small
    max_nodes: 16
    max_elements_per_tensor: 1048576
    timeout: "30s"
    exclude_ops: [transpose]
execution:
  backends: [cpu, af-mlir]
  devices: ["npu:0", "npu:1"]
  scheduling: staged
  static_only: false
  timeout: "120s"
  resources:
    policy: manual
    workers: 4
    compile_threads: 4
    worker_max_tasks: 16
    memory_budget: "16GiB"
evaluation:
  performance:
    enabled: false
    warmup_steps: 3
    measured_steps: 10
termination:
  mode: finite
  max_cases: 1000
results:
  save: true
  upload: true
```

重放模式：`workload: {mode: replay, seeds: ["C000000016"]}`。持续模式：`termination: {mode: continuous}`，禁止同时设置 max_cases。有限生成中的 max_cases 计数定义为生成用例数，不是后端执行次数；重放默认执行完整 seed 列表，不重复套用生成数量限制。

字段映射必须显式：workers → Budget.cpu_workers；max_elements_per_tensor → 当前生成配置对应字段；timeout 需要根据执行方式定义实际超时范围。P0 固化 schema 前逐一核对现有 CLI，无法映射的字段不得宣称已支持。

约束：upload=true 要求 save=true 且身份已绑定；资源不能突破设备策略；auto 模式不接受与其冲突的手动 workers/threads；不支持的 backend/设备/采样组合报错。精确复现仍依赖实际 seed、工具/环境版本和执行快照，不能只依赖模板名称。

### 5.5 优先级与修改语义

任务值优先级：内置默认 → 模板快照 → 任务 YAML → 本次显式 CLI 参数。设备策略是约束，不是可被最后一层覆盖的普通默认值。

- 对象按 schema 定义合并，列表整体替换；null 只用于明确允许为空的字段，禁止模糊删除语义。
- 记录每个覆盖字段的来源；网页可查看最终生效配置和本地覆盖。
- 显式超限申请拒绝，不静默裁剪；auto 规划展示决策理由。
- 旧 CLI 的 argparse 默认值不能伪装成用户显式覆盖。
- 服务端编辑采用 base_revision/If-Match 乐观锁，冲突返回 409，避免覆盖其他编辑。
- 回滚配置是发布新的 revision 指向旧内容，而不是倒退版本号。

### 5.6 生效分类

| 类别 | 参数示例 | 应用时机 |
| --- | --- | --- |
| 即时 | 日志级别、同步间隔、接单开关 | 校验后立即更新相关循环 |
| 新用例 | 生成策略、过滤、评估和超时 | 持续任务的新一批用例；记录新 epoch |
| worker 边界 | 并发、编译线程、worker 回收策略 | 排空必要 worker 后重建 |
| 环境/进程边界 | CANN、Torch、设备集合、工具版本 | 排空任务、验证并重启 |
| 本地受保护 | 密钥、信任来源、允许源码目录 | 本地授权流程，不作普通远程热更 |

模板变更不自动改写既有任务。持续任务需要显式“应用到后续批次”，生成新 epoch；已入队与已执行用例保持原快照。应用状态为 pending/received/validating/waiting_safe_point/applied/rejected；快照必须与 actual revision 一致发布。

## 6. 绑定、权限与通信协议

### 6.1 设备绑定

- 未绑定客户端仅能配对，不接受安装或执行指令。
- 配对包含短时用户码、设备端高熵秘密、过期时间、尝试上限和单次消费。
- 登录用户确认目标设备；客户端用自己的秘密领取设备专属凭据，用户码本身不能领取凭据。
- 支持本地 token 注册兼容路径，优先 stdin/秘密文件，避免命令行参数泄漏。
- 设备凭据限制到 user_id/device_id 和能力，支持轮换、撤销；不能以用户上传 token 自动授予所有远程管理权限。
- 普通用户只控制自己的设备；设备所有者授权依赖安装/构建；发布者管理工具制品；管理员权限独立审计。

### 6.2 下发与回执

首期复用心跳节奏通知新修订，配置与操作正文通过独立有界接口获取，不塞入现有有限 ACK。采用客户端主动拉取，不开放入站端口。

建议操作信封包含：operation_id、device_id、kind、schema_version、payload_hash、request_key、generation/fence、expires_at、expected_revision、payload。

- 服务端同一作用域 request_key 相同且载荷相同返回原操作；载荷不同返回冲突。
- 客户端先持久化收件箱，再 ACK 接收，再执行；进度/结果用 outbox 重试回传。
- 使用租约、续租和代次拒绝过期执行者的状态写入；本地资源锁保护实际副作用。
- 不能声称外部安装器可 exactly-once。崩溃后通过步骤检查和产物/进程证据判断 completed/resumable/needs_attention，不能仅因租约到期重复安装。
- 网络失联不取消已经开始的安装或已有任务；默认继续既有任务，但不执行过期的新命令。重连后先对账再接单。
- 设备离线时停机指令只能标记待送达，不能展示成已停止；需要立即停止时使用本地 CLI。
- 配置/操作协议通过 capability 和版本协商兼容旧客户端，不支持则明确拒绝。

### 6.3 拟议接口

| 类别 | 接口示意 |
| --- | --- |
| 绑定 | POST `/api/pairings`、POST `/api/pairings/{id}/approve`、POST `/api/pairings/{id}/exchange` |
| 设备 | GET `/api/devices`、GET `/api/devices/{id}` |
| 配置 | GET/PUT `/api/devices/{id}/config`、POST `/api/config/validate` |
| 环境 | POST `/api/devices/{id}/environment-plans`、POST `/api/devices/{id}/operations` |
| 任务 | POST `/api/devices/{id}/tasks`、POST `/api/tasks/{id}/actions` |
| 操作 | GET `/api/operations/{id}`、GET `/api/operations/{id}/logs?after=...` |
| 客户端通道 | POST `/api/control/poll`、POST `/api/control/ack`、POST `/api/control/progress` |

Web 写请求沿用安全会话、来源检查和 CSRF 防护；客户端用设备凭据。日志有分页、大小限制和背压；首期轮询，后续再评估 SSE/长轮询，避免大量长连接占满现有服务线程。

## 7. 环境安装与本地构建

### 7.1 安装计划

先生成不可变安装计划，展示目标、来源、版本组合、预计下载/展开空间、受影响任务、权限和回退能力，再确认执行。计划固定制品身份；URL 只是定位方式。空间估算未知时标明未知并设置保守预检，不给出虚假精确值。

环境状态：draft → preflight → downloading/building → verifying_artifacts → installing → validating → ready；错误进入 failed/needs_attention，取消进入 cancelling/cancelled。ready 环境不可原地任意变更；更新创建新 generation 或新 env-id。

### 7.2 下载与供应链

- 由目标客户端直接下载，服务端不作为任意 URL 代理。
- 默认 HTTPS；限制重定向和逐跳目标，阻断云元数据/回环等敏感目标；内网软件源必须在本机授权。
- 校验连接实际目标，考虑 DNS 重绑定；跨源重定向不转发认证头。
- 临时文件、磁盘配额、哈希校验、原子归档；断点续传仅在 ETag/对象一致性可证明时使用。
- 外部 RUN 包是可执行程序；HTTPS 和自行计算 SHA-256 不证明发布者可信。
- 有可信哈希/官方签名则验证；缺少可信证据的包只允许设备所有者明确批准其固定哈希后执行，默认拒绝无人值守安装。
- 安装自有工具/NNSmith 制品需要受信发布清单和签名；明确密钥轮换与撤销。
- URL token、认证头、环境秘密、安装输出中的敏感数据脱敏；避免 token 进入进程参数和审计明文。

### 7.3 CANN 安装适配器

- 识别包类型、版本、芯片、CPU 架构、系统支持范围和非 root 安装选项。
- 按 manifest 安排 Toolkit/算子包顺序；检查已有驱动兼容性。
- 默认安装到 `~/Ascend` 下适配器支持的受管路径；保护现有安装。
- 使用固定 argv 调用已认可安装器，不拼接 shell 命令；用户 URL 或路径不能成为命令片段。
- 接受许可等法律确认必须由用户按发行包要求完成，不能由工具代为假定。
- 不修改全局 shell 配置；把选定 CANN 的环境加载结果限制在执行子进程。
- 无法隔离且会修改共享状态的安装需要维护窗口、风险确认和明确恢复方案，不宣称原子回滚。

### 7.4 Python 依赖与 NNSmith

- 选择已验证 Python/系统架构/CANN/Torch/torch_npu 组合，锁定版本、来源与哈希。
- 初期依赖用户已有受支持 Python + venv 能力；不承诺自动安装系统 Python。缺少 venv/系统库时显示处理要求。
- 采用独立环境，默认不污染系统包；reuse 外部环境作为显式模式，不静默向其安装依赖。
- NNSmith 在发布端固定源码、补丁、Z3 和其他依赖并构建 wheel；验证安装后的源码指纹与现有检查兼容。
- 保留制品来源、许可证和补丁元数据；仅在满足分发许可的情况下提供镜像或打包。
- 下载大型包使用缓存去重；只有无任务、环境、回退版本引用的制品才可按策略回收。

### 7.5 本地源码构建

构建方案是本机登记、受信的 adapter/profile，包括：允许源码根目录、构建入口、类型化参数、构建依赖、输出清单、并行度、超时、预检和验证步骤。网页不可提交任意 shell 文本。

构建源码/脚本本身等同执行本地代码：首次登记需明确授权来源和权限，记录方案摘要；脚本或权限变化触发重新确认策略。检查真实路径和符号链接，防止越过授权根目录。

- incremental：允许 dirty 工作区，沿用受控构建缓存，记录构建前后源码状态；构建期间发生变化标记不可可靠复现。
- snapshot：复制所需工作树内容（包括选定未提交/未跟踪输入，排除秘密及无关文件），生成内容清单，在独立目录构建；不修改原工作树。
- 不执行 reset/clean/checkout/pull；显式清理仅作用于工具拥有的构建目录。
- 同一构建目录互斥；CPU/内存/磁盘预算与测试任务统筹，构建不默认抢占 NPU。
- 产物必须来自本次已完成构建的清单并校验类型/哈希，不能用模糊 glob 选择上一次留下的 RUN 包。
- 每次记录 build_id、commit、dirty 状态/快照哈希、方案、工具链、构建日志、产物 SHA-256。

### 7.6 验证与激活

分级验证：文件/manifest → Python 导入和版本 → CANN 路径/动态库 → NPU 可见性 → 小规模实际测试。

环境“已安装”和“通过所需验证”分开表示；设备忙时标记等待验证，不擅自抢卡。通过检查后发布 env-id；任务持有该环境引用，直到结束才释放。路径中的可变 `latest` 不能作为结果中的唯一版本依据。

## 8. 任务控制与执行语义

### 8.1 模板、运行批次和具体用例

- 模板可编辑、有修订；提交时固定模板和有效参数。
- 运行批次固定 env-id、工具构建、资源决策、配置指纹。
- 每个用例记录 seed、execution/evaluation 快照、epoch 和 attempt。
- 连续 fuzz 的后续配置变更创建新 epoch；历史结果不可改写。

### 8.2 状态与动作

状态：queued → preparing → running → completed/failed；附加 pausing/paused、draining/stopped、cancelling/cancelled、interrupted/needs_attention。

| 动作 | 精确定义 |
| --- | --- |
| start | 验证环境和资源，幂等创建运行批次；只有真实进程握手后标 running |
| pause | 停止新用例生成和新执行派发，在途用例完成；保留 pending 队列 |
| resume | 重验环境、资源、所有权后恢复派发，不重写已有用例参数 |
| graceful stop | 停止生成/派发；允许在途完成；未运行用例标记取消并记录原因 |
| force stop | 先请求退出，超时后仅终止工具拥有且身份核验通过的进程组 |
| daemon stop | 独立管理动作，明确离线后果；默认要求先处理正在运行的管理/测试操作 |

不要仅凭 PID 文件发 kill；结合所有权锁、进程启动标识和受管进程组防止 PID 复用误杀。失去锁或 lease 不代表已有进程已经退出，恢复前必须对账。

### 8.3 资源和恢复

- 首期每设备最多一个活动持续 campaign；复测与本地提交通过统一仲裁避免各自认为有空闲 NPU。
- 保留现有 seed/attempt/result 身份语义，新增关联字段，不修改既有结果事实。
- 内存、磁盘、设备占用与 CPU 预算在启动前和运行中检查；未知能力按保守策略处理。
- 中断不伪装成通过或失败测试；重试生成新 attempt，保留父操作和原因。
- 服务端/客户端重启、重复 ACK、网络延迟、上传积压不能导致重复启动。

## 9. 工具更新和回退

区分：工具版本、执行环境版本、构建产物版本。工具自动更新不得隐式升级 Torch/CANN。

流程：受信清单 → 下载独立版本 → 签名/哈希/协议检查 → 预检查 → 暂停接单和排空 → 保存状态 → 启动器切换 → 本地健康检查与服务端回执。

- 用稳定启动器管理新旧版本；不要在当前包目录执行 pip install -U 或 importlib.reload。
- 开发 editable 安装默认退出自动升级管理，不覆盖源码；用户显式切换到托管模式后才启用。
- 健康检查检查进程、本地控制、状态库、兼容协议；网络暂时不可用与版本启动失败分别处理，避免离线回退循环。
- 数据库优先采用向前/向后兼容的加法迁移；不可逆迁移需要私密一致性备份和维护窗口。
- 回退必须评估新版本已产生的数据；不能直接恢复旧库丢弃新结果。兼容性不满足时阻止自动回退，保留现场进入 needs_attention。
- 发布支持单设备灰度、批次暂停、版本吊销、审计；回退旧版本也是明确授权的新操作，防止降级攻击。

## 10. 存储、审计和可观测性

服务端建议新增独立命名空间/表：devices、device_credentials、config_revisions、device_config_state、environment_specs、environment_instances、builds、artifacts、task_templates、managed_runs、operations、operation_events、audit_events。

客户端 manager.sqlite 保存 inbox、operation_steps、outbox、effective_configs、environment_instances、artifact_refs 和 recovery。结果库/任务 ledger 继续保留职责边界，通过 ID 关联；跨库操作不假设原子事务，使用 outbox 和恢复对账。

- SQLite 事务保持短小，不在事务内等待下载、编译或网络。
- 外键、唯一键、请求幂等键和状态更新 fence 防止重复与越权。
- 审计包含操作者、目标、请求摘要、配置差异、结果、时间；不包含秘密正文。
- 日志按设备授权访问，分页/游标、脱敏、配额、轮转；CLI 与 Web 展示同一事件。
- 上报工具版本、协议能力、desired/actual revision、env-id、任务状态、最后联系时间。
- 所有新增写路由纳入现有维护 write allowlist，避免维护窗口绕过只读限制。
- 备份覆盖控制状态、身份和结果间的一致性关系；客户端凭据和服务端秘密采用私密备份策略。

## 11. Web 与 CLI 交互

### 11.1 Web 页面

1. **设备**：在线/离线、身份、版本、能力、资源、绑定/撤销。
2. **环境**：现有/下载/构建三种来源，版本组合，计划预览，安装日志和验证结果。
3. **任务**：模板、表单/YAML 切换、参数校验、启动/暂停/停止、最终有效配置。
4. **构建**：本地受信方案、增量/快照、构建产物、关联测试结果对比。
5. **更新与审计**：工具版本、灰度、回退能力、操作记录和配置历史。

按能力禁用不可用动作并显示原因；明确“已提交/已接收/等待安全点/已完成”，不使用只表示 HTTP 成功的绿色“已安装”。敏感字段不回显明文。危险操作展示具体影响设备、任务和路径。

### 11.2 CLI 示例

```sh
afx daemon start
afx device enroll --token-stdin
afx environment inspect --json
afx environment plan environment.yaml --json
afx environment apply --plan <plan-id> --json
afx build submit --profile local-dev --json
afx task validate task.yaml --json
afx task submit task.yaml --workers 2 --json
afx task inspect <task-id> --json
afx task pause <task-id> --json
afx task stop <task-id> --graceful --json
afx operation wait <operation-id> --json
afx config effective --json
```

默认本地 CLI 连接本机管理服务；远程控制需显式目标和认证。统一输出 envelope：schema_version、ok、data、error(code/message/details)、operation_id。约定提交成功仅表示 accepted；wait 才给最终退出码；stdout 只输出机器可读结果，诊断写 stderr。

保留现有 `fuzz run/replay/status/stop/setup`，通过适配器逐步接入；旧参数语义不得突然变化。安装器只写受管内容，卸载/清理不得默认删除用户结果、源码或外部 Ascend 目录。

## 12. 开发计划与验收门槛

按依赖排序交付；各阶段必须有自动化测试、错误路径和迁移说明。以下为工作拆分，不是工期承诺；P0 完成后依据首个 CANN 构建入口和实际包组合估算人日。

### P0：契约、兼容性和原型验证

交付：
- 三类配置 schema、字段映射、错误码、操作/任务状态机。
- 决定 YAML 解析库、规范化算法、CLI 覆盖规则。
- 审计轻量入口导入链和 wheel 依赖；确认支持的 Python/系统范围。
- 确认首组 CANN 包、安装参数、Torch 组合、本地构建 profile、制品分发许可。
- 明确数据库迁移、既有身份关联和开发安装退出更新策略。

验收：示例可被 schema 校验；字段可映射或明确延期；空白 Python 环境的引导路径原型不需要重型依赖；安装/回退能力有实际证据而非猜测。

### P1：轻量守护进程、绑定与可靠控制骨架

交付：
- 独立管理入口、惰性导入、依赖拆分、单实例锁、Unix socket。
- 配对/CLI 注册、设备凭据、轮换撤销、服务端设备列表。
- inbox/outbox、操作 ID、幂等、lease/fence、基础日志和恢复。
- systemd user 与容器前台启动方式，保留上传通路。

验收：没有 Torch/NNSmith/CANN 仍可绑定并在线；重复 start 不产生双实例；断网重连和双投递不重复执行；跨用户和过期配对请求被拒绝；秘密不出现在日志。

### P2：环境管理闭环（下载/已有环境）

交付：
- 环境计划、预检、客户端安全下载、CANN 安装适配器。
- 独立 Python 环境、锁定 Torch/torch_npu 组合、预构建 NNSmith 制品。
- 安装/验证/修复状态机、进度、失败恢复、Web 向导与 CLI。
- 现有环境只读检测和采用；环境引用和清理规则。

验收：从只安装 wheel 的客户端，经网页安装真实受支持 CANN 包和依赖并验证；未知包拒绝；下载损坏、磁盘不足、权限不足、中断后重试有明确状态；不污染已有 Python/Ascend 环境。

### P3：任务 YAML、Web 配置与启动/停止

交付：
- 配置 schema 校验、模板、修订、表单/YAML、有效参数和来源展示。
- 调度器适配、本地/远程统一操作、暂停/恢复/安全停止/强制停止。
- 环境/工具/配置/seed 与结果关联，连续任务 epoch。
- 新参数 CLI 与现有 fuzz 命令兼容。

验收：同一 YAML 经 Web/CLI 产生相同有效快照；资源超限明确报错；两张 NPU 在空闲且获准时完成实际路径验证；暂停/停止不杀其他任务；运行中修改模板不改变已有用例；重启恢复不重复测试提交。

**P1–P3 是首个完整可用闭环：安装 wheel → 绑定 → 网页准备环境 → 配置测试 → 启动 → 查看结果 → 安全停止。**

### P4：本地源码构建与开发验证

交付：
- 本机授权 build profile、构建依赖预检、增量/快照模式。
- 产物清单与哈希、build_id、dirty/源码快照证据。
- 构建→安装→验证→测试流水线与阶段单独操作。

验收：在实际开发源码中带未提交修改构建 RUN 包，安装并执行指定 seed；不切分支/清理用户文件；构建失败保留环境；构建产物不会选错为旧包；对比结果能关联到精确构建。

### P5：配置安全应用与工具自升级

交付：
- 热更新分类、worker 排空重建、desired/actual 回执。
- 稳定启动器、签名发布清单、独立工具版本、灰度与健康检查。
- 数据兼容门槛、失败回退、配置回滚、开发模式保护。

验收：网络不良时无虚假成功；升级期间身份/结果/outbox 保留；模拟新版本启动失败可在兼容条件下回退；不兼容数据库迁移阻止自动回退；不原地替换 Torch/CANN。

### P6：安全加固、性能与正式发布

交付：
- 完整权限/供应链/故障注入测试、操作审计、日志配额、缓存 GC。
- 旧客户端兼容、多版本协议、备份恢复演练、运维手册。
- 大量设备轮询和日志回传负载测试，按证据决定是否引入长轮询/SSE。
- 新用户指南、开发者构建指南、Agent CLI/API 文档。

验收：全量核心场景与回归通过；完成私密备份、线上部署和真实 UI/API 验证；回退流程有演练结果。P4、P5 均是最终需求的一部分，不因首个闭环上线而省略。

### 12.1 建议代码职责划分

- 新 `autofusex/fuzz/management/`：schema/config、daemon、transport、operations、environments、builds、updates、cli。
- `fuzz/central/`：新增设备/配置/环境/操作仓储和 API；沿用会话、维护限制、备份机制。
- `fuzz/scheduling/`：只增加管理适配、安全边界和关联元数据，不复制执行引擎。
- `fuzz/view/static/`：设备、环境、任务、构建和版本页面。
- `tools/`：签名制品、兼容清单、NNSmith wheel 和发布验证工具。
- `tests/`：管理契约、协议、权限、安装状态机、调度适配及 UI 测试。

## 13. 验证与部署要求

本项目测试与实际验证在 `npussh mlir` 进行，加载 `$HOME/env.sh` 和项目运行环境。**不在用户本机运行项目测试；oracle1 的服务冒烟不能替代 mlir 的真实 NPU 验证。** 先检查两张卡的占用，保护现有任务；忙时等待或选择不占卡阶段，不以“没有硬件”跳过。

| 测试层 | 必测内容 |
| --- | --- |
| 配置单元 | 重复键、未知字段、超限、单位、互斥分支、默认值、覆盖、指纹稳定 |
| 协议/存储 | 重复投递、ACK 丢失、租约过期、乱序回执、跨库恢复、并发修订冲突 |
| 安全 | 越权、CSRF、配对穷举、撤销凭据、URL 跳转/DNS、路径越界、日志泄密、签名失败 |
| 安装 | 真包成功路径、损坏包、磁盘不足、失败/重启、外部环境保护、许可/权限不足 |
| 构建 | dirty 工作区、并发构建、源码变化、旧产物误选、超时、取消、源码不被清理 |
| 调度/NPU | 双卡、资源约束、暂停/停止、在途任务、环境切换、复测仲裁、非本工具进程保护 |
| 更新 | 下载失败、启动失败、协议不兼容、数据库迁移、离线健康、回滚和结果保留 |
| UI/CLI | 最小 onboarding 闭环、状态真实性、错误可恢复、YAML/表单一致、JSON 输出与退出码 |

服务端功能变更完成后部署到 oracle1 的 `/home/ubuntu/af-fuzz`、`af-fuzz.service`，验证 `https://autofuse.online` 实际 UI/API。部署前重新检查活动 systemd overrides 和服务状态，做私密一致性备份，保留中心数据与身份；先兼容性迁移再渐进开放功能。本文档交付本身不修改运行服务、不触发部署。

## 14. 最终验收清单

- [ ] 干净客户端只安装 wheel 即可启动管理守护进程并完成绑定。
- [ ] 无 NNSmith/Torch/CANN 或执行环境损坏时，管理服务仍在线可诊断。
- [ ] 网页提供 CANN URL 后，可完成受支持包的用户态安装，默认根目录为 `~/Ascend`。
- [ ] Torch、torch_npu、NNSmith 按验证组合自动安装，无需用户 clone/打补丁。
- [ ] 支持已有 CANN、下载 RUN、本地源码构建 RUN 三条路径。
- [ ] dirty 源码的增量构建和可靠快照构建都有明确语义与产物追踪。
- [ ] Web 和 CLI 的配置、环境、任务和更新操作一致，CLI 适合 Agent 自动化。
- [ ] 参数可视化、YAML 导入导出、强校验、实际快照和修订回执完整。
- [ ] 任务启动/暂停/恢复/停止保护在途状态和其他用户进程。
- [ ] 新用例能采用新配置，既有结果和执行快照保持不变。
- [ ] 工具升级可灰度、可核验；回退受数据兼容约束，身份和结果不丢失。
- [ ] 网络故障、重复请求、进程崩溃均不造成盲目重复安装或重复启动。
- [ ] 在 mlir 完成实际环境/NPU 验证，服务端在 oracle1 部署并验证。

## 15. P0 需确认的具体输入

这些不影响总体架构，但会影响适配器、验证矩阵和工期：

1. 首批支持的 CANN RUN 包类型、可用下载来源及非 root 安装约束。
2. 本地构建仓库、实际构建入口、必需工具链和产物位置；不假定所有仓库都采用同一命令。
3. 首组已验证 Python/CANN/Torch/torch_npu/硬件组合，以及现有环境复用方式。
4. 制品托管位置、镜像与内网访问策略、签名发布和密钥管理责任。
5. CANN 安装器对多版本并存/回退的实际支持范围，以及需要用户接受的许可。
6. 第一版 UI/CLI 的操作授权策略、无人值守安装的本地显式授权方式。

在这些输入确定前，不承诺某个未知 RUN 包可安全自动安装或可逆回滚，也不填入猜测的版本兼容关系。


## 16. 本分支实现与验证记录（2026-11-12）

本分支 `codex/managed-client-control` 新增轻量管理包、设备绑定、配置 schema、环境计划/安装适配器、本地 build profile、任务控制、操作回执、Web 管理页、工具发布签名校验和稳定启动器。现有 fuzz/复测上传保持兼容，并增加设备凭据的结果上传隔离。

Docker 中已验证：

- 干净 wheel 安装后管理 CLI/守护进程可以启动，不导入 NumPy、Torch、NNSmith 或 torch_npu；
- YAML 重复键/别名/未知字段/越界值/危险标签被拒绝，表单与 YAML 共用 schema；
- 真实 venv/pip + 模拟 RUN 包安装和版本校验；本地 dirty 源码的增量/快照 build profile、产物哈希和路径约束；
- 配对、设备凭据、配置 revision、操作幂等/租约代次、断线 ACK 重试、审计和撤销；
- Chromium 实际页面登录、配对、表单、模板、操作、审计流程，浏览器控制台无错误；
- 任务暂停/安全停止/连续任务安全边界更新，以及工具签名/健康检查回退流程；
- 管理测试与既有 sync/retest/runtime/stage/ledger 回归测试均在 Docker 运行。

未做且刻意不做：没有在 `mlir` 环境安装 CANN/NNSmith，没有改动其 Python 环境，也没有执行实际 NPU 用例。真实 CANN RUN 包、真实 torch/torch_npu 版本组合和 NPU 双卡验证需要后续单独的维护窗口；验证前应先检查两张卡的占用。

## 17. Agent CLI 命令契约

`afx` 的管理命令是稳定的机器接口，不要求 Agent 解析人类日志。命令成功或失败都向 stdout 输出单个 JSON envelope：

```json
{"schema_version":1,"ok":true,"data":{}}
{"schema_version":1,"ok":false,"error":{"code":"...","message":"..."}}
```

stdout 不混入进度文本；长操作只返回 `data.id`（operation ID），使用 `operation wait` 查询最终状态。退出码固定为：`0` 成功/已接受，`1` 操作已进入 failed/needs_attention/cancelled，`2` 参数或校验错误，`3` 本地 daemon 不可用。`--json` 保留为兼容参数，JSON 是默认输出。

### 17.1 完整命令表

| 命令 | 用途 | 输入/输出 |
| --- | --- | --- |
| `afx daemon start [--foreground]` | 启动守护进程 | 返回本地 socket 已就绪 |
| `afx daemon status` | 查询守护进程、连接、运行和活动操作 | JSON 状态 |
| `afx daemon stop` | 无活动操作时停止守护进程 | 停止结果 |
| `afx daemon run` / `supervise` | 前台运行/稳定启动器 | 供 systemd 或测试使用 |
| `afx device enroll --token-stdin [--endpoint URL] [--name NAME]` | 从 stdin 读取注册 token | 设备 ID；token 不进入 argv |
| `afx device pair [--endpoint URL] [--name NAME]` | 创建浏览器配对 | 配对 ID/码 |
| `afx device status` | 查询本机配对和能力 | JSON 状态 |
| `afx device revoke` | 撤销当前设备凭据 | 撤销结果 |
| `afx environment inspect` / `list` | 检查/列出环境 | 环境列表和平台信息 |
| `afx environment plan FILE` | 校验 YAML 并生成不可变安装计划 | plan ID |
| `afx environment apply PLAN_ID` 或 `--plan PLAN_ID` | 执行已确认计划 | operation ID |
| `afx environment status ENV_ID` | 查询环境 manifest 和状态 | environment ID/name |
| `afx build submit FILE` | 提交本地 CANN 构建 | operation ID/build ID |
| `afx build inspect OPERATION_ID` | 查询构建操作 | operation 状态 |
| `afx task validate FILE` | 解析、补全并校验任务 | 有效任务快照/指纹 |
| `afx task submit FILE [--workers N] [--wait]` | 提交执行任务 | operation ID/run ID |
| `afx task list` | 列出任务 | run 列表 |
| `afx task inspect RUN_ID` / `status` | 查看有效快照和运行状态 | run 状态 |
| `afx task pause RUN_ID` | 安全暂停生成和派发 | operation 状态 |
| `afx task resume RUN_ID` | 恢复任务 | operation 状态 |
| `afx task stop RUN_ID [--graceful]` | 安全停止任务 | operation 状态 |
| `afx task force-stop RUN_ID` | 仅终止本工具拥有的进程组 | operation 状态 |
| `afx task reconfigure RUN_ID FILE [--wait]` | 连续任务在安全 epoch 应用新 YAML | 新 epoch/config hash |
| `afx operation inspect OP_ID` | 查询一次 | 完整操作 envelope |
| `afx operation wait OP_ID [--timeout SEC]` | 等待终态 | 终态和 result |
| `afx operation logs OP_ID [--after SEQ]` | 读取有界结构化事件 | event 列表 |
| `afx operation cancel OP_ID` | 取消排队/请求取消运行操作 | cancel 请求 |
| `afx config validate FILE` | 校验客户端策略 | 有效策略 |
| `afx config effective` | 查询当前策略 | 配置快照 |
| `afx config apply FILE [--wait]` | 应用到后续新任务 | revision/operation |
| `afx config export FILE` | 输出规范化 YAML | YAML 字符串 |
| `afx tool stage FILE [--wait]` | 校验并预安装签名版本 | release ID |
| `afx tool list` / `inspect RELEASE_ID` | 查询托管版本 | staged manifest |
| `afx tool activate RELEASE_ID [--wait]` | 空闲时切换工具版本 | launcher 状态 |
| `afx management trust FILE` | 设置本机信任/资源授权 | 仅本地 socket 可执行 |

所有会产生副作用的命令都支持 `--request-key KEY`。Agent 重试必须复用同一个 key；同一 key 搭配不同 payload 会返回冲突。输入文件使用安全 YAML 子集；传 `-` 表示 stdin。命令帮助可通过任意层级的 `--help` 获取，不能把帮助文本当作状态协议。
