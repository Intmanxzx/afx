# AFX 统一平台重构设计

更新日期：2026-11-23。目标架构、目录与 CLI 迁移已落地，导入与整网中心任务已接入；源码重构及迁移回归验收已完成；生产部署未执行。

## 1. 目标与现状

AFX 是统一的 Autofuse 开发与验证平台，支持自动泛化、人工导入和整网用例。三类任务共享客户端管理、资源管理、任务协议、结果及制品管理，以及适用的问题复测和修复闭环。

当前 platform 是中心服务端及 Web，不等于整个 AFX；其中 client.py 还包含上传、心跳和复测客户端。fuzz 包含生成器、通用执行能力和客户端管理。fuzz/server 只是本地只读查看与报告。根目录已有模型运行和分析能力，但尚未接入完整的平台整网任务链路。

当前 CaseSpec 依赖 Seed 和 NNSmith 图格式，上传及部分复测适配也限定 nnsmith。必须先拆协议与依赖，不能只移动目录。人工导入和整网平台接入是新增能力，目录迁移不代表业务完成。

现有平台业务设计继续作为问题、复测、知识库的依据，但其早期空库部署约定不适用于此次重构：必须保留中央身份、数据与历史执行事实。

## 2. 目标目录结构与功能

以下叶子模块为职责示意，不要求机械创建空文件。

    autofusex/
    ├── __main__.py                 # python -m autofusex
    ├── cli.py                      # 统一命令注册、分发和帮助
    ├── contracts/                  # 公共任务、用例引用、结果、制品协议和校验
    ├── server/                     # 当前 platform 的服务端部分
    │   ├── ...                     # API、账号、数据库、任务、统计、问题、复测、知识库
    │   └── static/                 # 中心 Web 与本地 viewer 唯一共用前端资源
    ├── client/
    │   ├── supervisor/             # 稳定启动器、监护、版本切换、健康检查、回滚
    │   ├── daemon.py               # 可升级客户端主循环，组装业务模块
    │   └── ...                     # 心跳、配置、任务、环境、上传与平台通信
    ├── execution/                  # 公共执行基础设施
    │   ├── ...                     # worker、调度、进程、设备租约、后端适配
    │   └── ...                     # 执行记录、制品、公共比较和结果查询
    ├── workloads/
    │   ├── fuzz/                   # seed、NNSmith、生成、确定性输入和重建
    │   ├── imported/               # 固定用例导入与运行适配，包括 Baize
    │   └── model/                  # 模型入口、输入、整网运行、图采集和融合定位
    └── viewer/                     # 本地只读 HTTP 服务和数据适配

### 2.1 代码迁移边界

| 当前位置 | 目标 | 边界 |
|---|---|---|
| platform 服务端及 static | server | 保留中心业务基础 |
| platform/client.py | client | 上传、心跳和复测适配归执行端 |
| fuzz/daemon/supervisor.py | client/supervisor | 独立部署的稳定层 |
| fuzz/daemon/runtime.py、management | client/daemon.py 及业务模块 | 不再嵌套庞大的 daemon/management |
| fuzz/client/generation、nnsmith、seed 专有协议 | workloads/fuzz | 仅 fuzz 依赖 seed/NNSmith |
| fuzz/client/scheduling、backends、storage 的公共部分 | execution | 去除对生成器规格的强制依赖 |
| fuzz/server/view、reporting | viewer 及 execution 查询模块 | 展示归 viewer，公共执行事实与查询归 execution |
| 模型 runner、adapter、图采集模块 | workloads/model | 专有分析留在 model，公共比较可抽取 |

使用 execution，避免 runtime 同时指环境、常驻进程和 NPU runtime。使用 imported，避免 Python 关键字 import；Baize 只是一种来源适配器。

### 2.2 依赖方向

client 调用 workloads，workloads 使用 execution；各层通过 contracts 交换公共数据。server 通过 HTTP 与 client 交互，不导入客户端管理实现。contracts 不依赖 Torch、NNSmith、数据库或服务器业务。

只有 fuzz 解释 seed 和 NNSmith 规格。server 可以保存来源元数据，不负责重建图。目标是 execution 通过适配器调用具体 workload；当前后端工厂与历史查询仍保留显式 fuzz 适配，新 imported/model 不经过 seed 队列。现有服务端引用客户端配置定义的依赖应迁入 contracts。

### 2.3 supervisor、daemon 与 worker

| 进程 | 职责 | 生命周期 |
|---|---|---|
| supervisor | 启动、异常监护、版本切换、健康检查和失败回滚 | 稳定且独立部署，普通客户端升级不替换 |
| daemon | 心跳、同步配置、领任务、启动 worker、取消与排空、上传 | 随客户端版本升级 |
| worker | 调用 workload 和 execution 执行任务 | 按任务隔离与回收策略管理 |

supervisor 源码虽位于 client 下，但安装位置和运行依赖独立。通过子进程启动目标版本，不导入目标客户端业务模块。IPC 带版本与能力信息，升级前校验兼容性。

升级时 daemon 协调任务排空，supervisor 检查目标版本、停止旧进程、切换并启动，健康检查失败则回滚。最终切换不能依赖被替换的 daemon 继续正常运行。

stop 表达主动停止业务的目标状态，supervisor 不得当作崩溃重新拉起。restart 重启客户端业务以重新加载环境。supervisor 自身维护使用独立流程。

### 2.4 viewer 与 Web

viewer 必须复用 server/static，不维护第二套前端资源。统一数据访问层连接中心 API 或本地只读接口，能力信息控制可用功能；本地接口必须实施只读限制，不能仅隐藏按钮。

viewer 不启动中心服务、不连接中心数据库，也不要求 Torch、NNSmith 或 NPU。静态资源通过包资源读取，不因此导入中心业务。现有独立 HTML 报告迁移须明确产物格式与旧报告读取方式。

## 3. 公共任务与数据协议

| 对象 | 信息 |
|---|---|
| 用例引用 | case_key、revision、workload 类型、来源、规格或制品引用 |
| 任务 | task/run 标识、执行目的、环境要求、资源预算、backend、评价配置 |
| 执行事实 | 用例版本、实际环境与软件版本、状态、阶段、指标和时间 |
| 制品 | 类型、摘要、大小、存储引用和所属执行 |

来源、workload 类型和执行目的分开。Baize 是来源，imported 是运行类型，verify/regress/retest 是执行目的。固定用例也能参加日常测试与回归。

fuzz 保存 seed 和生成协议版本；imported 保存不可变用例版本与入口；model 保存模型、权重、输入和参数引用。其他 workload 不提供伪造 seed，不强制使用小图 DAG 或同一种图格式。

统一任务提交与结果回执；暂停、恢复、改配置按 workload 能力声明。平台复测按类型分发，不统一拼接 fuzz replay。图或输入变化产生新版本，旧记录不得被解释为新图结果。

整网指标独立组织，不混入小图通过率；问题子图可登记为固定用例进入公共闭环。

## 4. CLI 原则

AFX 默认作为客户端，init/start/stop/restart/status 是正式顶层命令，不增加 afx client 层，也不作为另一套长命令的别名。

命令按用户操作组织，不机械对应目录。废弃或重复功能不保留无必要的兼容入口。仍有内部调用的入口先迁移脚本、启动器和调用方，再删除。内部启动与控制协议不需要暴露为日常用户命令。

统一完整根帮助、子命令帮助、公共参数、错误格式和退出码。export 的 stdout 必须可直接交给 edit/validate，不混入提示和外层响应包装。

## 5. 重构前后 CLI 对比

以下省略 afx 前缀。“已实现”指第一轮工作区修改；“目标”表示后续实施，不代表命令已可运行。目标组名沿用讨论方案，参数细节按调用方继续核对。

### 5.1 日常操作与配置

| 重构前 | 重构后 | 状态与语义 |
|---|---|---|
| init/start/stop/restart/status | 原命令 | 保留，调整内部进程边界 |
| --version / -v | 原命令 | 保留 |
| replay SEED... | 原命令 | 保留按本机历史配置排队重放 |
| config show/edit/reset | 原命令 | 已统一配置对象与校验，show 返回 config/source |
| config validate FILE（policy） | config validate FILE（任务） | 已实现，与 edit 同规则 |
| config export FILE（规范化 policy） | config export | 已实现，导出当前任务纯 JSON |
| config effective/apply | 删除 | 已实现，不增加 policy 命令层 |
| management trust/status | 删除 | 已实现，删除手工 trust 写入和重复状态入口 |
| env list/status/prepare/apply | 原命令 | 保留短入口 |
| environment list/status/plan/apply | env list/status/prepare/apply | 目标，合并重复入口 |
| environment inspect | env inspect | 目标 |

本地配置读写任务文件，托管模式访问 /api/config/me。validate 不提交修改。reset 在本地恢复默认并关闭上传，在托管模式撤销个人覆盖、恢复继承。未来客户端设置与 workload 参数在公共配置内分区，不再要求用户管理 client_policy。当前仍仅接入 fuzz 任务 schema。实际执行的冻结配置由 task inspect 查看。

### 5.2 任务、升级与进程

| 重构前 | 重构后 | 状态与语义 |
|---|---|---|
| task validate/submit/list/inspect/status/wait/results | 原命令 | 保留，扩展三类 workload |
| task pause/resume/stop/force-stop/reconfigure | 原命令 | 保留，按 workload 能力校验 |
| task replay SEED... --from-run RUN_ID | 保留并扩展用例版本引用 | 目标 |
| operation inspect/wait/logs/cancel | 原命令 | 保留异步操作管理 |
| build submit/inspect | 原命令 | 保留已有构建能力 |
| client update VERSION（有定义，顶层未路由） | update VERSION | 目标，只升级客户端 |
| client status（顶层未路由） | 合入 status | 目标 |
| tool stage/activate/list/inspect | release stage/activate/list/inspect | 目标，高级版本管理 |
| daemon start/status | start/status | 目标，迁移调用方后删除重复入口 |
| daemon run/supervise | 内部启动入口 | 目标，供 supervisor 或系统服务使用 |
| daemon stop | supervisor stop（维护） | 目标，先实现受控停机，不能仅改名 |
| device enroll/pair/status/revoke | 删除公开入口 | CLI 验收确认 /api/manage 服务端已删除；保留内部旧凭据兼容，普通用户使用 init |

### 5.3 泛化、结果与查看

| 重构前 | 重构后 | 状态与语义 |
|---|---|---|
| fuzz run / run --continue | 原命令 | 保留生成与按已存配置继续 campaign |
| fuzz replay SEED | 原命令 | 保留当前环境直接调试 |
| fuzz status/stop | 原命令 | 保留便利入口，底层统一任务身份和控制 |
| fuzz maintenance resume | fuzz resume | 目标，保留恢复语义 |
| fuzz maintenance seed/diagnose | fuzz seed/diagnose | 目标，生成器专有能力 |
| fuzz maintenance doctor | doctor --workload fuzz | 目标，公共检查加专项检查 |
| fuzz view DATABASE | view DATABASE | 目标，复用 server/static |
| fuzz maintenance results/compare | results list/compare | 目标，保留可比性检查 |
| fuzz maintenance snapshot/restore | results snapshot/restore | 目标，本地结果快照与恢复 |
| fuzz maintenance import | results merge | 目标，合并结果库，不是导入用例 |
| fuzz maintenance check-db | results check | 目标 |
| fuzz maintenance index-results/index-cases | results rebuild-index | 目标，支持选择索引类型 |

顶层 replay 查询本机历史配置，无历史报错；条件满足时插入运行队列，否则提交托管任务，可接受多个 seed。fuzz replay 不需历史，使用当前环境及参数直接执行一个 seed。历史配置引用不等于恢复当时的软件包版本。将来 replay --case CASE_KEY --revision REVISION 由对应 workload 解释规格。

### 5.4 服务端、同步与复测

| 重构前 | 重构后 | 状态与语义 |
|---|---|---|
| platform serve | server serve | 目标 |
| platform user add/disable/enable | server user add/disable/enable | 目标 |
| platform token issue/list/revoke | server token issue/list/revoke | 目标 |
| platform rebuild-stats | server rebuild-stats | 目标 |
| platform backup/verify-backup/restore | server backup/verify-backup/restore | 目标，中心数据与身份备份 |
| platform client setup | init；独立数据源用 sync setup | 目标，保留独立同步场景 |
| platform client run/once/status | sync run/once/status | 目标，日常 daemon 自动同步 |
| platform client retest-once | retest once | 目标 |
| platform client run --retest | 日常 start；独立 retest run 配合同步 | 目标，分清上传与复测执行 |

### 5.5 模型、分析与调试

| 重构前 | 重构后 | 状态与语义 |
|---|---|---|
| run-script | model run | 目标 |
| run-torch-model | model run --entry create_case | 目标，保留工厂入口语义 |
| normalize-script | model normalize | 目标 |
| make-manifest | analyze manifest | 目标 |
| compare/compare-three/compare-npy | analyze compare/compare-three/compare-npy | 目标，公共分析不限整网 |
| report-html | analyze report | 目标 |
| run | debug run | 目标，原功能是外部命令与日志采集 |
| run-debug-cpu | debug run --mode cpu | 目标 |
| camodel-info / run-camodel | debug camodel info/run | 目标 |
| python3 SCRIPT、bash SCRIPT、./SCRIPT.sh | 原快捷形式 | 保留，路由到模型运行能力 |

### 5.6 新增业务，不是改名

| 目标命令 | 功能 | 状态 |
|---|---|---|
| case import PATH | 固定用例导入，适配 Baize、脚本或用例包 | 已实现 |
| case list/inspect | 查询身份、来源、版本和运行规格 | 已实现 |
| case run CASE_KEY | 执行指定版本的固定用例 | 已实现 |
| doctor | 公共客户端与执行环境诊断 | 已整合 |

导入默认登记本地；case run CASE_KEY 执行本地登记版本，case run CASE_KEY --submit 提交中心派发。--upload 控制本地任务完成后的结果同步。中心页面列出导入与整网任务并提供已上传报告下载。

## 6. API 与内部协议清理

已删除本地 trust/effective IPC 和只返回 501 的导入、整网任务 HTTP 路由。后者现在返回 404，不进入 OpenAPI。保留数据表及规划，端到端实现后再注册路由。

仍有调用的配置 API、心跳、上传、复测领取与回执保留。supervisor/daemon 的排空、恢复、shutdown、提交与查询保留内部职责。内部 config.apply、effective_policy、trust 读取和自动初始化仍供同步、资源检查、安装、构建或签名升级使用，不因没有日常命令就删除。

旧设备配对和 central 协议在凭据与同步链路迁移后整体处理。删除接口同步清理路由、分发、调用方、帮助、OpenAPI、测试和文档。逐项依据见 [清理记录](cli_api_cleanup.md)。

## 7. 实施阶段与兼容

1. 配置与接口清理：已完成第一轮。
2. 抽取 contracts：来源无关的身份、任务、结果和制品协议，保留 fuzz 专有规格。
3. 拆 client 与 execution：稳定 supervisor 独立部署，迁移 daemon、worker、同步与资源管理。
4. 迁移 server、viewer、workloads：客户端逻辑归 client，本地查看复用唯一静态资源。
5. 接入 imported：完成固定 PyTorch 用例导入、版本登记、执行、上传、归类与复测，验证公共边界。
6. 接入 model：复用已有 runner，补齐平台任务、报告与问题子图导出。
7. CLI 收敛：更新脚本、启动器、systemd、wheel 资源路径与帮助，迁移调用方后删除旧入口。

每阶段独立验证。数据库、身份、历史用例与产物、任务和升级状态逐项列迁移方案。源码兼容、命令兼容与数据兼容分开处理，删除旧命令不等于删除旧数据。

部署前对中央数据与身份做私有一致性备份，升级后检查服务、UI 和 API；本文不代表已执行部署。

## 8. 当前状态与验收

已在工作区实现：统一 config；删除旧 policy 用户入口、management trust/status、trust/effective IPC；删除 501 占位路由；更新文档、页面提示及测试。

已实施：目录迁移、supervisor 固定快照、CLI 收敛、server/static 共享、人工导入与整网任务执行和中心派发。当前验收结果及保留边界见第 13 节；早期阶段记录按时间保留。

上一轮在 mlir 独立临时目录验证：客户端 13 项、平台 33 项、OpenAPI 4 项通过。平台测试使用临时 SQLite 和测试依赖，未修改服务运行环境；不是 NPU 执行验证。

后续验收要求：

- 根帮助完整，废弃命令不保留无功能别名，内部启动不混入日常帮助。
- config export → validate → edit 可往返，配置来源清晰。
- contracts、server、viewer 不加载执行框架，seed/NNSmith 专有逻辑限定 fuzz。
- supervisor 独立监护、切换、回滚；主动停止不被拉起；普通升级不替换稳定层。
- 三类 workload 共用任务、执行事实与制品协议，复测遵循用例版本及实际安装版本。
- viewer 与中心 Web 使用同一份 server/static，本地只读由接口保证。
- wheel、部署脚本及内部模块路径同步更新。
- 迁移保留 token、客户端身份、历史记录、用例版本和升级状态。
- 执行相关变更通过 npussh mlir 加载环境及项目运行时，检查两张 NPU，保留已有任务；本地 smoke 不代替真实 NPU 验证。

## 9. 相关文档

- [CLI 与 API 清理记录](cli_api_cleanup.md)：第一阶段实际范围与保留理由。
- [平台业务与数据设计](fuzz/platform_v3_design.md)：现有中心业务与复测协议。
- [执行引擎架构](fuzz/fuzz_design.md)：重构前执行链路。
- [客户端配置](fuzz/client_configuration.md)：当前可用行为。

## 10. 第一批协议抽取进展

阶段快照已提交为 1ff7642（配置/API 清理与设计文档）。随后开始配置协议拆分：

- 新增 contracts/configuration.py：解析、规范化、摘要、合并、数量单位及共用字段定义；不导入 workload 或执行框架。
- 新增 workloads/fuzz/configuration.py：保留现有配置 wire format、默认值及 fuzz 语义校验。
- 删除旧 fuzz/daemon/management/config.py，迁移服务端、daemon 和测试的调用路径，不保留转发模块。
- 服务端任务配置不再依赖 daemon 实现；当前仍明确调用 fuzz 配置校验，尚未成为三类 workload 的统一注册入口。
- 环境与内部策略 schema 暂随旧配置验证器迁移，后续继续从 fuzz 配置中拆出；当前不是完整的来源无关任务协议。

验证在 mlir 临时目录执行：公共协议隔离、配置、客户端、平台相关测试共 60 项最终通过。首次传输包含 macOS 元数据文件导致架构扫描失败，清理临时目录中的元数据后复测；架构检查显式记录四个现有平台客户端适配入口，待 platform/client.py 迁移时移除该过渡例外。

本批未变更执行算法、部署服务或升级运行中的客户端。

## 11. 目录迁移验证记录（2026-11-23）

本批迁移 server、client、execution、workloads/fuzz、workloads/model 和 viewer 的源码归属，更新内部导入、子进程模块入口、部署检查及 wheel 资源路径。公共 ExecutionConfig、EvaluationConfig、ShapeResult、IPC 与确定性 JSON 编码归 contracts。历史 CaseSpec 内的 materializer 类名仍使用原 wire namespace，不随 Python 包路径变化；历史身份与生成指纹不能因移动文件改变。

这是目录迁移阶段，尚未达到最终依赖边界：execution 的现有队列与统计仍有 fuzz 专用逻辑，supervisor 尚需部署隔离，viewer 尚有独立前端资源，CLI 收敛和 imported/model 平台接入继续实施。不能把空目录或移动源码计为新增业务完成。

mlir 独立临时目录全量回归结果为 1371 passed、7 failed、2 skipped（474 秒）。失败项均在迁移前基线复现：历史 CPU-only 状态期望、生成 golden fingerprints、已删除 session.js 的遗留测试，以及四项独立同步 setup 测试。同步 setup 的实际缺陷是未创建状态数据库父目录，本批修复；其余失败保留记录，不能通过重写历史指纹掩盖。全量中的后台执行测试已通过。加载项目环境后检查两个 NPU 芯片均为 Health OK；未部署或替换现有客户端。

同步初始化修复与序列化抽取后的定向回归：32 passed（96.75 秒），覆盖 contracts 隔离、架构边界、同步 setup、任务队列及后台执行。

CLI 第二阶段已实现 server 与 sync/retest 的职责分离：server 不再注册 client 子命令；sync 提供 setup/run/once/status，retest 提供 run/once（持续模式保留上传与心跳）。部署入口改用 server。mlir 路由和备份回归 10 passed；其他 CLI 分组仍待完成。

## 12. CLI、共享资源与稳定 supervisor（2026-11-23）

- model run/normalize、analyze manifest/compare/compare-three/compare-npy/report、debug run/camodel 已迁移；model run --entry 保留工厂入口的 fullgraph 默认值。
- results list/compare/snapshot/restore/merge/check/rebuild-index 与 view、doctor 成为顶层命令；fuzz 保留 run/replay/status/stop/resume/seed/diagnose。根帮助列出完整用户入口。
- update、release、env inspect 已路由；daemon 只通过内部 python -m autofusex.client 入口启动。
- viewer 不再有 static 目录；同一 server/static/index.html 和 bootstrap.js 按服务端能力信息加载中心模块或本地模块。统计与详情暂保留各自的数据适配和渲染模块，以保留原有功能；共享组件的进一步合并仍可继续。接口继续拒绝本地写入。
- supervisor 首次启动安装固定源码快照到 HOME/supervisor/v1，普通客户端更新不覆盖。快照只含监督与状态协议所需模块，不导入 daemon、workload 或 YAML。维护更新需单独流程；目标客户端仍通过独立 Python 子进程启动。

验证：mlir CLI/模型/结果相关 173 passed；共享资源与管理流程首轮 160 passed，修正遗留命令调用和 wheel 夹具后定向 68 passed，另 1 项缺失数据库错误处理复测通过。新增 supervisor 冻结包测试覆盖重复安装不替换与隔离导入。此阶段尚未接入非 fuzz 平台任务，后续实现及验收见第 13 节。


## 13. 导入、整网与最终边界验收

固定用例通过 case import 登记不可变源码包，不在导入时执行代码。人工脚本、Baize PyTorch/TF 来源保存在来源字段中；imported 与 model 分别使用 I: 和 M: 内容摘要身份，修改内容产生新 revision。case list/inspect 查询来源与版本；case run 支持指定环境、设备、CPU 对照、fullgraph、dynamic、误差阈值和脚本参数。

imported_task/model_task 与 fuzz_task 由 client/configuration 分派校验；环境及本机策略 schema 属于 contracts/control_configuration。固定任务规格不包含 seed、NNSmith 图或 fuzz DAG。client 复用持久化 operation、任务锁、环境解析和资源准入，execution/source_task 负责子进程组、停止、超时与内存约束；workloads/model/task 复用已有脚本和模型 runner，生成版本、软件环境、结果及产物摘要。

中心 schema 17 新增 source_cases/source_jobs/source_artifacts，不修改历史 fuzz 表及指纹。source API 实现登记、提交、领取、完成及产物上传下载，按用户隔离；重复领取与结果回执幂等。客户端将领取转为持久化 task.submit，重启后可以恢复；已有任务占用时排队。租约不自动跨机器抢占，避免旧 worker 仍执行时产生重复运行。报告文件按摘要校验，目前上传文本报告、JSON、CSV、SVG，单文件上限 1 MiB；原始大 tensor 仍保留本地。

seed ledger、campaign、staged worker/pool、队列 supervisor 和图制品适配归 workloads/fuzz/scheduling，NNSmith TorchAdapter 归 workloads/fuzz/torch_backend。execution 保留公共资源、设备锁、预算和执行事实；历史结果查询/报告中仍有 fuzz seed 适配，以维持旧数据库查询和展示，不要求 imported/model 使用该格式。

验证记录：

- mlir imported 在 npu:0、model 在 npu:1 的真实执行均通过，分别产生 52 和 48 个产物；既有 fuzz 任务未停止。
- 两张 NPU 均检查为 Health OK。
- 中心及本地 viewer 的浏览器检查均无页面错误，统一使用 server/static；wheel 包含共享资源和 supervisor，不包含旧 viewer/static。
- 新 source HTTP/任务/协议/管理与升级定向回归 26 passed，包含报告上传下载和未认证拒绝。
- 全量回归 1388 passed、5 failed、2 skipped。两项为已确认基线问题（旧状态期望、golden fingerprint）；共享入口断言已更新，其余两项 worker 时序失败继续定向核查。最终边界迁移之后的回归仍在执行。

这里记录的是源码与验证进度，未表示中心生产部署已完成。

补充验证：Manager.tick 领取任务到 operation 执行、停止结果回传的集成回归 5 passed。worker 超时起点修正为子进程启动之后，避免数据库初始化消耗执行预算；超时进程组清理及键盘中断两项定向复测通过。最终 wheel 构建及目录检查通过。


### 最终验收汇总

源码提交：127b01c 完成 imported/model 中心闭环和 fuzz 调度归位；906f419 补齐固定源码入口校验。此前 1ff7642、8498119、6fef085、d2a9430、99148de 分别覆盖配置、协议、目录、服务命令、共享 UI 与稳定 supervisor。

最后一批大范围回归为 1118 passed、8 failed、2 skipped。失败逐项闭环：两项缺失文档夹具和一项后台任务等待超时，在补齐临时测试目录后 3 passed；共享入口断言在后续组合回归通过；TorchAdapter 测试导入路径修正后阶段执行引擎 32 passed、2 skipped。worker 超时与键盘中断在修正计时起点后 2 passed。固定源码入口校验在 mlir 4 passed。没有重新修改历史 golden 来制造全绿。

最终保留两项基线失败：test_legacy_cpu_only_hidden_but_terminal_target_unknown_visible 的状态预期与现有 environment_failed 分类不一致；test_current_golden_fingerprints 与已有生成行为不一致。它们已在迁移前基线复现。本次未改变生成算法来掩盖差异。上述计数属于不同测试快照和定向复测，不能合并宣称最终单次全量全绿。

最终检查两张 NPU 均 Health OK；未停止既有业务任务。wheel 构建、共享静态资源与新模块归属检查通过。生产中心和已有客户端尚未部署此版本，部署仍需一致性备份及线上验证。

2026-11-23 补充：golden 失败已进一步定位并修复测试输入。测试隐式使用变化后的默认分布，实际生成了不同 seed；四个相同历史 seed 的完整 manifest 和旧 golden 均复现一致。同时修复实际 Torch 版本写入用例身份的问题，保留历史 wire 标记。详见 [seed 身份与分类边界核查](seed_identity_audit.md)。分类规则仍是本地与中心两套实现，统一前端不表示统一分类。
