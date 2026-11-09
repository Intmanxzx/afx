# 独立 fuzz 模块与扩展边界

fuzz 位于 `autofusedebugger/fuzz`，同一 pip 包分发，但不依赖旧 debugger/nnsmith_integration。
可独立执行 `python -m autofusedebugger.fuzz seed 42`；根 CLI 仅转发参数。

```
cli -> generation registry -> versioned generation/v1 -> contracts + tensors
    -> backend factory -> backend adapter             -> contracts + tensors
    -> execution (one sample loop) -> adapter protocol + generation oracle
    -> storage (lightweight results; never tensors/backend execution)
```

- **contracts.py**：不可变用例、seed、执行/评估配置；不 import Torch/存储/CLI。
- **tensors.py**：规范字节解码；不依赖设备。
- **generation/**：明确版本 dispatch 与冻结版本实现；新增版本不得改旧 seed golden。
- **backends/base.py**：adapter Protocol/unsupported；工厂延迟加载具体后端。
- **backends/torch.py**：Torch/AscendC/Triton 设备绑定和调用，重依赖只在 check 时加载。
- **execution.py**：静态/动态共用循环、正确性/测量状态、错误阶段；不包含 Torch 实现。
- **config/resources**：路径、身份引用、预算与只读资源诊断。
- **storage/**：版本化 schema、结果白名单持久化/分页/一致性快照，不导入生成/执行依赖。
- **cli.py**：参数和用例编排，不混入生成/数据库算法。

架构测试禁止 fuzz import 旧 debugger/NNSmith、禁止生成/契约反向依赖 CLI/后端/存储，并在拒绝重依赖的进程中运行独立模块入口。

扩展流程：新增生成版本保留旧 golden；新增 backend 实现 adapter/factory 并增加能力测试；新增结果 schema 用明确迁移（当前未知版本拒绝）；view/sync 只能依赖轻量 storage 契约。不能通过文件搬迁制造多份执行实现。

当前限制：生成 v1 因素范围有限；Torch adapter 仅支持已验证的最小图；历史 NNSmith 实验仍留在旧目录作调查材料，但不是独立模块的依赖。独立模块不意味着拆分为另一个 pip 包。

## 结果层首批接口

`storage/records.py` 负责白名单、短字段和数值校验；`storage/schema.py` 拥有 schema/application ID；`storage/repository.py` 提供事务 save、幂等 attempt ID、分页 list、detail 与 snapshot。

新 replay 默认只保存轻量结果到数据目录的 `results/results.sqlite`；`--database` 指定位置，`--no-save` 显式只运行。上下文（执行/评估/环境）内容寻址去重；每次 replay 新 attempt，调用者重传相同 attempt ID 和内容才幂等，不按 seed 删除重复实验。

`snapshot` 在 SQLite read transaction 中从白名单表分页复制，包含源 WAL 中已提交数据，排除额外表/lease/outbox/secrets。新文件完整性检查后原子无覆盖发布。不是直接复制 WAL 主文件，不执行源 SQL/pickle。

第一版仅 schema1；未知/损坏/无归属数据库拒绝写入，不伪装为迁移成功。离线网页与共享查询已实现（见下节）；outbox/revision、升级迁移与快照可信导入尚未实现。当前 snapshot 面向本工具生成的数据库，不是任意外来 SQLite 的安全清洗服务。

## 离线 view 与共享查询

`storage/queries.py` 拥有只读过滤/游标分页/汇总/详情契约，repository 的 list/detail 和 view 都复用该实现，不维护两套查询语义。`view/server.py` 只负责 loopback HTTP 适配；`view/static/` 是 wheel 内的预构建 HTML/CSS/JS，不需要 Node、前端构建器或模型依赖。

`view <database>` 是用户主动启动、Ctrl+C 退出的临时 PC 服务；只绑定 127.0.0.1，随机 capability 路径、Host/Origin 限制、CSP、只读连接、参数/分页/查询耗时限制。不是公网服务，也不是 NPU 常驻 API。该边界不是任意恶意 SQLite 文件的安全沙箱；仅打开可信来源的本工具结果/快照。中心服务的认证、授权、上传协议、线程/连接资源隔离将独立实现。

`storage/transfer.py` 单独负责可信恢复、合并和完整性审核，复用 records 的规范化和 schema 白名单。restore 新文件原子发布；merge 保持源 read transaction + 目标 IMMEDIATE write transaction。现有 results schema 不放任务状态，后续调度操作数据与长期结果分离，snapshot 不携带运行租约。

## 调度操作层

`scheduling/ledger.py` 仅依赖 contracts/标准库，提供有界持久化队列、事务领取、停止/中断/显式恢复、旧 claim 栅栏。任务操作库使用独立 application ID；不是长期结果归档或 seed 重放前提。运行时 supervisor/worker 将依赖 ledger + 统一 execution，不再复制一套执行引擎。当前尚未接执行器、设备锁或 CLI，不自动重领失联任务，必须先确认旧 worker 已终止。

### 前台管理层

- `scheduling/ownership.py`：POSIX 队列 flock，父进程关闭 FD 不显式解锁，继承的 worker FD 继续持有所有权；不是 NPU lease。
- `scheduling/supervisor.py`：进程生命周期、期限/停止、claim/result 恢复协调、队列→结果路径绑定；不重复执行逻辑。
- `scheduling/worker.py`：一个 task 的标准生成/后端/执行/持久化；只写结果，parent 确认任务。
- `scheduling/cli.py`：前台 CPU 命令编排；后台托管、NPU 设备映射/租约与 setup 完整资源预算仍待接入。

操作库及 `.results.json` 绑定文件应放在稳定本机路径，不可在运行中替换数据库 inode。备份结果使用 snapshot；不能用结果恢复命令恢复正在运行的队列。管理命令可运行而不导入 Torch，只有 worker 加载后端依赖。

`scheduling/migrations.py` 只处理操作库显式schema1→2，要求独立新备份及非活跃队列；普通连接拒绝旧/未知版本。schema2 的reason是短生命周期摘要，不是模型结果。enqueue_many拥有原子批次边界，CLI不再逐个提交可能失败的半批。
