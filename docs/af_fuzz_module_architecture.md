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

第一版仅 schema1；未知/损坏/无归属数据库拒绝写入，不伪装为迁移成功。网页、outbox/revision、升级迁移与快照可信导入尚未实现。当前 snapshot 面向本工具生成的数据库，不是任意外来 SQLite 的安全清洗服务。
