# 完整用例的展示与统计

2026-11-11（America/Phoenix）

- CPU 参考记录继续独立落库、增量同步；原始 attempts/详情/ACK 不变。
- 分阶段执行在每条结果中保存 `expected_backends`（本次 case.tasks 的后端集合）。
- 公共列表、总数、分类图、后端统计及处理耗时样本统一等待集合中的结果全部落库。
  失败、unsupported、超时是终态，不要求 Pass；真正 Unknown 不会被删除或伪装成功。
- 只有 CPU 参考结果不展示、不累计。历史结果没有后端清单时兼容为：至少有一条非 CPU
  后端终态结果才展示；无法从历史记录严格还原未上报的额外后端，不编造历史配置。
- `expected_backends` 不参与 provenance 分组，避免新旧客户端/重试拆成不同用例。
- 独立上传的效率窗口也过滤明确缺少后端结果的组；同组新结果到达触发窗口失效。
- 增量统计缓存版本升级，自动重算。翻页后仍刷新当前游标页，不自动跳回首页。

验证：本地专项 181 passed、2 skipped；浏览器实测 CPU-only 时列表/总数均为 0，
AF-MLIR 结果到齐后 3 个 Shape 同时显示、总数为 3，无 JS 错误。
全量测试并非全绿：macOS Linux /proc/flock 限制、缺 NNSmith 以及既有架构约束失败；
本次仅修复与展示口径变化相关的测试，未修改其他任务的执行/worker 改动。

部署：oracle1 当前 venv-69a6245 仅覆盖本次文件；一致性备份及旧文件在
`/home/ubuntu/af-fuzz/private/completion-1111/`。mlir 对应目录
`/mnt/workspace/af-fuzz/completion-1111/`；保留身份、结果与 ACK，未启动/恢复用户测试任务。
客户端同步 daemon 重启以读取新的结果字段；正在运行的旧 Python 不会自动热加载。
回滚只恢复 old/ 中的代码文件并重启相关服务，禁止用旧数据库覆盖新上传结果。

线上验收：af-fuzz.service active；静态资源哈希与本地一致；浏览器首屏 10 行、无 JS 错误；
summary/statistics 的 Shape 总数均为 18836（验收时快照，不是固定总量）。
升级前后 users=5、sources=6、receipts=38025、attempts=21259，身份记录未变。
远端 JS 测试受系统旧 Node 不支持 node:assert/strict 限制；该项在本地 Node 和真实 Chrome 中验证。
