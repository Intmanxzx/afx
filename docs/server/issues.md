# 修复闭环：聚类、筛选与处置 API

功能签名：首错误日志标准化结果 + function_status/fail_stage。主算子类别不参与功能签名；
同一个错误不会因整张图包含 Reduce、View 或 Transpose 而拆分。
精度与性能各自在自己的维度内仅按主算子类别聚类。后端、dtype、误差大小、
性能倍数和分类器版本不参与签名。当前 issue.backends 配置仍决定哪些后端进入问题池。

主类别优先级：Reduce > Transpose > Concat > Split > Gather > Scatter > View >
Reshape > Broadcast > Elementwise。大小写兼容旧数据，缺少有效类别时为 Unknown。

首日志保留 ERROR、op 名称、dtype、错误码、shape 和错误值。只标准化时间、PID、
源码行号、地址及 graph 标识等动态信息；原始执行日志不修改。无 ERROR 时读取具体
Python 异常、dtype 校验、环境/原生崩溃证据；超时使用稳定超时描述。缺证据使用
No diagnostic evidence，不用生成代码或弃用警告作为错误。issue.log_summary 保存
完整选中行；title 仅作为截断的展示标题。

## 查询

GET /api/issues（公开读取，标准 data envelope）：

| 参数 | 说明 |
|---|---|
| dimension | function / accuracy / performance |
| selected_backends | 逗号分隔后端；backend 为单后端兼容参数 |
| chips | 与总览共用：a2a3 / a5 / 精确芯片值，多选逗号分隔 |
| results | 与总览共用结果分类码；none 表示空选择。精度/性能页不发送此项 |
| categories | 主算子类别，可多选 |
| treatment | ignore / low / normal / high，可多选 |
| status | open / claimed / verifying / regressing / summarizing / resolved / closed，可多选 |
| scope | current：当前失败关联；all：包含历史关联，默认 all |
| q | issue ID、标题或日志文本搜索，最多 128 字符 |
| case_q | 复用 case 搜索表达式，例如 category:Reduce |
| page, limit | 页码从 1 开始，每页 1–200，默认 50 |
| after | 不透明分页游标，不能与 page 同用 |

后端、芯片、结果及 case_q 必须在同一条关联执行上满足。筛选完成后返回去重 issue。
返回 issues、page、page_size、total、total_pages、next_cursor。
每个 issue 增加 backends（完整历史影响后端）、primary_category、treatment、
matched_case_count（筛选命中的唯一 case 数）、current_case_count（当前关联唯一 case 数）。
case_count 保留累计唯一 case 数；列表展示匹配数 / 累计数。旧 backend 字段仅为兼容的
首次后端，不能用于判断完整影响范围。

修复页默认 scope=current、treatment=low,normal,high；不继承总览隐藏错误的默认值。
“全部 issue”清除条件，使用 scope=all，可找到已解决、忽略和已无当前失败的 issue。
侧边面板复用总览组件；后端、芯片、结果与阶段顺序一致，随后为主算子类别、处理标签、
修复状态。不展示具体 Op、dtype、独立精度结果。

GET /api/issues/{id}/cases 支持 scope=current|history|all，默认 all 保持兼容。
current 只返回 issue_current 指向的普通执行头；history 返回已不属于本 issue 的历史关联；
all 返回累计关联证据。历史行的 function_status 等字段属于 last_exec_id 证据，
run_kind 标明是否来自候选包验证；head_* 字段和 current_issue_id 表示当前执行结果和归属。
候选包 verify/regress 不推进普通执行头，其通过结果不代表当前运行版本已经通过。
分页支持 after 和 limit（1–200），游标以返回值为准。
同一 case 在不同后端或芯片的关联分别返回，不可用 case_id 单独去重任务。
GET /api/issues/{id} 返回 backends、主类别、处理标签、current_case_count 及已有事件和轮次信息。
classifications 只来自当前关联；没有当前关联时返回空数组，不再用历史验证证据补齐。
详情分别展示当前关联、历史关联与验证回归轮次，累计 case_count 保留原义。

旧执行头缺少当前关联时，在停止服务写入并备份后可显式补齐：

    PYTHONPATH=. python tools/issue_memberships.py DATA --offline
    PYTHONPATH=. python tools/issue_memberships.py DATA --offline --apply

默认预览会回滚。工具只补齐当前同版本功能/精度失败头的缺失关联，按现行签名归类；
保留旧 issue、last_exec_id、验证结论与轮次，已有当前关联不覆盖。
验证、回归、汇总状态 issue 的全部成员及同错误组跳过，并在报告中计数。

## 管理员设置处理标签

POST /api/admin/issues/{id}/treatment，需要管理员 Bearer token。

请求示例：{"request_key":"unique-operation-id","treatment":"high","reason":"优先修复公共模块"}

四个取值：ignore（忽略）、low（低优先级）、normal（普通优先级，默认）、high（高优先级）。
reason 可省略，最多 500 字符。返回 issue_id、treatment。相同 request_key 与相同请求
重试幂等；改用同 key 提交不同请求返回 409。非管理员返回 403，不存在返回 404，非法输入 400。

忽略不代表已解决。忽略中的 issue 不允许自动或指定领取；改回 high/normal/low 后，
原来 open 的 issue 可重新领取。标签本身不会重开 resolved/closed 的流程状态。
忽略正在处理的 issue 会取消待执行/已领取的复测任务，将未完成轮次标为 invalid，释放
issue 租约并返回 open，旧 worker 结果被拒绝。已完成结果保留。每次操作记录旧/新标签、
操作者、时间和原因。领取按 high、normal、low，再按影响 case 数排序。

验证与回归仍走既有 claim/verify/complete/knowledge API。验证任务按 case + backend +
芯片生成，候选制品的版本和 SHA 必须满足各后端任务；不会只验证首次出现的后端。

## 主动归还已领取的问题

`POST /api/issues/{issue_id}/release`，使用领取者自己的 Bearer token，
`Content-Type: application/json`。请求中的 `lease_id` 取自 claim 响应；
每次新操作使用新的 `request_key`，同一次请求重试保持请求体和 key 一致。

```json
{
  "request_key": "release-F123-001",
  "lease_id": "领取响应中的 lease_id"
}
```

成功返回标准 envelope：

```json
{
  "schema_version": 1,
  "ok": true,
  "data": {"issue_id": "F123", "status": "open"}
}
```

归还会清空领取人、agent 与租约，并记录 `released` 事件。`claimed` 状态退回
`open`，可再次自动或指定领取，原 agent 也可领取其他问题。若已进入验证、回归或
总结阶段，保留阶段、轮次和复测任务；后续领取者可按 issue ID 接手。

未登录返回 401，只读身份返回 403，缺少必填字段返回 400，不存在的问题返回 404。
租约过期、已归还、已被接手或属于其他用户时返回 409 `lease_lost`；
复用同一个 `request_key` 提交不同请求返回 409 `conflict`。
完全相同的请求重试返回首次结果，不重复生成事件，也不影响之后的新领取。

## 归因变更与性能

issue_current 是当前 case/backend/芯片到 issue 的派生关联，独立于验证轮次证据。
上传新执行头及历史归因 apply/rollback 在同一事务内更新受影响关联；不是等下一次上传。
旧 issue 保留历史和人工标签，新签名进入对应 issue；空的旧 issue 不计入 current 列表。
冻结的验证/回归任务不被后台归因变更覆盖。

按执行头、case/backend/芯片、issue/case 建索引。仅处理变化的当前执行；标杆性能结果
变化时更新该 case 同芯片下的目标后端。累计 case 数增量维护，不逐条扫描整个 issue。
生产规模耗时需用一致快照验证。

## 显式重建存量

功能签名移除主类别后，使用保留历史的专用迁移，不能用下方的清空重建替代：

    PYTHONPATH=. python tools/issue_signatures.py DATA --offline
    PYTHONPATH=. python tools/issue_signatures.py DATA --offline --apply

停止服务写入并备份后执行。迁移保留 ID、日志、事件及修复轮次，优先选择已有修复进度的
issue 为合并目标，将未处理副本关闭并记录 duplicate_of，旧签名作为别名继续路由。
含 verifying、regressing 或 summarizing 状态的同错误组整组跳过，不改签名、
用例关联、验证任务或历史执行签名；报告中标记 frozen verification group。
有独立修复进度、终态证据或跨组人工合并的记录列为 deferred，保留其证据与租约；
后续上传使用不含主类别的新签名。迁移支持预览和重复运行。精度、性能签名不受影响。

升级 schema 不自动清理 issue。停止服务写入并创建包含 core/content/stats/secret 的
私有一致备份，在隔离副本执行：

    PYTHONPATH=. python tools/issue_rebuild.py DATA --offline
    PYTHONPATH=. python tools/issue_rebuild.py DATA --offline --apply

默认预览在事务保存点内演练并回滚 issue 变化（Store 初始化仍可能升级 schema）。
apply 全量清理 issues、issue_cases、issue_current、签名别名、事件、轮次、issue 专属
复测任务及其执行关联、issue 范围回归 pin、knowledge_links 的 issue 链接，然后按当前
execution_heads 重建。知识正文及证据、手工复测、全局/feature pin、用户身份、case、
原始执行和日志均保留。知识中的旧 issue ID 属于历史证据。

新 ID 不复用旧 ID，防止旧请求或客户端误操作新问题。重建重置人工标签；不是日常刷新工具。
离线应用失败事务回滚；应用后的完整恢复使用匹配备份，不能覆盖恢复点之后的新上传。

## 开发验证记录

本机合成数据：50,000 条失败执行、2,000 个 issue，20 次首页查询平均约 26 ms；
增加后端和结果条件约 36 ms。数据写入约 9.5 s。这些数字不是线上 SLA，尚未使用
生产完整日志副本验证重建耗时。已有功能、HTTP/OpenAPI、分类刷新/回滚、跨后端验证、
权限和幂等测试；浏览器已验证共享筛选面板和算子类别过滤。

### 问题发现批量复测

“选择全部”选择当前搜索及结果、后端、芯片筛选命中的所有用例，包括其他页；
选择结果是点击时的固定用例列表，翻页保留，改变筛选或清空选择后重置。
复用操作栏的复测和加入回归集。单次支持至多 200,000 个用例。

管理员的复测侧栏支持多选复跑用户及全选，默认选择自己。任务按数量轮流分配，
各用户分得的任务数最多相差一个。requested_by 保留提交人，assigned_user
保存领取用户，不绑定客户端 ID。新客户端使用同一用户身份且匹配后端、芯片时
可领取待执行任务；旧客户端未完成的任务在租约过期后回到同一用户队列。
普通用户只能给自己创建任务，旧批次迁移时自动归属原提交人。
无匹配客户端的任务继续等待，不自动转交其他用户。

结果失败仍完成本次任务，不重新入队。沿用现有租约及有限重领机制。
“我提交的复测”按提交人查询，展示分配用户及数量；大批次明细只展开前 100 个，
汇总统计包含全部任务。

复测目标可选全部四个 NPU 后端：MLIR、Legacy、TritonNPU、EagerNPU，
默认勾选所展示记录中已有的后端。新后端没有历史记录时，沿用该用例已有
NPU 执行记录中的芯片；显式指定芯片优先。CPU 作为参考后端由执行流程处理。

### 无 Autofuse 覆盖的持久化

no_autofuse_coverage 表示真实调度正常返回，但运行探针没有观察到 Autofuse 候选；
已经进入 Autofuse 后发生的异常仍按失败处理。服务器将这类已验证租约的终态回执
保存到 case_coverage，包含用例版本、后端、芯片、软件包信息、来源任务和观测时间。
它是覆盖观测，不伪造精度通过或执行记录，也不把整张图在其他后端判为无效。

创建复测时默认跳过相同版本、后端、芯片的已知无覆盖目标，并报告跳过数量。
侧栏可勾选强制复测；后续有效复测结果会清除该目标的排除记录。用例详情页显示
这些观测。环境中断等其他 terminal_reason 不建立覆盖排除。旧无覆盖回执会迁移，
仅在任务创建时用例尚未更新的情况下归属当前版本；旧记录的软件包信息保留为空。

复测无覆盖目标在对应的当前结果上标为已有的 unsupported（不支持），保留
invalid_reason=no_autofuse_coverage 与标记前状态。列表显示“不支持 · 无 AF 覆盖”，
使用已有“不支持”筛选；通过率分母始终排除不支持，即使勾选显示也不计入。
服务端统计 valid_total 同样排除不支持。历史 invalid 覆盖标记迁移后重建统计。
fuzz 新生成的无覆盖用例继续在客户端跳过，不创建执行记录、不上报新增用例；
只有复测已有用例才提交终态回执，更新服务端已有目标的标记。

### 验证完成后的封存与相似问题

问题首次进入回归中、总结中或已关闭（包括 resolved）后，保留封存时间。重新分类和后续执行不再向该问题追加涉及用例，也不改写历史用例的结果、失效标记或计数；当前失败归属仍随执行头更新。验证与回归任务可以回写自身结果。即使回归失败退回领取状态，再次验证和回归也沿用原有集合。

同一错误再次出现时，创建新的问题，记录 similar_to 并设置高级处理标签；后续同类用例聚合到尚未封存的新问题。列表和详情提供原问题跳转，列表优先展示高级问题。新问题保留原问题的候选版本、摘要与代码链接，可领取后直接提交已有候选包验证。原问题不再自动重开。

提交总结完成修复或管理员关闭后，保留领取人、agent 和领取时间，只撤销租约；该 agent 可以继续领取其他问题。升级会依据未被归还、过期或撤回的最后一次领取事件补回历史已完成问题的领取信息，并记录 owner_restored 事件。

修复状态统一展示为：未领取、领取中、验证中、回归中、总结中、已关闭。“已关闭”过滤同时包含 resolved 和 closed。

### 封存后的补充验证 API

继续使用 POST /api/issues/{issue_id}/verify，不增加新路由或详情页按钮。
当前租约持有人携带 lease_id，管理员可省略 lease_id。request_key 必填；重复请求幂等。
封存问题验证全部涉及用例，含此前 Skip 和未覆盖用例，不受 verify.max_cases 截断。
fix_version 与 fix_sha256 每次必须显式提供，补验相同候选包也不例外。换包后未提供 fix_ref 时清空旧引用，避免把旧代码链接归到新包。
客户端仍须上报匹配的版本及 SHA 才能领取和完成任务；仅版本号相同、SHA 不同视为换包。
已有验证或回归正在运行时返回 409。
若最近一轮回归已通过且候选版本、SHA 都一致，响应 reuse_regression=true，补充验证通过后直接进入总结中；否则仍跑封存的回归集合。失败或无效结果仍退回修复。

### 管理员状态调整

详情页管理员可通过 POST /api/admin/issues/{issue_id}/status 调整状态，必须提供 request_key、expected_status 和 reason。状态已改变时返回 409，防止覆盖其他操作者的进展。

- 未领取：取消未完成任务、将未完成轮次标记为无效、释放领取人和租约。
- 领取中：取消未完成任务和旧租约，保留原领取人；无领取人时归当前管理员。agent 需重新领取获得租约。
- 验证中：使用已有候选包并实际创建验证任务；候选包不存在时拒绝，补验规则与 verify API 一致。
- 回归中：要求最近验证在当前候选包上通过，实际创建回归任务；已有封存回归集合按 case/backend/芯片原样复用。
- 总结中：要求最近轮次通过，已有回归集合须通过回归，或补验已复用通过的回归；不得把运行中或失败轮次手动标为通过。
- 已关闭：取消未完成任务和轮次、撤销租约并封存，保留领取人和 agent，记录关闭原因。

所有调整保留历史轮次、涉及用例、回归用例和已有封存标记；重新打开也不允许自动追加涉及用例。已合并问题不可重新打开，应操作合并目标。开始验证或回归前需恢复被忽略的问题。运行中的验证/回归必须先退回领取中或未领取以显式取消，然后重新开始。

换包后待领取任务不会自动改写目标包。若旧轮次因包不匹配而一直 pending，管理员先将状态退回 claimed 或 open 取消旧轮次，再使用当前客户端上报的版本和 SHA 提交 verify，创建新轮次。旧轮次及结果保留。
客户端在执行前和提交结果前重新读取 graph-autofusion 版本与 pyautofuse.so SHA，并与任务 required 比较（含重启后恢复的领取记录）。检测到变化时报告 environment_changed；服务端终止同轮剩余 pending/leased 任务、撤销租约，并要求重新提交验证。该指纹不覆盖完整 CANN、驱动和固件；运行期间仍须停止换包，避免动态库混用以及两次检测之间的瞬时替换。
