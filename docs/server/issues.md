# 修复闭环：聚类、筛选与处置 API

功能签名：首错误日志标准化结果 + function_status/fail_stage + 主算子类别。
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

GET /api/issues/{id}/cases 返回 backend，分页游标以返回值为准。
同一 case 在不同后端或芯片的关联分别返回，不可用 case_id 单独去重任务。
GET /api/issues/{id} 返回 backends、主类别、处理标签及已有事件和轮次信息。

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

## 归因变更与性能

issue_current 是当前 case/backend/芯片到 issue 的派生关联，独立于验证轮次证据。
上传新执行头及历史归因 apply/rollback 在同一事务内更新受影响关联；不是等下一次上传。
旧 issue 保留历史和人工标签，新签名进入对应 issue；空的旧 issue 不计入 current 列表。
冻结的验证/回归任务不被后台归因变更覆盖。

按执行头、case/backend/芯片、issue/case 建索引。仅处理变化的当前执行；标杆性能结果
变化时更新该 case 同芯片下的目标后端。累计 case 数增量维护，不逐条扫描整个 issue。
生产规模耗时需用一致快照验证。

## 显式重建存量

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
