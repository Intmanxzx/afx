# 总览覆盖统计口径

2026-11-12（America/Phoenix）

统计单位为当前筛选命中的完整 Shape 用例（group_id + shape_index），不因 CPU
参考执行、多个后端或同一后端重试而重复累计。

- 算子覆盖：每个用例在每个算子类别内计一次；同一用例可同时属于 Elementwise、
  Reduce、Transpose，因此各柱合计可以超过用例数。没有分类元数据时显示“未采集”。
- 数据类型：当前 Shape 输入/输出 dtype 的集合，每种类型计一次，不是张量个数。
  多类型用例可以进入多个柱；缺失元数据显示“未采集”。旧实现取 workload 的首个
  Shape 类型，并且会把没有 dtype 元数据的记录直接略去。
- 最大维度：当前 Shape 所有输入/输出张量中的最大单轴长度，不是 rank。
  独立区间为 0–8、9–64、65–512、513–4096、4097–16384、16385–65536、
  65537–131072、>131072；没有具体 Shape 的早期失败记录显示“未采集”。
- 张量规模：先计算每个输入/输出张量各轴长度的乘积，再取最大值；不是所有张量
  元素数之和。独立区间为 0–1K、(1K,16K]、(16K,64K]、(64K,256K]、
  (256K,1M]、(1M,16M]、(16M,64M]、>64M。K=1000，M=1000000。

两个数值图原本的 SQL CASE 已经互斥，不存在同一 Shape 被累计到多个阈值的问题。
本次将误导性的 ≤ 上界标签改为完整区间，并固定横轴数值顺序、分行展示端点。
算子类别/dtype 仍是多标签覆盖，不应改成互斥类别。

保留原有图例默认值：构图Failed、功能Failed、Unknown 默认隐藏，MLIR Failed、
精度Failed、Pass 显示。页面新增解释；要与总记录数对账，应打开全部状态。

验证全部在 npussh mlir 上加载 $HOME/env.sh 和项目 SQLite runtime 后完成。
已检查两张 NPU 的占用；本次统计/UI 测试不使用 NPU、不终止现有任务。
统计、边界、Shape、重试、筛选、增量缓存及公共缓存专项 50 passed；Node 20.20.0
下图例/UI 和新统计测试 12 passed（包含前一组中的 5 项）。系统 Node 10 不支持
现有测试所用的新语法/模块，因此使用独立目录下经官方 SHA256 校验的 Node 20。

部署只覆盖 reporting/statistics.py、reporting/incremental.py、view/static/charts.js、
view/static/index.html 四个文件，不携带工作区其他并行任务的修改。
增量统计 VERSION 从 4 升为 5，避免新旧标签/计数混用。
Oracle1 部署前对结果库、runtime 库做 SQLite 一致性备份，integrity_check 均为 ok；
备份、原文件及服务配置保存于 /home/ubuntu/af-fuzz/private/coverage-1112/。
回滚只恢复 old/ 中的代码并重启，不用备份数据库覆盖后续上传结果。

线上验收：版本切换后全量派生缓存重建曾返回 503；已完成预热并重启服务消除
重试退避，恢复正常。验收快照 overall=21064、dimension=21064、elements=21064、
dtype=21064，category=39111（多类别覆盖，非重复执行累计）。备份中 users、sources、
attempts 的 ID 在当前库内均无丢失。真实浏览器已加载数据，核对两张数值图的区间
顺序与 SVG 文本，控制台无 error。当前库持续接收结果，上述计数不是固定值。

工作区同时存在其他任务的 retest 修改；部署包保留本次独立验证的 statistics.py，
未夹带随后加入的 has_retests/rollup 变更，其余三个部署文件与本地哈希相同。

后续显示优化：最大维度的长档位仅在前端缩写为 (512,4K]、(4K,16K]、
(16K,64K]、(64K,128K]、>128K，注明 K=1024。API 原始桶、边界与计数不变，
无需缓存重建或服务重启。mlir 专项 12 passed，线上浏览器确认新标签与无控制台错误。
本次静态文件更新前的一致性备份及旧文件位于 private/dim-labels-1112/。

数据类型显示优化（2026-11-12）：总览数据类型图使用 FP16/BF16/FP32/FP64、
U8/U16/U32/U64、I8/I16/I32/I64、BOOL、C64/C128 缩写；未知类型和“未采集”
保持原样。仅映射前端图表显示，不改 API 标签、聚合计数或其他图表。
mlir 隔离目录 `/mnt/workspace/af-fuzz/dtype-labels-1112/` 上，加载 env.sh、
项目运行时及 Node 20 后，总览/覆盖回归 13 passed。已检查双卡占用，未使用 NPU
或中断现有任务。补齐总览导航测试桩缺少的展开状态，兼容当前筛选重置逻辑。

同日补充：各柱状图在当前可见状态下，“未采集”计数为 0 时不再保留柱位；
非零时仍显示，打开对应状态图例后可自动恢复。其他正常分桶的零值不受影响，
原始统计数据不变。mlir 总览/覆盖回归仍为 13 passed，包含隐藏/恢复测试。
Oracle1 的有效 PYTHONPATH release 和 venv 静态脚本已同步，HTTP 返回脚本哈希
与工作区一致，服务 active，无需重启。一致性备份和旧脚本在
`/home/ubuntu/af-fuzz/private/dtype-labels-1112/`，两个数据库 integrity_check 均 ok。
浏览器页面可加载，但本次图表内容未成功读取，不据此声称完成浏览器视觉验收。
