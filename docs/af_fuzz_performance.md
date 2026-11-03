# AscendC / Triton-Ascend 性能对比

在 `af-fuzz` 上启用独立的性能模式：

```bash
autofusedbg af-fuzz --compare-performance --out-dir /tmp/af_perf \
  --n-cases 10 --seed 42 --soc-tier A3 --op-categories elewise,reduce \
  --warmup-steps 3 --profile-steps 10
```

单 case 重跑：保持原生成参数，使用 `--repro-seed <case_seed>`，同时保留 `--compare-performance`。无需模型、图、权重或输入文件。性能模式在测量期间导出 profiling，默认解析并保存结果后删除原始 profiling 和本次测量的编译缓存，保留统计 CSV、请求配置、结果 JSON、日志及 JSON/HTML 汇总。

## 测量口径

每种后端先完成编译、一次正确性校验和配置数量的预热，然后同步 NPU。正式 profiling 只执行相同模型和已在设备上的输入，采集 `--profile-steps` 次；输入/权重拷贝、编译、预热和 CPU 校验均在采集区间外。

**每步 kernel 总耗时（μs）= `op_statistic.csv` 中所有算子类型的 `Total Time` 求和 / 实际采集步数。** `Total Time` 已包含该算子的所有出现次数，不再乘 Occurrences，也不累加 Average Time。该指标是 kernel 耗时之和，不是 Python 端到端延迟。默认 Level2 / Text 配置参考 `af_add_ge.py`。

AscendC 加速比 = Triton-Ascend 每步耗时 / AscendC 每步耗时，大于 1 表示 AscendC 更快。worker 超时、profile 缺失、耗时为零或重建出的 case 不同都会标为不可比，不生成有效加速比。精度结果独立展示，不参与性能 pass/fail 判定。数值不一致但仍能执行的后端保留实测耗时及正确性状态，供检查。

## 隔离与可比性

- 两个后端分别启动新的 Python 进程。AscendC 在启动前设置环境变量并传入 `torch.compile` 的 AscendC 选项；Triton-Ascend 在启动前删除继承的 `TORCHINDUCTOR_NPU_BACKEND`，且不传 `npu_backend` 选项，使用 NPU 默认后端。
- 父进程探测 AscendC 候选集合，只传 seed、生成配置和算子/dtype 候选快照。子进程使用共享生成模块重新构造 case，并核对图、完整权重、输入和 CPU oracle 的内容指纹。
- 使用 --jobs 2 --devices 0,1 时，不同 NPU 的常驻 worker 从公共任务池动态领取 case；每张卡最多一个性能 worker。两个后端始终在该 case 的同一卡上串行测量，逐例交替后端顺序。未指定 --devices 时只使用 --target 对应的一张卡；jobs 超过独立卡数时自动限制 worker 数。请在设备无其他负载时解读性能数据。
- 两个后端的每次测量使用独享的编译缓存（包括 Triton 缓存）和新的 profiling 目录，避免导出文件或缓存混用；默认在 worker 退出且结果保存后清理。编译及预热不计入性能数值。

## 参数与输出

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--compare-performance` | 关闭 | 开启后端性能对比 |
| `--n-cases` | 不限制 | 省略时持续运行；正整数指定整个会话的累计用例上限 |
| `--keep-perf-artifacts` | 关闭 | 保留完整 profiling 和编译缓存；默认只保留 CSV/JSON/日志/报告 |
| `--resume` | 关闭 | 从输出目录的检查点续跑 |
| `--warmup-steps` | 3 | 编译完成后的预热步数 |
| `--profile-steps` | 10 | 实际 profiling 的迭代次数 |
| `--perf-timeout-s` | 600 | 每个后端子进程总超时，包括编译、预热和导出 |

原有生成参数 `--method`、`--max-nodes`、`--tensor-scale`、`--max-elem-per-tensor`、`--timeout-ms`、`--value-dist-mode`、`--soc-tier`、`--op-categories`、`--seed` / `--repro-seed` 继续生效。`--timeout-ms` 只约束构图，`--perf-timeout-s` 约束整个后端 worker。性能模式在采集期间导出 profiling，`--debug` / `--no-debug` 不控制这些性能数据；原始产物保留策略由 `--keep-perf-artifacts` 控制，也不导出模型文件。

- `summary.json` / `summary.html`：沿用原汇总报告并添加性能对比和性能对比汇总页签；浏览器表头可排序。
- `cases/<case_id>/<backend>/run-*/`：本轮配置、worker 结果、日志和精简后的 `op_statistic.csv`（若生成了 CSV）。加 `--keep-perf-artifacts` 时还会保留 `profiling/` 原始导出目录。
- `compile_cache/<backend>/run-*/`：每次测量独享的临时编译缓存，默认测量结束后删除；`--keep-perf-artifacts` 可保留。

返回码：全部 case 有有效对比且精度通过时为 0；生成失败、后端失败、数值不一致或不可比时为 1；配置错误或会话级异常为 2；Ctrl+C 为 130，SIGTERM 为 143。性能 fail 本身不改变退出码。

## 模块边界

`nnsmith_integration/generation.py` 为正确性模式和性能模式共用的确定性生成实现。性能代码位于 `nnsmith_integration/performance/`：

| 模块 | 职责 |
|---|---|
| `backends.py` | 后端定义、启动环境和编译选项；新增后端的扩展入口 |
| `cases.py` | 候选快照及完整 case 身份校验 |
| `profiling.py` | Torch-NPU profiling 采集适配器 |
| `statistics.py` | CSV 定位、单位换算及计时聚合 |
| `worker.py` | 一个后端的重建、编译、校验、预热及采集 |
| `runner.py` | 进程启动、超时及子进程组清理 |
| `session.py` / `parallel.py` | 同卡串行配对、跨卡公共任务池、连续运行及乱序完成检查点 |
| `artifacts.py` | 结果落盘、CSV 精简保留及原始 profiling/独享缓存清理 |
| `lifecycle.py` | SIGINT/SIGTERM 停止、子进程父进程死亡保护 |
| `storage.py` / `checkpoint.py` | 原子写入、输出目录互斥锁、随机数状态与续跑配置校验 |
| `schema.py` / `report.py` / `cli.py` | 数据契约、报告与命令行接入 |

## 旧 Python 3.12 环境的编译器兼容性

此前 Python 3.12 环境中 PyTorch 2.14 的 ATen 头文件要求 C++20，而已安装的 Triton-Ascend launcher 仍使用 `-std=c++17`，会在 `precompiled.h.gch` 阶段编译失败。应优先安装与 torch / torch_npu / CANN 匹配的 Triton-Ascend 版本。工具将此情况报告为后端错误、耗时缺失及不可比，不会生成虚假的加速比。

为验证采集与对比链路，本次只在 `/tmp` 创建了编译器包装器，通过当前 Triton 支持的 `CC` 环境变量选择它，将 C++17 参数改为 C++20；没有修改虚拟环境的安装文件，也没有修改后端选择或 kernel 计时口径。该设置不是工具默认行为。需要在相同旧环境临时验证时，可使用：

```bash
cat > /tmp/af-perf-cxx20 <<'SH'
#!/bin/bash
args=()
for arg in "$@"; do
  if [ "$arg" = "-std=c++17" ]; then
    args+=("-std=c++20")
  else
    args+=("$arg")
  fi
done
exec /usr/bin/g++ "${args[@]}"
SH
chmod +x /tmp/af-perf-cxx20
CC=/tmp/af-perf-cxx20 autofusedbg af-fuzz --compare-performance --out-dir /tmp/af_perf --n-cases 1 --seed 42
```

JSON 的每个后端结果包含运行时版本、设备、实际传入的编译选项及相关环境变量，便于核对后端隔离和测试条件。

## summary.html 性能页签

性能模式沿用原 `summary.html` 的“汇总统计”和“用例列表”（功能/精度结果），并增加：

- **性能对比**：逐例展示序号、性能状态、seed、计算节点数、两种后端每步 kernel 数、执行耗时（秒）、每步 kernel 总耗时（μs）、加速比、算子集合、输入/输出 shape 与 dtype、独立精度状态、异常详情。支持按性能状态、算子/seed 筛选以及点击表头排序。
- **性能对比汇总**：总体占比，以及算子分类、dtype、最大维度值、最大元素数、节点数、AscendC 每步 kernel 数等维度的 pass/fail/error 分布。总体 pass rate 的分母为全部用例；多算子类别、多 dtype 可重复计入对应分组。

性能 **pass** 表示 AscendC kernel 时间小于或等于 Triton-Ascend；**fail** 表示 AscendC 更慢；**error** 表示没有有效配对计时（后端失败、耗时无效或指纹不匹配）。性能状态只按有效计时判断，精度结果独立展示。CSV 没有 Count 列时 kernel 数显示“—”；有 Count 列时为所有行计数求和再除以采集步数。

执行耗时是后端 worker 的墙钟时间，包含进程启动、构图、编译、精度校验、预热、采集和导出，**不参与性能判定**。每步 kernel 耗时始终使用 profiling CSV 的 Total Time 口径。JSON 增加 `performance_status`、`performance_passed`、`performance_failed`，保留既有 `status`/`comparable` 调度与采集状态；性能 fail 本身不改变退出码，采集或精度异常仍返回 1。

报告渲染拆分为共享 `summary_html.py`、性能适配器 `performance/report.py`、页签组件 `performance/tabs.py`、判定逻辑 `performance/metrics.py`，便于后续扩展指标和分组。


## 持续运行、停止和断点续跑

省略 `--n-cases` 即持续运行；`--repro-seed` 仍只执行指定的一个 case。显式 `--n-cases 500` 则在累计完成 500 个 case 后结束。功能/精度模式继续保留原来的默认 100 例行为。

```bash
autofusedbg af-fuzz --compare-performance --out-dir /tmp/af_perf_continuous --seed 42
# Ctrl+C，或在另一个终端执行 kill -TERM <主进程PID>
autofusedbg af-fuzz --compare-performance --out-dir /tmp/af_perf_continuous --seed 42 --resume
```

每例完成后原子替换 `summary.json` 和 `summary.html`，不会用部分写入的文件覆盖已有报告；输出目录带互斥锁，不能被两个运行中的会话同时使用。正常停止会终止正在测量的 worker 及其 profiler/compiler 子进程，然后更新报告。报告仅统计完整结束的 case，正在执行的 case 不计入完成数；构图超时、后端编译/采集失败等 case 级结果仍记入报告并继续下一例。会话级异常（例如结果写入失败、产物无法清理）会停止任务，不会无休止地累积大文件。

检查点记录随机数状态、候选算子/dtype 集合、已完成结果和当前未完成的 seed；不保存模型、权重或输入。续跑跳过已完成 case，重跑被中断的 case，后续 seed 序列与未中断运行保持一致。须保持生成参数、预热/采集步数、目标设备、算子候选和软件版本一致；超时和产物保留策略可以调整。有限会话续跑时，`--n-cases` 是累计总上限，例如已完成 100 例后传 `--n-cases 500`，最多再执行 400 例；不传则沿用保存的上限。

SIGKILL（`kill -9`）或断电不能触发退出回调，已原子保存的报告和检查点仍可读取；Linux 上 worker 注册了父进程死亡信号，主进程被强杀时会终止自己的子进程组。续跑时会清理上次未完成 case 的残留 profiling/缓存（除非显式要求保留），再重新测量。断点续跑只适用于新版创建、包含 `checkpoint` 的性能会话，旧报告不能直接当作检查点使用。

默认清理同样覆盖超时和异常场景，错误信息保留在 worker 日志和 JSON；CSV 若已生成则精简保留。清理前先持久化结果，清理失败记录 `cleanup_error` 并停止会话。需要完整现场时，在启动时加 `--keep-perf-artifacts`，或用相同生成配置加 `--repro-seed <case_seed> --keep-perf-artifacts` 重跑。

原始 profiling 与编译缓存默认只占当前测量所需空间；CSV、日志和汇总结果仍随用例数增长，持续运行时也需为这些轻量结果预留空间。


## 多卡动态性能调度

~~~bash
autofusedbg af-fuzz --compare-performance --out-dir /tmp/af_perf_multi --jobs 2 --devices 0,1 --seed 42
~~~

每个常驻 worker 固定绑定一张 NPU，完成当前 case 的两个后端测量后领取下一例。主进程统一生成 index/seed，记录所有已分配未完成任务，并独占汇总报告写入；完成顺序可以不同，报告按 index 排序。省略 --n-cases 继续表示持续运行。

停止时终止所有性能 worker 及其测量子进程，报告只计已收到的完整结果。--resume 会优先重跑全部未完成 case，保留已完成的高 index 结果，然后继续生成新 seed。--jobs 可以调整，--devices 和测量/生成参数仍需与原会话一致。不同卡的硬件状态和共享主机资源可能影响实测耗时，应在报告比较中保持设备条件一致。


## 动态 shape 性能

增加 --dynamic-shapes --dynamic-variants 3，可对同图同权重的三组合法输入尺寸分别采集。每组执行精度校验、预热和 --profile-steps 次采集，后端共享一个 torch.compile(dynamic=True) callable；严格标记变化输入轴，无法符号化时显式失败。case 总耗时为所有尺寸 Total Time 之和除以采集步数总和。逐尺寸结果在 shape_results 和 HTML 的“Shape模式 / 各尺寸耗时”列展示，默认清理后保留 op_statistic_shape_0.csv 等逐尺寸 CSV。

动态候选约束、生成失败含义、seed 复现及续跑条件见 [使用指南第5节](af_fuzz_guide.md#5-动态-shape-测试)。--dynamic-shapes 默认关闭；新参数也已加入使用指南参数表。
