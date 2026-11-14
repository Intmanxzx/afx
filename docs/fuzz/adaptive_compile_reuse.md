# 自适应编译并发与 worker 复用

> 本文记录早期 4×9/编译上限 10 的实验，不是当前默认配置。
> 当前启动策略和上限 6 以 [自适应资源策略 v2](resource_policy.md) 为准。

## 策略

`fuzz run --auto-resources` 在启动新任务时重新读取 host CPU、affinity、cgroup CPU 配额、
有效内存及配置的内存预算，不把 `os.cpu_count()` 当成可用 CPU 配额。

先选 pipeline worker 数（沿用 NPU 数量及 CPU/RAM 上限策略），再分配每 worker 编译并发：

```python
workers = max(1, min(effective_cores, memory_bytes // (2 * GiB), max(4, 2 * npu_count)))
compile_threads = max(1, min(10, effective_cores // workers,
                             memory_bytes // (workers * 2 * GiB)))
```

10 是已测得有效的编译并发目标，不是强制下限。每编译槽 2 GiB 是保守的初始预算启发式，
不是实际内存预测或硬隔离；现有进程组 RSS、缓存、worker 回收与取消保护仍生效。
Autofuse 内部其他线程池不一定受 Inductor 配置限制，不能把公式当作操作系统并发硬上限。

- 可用 40 核、双卡、内存充分：4 workers × 10。
- host/affinity 为 40、cgroup quota 为 36 核、双卡、约 105 GiB 预算：4 × 9。
- 有效 16 核、双卡、内存充分：4 × 4。
- 内存预算较低时进一步限制线程数；有效资源未知时保守降级。

`--auto-resources` 与显式 `--workers` / `--compile-threads` 仍互斥。
不覆盖用户显式配置，不改写历史任务；运行中不热改正在执行的编译契约。

## 复用边界

上一轮 3×10 已启用复用。此次比较的是“增加 CPU pipeline worker、相应将编译并发调为 9”
与复用的组合收益，而非重新发明复用。

固定双卡每卡顺序 E9SQECC3GJ1J8KJBR869MZ1XHC、C16、D16、再次 E，
对比 3×10/复用、4×9/不复用、4×9/复用。
每组 8 个后端 session、12 个 shape，使用隔离库、新 worker 池和私有缓存。
不复用使用 worker_max_tasks=1；复用使用 16。首次冷启动、不同新图和重复 E 分开看。
日志保存在 mlir `/mnt/workspace/af-fuzz/adaptive-reuse-1112/`。

可通过 `tools/fuzz_mlir_reuse_probe.py` 的 `--workers`、`--compile-threads`、
`--worker-max-tasks`、`--memory-gib` 复现。显式输出目录必须不存在，避免混入旧记录。

## 实测结果

| 配置 | 双卡窗口墙钟 | 正确性 |
| --- | --- | --- |
| 3×10、max_tasks=16 | 81.081 s | 8 session / 12 shape 全通过 |
| 4×9、max_tasks=1 | 109.232 s | 同上 |
| 4×9、max_tasks=16 | 57.411 s | 同上 |

4×9＋复用比 3×10＋复用减少 29.2%，比 4×9 不复用减少 47.4%。这是一次固定
小样本对照，包含最后重复 E 的缓存收益，不外推为整个随机 campaign 的提速比例。
3×10 组卡 1 的 worker 在首个 E 后发生更换，C16 为 cold；4×9 复用组的两卡 PID
均保持，各卡后三个 session 都是 warm。不能仅凭时间线断言每次更换的具体原因。

4×9 复用组按阶段细分：首次 E 为 20.252 / 20.417 秒；不同静态图 C16 为
10.982 / 10.851 秒；D16 为 11.489 / 11.712 秒；重复 E 为 1.567 / 1.566 秒。
结果结构化归档为 `docs/benchmarks/adaptive-reuse-20261112.json`。

## 验证与部署

本地资源/复用相关测试 43 passed；远端资源测试补齐 tools 测试依赖后 28 passed。
新增覆盖 40/36/16/3/1 核、不同 NPU 数、内存不足及资源未知场景。
现有 `--auto-resources` 入口使用新公式；未改变用户显式设置优先级。

mlir 已按实时资源自适应计算并应用 4×9，保留原采样、身份、预算及 max_tasks=16，
新无限 run id `5c9039bc-b63d-4710-b2dc-8e8feb7e4244`。
oracle1 结果库/runtime 库在私有 `adaptive-reuse-1112` 目录一致性备份，完整性均 ok；
只更新 resources.py，保留其他部署文件，服务重启后 active。
回滚只恢复该目录 resources-before.py；客户端停止后按保存的 campaign/budget/config
重启，不回写历史任务或数据库。测试数据未上传生产。
