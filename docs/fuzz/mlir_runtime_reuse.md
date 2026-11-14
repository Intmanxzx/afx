# AF-MLIR 冷启动与设备空等诊断

## 现象与计时边界

`E9SQECC3GJ1J8KJBR869MZ1XHC` 的输入为 float32 `[1]`，五个节点、三个标量输出。
历史记录 `073cf274-6705-4fd1-b4dc-3c1ac486bc80` 的首次调用为 41.417 秒，
combined 阶段为 48.811 秒。这不是 kernel 时间：计时覆盖惰性编译、加载、调用和设备同步。

在 `npussh mlir` 两张 Ascend910（逻辑 0/1、物理 12/13）上停止生产 campaign 后，
隔离单卡重放首次调用约 17.273 秒，同步约 0.271 毫秒。双卡基线又复现了约 41 秒首次调用。
这支持 CPU 编译竞争影响墙钟时间，但不是精确的 CPU/NPU 分段归因。

双卡基线对逻辑卡 1 每约 1.5 秒取样，同时用非阻塞 flock 尝试取得同一个物理 UUID 租约。
147 个采样点中有 123 个锁被占用；这 123 个点 AICore 利用率都为 0%，
其中 105 个点有 cc1plus/bisheng 编译进程。监测没有绕过设备锁执行额外工作。
因此确认了调度层面的“CPU 编译期间持卡、同卡托管任务无法进入”。
秒级利用率采样会漏掉很短的 kernel，不能据此宣称硬件全程没有执行指令。

## 有界复用，不拆分编译和执行

AF-MLIR 现在明确声明可复用运行时。成功执行及清理后，worker 才向池报告此能力。
未知能力、AF-legacy 和 Triton 编译后端仍保持逐 session 隔离。

- worker 按 backend、device、compile_threads 隔离；不跨卡共享运行时。
- 每个 case 仍独立生成与 CPU 比较；静态/动态编译选择和 seed 不变。
- 每 session 释放图、tensor、Dynamo 状态；保留 worker 的导入、设备初始化及临时磁盘缓存。
- 每 session 在取得设备锁后重置显存峰值，清理指定卡而非恢复后的默认卡。
- 保留 max-tasks（默认 16）、RSS 增长 256 MiB、设备驻留 16 MiB、缓存预算、异常、超时回收。
- `--worker-max-tasks 1` 可关闭复用。退出仍删除私有临时缓存。
- 设备独占锁不放松；`independent_compile` 和 `transferable_compilation` 仍为 false。

这是减少重复开销的优化，不是 CPU 编译与 NPU 执行的流水化：首次遇到新图时，
编译期间仍可能持卡空等。不能用重复 seed 的缓存命中收益代表随机新图吞吐收益。

## 可重复验证

使用目标环境的 Python，分别指定旧/新代码路径，各自在新目录运行：

```sh
python tools/fuzz_mlir_reuse_probe.py /path/to/new-private-output
```

探针在两张卡分别执行目标 E seed、C16、D16、目标 E seed，共 8 个后端 session、
12 个 shape，与 CPU 参考逐个比较。它使用隔离数据库，不加载生产上传凭据。
检查输出 `report.json` 的正确性、worker PID、cold/warm、RSS、首次调用、整个窗口时间。
不能只比较重复 seed，也不能把快速失败当作性能提升。

原始实验文件在 mlir `/mnt/workspace/af-fuzz/npu-idle-1111/`：
`baseline-trace.jsonl`、`batch-baseline-monitor.log`、`batch-*-result.log`、`summary.json`。

## 双卡对照结果

最终独立顺序对照（基线完成后才运行候选，没有同时运行回归测试）：

| 双卡每卡相同顺序 | 基线阶段时间（两卡） | 复用后阶段时间（两卡） |
| --- | --- | --- |
| 目标 E，首次冷启动 | 47.844 / 49.002 s | 47.866 / 49.128 s |
| C16，不同静态图 | 54.600 / 55.475 s | 46.330 / 46.346 s |
| D16，动态三组 shape | 56.052 / 57.117 s | 34.937 / 35.090 s |
| 再次目标 E | 47.719 / 48.234 s | 1.662 / 1.648 s |
| 整个窗口墙钟 | 223.863 s | 144.933 s |

两组均 8/8 session、12/12 shape 成功；窗口时间减少约 35.3%，吞吐约 1.54 倍。
最后重复 E 的首次调用为 0.727 / 0.722 秒；不能外推为所有新随机图都只需这一时间。
前一轮试验得到 223.939 / 145.021 秒，但基线期间执行过 focused UT，
因此正式结论采用后续不重叠对照。结构化记录见
`docs/benchmarks/mlir-runtime-reuse-20261111.json`。

## 回归与部署边界

- 最终 Linux 定向回归：92 passed、1 failed。唯一失败为既有架构门禁：
  `TorchAdapter._close` 导入 `autofusex.compare`，基线源码已有该依赖；本修改没有新增它。
- 初次完整测试为 646 passed、17 failed、5 skipped，不能声称全量通过。
  包括传输时的 macOS `._*.py` 元数据（已清理）、远端 Node 10 不支持网页测试依赖、
  旧模型捕获/seed/超时等失败；本任务未顺手修改这些无关代码。
- 本地设备锁/编译依赖/日志/阶段测试 36 passed、8 skipped、1 deselected；新增复用、
  回收和 worker key 相关测试通过。Linux 真双卡成功路径与上述 A/B 对照独立验证。
- oracle1 已备份中央结果库与 runtime 库，integrity_check 均为 ok；只覆盖三个运行时文件，
  保留并行任务部署的 UI/统计修改。服务 active，历史用例 API 与首页正常。
- mlir 保留原入口、运行时文件和任务库备份；恢复原无限随机 campaign 的设备、采样、
  dtype、静态/动态策略和 4 worker × 1 compile thread 预算，新 run id
  `60010441-d921-4f5c-9100-e136a491e901`。测试数据在隔离目录，没有上传到生产。

回滚运行时需先正常停止 fuzz，恢复 mlir `production-backup/runtime-before.tgz` 中的三个文件，
再按保存的 campaign 配置启动；不要用旧任务库覆盖新运行结果。服务器回滚同理仅恢复
`/home/ubuntu/af-fuzz/private/npu-reuse-1111/code/` 三个代码文件并重启服务，不回拷历史数据库。
