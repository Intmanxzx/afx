# 每个 AF-MLIR worker 的编译并发 1 / 10 对照

在 mlir 双卡环境暂停原 campaign 后，使用相同 seed
`E9SQECC3GJ1J8KJBR869MZ1XHC` 做冷启动对照。每轮使用全新 worker 和私有
Inductor/Triton 缓存，worker_max_tasks=1，两张卡同时运行；CPU 参考校验保持开启。
全局已有的运行时/PCH 缓存没有清空，因此这是 worker/cache 冷启动对比，不是裸机首次安装测试。

## 配置与测量

两组均 3 stage workers；分别 compile_threads=1 和 10。
在 invoke 入口读取 `torch._inductor.config.compile_threads`，确认实际为 1 / 10，
而非仅检查环境变量。机器有效 CPU 配额为 36 核，4×10 不符合现有预算准入，
3×10=30 在额度以内。并没有把不足十个源码文件的 host 线程池强行扩到十路，
也没有把全局默认或单核机器的默认编译并发改成 10。

| 配置 | 双卡首次调用 | 双卡窗口墙钟 |
| --- | --- | --- |
| 1，第一次 | 41.471 / 40.896 s | 62.791 s |
| 10，第一次成功 | 15.581 / 15.412 s | 38.158 s |
| 10，再次成功 | 15.978 / 16.041 s | 37.568 s |
| 1，回测 | 41.830 / 42.582 s | 64.867 s |

所有上述成功轮次均通过 CPU 精度校验，使用新 worker，不是同进程重复图缓存命中。
首次调用包括编译/加载/首次执行/同步，不能当作独立编译时间或 kernel 时间。

最初两轮 10 使用隔离测试的 32 GiB RSS 预算，worker 被 SIGKILL，结果为 error，
不计入性能比较。改为 105 GiB（与原生产预算相当）后重复通过。初始两轮没有采集
取消原因，不能仅凭 SIGKILL 证明是 OOM；当时未见当前容器 memory.failcnt 增长。
预算按进程组 RSS 求和，含共享页重复计数，不等同于物理内存实际占用。

原始脚本、完整日志、每轮结果及实际配置采样保存在远端
`/mnt/workspace/af-fuzz/compile-j10-probe/`，隔离结果没有上传到生产。

## 动态验证与生效配置

D16、compile_threads=10：双卡 6/6 shape 通过，首组调用 18.958 / 18.905 秒，
整个窗口 41.811 秒。

客户端配置已通过 setup 原子更新为 cpu_workers=3、compile_threads=10，保留身份、
凭据路径、内存/缓存配置；无限 campaign 保持原采样策略和双卡设备列表，仅新任务
使用新的 compile_threads，旧任务记录不改写。保留 worker_max_tasks=16 的有界复用。
新 run id 为 `c66689ba-24ba-4269-8235-086589539df8`。

配置备份为远端 probe 目录的 `config-before.json`（私有，不纳入版本控制）、
`budget-before.json`、`campaign-before.json`。此次仅调整该客户端配置，没有修改
编译器源码、全局默认或 oracle1 的服务代码；因此无需服务端代码重新部署。

回滚时正常停止 campaign，恢复 cpu_workers=4、compile_threads=1 的 setup 配置，
并用保存的原 campaign 策略和预算创建新任务；不要覆盖历史结果或回写旧任务契约。
结构化结果见 `docs/benchmarks/compile-j10-20261111.json`。
