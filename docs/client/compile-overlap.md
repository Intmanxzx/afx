# 编译等待与设备执行重叠

staged fuzz worker 在 AscendC 异步编译产物等待期间可以暂时让出 NPU 卡槽，让另一个用例使用同一张卡。一个用例的所有后端仍完成后统一发布结果。

worker 在让出设备前同步输入传输并释放物理 UUID 文件锁，然后通过私有 socket 通知调度器。CPU token、进程名额、主机内存与设备内存预留继续保留。编译完成后进入同卡恢复队列，优先于新任务取得卡槽，再取得物理文件锁，最后才返回编译调用并执行 kernel/autotune。

适用范围是当前 AscendC compiler 返回的异步构建 future，包括普通 JIT 和静态 Top-N。Top-N 的候选生成/构建在 CPU 上执行；实际 autotune 位于 kernel load 阶段，此时设备租约已恢复。显式 PGO、其他 backend、未知 future、缺失编译接口以及同步编译保留原来的独占执行方式。没有修改安装的 Torch/CANN 文件，也没有跨 worker 传递编译对象。

超时和取消沿用 worker 进程组回收；恢复设备所有权失败会终止 worker，不能进入普通的逐 shape 异常恢复。设备记录中的 device_intervals 表示实际持有卡槽的区间，旧记录仍按完整 lease 区间读取。卡槽占用率不等于 AICore 硬件利用率。

主要验证指标是固定用例集合的后端完成数/分钟、结果一致性、两张卡租约区间是否互斥，以及编译等待与另一会话持卡区间是否重叠。增加流水不会减少新图必须完成的 CPU 编译工作；最终吞吐仍受 CPU 编译能力限制。

## 2026-11-24 双卡验证

mlir 上以隔离源码目录运行候选版本，生产安装不变。固定6个真实用例，各跑CPU、af-legacy、af-mlir；4 workers、8 compile threads、独立worker缓存，比较同一组结果。

|指标|基线|候选|
|---|---:|---:|
|整批墙钟时间|159.63秒|101.78秒|
|NPU后端完成数/分钟|4.51|7.07|
|完成结果|18|18|
|失败/中断|0|0|

完成速率提高56.8%，墙钟时间减少36.2%。18条(seed, backend, status)一致，独立CPU比较均通过。12次NPU后端执行全部出现编译让出区间。两个UUID分别记录18段和6段持卡区间，各卡内部均无重叠；合计worker工作299.65秒中，持卡区间78.24秒，其余主要编译等待可交错。该持卡时间包含初始化/传输/执行/清理，不是kernel时间。

相关回归44 passed、2 skipped（显式NPU升级测试）；新增协议测试7项。基线与候选各跑一批，样本较小，56.8%代表这组用例，不是所有生成分布的固定收益。未测得连续AICore利用率，不能据此声称硬件算力已饱和。生产任务在用例边界排空后测试，finally恢复，验证control=run、state=running。

验证命令：

    python -m pytest -q tests/fuzz/test_compile_overlap.py tests/fuzz/test_fuzz_stage_engine.py tests/fuzz/test_fuzz_device_lease.py tests/fuzz/test_fuzz_runtime.py
