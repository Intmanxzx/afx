# Native 日志管道 / GIL 死锁修复

## 原因

2026-11-11（America/Phoenix）在 mlir 两张 Ascend910 上观察到 worker 主线程
阻塞于 `libunified_dlog -> write(stdout)`，调用链仍在 `rtSetDevice` 初始化。
同进程 `capture_tail` 的 Python 读线程在 `PyEval_RestoreThread -> take_gil` 等待 GIL。
原生调用持有 GIL、日志管道写满、读线程拿不到 GIL，形成循环等待。
`ASCEND_SLOG_PRINT_TO_STDOUT=1` 时可触发，不是 MLIR 编译耗时超过 120 秒。

## 修复

- `capture_tail` 使用独立 Python 子进程读取管道，不与生产日志的线程争抢 GIL。
- 子进程关闭无关 FD、不继承任务队列锁，留在 worker 进程组，随取消一起回收。
- 日志只保留有界内存尾部、错误上下文和 MLIR verdict，不落无限增长日志文件。
- 返回的诊断脱敏、限制 1600 字符；上下文退出后可通过 `tail()` 获取。
- 父进程使用 `read1` 及时接收短日志；超时回收保留日志，不再丢弃 `finish()` 结果。
- 保留原 120 秒限制和 CANN 日志开关，不靠增大超时/关日志掩盖死锁。

## 验证

- macOS 针对日志、分组及历史分类测试：102 passed。
- mlir 加载 env.sh、GCC libstdc++、SQLite runtime 和 NNSmith 后：105 passed。
  裸环境的首次测试缺 SQLite/NNSmith 等运行依赖，未将其当作有效环境验收。
- ctypes.PyDLL 模拟持有 GIL 的原生调用写 1 MiB：旧实现超时，新实现完成；
  覆盖 forward 开关、异常恢复、重复捕获、崩溃日志和超时诊断脱敏。
- 隔离重放两个此前超时的 seed，保持 120 秒限制并分别用 npu:0 / npu:1：
  - E9SQECC3GJ1J8KJBR868L9N2QS：capability 1.834 秒，总 execution 20.057 秒。
  - F9SQECC3GJ1J8KJBR8694VSOGP：capability 1.818 秒，总 execution 30.583 秒。
  - 均脱离设备初始化死锁并进入编译；不是测试 Pass。随后暴露环境缺失
    `autofuse.compile_adapter`，记录为实际编译异常，未在本次改动中修复该依赖。

## 部署

- 基于 upstream/refactor 的 5ca3b75（包含 dtype-aware tolerance 修复）。
- mlir 候选/安装/验证目录：`/mnt/workspace/af-fuzz/log-drain-1111/`；
  私有 SQLite 一致性备份、旧入口与 campaign 配置在 `private/`。
- 原 campaign 仅暂停 feeder，等待已有任务自然完成后切换正式入口
  `/mnt/workspace/af-fuzz/refactor-d0-Nh5vfp/bin/autofusedbg-npu`。
  原 home、身份、结果、ACK、预算、batch 配置与双卡设置保留，不重试旧失败结果。
- oracle1 仅叠加 core_logs.py / scheduling/pool.py，保留线上其他已部署变更。
  私有备份 `/home/ubuntu/af-fuzz/private/log-drain-1111/`；服务重启后 active，summary API 200。
