# Fuzz CLI（精简入口，实施状态）

普通入口：

```text
autofusedbg fuzz setup --user NAME --token TOKEN [--sync-interval 30]
autofusedbg fuzz run [SEED ...] [--count N] [--backend NAME] [--foreground]
autofusedbg fuzz stop
autofusedbg fuzz status
autofusedbg fuzz replay SEED [--backend NAME]
autofusedbg fuzz view RESULTS.sqlite
```

- setup默认连接https://autofuse.online，核对管理员登记姓名与token，不接受用姓名冒充另一身份。用户不再指定UUID或密钥文件。
- token作为命令参数可能留在shell历史或短暂可见于进程列表，请使用安全终端。无人值守同步需要持久凭据，工具内部以0600私有文件保存；不输出到日志/结果/返回JSON。这不等于不落盘。
- 资源默认按有效CPU/cgroup/亲和性/内存/磁盘余量保守计算。目前调度器串行，cpu_workers=1；compile_threads=min(8,有效核数的一半)，内存预算50%，缓存预算最多空闲磁盘20%且上限20GiB。不是未经基准测试宣称全局最优。
- run默认持续运行，直到`fuzz stop`。`--count 100`表示100个case，每个都执行CPU和AF-MLIR；`--backend`只增加其他后端，不能移除必选项。
- stop关闭持续生成器，取消全部待执行任务并请求停止运行任务；高级单任务控制仅保留内部参数。
- replay默认执行CPU和AF-MLIR；显式`--backend`用于已知故障的单后端诊断。所有后端使用同一执行流程。
- AF-MLIR与实际eager CPU输出不一致时，自动独立执行eager-npu复核；与eager-npu一致可判Pass，但保留CPU分歧与误差。编译/运行错误不会通过这一规则被掩盖。
- replay和run隐藏底层queue/database/data-dir/compile-threads参数，普通帮助不展示。工作目录可用AUTOFUSEDBG_HOME隔离。

低频维护：

```text
fuzz maintenance doctor
fuzz maintenance seed SEED [--show-spec]
fuzz maintenance diagnose EXPECTED ACTUAL
fuzz maintenance compare DATABASE LEFT RIGHT
fuzz maintenance results DATABASE
fuzz maintenance snapshot SOURCE DESTINATION
fuzz maintenance restore SOURCE DESTINATION
fuzz maintenance import SOURCE DESTINATION
fuzz maintenance check-db DATABASE
fuzz maintenance index-results DATABASE
fuzz maintenance upgrade-queue QUEUE BACKUP
fuzz maintenance resume TASK
fuzz maintenance sync ACTION --data-dir DIR
```

离线管理员：

```text
fuzz admin user DATABASE NAME KEY_OUTPUT
fuzz admin rotate-token DATABASE NAME KEY_OUTPUT
fuzz admin serve DATABASE [--port 8090]
fuzz admin backup DATABASE DESTINATION
```

管理员导出token的文件是离线分发产物，不是用户setup参数。轮换按姓名查询，不要求用户知道UUID。

AF-MLIR运行时必须实际加载带RunMlirCodegen的库，并使用严格MLIR模式；默认CANN库没有该能力时明确拒绝，不静默回退legacy。当前验证范围仅冻结Elementwise/Reduce/Transpose图，不代表任意图能力。错误日志与精度指标通过SQLite同步至详情页；worker异常退出/超时也保存失败摘要和已捕获的日志尾部。父进程本身被强杀时只能通过任务恢复发现中断，不能保证抓到最后日志。真实编译/kernel性能仍不能可靠拆分，保持为空。部署新版需使用`fuzz admin serve`。
