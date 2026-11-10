# Fuzz CLI（精简入口，实施状态）

普通入口：

```text
autofusedbg fuzz setup --token TOKEN [--sync-interval 30]
autofusedbg fuzz run [SEED ...] [--count N] [--backend NAME] [--foreground]
autofusedbg fuzz stop
autofusedbg fuzz status
autofusedbg fuzz replay SEED [--backend NAME]
autofusedbg fuzz view RESULTS.sqlite
```

- setup默认连接https://autofuse.online，用token向云端获取管理员登记的用户名，不允许客户端指定或冒充姓名。用户不再指定UUID或密钥文件。
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
fuzz admin user DATABASE NAME
fuzz admin rotate-token DATABASE NAME
fuzz admin serve DATABASE [--port 8090]
fuzz admin backup DATABASE DESTINATION
```

管理员命令在终端仅输出一次token（请私下交付，勿贴公开日志）；云端只存hash。轮换按姓名查询，不要求用户知道UUID。

AF-MLIR运行时必须实际加载带RunMlirCodegen的库，并使用严格MLIR模式；默认CANN库没有该能力时明确拒绝，不静默回退legacy。当前验证范围仅冻结Elementwise/Reduce/Transpose图，不代表任意图能力。错误日志与精度指标通过SQLite同步至详情页；worker异常退出/超时也保存失败摘要和已捕获的日志尾部。父进程本身被强杀时只能通过任务恢复发现中断，不能保证抓到最后日志。真实编译/kernel性能仍不能可靠拆分，保持为空。部署新版需使用`fuzz admin serve`。

### 网站查询

总览和测试记录共用搜索条件。`pass`、`Failed`、`功能Failed`、`精度Failed`、`Unknown`按当前shape用例结果过滤（CPU参考成功不会把目标后端失败用例变成Pass）。其他文字按用户、当前shape输入/输出、OP、后端、版本和错误摘要模糊匹配。

同一输入框接受只读SQL条件子集，例如：

```sql
WHERE status = 'Pass' AND backend = 'af-mlir'
WHERE user LIKE '%niuyuhu%' AND op LIKE '%transpose%'
WHERE backend = 'af-mlir' AND function_status = 'Failed'
WHERE seed = 'BD0MYWOXCJ' AND shape_index = 1
```

字段在页面“SQL条件与字段”列出。`status`是shape用例综合状态；`function_status`和`accuracy_status`属于指定backend。同一shape仍并排显示CPU及其他后端，不会隐藏必要的对照。只允许字段比较、LIKE、AND/OR、括号；无任意SELECT/写入/函数/子查询权限。`mode`取static或dynamic。现有耗时是当前shape输入绑定/首次调用/精度校验之和（秒，1位小数）；首次调用包含惰性编译，不能用于比较稳态kernel性能。
