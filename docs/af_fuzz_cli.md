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
- stop不传任务ID取消全部待执行任务并请求停止运行任务；高级单任务控制仅保留内部兼容参数。
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

本阶段尚未完成（不能按目标描述成已实现）：默认无限run、CPU+af-mlir强制配对、eager-npu二级精度判定、误差统计/核心日志展示、100条AF MLIR真实采集。当前run仍为有限队列；不传seed/count仍处理现有队列。AF MLIR继续fail closed直到验证已安装二进制真实能力。老版本公网服务命令仍保持原样，部署新版时必须同步切换成fuzz admin serve。
