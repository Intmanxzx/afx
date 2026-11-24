# 客户端配置与重放

## 快速开始

在 NPU 机器的当前 shell 中加载匹配的 PyTorch、torch_npu、CANN 和 Autofuse 环境，然后执行：

```bash
python -m pip install 'autofusex[execution]'
afx init
afx start
```

`init` 只需一次；`start` 自动启动 daemon，登记当前 Python 环境并准备基础资源授权，无需手工编写 trust 或 environment 文件。初始化不会安装 Torch/CANN，也不代表已通过完整 NPU 执行验证。默认持续生成、使用全部可见 NPU、自动计算资源并保存结果，性能测量关闭。

日常控制：

    afx status
    afx stop
    afx start

更换运行环境或安装包后，加载相应环境并执行 `afx restart`，同时重启 daemon 与执行进程。

## 接入平台与设备选择

带 token 初始化使用平台配置：

    afx init --token YOUR_TOKEN --endpoint https://autofuse.online
    afx start

token 由管理员发放，`--endpoint` 默认就是上述地址，可以省略。带 token 时 daemon 自动发送心跳、同步配置，并按 `results.upload` 上传结果，无需另起上传进程。平台默认任务启用保存与上传，个人或管理员配置可覆盖这些选项。

需要限定设备时，在首次初始化命令中增加 `--device npu:0`，多卡可重复指定，如 `afx init --device npu:0 --device npu:1`。显式选择的设备会覆盖平台配置中的设备列表。以上初始化方式按需选一个；已初始化的目录再次 `init` 会报错，不会覆盖身份或切换模式。

不带 token 的客户端不进行心跳、配置同步或结果上传。配置和结果位于客户端数据目录。测试请始终指定独立的 --data-dir；此选项不会修改全局默认目录指针。

## 配置与数据目录

```bash
afx config show
afx config edit task.yaml
```

`config edit` 读取已有 JSON/YAML 任务文件，不会打开编辑器。任务规格见[进阶命令](advanced.md)。本地模式写入本机配置，须设置 `results.upload: false`；带 token 时保存平台个人配置。

平台配置接口：GET/POST /api/config/me；管理员默认配置 GET/POST /api/config/default。POST 的 config 为 null 表示恢复继承。个人配置为完整快照；没有个人配置时继承管理员默认配置。配置页面位于客户端管理页面。保存配置在下次 start 生效，运行中任务继续使用原快照。带 token 时 `afx config reset` 清除个人配置；本地 `reset` 恢复默认任务配置并关闭上传。

数据目录可用 `AUTOFUSEX_HOME` 或命令末尾的 `--data-dir DIR` 指定，以 `init` 返回的 `home` 为准。指定独立目录时，每条命令都需保持一致，例如 `afx init --data-dir /path/to/afx-client`、`afx start --data-dir /path/to/afx-client`。托管任务结果分别保存在 `runs/RUN_ID/results.sqlite`，可用 `afx task list`、`afx task results RUN_ID` 查询。

资源由客户端根据设备数量、有效 CPU 和内存计算，仍受本机资源授权上限约束。本地模式关闭上传，带 token 时以上述平台配置为准。

## Replay

    afx replay SEED [SEED ...]

重放从本地已保存的 run 历史恢复执行配置，无需额外配置文件。请求里的所有 seed 都必须有历史记录。当前运行环境或设备与历史不兼容时明确报错。

运行中请求以优先级 100 原子追加到持久化队列；普通生成优先级为 0。同优先级保持 FIFO。调度器先派发重放用例和重放阶段，不中断已经执行的编译/kernel。重放不受当前随机生成算子过滤条件影响。返回 task_ids 可用于查询是否实际完成。

无运行任务时，恢复历史配置创建独立 replay run；多个请求按队列顺序执行。daemon 未运行时命令自动启动它。

## 更新

客户端从调用 `afx` 的 Python 环境启动。`afx update VERSION` 或管理员下发目标版本后，daemon 排空任务并缓存目标版与回退版 wheel，由 supervisor 在同一 Python 环境安装、重启和验证；失败则使用缓存回退。心跳、配置同步及任务管理由当前 daemon 执行，不依赖独立 platform agent 或活动版本指针。

更新保留身份、配置、结果与队列。完整流程、稳定 supervisor 边界及旧安装接入方式见[客户端安装与升级](upgrade.md)。

## 配置查看、校验与导出

所有 config 命令操作同一种任务配置，不再提供独立的 policy 命令。

- afx config（或 config show）：返回 config 和 source。本地读取任务配置，托管模式读取平台配置。
- afx config edit FILE：应用 JSON/YAML 任务配置，支持 - 从 stdin 读取。
- afx config validate FILE：按 edit 的规则校验，不提交修改。
- afx config reset：本地恢复默认并关闭上传；托管模式恢复平台继承配置。
- afx config export：stdout 输出纯任务 JSON，可保存后交给 edit 或 validate。

本机资源约束由客户端维护，任务执行和高级安装仍须通过资源准入。

## 固定用例与整网任务

导入只登记源码并计算版本，不执行脚本：

    afx case import ./case.py
    afx case import ./model-package --script model.py --entry create_case --workload model
    afx case list
    afx case inspect CASE_KEY
    afx case run CASE_KEY --device npu:0
    afx case run CASE_KEY --submit

CASE_KEY 使用导入输出的 I: 或 M: 标识。默认执行本地登记版本；--submit 提交已配置的中心，由客户端领取执行。--upload 将本地执行的结果同步到中心。已有任务占用时，中心任务排队等待，不抢占正在运行的任务。

脚本参数用 --script-arg 逐项传入；以短横线开头的参数可写为 --script-arg=--batch-size。--fullgraph、--dynamic、--with-cpu、--rtol 和 --atol 控制对应执行选项。工厂入口模型未指定 fullgraph/dynamic 时保留工厂定义。执行规格和结果保存在 runs/RUN_ID；中心“导入与整网”页面提供版本复测与报告下载。

本地包最大 256 MiB；中心可传输源码包当前限制为 512 KiB。报告上传支持不超过 1 MiB 的 HTML、JSON、TXT、CSV、SVG 文件，大型张量产物保留在执行机器。

## 手动开启编译调试 dump

Fuzz 默认关闭 Torch 编译诊断和 Autofuse 编译 debug dump。需要排查编译问题时，给本次启动或重放增加 --dump：

~~~sh
afx start --dump
afx restart --dump
afx replay SEED --dump
~~~

--dump 只覆盖本次任务，不会修改服务端或本地保存的配置。已有任务正在运行时，start 不会切换 dump 模式；使用 restart --dump 启动开启 dump 的新任务。恢复默认关闭可用 afx restart（任务配置未显式开启 dump 时）。已有保存的 campaign 通过 --continue 或升级恢复时沿用其记录的模式。

需要把开关保存在任务规格中，可通过 afx config edit FILE 设置 execution.dump: true；缺省为 false。服务端下发的任务配置和本地任务规格遵守同一字段，修改任务规格在下次启动生效。显式 --dump 优先开启，afx replay 每次须显式传入 --dump，不继承历史任务的 dump 开关。

调试文件保存在客户端数据目录 debug/autofuse_compile/worker-*，受自动清理策略管理。--dump 与 --save（保存结果数据库）独立，关闭 dump 仍保留正常任务日志和已启用保存的执行结果。继承环境中的 TORCH_COMPILE_DEBUG、编译 debug_dir/codegen_compile_debug 和图 dump 环境项由本次开关统一控制。

临时运行也支持 afx fuzz run --dump 和 afx fuzz replay SEED --dump；进阶命令列表见[进阶命令](advanced.md)。本开关用于 Fuzz 调试输出，模型分析命令生成报告所需的图和张量采集按模型分析流程执行。

## 自动清理客户端目录

服务端“测试任务配置 → 客户端自动清理”可分别设置 Debug 和 Runs 保留天数。个人配置覆盖管理员默认配置；没有个人配置时继承管理员默认值。两项默认均为 2 天，设为 0 关闭对应清理，可选范围为 0～3650 天。任务配置对应字段为 cleanup.debug_days 和 cleanup.runs_days。

客户端 daemon 启动后在独立后台线程检查，此后每小时执行一次。服务端配置同步后最多约 10 秒触发一次新检查，无需重启测试任务；离线时沿用本地最近配置，本地缺省值也是 2 天。daemon 未运行时不会自动清理。

Debug 目录按 worker 隔离并持有使用锁，仍运行的 worker 即使其父进程退出也会受到保护。停止使用的 worker 目录内全部文件超过保留期后才删除。旧版没有使用锁的 dump 在客户端任务或独立调度器仍占用队列时保留。清理不跟随符号链接。

Runs 只清理 completed、failed、stopped、cancelled 且结束时间超过保留期的目录；旧版本没有结束时间的任务从首次观察到已结束时开始计时。运行、暂停、待恢复、升级引用、未交付复测和未完成上传的任务均保留。上传游标尚未追上结果，或存在映射/服务端拒绝记录时跳过清理。环境不支持而被明确过滤的记录视为已处理。

清理删除 runs 下的结果数据库、日志和产物，保留 manager.sqlite 中的任务规格、结束摘要、结果条数及 cleanup 标记。清理后无法从该本地目录查看报告，按 seed 自动检索本地结果的 replay 也无法使用该目录；显式使用 task replay --from-run 仍可读取保留的任务规格；已上传的中心数据仍可查询。afx daemon status --json 可查看最近检查的时间、策略、删除数量、回收字节和错误类别，afx task list/inspect 可查询历史摘要。任务结束后上传仍失败的目录可能持续保留，应先解决上传问题。

本功能需要支持 cleanup-v1 的新版客户端；0.3.0 客户端继续收到兼容的任务配置，不会自动执行清理。编译调试 dump 默认关闭，按上面的 --dump 说明手动开启。
