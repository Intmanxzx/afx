# 客户端配置与重放

## 快速开始

在 NPU 机器的当前 shell 中加载匹配的 PyTorch、torch_npu、CANN 和 Autofuse 环境，然后执行：

```bash
python -m pip install 'autofusex[execution]'
afx init
afx start
```

`init` 只需一次；`start` 自动启动 daemon 并登记当前 Python 环境，无需手工编写 trust 或 environment 文件。初始化不会安装 Torch/CANN，也不代表已通过完整 NPU 执行验证。默认持续生成、使用全部可见 NPU、自动计算资源并保存结果，性能测量关闭。

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

托管模式的设备和其他任务参数以平台拉取的配置为准。修改个人配置后，从页面触发 restart，客户端会停止旧任务，再按新配置启动；无需另行 stop/start。本地模式可以在首次初始化时使用 `--device npu:0`，多卡可重复指定，后续通过 `afx config edit` 修改任务配置。以上初始化方式按需选一个；已初始化的目录再次 `init` 会报错，不会覆盖身份或切换模式。

不带 token 的客户端不进行心跳、配置同步或结果上传。配置和结果位于客户端数据目录。测试请始终指定独立的 --data-dir；此选项不会修改全局默认目录指针。

## 配置与数据目录

```bash
afx config show
afx config edit task.yaml
```

`config edit` 读取已有 JSON/YAML 任务文件，不会打开编辑器。任务规格见[进阶命令](advanced.md)。本地模式写入本机配置，须设置 `results.upload: false`；带 token 时保存平台个人配置。

平台配置接口：GET/POST /api/config/me；管理员默认配置 GET/POST /api/config/default。POST 的 config 为 null 表示恢复继承。个人配置为完整快照；没有个人配置时继承管理员默认配置。配置页面位于客户端管理页面。保存配置在下次 start 生效，运行中任务继续使用原快照。带 token 时 `afx config reset` 清除个人配置；本地 `reset` 恢复默认任务配置并关闭上传。

数据目录可用 `AUTOFUSEX_HOME` 或命令末尾的 `--data-dir DIR` 指定，以 `init` 返回的 `home` 为准。指定独立目录时，每条命令都需保持一致，例如 `afx init --data-dir /path/to/afx-client`、`afx start --data-dir /path/to/afx-client`。托管任务结果分别保存在 `runs/RUN_ID/results.sqlite`，可用 `afx task list`、`afx task results RUN_ID` 查询。

自动资源策略由客户端根据设备数量、有效 CPU 和内存计算；手动策略使用任务配置中的 workers、compile_threads 和 memory_budget。普通任务与复测不创建或校验 trust.json，也不受旧设备、资源授权限制；运行时仍检查资源可用性并使用设备锁。本地模式关闭上传，带 token 时以上述平台配置为准。

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
