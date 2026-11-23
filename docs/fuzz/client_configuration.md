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

`config edit` 读取已有 JSON/YAML 任务文件，不会打开编辑器。任务规格见[进阶命令](fuzz_usage.md)。本地模式写入本机配置，须设置 `results.upload: false`；带 token 时保存平台个人配置。

平台配置接口：GET/POST /api/config/me；管理员默认配置 GET/POST /api/config/default。POST 的 config 为 null 表示恢复继承。个人配置为完整快照；没有个人配置时继承管理员默认配置。配置页面位于客户端管理页面。保存配置在下次 start 生效，运行中任务继续使用原快照。带 token 时 `afx config reset` 清除个人配置；本地 `reset` 使用通用默认值，其中上传为开启，启动前需用 `config edit` 关闭上传。

数据目录可用 `AUTOFUSEX_HOME` 或命令末尾的 `--data-dir DIR` 指定，以 `init` 返回的 `home` 为准。指定独立目录时，每条命令都需保持一致，例如 `afx init --data-dir /path/to/afx-client`、`afx start --data-dir /path/to/afx-client`。托管任务结果分别保存在 `runs/RUN_ID/results.sqlite`，可用 `afx task list`、`afx task results RUN_ID` 查询。

资源由客户端根据设备数量、有效 CPU 和内存计算，仍受本机资源授权上限约束。本地模式关闭上传，带 token 时以上述平台配置为准。

## Replay

    afx replay SEED [SEED ...]

重放从本地已保存的 run 历史恢复执行配置，无需额外配置文件。请求里的所有 seed 都必须有历史记录。当前运行环境或设备与历史不兼容时明确报错。

运行中请求以优先级 100 原子追加到持久化队列；普通生成优先级为 0。同优先级保持 FIFO。调度器先派发重放用例和重放阶段，不中断已经执行的编译/kernel。重放不受当前随机生成算子过滤条件影响。返回 task_ids 可用于查询是否实际完成。

无运行任务时，恢复历史配置创建独立 replay run；多个请求按队列顺序执行。daemon 未运行时命令自动启动它。

## 更新

平台管理员继续使用客户端管理页面下发目标 afx 版本。bootstrap daemon 只负责接收升级、停止 worker、安装到独立 client-releases 目录并切换 active-client.json；平台 heartbeat、配置同步和硬件/软件事实采集由当前 active client release 的 platform agent 子进程执行。上一版本指针保存在 previous-client.json。连续任务在升级后重新提交。这样新增 heartbeat 字段只需发布并切换 client release，不依赖 bootstrap daemon 加载新模块。

每次管理员下发升级都有独立请求编号；同一次请求的重复心跳不会重复安装，重新下发相同版本会创建新操作。心跳上报 active-client.json 指向的执行端版本，服务端收到目标版本后清除升级目标。新初始化的数据目录生成独立持久化身份，已有 config.json 身份保持不变。旧 daemon 若不支持 active client agent，需要先手工升级并重启一次；支持该协议后，后续 heartbeat 和平台事实采集会随 active release 自动切换。

### 配置查看、校验与导出

所有 config 命令操作同一种任务配置，不再提供独立的 policy 命令。

- afx config（或 config show）：返回 config 和 source。本地读取任务配置，托管模式读取平台配置。
- afx config edit FILE：应用 JSON/YAML 任务配置，支持 - 从 stdin 读取。
- afx config validate FILE：按 edit 的规则校验，不提交修改。
- afx config reset：本地恢复默认并关闭上传；托管模式恢复平台继承配置。
- afx config export：stdout 输出纯任务 JSON，可保存后交给 edit 或 validate。

旧 config effective/apply、management trust/status 和 trust/effective 本地 IPC 接口已移除。底层资源约束仍保留供任务执行与高级安装使用。
