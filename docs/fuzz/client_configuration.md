# 客户端配置与重放

本地初始化：

    afx init
    afx config show
    afx config edit task.yaml
    afx start
    afx status
    afx stop

带 token 初始化使用平台配置：

    afx init --token TOKEN --endpoint https://autofuse.online

不带 token 的客户端不进行心跳、配置同步或结果上传。配置和结果位于客户端数据目录。测试请始终指定独立的 --data-dir；此选项不会修改全局默认目录指针。

平台配置接口：GET/POST /api/config/me；管理员默认配置 GET/POST /api/config/default。POST 的 config 为 null 表示恢复继承。个人配置为完整快照；没有个人配置时继承管理员默认配置。配置页面位于客户端管理页面。保存配置在下次 start 生效，运行中任务继续使用原快照。

资源由客户端根据设备数量、有效 CPU 和内存计算，仍受本机资源授权上限约束。性能和上传默认关闭。

## Replay

    afx replay SEED [SEED ...]

重放从本地已保存的 run 历史恢复执行配置，无需额外配置文件。请求里的所有 seed 都必须有历史记录。当前运行环境或设备与历史不兼容时明确报错。

运行中请求以优先级 100 原子追加到持久化队列；普通生成优先级为 0。同优先级保持 FIFO。调度器先派发重放用例和重放阶段，不中断已经执行的编译/kernel。重放不受当前随机生成算子过滤条件影响。返回 task_ids 可用于查询是否实际完成。

无运行任务时，恢复历史配置创建独立 replay run；多个请求按队列顺序执行。daemon 未运行时命令自动启动它。

## 更新

平台管理员继续使用客户端管理页面下发目标 afx 版本。daemon 心跳接收后执行 client.update，安装到独立 client-releases 目录，校验版本后切换执行端。上一版本指针保存在 previous-client.json。连续任务在升级后重新提交。该机制更新执行端，bootstrap daemon 本身的签名升级仍沿用 tool.stage/tool.activate。

每次管理员下发升级都有独立请求编号；同一次请求的重复心跳不会重复安装，重新下发相同版本会创建新操作。心跳上报 active-client.json 指向的执行端版本，服务端收到目标版本后清除升级目标。新初始化的数据目录生成独立持久化身份，已有 config.json 身份保持不变。旧 daemon 需要更新并重启才能采用上述协议；仅切换执行端不会更新 daemon。

## 升级修复验证（2026-11-20，America/Phoenix）

- 本地平台及相关测试 81 项通过；mlir 独立候选 daemon 回归测试 2 项通过。
- afx 服务端完成私有一致性备份后部署候选包；真实 HTTP 下发，mlir 独立客户端完成 0.2.0 → 0.2.1 → 0.2.0 → 0.2.1，每次均由真实心跳确认版本并清除目标。
- 使用当前源已发布的旧版本验证切换；候选 bootstrap 为 0.2.2。未覆盖生产批量升级或运行中连续任务恢复。
- 原 mlir daemon 和任务保持运行，未替换；本次使用独立 bootstrap 验证客户端修复。

## 本次验证记录（2026-11-19）

- 本地：management + platform ops 共 80 项通过；JavaScript 语法和 diff 空白检查通过。
- afx：独立目录 /tmp/afx-refactor-priority-test/final，管理/API 共 80 项通过。
- mlir：独立目录 /tmp/afx-refactor-priority-test；加载 ~/env.sh，独立 venv 引用现有 Torch/CANN 运行时；两张卡健康正常。
- mlir：调度器与新命令测试 37 项通过；真实 CPU + AF-MLIR 运行完成 28 条执行记录、0 失败，14 条跳过；运行时插入的高优先级 replay 两个后端完成。
- mlir：停止后重新启动 daemon，按历史配置独立 replay，CPU 与 AF-MLIR 均为 ok。
- afx：独立 upgrade-client 目录安装 PyPI 已发布版本 0.2.1，版本校验和导入成功。0.2.1.1119 未发布，安装拒绝且未切换版本。
- mlir 系统 SQLite 缺少 trigram tokenizer，平台服务测试在该运行时不能初始化；平台 API 已在 afx 验证。

以上未部署生产服务。浏览器交互尚未做视觉验收；升级测试验证执行端安装切换，未验证生产批量升级。
