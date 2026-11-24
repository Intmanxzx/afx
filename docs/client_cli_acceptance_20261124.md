# 0.2.5 客户端 CLI 逐项验收

环境：npussh mlir，Python 3.11，CANN 9.3。使用安装后的候选 wheel，隔离目录 /tmp/afx-cli-acceptance-025；生产 yuhu 的客户端目录与任务不作为启停测试对象。命令输入、返回码和输出保存在该目录 results.json，阶段脚本 /tmp/afx-cli-acceptance.py、/tmp/afx-cli-phase2.py 至 phase12.py。

以下区分真实业务执行、隔离协议/产物测试以及环境限制；帮助可加载不等于功能通过。测试脚本最初使用过非法设备值、termination 和资源策略，以及错误参数形式，这些输入被正确拒绝；改为合法配置后复验。task wait 对 stopped/cancelled 返回非零属于现有明确语义。

| 命令 | 验收与结果 |
|---|---|
| --version、各组 --help | 实际安装包可调用；25 个初始公开入口加载正常 |
| init | 本地初始化成功，重复初始化拒绝 |
| start / status / stop / restart | 隔离客户端实际启停、重启；两个 CPU 用例完成，状态和结果持久化 |
| config show / export / validate / edit / reset | 完整读写、导出、校验、恢复默认并再次应用；非法配置拒绝 |
| env list / inspect / status | 注册环境列表、环境检测、按名称读取通过 |
| env prepare / apply | 真实创建计划、异步执行、创建独立 venv 和版本校验通过；测试兼容配置，不是重装 CANN |
| task validate / submit | 实际 CLI/socket 提交与等待操作完成通过 |
| task list / inspect / status / wait / results | 已完成任务与结果可查询；停止/取消终态正确返回非零 |
| task pause / resume / stop / force-stop | 连续 CPU 任务实际控制；正常停止及取消终态确认 |
| task reconfigure | 连续任务在安全点切换，spec epoch=1；连续转有限配置被拒绝 |
| task replay / replay | 使用已完成任务历史配置提交重放；返回独立操作/任务 |
| operation inspect / wait / logs / cancel | 实际操作查询、等待、日志、取消请求执行通过 |
| build submit / inspect | 真实执行本地授权测试构建脚本，隔离快照与产物摘要生成成功；不是完整 CANN 构建验收 |
| update | 公开 update 0.2.5 --wait 使用本地 wheel 索引完成安装、supervisor 切换；另有生产 0.2.4.1→0.2.5 试点证据 |
| release stage / list / inspect / activate | 临时 Ed25519 签名及完整依赖 wheels，摘要缓存命中，真实 venv/pip 安装与激活，tool_switch=active；不包含外网下载测试 |
| case import / list / inspect / run | 注册不可变 Python 用例，npu:0 实际运行完成 |
| fuzz run / status / stop | CPU 前台有限任务真实执行、保存及查询通过，停止调用通过 |
| fuzz replay | CPU 实际重放并保存结果通过 |
| fuzz resume | 已完成任务的幂等恢复路径通过；中断恢复另有管理回归测试 |
| fuzz seed / diagnose | 实际物化 seed；相同 trace 比较无差异 |
| results list / check / compare | 使用本次真实生成的结果库，查询、完整性检查、同一 attempt 比较通过 |
| results snapshot / restore / merge / rebuild-index | 实际新建快照、恢复、合并和索引重建通过 |
| analyze manifest / compare / compare-three | 构造含真实 npy 数据的图，四种 side 清单及双向/三向比较通过 |
| analyze compare-npy / report | 实际比较 numpy 数组并生成 JSON、HTML 通过 |
| debug run | normal/cpu 两种模式均执行命令并保存输出 |
| debug camodel info / run | 当前 CANN 无模拟器，明确报缺失；人工模拟目录验证发现与环境包装命令成功，未执行真实 CAModel |
| model normalize | 本地 HTTP Agent 测试响应经真实 CLI 解析、校验、写入；不代表外部模型服务质量验证 |
| model run | npu:1 真实执行 PASS 1/1，生成 HTML 和 Autofuse 图；首次在不可写 / 下执行被 Torch 调试目录权限阻止，切换可写目录后通过 |
| view | 实际启动 CLI HTTP 服务，首页 200，api/capabilities 确认为 read_only；未做本轮浏览器交互验收 |
| doctor | 实际环境检测输出通过 |
| sync setup / once / status / run | 独立隔离服务端，上传、心跳、状态、持续轮询及中断退出通过 |
| retest once / run | 空队列调用、持续轮询与中断通过；真实任务领取成功，npu:1 三次执行均遭现有任务占卡，正确退避且未上报无效结果；真实完成回传尚未通过 |

## 发现与修复

device enroll/pair/revoke 使用 /api/manage/*，服务端实现已经删除（client/central.py 末尾明确记录）。本轮移除公开 device 命令和根帮助项，内部迁移兼容代码保留；普通客户端继续使用 init 与平台 token。新增回归确保公开入口拒绝 device pair。

管理、CLI surface、客户端 setup、白名单测试共 130 项通过。首次失败因测试部署漏传 test_platform.py 辅助文件，补齐后相关 3 项全部通过。最终收敛包的 daemon/旧升级协议/隐藏入口检查 5 项通过。

收敛包 SHA256：602a7aac47871702b5e42d8b8e2326266d7311e0144df85abcab2c254964a77d。与生产试点包相比仅删除公开 device 入口；未覆盖安装到生产 yuhu，未发布 PyPI。
