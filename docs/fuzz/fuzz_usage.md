# Fuzz 使用说明

本地 Agent 无需安装 afx，优先使用[中心六个 API](agent_api.md#本地-agent-使用中心-api)。以下管理命令在 NPU 客户端所在机器执行，默认输出 JSON；临时运行和重放使用 afx fuzz。两套入口的默认值不同，不能混用任务 ID 或状态协议。完整接口见 [Agent 与 API](agent_api.md)。

## 安装与准备

安装包含当前功能的 wheel；执行端需匹配的 Torch、torch_npu、CANN 和固定版本的 `nnsmith-autofusex`（由 pip 从 PyPI 安装）。已有 NPU 环境按[部署说明](../../deploy/fuzz/README.md)安装；中心和本地查看器只需基础 wheel。源码安装执行依赖：

~~~sh
python -m pip install -e '.[execution]'
~~~

数据目录用 AUTOFUSEX_HOME 或命令末尾的 --data-dir DIR 指定。管理入口需要 daemon 已启动、执行环境已登记为 ready、设备已获本地授权；首次准备见[运维说明](operations.md)。

## Agent：提交、查询、停止

先查询可用环境，将其唯一 ID 或名称填入 task.yaml 的 environment：

~~~sh
afx daemon start
afx environment list
~~~

~~~yaml
schema_version: 1
kind: fuzz_task
environment: npu-env
workload:
  mode: generate
termination:
  mode: finite
  max_cases: 100
results:
  save: true
  upload: true
~~~

upload 为 true 要求设备已绑定；仅本地保存可设为 false。省略 execution 时使用 CPU + AF-MLIR、npu:0 和 staged 调度。

~~~sh
afx task submit task.yaml --request-key campaign-001 --wait
afx task wait RUN_ID
afx task results RUN_ID
afx task stop RUN_ID --wait --request-key stop-campaign-001
~~~

提交成功从 data.result.run_id 取得 RUN_ID。submit --wait 等待提交操作，task wait 等待运行结束；超时不取消任务，可用同一 ID 继续等待。完成返回退出码 0，失败、停止、取消、中断或需要处理返回 1。completed 表示调度结束，模型是否通过要查询 results；结果以 attempt 为单位，包含 seed_short 和状态，支持 --status fail、--status error、--after 游标和 --attempt ID 详情。

重放结果中的 seed，沿用原任务环境、后端、资源和评估配置：

~~~sh
afx task replay SEED --from-run RUN_ID --wait --request-key replay-001
~~~

返回新 run_id，再用 task wait / results 查询。原运行必须已释放任务所有权。重放保留原保存/上传设置，不覆盖旧记录。FILE 参数统一支持 - 从 stdin 读取；可先执行 afx task validate task.yaml --wait 验证环境与规格。重试同一操作沿用 request key，新操作使用新 key。

## 临时运行与重放

在已准备好的执行环境内运行：

~~~sh
afx fuzz run --count 100 --save
afx fuzz status
afx fuzz stop
afx fuzz replay SEED --save
~~~

将 SEED 替换为结果中的当前 E/F seed。直接 run 不传 count 会持续生成，--fg 前台运行，--save 才归档结果；已配置同步时保存结果进入上传队列。首次上传配置为 afx fuzz setup --token TOKEN，TOKEN 由管理员发放；托管设备绑定与此配置是不同入口。

| 目的 | 命令 |
|---|---|
| 单后端重放 | afx fuzz replay SEED --backend af-mlir --save |
| 人工 eager-npu 复核 | afx fuzz replay SEED --backend eager-npu --save |
| 过滤算子 | afx fuzz run --ops add,ge,sum,reshape --save |
| 双卡指定资源 | afx fuzz run --device npu:0 --device npu:1 --workers 4 --compile-threads 1 --save |
| 查看同步 | afx fuzz maintenance sync status |
| 检查环境 | afx fuzz maintenance doctor |

run --backend 增加被测后端，replay --backend 只执行指定后端。不要与同一 home 的活动托管任务同时运行。

## 查询结果

中心提供[网页](https://autofuse.online)和只读 API：

~~~sh
curl --fail-with-body 'https://autofuse.online/api/cases?limit=10'
curl --fail-with-body 'https://autofuse.online/api/summary'
~~~

本地结果可以导出快照后查看：

~~~sh
afx fuzz maintenance snapshot /path/results.sqlite ./results.snapshot.sqlite
afx fuzz view ./results.snapshot.sqlite --no-browser
~~~

快照用于离线查询，不能替代包含身份和 ACK 的中心备份。更多参数查对应命令的 --help。
