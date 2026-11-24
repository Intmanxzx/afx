# Seed 身份与分类边界核查（2026-11-23）

## 环境无关约束

完整 seed 决定用例图、算子属性、shape、dtype、输入与权重。A2/A3/A5、CANN、Python、Torch 等执行环境版本不得改变用例或其身份。不支持执行时报告不支持，不能通过换算子、过滤候选或改输入产生另一个用例。实际环境信息归执行记录。

## Golden 失败的直接原因

81f2e5e 保存 golden 时默认 value_dist_mode=uniform；92d5b23 改为 log_scale。测试调用 SamplingConfig.profile 而未指定该字段，实际 seed 已变化。此前将此失败描述为“相同 seed 指纹漂移”不准确。

在 mlir 同一运行环境下，分别执行 81f2e5e 与当前源码：四个完整历史 seed 的完整 content_manifest（规格、图、存储长度及摘要）一致，全部旧 golden 可复现。当前默认生成的四个 seed 也在清理 plan 缓存、两个独立进程（PYTHONHASHSEED=1/73）中一致。golden 测试现固定完整 seed，未替换参考摘要。

## 独立发现的 Torch 元数据耦合

ca7b5ce 将 graph_json.torch 从固定字符串改为实际 torch.__version__；graph_json 参与 workload_id，导致同一用例在不同 Torch 版本字符串下可能得到不同身份。本次恢复为 E/F 历史固定 wire 标记 2.10.0+cpu，保持既有 golden 及 CPU 标记用例身份。该字段不再描述实际运行环境；实际版本由 execution.environment 记录。不能直接删除该字段，否则所有历史摘要都会改变。

此前以其他实际 Torch 版本写入的图仍可解码；其已存摘要不重写。它们与恢复后的规范生成摘要可能不同，跨旧版本历史匹配应使用已存 CaseSpec，不应仅凭 seed 重新生成后假定旧摘要相等。后续协议升级可移除这个历史字段，并显式处理旧身份映射。

生成依赖的 NNSmith/Z3 约束属于 seed 算法实现约束，不能随执行环境更换；目前生成器仍需要安装相应依赖及 Torch 算子定义。测试不等于证明任意 Python/Torch 版本或所有 NPU 平台均可执行。

## 分类尚未统一

- 本地查询及 viewer：execution/reporting/failure_origin.py，VERSION=8。
- 上传转换：client/platform_transport.function_status，复用本地规则并有额外超时、schedule 等映射。
- 中心入库：server/failure_attribution.py，VERSION=1；server/ingest.py 会依据上传证据重新计算 function_status/fail_stage/failure_owner。

共享静态资源不等于分类规则统一。未来应抽出公共纯函数，统一输入证据和输出协议；本地与中心调用同一规则，中心保持最终裁定。现有中心按入库结果持久化，升级分类逻辑还需要历史重分类及聚合重建；本地缓存已包含分类版本。此次未调整分类规则。

## 验证

mlir 当前协议测试 24 passed；新增环境元数据扰动测试覆盖完整 manifest，清理生成缓存后修改 Torch 版本字符串及 CANN 路径。A2/A3/A5 字符串只用作环境扰动，未宣称三种硬件实测。
