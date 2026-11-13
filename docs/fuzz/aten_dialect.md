# NNSmith ATen 四类算子扩展（I/J）

## 范围

本轮只扩展逐元素、比较、归约和 view，不扩展矩阵乘、卷积。新随机 campaign
使用 I（静态）/J（动态）seed；C–H 继续按原协议重放，已有 campaign 不自动换目录。

| 类别 | 精确 ATen overload 数 | 示例 |
| --- | ---: | --- |
| 逐元素及比较 | 256 | Tensor/Scalar 算术、逻辑/位运算、比较、where、激活、special 函数、函数式 backward 公式 |
| Reduce 及相关操作 | 58 | sum/mean/prod、min/max/arg、all/any、std/var、norm、median、topk、累计运算、softmax |
| View 及形状操作 | 54 | reshape/view、transpose/permute、squeeze、expand、slice、split/chunk/unbind、diagonal、unfold、复数视图 |
| 合计 | **368** | 按 overload 计数，不是 368 个不同算子名 |

精确目录是 `autofusedebugger/nnsmith_integration/aten_catalog.py` 的 `RULES`。
`aten_dialect.py` 为每条规则提供 NNSmith `AbsOpBase` 子类、dtype/shape 传递、
Z3 约束及 materializer。仍由 `nnsmith.graph_gen.model_gen` 构图，不是绕过
NNSmith 的独立随机脚本，也不修改已安装的 NNSmith 文件或它的旧 Torch registry。
执行目标是目录内的真实 `torch.ops.aten.<name>.<overload>`。

这里的“覆盖”指已建模 overload，不代表所有 dtype、rank、参数组合或后端均支持。
当前 rank 上限为 5，标量和轴参数采用确定的有界选择；一些形状变换也有明确约束，
例如 split/chunk 建模两个等长输出，unbind 建模长度为 2 的轴，topk 固定 k=1。
多输出逐项保存和绑定，下游节点可以使用第二个返回值。

### 尚未覆盖

固定 Torch 环境的 `torch.Tag.pointwise` 审计中，函数式 tensor 输出的逐元素
schema 已全部进入目录；`aten.equal.default` 返回 Python bool，不作为 Tensor 边建模。
这不是声称所有 ATen 或所有 reduce/view overload 已补齐。

- 原地及 `out=` 变体不纳入本轮，避免引入尚未实现的写别名语义。
- `as_strided`/`as_strided_scatter` 等需要额外 storage/alias 模型，未启用。
- quantile、unique、sort/argsort 等及其他未建模 overload 仍需补充规则。
- `inventory()` 对目录涉及的 packets 及明确列出的缺口输出 schema/状态，
  不是整个 ATen 命名空间的完整分类器。
- 某些 half/bfloat16 special 函数没有 CPU 实现，仍保留在生成目录中，执行时跳过；
  不按设备能力删目录，也不替换成另一张图。

## Seed 与筛选

I/J 校验固定 NNSmith 源码摘要、Z3 4.12.2、Torch 2.10.0+cpu 及默认 float32；
不匹配时拒绝启动，不自动缩小算子目录。该 Torch 版本字符串来自 mlir 现有运行环境，
不表示没有加载 NPU 扩展。它不是跨任意 Torch 构建或任意硬件的兼容承诺。
同 seed 在相同协议和固定依赖下重放同图及输入；跨架构 Z3 确定性仍需专门验证。
生成超时或求解失败直接跳过，不返回受机器速度影响的部分图，不换 seed 重试。

`--ops` 在完整图物化后过滤，不影响随机流：

```sh
# 图中所有节点都必须在白名单内；add 接受该 packet 的所有已建模 overload
# 使用完整 target 可以仅允许一个 overload
autofusedbg fuzz run --ops add,ge,sum,reshape --workers 4 --compile-threads 1 --save
autofusedbg fuzz run --ops aten.add.Tensor,aten.ge.Tensor --count 10 --save
# 保留原有排除入口；按 transpose packet 过滤
autofusedbg fuzz run --exclude-op transpose --save
```

白名单很窄时拒绝率可能很高，`--count` 也不承诺凑满这么多个有效图。
白名单与排除参数同用的既有优先规则没有在本轮改动。

## 无效用例与执行模型

保留长期复用 worker、动态设备租约和单次真实编译。没有引入一 case 一进程，
也没有增加“先探测、再完整编译”。生成失败、CPU 不可执行及无 Autofuse 覆盖
沿用 case 级跳过：不写 attempts、不进入有效用例统计，只写有界 skip audit。
已进入 Autofuse 后的编译错误以及 CPU 对照下的 NPU 精度错误仍是失败，不吞掉。

无覆盖依据是实际 `codegen_kernel_def` 事件和 Scheduler 正常结束，不是原始算子白名单。
`AF_CANDIDATE` 只证明到达 Autofuse 编译边界；Scheduler 的 enter/ok 不冒充
Autofuser.schedule 的 enter/ok。`AF_CODEGEN_OK` 表示原 codegen 入口正常返回。

## mlir 验证记录

- 每个 overload 都进行 CPU shape/dtype 建模对照及序列化后 materializer 重放；
  遍历声明的 dtype 签名及标量、二维、三维样本。CPU 明确未实现的低精度 special
  组合允许跳过，但每个 overload 至少必须有成功样本。这不是全部组合的穷举证明。
- 64 个随机组合图：55 个 CPU 成功、7 个生成跳过、2 个 CPU 未实现
  （special_log_ndtr 的 BF16），触及 75 个不同 overload。
- 四 worker、双 NPU AF-MLIR：ReLU 与 ge.Tensor 编译/执行/CPU 对照通过；
  transpose.int 与 prod.dim_int 在当前后端路径没有 Autofuse 覆盖，按预期熔断。
  数据库仅保留有效两例的 CPU/AF 共 4 条成功记录，另有 2 条 case skip audit。
  本次小批次耗时 41.83 秒，不作为吞吐无回退证明。
- 两个独立进程分别设置 PYTHONHASHSEED=1/9127，8 个 I/J seed 的 workload_id 一致；
  另验证旧 G seed 指纹不变。未据此宣称跨架构已验证。

远端日志：`/tmp/aten-graphs-final.out`、`/tmp/aten-npu-final.out`、
`/tmp/aten-hash-{1,9127}.out`。这些是本轮临时验证产物，不是生产部署记录。

最终同一份源码在 mlir 的专项及集成回归：**510 passed、6 skipped，198.05 秒**。
6 项是远端 Node.js 10 无法运行的浏览器 DAG 测试（需要 Node.js 16+），未计为通过；
另有 pygraphviz 可选可视化依赖警告。日志 `/tmp/aten-complete-final.out`。
覆盖逐 overload、I/J 静动态重放、多输出下游绑定、落库分类、旧 seed 指纹、
CLI/campaign、CPU/no-AF 跳过及四 worker 调度。测试适配修正了旧 campaign 元组
长度和 CPU-only 记录查询入口；没有放宽精度断言。
