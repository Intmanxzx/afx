# 当前 ATen 算子目录

当前 fuzz 生成器只保留一套 NNSmith + ATen 实现，静态 seed 使用 C2，动态 seed 使用 D2。旧内测 seed 仍可在历史记录中显示，但不支持重放，也不会被解释成当前协议。

## 范围

当前目录覆盖逐元素、比较、归约和 view/shape 四类，共 368 个精确 ATen overload；矩阵乘、卷积和原地/out 变体暂不纳入。

| 类别 | overload 数 | 示例 |
| --- | ---: | --- |
| 逐元素及比较 | 256 | Tensor/Scalar 算术、逻辑/位运算、比较、where、激活和 special 函数 |
| Reduce 及相关操作 | 58 | sum/mean/prod、min/max/arg、all/any、std/var、norm、median、topk、累计运算、softmax |
| View 及形状操作 | 54 | reshape/view、transpose/permute、squeeze、expand、slice、split/chunk/unbind、diagonal、unfold、复数视图 |
| 合计 | **368** | 按 overload 计数，不是 368 个不同的算子名 |

精确目录是 `autofusedebugger/nnsmith_integration/aten_catalog.py` 的 `RULES`。`aten_dialect.py` 为每条规则提供 NNSmith `AbsOpBase` 子类、dtype/shape 传递、Z3 约束及 materializer。图仍由 `nnsmith.graph_gen.model_gen` 构造，执行时调用目录中对应的真实 `torch.ops.aten.<name>.<overload>`。

逐个 overload 的可读清单见 [ATen 算子清单](aten_op_list.md)。

图节点保存完整的 ATen 目标、构造属性、算子对象状态和全部输出。CPU 与目标后端使用同一份 CaseSpec 和输入字节；CPU materializer 无法执行的生成结果直接跳过，不能替换成另一张图。

## 未建模边界

- 原地及 `out=` 变体不纳入本轮，避免引入尚未实现的写别名语义。
- `as_strided`/`as_strided_scatter` 等需要额外 storage/alias 模型，未启用。
- quantile、unique、sort/argsort 等未建模 overload 仍需补充规则。
- `aten.equal.default` 返回 Python bool，不作为 Tensor 图边建模。
- 某些 half/bfloat16 special 函数没有 CPU 实现，执行时跳过；不会按设备能力删目录。

目录扩展必须同步更新固定源码指纹/协议版本，避免旧 seed 在规则改变后静默生成另一张图。

## Seed 与过滤

同一个 C2/D2 seed 在固定 NNSmith 源码、Z3、Torch 版本和 seed 配置下生成同一个图和输入字节。`--ops` 在完整图物化后过滤，不参与随机流；`--exclude-op transpose` 也只过滤已经生成的图。因此过滤参数不会改变同一个 seed 的 workload。

```sh
autofusedbg fuzz run --ops add,ge,sum,reshape --workers 4 --compile-threads 1 --save
autofusedbg fuzz run --ops aten.add.Tensor,aten.ge.Tensor --count 10 --save
autofusedbg fuzz run --exclude-op transpose --save
```

## 跳过与 Autofuse 覆盖

生成失败、CPU eager 无法执行和确认没有 Autofuse 候选的 case 都跳过，不写入有效 attempts，也不计入有效用例分母。已经进入 Autofuse 后的 lowering、schedule、codegen 或 CPU 对照失败仍保留为失败。

无 Autofuse 覆盖的判定来自运行时 `codegen_kernel_def` 候选事件以及 Scheduler 正常完成后的汇总，不来自算子白名单。探针在长期 worker 内临时安装，不采用一 case 一进程，也不进行“先探测再完整编译”。
