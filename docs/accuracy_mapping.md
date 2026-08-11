# CPU/NPU 算子输出映射策略

本文说明当前 `autofusedebugger` 在逐算子精度比较时，如何把 NPU 输出映射到 CPU 输出，以及出现未映射时应该如何排查。

本文描述的是当前代码行为，不是理想化的编译器 IR 对齐方案。对应实现主要位于：

- `autofusedebugger/torch_model.py`：CPU `torch.export` 输出采集，以及 native Inductor 路径的辅助命名匹配。
- `autofusedebugger/inductor_autofuse.py`：native Inductor + AscendC/Autofuse 路径的 post-fusion 图解析和 CPU origin 匹配。
- `autofusedebugger/npu_examples.py`：旧 GE/Qwen 适配路径中的 compiler provenance 解析。
- `autofusedebugger/graph.py`：把 NPU 图输出属性转换为 manifest 的 `fusions` 映射。
- `autofusedebugger/compare.py`：根据 manifest 执行最终的输出对照。

## 1. 映射的目标

比较的基本单位是一个输出 tensor，而不是一个显示名称：

```text
NPU (op_name, output_index) -> CPU (op_name, output_index)
```

对于融合算子，只比较融合算子的出口输出：

```text
NPU fused_kernel.output.0 -> CPU 融合子图出口单算子.output.0
```

融合子图内部的 `Add`、`Mul`、`Cast` 等节点不会因为出现在 ASCIR 子图中就被单独加入主精度对比结果。

## 2. 当前推荐路径：native Inductor + Autofuse

`run-script` 和 `run-torch-model` 使用以下链路：

```text
CPU:
  torch.export
    -> run_decompositions()
    -> ExportedProgram FX Interpreter
    -> CPU FX/ATen 节点输出 dump

NPU:
  torch.compile(options={"npu_backend": "ascendc"})
    -> TorchInductor
    -> post-fusion IR
    -> AscendC/Autofuse
    -> intermediate hook 输出 dump
```

### 2.1 CPU 节点名从哪里来

CPU 侧首先执行：

```python
exported = torch.export.export(model, inputs, kwargs)
exported = exported.run_decompositions()
```

随后用 FX `Interpreter` 执行 `ExportedProgram.graph_module`。对于每个 `call_function` 节点，工具保存第一个 Tensor 输出，CPU dump 文件名使用 FX 节点名：

```text
cpu_dump/<fx_node_name>.output.0.npy
```

因此 CPU 的候选名通常是 `add`、`add_1`、`_to_copy`、`matmul`、`mm` 等 ATen/ExportedProgram 节点名。这个名字受以下因素影响：

- PyTorch 版本和 `torch.export` 命名规则。
- decomposition 是否把一个高层算子拆成多个 Core ATen 算子。
- 模型是否产生 graph break、多个编译分区或重复调用。
- 一个 FX 节点是否返回多个 Tensor；当前采集器主要保存第一个 Tensor。

### 2.2 NPU 融合节点的出口信息

工具解析 TorchInductor 编译目录中的：

- `ir_post_fusion.txt`：真实的 post-fusion 调度拓扑。
- `fx_graph_readable.py`：输入输出 layout 辅助信息。
- `autofused_*/benchmark.py`：融合 kernel 的 FX 片段和 ASCIR `graph_hint`。

对于每个融合 kernel，`benchmark.py` 中的 `# return %xxx` 被视为融合子图出口，工具记录：

```text
origin_ops  = 融合子图中的原始 FX/ATen 节点列表
exit_name   = # return 对应的 FX 节点名
exit_type   = 出口节点的 ATen 类型
```

例如：

```text
origin_ops = [add, add_1, add_2, ... add_15]
exit_name  = add_15
exit_type  = add
```

这意味着 16 个 `add` 融合成一个 NPU kernel 后，主图只建立：

```text
autofused_add_... -> CPU add_15
```

而不是把 NPU 融合 kernel 映射到 16 个 CPU 节点。

### 2.3 native 路径的候选筛选和排序

实现函数是 `inductor_autofuse._match_cpu_output()`。对每个 NPU 主图节点，目前只尝试其 `output.0`：

1. 找到 NPU intermediate hook 采集的输出文件；没有文件时不进入 CPU 映射，最终状态为 `NPU_OUTPUT_NOT_CAPTURED`。
2. 使用融合出口的 `exit_name/exit_type`；同时从 `benchmark.py` 的 `Original ATen` 注释中提取原始 ATen 类型。未融合或 fallback 节点使用其生成的 kernel/capture 名称和类型。
3. 遍历尚未使用的 CPU 输出，只保留 `element_count` 相同的候选。
4. 候选必须满足下面至少一个条件：
   - CPU 输出名与期望名称相同。
   - CPU 输出名和期望名称规范化后相同。
   - CPU 节点类型和期望类型规范化后相同。
   - CPU 输出名或节点类型与 `Original ATen` provenance 匹配。
5. 按以下顺序选择候选，越靠前优先级越高：

```text
shape 完全相同
Original ATen 名称（含调用序号）相同
Original ATen 类型相同
精确名称相同
规范化名称相同
规范化类型相同
CPU FX 拓扑中的稳定顺序更靠前
```

6. 一个 CPU 输出只能被使用一次，避免多个 NPU 节点都映射到同一个 CPU 节点。
7. 找到候选后，把下面两个属性写到 NPU 图的输出描述中：

```text
_autofusedebugger_cpu_origin_name
_autofusedebugger_cpu_origin_output_index
```

需要特别注意：当前 native 匹配只使用编译器 provenance、名称、类型、shape 和稳定顺序，不读取 CPU/NPU 数值来决定映射。否则真正发生精度错误时，按数值距离选择候选会把失败节点错误映射到数值更接近但拓扑无关的 CPU 节点。该策略仍不是完整的拓扑图同构匹配。

### 2.4 名称规范化规则

native 路径会统一大小写、去掉 `aten.` 前缀和 overload 后缀，并使用以下别名：

| 输入名称 | 规范化名称 |
|---|---|
| `_softmax` | `softmax` |
| `_unsafe_view`、`view` | `reshape` |
| `cat` | `stack` |
| `clone` | `contiguous` |
| `permute`、`t` | `transpose` |
| `reducesum`、`reduce_sum` | `sum` |
| `reducemean`、`reduce_mean` | `mean` |
| `realdiv` | `div` |
| `mm`、`bmm` | `matmul` |
| `addmm` | `linear` |
| `baddbmm` | `add` |
| `masked_fill` | `where` |

名称末尾的数字后缀也会被规范化，例如 `add_3` 的类型匹配部分会按 `add` 处理；但精确名称 `add_3` 仍然优先于类型匹配。

`baddbmm -> add` 用于 Inductor 将 CPU FX 中的 `bmm -> add` 重组为一个 `baddbmm` kernel 的场景。二者的可观测出口等价，因此 NPU `baddbmm` 输出应与 CPU 最后的 `add` 输出比较。

`masked_fill -> where` 对应 `torch.export` 将 `masked_fill(self, mask, value)` 分解为 `where(mask, value, self)` 的场景。NPU fallback 可能仍保留 `aten.masked_fill`，两者的可观测出口等价。

### 2.5 compare 阶段的最终规则

`autofusedebugger/graph.py` 会从 NPU `output_desc[].attr` 中提取 origin 属性，生成：

```json
{
  "fused_op": "autofused_add_...",
  "fused_index": 0,
  "origin_op": "add_15",
  "origin_index": 0
}
```

之后 `compare_manifests()` 对每个 NPU 输出按以下顺序选择 CPU 对照：

1. 优先使用 `fusions` 中的 `(origin_op, origin_index)`。
2. 没有融合映射时，尝试同名同输出序号 `(npu_op, npu_index)`。
3. 两者都不存在时，记录 `UNMATCHED_CPU_OP`。

最终比较阶段不会再次执行上面的名称候选、误差距离或类型推断；这些工作只发生在 native 图构建阶段。

### 2.6 历史文件名候选 helper

`torch_model.py` 中还保留了 `cpu_candidates_for_npu()` 和 `match_npu_to_cpu()`。这组函数从 CANN 转换后的 `.npy` 文件名中提取 `Add`、`MatMul`、`ReduceSum` 等 token，再用 `_NPU_CPU_PREFIXES` 生成 `add`、`matmul`、`sum` 等 CPU 候选，并按元素数和最大绝对误差选择候选。

它们主要用于历史适配和单元测试，当前 `run-script` / `run-torch-model` 的 native Inductor 主流程不调用这组函数。排查新环境时，不能把这组 helper 的规则误认为 native 主流程的唯一映射规则。

## 3. 旧 GE/Qwen 适配路径

部分历史 example/Qwen 适配仍使用 `npu_examples.py` 的 CANN GE 图。该路径优先使用编译器写入的 provenance，而不是通过数值误差猜测：

### 3.1 `_fx_tensor_name`

如果 GE 输出描述中存在：

```text
_fx_tensor_name = "<fx_node_name>.OUTPUT.<index>"
```

工具取出 `<fx_node_name>` 和输出序号，并在 CPU `torch.export` 节点中查找。当前显式别名包括：

```text
mm -> matmul
```

### 3.2 `_datadump_origin_name`

如果当前 GE 图输出没有直接的 `_fx_tensor_name`，工具会读取 pre-run GE 图，先建立：

```text
(GE node name, output index) -> (CPU FX node name, output index)
```

然后根据当前输出的：

```text
_datadump_origin_name
_datadump_origin_output_index
```

查找对应 CPU 输出。

### 3.3 `_datadump_original_op_names`

这个属性用于展示融合节点由哪些原始算子组成，以及构造融合子图；它本身不是最终精度映射的充分条件。真正决定 compare 结果的是写入 `fusions` 的 origin name/index 属性。

## 4. 为什么会出现“映射不上”

报告中的两类状态含义不同：

| 状态 | 含义 |
|---|---|
| `NPU_OUTPUT_NOT_CAPTURED` | NPU 图中存在该输出，但没有找到对应 dump 文件；这是采集问题，不一定是映射问题。 |
| `UNMATCHED_CPU_OP` | NPU 输出 dump 已存在，但没有建立到 CPU 输出的映射。 |
| `NOT_OBSERVABLE_ON_NPU` | CPU 节点属于 NPU 融合内部节点，且不是当前 dump 的融合出口；当前 dump 粒度下没有独立 NPU 输出。 |
| `CPU_FILE_NOT_FOUND` | 已经有映射，但映射指向的 CPU dump 文件不存在。 |

常见原因如下：

1. **编译器版本差异**：不同 `torch/torch_npu/inductor_npu_ext` 版本会改变 decomposition、FX 节点名、`benchmark.py` 结构或 provenance 属性名。
2. **融合出口不在 CPU dump 中**：融合 kernel 的 `# return` 节点可能在 CPU 侧被进一步 decomposition、被过滤，或不是 `call_function` Tensor 输出。
3. **输出元素数不同**：native 候选筛选首先要求 CPU/NPU 输出的元素数量相同；broadcast、reshape、动态 shape 或错误的输出序号都会导致候选被排除。
4. **多个同类型同元素数节点**：当模型中有多个 `add`、`mul`、`matmul` 形状相同且名称已被重写时，候选可能不唯一。工具会使用 provenance、未使用约束和 CPU FX 稳定顺序选择，但当前不会把候选集合完整写进报告。
5. **多输出算子**：当前 native 自动映射主要以输出 0 为中心，不能保证一个多输出算子的每个输出都有独立 origin 映射。
6. **图结构发生变化**：`fullgraph`、`dynamic`、graph break、参数常量化、layout/decomposition 变化都可能让 CPU 图和 NPU post-fusion 图不再拥有相同的出口语义。
7. **hook 没有覆盖该 kernel**：NPU kernel 没有 live output、生成代码格式变化、hook 安装时机不兼容或 dump 转换失败时，会表现为 `NPU_OUTPUT_NOT_CAPTURED`。
8. **不应比较的节点**：`Const`、`Data`、`NetOutput` 在 manifest 索引阶段会被排除，不属于逐算子精度比较对象。
9. **GE provenance 缺失**：旧 GE/Qwen 路径没有 `_fx_tensor_name`、`_datadump_origin_name` 或对应 pre-run 图时，不能可靠地恢复 CPU origin。

对于 native Autofuse 融合节点，报告会根据 `origin_ops` 额外生成 `NOT_OBSERVABLE_ON_NPU` 行。例如 CPU 图中有 `addmm`，而 NPU 将 `mm -> add -> mul -> relu` 融合，并且只 dump `relu` 出口，则 `addmm`/`mm`/`add` 等非出口节点不会被伪造映射到融合最终输出，而会标记为融合内部不可观测。此类行会保留在报告明细中，但不会计入 `coverage` 的可观测分母；只有 `UNMATCHED_CPU_OP` 和 `NPU_OUTPUT_NOT_CAPTURED` 会使总体状态变成 `INCOMPLETE`。

## 5. 另一个环境中的排查顺序

建议先判断是“没采到”还是“没映射”：

```bash
jq '.summary, .results[] | select(.status == "UNMATCHED_CPU_OP" or .status == "NPU_OUTPUT_NOT_CAPTURED")' \
  /tmp/model_accuracy/report.json
```

然后检查四类中间产物：

```text
cpu_graph.json
  CPU export 后实际生成的节点名、类型和输出 shape。

npu_graph.json
  post-fusion 节点、融合 origin 属性、输出 dtype/shape。

cpu_manifest.json / npu_manifest.json
  节点输出是否真的关联到了 .npy 文件。

inductor_compile/
  ir_post_fusion.txt、fx_graph_readable.py、autofused_*/benchmark.py、output_code.py。
```

对某个未映射 NPU 节点，重点核对：

1. `npu_graph.json` 是否存在输出文件路径。
2. `npu_graph.json` 的 `output_desc[].attr` 是否有 `_autofusedebugger_cpu_origin_name`。
3. 融合节点的 `origin_ops` 和 `benchmark.py` 的 `# return` 是否符合预期。
4. `cpu_graph.json` 是否真的存在同名出口节点，且该节点有输出 dump。
5. CPU/NPU 输出元素数量是否相同，输出 index 是否都是 0。
6. 当前运行使用的 `torch`、`torch_npu`、CANN 和 AscendC 后端是否来自同一套兼容环境。

可以直接查看一个 NPU 图节点的 origin 属性：

```bash
jq '.graph[0].op[] | {
  name,
  type,
  origin_ops,
  outputs: [.output_desc[] | {
    dtype: (.data_type // .dtype),
    shape: .shape.dim,
    attrs: [.attr[]? | select(.key | startswith("_autofusedebugger_cpu_origin"))]
  }]
}' /tmp/model_accuracy/npu_graph.json
```

## 6. 当前策略的边界和后续改进方向

当前策略已经满足“融合算子比较出口”和大部分常规 pointwise/reduce/matmul 场景，但它仍然不是完整的 FX/Inductor/ASCIR 图对齐器。特别是名称和类型相同的重复节点、多个输出、复杂 decomposition 和编译器版本差异场景，仍可能出现未映射或错误候选。

如果需要继续提高跨环境稳定性，建议按以下顺序演进：

1. 优先消费编译器提供的结构化 provenance，名称启发式只作为 fallback。
2. 对每个 output index 独立建立映射，禁止默认把所有输出都标成 origin index 0。
3. 把候选列表、排除原因和最终 score 写入 `report.json`，便于判断是名称、元素数还是 provenance 缺失。
4. 在候选相同的情况下增加输入边和拓扑邻域约束，减少仅靠名称、类型和顺序误配的可能。
5. 对版本、decomposition 和后端 provenance 字段做兼容层，并为每种真实 dump 格式增加回归测试。
