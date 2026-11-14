# CPU/NPU 算子输出映射策略

本文说明当前 `afx` 在逐算子精度比较时，如何把 NPU 输出映射到 CPU 输出，以及出现未映射时应该如何排查。

本文描述的是当前代码行为，不是理想化的编译器 IR 对齐方案。对应实现主要位于：

- `autofusex/torch_model.py`：从同一次 Inductor 编译截获 pre-fusion transformed FX 图和真实 AOT 输入，并在 CPU/NPU eager 回放。
- `autofusex/inductor_autofuse.py`：native Inductor + AscendC/Autofuse 路径的 post-fusion 图解析和 CPU origin 匹配。
- `autofusex/npu_examples.py`：旧 GE/Qwen 适配路径中的 compiler provenance 解析。
- `autofusex/graph.py`：把 NPU 图输出属性转换为 manifest 的 `fusions` 映射。
- `autofusex/compare.py`：根据 manifest 执行最终的输出对照。

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

## 2. 当前推荐路径：同一张 pre-fusion FX 图

`run-script` 和 `run-torch-model` 使用以下链路：

```text
Inductor pre-fusion CPU/NPU eager:
  同一次 torch.compile 的 transformed FX GraphModule
    -> compiled FX wrapper 真实 boxed AOT 输入
    -> CPU / NPU eager FX Interpreter 回放
    -> 两侧共享 canonical_id 的节点输出 dump

NPU Autofuse:
  torch.compile(options={"npu_backend": "ascendc"})
    -> TorchInductor
    -> post-fusion IR
    -> AscendC/Autofuse
    -> intermediate hook 输出 dump
```

### 2.1 canonical_id 从哪里来

工具不再分别执行 CPU/NPU `torch.export` 来构造两张基准图。NPU Autofuse 编译时，`GraphLowering` 会截获进入 Inductor 的 transformed FX `GraphModule`；compiled FX wrapper 第一次真实执行时，再截获与该图 placeholder 顺序一致的 boxed AOT 输入。随后对同一份 GraphModule 做独立副本回放：

```python
runtime_inputs = compiled_fx_wrapper_boxed_inputs
prefusion_graph = captured_graph_lowering_graph
```

每张图中的 placeholder 和 `call_function` 节点使用：

```text
graph{graph_index}::{fx_node_name}
```

该值是跨 CPU/NPU eager 回放和 Autofuse provenance 的稳定键；节点在不同后端显示名改变时，仍按 canonical ID 映射。显示名仍保留在 `name`/`display_name` 字段中，便于阅读。工具只保存每个 `call_function` 的第一个 Tensor 输出，文件名使用显示名：

```text
cpu_dump/<display_name>.output.0.npy
```

编译器删除未使用输入、重命名 placeholder 或产生多个 graph break 时，canonical ID 仍以实际 captured GraphModule 为准；多张图通过 `graph0`、`graph1` 区分。

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

### 2.3 native Autofuse 的 canonical provenance

实现函数是 `inductor_autofuse.build_inductor_autofuse_graph()`。当提供 pre-fusion graph data 时，NPU post-fusion 节点的 `origin_ops` 和 `# return` 出口名会解析为 pre-fusion canonical ID：

```text
NPU Autofuse fused output
  -> origin_canonical_id = graph0::relu_15
  -> Inductor pre-fusion CPU graph0::relu_15
```

普通未融合节点也使用出口节点的 canonical ID。这个映射发生在图构建阶段，不读取 Tensor 数值；因此不会因精度异常而把节点错误地映射到“数值更接近”的候选。每个 post-fusion 输出只关联一个可观测出口，融合内部节点不会伪造独立 NPU 输出。

只有旧环境没有提供 pre-fusion canonical graph 时，才回退到 `inductor_autofuse._match_cpu_output()` 的名称、类型、shape 和拓扑候选规则。回退流程对每个 NPU 主图节点目前只尝试其 `output.0`：

1. 找到 NPU intermediate hook 采集的输出文件；没有文件时不进入 CPU 映射，最终状态为 `NPU_OUTPUT_NOT_CAPTURED`。
2. 使用融合出口的 `exit_name/exit_type`；同时从 `benchmark.py` 的 `Original ATen` 注释中提取原始 ATen 类型。未融合或 fallback 节点使用其生成的 kernel/capture 名称和类型。
3. 遍历尚未使用的 CPU 输出，只保留 `element_count` 相同的候选。完整 shape 必须相同才会真正建立 provenance 映射；仅有 `()` 与 CANN 常见的 `(1,)` 这一标量表示差异时视为兼容。元素数相同但维度顺序不同的候选只用于记录冲突，不会被接受。
4. 候选必须满足下面至少一个条件：
   - CPU 输出名与期望名称相同。
   - CPU 输出名和期望名称规范化后相同。
   - CPU 节点类型和期望类型规范化后相同。
   - CPU 输出名或节点类型与 `Original ATen` provenance 匹配。
5. 按以下顺序选择候选，越靠前优先级越高：

```text
shape 完全相同或标量 shape 兼容
NPU 上游输出的 compiler origin 与 CPU 候选输入边一致
Original ATen 名称（含调用序号）相同
Original ATen 类型相同
精确名称相同
规范化名称相同
规范化类型相同
CPU FX 拓扑中的稳定顺序更靠前
```

6. 对没有 compiler provenance 的未融合节点，精确名称优先于去掉数字后缀的规范化名称，避免 `clone_6` 被映射到另一个同 shape 的 `clone`。如果最高候选仍然存在非标量 shape 冲突，记录 `PROVENANCE_CONFLICT`，不写入 origin 属性，也不在 compare 阶段按相同元素数展平比较。一个 CPU 输出只能被使用一次，避免多个 NPU 节点都映射到同一个 CPU 节点。
7. 找到候选后，把下面两个属性写到 NPU 图的输出描述中：

```text
_autofusex_cpu_origin_name
_autofusex_cpu_origin_output_index
```

需要特别注意：当前 native 匹配只使用编译器 provenance、名称、类型、shape 和稳定顺序，不读取 CPU/NPU 数值来决定映射。否则真正发生精度错误时，按数值距离选择候选会把失败节点错误映射到数值更接近但拓扑无关的 CPU 节点。该策略仍不是完整的拓扑图同构匹配。

### 2.3.1 FX 执行图与 Inductor IR 映射层

当前 native runner 对 `fx_graph_transformed.py` 和 `ir_pre_fusion.txt` 采用分层职责：

```text
fx_graph_transformed.py / transformed FX GraphModule
  -> CPU 回放和 NPU eager 回放
  -> 采集每个 call_function 的 Tensor 输出

ir_pre_fusion.txt
  -> 解析 pre-fusion SchedulerNode、写入 Buffer 和依赖
  -> 校验 post-fusion 节点属于同一个 model__N 编译图
  -> 为 Autofuse 出口提供 SchedulerNode 级映射证据

ir_post_fusion.txt + benchmark.py
  -> 解析 post-fusion 调度拓扑、融合 kernel 和融合出口
  -> 映射到 transformed FX canonical_id
```

`fx_graph_transformed.py` 是可执行的 FX/ATen 图，适合构造 CPU 和 NPU eager。`ir_pre_fusion.txt` 是 Inductor lower 后的 Scheduler IR，包含 `SchedulerNode`、`ComputedBuffer`、`ExternKernel`、`MemoryDep` 和 Layout 等信息，文本本身不用于回放 eager。两者都处于 scheduler fusion 之前，但抽象层次不同，不能把 IR 文本直接当作 CPU/NPU eager 图执行。

Autofuse 主图构建时使用以下映射链路：

```text
NPU post-fusion kernel
  -> ir_post_fusion.txt 中的 post SchedulerNode
  -> 同一 model__N/ir_pre_fusion.txt 中的 pre SchedulerNode 集合
  -> benchmark.py 的 origin_ops 和 # return 出口
  -> 同一 graphN 的 transformed FX canonical_id
  -> CPU/NPU eager 对应输出 Tensor
```

具体步骤如下：

1. 从 trace 目录名 `model__N_inference_...` 提取 `N`，同时用于 post-fusion 和 pre-fusion 图的 `graphN`，不再按文件名字典序或 trace 遍历位置重新编号。
2. 解析同一 trace 目录下的 `ir_pre_fusion.txt`，记录每个 `op0`、`op1` 等 SchedulerNode 的写入 Buffer 和输入依赖。
3. 对融合 kernel，读取 `benchmark.py` 中的 scheduler ID；对 `FusedSchedulerNode`，展开 `op0_op1` 中的组件 ID，并检查这些 ID 是否都存在于 `ir_pre_fusion.txt`。
4. 使用 `benchmark.py` 的 `origin_ops` 和 `# return %exit` 找到融合出口，再在同一个 `graphN` 的 transformed FX 图中解析 `graphN::exit`。
5. 对候选 FX 出口和 NPU dump 输出进行完整 shape 校验。只有 shape 相同，或仅存在标量 `()` 与 `(1,)` 的兼容差异时，才写入 canonical provenance。
6. 校验通过时，NPU 输出属性会记录 `_autofusex_prefusion_origin_id`、`_autofusex_mapping_method` 和对应的 pre-fusion SchedulerNode ID；CPU/NPU eager 仍通过同一 GraphModule 的 canonical ID 直接映射。
7. `ir_pre_fusion.txt` 缺失或 SchedulerNode 无法对应时，保留 compiler provenance 作为兼容回退，但将映射方法标记为 `canonical_source_id_ir_unverified`。shape 冲突时标记 `PROVENANCE_CONFLICT`，不建立映射，避免把不同 graph variant 或不同出口按元素个数强行比较。

当前主要 `mapping_method` 含义如下：

| mapping_method | 含义 |
|---|---|
| `ir_pre_fusion_canonical_id` | pre-fusion IR SchedulerNode 校验通过，再映射到 transformed FX canonical ID |
| `canonical_source_id_ir_unverified` | canonical provenance 可以使用，但缺少完整 pre-fusion IR 证据 |
| `canonical_source_id` | 历史或离线 manifest 按 canonical ID 映射 |
| `provenance_conflict` | graph、SchedulerNode 或输出 shape 证据冲突，没有建立映射 |

这套设计保留了 CPU/NPU eager 的真实单算子执行语义，同时利用 Inductor IR 约束 Autofuse 的编译拓扑，避免仅按节点名或遍历顺序映射。

### 2.3.2 CPU 回放中的 NPU 专用算子

部分 `torch_npu` lowering 会把原始 ATen 算子替换为 CPU 不具备 kernel 的 NPU 专用算子。例如，`scaled_dot_product_attention` 可能在 transformed FX 图中表现为 `npu_fusion_attention`。为保证同一张 pre-fusion FX 图仍能在 CPU 回放，工具对常见的推理形式注册语义回退：将 `BNSD`、`BSND` 或 `BSH` 布局的 `npu_fusion_attention` 回放为 CPU `scaled_dot_product_attention`。

回退只作用于 CPU Interpreter，不修改原始 FX 节点 target；NPU eager 仍执行原始 NPU 算子，Autofuse 仍走原始编译路径。当前 attention 回退要求 `pse=None`、`keep_prob=1.0`。另外，NPU `mm` 接受混合 dtype 而 CPU `mm` 不接受时，CPU 回放会将右矩阵转换为左矩阵 dtype，以匹配 NPU 的输出 dtype 规则；输入 dtype 已一致时不介入。不支持的布局或参数会明确报告为 CPU pre-fusion 回放失败。实际发生的回退会记录在 `cpu_graph.json` 的 `metadata.cpu_replay_fallbacks` 中。

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

### 2.5 旧报告的 CPU/NPU eager 语义对齐回退

旧版本报告没有 canonical ID 时，CPU 和 NPU eager 曾分别执行设备相关的 `torch.export + run_decompositions()`。不同设备可能选择不同 decomposition，例如 CPU 图中的：

```text
expand -> view -> bmm -> view -> add
```

在 NPU eager 图中可能变成：

```text
matmul -> add
```

这会让后续 FX 自动编号整体偏移。即使两侧都存在 `view_4`，它们也可能不是同一个语义节点。工具不会再仅凭同名映射 CPU/NPU eager，而是对两张有序 FX 图执行全局序列对齐：

1. 候选输出的完整 shape 和 dtype 必须一致。
2. 算子类型必须一致或属于明确等价组：`view/reshape`、`permute/transpose`、`bmm/matmul`、`_softmax/softmax`。
3. 在候选约束下按拓扑稳定顺序做全局对齐。
4. 同名只作为对齐加分项，不能覆盖类型、dtype 或 shape 不一致。
5. 映射过程不读取 Tensor 数值。

此类旧报告结果的 `mapping_method` 为 `topology_signature`；语义和名称都相同时仍显示 `same_name`。当前 native 三方 runner 优先使用 `canonical_source_id`，只有读取历史 manifest 或 canonical provenance 缺失时才进入此回退。

### 2.6 旧报告的 NPU eager 与 NPU Autofuse provenance 转换

历史 Autofuse compiler provenance 可能只指向 CPU export 节点名，不能直接拿该名称查找设备相关 decomposition 后的 NPU eager 图。旧三方比较先建立：

```text
CPU output -> NPU eager semantic output
```

再把 Autofuse 的 CPU origin 转换为 NPU eager 节点：

```text
NPU Autofuse output -> CPU origin -> NPU eager semantic output
```

成功转换时旧报告 `mapping_method` 为 `compiler_origin_via_eager`。当前 runner 因 CPU/NPU eager 共享 canonical ID，直接按 `canonical_source_id` 映射，不再依赖这条二段转换。

### 2.7 compare 阶段的最终规则

`autofusex/graph.py` 会从 NPU `output_desc[].attr` 中提取 origin 属性，生成：

```json
{
  "fused_op": "autofused_add_...",
  "fused_index": 0,
  "origin_op": "add_15",
  "origin_index": 0,
  "origin_canonical_id": "graph0::add_15"
}
```

之后 `compare_manifests()` 根据比较组合选择基准输出：

1. pre-fusion CPU ↔ NPU Autofuse：优先使用 `fusions.origin_canonical_id`。
2. pre-fusion CPU ↔ NPU eager：使用同一 GraphModule 的 `canonical_id` 直接映射。
3. pre-fusion NPU eager ↔ NPU Autofuse：使用同一 `canonical_id` 和 Autofuse 出口 provenance 直接映射。
4. 历史 manifest 没有 canonical ID 时，才使用 2.5/2.6 节的拓扑、类型、dtype、shape 和名称回退。
5. 不能可靠对齐时记录 `UNMATCHED_CPU_OP`；不会静默按同名节点强行比较。

最终比较阶段不会再次执行上面的名称候选、误差距离或类型推断；这些工作只发生在 native 图构建阶段。

### 2.8 历史文件名候选 helper

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
| `PROVENANCE_CONFLICT` | provenance 候选存在，但 CPU/NPU 完整输出 shape 不兼容；该节点不参与数值比较，避免按相同元素个数展平后误报精度失败。标量 `()` 与 CANN `(1,)` 视为兼容。 |
| `NOT_OBSERVABLE_ON_NPU` | CPU 节点属于 NPU 融合内部节点，且不是当前 dump 的融合出口；当前 dump 粒度下没有独立 NPU 输出。 |
| `NOT_OBSERVABLE_IN_REFERENCE` | Autofuse origin 在当前基准图中被 decomposition 合并或消除，没有可独立比较的 NPU eager 输出。 |
| `CPU_FILE_NOT_FOUND` | 已经有映射，但映射指向的 CPU dump 文件不存在。 |

常见原因如下：

1. **编译器版本差异**：不同 `torch/torch_npu/inductor_npu_ext` 版本会改变 decomposition、FX 节点名、`benchmark.py` 结构或 provenance 属性名。
2. **融合出口不在 CPU dump 中**：融合 kernel 的 `# return` 节点可能在 CPU 侧被进一步 decomposition、被过滤，或不是 `call_function` Tensor 输出。
3. **输出元素数不同**：native 候选筛选首先要求 CPU/NPU 输出的元素数量相同；broadcast、reshape、动态 shape 或错误的输出序号都会导致候选被排除。
4. **多个同类型同元素数节点**：当模型中有多个 `add`、`mul`、`matmul`、`clone` 形状相同且名称已被重写时，候选可能不唯一。工具先要求候选可从已映射的上游 origin 拓扑到达；融合出口优先使用 benchmark 中的精确出口语义，再用最短路径消歧。非融合节点没有可达候选时按编译器插入节点处理，不会强行映射到同类型节点。不能把 Inductor 节点编号与 CPU FX 节点编号视为稳定对应关系。
5. **多输出算子**：当前 native 自动映射主要以输出 0 为中心，不能保证一个多输出算子的每个输出都有独立 origin 映射。
6. **图结构发生变化**：`fullgraph`、`dynamic`、graph break、参数常量化、layout/decomposition 变化都可能让 CPU 图和 NPU post-fusion 图不再拥有相同的出口语义。
7. **hook 没有覆盖该 kernel**：NPU kernel 没有 live output、生成代码格式变化、hook 安装时机不兼容或 dump 转换失败时，会表现为 `NPU_OUTPUT_NOT_CAPTURED`。
8. **不应比较的节点**：`Const`、`Data`、`NetOutput` 在 manifest 索引阶段会被排除，不属于逐算子精度比较对象。
9. **GE provenance 缺失**：旧 GE/Qwen 路径没有 `_fx_tensor_name`、`_datadump_origin_name` 或对应 pre-run 图时，不能可靠地恢复 CPU origin。

报告中的红色首次失败表示当前节点是首个已观测到的整体 Tensor 失败；后续整体失败节点如果已经存在失败祖先，会显示为“上游失败传播”。这表示当前输出确实未通过，但不能据此断言该单算子又新增了精度误差。

整体 Tensor 指标还会记录适用性说明：KL 散度仅用于非负、非近零 Tensor；余弦相似度和欧氏相对距离对近零向量不稳定时显示 `N/A`。当所有元素均满足 dtype-aware 的 `rtol/atol` 时，整体判定不会被这些不适用或不稳定的聚合指标反向判失败。这样可以避免 `var_mean`、`getitem` 等输出只有极小误差，却因均值接近零或包含负值而被统计成多个整体 Tensor 失败。

`summary.cann_failed` 是未通过整体判定的结果行总数；`summary.cann_direct_failed` 是没有失败上游的直接失败行，`summary.cann_propagated_failed` 是已有失败上游的传播行，`summary.cann_uncertain_failed` 是上游存在未映射或不可独立观测节点的行。HTML 报告会同时展示这三个分项，避免把一条首失败沿数据依赖链产生的多个传播结果误认为多个独立算子问题。

报告 JSON 中的 `failure_attribution` 也会记录该归因：`DIRECT_FAILURE` 表示疑似当前节点首次引入失败，`UPSTREAM_FAILURE` 表示已有失败祖先，`UPSTREAM_UNCERTAIN` 表示上游有未映射或不可独立观测节点，无法确认当前节点是否首次引入误差。

对于 native Autofuse 融合节点，报告会根据 `origin_ops` 额外生成 `NOT_OBSERVABLE_ON_NPU` 行。例如 CPU 图中有 `addmm`，而 NPU 将 `mm -> add -> mul -> relu` 融合，并且只 dump `relu` 出口，则 `addmm`/`mm`/`add` 等非出口节点不会被伪造映射到融合最终输出，而会标记为融合内部不可观测。此类行会保留在报告明细中，但不会计入 `coverage` 的可观测分母；`UNMATCHED_CPU_OP`、`PROVENANCE_CONFLICT` 和 `NPU_OUTPUT_NOT_CAPTURED` 会使总体状态变成 `INCOMPLETE`。

## 5. 另一个环境中的排查顺序

建议先判断是“没采到”还是“没映射”：

```bash
jq '.summary, .results[] | select(.status == "UNMATCHED_CPU_OP" or .status == "PROVENANCE_CONFLICT" or .status == "NPU_OUTPUT_NOT_CAPTURED")' \
  /tmp/model_accuracy/report.json
```

然后检查四类中间产物：

```text
cpu_graph.json
  同次 Inductor 编译截获的 pre-fusion CPU 回放图、canonical ID、类型和输出 shape。

npu_graph.json
  post-fusion 节点、融合 origin canonical 属性、输出 dtype/shape。

cpu_manifest.json / npu_manifest.json
  节点输出是否真的关联到了 .npy 文件。

inductor_compile/
  ir_post_fusion.txt、fx_graph_readable.py、autofused_*/benchmark.py、output_code.py。
```

对某个未映射 NPU 节点，重点核对：

1. `npu_graph.json` 是否存在输出文件路径。
2. `npu_graph.json` 的 `output_desc[].attr` 是否有 `_autofusex_cpu_origin_name`。
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
    attrs: [.attr[]? | select(.key | startswith("_autofusex_cpu_origin"))]
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
