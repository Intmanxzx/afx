# Manifest Schema

## 目标

manifest 用于解耦“如何采集 CPU/NPU/CAModel dump”和“如何逐算子比较精度”。比较器只读取 manifest，不直接依赖 PyTorch、GE、ACL 或 CAModel API。

## 顶层字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `schema` | string | 固定为 `autofusex-manifest/1` |
| `side` | string | `cpu`、`npu_eager`、`npu_autofuse`，或 `inductor_prefusion_cpu` / `inductor_prefusion_npu_eager` |
| `model` | object | 模型名、图文件等描述信息 |
| `artifacts.root` | string | tensor dump 文件根目录 |
| `nodes` | array | 图中节点及其输出 tensor |
| `fusions` | array | 仅 NPU 侧需要，描述融合输出到 CPU 单算子输出的映射 |

## nodes

```json
{
  "name": "Relu_1",
  "type": "Relu",
  "outputs": [
    {
      "index": 0,
      "path": "Relu_1.output.0.npy",
      "format": "ND",
      "dtype": "DT_FLOAT",
      "shape": [1, 128]
    }
  ]
}
```

`path` 可为相对路径或绝对路径。相对路径基于 `artifacts.root` 解析。`format`、`dtype`、`shape` 当前用于校验和容错；实际比较以 `.npy` 内容为准。Inductor pre-fusion manifest 额外包含 `canonical_id`，格式为 `graph{index}::{fx_node_name}`；CPU 和 NPU eager manifest 来自同一张 transformed FX 图，因此优先按该 ID 映射。

native NPU Autofuse graph node 还可能包含以下映射诊断字段：

| 字段 | 说明 |
|---|---|
| `ir_pre_fusion` | 同一 `model__N` trace 的 pre-fusion SchedulerNode、写入 Buffer、输入依赖和匹配状态 |
| `mapping_method` | `ir_pre_fusion_canonical_id`、`canonical_source_id_ir_unverified` 或 `provenance_conflict` |
| `provenance_conflicts` | graph、SchedulerNode 或输出 shape 不一致时的详细原因 |

其中 `ir_pre_fusion` 只用于 NPU post-fusion 到 pre-fusion FX 的映射校验，不用于执行 CPU/NPU eager。CPU/NPU eager 节点仍来自同一 transformed FX GraphModule，并通过 `canonical_id` 对齐。

## fusions

```json
{
  "fused_op": "autofuse_0_Add_Relu",
  "fused_index": 0,
  "origin_op": "Relu_1",
  "origin_index": 0,
  "origin_canonical_id": "graph0::relu_1",
  "mapping_method": "ir_pre_fusion_canonical_id",
  "prefusion_ir_scheduler_ids": ["op0"]
}
```

比较时，若 NPU 输出 `(fused_op, fused_index)` 命中 `fusions`，则优先使用 `origin_canonical_id` 找到 pre-fusion 对照项，再回退到 `(origin_op, origin_index)`。普通 pre-fusion 输出优先按 `canonical_id` 映射；没有 canonical provenance 的旧 manifest 才使用同名或拓扑语义对齐。

## 状态码

| 状态 | 说明 |
|------|------|
| `OK` | 正常比较 |
| `OK_FORMAT_CONVERTED` | 已完成支持的 format 转换 |
| `OK_DTYPE_CAST` | dtype 不一致，提升到公共 dtype 后比较 |
| `OK_SHAPE_FLATTENED` | shape 不一致但元素数相同，展平后比较 |
| `CPU_FILE_NOT_FOUND` | CPU 对照 tensor 缺失 |
| `NPU_FILE_NOT_FOUND` | NPU tensor 缺失 |
| `LOAD_ERROR` | tensor 文件加载失败 |
| `FORMAT_UNSUPPORTED` | format 不一致且无内置转换 |
| `SHAPE_MISMATCH` | shape 不一致且元素数不同 |

## 支持的 format 转换

| 源 format | 目标 format |
|-----------|-------------|
| `NC1HWC0` | `NHWC` / `ND` |
| `NDC1HWC0` | `NDHWC` / `ND` |

后续如需要支持 FRACTAL、NZ 等格式，应在 `autofusex/execution/compare.py` 中补充确定的布局转换函数，并添加测试。
