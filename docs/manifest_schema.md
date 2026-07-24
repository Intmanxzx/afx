# Manifest Schema

## 目标

manifest 用于解耦“如何采集 CPU/NPU/CAModel dump”和“如何逐算子比较精度”。比较器只读取 manifest，不直接依赖 PyTorch、GE、ACL 或 CAModel API。

## 顶层字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `schema` | string | 固定为 `autofusedebugger-manifest/1` |
| `side` | string | `cpu` 或 `npu` |
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

`path` 可为相对路径或绝对路径。相对路径基于 `artifacts.root` 解析。`format`、`dtype`、`shape` 当前用于校验和容错；实际比较以 `.npy` 内容为准。

## fusions

```json
{
  "fused_op": "autofuse_0_Add_Relu",
  "fused_index": 0,
  "origin_op": "Relu_1",
  "origin_index": 0
}
```

比较时，若 NPU 输出 `(fused_op, fused_index)` 命中 `fusions`，则 CPU 对照项使用 `(origin_op, origin_index)`。否则默认使用同名同 index 的 CPU 输出。

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

后续如需要支持 FRACTAL、NZ 等格式，应在 `autofusedebugger/compare.py` 中补充确定的布局转换函数，并添加测试。
