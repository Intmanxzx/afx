# autofusedebugger

`autofusedebugger` 用于比较同一个 PyTorch 模型在 CPU 和 NPU Autofuse 路径上的逐 kernel 输出精度，并生成单文件 HTML 报告。

自动化模型命令只使用以下两条数据链路：

- CPU：`torch.export` 后执行默认 Core ATen decomposition，与 Inductor 图对齐，并用 FX Interpreter 保存每个 `call_function` 节点输出。
- NPU：加载 `inductor_npu_ext` 后调用默认 `torch.compile(model)`，执行 PyTorch Inductor -> AscendC/Autofuse 路径。工具不会调用 `torchair.get_npu_backend()`，也不会使用 GE 图作为 NPU 主图。

NPU 中间输出由 PyTorch Inductor intermediate hook 采集。工具为 `inductor_npu_ext` 的 `autofused_*` 调用补充相同 hook，只保存融合 kernel 的出口；`mm`、`tanh` 等 fallback kernel 使用 Inductor 原生 hook。主图拓扑来自 `ir_post_fusion.txt`，融合子图来自对应 kernel `benchmark.py` 中的 ASCIR `graph_hint`。

## 安装

在已安装 PyTorch 2.9、torch_npu 2.9、CANN 和 `inductor_npu_ext` 的 NPU 环境执行：

```bash
cd /home/xingzhixiong/dfx_dir/autofusedebugger
python3 -m pip install -e .
```

也可以不安装：

```bash
PYTHONPATH=. python3 -m autofusedebugger --help
```

## 普通脚本一键运行

推荐使用 `run-script`。用户脚本不需要实现工具接口，只需要像平常一样构造模型、输入并发起目标推理：

```python
import torch


class Model(torch.nn.Module):
    def forward(self, x, bias):
        return torch.sum(torch.relu(x + bias), dim=1)

if __name__ == "__main__":
    torch.manual_seed(1)
    model = Model().eval()
    model(torch.randn(8, 16), torch.randn(16))
```

在 NPU 机器本地运行：

```bash
source /home/developer/Ascend/cann-9.1.0/set_env.sh

autofusedbg run-script \
  --framework pytorch \
  --backend inductor_autofuse \
  --script /path/to/model.py \
  --cann-root /home/developer/Ascend/cann-9.1.0 \
  --out-dir /tmp/my_model_accuracy \
  --html /tmp/my_model_accuracy.html
```

工具执行脚本并在目标最外层 `torch.nn.Module` 调用发生前截获模型、位置参数和关键字参数。原脚本中的该次推理不会实际执行，CPU reference 和 NPU Autofuse 执行均由工具使用同一份输入快照发起。

默认按比较双方精度较低的 dtype 自动选择逐元素 `np.isclose` 阈值：

| dtype | rtol | atol |
|---|---:|---:|
| `float64` | `1e-7` | `1e-7` |
| `float32` | `1e-5` | `1e-5` |
| `float16` | `1e-3` | `1e-3` |
| `bfloat16` | `1e-2` | `1e-2` |
| integer / bool | `0` | `0` |

显式传入 `--rtol`、`--atol` 可覆盖自动阈值；JSON、CSV 和 HTML 明细会记录每个节点实际使用的阈值。

报告同时保留两套判定：`Strict` 是逐元素 `np.isclose`，要求所有元素通过；`CANN` 使用 Cosine Similarity、Relative Euclidean Distance 和 KL Divergence，阈值分别为 `cosine > 0.99`、`relative_euclidean < 0.05`、`KL < 0.005`，三项都通过才算通过。报告总状态要求两套标准同时通过。包含 `Inf/NaN` 时，相关聚合指标会标记为 `N/A`，并在详情中显示非有限值类型、数量和位置。

脚本有多个模型或多次推理时，可选择目标调用：

```bash
autofusedbg run-script \
  --script /path/to/model.py \
  --out-dir /tmp/model_accuracy \
  --module-pattern 'QwenModel' \
  --capture-call 1 \
  -- --model-path /path/to/weights --batch-size 1
```

- `--module-pattern`：按模块完整类名或类名正则筛选。
- `--capture-call`：筛选结果中从零开始的调用序号。
- `--` 后的参数原样传递给模型脚本。
- `--allow-graph-breaks` 对应 `fullgraph=False`，默认使用 `fullgraph=True`。
- `--dynamic` 开启动态 shape 编译。

### 框架和后端适配器

当前支持矩阵：

| Framework adapter | Backend adapter | 状态 |
|---|---|---|
| `pytorch` | `inductor_autofuse` | 已实现 |
| `tensorflow` | - | 接口占位，未实现 |
| `jax` | - | 接口占位，未实现 |

`--framework auto` 当前解析为 `pytorch`。选择尚未实现的 TensorFlow/JAX adapter 会直接报告明确错误，不会回退到其他执行路径。

### `create_case()` 兼容入口

原 `create_case()` 方式继续保留，用于脚本不执行推理、全部使用 functional API，或自动截获存在歧义的场景：

```bash
autofusedbg run-torch-model \
  --script /path/to/my_case.py \
  --out-dir /tmp/my_model_accuracy
```

`create_case()` 返回 `(model, inputs)`，或包含 `model`、`inputs`、可选 `kwargs/fullgraph/dynamic/name` 的字典。

命令只执行一次 NPU 模型，会自动完成：

1. 采集 CPU ATen FX 每个节点输出。
2. 通过默认 `torch.compile` 执行原生 Inductor + `inductor_npu_ext` + Autofuse。
3. 采集每个 NPU 调度 kernel 输出；融合 kernel 只采集出口。
4. 解析真实 Inductor post-fusion 拓扑、compiler provenance 和 ASCIR 融合子图。
5. 收集同次编译生成的所有 `AutoFuseAfterOptimize` 图。
6. 生成 manifest、JSON、CSV 和 HTML 精度报告。

主要产物：

```text
OUT_DIR/
  cpu_dump/              CPU 节点输出 .npy
  npu_dump/              NPU Autofuse/fallback kernel 输出 .npy
  inductor_compile/      FX、post-fusion IR、generated code、provenance
  autofuse_dump/         AutoFuseAfterOptimize 等 AscendC DFX 图
  cpu_graph.json
  npu_graph.json
  cpu_manifest.json
  npu_manifest.json
  report.json
  report.csv
  report.html
```

## HTML 报告

报告以 NPU Autofuse 主图为主要区域：

- 主图节点对应真实 Inductor 调度 kernel；绿色表示 Strict 和 CANN 都通过，橙色表示仅 Strict 失败，红色表示 CANN 失败，蓝色表示输入/权重，灰色表示未映射或未采集。
- 单击节点查看精度和输入输出；可同时向前追溯输入链、向后追溯输出链，并选择一层、两层或全部链路。
- 融合节点右上角蓝色入口用于进入 ASCIR 融合子图。
- 图支持滚轮缩放、鼠标拖动和自动适配。
- `AutoFuseAfterOptimize` 按 `autofused_*` kernel 分组，每组保留全部优化变体；选中主图融合节点时会高亮对应分组，并可从详情抽屉直接进入。
- `AutoFuseAfterOptimize` 节点会解析 `sched.exec_condition`；启用 broadcast 条件缓存的节点使用青色 `BC` 圆标和 `BROADCAST CACHE` 标签，详情中展示完整 condition，生命周期节点表头同步标记。
- 每张优化图提供默认收起的 Queue/Buffer 时间线：Queue 按 `queue_id/reuse_id` 分层，横条表示按拓扑推导的静态占用区间，实心/空心节点分别表示生产者和直接消费者；支持 Queue、Buffer、全部视图切换、状态过滤、横轴缩放，并与上方 ASCIR 拓扑双向高亮。展开后可拖动顶部把手调整时间线高度，双击把手会尽量显示全部资源。
- 误差详情默认收起，可查看 Strict/CANN 判定、max abs、max rel、RMSE、Cosine、Relative Euclidean Distance、KL Divergence、mismatch 和最大有限误差位置。输出含 `Inf/NaN` 时还会显示 CPU/NPU 两侧的数量、同位置数量和首个位置。

已有 `report.json` 时可单独重建 HTML：

```bash
autofusedbg report-html --report work/report.json --out work/report.html
```

## 离线 manifest 比较

`make-manifest`、`compare` 和 `compare-npy` 仍可用于已有 `.npy` 数据：

```bash
autofusedbg compare \
  --cpu-manifest work/cpu_manifest.json \
  --npu-manifest work/npu_manifest.json \
  --autofuse-dump work/autofuse_dump \
  --out work/report.json \
  --csv work/report.csv \
  --html work/report.html
```

`--autofuse-dump` 用于把 `AutoFuseAfterOptimize` AscendC DFX 图挂载到离线报告中；如果不传，工具会自动查找 NPU manifest 同级的 `autofuse_dump/` 目录。没有该目录时，报告仍可完成 CPU/NPU 精度比较，但不会显示 AutoFuseAfterOptimize 页签。

## 无卡环境

`TORCHINDUCTOR_NPU_EXT_DEBUG=cpu` 只能验证 host 编译流程，不代表 NPU 数值。完整逐 kernel 精度报告必须在可运行 torch_npu 和 AscendC kernel 的 NPU 环境执行。模型 runner 会主动清除该变量，避免误把 DEBUG=cpu 当成真实 NPU Autofuse 结果。

CAModel 环境探测和外部命令封装仍可使用：

```bash
autofusedbg camodel-info --cann-root /home/xingzhixiong/Ascend/cann-9.1.0
autofusedbg run-camodel --cann-root /path/to/cann --out-dir work/camodel --command 'bash run.sh'
```

更多 manifest 字段见 [docs/manifest_schema.md](docs/manifest_schema.md)。
