# autofusedebugger

`autofusedebugger` 用于比较同一个 PyTorch 模型在 CPU、NPU eager 和 NPU Autofuse 三条路径上的逐算子/逐 kernel 输出精度，并生成 HTML 报告。

自动化模型命令只使用以下两条数据链路：

- CPU：`torch.export` 后执行默认 Core ATen decomposition，与 Inductor 图对齐，并用 FX Interpreter 保存每个 `call_function` 节点输出。
- NPU eager：复用导出的 ATen/FX 图，在 NPU 上直接用 FX Interpreter 执行，不经过 Inductor 和 AscendC，用于区分 NPU 原生 eager 误差与融合误差。
- NPU：加载 `torch_npu` 内置的 `torch_npu._inductor.ascendc` 后端，并调用 `torch.compile(model, options={"npu_backend": "ascendc"})`，执行 PyTorch Inductor -> AscendC/Autofuse 路径。工具不会调用 `torchair.get_npu_backend()`，也不会使用 GE 图作为 NPU 主图。

NPU 中间输出由 PyTorch Inductor intermediate hook 采集。工具为 AscendC 后端的 `autofused_*` 调用补充相同 hook，只保存融合 kernel 的出口；`mm`、`tanh` 等 fallback kernel 使用 Inductor 原生 hook。主图拓扑来自 `ir_post_fusion.txt`，融合子图来自对应 kernel `benchmark.py` 中的 ASCIR `graph_hint`。

## 安装

在已安装 PyTorch 2.9、torch_npu 2.9 和 CANN 的 NPU 环境执行：

```bash
cd /home/xingzhixiong/dfx_dir/autofusedebugger
python3 -m pip install -e .
```

当前 `inductor-npu-ext` 已随 `torch_npu` 打包在 `torch_npu/_inductor/ascendc`，无需单独安装。执行工具前可确认内置后端能够导入：

```bash
python3 -c "import torch_npu; import torch_npu._inductor.ascendc; print(torch_npu._inductor.ascendc.__file__)"
```

工具同时兼容仍使用独立 `inductor_npu_ext` 包的旧环境，但优先使用 `torch_npu` 内置后端。

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

工具执行脚本并在目标最外层 `torch.nn.Module` 调用发生前截获模型、位置参数和关键字参数。原脚本中的该次推理不会实际执行；工具为 CPU、NPU eager 和 NPU Autofuse 分别创建独立的模型/输入副本，避免任一采集阶段修改后续阶段的状态。捕获时的 train/eval、grad 和 autocast 状态会随调用保存并在对应阶段恢复。

默认只迁移模型参数和 Tensor 输入到目标设备，不会修改模型的 `dev`、`device` 等普通属性，也不会强制 `eval()` 或 `no_grad()`。对于依赖这类属性创建张量的旧模型，可显式传入 `--rewrite-device-hints` 兼容开关；该开关可能改变模型图，仅应在确认需要时使用。

若 CPU 的 `torch.export` 明确报出 `Unhandled FakeTensor Device Propagation`，工具会仅对独立的 CPU 模型副本自动重试一次，将其 `dev/device` 属性临时映射为 `cpu`。这用于处理模型在 `forward` 内通过 `torch.arange(..., device=self.dev)` 等方式创建临时张量的场景；原模型和两条 NPU 路径均不受影响。重试仍失败时，错误信息会同时保留首次和重试错误。

默认按比较双方精度较低的 dtype 自动选择逐元素 `np.isclose` 阈值：

| dtype | rtol | atol |
|---|---:|---:|
| `float64` | `1e-7` | `1e-7` |
| `float32` | `1e-5` | `1e-5` |
| `float16` | `1e-3` | `1e-3` |
| `bfloat16` | `1e-2` | `1e-2` |
| integer / bool | `0` | `0` |

显式传入 `--rtol`、`--atol` 可覆盖自动阈值；JSON、CSV 和 HTML 明细会记录每个节点实际使用的阈值。

报告同时保留两套判定：报告中的“逐元素对比”（内部字段 `strict`）使用逐元素 `np.isclose`，要求所有元素通过；“整体 Tensor 对比”（内部字段 `cann`）将整个输出 Tensor 作为整体，使用 Cosine Similarity、Relative Euclidean Distance 和 KL Divergence，阈值分别为 `cosine > 0.99`、`relative_euclidean < 0.05`、`KL < 0.005`，三项都通过才算通过。报告总状态要求两套标准同时通过。

非有限值按位置和类型比较：CPU/NPU 同一位置均为 `NaN`、均为 `+Inf` 或均为 `-Inf` 时，该元素按一致通过处理；若 Tensor 还包含有限值，整体指标只在有限值子集上计算；若全部元素都是对齐的非有限值，则整体判定通过、聚合指标显示 `N/A`，节点以绿色“非有限值一致 · 通过”展示。位置不同、`NaN`/`Inf` 类型不同、`+Inf`/`-Inf` 符号不同或有限值与非有限值不一致时仍判定失败。这个结论只表示 CPU/NPU 结果一致，不表示模型输出本身在业务上有效。

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
- 默认使用 `fullgraph=False`，与 `torch.compile` 默认行为一致；`--fullgraph` 才会强制单一完整图。
- `--rewrite-device-hints` 临时改写模块的 `dev/device` 属性以兼容旧模型，默认关闭。
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

`create_case()` 返回 `(model, inputs)`，或包含 `model`、`inputs`、可选 `kwargs/fullgraph/dynamic/name/rewrite_device_hints` 的字典。
未指定时使用 `fullgraph=False`；`run-torch-model --fullgraph` 可显式覆盖该设置。

仓库提供了一个包含 17 个 `bf16` 输入、16 个串联 `torch.add` 的验证用例：

```bash
autofusedbg run-torch-model \
  --script examples/af_bf16_add_chain16.py \
  --out-dir /tmp/bf16_add_chain16
```

命令只执行一次 NPU 模型，会自动完成：

1. 采集 CPU ATen FX 每个节点输出。
2. 在 NPU 上直接执行导出的 ATen FX 图，采集 NPU eager 每个节点输出。
3. 通过 `torch.compile(options={"npu_backend": "ascendc"})` 执行原生 Inductor + AscendC/Autofuse。
4. 采集每个 NPU 调度 kernel 输出；融合 kernel 只采集出口。
5. 解析真实 Inductor post-fusion 拓扑、compiler provenance 和 ASCIR 融合子图。
6. 收集同次编译生成的所有 `AutoFuseAfterOptimize` 图。
7. 生成三方 manifest、JSON、CSV 和 HTML 精度报告。

主要产物：

```text
OUT_DIR/
  cpu_dump/              CPU 节点输出 .npy
  npu_eager_dump/        NPU eager 节点输出 .npy
  npu_dump/              NPU Autofuse/fallback kernel 输出 .npy
  inductor_compile/      FX、post-fusion IR、generated code、provenance
  autofuse_dump/         AutoFuseAfterOptimize 等 AscendC DFX 图
  cpu_graph.json
  npu_eager_graph.json
  npu_graph.json
  cpu_manifest.json
  npu_eager_manifest.json
  npu_manifest.json
  report.json
  report.csv
  report.html
  report_data/          大 Tensor 逐元素数据，打开 report.html 时按需加载
```

### 运行进度与卡顿判断

`run-script` 和 `run-torch-model` 会为 CPU/NPU 采集、图重建、三方精度比较、AutoFuse 图收集、报告写入和 HTML 渲染输出 `START/DONE/FAILED` 日志。单个阶段超过 30 秒时，每 30 秒输出一次包含墙钟耗时、CPU 时间和 RSS 内存的 `RUNNING` 心跳；连续 120 秒没有可测 CPU 进展时输出 `WARNING`，提示该阶段可能正在等待外部任务或已经卡住。

可通过环境变量调整两个时间间隔：

```bash
export AUTOFUSEDEBUGGER_HEARTBEAT_SECONDS=10
export AUTOFUSEDEBUGGER_STALL_SECONDS=60
```

HTML 阶段会分别显示 `HTML template rendering` 和 `HTML streaming write`，完成后报告 HTML 大小与绝对路径。小 Tensor 可直接内嵌；大 Tensor 写入与 HTML 同目录的 `<HTML文件名>_data/tensor_*.js`，点击节点的 `T` 按钮时才从本地加载和解码。三方比较中内容相同的 Tensor 只保存一份。看到 `run completed html=...` 表示所有报告产物已经完整落盘。

复制或归档报告时，必须同时保留 HTML 及其同名 `_data` 目录。例如 `report.html` 与 `report_data/` 应放在同一目录。只移动 HTML 会导致逐元素 Tensor 查看器提示数据文件加载失败，但主图和精度指标仍可查看。

默认单个不超过 64 KiB、累计不超过 8 MiB 的 Tensor 数据内嵌到 HTML，可通过环境变量调整；均设为 `0` 可让所有 Tensor 数据外置：

```bash
export AUTOFUSEDEBUGGER_EMBED_TENSOR_BYTES=0
export AUTOFUSEDEBUGGER_EMBED_TOTAL_BYTES=0
```

已有超大 `report.json` 时，`report-html` 会流式扫描 JSON，把全部逐元素 Tensor 数据拆到同名 `_data` 目录，不把整份报告加载进内存：

```bash
autofusedbg report-html --report work/report.json --out work/report.html
```

## HTML 报告

报告以 NPU Autofuse 主图为主要区域：

- 主图节点对应真实 Inductor 调度 kernel；绿色表示逐元素对比和整体 Tensor 对比都通过，橙色表示仅逐元素对比失败，红色表示整体 Tensor 对比失败，蓝色表示输入/权重，灰色表示未映射或未采集。
- 单击节点查看精度和输入输出；可同时向前追溯输入链、向后追溯输出链，并选择一层、两层或全部链路。
- 融合节点右上角蓝色入口用于进入 ASCIR 融合子图。
- 图支持滚轮缩放、鼠标拖动和自动适配。
- `AutoFuseAfterOptimize` 按 `autofused_*` kernel 分组，每组保留全部优化变体；选中主图融合节点时会高亮对应分组，并可从详情抽屉直接进入。
- `AutoFuseAfterOptimize` 节点会解析 `sched.exec_condition`；启用 broadcast 条件缓存的节点使用青色 `BC` 圆标和 `BROADCAST CACHE` 标签，详情中展示完整 condition，生命周期节点表头同步标记。
- 每张优化图提供默认收起的 Queue/Buffer 时间线：Queue 按 `queue_id/reuse_id` 分层，横条表示按拓扑推导的静态占用区间，实心/空心节点分别表示生产者和直接消费者；支持 Queue、Buffer、全部视图切换、状态过滤、横轴缩放，并与上方 ASCIR 拓扑双向高亮。展开后可拖动顶部把手调整时间线高度，双击把手会尽量显示全部资源。
- 主图节点只显示一行精度摘要 `Precision: 逐元素pass, 整Tensor pass/fail`；逐算子明细表按“逐元素精度对比指标”和“整 Tensor 精度对比指标”分组展示 max abs、max rel、RMSE、`CosineSimilarity（余弦值相似度）`、`RelativeEuclideanDistance（欧氏相对距离）`、`KullbackLeiblerDivergence（KL散度）`，表格和节点详情抽屉均支持横向滚动。误差详情默认收起。输出含 `Inf/NaN` 时还会显示 CPU/NPU 两侧的数量、同位置数量和首个位置。
- 主图沿真实数据依赖 DAG 识别首次“整体 Tensor 失败”的节点。仅逐元素失败但整体 Tensor 通过的橙色告警不会向下游传播为首失败；上游比较链完整时标记 `FIRST FAIL`，融合出口标记 `FUSION FIRST FAIL`。如果当前节点已确认整体失败、但上游存在未映射或未采集节点，则标记红色 `FIRST OBSERVED FAIL`，表示“当前是首个已观测失败，但不能排除不可比较的上游更早失败”。详情抽屉展示判断依据、输入传播路径和上游不确定节点；后续失败节点展示其上游首失败边界。并行分支可以各自存在首失败节点。
- 三方报告顶部提供 `CPU ↔ NPU Autofuse`、`CPU ↔ NPU eager`、`NPU eager ↔ NPU Autofuse` 三组页签；切换时会同步更新统计、主图和逐算子明细。
- 有逐元素 dump 的节点会显示蓝色 `T` 按钮，可查看二维展开后的 CPU 值、NPU 值、绝对误差、相对误差和逐元素 PASS/FAIL。

已有 `report.json` 时可单独重建 HTML：

```bash
autofusedbg report-html --report work/report.json --out work/report.html
```

功能截图、三方比较示例、未融合原因和 TQue/TBuf 生命周期展示见
[docs/feature_gallery.md](docs/feature_gallery.md)。

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

算子输出映射规则、融合出口定义、未映射状态和跨环境排查方法见
[docs/accuracy_mapping.md](docs/accuracy_mapping.md)。
