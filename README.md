# autofusedebugger

`autofusedebugger` 是一个用于分析并可视化各类机器学习网络（包括大模型训练、推荐等）的功能、精度、性能以及能力评估的自动化（agent化）工具。
当前仅支持pytorch+ascendc后端的网络分析，后续计划拓展到tensorflow/jax前端框架以及af-mlir等后端。

## 安装

请先准备好 PyTorch、torch_npu 和 CANN 环境，再选择以下一种方式安装。

### 方式一：源码安装

```bash
git clone https://gitcode.com/xingzhixiong/Autofuse-Debugger.git
cd ${HOME}/Autofuse-Debugger    # ${HOME}表示用户拉取工具代码的目录
python3 -m pip install -e .
```

### 方式二：wheel 包安装

进入 [v0.1.0 发行版](https://gitcode.com/xingzhixiong/Autofuse-Debugger/releases/v0.1.0)，下载
`autofusedebugger-0.1.0-py3-none-any.whl`，然后在 wheel 所在目录执行：

```bash
python3 -m pip install ./autofusedebugger-0.1.0-py3-none-any.whl
```

安装完成后可以执行以下命令验证：

```bash
autofusedebugger --help
```

## 使用

对于已有的 PyTorch 推理脚本，最简单的方式是在原始 `python3` 命令前加上
`autofusedebugger`。工具会自动识别脚本路径，原样传递脚本参数，并在当前执行目录生成报告：

```bash
autofusedebugger python3 /path/to/model.py \
    --device npu \
    --device_id 0
```

默认输出为：

```text
./model_autofusedebugger/
./model_autofusedebugger.html
```

其中 `model` 是脚本文件名（不含 `.py`）。该快捷方式会根据脚本中的框架导入自动选择
`pytorch`、`tensorflow` 或 `jax`；当前只有 PyTorch + Inductor Autofuse 路径可执行，TF/JAX
识别结果会给出明确的未实现提示。快捷方式默认执行三方精度采集；需要指定
`--capture-call`、`--save-elementwise-data`、输出目录或其他工具参数时，使用下面的完整命令。

直接运行一个会构造模型、输入并执行推理的 PyTorch 脚本：

```bash
python3 -m autofusedebugger run-script \
    --framework pytorch \
    --backend inductor_autofuse \
    --script /path/to/model.py \
    --out-dir /tmp/model_accuracy \
    --html /tmp/model_accuracy.html \
    -- \
    --device npu \
    --device_id 0
```

`--` 后面的参数会原样传给模型脚本。如需在 HTML 中查看输出 Tensor 的逐元素数据时，增加 `--save-elementwise-data`；不指定时仍会正常进行逐元素精度对比，但不保存逐元素数据。

## 输出

`--out-dir` 保存运行中间产物和 JSON/CSV 数据，`--html` 保存可视化报告。常见文件包括：

```text
OUT_DIR/
  cpu_dump/              CPU 节点输出
  npu_eager_dump/        NPU eager 节点输出
  npu_dump/              NPU Autofuse/fallback kernel 输出
  inductor_compile/      Inductor 图、IR 和编译产物
  autofuse_dump/         AutoFuseAfterOptimize 图
  *_manifest.json        各执行路径的输出清单
  report.json             精度对比结果
  report.csv              表格结果
  report.html             HTML 报告
```

启用 `--save-elementwise-data` 后，还会生成 HTML 使用的逐元素数据目录。完整的执行阶段、算子映射、精度指标、报告字段和离线命令见：

| 文档 | 说明 |
|---|---|
| [docs/usage.md](docs/usage.md) | 工具使用方式、执行阶段、输出目录和离线分析命令 |
| [docs/accuracy_mapping.md](docs/accuracy_mapping.md) | CPU、NPU eager 和 NPU Autofuse 之间的算子映射策略 |
| [docs/manifest_schema.md](docs/manifest_schema.md) | manifest 文件的字段、数据格式和状态码说明 |
| [docs/feature_gallery.md](docs/feature_gallery.md) | 精度对比、图展示和生命周期分析等功能示例 |
