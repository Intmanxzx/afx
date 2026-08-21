# autofusedebugger

`autofusedebugger` 是一个用于分析并可视化各类机器学习网络（包括大模型训练、推荐等）的功能、精度、性能以及能力评估的自动化（agent化）工具。
当前仅支持pytorch+ascendc后端的网络分析，后续计划拓展到tensorflow/jax前端框架以及af-mlir等后端。

## 安装

在已经准备好 PyTorch、torch_npu 和 CANN 的环境中执行：

```bash
git clone https://gitcode.com/xingzhixiong/Autofuse-Debugger.git
cd ${HOME}/Autofuse-Debugger    # ${HOME}表示用户拉取工具代码的目录
python3 -m pip install -e .
```

## 使用

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

- [docs/usage.md](docs/usage.md)
- [docs/accuracy_mapping.md](docs/accuracy_mapping.md)
- [docs/manifest_schema.md](docs/manifest_schema.md)
- [docs/feature_gallery.md](docs/feature_gallery.md)
