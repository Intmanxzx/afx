# 使用与执行阶段

本文记录 `afx` 的完整使用方式，以及 CPU、NPU eager 和 NPU Autofuse 三条执行路径分别使用的图和数据。根目录 [README.md](../README.md) 只保留快速上手内容。

## 环境要求

工具当前完整实现的是 `torch + Inductor + Autofuse` 路径，通常需要：

- Python 3.9 或更高版本
- PyTorch 2.9
- 与 PyTorch 匹配的 `torch_npu`
- CANN
- 可执行 NPU kernel 的真实 NPU 环境

`inductor_npu_ext` 如果已经随 `torch_npu` 打包，无需再次安装。

Fuzz 的客户端、云端和离线查看命令见 [Fuzz 使用说明](client/advanced.md)。

安装工具：

```bash
cd /home/xingzhixiong/dfx_dir/Autofuse-Debugger
python3 -m pip install -e .
```

安装后使用命令 `afx`；在未安装或需要调试仓库代码时，也可以使用：

```bash
PYTHONPATH=. python3 -m afx --help
```

TensorFlow 和 JAX adapter 目前只保留接口，尚未实现执行和精度采集。

## 一键运行模型脚本

### 直接在原始命令前加工具名

对于已有的模型启动脚本，不需要改动模型脚本内容或为 `afx` 手工填写模型入口，
可以将原来的：

```bash
python3 sasrec.py --device npu --device_id 0
```

改为：

```bash
afx python3 sasrec.py --device npu --device_id 0
```

工具会把 `sasrec.py` 识别为目标脚本，将它后面的参数原样传给脚本，并继续使用当前目录作为
报告输出位置。对于 `/path/to/model.py`，默认产物为：

```text
./model_autofusex/       # dump、图、manifest、JSON/CSV 等中间产物
./model_autofusex.viewer.report   # 最终 HTML 报告
```

再次执行会清理并重新生成同名中间产物目录，同时覆盖同名 HTML。快捷模式会先静态扫描脚本的
`import`，识别 `torch`、`tensorflow` 或 `jax`，再选择对应的 framework adapter；不会执行脚本来
判断框架，也不会修改脚本。当前只有 `pytorch + inductor_autofuse` 路径可执行，TF/JAX 会被识别后
明确提示对应 adapter/backend 尚未实现。未识别到框架时为兼容从其他模块导入模型的旧脚本，回退到
PyTorch；同时识别到多个框架时会报歧义并要求显式传入 `--framework`。需要调整捕获调用、保存逐
元素数据、阈值或输出路径时，使用下面的完整 `run-script` 命令。

### 通过 shell 启动脚本运行

网络的启动方式是 shell 脚本（内部 export 环境变量、调用 python）时，直接把脚本交给工具：

```bash
afx bash launch.sh     # 或 afx ./launch.sh
```

工具对启动脚本做静态解析（不执行）：

- 提取所有 `export KEY=VALUE` 行（含变量引用展开与数组变量第一项），注入当前进程环境。
- 定位 `python xxx.py ...` 命令行作为模型入口；`for model in ${MODELS[@]}` 之类的循环变量按
  第一项展开，续行（反斜杠）自动拼接；`.py` 路径按启动脚本所在目录解析。
- `rm`、`mkdir`、`cd`、`source` 等语句不会执行；`source` 的外部脚本（如 CANN `set_env.sh`）
  不会被展开，需要先在 shell 中 source 后再运行工具。缓存清理等副作用由用户自行决定。
- `TORCHINDUCTOR_NPU_BACKEND`、`TORCH_COMPILE_DEBUG`、`TORCHINDUCTOR_FORCE_DISABLE_CACHES`
  由工具管理，脚本中的同名值会被忽略并打印提示。

只继承环境文件、不改入口方式时使用 `run-script` 的环境参数：

```bash
afx model run \
    --script /path/to/model.py \
    --env-file /path/to/launch.sh \
    --env EXTRA_KEY=VALUE \
    --out-dir /tmp/model_accuracy \
    -- \
    --device npu
```

- `--env-file` 读取 shell 文件中的 `export` 行并注入（不执行文件），适合复用 launch.sh 或
  单独的 env 文件。
- `--env KEY=VALUE` 直接注入单个变量，可重复。
- 先在 shell 中 `export` 再运行 `afx python3 model.py` 时变量天然生效：工具与
  模型同进程执行，继承当前 shell 的全部环境。

`run-script` 面向用户已有的推理脚本。脚本不需要实现 `create_case()`，工具会捕获目标 `torch.nn.Module` 的一次调用，并使用捕获到的模型和输入完成三条路径的采集。

```bash
afx model run \
    --framework pytorch \
    --backend inductor_autofuse \
    --script /path/to/model.py \
    --out-dir /tmp/model_accuracy \
    --html /tmp/model_accuracy.html \
    -- \
    --device npu \
    --device_id 0
```

参数规则：

- `--script` 是实际执行的模型脚本。
- `--out-dir` 是中间产物、dump、manifest 和 JSON/CSV 结果目录。
- `--html` 是最终 HTML 文件路径；不传时默认写到 `OUT_DIR/report.html`。
- `--framework pytorch` 选择 PyTorch adapter。
- `--framework auto` 根据脚本导入自动识别框架；这是默认值。
- `--backend inductor_autofuse` 选择原生 Inductor + AscendC/Autofuse 后端。
- 最后的 `--` 是工具参数和模型脚本参数的分隔符，后面的内容原样传给模型脚本。
- `--capture-call` 选择匹配模块调用中的第几个调用，序号从 `0` 开始。
- `--module-pattern` 按模块完整类名或类名正则筛选捕获目标。
- `--fullgraph` 强制使用单一完整图；默认允许 graph break。
- `--dynamic` 开启动态 shape 编译。
- `--rewrite-device-hints` 是兼容旧模型的显式开关，用于临时改写模型的 `dev/device` 属性，默认关闭。
- `--save-elementwise-data` 保存 HTML 中逐位置查看所需的 Tensor 数据，默认关闭。
- `--env-file` 从 shell 文件读取 export 行注入环境（见上文"通过 shell 启动脚本运行"）。
- `--env KEY=VALUE` 注入单个环境变量，可重复。

工具不会修改模型脚本文件，也不会改变脚本中模型的 forward 逻辑。默认只将模型参数和 Tensor 输入迁移到当前比较路径的目标设备，不强制调用 `eval()` 或 `no_grad()`。如果模型依赖普通属性 `dev/device` 创建张量，才需要显式使用 `--rewrite-device-hints`。

### `cann-ge` 测试框架脚本

`feature/AutoFuse/PyTorch/Inductor/` 下的 `tc_*.py` 用例由 `xfusion_fmk.py` 调用，通常不在脚本顶层执行模型，也不提供 `create_case()`。工具会根据脚本向上查找同时包含 `libs/__init__.py` 和 `feature/` 的 `cann-ge` 根目录，将该目录加入导入路径并临时作为工作目录；如果顶层没有捕获到模块调用，再调用脚本中与文件同名的 `tc_*` 函数。

因此可以直接使用快捷模式：

```bash
cd /workspace/xingzhixiong/test-dev/smoke-testing/cann-ge
afx python3 \
    feature/AutoFuse/PyTorch/Inductor/tc_af_torch_inductor_codegen_0001.py
```

对于 PyTorch 用例，如果 `tc_*` 调用了 `run_af(..., framework="pytorch")`，工具会在 `run_af` 的模型/输入边界截获 `Model` 和 Baize 的 `gen_inputs` 生成的最后一组输入，并保持 Baize 的 `eval()`、`no_grad()` 语义，然后立即停止用例，由工具自己执行 CPU pre-fusion、NPU eager pre-fusion 和 NPU Autofuse 三条路径。因此不会启动 Baize 自己的 benchmark 子进程，也不会让工具的捕获 hook 泄漏到 Baize worker。未使用 `run_af` 而是在 `tc_*` 内直接调用模型的用例，继续使用模型调用捕获作为兼容路径。适配调用只使用 `kernel_path`、`case_path`、`case_name`、`soc_version` 和 `device_id` 这五个框架参数启动用例；框架原有的性能测试、SLOG 检查和结果校验不会作为报告执行流程的一部分。`soc_version` 和 `device_id` 优先读取当前进程的 `SOC_VERSION`、`ASCEND_DEVICE_ID`/`DEVICE_ID` 环境变量。该适配仍要求在与框架一致的 Python、PyTorch、torch_npu 和 CANN 环境中执行。

### Agent 归一化和失败诊断

复杂 PyTorch/Baize 脚本可以先交给可配置的 Agent 生成固定格式的归一化 JSON：

```bash
export AUTOFUSEX_AGENT_ENDPOINT=https://agent.example.com/v1/chat/completions
export AUTOFUSEX_AGENT_API_KEY='your-api-key'
export AUTOFUSEX_AGENT_MODEL='your-model-name'

afx model normalize \
    --script /path/to/model.py \
    --framework pytorch \
    --out /tmp/model_accuracy/agent_normalization.json
```

需要与执行联动时使用：

```bash
afx model run \
    --script /path/to/model.py \
    --out-dir /tmp/model_accuracy \
    --html /tmp/model_accuracy.html \
    --agent-normalize \
    --agent-recover
```

`--agent-normalize` 会把入口、模型、图阶段、输入、权重和执行约束保存为 `agent_normalization.json`，并只将入口筛选建议交给现有 PyTorch adapter。`--agent-recover` 在执行失败后生成 `agent_recovery.json`，其中包含失败阶段、证据和用户可审阅的修复建议。Agent 不执行用户源码、命令或补丁，也不静默修改模型逻辑；完整 schema 和安全边界见 [agent_normalization.md](agent_normalization.md)。

Agent 的 HTTP、CANNBot、Codex/Claude/OpenCode CLI 配置，以及各环境变量的含义和默认值，见
[agent_normalization.md](agent_normalization.md)。工具只接收归一化结果和诊断结果，实际模型执行仍由
AutofuseX 完成。

## 兼容 `create_case()` 用例

不执行推理、而是通过工厂函数返回模型和输入的用例可以使用 `run-torch-model`：

```bash
afx model run --entry create_case \
    --script /path/to/case.py \
    --entry create_case \
    --out-dir /tmp/model_accuracy \
    --html /tmp/model_accuracy.html
```

`create_case()` 返回 `(model, inputs)`，或包含 `model`、`inputs` 以及可选 `kwargs`、`fullgraph`、`dynamic`、`name`、`rewrite_device_hints` 的字典。这个入口主要用于 functional API 用例或自动捕获目标调用存在歧义的脚本。

## 执行路径与图阶段

一次 `run-script` 或 `run-torch-model` 运行会围绕同一次 Inductor 编译建立数据路径。**默认只执行
NPU eager pre-fusion 回放和 NPU Autofuse 执行，生成两方比较**；大网络下 CPU 回放（逐节点 CPU 参考计
算）是最耗时的部分，默认跳过可显著缩短采集时间。需要三方 CPU 对比时增加 `--with-cpu`：

```text
模型脚本
  -> torch.compile + torch_npu._inductor.ascendc
  -> transformed FX GraphModule / ir_pre_fusion.txt
  -> [--with-cpu] CPU pre-fusion 回放
  -> NPU eager pre-fusion 回放
  -> NPU Autofuse post-fusion 执行
  -> manifest / compare / HTML
```

具体含义如下：

1. **Inductor pre-fusion CPU**（仅 `--with-cpu`）：从 Autofuse 编译过程中截获实际进入 Inductor 的 transformed FX `GraphModule`，使用真实的 AOT 扁平化输入在 CPU 上回放，采集每个 pre-fusion 节点输出。
2. **Inductor pre-fusion NPU eager**：使用同一个 transformed FX `GraphModule` 和同一份 AOT 输入，在 NPU eager 上回放，采集未融合算子输出。CPU 与 NPU eager 共享 `canonical_id`，用于确认单算子 CPU/NPU 执行是否存在差异。
3. **NPU Autofuse**：执行原生 `torch.compile(..., options={"npu_backend": "ascendc"})`，采集 post-fusion 调度 kernel 输出。融合 kernel 只比较其出口输出节点；融合内部节点不作为主精度明细中的独立输出。
4. **`ir_pre_fusion.txt`**：只用于解析 Inductor pre-fusion 的 SchedulerNode、Buffer、依赖和图 ID，校验 post-fusion kernel 与 pre-fusion 图的关系，不直接作为 CPU/NPU eager 的执行输入。
5. **post-fusion 图**：主 HTML 图来自真实的 Inductor post-fusion 拓扑；融合子图来自对应 kernel 的 ASCIR `graph_hint`。`AutoFuseAfterOptimize` 图来自同一次 AscendC 编译生成的 DFX dump。

默认报告是 **NPU eager ↔ NPU Autofuse** 两方比较（单页签），直接回答"Autofuse 融合是否引入精度
差异"。开启 `--with-cpu` 后报告包含三组比较，在 HTML 顶部的页签中切换：

- `Inductor pre-fusion CPU ↔ NPU Autofuse`
- `Inductor pre-fusion CPU ↔ Inductor pre-fusion NPU eager`
- `Inductor pre-fusion NPU eager ↔ NPU Autofuse`

算子映射规则和未映射原因见 [accuracy_mapping.md](accuracy_mapping.md)。

## 精度结果与逐元素数据

工具默认按比较双方较低 dtype 自动选择 `rtol/atol`，并同时计算：

- 逐元素指标：`max abs`、`max rel`、`RMSE`、`rtol/atol`、`mismatch`、有限值和非有限值位置。
- 整 Tensor 指标：`CosineSimilarity（余弦值相似度）`、`RelativeEuclideanDistance（欧氏相对距离）`、`KullbackLeiblerDivergence（KL散度）`。

主图中同一精度结果类别使用渐变色表示误差严重度：颜色越深表示最大逐元素误差相对于该位置 `rtol/atol` 容差的倍数越大。颜色采用对数尺度，达到 `1000x` 及以上按最深色显示，避免极端异常节点压低其他节点的颜色差异；节点详情和悬浮提示中的“相对容差误差倍数”仍显示具体值。整体 Tensor 指标单独失败且逐元素通过时，会使用整体指标相对于对应阈值的倍数参与颜色计算。未映射、未采集和融合内部不可观测节点不参与该颜色计算。

默认不会保存每个 Tensor 位置的 CPU/NPU 原值和 PASS/FAIL mask，以避免整网 HTML 和中间数据过大；逐元素判定、mismatch 数和汇总指标仍会正常计算。需要打开 HTML 的 Tensor 逐位置查看器时，在工具参数中增加：

```bash
afx model run \
    --script /path/to/model.py \
    --out-dir /tmp/model_accuracy \
    --html /tmp/model_accuracy.html \
    --save-elementwise-data
```

启用后，报告会为有数据的节点显示 `T` 入口，并将较大的 Tensor 数据外置到 HTML 同名数据目录。复制报告时需要同时保留 HTML 和该目录。

默认阈值可用命令行覆盖：

```bash
afx model run \
    --script /path/to/model.py \
    --out-dir /tmp/model_accuracy \
    --rtol 1e-5 \
    --atol 1e-5
```

`NaN`、`+Inf`、`-Inf` 会按位置和类型分别判断；两侧同位置同类型的非有限值计为一致通过，但这只说明两条路径结果一致，不说明该模型输出在业务上有效。

## 主要输出

```text
OUT_DIR/
  cpu_dump/              Inductor pre-fusion FX 在 CPU 上的节点输出（仅 --with-cpu）
  npu_eager_dump/        同一张 pre-fusion FX 在 NPU eager 上的节点输出
  npu_dump/              NPU Autofuse/fallback kernel 输出
  inductor_compile/      transformed FX、pre/post-fusion IR 和编译产物
  autofuse_dump/         AutoFuseAfterOptimize 等 AscendC DFX 图
  cpu_graph.json         CPU pre-fusion 图描述（仅 --with-cpu）
  npu_eager_graph.json   NPU eager pre-fusion 图描述
  npu_graph.json         NPU Autofuse post-fusion 图描述
  *_manifest.json        各路径的 dump 清单
  report.json             精度比较结果
  report.csv              表格结果
  report.html             HTML 报告
```

HTML 以 NPU Autofuse post-fusion 图为主图，支持融合子图、逐算子精度明细、首次整体失败节点、未融合原因、`AutoFuseAfterOptimize` 图以及 TQue/TBuf 生命周期视图。功能示例见 [feature_gallery.md](feature_gallery.md)，manifest 字段见 [manifest_schema.md](manifest_schema.md)。

## 离线处理

已有 dump 和 manifest 时可以跳过模型执行，单独生成比较结果或 HTML：

```bash
afx analyze manifest \
    --side npu_autofuse \
    --graph work/npu_graph.json \
    --dump-dir work/npu_dump \
    --out work/npu_manifest.json

afx analyze compare-three \
    --cpu-manifest work/cpu_manifest.json \
    --npu-eager-manifest work/npu_eager_manifest.json \
    --npu-autofuse-manifest work/npu_manifest.json \
    --out work/report.json \
    --csv work/report.csv \
    --html work/report.html

afx analyze report \
    --report work/report.json \
    --out work/report.html
```

`--autofuse-dump` 可以将 `AutoFuseAfterOptimize` dump 挂载到离线报告；未指定时工具会查找 NPU manifest 同级的 `autofuse_dump/` 目录。

## 无卡与 CAModel

`TORCHINDUCTOR_NPU_EXT_DEBUG=cpu` 只能验证 host 编译流程，不能代表真实 NPU 数值。完整的逐 kernel 精度报告应在可执行 AscendC kernel 的 NPU 环境运行。CAModel 相关目录探测和外部命令封装可以使用：

```bash
afx debug camodel info --cann-root /path/to/cann
afx debug camodel run \
    --cann-root /path/to/cann \
    --out-dir work/camodel \
    --command 'bash run.sh'
```
