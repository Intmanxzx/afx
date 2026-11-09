# autofusedebugger

`autofusedebugger` 是一个用于分析并可视化各类机器学习网络（包括大模型训练、推荐等）的功能、精度、性能以及能力评估的自动化（agent化）工具。
当前仅支持pytorch+ascendc后端的网络分析，后续计划拓展到tensorflow/jax前端框架以及af-mlir等后端。

## 新版 fuzz（开发中）

完整重放只需要一个数字 seed：

```sh
autofusedbg fuzz replay 42
```

可选执行参数：`--backend ascendc`、`--static-only`、`--performance`；它们不改变 seed 对应的图、权重、输入与完整用例指纹。`autofusedbg fuzz seed 42 --show-spec` 可在无 Torch/NNSmith 环境查看规格。

新版不支持旧 af-fuzz、长 token、profile 或 --seed-token。旧版发行包不包含此新入口。协议与当前限制见 [数字 seed 说明](docs/af_fuzz_seed_v1.md)。

## 安装

请先准备好 PyTorch、torch_npu 和 CANN 环境，再选择以下一种方式安装。

### 方式一：wheel 包安装

进入 [v0.1.2 发行版](https://gitcode.com/xingzhixiong/Autofuse-Debugger/releases/v0.1.2)，下载
`autofusedebugger-0.1.2-py3-none-any.whl`，然后在 wheel 所在目录执行：

```bash
python3 -m pip install ./autofusedebugger-0.1.2-py3-none-any.whl
```

### 方式二：源码安装

```bash
git clone https://gitcode.com/xingzhixiong/Autofuse-Debugger.git
cd ${HOME}/Autofuse-Debugger    # ${HOME}表示用户拉取工具代码的目录
python3 -m pip install -e .
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
识别结果会给出明确的未实现提示。快捷方式默认只执行 **NPU eager ↔ NPU Autofuse 两方精度
采集与比较**（大网络下更快）；需要三方 CPU 对比时增加 `--with-cpu`（完整命令
`run-script`/`run-torch-model` 同名参数）。需要指定 `--capture-call`、
`--save-elementwise-data`、输出目录或其他工具参数时，使用下面的完整命令。

### 通过 shell 启动脚本运行

很多网络通过 shell 脚本启动：脚本里 export 一批环境变量，再调用 python。此时可以直接把
启动脚本交给工具：

```bash
autofusedebugger bash launch.sh
# 或
autofusedebugger ./launch.sh
```

工具静态解析启动脚本（不执行它）：提取 `export KEY=VALUE` 注入当前进程环境，找到
`python xxx.py ...` 命令行作为模型入口（`for` 循环变量取第一项展开），然后进入正常的采集
流程。启动脚本中的 `rm`/`mkdir`/`source` 等语句不会执行，缓存清理由用户自行决定；
`source` 的外部脚本（如 CANN `set_env.sh`）不会被展开，需要先在 shell 中 source 后再运行
工具。`TORCHINDUCTOR_NPU_BACKEND` 等少量由工具管理的变量以工具为准，脚本中的同名值会被
忽略并提示。

也可以只继承环境文件而不改变入口方式：

```bash
autofusedebugger run-script \
    --script /path/to/model.py \
    --env-file /path/to/launch.sh \
    --env EXTRA_KEY=VALUE \
    --out-dir /tmp/model_accuracy \
    -- \
    --device npu
```

`--env-file` 只读取文件中的 export 行（同样不执行），`--env KEY=VALUE` 可重复使用。
先在 shell 中 `export` 再直接运行 `autofusedebugger python3 model.py` 时，这些变量天然
生效（工具与模型同进程执行）。

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

### 可选 Agent 配置

Agent 只在显式启用归一化或失败诊断时调用：

```bash
export AUTOFUSEDEBUGGER_AGENT_NORMALIZE=1
export AUTOFUSEDEBUGGER_AGENT_RECOVER=1
autofusedebugger python3 /path/to/model.py --device npu
```

等价地，也可以在 `run-script` 中增加 `--agent-normalize --agent-recover`，或使用
`normalize-script` 单独生成归一化 JSON。Agent 只做静态分析和失败诊断，不执行、修改或自动修补模型脚本。

工具通过传输通道连接 Agent，不绑定某一个模型厂商或客户端。目前支持：

| 通道 | 含义 | 适用场景 |
|---|---|---|
| `http` | 调用 OpenAI-compatible Chat Completions HTTP endpoint | 推荐；适用于任何能提供兼容 HTTP 接口的 Agent 服务 |
| `cannbot` | 启动本地 CANNBot CLI | 当前工具内置的本地 CLI 适配 |
| `auto` | 优先选择 HTTP；否则选择本地 CANNBot | 不确定通道时使用，默认值 |

HTTP Agent 的通用配置如下，`endpoint`、`model` 的具体值由 Agent 服务提供方决定：

```bash
export AUTOFUSEDEBUGGER_AGENT_TRANSPORT=http
export AUTOFUSEDEBUGGER_AGENT_ENDPOINT=https://agent.example.com/v1/chat/completions
export AUTOFUSEDEBUGGER_AGENT_API_KEY='your-api-key'
export AUTOFUSEDEBUGGER_AGENT_MODEL='your-model-name'
```

各环境变量的含义如下：

| 环境变量 | 含义 | 默认值 |
|---|---|---|
| `AUTOFUSEDEBUGGER_AGENT_TRANSPORT` | Agent 连接通道：`auto`、`http` 或 `cannbot` | `auto` |
| `AUTOFUSEDEBUGGER_AGENT_ENDPOINT` | OpenAI-compatible HTTP POST 地址；使用 `http` 时必填 | 空 |
| `AUTOFUSEDEBUGGER_AGENT_API_KEY` | HTTP 请求的 Bearer API key；不需要鉴权时可不设置 | 空 |
| `AUTOFUSEDEBUGGER_AGENT_MODEL` | 传给 Agent 服务的模型标识，不是工具内部固定枚举 | `default` |
| `AUTOFUSEDEBUGGER_AGENT_COMMAND` | CANNBot 可执行文件路径；例如 CANNBot 不在 `PATH` 时使用 | 自动查找 `cannbot` |
| `AUTOFUSEDEBUGGER_AGENT_TIMEOUT` | 单次 Agent 请求超时时间，单位为秒 | `90` |
| `AUTOFUSEDEBUGGER_AGENT_JSON_MODE` | 是否请求服务端返回 JSON：`1/true` 开启，`0/false` 关闭 | 开启 |
| `AUTOFUSEDEBUGGER_AGENT_NORMALIZE` | 快捷模式是否执行脚本归一化：`1/true` 开启 | 关闭 |
| `AUTOFUSEDEBUGGER_AGENT_RECOVER` | 快捷模式失败后是否请求诊断建议：`1/true` 开启 | 关闭 |

如果使用的是 Codex、Claude、OpenCode 等非 CANNBot 工具，有两种方式：

1. 如果它们通过网关或服务提供 OpenAI-compatible HTTP 接口，按上面的 `http` 方式设置
   `ENDPOINT/API_KEY/MODEL`；工具不需要知道 Agent 的品牌名称。
2. 如果只有本地 CLI，当前版本不能把 `codex`、`claude` 或 `opencode` 直接填入
   `AUTOFUSEDEBUGGER_AGENT_COMMAND`。该变量当前只适用于内置的 CANNBot 调用格式
   `cannbot run --format json --model MODEL PROMPT`。此时应提供一个兼容包装器，将工具传入的
   prompt 转交给本地 CLI，并让标准输出只返回一个 JSON 对象；或者将该 CLI 暴露为 HTTP
   endpoint，再使用 `http` 通道。不能只把 `AUTOFUSEDEBUGGER_AGENT_MODEL` 设置为 `codex` 或
   `claude` 来完成切换。

完整的请求格式、归一化 JSON、失败诊断和本地 CLI 包装器约定见
[docs/agent_normalization.md](docs/agent_normalization.md)。

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
| [docs/agent_normalization.md](docs/agent_normalization.md) | Agent 归一化协议、配置方式和失败诊断流程 |
