# Agent 模型脚本归一化

## 作用

`afx` 可以把 PyTorch 模型脚本发送给 Agent，由 Agent 静态识别模型入口、图构造方式、输入生成方式、权重来源和运行约束，并返回固定格式的 `autofusex-normalization/1` JSON。工具通过 OpenAI-compatible HTTP 或本地 CANNBot CLI 连接 Agent，不绑定某一个模型厂商或客户端。

Agent 的职责是提供结构化分析和入口选择建议；模型脚本仍由本地 PyTorch adapter 加载和执行。Agent 不执行用户代码、不读取本地权重、不修改原始脚本，也不会自动执行它返回的命令或补丁。

## 配置

当前支持 `http` 和 `cannbot` 两种传输通道，没有 `opencode`、`codex` 或 `claude` 专用适配器。Agent 品牌不是工具的配置项，关键是选择一个工具支持的传输协议。

### HTTP 通道

HTTP 通道适用于任何提供 OpenAI-compatible Chat Completions 接口的 Agent 服务。使用环境变量保存 endpoint 和 API key，避免把密钥直接写进命令行历史：

```bash
export AUTOFUSEX_AGENT_TRANSPORT=http
export AUTOFUSEX_AGENT_ENDPOINT=https://agent.example.com/v1/chat/completions
export AUTOFUSEX_AGENT_API_KEY='your-api-key'
export AUTOFUSEX_AGENT_MODEL='your-model-name'
export AUTOFUSEX_AGENT_TIMEOUT=90
```

endpoint 必须是兼容 Chat Completions 请求的 HTTP POST 地址。工具发送的请求包含 `model`、`messages`、`temperature=0`，默认还带 `response_format={"type":"json_object"}`；如果服务端不支持 JSON mode，可以设置：

```bash
export AUTOFUSEX_AGENT_JSON_MODE=0
```

源码内容会发送到配置的 Agent endpoint。生产环境应使用组织内的可信 endpoint，并根据模型脚本的敏感信息策略决定是否启用。

HTTP 请求使用 `AUTOFUSEX_AGENT_MODEL` 作为请求中的 `model` 字段。这个值由 endpoint 支持的模型决定，例如 `gpt-...`、`claude-...` 或组织内部模型名称；工具不会校验或改写它。

### 环境变量速查

| 环境变量 | 作用 | 默认值或要求 |
|---|---|---|
| `AUTOFUSEX_AGENT_TRANSPORT` | 选择 `auto`、`http` 或 `cannbot` 通道 | `auto` |
| `AUTOFUSEX_AGENT_ENDPOINT` | HTTP Agent 的 POST 地址 | `http` 时必填 |
| `AUTOFUSEX_AGENT_API_KEY` | HTTP `Authorization: Bearer ...` 鉴权值 | endpoint 要求鉴权时设置 |
| `AUTOFUSEX_AGENT_MODEL` | 传给 endpoint 或 CANNBot 的模型名称 | `default` |
| `AUTOFUSEX_AGENT_COMMAND` | CANNBot 可执行文件路径或命令 | 自动查找 `cannbot` |
| `AUTOFUSEX_AGENT_TIMEOUT` | 单次 Agent 请求超时，单位为秒 | `90` |
| `AUTOFUSEX_AGENT_JSON_MODE` | 是否向 HTTP 服务请求 JSON mode | 默认开启，设为 `0` 关闭 |
| `AUTOFUSEX_AGENT_NORMALIZE` | 快捷模式执行前是否归一化 | 默认关闭，设为 `1` 或 `true` 开启 |
| `AUTOFUSEX_AGENT_RECOVER` | 快捷模式失败后是否诊断 | 默认关闭，设为 `1` 或 `true` 开启 |

`AUTOFUSEX_AGENT_NORMALIZE` 和 `AUTOFUSEX_AGENT_RECOVER` 只控制是否调用 Agent，不决定使用哪个 Agent。`--agent-*` 命令行参数优先于同名配置环境变量；模型脚本参数仍放在命令末尾的 `--` 之后。

### 使用本地 CANNBot

如果执行模型脚本的环境已经安装 CANNBot，可以让工具启动本地 CANNBot 子进程。CANNBot 需要先完成自身的 provider/OAuth 配置；如果 `cannbot` 已经在 `PATH` 中，工具会自动探测：

```bash
export AUTOFUSEX_AGENT_TRANSPORT=cannbot
export AUTOFUSEX_AGENT_MODEL=cannbot/qwen3.8-flash
```

如果 CANNBot 由 Node 环境提供但没有加入 `PATH`，显式指定可执行文件：

```bash
export AUTOFUSEX_AGENT_TRANSPORT=cannbot
export AUTOFUSEX_AGENT_COMMAND=/home/developer/.nvm/versions/node/v18.19.1/bin/cannbot
```

也可以在命令行传入 `--agent-transport cannbot --agent-command /path/to/cannbot`。工具调用的是 `cannbot run --format json --model MODEL`，解析其 JSON event stream 中的文本结果，然后继续使用同一套归一化/诊断 schema。当前 Agent prompt 明确要求不使用工具，因此 CANNBot 负责分析，模型脚本仍由 AutofuseX 本地执行；不会自动把模型脚本交给 CANNBot 执行。

### 使用 Codex、Claude 或 OpenCode CLI

如果这些工具所在环境提供 OpenAI-compatible HTTP endpoint，直接使用 HTTP 通道，不需要设置 `AUTOFUSEX_AGENT_COMMAND`：

```bash
export AUTOFUSEX_AGENT_TRANSPORT=http
export AUTOFUSEX_AGENT_ENDPOINT=https://your-agent-gateway/v1/chat/completions
export AUTOFUSEX_AGENT_API_KEY='your-api-key'
export AUTOFUSEX_AGENT_MODEL='your-model-name'
```

如果只有 CLI，当前版本不能直接执行任意 CLI。特别是不能仅把 `AUTOFUSEX_AGENT_COMMAND` 改成 `codex`、`claude` 或 `opencode`，因为该变量当前的调用协议固定为 CANNBot CLI 协议。可以增加一个兼容包装器：它接收命令行最后一个参数作为完整 prompt，调用实际 CLI，并将最终的归一化或诊断 JSON 原样写到 stdout。例如：

```bash
#!/usr/bin/env bash
set -euo pipefail
prompt="${!#}"
# 按实际 CLI 的非交互、纯文本输出参数替换下一行。
exec claude -p "$prompt" --output-format text
```

包装器必须满足：不要求交互输入；日志写到 stderr；stdout 最终只包含 Agent 返回的 JSON 对象。然后将包装器作为 `AUTOFUSEX_AGENT_COMMAND`，并使用 `cannbot` 兼容通道：

```bash
export AUTOFUSEX_AGENT_TRANSPORT=cannbot
export AUTOFUSEX_AGENT_COMMAND=/path/to/agent-cli-wrapper
export AUTOFUSEX_AGENT_MODEL=default
```

这是兼容现有 CLI 协议的临时接入方式；如果希望避免包装器，建议将 CLI 接入一个 OpenAI-compatible 网关后使用 `http` 通道。

## 单独归一化

```bash
afx model normalize \
    --script /path/to/model.py \
    --framework pytorch \
    --out /tmp/model_accuracy/agent_normalization.json \
    --agent-transport cannbot \
    --agent-command /home/developer/.nvm/versions/node/v18.19.1/bin/cannbot \
    -- \
    --device npu \
    --device_id 0
```

不指定 `--out` 时，默认输出到当前目录的：

```text
<script_stem>_autofusex/workloads/model/agent_normalization.json
```

执行成功后，JSON 中的 `status` 为 `ready`；Agent 无法确认必要信息时返回 `needs_user_input`，命令仍会保留结果供用户检查，但不会把不确定信息当作事实执行。

## 与模型执行联动

```bash
afx model run \
    --framework auto \
    --backend inductor_autofuse \
    --script /path/to/model.py \
    --out-dir /tmp/model_accuracy \
    --html /tmp/model_accuracy.html \
    --agent-normalize \
    --agent-recover \
    -- \
    --device npu \
    --device_id 0
```

`--agent-normalize` 会在执行前生成 `OUT_DIR/agent_normalization.json`。对于普通 PyTorch 脚本，当前版本会使用其中的 `source.entrypoint.module_pattern` 和 `capture_call` 建议辅助 PyTorch adapter 选择调用目标；模型对象、输入 Tensor 和权重仍由本地现有捕获机制获取，避免 Agent 伪造运行时 Tensor。对于 CANN GE 用例，工具保留 adapter 原生的 `run_af` 入口选择，不采用 Agent 对 `capture_call/module_pattern` 的静态猜测。

显式传入的 `--capture-call` 和 `--module-pattern` 优先于 Agent 建议。未显式传入 `--capture-call` 时，普通 PyTorch 脚本默认值为 `0`；只有 Agent 返回其他合法调用序号时才会采用该建议。CANN GE 脚本始终由原生 `run_af` 适配逻辑选择入口。`fullgraph`、`dynamic`、设备迁移和脚本参数不会被 Agent 静默改写。

## 失败诊断

`--agent-recover` 只有在本地捕获或后端执行失败时才调用 Agent。它会把失败阶段、异常、traceback 和归一化结果发送给 Agent，并将结果保存为：

```text
OUT_DIR/agent_recovery.json
```

恢复 JSON 使用 `autofusex-recovery/1` schema，包含诊断、证据、建议动作和是否适合重试。`script_patch` 仅供用户审阅，当前版本不会自动修改脚本、安装依赖、执行 shell 命令或循环重试。这样可以避免模型建议改变原始模型逻辑或污染运行环境。

## JSON 协议

归一化结果的顶层结构如下：

```json
{
  "schema": "autofusex-normalization/1",
  "status": "ready",
  "framework": "pytorch",
  "source": {
    "script": "/path/to/model.py",
    "project_root": "/path/to/project",
    "imports": ["torch"],
    "entrypoint": {
      "kind": "module_call|create_case|cann_ge_run_af|function|factory_function|script_main|run_af|unknown",
      "symbol": "Model",
      "module_pattern": "Model",
      "capture_call": 0
    }
  },
  "model": {
    "kind": "nn.Module",
    "class_name": "Model",
    "construction": {
      "strategy": "script_runtime",
      "symbol": "Model",
      "location": "model.py:20",
      "arguments": {}
    },
    "training": "eval|train|unknown"
  },
  "graph": {
    "strategy": "runtime_compile|torch_export|script_ir|unknown",
    "forward_symbol": "forward",
    "graph_stage": "inductor_transformed_fx",
    "graph_breaks": "unknown",
    "notes": []
  },
  "inputs": {
    "strategy": "captured_call|create_case|cann_ge_run_af|literal|saved_artifact|unknown",
    "parameters": [],
    "kwargs": [],
    "artifacts": []
  },
  "weights": {
    "strategy": "inline|checkpoint|none|unknown",
    "artifacts": [],
    "load_sites": [],
    "required": false
  },
  "execution": {
    "backend": "inductor_autofuse",
    "device": "npu:0",
    "script_args": [],
    "capture_call": 0,
    "module_pattern": null,
    "fullgraph": false,
    "dynamic": false
  },
  "diagnostics": {
    "confidence": 0.95,
    "assumptions": [],
    "missing": [],
    "candidates": []
  }
}
```

### 字段含义

| 字段 | 作用 |
|---|---|
| `source` | 脚本、工程根目录、导入和入口调用信息。 |
| `model` | `nn.Module`/callable 的构造位置、训练状态和构造参数线索。 |
| `graph` | 模型图的获取方式及其阶段；当前执行主路径使用 Inductor transformed FX。 |
| `inputs` | 输入来自模块调用、`create_case`、Baize `run_af`、字面量还是已保存 artifact。 |
| `weights` | 权重是脚本内构造、checkpoint、无权重还是未知，并记录加载位置线索。 |
| `execution` | 后端、设备、脚本参数和入口选择建议。 |
| `diagnostics` | 置信度、假设、缺失信息和候选入口。 |

Agent 不应在 `inputs` 或 `weights` 中写入完整 Tensor/权重内容，而应返回 shape、dtype、路径、变量名和获取策略等元信息。`source.entrypoint.kind` 和 `inputs.strategy` 保留为非空字符串，允许不同模型框架或 Agent 使用扩展命名；它们不会直接驱动本地执行分支。

## 当前边界和后续演进

当前只对 `pytorch + inductor_autofuse` 提供执行支持，TensorFlow/JAX 仍保留 adapter 接口。后续可以在不改变比较核心的前提下，将 `inputs.strategy`、`weights.strategy` 和 `graph.strategy` 扩展为框架专用实现，并增加经过用户确认的 patch/retry 工作流。
