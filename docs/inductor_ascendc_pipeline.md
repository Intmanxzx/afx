# Inductor + AscendC（Autofuse）编译执行全链路流程

本文梳理从一个 PyTorch 模型脚本出发，经 Dynamo/Inductor、torch_npu 的 ascendc 后端（inductor_npu_ext 同源机制）、Autofuser 编译生成融合 kernel，到运行时通过 wrapper 调用 `AutofuseLaunch` 下发执行的完整流程。

依据代码版本：torch 2.10 + torch_npu 2.10.0.post7（Ascend950PR 环境实测），文中行号对应该版本。

## 总览

```text
用户模型脚本 (torch.nn.Module)
    │  torch.compile(model, backend="inductor")   ← TORCHINDUCTOR_NPU_BACKEND=ascendc
    ▼
[1] TorchDynamo 捕获 → FX Graph（symbolic traced）
    ▼
[2] AOTAutograd / 分解 → aten 子集的 FX GraphModule
    ▼
[3] Inductor: IrGenerator → Scheduler（预融合 IR）
    │   trace 产物: fx_graph_readable.py / ir_pre_fusion.txt / ir_post_fusion.txt / output_code.py
    ▼
[4] NPUScheduling.codegen_nodes（torch_npu/_inductor/ascendc/npu.py）
    │   融合决策(can_fuse) → NPUKernel.tracing_asc() 生成 ASCIR HintGraph
    │   产物: autofused_<ops>_<md5>/asc_graph.py、benchmark.py
    ▼
[5] Autofuser（CANN pyautofuse）: schedule + codegen
    │   产物（内嵌于 output_code.py 的 artifacts 字符串）:
    │     tiling_def / host_impl / device_impl / cpp_wrapper / cpp_tiling_wrapper
    │   DFX 开关(AUTFUSE_DFX_FLAGS)额外产出 GE pbtxt + ascgraph txt
    ▼
[6] async_compile_ascendc（compiler/__init__.py:508）
    │   g++ 编译 wrapper.so；autofuse.compile_adapter.jit_compile 编译 kernel.so
    │   （静态 kernel 走 topn tiling 多版本编译 + tiling.so）
    │   返回 NpuInductorKernel（Python 可调用对象）
    ▼
[7] 运行时: output_code.py 的 call() 直接函数调用
    │   autofused_xxx(tensor..., stream, kernel_key)
    ▼
[8] NpuInductorKernel.__call__ → ctypes → wrapper.so 的 wrapper()
    │   ├─ dlsym(kernel.so, "AutofuseTiling")   tiling 计算
    │   ├─ workspace 分配（NPU caching allocator）
    │   └─ OpCommand::RunOpApiV2 → launch() → kernel.so 的 AutofuseLaunch
    ▼
[9] AutofuseLaunch 内部 <<<blockDim, nullptr, stream>>>(...)（aclrtLaunchKernel）→ NPU 执行
```

## 阶段 1：torch.compile 入口与后端选择

```python
compiled = torch.compile(model, backend="inductor")
```

后端由环境变量决定：`TORCHINDUCTOR_NPU_BACKEND=ascendc`。torch_npu 在 `torch_npu/_inductor/__init__.py` 的 `_BACKEND_LOADERS` 中注册了各后端加载器，import torch_npu 时执行 `_load_backend()`：

```python
_BACKEND_LOADERS = {
    "mlir":  _load_mlir_backend,
    "dvm":   _load_dvm_backend,
    "ascendc": _load_ascendc_backend,   # ← 本路线
    "default": _load_triton_backend,
}
```

`_load_ascendc_backend` 导入 `torch_npu._inductor.ascendc` 包，其 `__init__.py:23` 完成关键注册：

```python
register_backend_for_device("npu", NPUScheduling, NpuWrapperCodeGen, CppWrapperAscendC)
```

把 Inductor 的设备调度器/代码生成器替换为 NPU 版本，同时打一系列补丁（NPU lowering 设备选择、动态 shape 处理、分解白名单等）。

> `inductor_npu_ext`（torchair 仓 experimental 下的独立包）是同一机制的先行版本：同样 `register_backend_for_device("npu", NPUScheduling, NpuWrapperCodeGen)`，只是实现位置和成熟度不同。工具 run_summary 里的 `topology: inductor_npu_ext Autofuse` 即该版本产出。

## 阶段 2-3：Dynamo 捕获与 Inductor 调度（框架原生）

1. **TorchDynamo** 捕获模型 forward 的 Python 字节码，trace 成 FX Graph。
2. **AOTAutograd** 与分解（decomposition）把高阶算子降到 aten 子集，得到 transformed FX GraphModule。
3. **Inductor IrGenerator** 把 FX 节点变成 buffer/kernel 级 IR，**Scheduler** 做缓冲区分析、算子分组与预融合，形成 `SchedulerNode` / `FusedSchedulerNode` 集合。

开启 `TORCH_COMPILE_DEBUG=1` 后，每个 `model__N_inference_M` trace 目录下会落中间产物：

| 文件 | 内容 |
|---|---|
| `fx_graph_readable.py` | 进入 Inductor 的 FX 图（pre-fusion） |
| `ir_pre_fusion.txt` | 预融合调度 IR（SchedulerNode 列表、依赖、writes） |
| `ir_post_fusion.txt` | 融合后调度 IR（FusedSchedulerNode） |
| `output_code.py` | 最终 Python wrapper（见阶段 7） |
| `autofused_*/asc_graph.py`、`benchmark.py` | 每个 ASC 融合 kernel 的图脚本与基准脚本 |

## 阶段 4：NPUScheduling——融合决策与 ASCIR 生成

位置：`torch_npu/_inductor/ascendc/npu.py`。

```python
class NPUScheduling(BaseScheduling):          # npu.py:2335
    def codegen_nodes(self, nodes):           # npu.py:4386
        kernel = NPUKernel(nodes, comments=comments).tracing_asc()
        ...
        kernel_code = kernel.codegen()        # 生成 asc_graph.py 内容
```

1. **融合决策**：`can_fuse`（垂直/水平融合判断，每次决策经 `_log_can_fuse_result` 落日志，即工具 `autofuse_diagnostics.json` 里 fusion_decision_groups 的来源）。能融合的一组 SchedulerNode 交给一个 `NPUKernel`。
2. **ASCIR tracing**：`NPUKernel._tracing_asc_impl`（npu.py:859）把调度节点翻译成 **ASCIR HintGraph**（ascir.ops.Data/Load/Add/Relu/Output…），写出 `autofused_<op列表>_<md5>/asc_graph.py`。md5 由 asc_graph 文本 + cpp_wrapper 计算，保证同构 kernel 去重。
3. **benchmark.py**（`common/precision_check.py` 生成）：包含 FX 片段注释、Original ATen、graph_hint —— 工具 `_benchmark_metadata` 解析的就是它。

无法融合（或显式 fallback）的节点走两条旁路：

- **aten 原生路径**：wrapper 里直接 `torch.ops.aten.xxx(...)`，运行时经 PTA 落到 aclnn 单算子；
- **triton 路径**：由 triton-ascend 生成 `triton_unk_fused_xxx` kernel（`triton_xxx.run(...)` 调用）。

## 阶段 5：Autofuser 编译——从 ASCIR 到 kernel 源码

asc_graph.py 中的典型片段：

```python
autofused_add_relu_xxx = ascir.FusedGraph('autofused_add_relu_xxx')
graph = ascir.ops.AscBackend('graph', graph_hint, autofused_add_relu_xxx)
...
fuser = Autofuser(AutofuserOptions(graph_type=1))
scheduled = fuser.schedule(autofused_add_relu_xxx)      # 调度（多级流水/TQue/TBuf 分配）
tiling_def, host_impl, device_impl = fuser.codegen(scheduled)  # 三段代码生成
```

`fuser.codegen` 返回三段字符串：

| 产物 | 内容 |
|---|---|
| `tiling_def` | C 结构体 `AutofuseTilingData`（block_dim/corenum/各 buffer size/tiling_key 等） |
| `host_impl` | host 侧 tiling 计算函数 `AutofuseTiling()`（按 shape/资源限制算分块参数） |
| `device_impl` | `__aicore__` 设备函数 `autofused_...()`，内部即 `<<<blockDim, nullptr, stream>>>` 形式的 launch 宏（封装 aclrtLaunchKernel） |

加上 npu.py 侧生成的 host 包装（prof 埋点、workspace 分配、stream 获取、符号解析），共同构成 5 个 artifacts 字段：`name / tiling_def / host_impl / cpp_wrapper / cpp_tiling_wrapper`（`is_static` 标记静态 shape）。

**DFX 图 dump**：设 `AUTOFUSE_DFX_FLAGS="--codegen_compile_debug=true"`（可加 `--debug_dir=...`）时，Autofuser 在各优化 pass 前后额外导出 `ge_onnx_*.pbtxt`（GE ONNX proto 图）与 `ascgraph_*_AutoFuse*.txt`（ASCIR 文本图）到 `autofuse_compile_debug/ascgen_dump_pid_<pid>/<kernel>/`。这就是工具 `autofuse_dump/` 与 parse-input 外部图输入的来源。

## 阶段 6：async_compile_ascendc——编译两个 .so

位置：`torch_npu/_inductor/ascendc/compiler/__init__.py:508`。wrapper 的 output_code.py 里每个 kernel 一行：

```python
autofused_xxx = async_compile_ascendc(None, autofused_xxx_artifacts)
```

流程（编译产物落 `TORCHINDUCTOR_NPU_EXT_CACHE_DIR`（默认 `/tmp/torchinductor_root/`）下按 kernel md5 分目录，带 `compile.lock` 文件锁与缓存命中检查）：

```text
artifacts
 ├── compile_wrapper()  → wrapper.so
 │     g++ -shared 包装：common header + tiling_def + debug 支撑 + cpp_wrapper(+cpp_tiling_wrapper)
 │     导出符号: wrapper() / init() / release() / set_device_limit() / set_prof_enabled()
 │
 └── kernel.so  （二选一）
       ├─ JIT 路径（动态 shape / task_queue 模式）:
       │     setup_asc_jit_compile_command → autofuse.compile_adapter.jit_compile
       │     （tiling_def + host_impl + device_impl → kernel.so，导出 AutofuseTiling / AutofuseLaunch）
       └─ 静态 topn 路径（is_static 且启用 autotune）:
             setup_asc_host_compile_command → tiling.so（host tiling 库）
             compile_tiling_wrapper()       → tiling_wrapper.so
             get_topn()                     → 用 tiling.so 穷举/评分 Top-N 个 tiling 方案
             每个 tiling 方案单独 kernel_compile → kernel.so（含 tiling_map 查表,
             运行时按 kernel_key 选静态 kernel, release() 释放未选句柄）
```

缓存命中时直接 `cdll.LoadLibrary(wrapper.so)` + `NpuInductorKernel(...)` 返回。编译可进 Inductor 进程池并行（`compile_threads > 1`）。

## 阶段 7-9：运行时下发执行

### 7. wrapper 代码（output_code.py 的 call()）

执行就是普通 Python 调用序列：

```python
def call(self, args):
    buf0 = torch.ops.aten.mm.default(arg4_1, arg3_1)        # 未融合 → aclnn 单算子
    buf2 = empty_strided((128, 96), (96, 1), device='npu', ...)  # 输出分配
    autofused_add_mul_...(arg1_1, arg2_1, arg4_1, buf1, buf2)    # 融合 kernel → NpuInductorKernel
    return (buf150,)
```

### 8. NpuInductorKernel（compiler/_kernel.py:48）——dlopen 与符号解析

```python
class NpuInductorKernel:
    def load(self, args):                       # 首次调用
        self.dl = cdll.LoadLibrary(self.lib_wrapper)     # ① dlopen wrapper.so（ctypes）
        self.kernel = self.dl.wrapper                   # ② 取 extern "C" int wrapper(...) 符号
        _inject_device_limit(self.dl)                   #    注入 AIV/AIC 核数限制
        self.dl.init(self.lib_kernel.encode(), kernel_key)  # ③ wrapper.so 内部再 dlopen kernel.so

    def __call__(self, *args):
        self.load(args)
        result = self.run(args)                 # 执行

    def run(self, args):
        return self.kernel(                     # ④ 直接调用 C 函数
            *self.to_ctypes_args(args),         #    tensor → device 数据指针
            self.default_stream,                #    torch stream
            kernel_key,                         #    静态 kernel 选择键（JIT 路径为 None）
        )
```

wrapper.so 内部的 `init()`（output_code.py 内嵌源码）完成第二层加载：

```c
extern "C" int init(const char *kernel_file, const char *kernel_key) {
    handle = dlopen(kernel_file, RTLD_NOW | RTLD_LOCAL);              // dlopen kernel.so
    tiling_fn = dlsym(handle, "AutofuseTiling");                      // host tiling 函数
    launch_fn  = reinterpret_cast<LaunchFuncType>(dlsym(handle, "AutofuseLaunch"));  // launch 函数
    // 静态 topn: 按 kernel_key 把多个 tiling 版本的 launch 函数收进 static_kernel_launch_fns
}
```

### 9. wrapper() → AutofuseLaunch → 设备执行

`wrapper()` 按 `TASK_QUEUE_ENABLE` 分派 task_queue2/01 变体，核心路径：

```c
extern "C" int wrapper(void *in_ptr0, ..., void *stream, char *kernel_key) {
    void *current_stream = GetStream(stream);          // 默认 stream=0 时取当前 torch stream
    // ① tiling：静态查 tiling_map；动态调 tiling_fn(&tiling_data, &workspace_size, &block_dim, ...)
    // ② workspace：AllocateWorkspaceTensor(workspace_size, current_stream)  // NPU caching allocator
    // ③ 经 PTA host 队列下发：
    at_npu::native::OpCommand::RunOpApiV2("autofused_xxx", launch_call);
    //    launch_call 内部：
    int64_t r = launch(block_dim, current_stream, asc_input0..., asc_output0, workspace, &tiling_data);
    //    launch == kernel.so 的 AutofuseLaunch，内部：
    //    autofused_xxx<<<blockDim, nullptr, stream>>>(...)   → aclrtLaunchKernel → NPU
}
```

完整调用栈：

```text
Python: autofused_xxx(tensor, ...)                     ← NpuInductorKernel.__call__
  └─ ctypes → wrapper.so: wrapper(ptrs, stream, key)
       ├─ GetStream / tiling_fn 或 tiling_map 查表
       ├─ AllocateWorkspaceTensor
       └─ OpCommand::RunOpApiV2(launch_call)
            └─ launch(...)                              ← dlsym(kernel.so, "AutofuseLaunch")
                 └─ <<<blockDim, nullptr, stream>>>(…)  ← aclrtLaunchKernel
```

- 算子下发到device后的运行和调度（stream队列管理、异步执行、事件同步等）由ACL runtime负责

## 关键结论（精度调试视角）

1. **无 GE 会话**：全链路只在 `aclrtLaunchKernel` 级别下发，kernel 对 CANN 是匿名单次 launch，GE 图 dump / `aclmdlSetDump` / CANN 算子级 datadump 均无受体（实测 `DUMP_GE_GRAPH=2` 零产出）。
2. **每个融合 kernel 的唯一 Python 观测点**是 `NpuInductorKernel.__call__` / `torch._inductor.hooks.intermediate_hook`（wrapper 的 call 流程上）——AutofuseDebugger 的 npu_dump 采集即挂此处；这也是此路线下唯一的逐 kernel 数据采集方式。
3. **GE 图仅存在于编译期 DFX dump**（`AUTOFUSE_DFX_FLAGS`），供离线结构分析，与执行路径无关。
4. **torchair GE 路线（`get_npu_backend`）与本路线互斥**：前者整图进 GE Session（dump_config → `AclmdlSetDump` → GE 逐节点落 dump，数据可采）；后者如上，图与数据都只能靠 DFX dump + hook。

## 附：与工具（AutofuseDebugger）的衔接点

| 流程位置 | 工具衔接 |
|---|---|
| transformed FX GraphModule（阶段 3 截获） | CPU / NPU eager 回放（cpu_dump、npu_eager_dump），`canonical_id` 对齐 |
| ir_pre_fusion.txt / ir_post_fusion.txt | post-fusion ↔ pre-fusion SchedulerNode 映射校验 |
| benchmark.py 的 FX 片段与 graph_hint | 融合子图展示、origin 对齐（`_benchmark_metadata`） |
| intermediate_hook（阶段 7 调用点） | npu_dump 逐 kernel 输出 |
| fusion 决策日志（阶段 4） | unfused_reasons / fusion_decisions |
| AUTOFUSE_DFX_FLAGS 产物（阶段 5） | AutoFuseAfterOptimize 图（TQue/TBuf 生命周期视图）、parse-input 外部图输入 |
