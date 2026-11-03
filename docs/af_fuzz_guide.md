# af-fuzz 使用指南

## 1. 执行方式

### 1.1 执行命令

```bash
git clone https://gitcode.com/xingzhixiong/Autofuse-Debugger.git
cd Autofuse-Debugger
bash autofusedebugger/nnsmith_integration/install_af_fuzz.sh   # 一次性装好 Autofuse-Debugger 和配套的 nnsmith

source autofusedebugger/nnsmith_integration/npu_env.sh  # 需将其中的 python 库路径和 CANN 包路径换成用例执行环境对应路径
autofusedbg af-fuzz --out-dir /tmp/af_fuzz_run
```

- `install_af_fuzz.sh` 做两件事：`pip install -e` 当前这份 Autofuse-Debugger checkout；再把 nnsmith（af-fuzz 依赖的图生成库，不在 PyPI 上，且必须是打了 ascendc 专用补丁基线的版本）clone 到同级目录 `../nnsmith` 并固定到补丁所对应的 commit（`bc0af42`，不要用更新的版本），然后 `pip install -e`。补丁在 import 时自动生效（见 `_nnsmith_patch.py`），不需要手动 `git apply`。
- 这个脚本只需要跑一次（`pip install -e` 是可编辑安装，装到对应 venv/环境后长期有效）；如果 nnsmith 目录已存在会跳过 clone，重复执行是安全的。但如果 pod 实例本身是动态分配的、配置好的 venv 不持久化，就要在新 pod 上重新跑一遍这个脚本（细节见 `docs/af_fuzz_npu_env.md` 第 3 节）。
- `source npu_env.sh` 每次新开 shell / 新连环境都要重新执行一遍（解决 `import torch`/`torch_npu` 时的库路径问题，环境排坑细节见 `docs/af_fuzz_npu_env.md`）。
- 跑完会在终端打印一行汇总，例如：
  ```
  total=100 passed=97 failed=2 gen_errors=1; summary: /tmp/af_fuzz_run/summary.html
  ```
  进程退出码：`failed == 0` 时为 `0`，否则为 `1`（`gen_error` 不影响退出码）。

### 1.2 参数说明

| 参数 | 默认值 | 可配置范围 / 含义 |
|---|---|---|
| `--out-dir` | **必填** | 结果输出目录（`summary.json`/`summary.html`/失败用例目录都落在这里） |
| `--n-cases` | 功能/精度模式 `100`；性能模式不限制 | 本次会话的用例总数上限，显式指定时为正整数。性能模式不传时持续运行，直到 Ctrl+C、SIGTERM 或会话异常退出；配合 `--resume` 时是不包含重复执行已完成用例的累计总上限，不传则沿用检查点的上限 |
| `--max-nodes` | `5` | 单个用例计算图节点数的**上限**；每个 case 的真实节点数会在 `[2, max-nodes]` 区间内随机抽取，不是固定等于这个值（至少 2 个节点才有融合的意义） |
| `--method` | `"symbolic-cinit"` | nnsmith 图生成算法，一般不需要改 |
| `--tensor-scale` | `"default"` | `default`/`small`/`medium`/`large`/`16k`/`32k`/`64k`/`128k` 八档，每档同时固定 `--timeout-ms`/`--max-elem-per-tensor`/维度取值区间/分布形态，具体数值见本表后面的"`--tensor-scale` 各档具体数值"表；`default` 是 2026-10-29 新增的开箱即用档位，区间 `[1, 131072]` 配合 half-normal 分布（小 shape 常见、大 shape 罕见但可达）；`small` 等价于改动前的硬编码默认行为，保留作回归基线；`16k`/`32k`/`64k`/`128k` 专门用来压测单维度大小本身（长上下文 `seq_len` 场景），上界直接对应档位名的数值 |
| `--timeout-ms` | `None`（跟随 `--tensor-scale` 档位） | 单 case 生成超时（毫秒）。不传时使用 `--tensor-scale` 档位自带的值（见后面的档位数值表），显式传值会覆盖档位值 |
| `--max-elem-per-tensor` | `None`（跟随 `--tensor-scale` 档位） | 单张量最大元素数上限。不传时使用 `--tensor-scale` 档位自带的值（见后面的档位数值表），显式传值会覆盖档位值 |
| `--value-dist-mode` | `"uniform"` | 权重/输入的数值分布档位，四选一：`uniform`（原始 `[3,7)` 均匀分布，默认兼容旧行为）/ `log_scale`（跨 `1e-4~1e4` 量级）/ `llm`（模拟 LLM 激活值分布：主体 `N(0, 0.2²)`，5% 元素随机放大 100~10000 倍，模拟"massive activations"现象）/ `recommendation`（模拟推荐系统归一化特征，`uniform[0,1)`）。具体分布公式见 `docs/af_fuzz_randomization.md` |
| `--seed` | `None`（随机） | 整批的顶层随机种子，决定派生的 `case_seed` 序列；批量复跑须保持 `--jobs` 等分片参数一致。单 case 用 `--repro-seed`，复现条件见第 3 节 |
| `--target` | `"npu"` | 执行目标后端 |
| `--jobs` | `2` | 按进程切分 `--n-cases`，并行跑 N 个进程。每个 worker 真实占用 NPU 编译/执行资源，偏重 CPU+内存，建议先用低值观察负载再往上调；并发调优细节和实测数据见 `docs/af_fuzz_npu_env.md` 第 5 节 |
| `--devices` | `None`（不指定，所有 `--jobs` worker 都跑在 `npu:0` 上） | 逗号分隔的 NPU 卡号列表，比如 `0,1`；把 `--jobs` 分出的每个 worker 按 `shard_idx % len(devices)` 轮询分配到对应的卡上（`--jobs 4 --devices 0,1` 时 shard 0/2 → `npu:0`，shard 1/3 → `npu:1`），让多卡机器真正并发利用所有卡，而不是全部挤在一张卡上排队。只在 `--target npu`（默认）下生效，`cuda`/`cpu` 忽略此参数。多卡用法和实测见 `docs/af_fuzz_npu_env.md` 第 5.3 节 |
| `--compile-threads` | `None`（跟随环境变量 `TORCHINDUCTOR_COMPILE_THREADS`，否则 `4`） | **每个** `--jobs` 进程内部 inductor 并行编译 worker 池大小。inductor 自身默认值是 `min(32, 宿主机 CPU 核数)`，在容器里容易把内存撑爆，所以这里改成了偏保守的默认 `4` |
| `--soc-tier` | `"A3"` | `A3` / `A5` 二选一，决定用哪一代 NPU 的融合算子白名单来约束图生成（见 `ascendc_opset.py`）；在 A5 硬件上验证时要显式传 `A5` |
| `--op-categories` | `None`（不限制，用 `--soc-tier` 对应的全量算子集合） | 逗号分隔的算子分类子集，比如 `elewise,reduce`；传了之后**完全替换**默认的全量算子集合（不是在默认集合基础上再收窄），只会从这些分类里的算子生成用例。可选分类及具体数值见本表后面的"`--op-categories` 分类明细"表。会在 `--soc-tier` 过滤之后再应用，所以只能让生成范围变得更窄，不会出现某个分类在当前 soc-tier 上本来不支持却被生成出来的情况；如果选的分类在当前 soc-tier 上一个算子都没有（比如 A3 下选 `matmul`），会直接报错 |
| `--compare-performance` | 关闭（`False`） | 开启 AscendC 与默认 Triton-Ascend 的性能对比；同一 case 在同一张 NPU 上分别运行两个独立后端进程，比较 profiling 得到的每步 kernel 总耗时。仅支持 NPU，按 case 和后端串行执行（忽略 `--jobs`）；`--devices` 按 case 轮询分配卡号。报告增加“性能对比”和“性能对比汇总”页签，详见第 4 节 |
| `--keep-perf-artifacts` | 关闭（`False`） | 性能模式专用；显式保留原始 profiling 和每次测量的编译缓存。默认在结果持久化后删除这些大体积产物，保留统计 CSV、JSON、日志和 HTML；异常与超时用例同样清理 |
| `--resume` | 关闭（`False`） | 性能模式专用；从 `--out-dir` 的检查点继续运行，跳过已完成用例并重跑被中断的 case。须保持生成参数、预热/采集步数、目标设备与软件版本一致；只支持新版创建的会话 |
| `--warmup-steps` | `3` | 性能模式专用，正整数；每种后端完成编译和首次精度校验后的预热步数，不计入 kernel 耗时 |
| `--profile-steps` | `10` | 性能模式专用，正整数；正式 profiling 的采集步数。每步 kernel 总耗时 = `op_statistic.csv` 中所有算子 `Total Time` 求和 ÷ 实际采集步数 |
| `--perf-timeout-s` | `600` | 性能模式专用，正数，单位秒；每个后端 worker 的总超时，包含进程启动、构图、编译、校验、预热、采集和导出，与 `--timeout-ms` 的构图超时分别控制 |
| `--repro-seed` | `None` | 只跑这一个 `case_seed`，不跑整批（见第 3 节） |
| `--debug` / `--no-debug` | 不传（`None`） | 是否开启调试落盘。只在配合 `--repro-seed` 时生效：不传时 `--repro-seed` 自动当作开了 `--debug`；显式 `--no-debug` 可以在单 case 复现时关掉落盘，只看 pass/fail（见第 3 节） |

`--tensor-scale` 各档具体数值：

| 档位 | `max_elem_per_tensor`（单张量最大元素数） | `concr_ph_dim_rng`（每个维度大小的随机取值区间） | `timeout_ms`（单 case 生成超时） | `dim_dist_mode` |
|---|---|---|---|---|
| `default`（默认） | `524288` | `(1, 131072)` | `300000`（300s） | `half_normal` |
| `small` | `2^16` = 65536 | `(1, 64)` | `10000`（10s） | `uniform` |
| `medium` | `2^20` ≈ 1,048,576 | `(8, 4096)` | `30000`（30s） | `uniform` |
| `large` | `2^24` ≈ 16,777,216 | `(8, 8192)` | `60000`（60s） | `uniform` |
| `16k` | `65536` | `(8, 16384)` | `90000`（90s） | `log_normal` |
| `32k` | `131072` | `(8, 32768)` | `120000`（120s） | `log_normal` |
| `64k` | `262144` | `(8, 65536)` | `180000`（180s） | `log_normal` |
| `128k` | `524288` | `(8, 131072)` | `300000`（300s） | `log_normal` |

`concr_ph_dim_rng=(l, r)` 是输入占位符**每一维**大小的随机取值区间，各维连乘得到总元素数；一旦某一维累乘会超过 `max_elem_per_tensor`，那一维会退化成 `1` 而不是真的生成超大张量。档位越大，单 case 生成的张量可能越大、形状越多样，但耗时和显存/内存占用也越高——批量跑时优先用默认档或 `small`，确认要测大 shape 场景时再切 `medium`/`large`。

`small`/`medium`/`large` 三档在 `[l, r]` 区间内走**均匀分布**采样（`random.randint(l, r)`，区间内每个值概率相等，和改动前的行为一致）。`16k`/`32k`/`64k`/`128k` 四档改成**对数正态分布**：小尺寸的概率明显更高，大尺寸（接近上界）概率很低但仍有机会出现，不是均匀铺开——设计意图是"这四档本来就是为了偶尔探一下大 shape 边界，不应该让一半以上的 case 都扎堆在大值附近，拖慢整批生成/编译速度"。具体效果（A5 实测，`32k` 档 `(8, 32768)` 区间）：中位数在个位数~百位数量级，p90 在千级，极少数会摸到上万甚至接近 `32768` 上界。`16k`/`32k`/`64k`/`128k` 是专门为长上下文（`seq_len` 4K~128K）场景设计的，目标是偏重压测**单维度**能不能长到这么大，不是让"整张张量的总元素数"同步变大：`max_elem_per_tensor` 固定取"维度上界 × 4"（不是上界的平方），意思是允许大概 2 个维度同时取到接近上界的大值，其余维度仍会被 `max_elem_per_tensor` 的总量约束压缩成 `1`——如果把 `max_elem_per_tensor` 设成上界的平方，单 case 会瞬间 OOM/超时。

`default` 档（2026-10-29 新增，现在的 `--tensor-scale` 默认值）走**半正态分布**（`dim_dist_mode="half_normal"`）：和 `log_normal` 在对数空间有一个内部峰值不同，half-normal 的密度从区间下界 `1` 开始就是最高点，然后单调递减到上界 `131072`，没有任何内部峰值——即"越小越常见，越大越少见"是严格单调的，不会在中间某个值上扎堆。上界/超时直接复用 `128k` 档已验证过的数值（`max_elem_per_tensor=524288`，`timeout_ms=300000`），下界改成 `1`（而不是 `16k`/`32k`/`64k`/`128k` 的 `8`），因为这个档位不针对特定的长上下文场景，只是给默认批次更广的覆盖面。A5 实测（100000 次采样）：中位数约 2.9 万，p90 约 7.2 万，恰好摸到上界 `131072` 的概率约 0.3%（罕见但可达），分桶统计显示每个桶的样本数随桶起点增大严格递减。

`--op-categories` 分类明细（具体算子列表以 `autofusedebugger/nnsmith_integration/ascendc_opset.py` 的 `OP_CATEGORIES` 为准，这里列的是分类含义和代表性算子；每个分类都已限定在 `--soc-tier` 对应的融合算子白名单内，不含白名单之外的算子）：

| 分类 | 含义 | 代表算子（不含 `Input`/`Constant` 这类图生成器自带的叶子节点，它们不受 `--op-categories` 限制） |
|---|---|---|
| `elewise` | 逐元素算子：输出每个位置只依赖输入同位置的值，二元广播算子（Add/Sub/...）也算在内 | `Add`/`Sub`/`Mul`/`Div`/`Pow`/`Abs`/`Neg`/`Sigmoid`/`Clip`/`Sqrt`/`Equal`/`And`/`Or`/`Where`/`Sin`/`Cos`/`Tan`/`GELU` 等 |
| `reduce` | 归约算子：按某个轴把多个位置合并成更少的输出。`TorchReduceSum`/`ReduceMin`/`ReduceMax`/`ReduceMean` 支持单轴或多轴归约（`dim` 是单个 int 或 int 列表，轴数随机，详见 `ascendc_multi_axis_reduce.patch`）；`ArgMin`/`ArgMax`/`Squeeze` 语义上只支持单轴，不受影响 | `TorchReduceSum`/`ReduceMean`/`ReduceMin`/`ReduceMax` |
| `view` | 纯形状/布局变换，不做算术运算 | `Reshape`/`Transpose`/`Squeeze`/`Concat`/`Cast`/`Slice`/`ExpandLast1~4`/`Tril`/`Triu` |
| `gather` | 索引/查表类算子，输出位置由单独的 index 张量决定 | `IndexSelect`/`Gather` |
| `norm` | 归一化复合算子，内部是 reduce+elewise 的组合，单独归一类 | `LayerNorm` |
| `matmul` | 矩阵乘法类，只在 A5 上才会真正融合（A3 上 ascendc 的 matmul lowering 内部会判断 SoC 版本直接 fallback） | `MatMul`/`Linear` |

用户场景示例：只想生成"elewise + reduce"组合的用例，用：

```bash
autofusedbg af-fuzz --out-dir /tmp/af_fuzz_run --op-categories elewise,reduce
```

### 1.3 dtype 是怎么生成的

每个 case 用什么 dtype 不是"固定 fp32"也不是完全均匀随机，是加权随机：候选集是 `--soc-tier`/`--op-categories` 对应的算子实际支持的 dtype 并集（见 `ascendc_opset.py` 的探测缓存），float 类（f16/f32/f64/bf16）权重是 int/uint/bool 类的 4 倍，但两边都有机会被抽到——一个 case 内所有张量的 dtype 由图的第一个节点决定后沿图继承（大多数算子是"输入 dtype = 输出 dtype"），只有遇到 `Cast` 这类显式允许转换 dtype 的算子，或者图里出现独立的新分支，才会重新随机换一次。

**这里的候选集口径 2026-10-29 起变了一次**：之前"候选集"只看探测缓存——某个 dtype 组合只要编译+执行不报错（哪怕是走 eager fallback，根本没碰到 ascendc 的融合 lowering）就会被当成合法候选，典型例子是 `torch.TorchReduceSum`：`aten.sum.dim_IntList` 真实只注册了 `float16`/`float32`/`bfloat16`/`int32` 四种，但探测缓存里曾经出现过 `complex64`/`complex128`/`uint8`/`int8`/`int16`/`int64`/`float64` 一共 10 种组合——后面这 7 种全是沿 eager fallback 跑通的，从来没有走过融合，被抽到也只是在拖低"真实验证了融合能力"的 case 占比。现在 `probe_ascendc_opset()` 会把探测结果再跟 `inductor_npu_ext` 真实的 `_LoweringGuard` 注册表交叉核对一遍（见 `lowering_dtype_filter.py`），只保留真的在融合白名单里的组合；`matmul`/`Linear` 额外处理了 `_LoweringGuard` 查不出来的 A3 SoC 门槛。只覆盖有直接 `_LoweringGuard` 注册的算子——`core.LayerNorm`/`core.GELU`/`core.LeakyReLU` 这类只能靠 decomposition 达成融合、自己没注册记录的算子不受影响，继续信任探测结果。

实测（此修复前）：同样的 `--op-categories elewise,reduce`，40 个 case 的 dtype 分布是 `{f32: 30, f64: 27, f16: 17, i8: 12, i16: 8, i32: 7, i64: 6, bool: 5, u8: 2, c128: 2}`——这份分布里的 `f64`/`c128` 占比明显偏高，现在已知主要是上面说的"探测缓存混入了纯 fallback 组合"造成的，不代表这些算子真的有这么高比例的 case 在融合路径上用到这些 dtype；`bf16` 目前暂未出现，是因为这批算子在当前 A5 环境的探测结果里本身就没有 bf16 组合（不是生成逻辑的问题，是 opset 探测阶段的已知限制）。

## 2. 输出结果

### 2.1 输出文件

`--out-dir` 目录下固定产出：

| 文件/目录 | 内容 |
|---|---|
| `summary.json` | 本批次全部 case 的结构化结果（`total`/`passed`/`failed`/`gen_errors` 汇总 + 每条 `CaseResult`） |
| `summary.html` | 可视化报告，默认包含「汇总统计」（默认打开，图表总览）和「用例列表」（逐 case 表格），字段见下面 2.2/2.3；开启 `--compare-performance` 后增加「性能对比」和「性能对比汇总」两个页签 |
| `bug-<index>-<case_seed>/` | 每个 **失败**（`status == "fail"`）用例单独落盘的目录，内容见下表；`pass`/`gen_error` 的 case 默认不落盘 |

`bug-<index>-<case_seed>/` 目录内容：

| 文件 | 内容 |
|---|---|
| `model.pth` | 模型权重（`state_dict`） |
| `gir.pkl` | 图结构（nnsmith `GraphIR`，pickle 序列化） |
| `oracle.pkl` | 期望输入/输出（CPU eager 跑出来的基准值） |
| `err.log` | 失败时的异常堆栈/日志 |
| `meta.json` | `system`/`symptom`/`stage`/`version` 等元信息 |
| `repro.py` | 可直接运行的单 case 复现脚本 |

### 2.2 「汇总统计」页签

打开 `summary.html` 默认看到的页签，五张图，都是基于当前批次全部 case 的 `op_types`/`input_shapes`/`output_shapes`/`input_dtypes`/`output_dtypes`/`status` 在浏览器端实时聚合出来的（不需要额外的命令行参数，每次生成报告都自动带上）：

| 图 | 统计口径 |
|---|---|
| 总体 pass/fail/gen_error | 环形图，中心是 pass rate 大数字；三态计数 + 占比在图例 |
| 按算子分类 | 竖直堆叠柱状图，按 `ascendc_opset.py` 的 `OP_CATEGORIES`（`elewise`/`reduce`/`view`/`gather`/`norm`/`matmul`）分组统计每类的 pass/fail/gen_error 数。**一个 case 可能同时落入多个分类**（比如一张图里同时有 `Add` 和 `ReduceMin`，会同时计入 `elewise` 和 `reduce`），所以各分类计数之和可能大于 `total`；算子不属于任何已知分类时归到「(未分类)」 |
| 按 shape 档位（最大单维度值） | 把 case 所有 input/output 张量形状里最大的单个维度值分到 `≤8`/`≤64`/`≤512`/`≤4096`/`>4096` 五档统计 |
| 按 shape 档位（最大元素数） | 同上，但用张量形状各维度的乘积（元素总数）分档：`≤1K`/`≤64K`/`≤1M`/`≤64M`/`>64M` |
| 按 dtype | 汇总 case 所有 input/output 张量出现过的 dtype（去重），一个 case 可能同时出现多种 dtype（比如混合精度图），同样会被重复计入多个 dtype 桶 |

这五张图都是纯前端 JS 计算（`summary_html.py` 里的 `renderStats()`），刷新页面/切换页签不需要重新生成报告。

图表配色用的是经过色觉障碍（CVD）安全验证的三色（pass `#12805e` / fail `#d03b3b` / gen_error `#6250d6`，与「用例列表」表格里状态徽章的颜色是两套独立配色）；柱状图为细柱 + 段间留白 + 整洁刻度 y 轴，悬停/Tab 键聚焦任一柱段会弹出该分类的 pass/fail/gen_error 细分浮层。

### 2.3 「用例列表」页签：字段说明

| 列名 | 对应字段 | 含义 |
|---|---|---|
| `#` | `index` | case 在本批次里的序号（从 0 开始） |
| 状态 | `status` | `pass`（通过验证）/ `fail`（生成成功但验证不一致或执行异常）/ `gen_error`（图生成阶段本身失败，没走到验证） |
| Seed | `seed` | 这个 case 的 `case_seed`，控制图结构、权重和输入的生成；用 `--repro-seed <这个值>` 单独复现，条件见第 3 节 |
| 节点数 | `n_nodes` | 生成模型中的计算节点数（不含 `Input` 占位符，`gen_error` case 为 `0`） |
| Kernel数 | `n_kernels` | 这个 case 最终真正生成并在 NPU 上执行的 kernel 总数。**这个数字本身不能单独说明"有没有发生融合"**：`n_nodes` 比 `n_kernels` 大，说明至少有节点被融合进了同一个 kernel；`n_nodes == n_kernels` 通常意味着没有发生融合，但也可能是节点之间本身就没有可融合的依赖关系（比如两条独立并行的分支），不能只凭这一个数字下结论。`gen_error` case 为 `0` |
| 耗时(s) | `duration_s` | 这个 case 从开始生成到验证完成（或失败）的总耗时 |
| Symptom | `symptom` | 只有 `fail` 才有值：`exception`（执行时抛异常）/ `segfault`（进程崩溃）/ `inconsistency`（NPU 和 CPU eager 输出数值不一致）/ `timeout`（执行超时）/ `worse_perf`（性能回退，当前流程不触发） |
| Stage | `stage` | 只有 `fail` 才有值：`compilation`（编译阶段）/ `execution`（执行阶段）/ `verification`（数值校验阶段） |
| 算子集合 | `op_types` | 这个 case 用到的算子类名去重列表（同一种算子出现多次只算一条） |
| Input Shape | `input_shapes` + `input_dtypes` | 显示成 `张量名: dtype[shape]`（例如 `v2_0: f32[4, 8]`），dtype 是 `DType.short()` 缩写（`f32`/`f16`/`bf16`/`i8`/`i32`/`u8`/`b`=bool/`c128`=complex128 等）；`gen_error` case 为空 |
| Output Shape | `output_shapes` + `output_dtypes` | 同上，针对输出张量 |
| 复现路径 | `bug_dir` | 失败用例的 `repro.py` 链接；`pass`/`gen_error` 没有 |

点击表头可排序，顶部可按算子名子串 / 状态筛选。

## 3. 失败用例如何复跑

### 3.1 复跑方式

从 `summary.html`（或 `summary.json` 的 `case_id`/`seed` 字段）拿到失败 case 的 `case_seed`，用 `--repro-seed` 单独重跑：

```bash
autofusedbg af-fuzz --out-dir /tmp/repro --target npu \
  --repro-seed 389609433 --max-nodes 4 --no-debug
```

- OP 使用 `random.choice` 均匀采样，**均匀分布也受 seed 控制**。构图开始时会用 `case_seed` 重置随机数状态；OP 候选按名称排序，探测缓存中的输入/输出 dtype 组合按稳定顺序成对排序，rank 候选也固定顺序，因此同一个随机下标不会因为进程或集合遍历顺序变化而指向另一个候选。不需要额外设置 `PYTHONHASHSEED`。
- 用 `--repro-seed` 重新生成完整 case（图结构、权重、输入及 CPU oracle）时，生成参数（`--method`/`--max-nodes`/`--timeout-ms`/`--max-elem-per-tensor`/`--target`/`--tensor-scale`/`--value-dist-mode`/`--soc-tier`/**`--op-categories`**）须和原批次一致，`NNSMITH_MAX_RANK` 等生成环境、软件版本及探测得到的 OP/dtype 候选集合也须一致。seed 不能替代这些配置；跨版本或候选集合改变时不保证生成相同的图。
- 构图达到时间上限时直接记录 `gen_error`，不再把截止时的部分图当作成功用例；Z3 返回 `unknown`（例如求解超时）也记为 `gen_error`，避免受负载影响而跳过当前 OP 后生成另一张图。繁忙环境可能出现生成失败，但不应把它解释为 seed 对应另一张有效图。
- **新生成的 case 只需保存 `case_seed`，不依赖 `repro.py`、模型结构、权重或输入文件。** 每次构图使用独立的 Z3 上下文，避免前序 case 的求解历史改变 reshape 等算子的解。数值生成前重新设置 Python/NumPy/Torch seed，在 CPU 单线程、Torch 确定性算法模式下创建模型、权重、输入和 CPU oracle；完成后恢复调用方的线程及确定性算法设置。
- 数值稳定性搜索改为最多 **2 组采样，每组最多 16 步梯度修正**。提前结束由数值结果决定，不再使用原来的 20 ms / `NNSMITH_GRAD_TIME` 时间预算，因此日志、机器负载或先前运行过哪些 case 不会改变最终权重和输入。没有确定性实现的操作直接记为 `gen_error`，不降级到非确定性生成。
- `--no-debug` 下 pass/fail 均不落盘模型、权重、输入、图结构或 `repro.py`。`--debug` 仍可用于主动导出诊断文件；这些文件不是 seed 复现的依赖。
- 此保证针对当前确定性生成版本。旧版本的候选顺序、求解器上下文或搜索耗时已经影响过的 case，不能由新版根据同一个 seed 反推出旧产物；升级生成算法也可能改变 seed 的映射。
- 复跑只跑这一个 case，终端会打印：
  ```
  case_id=00000-389609433 status=fail symptom=inconsistency stage=verification
  ```
- 若主动保留了导出文件，也可以直接运行失败目录里的 `repro.py`（`bug-<index>-<case_seed>/repro.py`），不需要重新走 `--repro-seed`。

### 3.2 如何开启 debug 日志、落盘 dump 图

`--repro-seed` 默认就是开着 debug dump 的（等价于自动加了 `--debug`），会把以下内容落到 `--out-dir` 下的 `case-<case_id>/` 目录：

| 内容 | 说明 |
|---|---|
| `model.pth` / `gir.pkl` / `oracle.pkl` / `repro.py` | 和失败用例目录同一套产物，无论这次复现是 pass 还是 fail 都会落盘 |
| `torch_compile_debug/` | `TORCH_COMPILE_DEBUG=1` + `TORCH_LOGS=+dynamo,+inductor,+graph_breaks` 打开后 TorchInductor 自己的调度/编译日志（注意：这个环境变量在部分环境下需要在**进程启动前**设置才会生效，运行期临时设置可能不产出这个目录——若需要完整 FX 图 dump，建议改用 `inductor_cache/` 下生成的 kernel 源码，里面带有完整的 "Graph fragment" 注释） |
| `inductor_cache/` | 本次复现独立使用的 inductor 编译缓存目录，包含真正生成的 kernel 源码（可以直接看到 FX 图结构和 codegen 细节） |
| `autofuse_dump/` | autofuse 侧的图优化过程 dump（`AUTOFUSE_DFX_FLAGS` 控制），每个真正走完融合流程的 kernel 会在这里留下 `ascgraph_*`/`ge_onnx_*` 系列文件（从 `AutoFuseBeforeOptimize` 到 `AutoFuseAfterOptimize` 各个阶段），`ascgraph_*AutoFuseAfterOptimize*.txt` 的存在与否是确认这个 case 是否真正走到了 autofuse 融合 codegen（而不是静默 fallback 到逐算子执行或 eager）的直接依据 |

这些产物各自落在该 case 自己的子目录下，不会和其它 case 混在一起。

传 `--no-debug` 可以在复现时关掉上述全部落盘，只看 pass/fail 结果：

```bash
autofusedbg af-fuzz --out-dir /tmp/repro --target npu \
  --repro-seed 389609433 --no-debug
```

批量跑（不带 `--repro-seed`）时不受这个开关影响：只有失败用例才落盘 `bug-<index>-<case_seed>/`，不会打开上面这些 debug 环境变量——避免整批几十上百个 case 都落一份 inductor debug 目录把磁盘撑爆。

## 4. 后端性能对比

使用 `af-fuzz --compare-performance` 对比同一 seed 下 AscendC 和默认 Triton-Ascend 的每步 kernel 总耗时，默认预热 3 步、采集 10 步。例如：

```bash
autofusedbg af-fuzz --compare-performance \
  --out-dir /tmp/af_perf_run --n-cases 10 --seed 42 \
  --target npu:0 --soc-tier A3 \
  --warmup-steps 3 --profile-steps 10 --perf-timeout-s 600
```

性能模式在运行期间采集并导出 profiling，默认在保存统计结果后清理原始 profiling 和每次测量的编译缓存；`--keep-perf-artifacts` 可保留完整产物，`--debug` / `--no-debug` 不控制这项策略。

持续运行时省略 `--n-cases`：

```bash
autofusedbg af-fuzz --compare-performance --out-dir /tmp/af_perf_continuous --seed 42
```

Ctrl+C 或 `kill -TERM <主进程PID>` 会停止当前测量及其子进程，并保留已完成用例的 `summary.json` / `summary.html`。未完成的 case 不计入完成数。使用相同配置加 `--resume` 可继续：

```bash
autofusedbg af-fuzz --compare-performance --out-dir /tmp/af_perf_continuous --seed 42 --resume
```

报告和检查点逐例原子保存；SIGKILL（`kill -9`）或断电无法执行退出回调，可读取最后一次保存的报告，并在下次 `--resume` 时恢复。具体口径、参数和模块结构见 [性能对比指南](af_fuzz_performance.md)。
