# af-fuzz：用例生成的随机变量参考

这份文档回答一个问题：`af-fuzz` 生成每个用例时，哪些维度是随机的、随机范围
是什么、谁在控制这个范围、怎么从 CLI 配置它。不讲环境搭建（见
`af_fuzz_npu_env.md`）或内部实现细节本身，只讲"作为用户我能调什么、调了会
发生什么"。

## 总览表

| 维度 | 随机范围 / 真实机制 | CLI 配置入口 | 默认值 |
|---|---|---|---|
| op 池（算子种类） | 不是"从白名单随机抽一个子集再生图"，而是生成算法在构图过程中每插入一个新节点时，从**整个白名单**里按内部权重随机选一个算子类型——一个 case 用到哪些算子纯粹看运气，`--max-nodes` 越大越可能覆盖更多种类，但不保证任何特定算子一定出现。白名单分两层：静态登记（`ascendc_opset.py` 里的 `ASCENDC_OPSET_A3`/`_ASCENDC_A5_ONLY_ADDITIONS`，人工确认"理论上支持"）∩ 真实硬件探测确认结果（`probe_ascendc_opset()`，在真实 A3/A5 硬件上逐个编译+执行确认"实际能跑通"，结果增量缓存在 `~/.cache/nnsmith-<version>/...yaml`）。**这一步"能跑通"只证明不报错，不证明走了 ascendc 的融合 lowering**——2026-10-29 起，`probe_ascendc_opset()` 在拿到探测缓存后，还会用 `lowering_dtype_filter.py` 的 `LOWERING_GUARD_TARGETS` 映射表把每个 dtype 组合再跟 `inductor_npu_ext` 真实的 `_LoweringGuard._data` 注册表交叉核对一遍，把"能跑通但其实是纯 fallback"的组合过滤掉（典型例子：`torch.TorchReduceSum` 在没加这层过滤前会把 `complex64`/`complex128`/`uint8`/`int8`/`int16`/`int64`/`float64` 都当成合法候选，但 `aten.sum.dim_IntList` 真实只注册了 `float16`/`float32`/`bfloat16`/`int32`——前面那一串全是沿着 eager fallback 跑通的，从来没进过融合路径）；`matmul`/`Linear` 系列还额外处理了 `_LoweringGuard` 查不出来的 SoC 内部门槛（`mm_lowering.py` 的 `_is_matmul_supported_on_soc()`，A3 上硬性排除，跟 dtype 无关）。只覆盖有直接 `_LoweringGuard` 注册的算子，`core.LayerNorm`/`core.GELU`/`core.LeakyReLU` 这类只能靠 decomposition 达成融合、自己没有注册记录的算子不受这层过滤影响，继续信任探测结果本身。`--op-categories` 在这之后再加一层过滤：把白名单按语义分成 `elewise`/`reduce`/`view`/`gather`/`norm`/`matmul` 六类（`OP_CATEGORIES`），传了就只从指定分类里选，不传就是全量白名单。`ReduceMin`/`ReduceMax` 2026-10-27 起也在白名单内（`ascendc_reduce_min_max_amin_amax.patch`，自动应用，见 `_nnsmith_patch.py`）：NNSmith 原本的 `forward_fn` 是 `x.min(dim).values`/`x.max(dim).values`，会追踪成 `aten.min.dim`/`aten.max.dim`——返回 `(values, indices)` 元组的重载，而 `indices` 恒为 `int64`，永远不在 ascendc 注册的输出 dtype 集合里，导致这俩算子永远 fallback；patch 把 `forward_fn` 改成 `x.amin(dim)`/`x.amax(dim)`，追踪到单 Tensor 输出、无 indices 伴生的 `aten.amin`/`aten.amax`（ascendc 单独注册，A3 起即可融合），语义等价，但能真正走到融合逻辑。`TorchReduceSum`/`ReduceMin`/`ReduceMax`/`ReduceMean` 还支持**多轴归约**（`ascendc_multi_axis_reduce.patch`，自动应用）：`dim` 不再固定是单个 int，而是先随机决定归约的轴数 `k`（`[1, rank]` 内均匀），再用 `weighted_axis_sample`（`weighted_axis_choice` 的不放回版本，边界轴权重更高）无重复选出 `k` 个轴，和原来的单轴选择共享同一套边界加权偏好；`Squeeze`/`ArgMin`/`ArgMax` 语义上要求单轴（`ArgMin`/`ArgMax` 的 PyTorch 原语就是单轴），继续走未改动的单轴 `ReduceBase`，不受影响 | `--soc-tier {A3,A5}`，`--op-categories <逗号分隔分类>` | `--soc-tier` 默认 `A3`；`--op-categories` 默认 `None`（不限制） |
| 节点数（op 数量） | `[2, max_nodes]` 闭区间，由 `case_seed` 驱动、可复现：`run_single_case` 在调用 `model_gen()` 前先用 `case_seed` 重置的全局 RNG 做一次 `random.randint(2, max_nodes)`，抽出这次 case 实际的节点数上限，再传给生成算法（生成算法本身会插入节点直到达到这个上限）。`CaseResult.n_nodes` 记录的是生成完成后图里**真实的**计算节点数，不是这个上限本身 | `--max-nodes`（是**上限**，不是固定值） | `5`（实际节点数在 `[2,5]` 间随机） |
| dtype（每个张量的数据类型） | 不是独立配置项，是两层交集：nnsmith 自身的 `DTYPE_GEN_ALL`（`float16`/`float32`/`float64`/`bfloat16` + `int8`/`int16`/`int32`/`int64`/`uint8` + `complex64`/`complex128`）∩ 当前 `--soc-tier`/`--op-categories` 范围内的算子在真实硬件探测里确认支持的 dtype 组合。图里**第一个节点**的 dtype 用这个交集做一次加权随机（`random_dtype_gen()`：float 类权重是 int/uint/bool 类的 4 倍），之后几乎所有算子都是"输入 dtype = 输出 dtype"直接继承前一个节点的 dtype，只有遇到 `Cast` 这类显式转换 dtype 的算子或图分出独立新分支时才会重新加权随机一次——**不是**每个节点独立随机选 dtype。2026-10-27 前这一步被硬编码成固定 `DType.float32`（`ascendc_init_fp_dtype_diversity.patch` 修复前），导致几乎所有 case 不论 op 池/soc_tier 如何配置，dtype 几乎永远是 `f32`（偶尔 `bool`，来自 `Logical`/`Comparator` 类算子输出 dtype 被结构性硬编码），现已修复 | 无直接参数；间接受 `--soc-tier`/`--op-categories` 影响 | 无（由 op 池+探测结果决定） |
| shape（rank + 每维大小） | rank（张量维度数）：`[0, __MAX_RANK__]`，还要受该算子自身 `inp_ranks`/`out_ranks` 约束进一步收窄；每维大小：默认方法 `symbolic-cinit` 实际走的是非符号化的 `make_random_concrete_placeholder`（不是 z3 求解器约束求解——`SymbolicGen` 的符号化路径只在 `method="symbolic"`，不带 `-cinit` 后缀时才会用，af-fuzz 没用这个方法），在 `concr_ph_dim_rng`（一个 `(low, high)` 区间）内用 `_sample_dim_size()` 采样，分布形态由 `dim_dist_mode` 决定（见下方"`--tensor-scale` 各档位的具体数值"小节）：`uniform`（区间内每个值概率相等）或 `log_normal`（偏向小值，大值概率低但非零） | `--tensor-scale {small,medium,large,16k,32k,64k,128k}` 联动控制 `concr_ph_dim_rng` 和采样分布（`dim_dist_mode`，无独立 CLI 参数，完全由档位决定）；rank 上限单独用环境变量 `NNSMITH_MAX_RANK`（无 CLI 参数） | `--tensor-scale` 默认 `small`（`concr_ph_dim_rng=(1,64)`，`dim_dist_mode="uniform"`）；`NNSMITH_MAX_RANK` 默认 `5` |
| 输入数值分布 | `uniform`：`base + (rand()-0.5) * margin`，默认 `margin=4, base=5`，即 `[3,7)` 区间线性均匀分布；`log_scale`：`sign * 10^exponent`，`exponent` 在 `[-margin, margin)` 均匀采样，即数值覆盖 `±10^-4` 到 `±10^4` 的跨数量级范围，用来测试算子在极端数值（接近零/很大）下的表现；`llm`：模拟真实 LLM 激活值分布，95% 的元素走 `N(0, 0.2²)` 主体分布，另外 5% 的元素（逐元素独立判定）在此基础上额外乘上一个 `[100, 10000)` 区间内独立采样的放大系数，模拟"massive activation"现象（少数元素幅值远超其余元素）；`recommendation`：`uniform[0,1)`，模拟推荐系统输入特征做过 min-max 归一化后的分布 | `--value-dist-mode {llm,log_scale,recommendation,uniform}` | `uniform` |
| 权重数值分布 | 受 `--value-dist-mode` 部分控制：`uniform`/`log_scale` 档位下权重分布和输入分布一样（`random_tensor()` 的 `mode` 分支不区分 `role`）；`llm` 档位权重分布是 `N(0, 0.02²)`（BERT/GPT 风格初始化）；`recommendation` 档位权重分布是 `N(0, 1/√numel)`（`numel` 是该权重张量的元素总数，recommendation embedding 风格初始化）。**但这几个新档位目前只在 `random_tensor()` 函数本身生效**——`af-fuzz` 实际跑的 `TorchModel.refine_weights()` 内部调用 `PracticalHybridSearch.search()` 搜索一组能让模型数值稳定的权重时，其权重采样循环（`nnsmith/materialize/torch/input_gen.py` 的 `search()` 方法）调用 `random_tensor()` 从未传 `mode`/`role`，永远走 `uniform` 分支，不受 `--value-dist-mode` 影响——换言之，`--value-dist-mode` 目前只改变输入的分布，不改变 `af-fuzz` 实际生成的权重分布 | `--value-dist-mode {llm,log_scale,recommendation,uniform}`（对权重搜索循环暂不生效，见左侧说明） | 固定 `uniform` |
| 轴选择（Slice/ReduceBase/Transpose/Unsqueeze/Concat 选轴） | 统一走共用函数 `weighted_axis_choice(ndims, boundary_weight=3)`：边界轴（第一个 `0` 和最后一个 `ndims-1`）被选中的概率是任意内部轴的 3 倍，其余轴之间仍然均匀（`ndims<=1` 固定返回 `0`；`ndims==2` 两个轴都是边界，保持均匀）。`Unsqueeze` 的 `expand_dim` 范围是 `[0, ndims]`（`ndims+1` 个有效插入位置），调用时按 `ndims+1` 加权。`ReduceBase` 对 0 维（标量）输入保留返回 `None`（没有轴可以 reduce），不走这个加权函数。`Transpose` 只对 `dim0` 做加权，`dim1` 仍用原有"偏移取模"方式保证和 `dim0` 不同（避免重试）。`TorchReduceSum`/`ReduceMin`/`ReduceMax`/`ReduceMean`（`MultiAxisReduceBase`）走姊妹函数 `weighted_axis_sample(ndims, k, boundary_weight=3)`：不放回地重复 `k` 次"按剩余候选轴的边界权重抽一个、移出候选集"，保持和单轴版本一致的边界偏好；轴数 `k` 本身在 `[1, rank]` 内均匀随机，且必须在反向推导阶段（只知道输出 rank，还没有输入张量）就决定并缓存，和单轴版本"一次性决定、缓存进 `extra_attrs`"是同一套机制的推广。两个不适用场景：Broadcast（`Add`/`Sub`/`Mul` 等二元算子）没有显式"选轴"步骤，广播完全由 shape 约束决定，唯一的随机性是随机每个输入的 rank（不是选轴），所以不走这个机制；`Transpose` 在 NNSmith 里只支持两轴互换，没有多轴全排列（`torch.permute` 等价物），"多轴顺序变换"这个场景目前无法生成 | 无 CLI 参数，`boundary_weight=3` 硬编码在函数默认值 | `boundary_weight=3`，无法从 CLI 配置 |
| 单张量元素总数上限 | 档位联动派生量，不是独立随机变量 | `--tensor-scale`（或显式 `--max-elem-per-tensor` 覆盖，显式传参优先于档位默认值） | `--tensor-scale` 默认 `small`（`2**16`=65536）；`--max-elem-per-tensor` 默认 `None`（不覆盖档位） |
| 单 case 生成超时 | 档位联动派生量，不是独立随机变量 | `--tensor-scale`（或显式 `--timeout-ms` 覆盖，显式传参优先于档位默认值） | `--tensor-scale` 默认 `small`（`10000` ms）；`--timeout-ms` 默认 `None`（不覆盖档位） |
| 顶层随机种子 | 决定性地控制以上全部随机变量（op 池筛选、节点数、dtype、shape、输入数值、轴选择）：同一个 `--seed` + 同一组生成参数永远生成同一批用例 | `--seed`（批量起始种子）/ `--repro-seed`（单独复现某一个 case） | `--seed` 默认 `None`——`random.Random(None)` 会用系统熵源播种，**每次运行都会生成不同的一批 case**，想要可复现务必显式传一个具体的 `--seed` 值；`--repro-seed` 默认 `None`（不复现，跑批量） |

### `--tensor-scale` 各档位的具体数值

| 档位 | `max_elem_per_tensor` | `concr_ph_dim_rng` | `timeout_ms` | `dim_dist_mode` |
|---|---|---|---|---|
| `small`（默认） | `2**16` = 65536 | `(1, 64)` | `10000` | `uniform` |
| `medium` | `2**20` ≈ 104万 | `(8, 4096)` | `30000` | `uniform` |
| `large` | `2**24` ≈ 1677万 | `(8, 8192)` | `60000` | `uniform` |
| `16k` | `65536` | `(8, 16384)` | `90000` | `log_normal` |
| `32k` | `131072` | `(8, 32768)` | `120000` | `log_normal` |
| `64k` | `262144` | `(8, 65536)` | `180000` | `log_normal` |
| `128k` | `524288` | `(8, 131072)` | `300000` | `log_normal` |

`16k`/`32k`/`64k`/`128k` 和 `small`/`medium`/`large` 的设计目标不同：`small`/`medium`/`large` 是"张量总体规模"从小到大递增（`max_elem_per_tensor` 每档乘 16），`concr_ph_dim_rng` 的上界只是跟着总量被动抬高，内部仍走 nnsmith 原有的均匀分布（`random.randint(l, r)`）；`16k`/`32k`/`64k`/`128k` 反过来，是先确定要测的**单维度上界**（对应大模型长上下文场景的 `seq_len` 典型值），再把 `max_elem_per_tensor` 设成"上界 × 4"（刚好允许 2 个维度同时取到接近上界的值，不是上界的平方——那样会让单 case 直接 OOM/超时），`concr_ph_dim_rng` 的下界固定在 `8`，和 `medium`/`large` 一致；但这四档的每维大小采样改用对数正态分布（`dim_dist_mode="log_normal"`），不再均匀——用户反馈"大 shape 应该是偏门场景，权重该比小 shape 低"，均匀分布会让一半以上的 case 都落在区间上半部分，既拖慢整批生成/编译速度，也不符合真实业务里大多数张量仍是中小规模、只是偶尔出现超大 `seq_len` 的分布特征。

`dim_dist_mode="log_normal"` 的具体实现（`nnsmith/graph_gen.py` 的 `_sample_dim_size()`，patch 文件 `ascendc_dim_size_distribution.patch`）：在对数空间做正态采样再指数还原、截断回 `[l, r]`：

```
log_l, log_r = log(l), log(r)
mu    = log_l + 0.3 * (log_r - log_l)   # 均值偏向下界一侧，但不等于下界本身
sigma = (log_r - log_l) / 3.5
v     = round(exp(gauss(mu, sigma)))
v     = clamp(v, l, r)
```

`mu` 没有直接取 `log_l`：如果均值正好卡在下界，截断之后中位数会直接退化成 `l` 本身（对数正态分布一半以上的概率质量本来就在均值以下），所以刻意让均值往上偏移 30%，让中位数落在区间内部一个合理的小值，而不是死死卡在下界。`mu`/`sigma` 的具体系数（`0.3`、`3.5`）是针对这四档的实际 `(l, r)` 区间（`8~16384` 到 `8~131072`）用模拟方式挑出来的，不是通用公式——若区间数量级发生较大变化，需要重新模拟校验。

A5 实测（`--tensor-scale 32k --op-categories elewise,reduce`，30 case）：全部维度采样里 74.7% ≤ 64，中位数在个位数~两位数量级，p90 在五百左右，没有任何维度超过区间一半（`16384`）——和改动前（均匀分布）相比，大值被明显压低了概率但理论上仍有机会出现（样本量不够大时可能暂时没摸到，不代表上界不可达）。

## 结果报告字段说明

`--out-dir` 下会生成 `summary.json`（结构化数据，`FuzzSummary.to_dict()`
的直接产出）和 `summary.html`（基于 `summary.json` 渲染出的可排序/
可筛选表格，`autofusedebugger/nnsmith_integration/summary_html.py`）。
两者字段含义一致，HTML 只是对 JSON 的可视化封装，少数字段（比如
`n_vars`）只出现在 JSON 里，HTML 表格没有单独展示列。

批次级汇总（`summary.json` 顶层）：

| 字段 | 含义 |
|---|---|
| `total` | 本次批次的 case 总数 |
| `passed` | 通过验证的 case 数（`status == "pass"`） |
| `failed` | 验证不一致/异常的 case 数（`status == "fail"`） |
| `gen_errors` | 生成阶段本身失败的 case 数（`status == "gen_error"`） |

每个 case 的字段（`summary.json` 的 `cases` 数组元素，对应
`fuzz_loop.py` 的 `CaseResult` dataclass）：

| 字段 | HTML 列名 | 含义 |
|---|---|---|
| `index` | `#` | case 在本批次里的序号（从 0 开始），和 `seed` 一起构成 `case_id`（格式 `{index:05d}-{seed}`） |
| `status` | 状态 | 三态之一：`pass`（通过验证）/ `fail`（生成成功但验证不一致或执行异常）/ `gen_error`（图生成阶段本身就失败，没能走到验证这一步） |
| `seed` | Seed | 这个 case 的 `case_seed`，决定性地控制它的全部随机内容（见上面"用例生成的随机变量"总览表）。用 `--repro-seed <这个值>` 可以单独复现这一个 case |
| `n_nodes` | 节点数 | 生成完成后图里**真实的**计算节点数（`ir.n_compute_inst()`），不是 `--max-nodes` 这个上限本身——节点数本身是 `[2, max_nodes]` 区间内随机抽取的。`gen_error` case 没有真正生成出图，这个字段保持默认值 `0`，不代表节点数随机化出了问题 |
| `n_kernels` | Kernel数 | 这个 case 最终真正生成并在 NPU 上执行的 kernel 总数（`len(generated_kernels)`，来自 `autofusedebugger/inductor_autofuse.py` 的 `codegen_nodes` 钩子：调度器每次真正产出一个 kernel 就记一条，`fuzz_loop.py` 的 `run_single_case` 在每个 case 编译+执行前后调用 `reset_autofuse_diagnostics()`/`autofuse_diagnostics()` 采集）。这个数字本身不能单独说明"融合了几次"——`n_nodes`（计算节点数，不含 `Input` 占位符，且不去重）比 `n_kernels` 大，说明至少有节点被融合进了同一个 kernel；`n_nodes == n_kernels` 通常意味着没有发生融合，但也可能是节点本身就没有互相融合的条件，具体原因要看调度器的融合决策记录，不能只凭这一个数字下结论。`gen_error` case 没有走到编译这一步，保持默认值 `0` |
| `n_vars` | （HTML 未展示列，仅 JSON 有） | 图里的变量（张量）总数（`ir.n_var()`），通常比 `n_nodes` 大（每个节点可能产生多个输出张量，加上输入占位符） |
| `duration_s` | 耗时(s) | 这个 case 从开始生成到验证完成（或失败）的总耗时 |
| `symptom` | Symptom | **只有 `status == "fail"` 时才有值**，取自 nnsmith 的 `Symptom` 枚举：`exception`（执行时抛异常）/ `segfault`（进程崩溃）/ `inconsistency`（NPU 和 CPU eager 的输出数值不一致，超出容忍度）/ `timeout`（执行超时）/ `worse_perf`（性能回退，当前 af-fuzz 流程不会触发这个）。`pass`/`gen_error` 的 case 这个字段是 `null`，HTML 里显示为 `-`，这是正常行为，不代表字段没用 |
| `stage` | Stage | **只有 `status == "fail"` 时才有值**，取自 nnsmith 的 `Stage` 枚举：`compilation`（编译阶段）/ `execution`（执行阶段）/ `verification`（数值校验阶段）——结合 `symptom` 能快速定位"错在哪一步、错成什么样"，不用每次都打开 `err.log` |
| `op_types` | 算子集合 | 这个 case 用到的全部算子类名去重排序后的列表（如 `["core.Add", "core.Gather"]`），注意这是"用到了哪些种类"，不是"用了几次"——同一种算子在图里出现多次只算一条 |
| `input_shapes` | Input Shape | `{输入张量名: shape}` 的映射，比如 `{"v0_0": [2, 3, 4]}`。张量名是 nnsmith 内部按 `v{指令id}_{返回值序号}` 规则生成的（不是用户可读的语义名），多个张量在单元格内换行列出，鼠标悬停可看完整内容。`gen_error` case 这个字段是空对象 `{}`，HTML 显示为 `-` |
| `input_dtypes` | （并入 Input Shape 列展示，无独立列） | `{输入张量名: DType.short() 字符串}`，和 `input_shapes` 同一组 key，比如 `{"v0_0": "f32"}`。HTML 把两者合并渲染成 `v0_0: f32[2, 3, 4]` 的形式（`formatShapes()` 的 dtype 前缀），不单独占一列。`gen_error` case 为空对象 `{}` |
| `output_shapes` | Output Shape | 同 `input_shapes`，针对输出张量 |
| `output_dtypes` | （并入 Output Shape 列展示，无独立列） | 同 `input_dtypes`，针对输出张量 |
| `bug_dir` | 复现路径 | 只有触发了落盘（失败 case，或 `--debug`/`--repro-seed` 模式下的 pass case）才有值，指向一个含 `model.pth`/`gir.pkl`/`oracle.pkl`/`err.log`/`repro.py` 的目录，`repro.py` 可以直接 `python3 repro.py` 独立重跑这一个 case。没有落盘的 case（批量模式下的 pass case）这个字段是 `null`，HTML 里这一列显示为 `-` |
| `error_summary` | （HTML 未展示列，仅 JSON 有） | `fail` case 的报错堆栈/不一致详情的截断文本（完整内容在 `bug_dir/err.log` 里），`pass`/`gen_error` case 是 `null` |

## 和 `af_fuzz_npu_env.md` 的关系

这份文档讲"生成算法层面有哪些可调的随机维度"以及"结果报告里每个字段是
什么意思"；`af_fuzz_npu_env.md` 讲"怎么在 A3/A5 NPU pod 上把 `af-fuzz`
跑起来"（环境变量、入口点、并发/内存配置、已知 flaky 现象）。两者独立，
互不重复。
