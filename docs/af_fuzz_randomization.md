# af-fuzz：用例生成的随机变量参考

这份文档回答一个问题：`af-fuzz` 生成每个用例时，哪些维度是随机的、随机范围
是什么、谁在控制这个范围、怎么从 CLI 配置它。不讲环境搭建（见
`af_fuzz_npu_env.md`）或内部实现细节本身，只讲"作为用户我能调什么、调了会
发生什么"。

## 总览表

| 维度 | 随机范围 | 由什么决定 | CLI 配置入口 |
|---|---|---|---|
| op 池（算子种类） | 固定白名单内的算子集合（soc-tier 不同，数量不同） | `ascendc_opset.py` 里 `ASCENDC_OPSET_A3`/`ASCENDC_OPSET_A5` 静态白名单 ∩ 真实硬件探测确认结果 | `--soc-tier {A3,A5}` |
| 节点数（op 数量） | `[2, max_nodes]` 闭区间，`case_seed` 驱动、可复现 | `run_single_case` 内部 `random.randint(2, max_nodes)` | `--max-nodes`（现在是**上限**，不是固定值） |
| dtype（每个张量的数据类型） | nnsmith 全集 `DTYPE_GEN_ALL` ∩ 该算子探测确认支持的 dtype 组合 | `BaseGen.__init__` 做交集，不是独立随机变量 | 无直接参数；间接受 `--soc-tier` 影响 |
| shape（rank + 每维大小） | rank：`[0, __MAX_RANK__]`（默认 5）；每维大小：档位区间内 z3 求解 | 算子自身的 `inp_ranks`/`out_ranks` 约束 + `concr_ph_dim_rng` | `--tensor-scale {small,medium,large}`；rank 上限另需设环境变量 `NNSMITH_MAX_RANK`（无 CLI 参数） |
| 输入数值分布 | `uniform`：`[3,7)` 线性区间；`log_scale`：`±10^[-4,4)` 跨数量级 | `random_tensor(mode=...)` | `--value-dist-mode {uniform,log_scale}` |
| 权重数值分布 | 固定 `uniform [3,7)`，**不受 `--value-dist-mode` 控制** | 同一个 `random_tensor()`，调用时未传 `mode` | 无——设计上的范围限定，不是遗漏 |
| 轴选择（Slice/ReduceBase/Transpose/Unsqueeze/Concat） | 边界轴（首/末）概率是内部轴的 3 倍 | `weighted_axis_choice(ndims, boundary_weight=3)` | 无 CLI 参数，`boundary_weight=3` 硬编码在函数默认值 |
| 单张量元素总数上限 | 档位联动派生量 | 同 `--tensor-scale` | `--tensor-scale`（或显式 `--max-elem-per-tensor` 覆盖） |
| 单 case 生成超时 | 档位联动派生量 | 同 `--tensor-scale` | `--tensor-scale`（或显式 `--timeout-ms` 覆盖） |
| 顶层随机种子 | 用户指定或自动生成 | 驱动以上全部随机性的可复现性来源 | `--seed`（批量起始种子）/ `--repro-seed`（单 case 复现） |

## 逐项说明

### 1. op 池（算子种类）

不是"从白名单随机抽一个子集再生图"，而是生成算法在构图过程中每插入一个新
节点时，从**整个白名单**里按内部权重随机选一个算子类型。所以一个 case 用
到哪些算子纯粹看运气：`--max-nodes` 越大、越可能覆盖更多算子种类，但不
保证任何特定算子一定出现。

白名单本身分两层：

1. **静态登记**（`ascendc_opset.py` 里 `ASCENDC_OPSET_A3`/
   `_ASCENDC_A5_ONLY_ADDITIONS`）：人工读 ascendc 的
   `_LoweringGuard.support()` 注册代码确认"理论上支持"的算子。
2. **真实硬件探测**（`probe_ascendc_opset()`）：对静态登记里的每个算子，
   在真实 A3/A5 硬件上逐个单独跑一次编译+执行，确认"不仅注册了、实际也
   真的能跑通"，结果缓存到
   `~/.cache/nnsmith-<version>/TorchModelCPU_ascendc_<ver>_npu_ascendc_<tier>.yaml`。

最终可用 op 池 = 两层的交集。**缓存支持增量更新**：往静态白名单加新算子
后，下次调用只会对新算子跑探测，不需要删除整个缓存文件重新探测全部
~47 个算子。

已通过真实 A5 硬件探测确认的新增算子（2026-10-26）：`core.Gather`、
`core.LayerNorm`（后者通过 `decomposition.py` 的分解路径融合，不是直接
lowering 注册）。

**已知的探测波动性**：某个算子的探测结果在不同调用之间可能翻转（比如
`core.Reshape` 曾经缺失、后来又出现），根因是 nnsmith 自身一个已知 bug
（`Reshape.type_transfer()` 在 z3 模型给多个 `target_shape` 槄赋 `-1` 时
触发断言失败），这个 bug 只对探测过程里随机构造的部分单算子 IR 触发，不
是每次都稳定复现。如果某个算子在没有改动白名单/环境的情况下跨批次忽隐
忽现，先怀疑这类求解器相关的偶发性，不要假设缓存机制本身坏了。

### 2. 节点数（op 数量）

2026-10-26 之前，`--max-nodes` 是生成算法的硬性目标——`abstract_gen()`
的循环条件是 `while num_op() < max_node_size: ...`，意味着几乎每个 case
都固定恰好有 `max_nodes` 个节点，节点数本身完全不随机。

现在改为：`run_single_case` 在调用 `model_gen()` 之前，先 `set_seed
(case_seed)`，再 `random.randint(2, max_nodes)` 抽一个这次 case 实际要
用的节点数上限（下限固定 2，因为至少两个节点才有融合意义），再把这个
抽到的具体值传给 `model_gen(max_nodes=<抽到的值>)`。

- 这次抽取用的是 `case_seed` 已经重置过的全局 RNG，而 `model_gen()` 内部
  的 `BaseGen.__init__` 会对同一个 `case_seed` 再做一次完全重置——两次
  `set_seed(case_seed)` 之间不管消耗了多少次随机数，重置后的状态都一样，
  所以这次抽取不影响图结构生成的确定性，`--repro-seed` 的复现保证不受
  影响：同一个 `case_seed` 每次都抽到同一个节点数上限。
- `CaseResult.n_nodes` 记录的是生成完成后图里**真实的**计算节点数
  （`ir.n_compute_inst()`），不是这次抽到的上限本身——如果生成过程本身
  失败（`gen_error`），`n_nodes` 保持默认值 `0`，这不代表节点数随机化
  违反了下限 2，只是这个 case 从未真正生成出图。

### 3. dtype（数据类型）

不是独立配置项。每个张量的 dtype 来自两层交集：

1. nnsmith 自身的 `DTYPE_GEN_ALL`（patch 后：`float16`/`float32`/
   `float64`/`bfloat16` + `int8`/`int16`/`int32`/`int64`/`uint8` +
   `complex64`/`complex128`）。
2. 实际被选中的算子，在真实硬件探测里确认支持的 dtype 组合
   （`BaseGen.__init__` 做交集，取并集再取交集，具体见 `graph_gen.py`）。

没有单独的 `--dtype` 参数——想控制 dtype 范围，只能通过改白名单（影响第 2
层）或设 `NNSMITH_MAX_RANK` 之类的环境变量间接影响。

### 4. shape（rank + 每维大小）

- **rank（张量维度数）**：`[0, __MAX_RANK__]`，`__MAX_RANK__` 默认 `5`，
  可用环境变量 `NNSMITH_MAX_RANK` 覆盖（没有对应的 `af-fuzz` CLI 参数，
  需要在跑之前 `export NNSMITH_MAX_RANK=<值>`）。具体某个张量的 rank 还
  要受它所在算子的 `inp_ranks`/`out_ranks` 约束进一步收窄。
- **每维大小**：在 `concr_ph_dim_rng`（一个 `(low, high)` 区间）内由 z3
  求解器约束求解决定，不是简单均匀采样。

`concr_ph_dim_rng` 本身由 `--tensor-scale` 档位决定，不单独开放 CLI 参数：

| 档位 | `max_elem_per_tensor` | `concr_ph_dim_rng` | `timeout_ms` |
|---|---|---|---|
| `small`（默认） | `2**16` = 65536 | `(1, 64)` | `10000` |
| `medium` | `2**20` ≈ 104万 | `(8, 4096)` | `30000` |
| `large` | `2**24` ≈ 1677万 | `(8, 8192)` | `60000` |

`small` 档位的值和改动前的硬编码默认值完全一致，保证不传 `--tensor-scale`
时行为不变。

**可以用 `--max-elem-per-tensor`/`--timeout-ms` 单独覆盖档位里的对应值**
（显式传参优先于档位默认值），但 `concr_ph_dim_rng` 本身没有独立的
CLI 参数，只能通过切换档位整体改变。

**已修复的 bug（2026-10-26）**：`--tensor-scale` 曾经只对 `concr_ph_dim_
rng` 真正生效，`max_elem_per_tensor`/`timeout_ms` 这两项实际不受影响。
根因是 `cli.py` 里 `--max-elem-per-tensor`/`--timeout-ms` 这两个 argparse
参数的 `default=` 当时写死成具体数字（`2**16`/`10000`），不是 `None`——
`run_single_case` 内部"只有传入值是 `None` 才用档位默认值覆盖"这条回退
逻辑因此永远不会触发，不管选哪个档位，这两项实际传进去的永远是 argparse
那两个硬编码的旧默认值，和 `small` 档位刚好一样，所以只有切到 `medium`/
`large` 才会看出差异（`concr_ph_dim_rng` 变了，但张量实际大小/超时没跟
着变，容易被误以为"档位没完全生效"）。现在这两个参数的 `default` 已经
改成 `None`，和 `--max-nodes` 等其他档位相关参数的处理方式一致：不传时
让档位说话，传了才覆盖。已在 A5 上实测验证：`--tensor-scale medium` 现在
确实让 `model_gen()` 收到 `max_elem_per_tensor=2**20`、`timeout_ms=30000`，
不再被旧默认值盖掉。

### 5. 输入数值分布

`--value-dist-mode {uniform,log_scale}`（默认 `uniform`）：

- `uniform`：`base + (rand()-0.5) * margin`，默认 `margin=4, base=5`，即
  `[3,7)` 区间线性均匀分布——这是改动前的原始行为，默认值完全不变。
- `log_scale`：`sign * 10^exponent`，`exponent` 在 `[-margin, margin)`
  区间均匀采样（默认 `margin=4`），即数值覆盖 `±10^-4` 到 `±10^4` 的跨
  数量级范围，专门用来测试算子在极端数值（接近零/很大）下的表现，而不
  是挤在 uniform 模式那条很窄的线性带里。

**生效路径（2026-10-26 之前有缺陷，已修复）**：`run_single_case` 先调
`model.refine_weights()` 再调 `model.make_oracle(mode=value_dist_mode)`。
`refine_weights()` 内部的 `PracticalHybridSearch` 搜索过程，如果搜到一组
"验证通过"的权重+输入（`self.sat_inputs` 非空），`make_oracle()` 会直接
用这组缓存的输入，完全忽略 `mode` 参数——这个搜索过程本身在修复前永远用
默认 `uniform` 采样权重/输入，和 `--value-dist-mode` 完全无关。

实测（40 个 case 的批次）：修复前只有 **15%** 的 case 真正体现指定的
`log_scale` 分布，其余 75% 被这个短路悄悄吃掉、3 个异常（约7.5%）。修复
后：`refine_weights()` 现在接受 `value_dist_mode` 参数，当它不是默认值
`uniform` 时，会提前用 `random_tensor(mode=value_dist_mode)` 生成
`max_sample`（=2，和 `PracticalHybridSearch.search(max_sample=2)` 保持
同一个变量，不会脱节）份候选输入，传给搜索过程作为起点——只要这 2 次
尝试里有任意一次验证通过，最终 `sat_inputs` 就会保留指定的分布特征，不
会静默回退到 uniform。修复后实测生效比例提升到约 **65%**（9 个 case 因
`sat_inputs` 本身搜索失败走到 `get_random_inps()` 这条本就受 `mode`
控制的路径，另外至少 17 个 case 的 `sat_inputs` 确认带有 log_scale 特征）。

注意：**权重数值分布不受这个参数控制**，`refine_weights()` 内部的
`weight_sample = random_tensor(param.shape, dtype=param.dtype, ...)`
调用从未传过 `mode`，永远是 `uniform [3,7)`。如果需要权重也按指定分布
生成，目前没有实现，需要另外扩展。

### 6. 轴选择（哪个维度被选中做 Slice/Reduce/Transpose/Unsqueeze/Concat）

改动前：`Slice`/`ReduceBase`/`Transpose`/`Unsqueeze`/`Concat` 这五处轴
选择逻辑各自独立调用 `random.randint(0, ndims-1)`，均匀随机。

改动后：统一走共用函数 `weighted_axis_choice(ndims, boundary_weight=3)`
——边界轴（第一个 `0` 和最后一个 `ndims-1`）被选中的概率是任意内部轴的
3 倍，其余轴之间仍然均匀。`ndims<=1` 时固定返回 `0`；`ndims==2` 时两个轴
都是边界，保持均匀（不做无意义的加权计算）。`boundary_weight=3` 是硬编码
的函数默认值，没有 CLI 参数，想调整只能改 patch 源码里这个默认值。

这个改动的动机：实践中边界轴更容易暴露算子实现的边界 bug（比如首/末维度
的特殊处理逻辑），加权后能更密集地覆盖这类场景，同时保留内部轴也能被
选中（不是排除，只是概率更低）。

`Unsqueeze` 的 `expand_dim` 范围是 `[0, ndims]`（`ndims+1` 个有效插入
位置，包括"插入到最后一维之后"），比普通 `[0, ndims)` 多一个选项，调用
`weighted_axis_choice(ndims+1)` 时让位置 `0` 和位置 `ndims` 都算作边界。

`ReduceBase` 有一个特殊情况：0 维（标量）输入没有任何轴可以 reduce，这
个分支保持返回 `None`（不走 `weighted_axis_choice`，因为
`weighted_axis_choice(0)` 会返回 `0` 而不是 `None`，错误地暗示"选中了
轴 0"）。

### 7. 顶层随机种子

`--seed <int>`：批量生成时的起始种子，驱动每个 case 的 `case_seed`
派生，决定性地控制上面全部随机变量（op 池筛选、节点数、dtype、shape、
输入数值、轴选择）。同一个 `--seed` + 同一组生成参数（`--max-nodes`/
`--tensor-scale`/`--value-dist-mode`/`--soc-tier` 等）永远生成同一批
用例。

`--repro-seed <case_seed>`：不跑批量，只复现 `summary.json` 里某个具体
`CaseResult.seed`（或 `case_id` 里 `-` 后面的数字）对应的那一个 case，
默认打开 debug dump。

## 和 `af_fuzz_npu_env.md` 的关系

这份文档讲"生成算法层面有哪些可调的随机维度"；`af_fuzz_npu_env.md` 讲
"怎么在 A3/A5 NPU pod 上把 `af-fuzz` 跑起来"（环境变量、入口点、并发/
内存配置、已知 flaky 现象）。两者独立，互不重复。
