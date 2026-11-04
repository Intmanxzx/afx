# af-fuzz 能力总览

本文汇总当前 af-fuzz 的用例生成、功能/精度验证、动态 shape、性能、多卡、持续运行、断点续跑、报告和产物能力。详细参数见 af_fuzz_guide.md 与 af_fuzz_performance.md。

## 1. Case 生成

### seed 与复现

--seed 初始化整批随机数生成器，按顺序派生每个 case_seed。同一代码、NNSmith、算子/dtype 候选、生成参数和环境会得到相同 case 序列。--repro-seed 只重跑一个 case，不需要保存模型、权重、输入或图结构。

### 算子组合与融合 opset

工具按 --soc-tier A3/A5 探测 AscendC 实际 lowering/decomposition 和 dtype 组合，而不是直接使用 NNSmith 全量算子。--op-categories 可选 elewise、reduce、view、gather、norm、matmul。每个 case 的计算节点、算子组合、图属性和 dtype 由 seed 控制；候选还会经过 lowering guard 过滤。

### dtype 随机选择

候选 dtype 来自目标 SoC 和算子真实支持组合，浮点 dtype 相对整数/布尔有更高抽样权重。NNSmith patch 已加入 bf16 枚举与转换，但环境没有对应 lowering 时不会生成 bf16 case。报告记录输入/输出 dtype。

### 数值分布

--value-dist-mode 支持 uniform（兼容原约 [3,7)）、log_scale（约 1e-4 到 1e4）、llm（小幅主体并注入少量大幅激活）、recommendation（[0,1)）。该设置贯穿输入、权重搜索和 oracle。

### shape、rank、规模

--tensor-scale 联动元素上限、单维度范围和构图超时，支持 small、medium、large、16k、32k、64k、128k、default。普通档均匀采样，长维度档偏向小值的 log-normal，default 使用 half-normal。大 shape 低频但可达。更高 rank 可通过 NNSMITH_MAX_RANK 设置。

## 2. 静态与动态 shape

默认是静态 shape，一个 case 只有一组具体尺寸。

--dynamic-shapes --dynamic-variants 3 表示同图、同权重生成多组合法尺寸：固定 rank；工具选择大于 1 的尺寸值，联动变化相同尺寸的输入轴；显式调用 torch._dynamo.mark_dynamic；使用 torch.compile(dynamic=True)；先用 CPU eager 过滤非法 shape；无法形成足够变体则为 gen_error。动态性能逐尺寸校验、预热和 profiling，报告记录 dynamic_axes、shape_variants、shape_results 和逐 shape CSV。

dynamic=True 不会自动标记所有 axis；动态轴选择由工具 workload 生成器决定，Dynamo/Inductor 负责符号化和 guard。

--mixed-shapes 仅用于性能模式，按 case seed 稳定选择 static/dynamic，约一半一半；报告增加按模式统计和筛选。两组来自不同随机图，不是同图配对开销实验。

## 3. 功能与精度

不加 --compare-performance 时，工具生成 CPU oracle，在 AscendC 上编译运行并比较结果：pass 表示通过，fail 表示输出不一致或后端精度异常，gen_error 表示构图或动态 workload 失败。

## 4. 性能对比

AscendC worker 设置 TORCHINDUCTOR_NPU_BACKEND=ascendc 和 compile option；Triton worker 清除该环境变量、不传 npu_backend，使用 NPU 默认后端。两个 worker 独立运行，同一 case 在同一张卡上先后测量。

默认预热 3 步、采集 10 步。每步 kernel 时间是 op_statistic.csv 所有 Total Time 求和除以实际采集步数；加速比是 Triton-Ascend / AscendC，>=1 为 pass，<1 为 fail。缺 CSV、耗时、指纹或 worker 结果为 error；error 不一定是功能失败，纯 view 图可能两边精度通过但没有 device kernel。

严格核验融合路径时查看 OP Type：AscendC 常见 autofused_*，Triton 常见 triton_poi_fused_*。当前性能有效性主要依据合法耗时，不强制前缀；fallback kernel 可能进入总时间比较。

## 5. 多卡任务池

功能/精度和性能模式都支持公共任务池。每个 worker 固定绑定一张卡，完成当前 case 后领取下一例；不再固定均分 shard。性能模式中同一 case 的两个后端仍在同卡串行执行。

示例：

    autofusedbg af-fuzz --compare-performance --mixed-shapes --out-dir /tmp/af_mixed --jobs 2 --devices 0,1 --seed 42

父进程统一生成 index/seed、写报告，完成顺序不同也不影响最终排序。

## 6. 持续运行和断点续跑

性能模式省略 --n-cases 即持续运行。Ctrl+C 或 SIGTERM 保存已完成报告；--resume 会优先重跑 pending，已完成 case 不重跑。检查点保存 batch seed、随机数状态、已完成 case、pending task、候选快照、生成/测量配置、设备和软件指纹。kill -9 或断电无法执行回调，但最后一次原子保存的检查点仍可恢复。

## 7. 产物与报告

性能模式默认清理原始 profiling 和独享编译缓存，保留 summary.json、summary.html、worker.log、精简 CSV 和动态逐 shape CSV。--keep-perf-artifacts 保留完整 profiling/缓存。HTML 保留汇总统计、用例列表，并增加性能对比、性能对比汇总、逐 shape 数据和 static/dynamic 分组表。

## 8. NNSmith 改造缺口对应

bf16：枚举和转换已接入，实际生成取决于 lowering 探测。元素数与大 shape：tensor scale 和 max-elem 控制。数值范围：value-dist-mode。rank：NNSMITH_MAX_RANK。单维度区间和分布：tensor scale、uniform/log-normal/half-normal。AscendC opset：SoC/lowering guard 探测与 dtype 过滤。多进程多卡：公共任务池。动态 shape：固定 rank、部分 axis 动态、多尺寸同图执行。性能：双后端 profiling、逐 shape 统计、static/dynamic 分组。

已接入不代表任意 SoC、算子或 shape 都能通过后端 lowering；实际覆盖由 Torch、torch_npu、Triton-Ascend、CANN、SoC 和 NNSmith 版本共同决定。
