# 可重放采样协议 E/F

E 为静态、F 为同一固定图的三组合法动态 shape。C/D 的图和数值字节不变；已经持久化的旧 campaign 继续使用 C/D，新建 campaign 默认 E/F。配置编码进 128-bit seed entropy，短码因此比 C/D 长；无需外部配置即可 replay。不要给已有 seed 追加采样参数。

```sh
autofusedbg fuzz run --tensor-scale default --count 100 --save
autofusedbg fuzz run --tensor-scale large --value-dist-mode llm --dtype-mode mixed --count 100
autofusedbg fuzz run --dim-range 1 1000000 --max-elem-per-tensor 10000000 --dim-dist-mode half_normal --count 100
```

## 历史因子逐项核对

| 因子 | 历史行为 | E/F 行为 |
|---|---|---|
| 轴长 | default `[1,131072]` 半正态；small/medium/large 均匀；16k–128k 对数正态 | 同档位、同 NNSmith 抽样函数；范围、分布和元素预算可覆盖 |
| 元素预算 | small 65,536；medium 1,048,576；large 16,777,216；default 524,288 | 保留；预算是每张量上限，不是单轴范围 |
| 图节点数 | 2 到 max_nodes 均匀，默认上限 5；上限 1 时固定 1 | 恢复；超时未完成节点预算明确报错 |
| rank | `random.choice(rank_choices)`，默认初始 rank 0–5 | 同固定源码；不是正态抽样，最终 rank 可能经反向插图超过初始范围 |
| 轴位置 | 首尾轴权重 3，中间 1；多轴无放回加权抽样 | 保留；Transpose 第二轴仍在其余轴中均匀抽样 |
| Reduce 轴数 | 固定源码 MultiAxisReduceBase 的均匀轴数抽样 | 保留，不退化为单轴 |
| dtype | 旧硬件探测后可用 dtype 池，浮点候选有额外权重 | 恢复 float16/float32 候选（默认 mixed），可指定单一精度；不使用机器相关探测改变 seed |
| 数值 | uniform、log_scale、llm、recommendation，区分 input/weight | 恢复公式与角色区分；独立 seed-local RNG，不再固定为 C/D 的 65 个离散值 |
| 动态候选 | 小尺寸、原尺寸附近、半尺寸候选，seeded shuffle；不改常量 | F 恢复该候选范围与随机顺序，并验证中间张量预算；D 不变 |

数值公式：uniform `[3,7)`；log_scale 随机符号乘 `10**U(-4,4)`；llm 权重 `N(0,.02²)`、输入 `N(0,.2²)` 且 5% 元素放大 100–10000 倍；recommendation 权重 `N(0,1/numel)`、输入 `[0,1)`。新协议复现的是这些分布及固定字节，不承诺与旧 Torch RNG 同 seed 逐字节一致。不会用数值搜索替换为另一分布；溢出/非有限输出作为测试结果保留。

## 范围与限制

- E/F 编码支持轴上下界至 16,777,215、元素预算至 33,554,431、节点上限 255、solver timeout 至 511 秒。原始 32-bit 随机种子和配置共同决定身份。
- `--generation-timeout` 是写入 seed 的求解上限；`--timeout` 是外部 worker/stage 墙钟安全上限（默认 120 秒），大预算实验需同时调高后者。
- 实际输入分布受符号约束、broadcast、Reduce、元素预算挤压影响，不能宣称最终轴长严格服从初始分布。预算挤压可产生 1，即使请求范围下界大于 1。
- 不扩展当前九种 op 的执行词汇，不声称恢复旧探测池的所有算子/整数 dtype；当前存储协议支持 float16/float32。bf16 在历史实现也未完成。
- rank 环境覆盖不属于 E/F 配置；非默认 `NNSMITH_MAX_RANK` 明确拒绝，避免环境改变重放。
- Case/output artifact 仍有独立 64 MiB 安全上限，多张量大图可能明确拒绝，不会降低 shape 伪装成功。
- F 无法找到三组合法输入时明确失败。采样失败必须纳入统计，不能只报告接受样本。
- 源码/Z3 指纹继续固定；跨平台求解器确定性仍未宣称完全保证。

## 验证

`tools/fuzz_shape_distribution.py --count 128 --output ...` 同时审计 C/D/E/F，按最终输入、rank、轴位置、动态 sample、Transpose 过滤前后分层，并记录错误与 dtype/元素数。单测同时覆盖配置往返、历史档位逐字段一致、边界轴权重、最终输入大轴、角色数值统计、C/D 重放不变、保存元数据重新物化、动态和 dtype CPU 执行。

2026-11-10 的 128 组审计：C 128 接受，D 91 接受/37 失败，E 128 接受，F 98 接受/30 失败。C/D 最大基准输入轴为 8；E/F 为 122,468。E 基准输入共 499 个轴，其中 102 个大于 8、88 个大于 8,192，包含 float16/float32。原始报告为本次验证产物，不作为协议定义。

部署：oracle1 使用 `/home/ubuntu/af-fuzz/venv-sampling-1110`，部署前私有一致性备份及 service 原配置在 `/home/ubuntu/af-fuzz/private/sampling-1110/`。服务切换不改生产库；回滚时恢复该目录的 service 配置并重启，不回拷旧数据库覆盖新结果。mlir 客户端安装在 `/mnt/workspace/af-fuzz/sampling-1110/installed`，原入口保存在同目录 `entry.before`。没有终止或改写原有长跑 campaign；新建 campaign 才启用 E/F。

硬件检查：通过 `npussh mlir` 使用两张 Ascend910（逻辑 0/1，物理 12/13），对轴长 66,371 的静态 E 和三组动态 F，eager-NPU/AF-MLIR 均通过 CPU 精度对比（共 8 个 device/backend/mode 组合）。百万轴覆盖另以三个分布的固定 `[1000000,1000000]` 配置验证最终输入包含 1,000,000，而非只检查抽样函数。
