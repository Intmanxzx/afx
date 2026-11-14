# 当前生成规则：单实现 E/F

## 历史记录和执行边界

历史内测 seed 不再承担跨版本重放兼容，也不继续维护旧生成器或解释器。现有数据库不清理、不改写：
仍能查看记录和已保存的图，但执行入口明确拒绝历史 seed，详情页不再提供可执行的重放命令。
旧 campaign 不会自动变成新规则；再次 refill 时停止，用户应启动新 campaign。

当前只有一套 NNSmith + ATen 生成/序列化/执行实现。E 表示静态，F 表示动态；
短 seed 格式为 `E<26位base36>` / `F<26位base36>`，两者不是两套算子实现。历史 C2-/D2- 仍可解析。
新的独立编码空间避免与历史 token 发生歧义，不会把旧 seed 解释成另一张图。
历史解析仅用于读记录，不包含历史构图、数值生成或执行分支。

## 恢复与新增的因子

- 继续使用 368 个精确 ATen overload，限定逐元素/比较、reduce、view；没有加入矩阵、卷积。
- 逐个 overload 的可读清单见 [ATen 算子清单](aten_op_list.md)；规则源代码仍是 `aten_catalog.py`。
- 图节点保存实际目标、完整模型状态、属性、输入及所有输出；执行器只解析注册的 ATen 目标。
- 仍是 NNSmith/Z3 构图。`--method` 支持 symbolic-cinit（默认）、symbolic、concolic，写入 seed。
- `--dynamic-variants 2..8` 恢复动态样本数（默认 3）。变化尺寸组按 seed 打乱；不再仅因
  与权重某一维数值相同就排除输入轴；权重本身保持不变。变体不再受初始 dim-range 限制，
  但每个输入及中间结果仍受元素预算和模型约束限制。
- 变体只由完整模型约束筛选，不重复 CPU probe。生成完成后共享 CPU 阶段验证所有样本；
  任一样本不可执行就跳过整例，不改图、不偷偷回退静态、不搜索另一套数值。
- 整数先 round（ties-to-even），再按 dtype 范围编码；不再直接截断。
- complex 的实部/虚部分别采样；非零虚部是新增因子，旧实现本来也没有覆盖。
- 不恢复数值搜索/权重 refine，也不恢复 finite-only 预筛。NaN 的比较仍遵守原精度规则，
  不能把成功运行但精度失败的用例当成 CPU 不可执行而跳过。
- `--ops` 仍然生成后过滤，不改变 seed 对应的图和输入；不按设备能力裁剪算子目录。

## 资源与传输上限

统一定义在 `fuzz/limits.py`：最多 64 个计算节点、8 组 Shape、每组最多 256 个张量，
历史只读记录仍接受原来的最多 16 组 Shape。CaseSpec 最多 512 KiB，执行 context 最多 768 KiB。seed 仍对轴长、单张量元素数、
求解时间设上限，CLI 配置在生成前验证；配置可表达的上限不是生成成功保证。

原来 12,000 字节 CaseSpec / 16 KiB context 的隐含限制已移除。生成阶段先做序列化
大小检查，过大则跳过，不等到 NPU 执行完才保存失败。存储、读取、快照恢复共用预算；
上传仍保持单请求 2 MiB 上限，outbox 根据实际编码大小截断批次，避免增大图后堆出超大请求。

## 重放与测试口径

seed 固定采样配置、动态样本数、method 和随机熵；固定代码/依赖下重建同图及输入。
依赖仍校验 NNSmith 源码摘要、Z3 4.12.2、Torch 2.10.0+cpu 和默认 float32。
求解超时或 unknown 都终止生成，不以耗时选择不同的可接受图；不承诺任意架构或
不同 PyTorch/Z3 构建均逐字节一致。

验证必须分开：模型约束、原生执行语义、CPU 可执行、AF 编译路径、NPU 对照。
单看两个后端都通过不能证明序列化保留了原生语义。回归包含：原 NNSmith 可对应算子
与 JSON 往返后执行结果对照；独立 Torch API 对照多轴 sum、keepdim、切片步长、
transpose/permute 及裁剪参数；所有 overload 的 shape/dtype 验证。
未建模的 Cast/Gather/LayerNorm/Concat 等不能因为删除旧解释器就算成已恢复覆盖。

## 本轮 mlir 实测

- 新静态 C2 四 worker、双卡：ReLU 与 ge.Tensor 通过 AF-MLIR 及 CPU 对照；
  transpose/prod 本次图无 AF 覆盖，整例跳过。结果只有两例 CPU/AF 的 4 条成功记录，
  skip audit 两条，无残留单边 CPU 记录。小批次 39.69 秒。
- 新动态 D2：2 组与 8 组 Shape，两例共 10 个 AF 目标 Shape 全部对照通过；
  CPU/AF 共 4 条成功 attempts，无 skip。小批次 48.78 秒。
- symbolic-cinit 和 concolic 均实测动态 2/3/8 样本成功；symbolic 静态成功，
  该固定 seed 的动态图没有足够变化尺寸，按预期跳过，不偷偷改为其他 method。
- 两个独立进程、PYTHONHASHSEED=1/9127 对 6 个 C2/D2 seed 得到相同 workload_id；
  将其中四个指纹作为当前协议的固定回归基线。未验证跨架构逐字节重放。
- 原生语义/多轴参数、2–8 动态样本、64 节点和 8 Shape 的产物/保存/快照/上传，
  分别有独立测试。吞吐没有新增双编译或一 case 一进程，但上述小批次不构成无性能回退证明。

远端临时日志：`/tmp/af-current-smoke.out`、`/tmp/af-current-dynamic-npu.out`、
`/tmp/af-current-hash1.out` 和 `/tmp/af-current-hash9127.out`。未清理或迁移生产历史数据。
