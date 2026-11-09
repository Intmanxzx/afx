# 数字 seed：一个值完整重放

当前默认入口：

```sh
autofusedbg fuzz replay 42
# 只运行基础样本，不改变图、权重、输入或完整 workload ID
autofusedbg fuzz replay 42 --static-only
# 换后端，仍是同一用例
autofusedbg fuzz replay 42 --backend ascendc
# 不执行模型；标准库即可检查完整用例规格与指纹
autofusedbg fuzz seed 42 --show-spec
```

无需 profile、版本参数、生成配置、数据库或原机器。`replay 42` 默认运行同一图/权重的全部三个确定样本；当前默认后端是 cpu-eager，NPU 通过 backend 参数选择，设备选择不改变用例。

## 新契约与破坏性变更

按用户要求全新重构，不兼容先前设计：长格式 `v1:preview-…`、static/dynamic profile、`run --seed-token`、旧 `af-fuzz` 子命令/console entry 均不再支持。数字 seed 42 **不是**旧 batch seed，也不是旧长 token 的别名。旧协议黄金向量已撤下，当前黄金向量为 `tests/fixtures/fuzz/numeric_v1_vectors.json`。

历史 NNSmith 模块/测试暂留用于 D0 故障调查，新的 fuzz CLI 不再调用它们；非 fuzz 调试命令不受此破坏性变更影响。旧诊断文档的命令和指纹属于历史证据，不是当前用法。

## 自包含编码

```
seed = (generation_version - 1) * 2^128 + entropy
```

接受非负整数或规范 ASCII 十进制字符串（不接受符号、空格、前导零）。版本 1 占用 0..2^128-1，42 永远解码为版本 1/entropy 42。更高区间目前明确报 unsupported version，不能回退到版本 1。

后续生成规则改变必须实现新版本，保留**从本轮确定的新数字协议开始**的旧实现和向量。全新重构取消的是废弃设计兼容，不取消未来数字 seed 的长期稳定性要求。UI/JSON 中 seed 用十进制字符串保存，避免 JavaScript 大整数精度丢失。

## 固定生成算法

字段随机源：

```
SHA256(ASCII("autofusedbg/numeric/v1\0") || ASCII(decimal_seed) || 0x00
       || ASCII(field_name) || 0x00 || uint64_big_endian(attempt))
```

digest 是 big-endian uint256；N 个候选使用 rejection sampling，接受 `< 2^256 - 2^256 mod N`，结果 mod N，attempt 从零增加。不使用时间、Python hash、库 RNG、Z3、设备能力或外部文件。

固定因素顺序：dtype(float16/float32)、broadcast_axis(0/1)、fixed_size/base_size(2/3/4)、input_layout(contiguous/transposed)、operation(relu/neg/mul)、reduction(none/axis0/axis1)。

用例为 `x + weight -> relu|neg|mul(weight) -> optional sum(axis)`。weight 在广播轴上恒为 1；基础输入这条轴为 base_size，变体为 base_size+1/+2。三组输入、同一图、同一权重一起构成完整用例；不再存在“静态 seed 和动态 seed 分别生成”。

所有 storage 元素按 `storage/{id}/element/{index}` 取整数 -8..8，再除以 8。规范存储为 little-endian IEEE float16/float32。weight ID=0，样本 ID=1..3。CaseBundle 是 frozen spec + tuple[bytes]；验证存储引用、对齐、view 边界。无需落盘。

当前有界 dyadic 算术可精确表示；独立 scalar reference 版本为 `af-mini-reference-v1-exact-dyadic`。不是通用浮点 oracle。内容 workload ID 只包含规范规格和存储长度/SHA256，不包含执行模式/后端/设备/时序。

## 执行语义

`--static-only` 只选择样本 0，并让编译使用 static shape；其余两个样本依然属于完整 bundle。默认使用 dynamic 编译并遍历所有样本。两种模式都不修改图、权重、基础输入或 workload ID。

后端适配器只处理准备/绑定/执行/同步/输出检查，不改变生成。正确性和性能沿同一循环，逐 shape 分别记录；首次调用为 combined，不把 lazy callable 当编译完成。CPU 可选 wall-time 测量；NPU 严格计时因设备 lease 未实现而明确 unsupported。不同后端使用相同 bundle，设备不支持就报错/unsupported，绝不重抽 seed。

此版仍是有限算子/布局/shape 子集预览：未完成跨架构验收、完整 NPU 因素矩阵、Triton 可用环境、任意 alias/state、编译产物跨进程移交、后台批测/SQLite/view。不能以本地算法推导代替跨设备/长期版本实测。

## 结果归档（D3 首批）

replay 默认将轻量结果写到 data home/results/results.sqlite，可用 `--database ./results.sqlite` 指定位置或 `--no-save` 显式禁用。只保存 seed/指纹/上下文/每 shape 状态指标和短错误，不保存 CaseBundle/tensor。

```sh
autofusedbg fuzz replay 42 --database ./results.sqlite
autofusedbg fuzz results ./results.sqlite --limit 50
autofusedbg fuzz snapshot ./results.sqlite ./snapshot.sqlite
python -m autofusedebugger.fuzz results ./snapshot.sqlite
```

查询与快照不需要 Torch/NNSmith/CANN/Node。view 尚未实现；此阶段 CLI 查询不是个人 NPU 常驻查看服务。
