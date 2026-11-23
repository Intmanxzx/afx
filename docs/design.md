# 简介

> 更新说明：当前自动化模型 runner 已切换为默认 `torch.compile` + `inductor_npu_ext` 原生 Autofuse 路径，NPU 主图来自 TorchInductor post-fusion IR；本文中 GE manifest 内容仅描述历史离线适配层。当前使用方式和产物约定以仓库根目录 `README.md` 为准。

## 目的

本文档描述 Autofuse 逐算子精度对比工具设计。目标读者是 Graph-autofusion/Autofuse 开发与精度定位人员。工具比较同一次 Inductor 编译得到的 pre-fusion CPU、pre-fusion NPU eager 和 NPU Autofuse 输出，并对 NPU 侧融合算子按 compiler provenance 对齐到同一张 pre-fusion FX 图中的出口节点。

## 范围

包含：

- Inductor pre-fusion CPU/NPU eager、NPU Autofuse/CAModel dump manifest 约定。
- GE dump 图中 `AscBackend` / `FusedAscBackend` 输出到原始单算子输出的映射解析。
- `.npy` tensor 离线加载、format/dtype/shape 容错和误差指标计算。
- JSON/CSV/控制台报告。
- 外部 CAModel 命令封装，基于 CANN `x86_64-linux/simulator` 解析环境并保存 stdout/stderr。

不包含：

- 修改 Graph-autofusion 仓内构建、打包、Python/C++ ABI。
- 直接实现 PyTorch/GE/ACL/CAModel dump 采集逻辑。
- 真实 NPU runtime 资源管理。

# 总体概述

## 软件概述

### 项目介绍

Autofuse 会把多个单算子融合为 NPU 侧融合算子。整网精度劣化时，需要快速判断是哪个融合输出与 pre-fusion 单算子语义输出发生偏差。仓内已有 `autofuse/tools/nwa_tool/fusion_precision_analyzer.py` 支持开启/关闭融合后的 datadump 对比，本工具进一步抽象同一张 pre-fusion FX 图的 CPU/NPU eager 输出与 NPU/CAModel dump，覆盖单算子执行误差和融合出口误差的对齐场景。

### 产品环境介绍

工具以独立 Python 包形式放置在 `/home/xingzhixiong/dfx_dir/Autofuse-Debugger`，依赖 Python 3.9+ 和 NumPy。真实环境中 CPU dump 可来自框架侧 hook，NPU dump 可来自 CANN datadump；当前无真实 NPU 卡时，NPU dump 可由 `/home/xingzhixiong/Ascend/cann-9.1.0/x86_64-linux/simulator` 下的 CAModel 仿真命令产生。工具只消费 `.npy` 与 JSON manifest，不调用非开放 runtime 接口。

## 软件功能

- 从 pre-fusion FX/post-fusion 图和 `.npy` dump 目录生成 CPU/NPU/Autofuse manifest。
- 解析 NPU 侧融合算子输出到 CPU 侧原始单算子输出的映射。
- 逐输出计算 `max_abs_error`、`mean_abs_error`、`max_rel_error`、`rmse`、`cosine_similarity`、不匹配元素数。
- 支持 JSON、CSV 和控制台摘要。
- 支持 `camodel-info` 检查 CANN simulator 目录。
- 支持 `run-camodel` 子命令执行外部 CAModel 命令并保存日志。

## 设计约束

- 不修改 Autofuse C++/Python 对外接口，不影响 ABI/API。
- 不把 CAModel 路径、芯片型号或框架类型硬编码为核心逻辑；具体命令由用户通过 CLI 传入。
- 默认不产生高频 runtime 日志；报告只在离线比较后输出。
- 所有外部路径、manifest、shape、文件缺失都要显式报错或标记状态。
- 不引入除 NumPy 外的强依赖，便于在调试环境中直接运行。

## 假设和依赖关系

- CPU 与 NPU eager 回放使用同一张 Inductor pre-fusion GraphModule 和同一份真实 AOT 输入；Autofuse 使用同次编译的输出。`ir_pre_fusion.txt` 只作为 SchedulerNode/Buffer 映射和 shape 校验层，不直接构造 eager 图。
- NPU 侧 GE dump 图保留 `_datadump_origin_name` 和 `_datadump_origin_output_index` 属性。
- dump tensor 当前主要以 `.npy` 形式提供。
- CAModel 如何执行模型和 dump tensor 的具体命令待接入环境确认。

# 需求分析与设计

## 整体介绍

整体数据流：

1. Autofuse 编译时截获 pre-fusion GraphModule、真实 AOT 输入和 NPU kernel 输出。
2. 在 CPU 和 NPU eager 上回放同一张 pre-fusion 图，产出单算子输出 `.npy`。
3. 解析同一 `model__N` trace 下的 `ir_pre_fusion.txt`，建立 post-fusion SchedulerNode 到 pre-fusion SchedulerNode 的证据链。
4. `make-manifest` 根据 pre-fusion/post-fusion 图和 dump 目录生成 manifest。
5. `compare` 构建 NPU 输出到 pre-fusion 输出的对照关系；融合输出优先使用 IR 校验后的 canonical provenance，普通输出优先使用同一 `canonical_id`。
6. 加载 tensor，处理 format/dtype/shape 差异，输出报告。

## 功能需求

### 功能需求 1：生成 manifest

1. 介绍

   从 pre-fusion/post-fusion 图 JSON 和 `.npy` dump 目录生成 CPU、NPU eager 或 NPU Autofuse manifest。

2. 输入

   `--side inductor_prefusion_cpu|inductor_prefusion_npu_eager|npu_autofuse`、`--graph`、`--dump-dir`、`--out`。`--graph` 必须存在，`--dump-dir` 必须为目录。

3. 处理

   解析 `graph[0].op`，遍历每个 op 的 `output_desc`，按 CANN datadump 常见命名匹配 `.npy` 文件。NPU 侧额外解析融合输出 origin mapping。

4. 输出

   写出 `autofusex-manifest/1` JSON。

### 功能需求 2：逐算子精度比较

1. 介绍

   对每个 NPU 输出找到 CPU 对照输出并计算误差。

2. 输入

   `--cpu-manifest`、`--npu-manifest`、`--rtol`、`--atol`、可选 `--out`、`--csv`。

3. 处理

   对 NPU 输出 `(op, index)`，若命中融合映射则使用 CPU `(origin_op, origin_index)`，否则使用同名同 index。加载 `.npy`，处理支持的 format 转换、dtype 提升和可展平 shape。

4. 输出

   控制台摘要、JSON 报告和 CSV 报告。存在失败项时进程返回 1。

### 功能需求 3：CAModel 目录发现与命令封装

1. 介绍

   提供 CANN simulator 目录发现和外部命令执行入口，便于把 CAModel 仿真命令纳入同一工作流。

2. 输入

   `camodel-info` 输入 `--cann-root`。`run-camodel` 输入 `--cann-root`、可选 `--soc`、`--out-dir`、`--command`、可重复 `--env KEY=VALUE`。

3. 处理

   解析 `x86_64-linux/simulator` 下的 SoC 目录，构造 `ASCEND_TOOLKIT_HOME`、`ASCEND_HOME_PATH`、`AUTOFUSE_CAMODEL_SIMULATOR_ROOT`、`AUTOFUSE_CAMODEL_SOC_DIR` 和 `LD_LIBRARY_PATH`。创建工作目录，替换 `{out_dir}` 占位符，执行 shell 命令，保存 stdout/stderr/command。

4. 输出

   `run.stdout`、`run.stderr`、`run.command`，返回外部命令退出码。

## 非功能需求

### 可维护性

代码按 `graph.py`、`compare.py`、`artifact_io.py`、`runner.py`、`cli.py` 拆分。manifest 是稳定边界，后续接入 CAModel 专用 adapter 时不影响比较核心。

### 可测试性

比较核心可用临时 `.npy` 和手写 manifest 做 pytest。融合映射、shape mismatch、format 转换都可独立覆盖。

### 可移植性

Python 3.9+，仅依赖 NumPy。CAModel 命令、CANN 路径和 simulator 环境由 CLI 参数传入。

### 可靠性

所有文件缺失、schema 不匹配、shape 不兼容、format 不支持都进入明确状态。外部命令日志保存到工作目录，便于复盘。

### 特性交叉影响

| 场景 | 适用性 | 分析说明 |
|------|--------|----------|
| SuperKernel Python 接口 | 不适用 | 不修改 SuperKernel Python 包、选项或 wheel |
| SuperKernel C++/AOT 接口 | 不适用 | 不影响 `libascendsk.so`、AOT 测试或 ABI/API |
| Autofuse 图优化 | 适用 | 只读取 GE dump 图与融合 origin 属性，不改图、不改 pass 时序 |
| Autofuse Codegen/Backend | 适用 | 消费 codegen/debug/datadump 产物，不影响代码生成、tiling 或 kernel 编译 |
| AscendC API / Runtime 交互 | 不适用 | 工具不直接调用 `rt*`、`aclrt*` 或 AscendC API |
| Python/C++ 混合绑定 | 不适用 | 不修改 `pyautofuse` 或 CPython 扩展 |
| 构建与打包 | 不适用 | 独立目录原型，不修改 CMake、`build.sh` 或 run 包 |
| 测试与覆盖率 | 适用 | 新增 pytest 覆盖比较核心；后续接入仓内时应补 ST |
| 性能与日志 | 适用 | 离线逐 tensor 比较为 O(n)，默认只输出摘要，不在运行时高频路径打日志 |
| 兼容性 | 适用 | 不改变既有 API/ABI；manifest schema 使用版本号保证后续兼容演进 |

## 性能

### 编译时长

不修改编译链路，编译时长无影响。

### 执行性能

比较耗时与 dump tensor 总元素数线性相关。工具离线运行，不影响模型执行和 Autofuse 编译执行路径。

### 内存和产物大小

比较时一次加载一对 tensor，峰值约为 CPU tensor + NPU tensor + float64 临时数组。报告为 JSON/CSV，大小与输出数量线性相关。

## 接口设计

### 新增/修改接口描述

新增 CLI：

```bash
afx make-manifest --side inductor_prefusion_cpu|inductor_prefusion_npu_eager|npu_autofuse --graph GRAPH --dump-dir DIR --out OUT
afx compare --cpu-manifest CPU --npu-manifest NPU [--out JSON] [--csv CSV] [--rtol R] [--atol A]
afx compare-npy --golden A.npy --actual B.npy [--rtol R] [--atol A]
afx camodel-info --cann-root CANN_ROOT
afx run-camodel --cann-root CANN_ROOT [--soc SOC] --out-dir DIR --command CMD [--env KEY=VALUE]
```

### 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|--------|----------|----------|------|
| 接口说明 | 是否需要接口评审 | 否 | 独立原型 CLI，不进入 Autofuse 对外接口 |
| 接口说明 | 是否需要补充文档 | 是 | README 和 manifest schema 已补充 |
| 接口兼容 | 行为是否兼容 | 是 | 不影响既有接口 |
| 接口兼容 | ABI/API 是否兼容 | 否 | 不修改 C++/Python 绑定 |
| 接口约束 | 约束不满足时是否清晰报错 | 是 | 文件、schema、shape、format 均明确状态 |
| 接口测试 | 是否需要独立接口用例 | 是 | pytest 覆盖核心比较；CLI 可通过 `--help` 自检 |

## 软件设计

### 关键数据结构

- `TensorRef`：一次比较中的 tensor 位置、format、文件路径和节点类型。
- manifest `nodes`：描述图中节点及输出文件。
- manifest `fusions`：描述 NPU 融合输出到 CPU 单算子输出的映射。
- report `results`：逐输出比较结果。

### 关键技术/算法

对齐算法：

1. 建立 CPU 输出索引 `(op_name, output_index) -> TensorRef`。
2. 建立 NPU 输出索引。
3. 建立融合映射 `(fused_op, fused_index) -> (origin_op, origin_index)`。
4. 遍历 NPU 输出，优先用融合映射找 CPU 对照，否则同名同 index。

误差算法：

- 绝对误差：`abs(cpu - npu)`。
- 相对误差：`abs(cpu - npu) / max(abs(cpu), abs(npu), 1e-12)`。
- pass/fail：`np.isclose(cpu, npu, rtol, atol, equal_nan=False)`。

### 流程设计

主流程为 manifest 生成、外部 runner、离线 compare。异常流程通过状态码进入报告或返回非零退出码。工具不持有 runtime 资源，不需要额外回滚。

### 对子模块的修改

当前不修改 Graph-autofusion 仓内子模块。新增独立目录：

- `autofusex/execution/graph.py`：GE dump 图解析和 manifest 生成。
- `autofusex/execution/compare.py`：精度比较核心。
- `autofusex/execution/camodel.py`：CANN simulator/CAModel 目录发现与环境变量构造。
- `autofusex/execution/runner.py`：外部命令执行封装。
- `docs/`：设计与 manifest 文档。
- `tests/`：最小 pytest。

### 错误处理

#### 系统错误

文件不存在、目录不存在、JSON 非法、外部命令失败均返回异常或外部命令退出码。比较过程中单个 tensor 失败时记录状态，不中断整个报告。

#### 接口错误

CLI 参数非法由 argparse 报错。manifest schema 不匹配返回错误。比较失败返回 1，工具内部错误返回 2。

## 安全检查

- 敏感信息和公网地址硬编码：未新增。
- 外部输入作为数组索引、长度、偏移或内存申请大小：manifest index 转 int 后只用于字典 key；shape 以 NumPy 实际数组为准。
- 整数溢出、反转、除 0：相对误差分母使用 `1e-12` 下限，未做手工大块内存申请。
- 资源生命周期和异常分支释放：文件使用上下文管理器；外部命令由 `subprocess.run` 管理。
- ABI/API 兼容：不修改既有 ABI/API。
- 图改写等价性和确定性：只读图，不做图改写；输出按 NPU key 排序，结果确定。
- runtime/aclrt/AscendC 接口生命周期约束：不直接调用相关接口。

## 兼容性检查

不影响已有脚本参数、配置文件、安装目录、动态库、Python 包、旧产物或用户代码。后续如纳入 Autofuse 仓，应作为新增工具目录接入，不改变既有行为。

## 测试设计

### 测试边界

入口为 Python API 和 CLI。出口为 report JSON/CSV 和进程返回码。测试使用临时 `.npy` 和 manifest，不依赖真实 CANN/CAModel。

### 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|----------|------------|----------|----------|
| 功能 | 融合输出映射到 CPU origin | 手写 manifest + npy | UT |
| 异常 | shape 元素数不一致 | 手写 manifest + npy | UT |
| 性能 | 大 tensor 离线比较耗时 | 后续手工 benchmark | Benchmark |
| 兼容性 | 旧 manifest schema 拒绝 | 后续补充 schema 用例 | UT |
| 特性交叉 | NPU 融合与普通单算子混合 | 后续用真实 GE dump ST | ST |

### 测试命令

```bash
cd /home/xingzhixiong/dfx_dir/Autofuse-Debugger
PYTHONPATH=. python3 -m pytest tests
PYTHONPATH=. python3 -m afx --help
PYTHONPATH=. python3 -m afx camodel-info --cann-root /home/xingzhixiong/Ascend/cann-9.1.0
```

若后续集成到 Graph-autofusion 仓内，再按仓内测试要求补充：

```bash
sh build.sh -u --module=autofuse_framework -j 8
sh build.sh -s --module=autofuse_e2e -j 8
```

## 验收标准

- 能从 pre-fusion/post-fusion 图和 CPU/NPU `.npy` 目录生成 manifest。
- 能对融合输出使用 origin mapping 对齐 CPU 单算子输出。
- 能输出 JSON/CSV 报告，并在存在失败项时返回非零。
- 缺失文件、shape mismatch、format 不支持等异常场景有明确状态。
- 不修改 Graph-autofusion 仓内源码，不影响构建、打包和 ABI/API。

### 设计文档检查结果

- [x] 跨特性交叉影响：已按 `cross_feature_check.md` 检查 SuperKernel、Autofuse optimize/codegen、Python/C++ 接口、构建交付、测试和性能日志影响。
- [x] 编码红线：已检查资源生命周期、ABI/API、图改写确定性、外部输入、除 0 和 runtime 接口约束，无违反。
- [x] 测试设计：新增 pytest UT，覆盖融合输出映射和 shape mismatch；后续接入真实 dump 后补充 ST。
- [x] 性能评估：离线逐 tensor O(n) 比较，不进入编译期 pass、codegen、runtime 调度或 kernel 高频路径。
