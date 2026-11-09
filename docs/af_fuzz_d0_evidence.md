# D0/D1 实施证据（2026-11-09）

本次执行位置：`npussh mlir`。隔离根目录：
`/mnt/workspace/af-fuzz/refactor-d0-Nh5vfp`（下文日志路径相对此目录）。
本地只保留源码/测试/文档，不安装或运行后续测试。最初误建的本地临时 venv、日志、传输包已清理。

## 1. 基线与消费者

- 本地起点 `6d6190c`；两份评审文档原为未跟踪文件，保留，未提交/推送。
- 无适用 AGENTS.md。旧入口 `autofusedbg af-fuzz`，正确性和性能仍是不同会话实现；本次只做转发，不增加第三套执行器。
- `generation.py` 为已有共用生成入口；`performance/{session,parallel,worker,dynamic_worker}.py` 为性能消费者，`fuzz_loop.py` 为正确性消费者。
- 旧数据消费者包括 summary JSON/HTML、性能 CSV、checkpoint/resume、重放 dump。D1 不更换其格式，不伪装为新协议数据。
- 项目元数据 Python >=3.9、numpy 必需，NNSmith 另装。本次实际执行：Linux aarch64、Python 3.11.4、SQLite 3.31.1、Torch 2.10.0+cpu、torch_npu 2.10.0.post7.dev20261015、NNSmith 0.2.0.dev25。
- `~/autofuse` HEAD `c1b46272`，初检无修改。`~/niu/nnsmith` HEAD `bc0af42`，已有 9 个修改文件；复制到隔离目录后测试，不改原件。旧 baseline import 可能补齐**隔离副本**的补丁。
- 初检 NPU 6 / 两个 chip、健康 OK、AICore 0%，无运行进程；约 3GB/chip 已占用内存。并非严格性能独占测量；不使用此探针声称吞吐提升。
- 最初本地缺 NNSmith 导致普通 import 触发补丁错误、测试收集失败；后续按用户要求所有执行转远端。新的 import 无该副作用。

基线命令（在隔离 `repo` 下，使用 `PYTHONPATH=$PWD:../nnsmith`）：

```sh
source ~/env.sh
python -m pytest -q
```

`baseline.log`：**256 passed, 5 failed / 48.30s**。五项失败原样保留，不通过跳过/xfail 隐藏：

1. `test_pytorch_adapter_captures_model_from_cann_ge_fork_worker`：未捕获 fork 子进程 module call。
2. `test_run_script_keeps_script_directory_importable_during_backend_execution`：前序测试的 `libs` 模块污染，`add_one` import 失败。
3. `test_seed_survives_candidate_order_and_previous_cases[42]`：具体 shape 漂移。
4. `test_capture_prefusion_graphs_skips_cpu_replay_by_default`：`Linear` 无 `.graph`。
5. `test_capture_prefusion_graphs_with_cpu_captures_both_sides`：同上。

## 2. Seed / input size 诊断

新增 `tools/fuzz_solver_probe.py` 和 `tools/fuzz_seed_probe.py`，仅显式诊断时写 JSON，不用这些文件保证重放。

```sh
export PYTHONPATH=$PWD:../nnsmith
python tools/fuzz_solver_probe.py --output-dir ../solver
PYTHONHASHSEED=0 python tools/fuzz_seed_probe.py --seed 42 --output ../traces/a.json
PYTHONHASHSEED=123 python tools/fuzz_seed_probe.py --seed 42 --output ../traces/b.json
python -m autofusedebugger fuzz diagnose ../traces/a.json ../traces/b.json
```

诊断顺序：配置/候选池 -> 具体图/有序变量 -> 初始权重 -> refine 权重 -> 输入字节/签名 -> 动态轴/逐 shape 输入。记录 dtype、shape、stride、storage offset、alias group 和字节 SHA256；首个差异返回 stage、JSON path、预期/实际。诊断物化时对比模型声明输入和实际输入的名称、顺序、dtype、shape；绝不 reshape/pad 掩盖错误。当前未接入所有编译加载/后端边界，此项留 D2。

已捕获的事实（`solver/comparison.json`）：

- 裸 `model_gen` 按 0/42/389609433 顺序运行，并在重试前生成 seed+1、反转候选池时，seed=42 首个差异在 **graph / `$.payload.variables[1].shape[4]`：2 -> 1**。
- 图指令文本相同不意味着广播形状相同；不同变量也有 [1]/[2] 变化。
- 同一实验使用已有 `_seeded_solver_context` 后，三个 seed 均无差异。说明该捕获现象与求解器上下文/调用历史有关；不声称已查明 Z3 内部全部原因。
- **生产 `generate_case` 已有 context 隔离**。本次补充其回归，不将裸 API 的故障冒充生产修复，也不更改共享 NNSmith。
- seed=42、受限 Add/Mul/Neg/ReLU、小规模静态：hash seed 0/1/123 和前序 seed=43 后，完整 trace digest 都为 `4259ae2b2184a5f14ef415f4bcc6319b5ed2d7d8a58bf67569e71e0948595398`。fixture 明确标为环境受限的 legacy diagnostic golden，而非 v1 承诺。
- 静态 fresh process 生成约 0.399–0.402s，调用历史热身后约 0.024s；RSS 约 1.09–1.11 GiB（Linux ru_maxrss KiB，包含 Torch 进程），trace 3529 bytes。不是全流水线性能基线。
- 动态 seed=42 生成 5867-byte trace，约 0.415s；完整 digest `b3e04667a7da3baf511d85b7543d79d60ec3977f0d208c61b36c749d941a81cd`。
- 手工固定诊断样本覆盖广播、规约、两个 shape、float32/float16/int32、转置/alias 与字节哈希；它们是诊断工具测试，**不是新生成器覆盖率或跨架构黄金测试**。

仍未定位用户原始广播轴/input size 故障。动态候选合法性拒绝单独报告，不归为同一故障。完整图属性的结构化表示、参考输出/可变 buffer、真实设备布局和所有输入绑定检查点仍需 D2 完成。

## 3. 后端能力矩阵

`tools/fuzz_backend_probe.py` 使用广播 Add+ReLU；调用后同步并检查输出。每后端独立进程/缓存、串行执行。

| 能力 | AscendC | Triton-Ascend |
|---|---|---|
| 当前环境 eager 输入准备 | 通过 | 通过 |
| callable 创建 | 约 0.688s；不是编译结束 | 已进入编译流程 |
| 首次 combined compile+execute | 约 15.958s，通过 | 失败，见下 |
| 再次执行（非正式性能） | 约 0.000634s，通过 | 未测 |
| 纯 CPU 完整图编译 | 未验证 | 未验证 |
| 跨进程可移交执行包 | 未验证 | 未验证 |
| 调度契约 | 只允许 combined | 不声明可用/split |

日志：`backend-ascendc.log`、`backend-triton.log` 记录初次环境失败；`overlay-ascendc.log`、`overlay-triton.log` 记录补齐隔离依赖后的验证。

AscendC 首次失败为未加载 autofuse Python 包；重试只读使用已有预构建 overlay 的 paths.json，设置独立缓存及禁止 pyc 写入，不改共享源码，也未重新编译 Autofuse。

Triton 预检不存在。使用 `pip --target <隔离根>/triton-deps --no-deps` 安装 `triton-ascend==3.2.0` 及缺失的 pybind11，不升级 Torch/CANN。最终运行失败：当前 CANN 9.3.0 头文件没有 `rtLimitType_t::RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE`，Triton 的 npu_utils.cpp 无法编译。不能声称双后端环境已补齐；需寻找与该 Torch/CANN 组合匹配的 Triton 构建，禁止通过修改共享 SDK 枚举强行绕过。

已复查当前 torch_npu 源码 `ascendc/compiler/_autotune.py:102–143`：使用 NPU current_stream、NPU profiler 和 stream synchronize，因此此 autotune 路径必须占用设备，不能归类为纯 CPU 工作。本探针未覆盖触发该路径的完整调优用例。

## 4. D1 交付范围与边界

- 普通包/CLI 导入不再初始化 NPU 环境、不再改 NNSmith。兼容检查默认只读，缺补丁显式失败；显式 `isolated_copy=` 才允许旧工具给用户自建副本打补丁。
- `autofusedbg fuzz setup|doctor|run|diagnose`；旧 `autofusedbg af-fuzz` 和 `af-fuzz` console script 都转发同一旧执行处理器。不是新统一执行引擎。
- `setup` 原子写配置、持久 source_id、独占 setup lock、保留未知字段、路径规范化、CPU worker 预算、管理员预发 user_id/display_name/key_file 引用。密钥内容不打印、不复制。没有在线注册、密钥申请或伪造在线校验。
- `doctor` 仅读取包元数据与 Python/SQLite/OS，不 import 执行依赖、不探测设备、不安装软件；明确报告未实现/未验证能力。
- seed token 编解码及不可变 CaseSpec/CaseBundle/配置/短结果为**预览契约**。未注册任何 v1 generator，调用显式失败，不用当前设备能力或旧整数 seed 偷换语义。CaseBundle 使用 bytes，不要求保存；完整 storage/alias 边界、oracle 与 materializer 验收留 D2。
- `view/snapshot/status/stop/resume` 明确报未实现，不启动服务。离线身份上传、资源检测、同步器、前端资源、三系统 CI 尚未实现。
- Linux aarch64 的无 Torch/NNSmith/numpy venv 已成功安装本项目 wheel（`--no-deps`），doctor/setup/run help/af-fuzz help 正常。仍保留旧项目 numpy 基础依赖，不宣称 Windows/macOS 运行通过。

## 5. 下一步验收门槛

1. 保留原五项基线失败；继续对生产生成器更广候选、dtype、动态/权重约束采样，获取用户原故障 seed/config/阶段证据。
2. 固定最小 v1 profile 与 PRNG/物化协议，加入无缓存、多架构及旧版本 golden 验收后才开放 v1 run。
3. 找到匹配的 Triton 构建，实测完整图编译、设备调优、产物序列化/异进程加载边界，未证明前保持 combined。
4. D1 完善资源预算、依赖兼容/离线管理员配置检查和三 OS 干净安装 CI；D3 才交付预构建 Web 与 SQLite view。
5. 全流程生成/编译/执行/报告的时间、RSS、磁盘预算尚未完成，不能用上面的单算子计时代替。

## 6. 最终回归结果

- `final.log`：**279 passed / 5 failed / 27.91s**；失败集合与基线完全相同。
- `focused-final.log`：**38 passed / 3 deselected / 17.95s**；聚焦命令排除已有裸 API 候选顺序组三项，全量未排除。
- `wheel-final.log`、`doctor-final.json`、`setup-final.json`、`run-help.log`、`alias-help.log`：最终 wheel 与轻量 CLI 验证。重复 setup 保留相同 source_id。初次安装被工作目录同版本 egg-info 误判为已安装；切出源码目录、清除 PYTHONPATH 后 force-reinstall 到隔离 venv 验证通过，不影响共享环境。
- 动态 seed=42 在 PYTHONHASHSEED=123 新进程重放，trace 比较相同；`dynamic42-history.log` 的前序 seed=43 在目标 case 运行前因 unit/scalar-only graph 无合法动态轴而拒绝，故这次**动态前序历史实验未完成**，不能计为通过。
- 最后复查共享仓库 status 与初检一致；没有在共享源码或依赖上应用本次修改。

## 7. 第二批：资源与扩展重放（2026-11-09）

远端工作副本 `round2/`；重新核对仓库、Python 与设备：共享仓库状态未变，NPU 无运行进程。未执行新的 NPU 性能测量，也未重试不匹配的 Triton 构建。

### 资源/配置

`round2-doctor.json` 实测宿主 CPU=40、affinity=40，但 cgroup CPU quota=36 核；memory limit=225485783040 bytes（210 GiB），物理内存=246360707072 bytes。解析的是本容器真实 cgroup v1 controller mount/member 路径，包含可见父级限制，不只看 `/sys/fs/cgroup/cpu.max`。磁盘探测使用数据目录最近存在的祖先，不为 doctor 创建目录。

新预算只作持久配置/告警，不声称实现调度强制约束。密钥元数据检查不等于在线认证/唯一展示名校验，不读密钥内容。

复现：

```sh
cd /mnt/workspace/af-fuzz/refactor-d0-Nh5vfp/round2
source ~/env.sh
export PYTHONPATH=$PWD:../nnsmith
python -m autofusedebugger fuzz doctor --data-dir ../setup-home
python tools/fuzz_replay_audit.py --output-dir ../NEW-audit-directory
```

### 重放调查

矩阵：core(Add/Mul/Neg/ReLU) 或 extended(另加 Transpose/Reshape/ReduceMin)，float32/float16，static/dynamic，seed 0/42/389609433，共 24 对、48 个新进程。第二个进程使用 hash seed 123（第一个 0）、OMP/MKL 线程环境 4（第一个 1），先调用 seed+1。生产物化自身仍固定 Torch 线程数，不能将环境变量实验描述成已验证多线程物化。

第一遍 `replay-round2/summary.json`：24 对生成重放相同，16 个前序成功、8 个前序因动态轴不合法拒绝。扩展样本确实出现 ReduceMin、Transpose，不是只换候选目录但仍全是逐元素图。未承诺每个候选算子都被选中。

检查 IR 后发现 `Transpose.extra_attrs['dim0'/'dim1']` 不在 `str(ir)` 中，首批文档“文本包含所有属性”的判断不充分。已修正诊断捕获并重跑，新增阶段为 graph_extra_attrs。原 legacy_seed42 fixture 指纹只适用于首批诊断格式，不是稳定生成协议。

### 中间测试异常保留

`round2-full-final.log` 一次与审计并发的全量测试：297 passed / 6 failed。额外失败为 `test_real_spawn_pool_rebalances_complete_performance_cases`：任务全部完成，但 slow worker 得到 4 个任务，不满足硬编码的 1:9 分配断言。未修改或跳过该测试；调查结束后串行复跑确认。该事件不能从记录中删除，也不能直接认定为确定性调度回归。

新增轴属性后的矩阵位于 `replay-round2-attrs/summary.json`：仍为 **24 对成功重放 / 48 个新进程**，包括具体 extra_attrs。它只证明本次选定 seed/目录/环境的 trace 一致；没有复现用户原始故障，也不覆盖所有算子、设备或长期协议。

新版 core/native seed42 trace digest 为 `d1155e01b59b591898ef08cf245b5fa8ef6c6e68e0188bab6f81ff2543b9368c`（4393 bytes）。fixture 同时保留旧诊断格式 digest，不能跨格式直接比较。

资源/配置+诊断新增测试覆盖 cgroup v1/v2、嵌套父限制、unlimited sentinel、非法路径、损坏配置不覆盖、凭据不泄漏、预算更新保留 source ID、三平台路径逻辑和前序拒绝后继续重放。远端干净 wheel 环境再次验证新版 setup/doctor；没有 Torch/NNSmith/numpy，预算更新及再次 setup 保留原 source ID。

审计结束后串行全量 `round2-serial-final.log`：**299 passed / 原 5 failed / 31.53s**。负载相关 1:9 分配断言此次通过；没有修改测试或调度代码来掩盖它。最后 wheel 的 pip list 仅 autofusedebugger 0.1.2、pip、setuptools（`round2-wheel-final.log`、`round2-light-doctor-final.json`、`round2-light-setup-final.json`）。

## 8. 第三批：v1 纵向切片证据

远端副本 `round3/`。复检共享源码/设备状态不变；所有安装、测试、生成 golden 都在远端；本地只接收需版本管理的 golden JSON，不保存运行缓存或日志。

### 生成协议

新 profile 的生成完全独立于 NNSmith/设备/历史数据，详见 `af_fuzz_seed_v1.md`。仅预览因素子集，不是对 legacy generator 的确定性背书。无 Torch/NNSmith/numpy 的 `light-venv` 使用 wheel 成功运行：

```
autofusedbg fuzz seed v1:preview-dynamic-v1:42
```

得到 workload_id `a72251ef070cf55af01a56c15c2f7a29bc86df64e5060cb501d6f0d11f952a9a`，三个 shape。同一轻量环境对六个 frozen golden 的 workload 与 oracle 哈希检查全部通过。记录：`round3-wheel.log`、`round3-light-seed.json`。

### 执行

- CPU eager 两 profile 各 32 seed（32 static + 96 dynamic shape）精确通过独立标量参考；覆盖全部 dtype/operation 组合。另有 128 seed 的广播轴/布局/规约因素可达性测试。
- `round3-cpu-compile.json`：dynamic seed42 在 CPU Inductor 三个 shape 均通过，warmup=1、steps=2，mean wall-time 约 67.8/59.5/49.2 µs。只是功能性测量，不是稳定性能基线。首 shape combined 首次调用约 10.79s。
- `round3-preview-static-v1-ascendc.log`：static seed42 通过，首次 combined 约 21.34s。
- `round3-preview-dynamic-v1-ascendc.log`：三个 shape 均通过，首次 combined 约 18.72s / 0.255ms / 0.202ms。不能据此断言全部动态用例只编译一次。
- NPU 使用既有 overlay paths.json 只读依赖，`round3-npu-cache`、`round3-triton-cache`、`round3-tmp` 为隔离产物目录。未修改共享源码/依赖，没有向默认归档保存 tensor。
- 新适配器识别 profile 之外的图为 unsupported，不悄悄将未知 op 替换成已支持算子；后端失败不修改 seed/CaseSpec。
- 严格 NPU performance 请求在依赖加载前拒绝：lease 尚未实现。Triton 只接入适配路径，本轮未绕过前轮 CANN 不匹配错误，不宣称已可用。

本轮依然没有跨 CPU 架构验证。固定 IEEE 打包和 dyadic 算术是实现依据，不等于已经实测跨平台；同名 profile 的长期兼容必须由未来版本持续跑这些黄金向量保证。

第三批最终测试：`round3-full-final.log` **322 passed / 原 5 failed / 31.58s**，比第二批增加 23 项通过，无新增全量失败。`round3-protocol-final.log` 是增加最终设备匹配测试之前的专项结果（22 passed / 2.68s）。所有最终改动均已纳入全量；最终 wheel 重建验证见 `round3-wheel-final.log`。

## 9. 第四批：单数字 seed 重放

用户撤销旧设计兼容要求，本轮使用全新数字协议。前几节长 token/两个 profile 的测试和指纹只作为历史证据，不代表当前协议支持。

隔离目录 `round4/`，初检共享仓库状态未变，NPU 无运行进程。结果：

- `round4-full.log`：**319 passed / 原 5 failed / 36.23s**；相对第三批去除重复 profile 参数化和旧初始化测试，新增数字 seed/未知版本/static-only 身份不变/单 seed CLI 回归。
- `round4-cpu.json`：CPU Inductor 全三 shape 正确性及可选性能阶段通过。
- `round4-ascendc-all.json`：`replay 42 --backend ascendc` 三个 shape 全部 pass。
- `round4-ascendc-static.json`：同 seed 加 `--static-only`，基础样本 pass。
- CPU、NPU 与 static-only 共用 workload ID `610516f468114ce745c01ab2d1cb78129c87e4395d60e19e034aee9a714e9daa`；seed 对外表示为十进制字符串 `42`。
- `round4-wheel-final.log`：干净构建后轻量 venv 验证五个数字 golden，wheel 不含旧 console entry 或 fuzz/runtime.py。标准库生成无需 Torch/NNSmith/numpy。
- 隔离 `round4-npu-cache`、`round4-cpu-cache`、`round4-triton-cache`、`round4-tmp` 存放运行产物；本地只保留需交付的源码、文档、golden JSON。

未知数字版本不回退；旧长 token 不兼容；`--dynamic/--max-nodes/--dtype/--seed-token` 不接受作为 replay 生成覆盖参数。静态选择和性能开关不改变 bundle 指纹。尚无跨架构实测，不以这次单 NPU 环境结果替代跨设备验收。

## 10. 阶段 commit、模块边界与 D3 存储

用户已授权阶段性 commit，仍不推送。checkpoint `c9039fa` 保存前轮成果；`e20e44d` 保存独立 fuzz 模块化。模块化远端 `round5-modules.log`：322 passed / 原 5 failed / 32.56s。

模块新增 generation registry/version、backend interface/factory/Torch adapter、tensors、独立 __main__；所有数字黄金向量保持原值。3 个架构测试覆盖依赖方向与不依赖旧 debugger 的模块入口。

D3 首批工作副本 `round5-storage/`。`round5-storage-final-focused.log` 10 个存储专项测试全部通过，包含多 writer 首次创建。`round5-saved-replay.json` 保存实际 CPU replay42，生成 `round5-results.sqlite`；`round5-snapshot.sqlite` 与 results 命令读到同 workload ID。DB 仅 contexts/attempts/shapes，白名单记录，不含 tensor/原始日志/密钥配置。

快照测试保留一个 WAL writer 的未提交修改，另一个连接一致性导出已提交状态；额外 secrets 表不出现在导出 DB。故障注入 trigger 验证 shapes 失败时 attempts 和新增 context 一起回滚。未知 schema、损坏文件及其它 SQLite 均拒绝，不覆盖用户文件。

最终 `round5-final.log`：332 passed / 原 5 failed / 42.15s。`round5-wheel.log` 构建安装同包 wheel 到无重依赖 light-venv；`round5-light-query.json` 验证独立模块查询快照成功。本轮所有测试/构建在 npussh mlir；本地只编辑、Git 检查和 commit，没有安装/运行测试或创建临时测试产物。

## 11. D3 离线查看器验收（2026-11-09）

隔离根目录 `/mnt/workspace/af-fuzz/refactor-d0-Nh5vfp`；本阶段源码 `round6-view/`。实施前复查 Python 3.11.4、共享仓库及设备：Autofuse 无改动，NNSmith 仍为既有 9 处修改；npu-smi 当时 AICore 为 0%，无列示进程（仍有 HBM 占用），本阶段不执行 NPU 工作。

证据：
- `round6-focused.log`：初始 view+storage 专项 33 passed。
- `round6-full.log`：357 passed / 原 5 failed / 73.35s（最终架构测试新增前）。
- `round6-final.log`：**358 passed / 原 5 failed / 33.85s**；新增 26 个通过项，无新增失败。
- `round6-scale.json`：100,000 attempt / 300,000 shape；74,817,536 bytes；page 0.564ms、deep_page 0.548ms、seed_filter 0.433ms、status_filter 0.576ms、summary 36.124ms；单页 12,231 bytes。
- `round6-wheel.log` / `round6-light-view.log`：wheel 内资源 + 无模型/Node 环境，安装 CLI HTTP 启动/资产/API/退出验收成功。
- `round6-browser.log` / `round6-screenshots/{desktop,mobile}.png`：真实 Chromium 130 浏览器交互、响应式无整页横向溢出、零 page error。
- `round6-browser-install.log` / `round6-browser-libs.log`：新版 Playwright 平台不支持；旧版所需 .deb 下载后只解包到 `view-browser-libs`，以该测试进程 LD_LIBRARY_PATH 使用，没有系统安装。

中间失败均保留原因：首次 wheel 使用 --no-build-isolation 时 light-venv 缺 bdist_wheel，改回 pip 隔离构建成功；首次轻量检查在含 nnsmith 源码目录的父目录运行，发现 namespace spec，移至空 cwd 后无该依赖；首次浏览器测试发现 reset DOM 命名冲突，修复后重新验证。没有通过忽略断言/静默跳过来宣称通过。

未验证：跨 CPU 架构/跨 OS 重放、Windows/macOS 浏览器与 wheel 运行、任意恶意数据库沙箱隔离、schema 升级/恢复导入、中心服务及自动上传。性能数据是合成查询单次测量；不代表 NPU 测试性能或真实生产库上界。

## 12. D3 可信恢复/导入验证

隔离源码 `refactor-d0-Nh5vfp/round7-recovery`。实施前复查 Autofuse HEAD c1b46272 且干净、NNSmith 原有 9 项修改；NPU 无列示运行进程（仍有 HBM 占用），本阶段不运行 NPU。

`round7-focused.log` 初始 transfer+storage 31 passed；新增子进程崩溃回滚/WAL 未提交数据隔离后以 `round7-transfer.log`、`round7-final.log` 为最终记录。`round7-light.log` 保存 installed-wheel restore/import/check-db 实际命令输出。异常注入并非真实填满磁盘；崩溃回滚以子进程 os._exit(77) 在事务写入后、提交前终止验证。未测断电和目录 fsync 持久化保证。

最终：`round7-transfer.log` **23 passed / 40.50s**；`round7-final.log` **381 passed / 原 5 failed / 74.14s**。`round7-light.log` restore added=1、再次 import skipped=1、check-db schema1/integrity=ok。新增测试全部通过；全量耗时不是性能基准。

## 13. D4 操作任务账本首批

隔离源码 `refactor-d0-Nh5vfp/round8-ledger`。测试日志 `round8-focused.log`、`round8-final.log`；构建及无执行依赖安装验证为 `round8-wheel.log` / `round8-light.log`。

覆盖 case/backend 分开入队、任务幂等、容量背压、并发首次建库/领取、停止/显式恢复、过期 claim 拒绝、六个子进程只允许一个领取、子进程退出后禁止隐式重领、恢复时容量检查及跨客户端持久化容量。没有 NPU 执行或假冒后端 split；任务账本完成不代表调度系统完成。测试子进程退出后由父进程 communicate/wait 确认退出，才 interrupt/resume。

最终 `round8-focused.log` **10 passed / 0.97s**；`round8-final.log` **391 passed / 原 5 failed / 35.75s**。`round8-light.log` 安装 wheel 内部生命周期测试成功，无 Torch/NNSmith/numpy 导入。所有构建/测试在远端隔离目录；本地没有新增临时产物，不修改共享依赖，不推送。

## 14. D4 前台 CPU supervisor 验收

远端隔离根 `refactor-d0-Nh5vfp`，源码 `round9-supervisor`。实施前复查 Python3.11.4、Autofuse c1b46272 clean、NNSmith bc0af42 原9项修改；NPU无列示运行进程但有HBM占用，本批不运行NPU。

- `round9-final.log`：**407 passed / 原5 failed / 50.31s**。新增16项 supervisor/CLI 测试通过。
- `round9-focused-final.log`：最终 supervisor + ledger 专项（含真实 worker、进程组超时、SIGINT、父死后worker锁、保存后崩溃确认、轻量CLI）。
- `round9-cli-run.json` / `round9-cli-status.json` / `round9-cli-results.json`：`run 42 --foreground --backend cpu-eager --backend cpu-compile` completed=2，两个后端结果均ok，workload ID均为 `610516f468114ce745c01ab2d1cb78129c87e4395d60e19e034aee9a714e9daa`。
- `round9-cli-check.json`：结果库2条完整性与契约检查通过。
- `round9-static-run.json` / `round9-static-check.json`：单独静态样本+CPU可选性能路径验证。
- `round9-wheel.log` / `round9-light.log`：重建安装wheel，源码目录外无模型依赖环境 status 成功。

中间失败：`round9-full.log` 首次全量 **404 passed / 6 failed / 94.96s**，除了原5项，旧 `test_real_spawn_pool_rebalances_complete_performance_cases` 的任务分配时序断言期望1:9、实际3:7。任务本身10条完成。最终全量同项通过；没有修改或删除该测试，也不能据一次复验保证其不再波动。独立重跑见 `round9-legacy-timing-rerun.log`。

本轮验证的是POSIX CPU进程树，不是设备租约。子进程清理使用独立进程组TERM/KILL+wait；测试允许被init尚未回收的僵尸，但不允许继续执行。自行脱离进程组的helper、孤儿设备进程以及SSH断连后的自动托管未验收。父进程SIGKILL时worker可继续完成，但继承锁阻止另一supervisor并发接管；不得删除/替换活跃队列或自行强制解锁。

最终专项 `round9-focused-final.log` **26 passed / 35.14s**；旧时序测试独立复验 `round9-legacy-timing-rerun.log` **1 passed / 4.18s**。静态CPU性能用例断言 status=ok、1个shape、performance=measured；不与NPU kernel profiler比较。

## 15. D4 原子批次与任务 schema2

隔离源码 `refactor-d0-Nh5vfp/round10-batches`；实施前复查Autofuse c1b46272 clean、NNSmith bc0af42 原9项修改；NPU无列示运行进程但HBM有占用，本阶段不运行NPU。

`round10-focused.log` **33 passed / 13.76s**，包含7项新增批次/升级测试：容量及晚到冲突全回滚、非法规格不入队、并发批次预算、schema1一致性备份→schema2保留task身份、备份发布失败不升级、活跃锁拒绝、reason限长与重新打开持久化。最终全量见下方。未知版本不猜测迁移；不将此任务库升级等同结果库未来升级或断电验收。

`round10-full.log` 最终 **414 passed / 原5 failed / 52.19s**，新增7项全部通过，未出现上一轮额外时序失败。

## 16. D4 后台 CPU 启动握手

隔离源码 `refactor-d0-Nh5vfp/round11-hosting`。

- `round11-focused.log` 首轮 **30 passed / 60.65s**，覆盖后台launcher退出后真实seed42完成、启动失败/超时保持pending、错误结果库绑定/活跃锁不入队、旧状态文件不冒充存活、未授权断开不领取、SIGTERM清理worker。后续增加后台API拒绝NPU和升级ALTER后失败回滚测试。
- `round11-migration-rollback.log` **1 passed / 0.10s**：注入ALTER后、版本提交前失败，原库和独立备份仍为schema1八列，无残留临时备份。
- `round11-ssh-start.json`：独立npussh连接提交CPU Inductor seed42，收到running ACK后该连接退出。
- 随后新npussh连接读取 `round11-ssh-status.json` / `round11-ssh-results.json`：task completed、supervisor finished/unowned、result ok，workload ID `610516f468114ce745c01ab2d1cb78129c87e4395d60e19e034aee9a714e9daa`。验证正常SSH命令结束后的有限后台任务，不冒充网络强断/重启自动恢复或systemd/cgroup验收。
- `round11-wheel.log` / `round11-light-start.json`：同包wheel安装与轻量环境后台空队列启动，最终状态另见light-status。

后台进程只有本地socketpair握手，不监听网络，不安装NPU查看API，也未修改共享Torch/CANN/NNSmith。子supervisor不自动重跑interrupted任务，有限队列耗尽即退出；后台执行不是自动上传，也不是机器重启托管。

`round11-final.log` 最终 **423 passed / 原5 failed / 46.68s**，包含后台8项新增测试和额外升级回滚1项，无新增失败。`round11-light-status.json` 安装wheel后台空队列最终finished/unowned；不是仅检查启动ACK。无本地测试/安装/临时产物，远端共享仓库状态保持不变。

## 17. 托管资源预算首批

远端round12-budgets：新增预算专项 **17 passed / 3.41s**，全量 **440 passed / 原5 failed / 54.44s**。`round12-run.json` 真实seed42 eager/Inductor均completed，`round12-results.json`两后端ok且workload指纹不变；4GiB RSS/1GiB scratch预算生效，结束无.fuzz-worker临时目录。采样超额终止不是硬隔离；没有声称峰值绝不越限。

## 18. 公网与同步首批（goal进行中，非最终验收）

oracle1 x86_64/Python3.12，babynews源码180KiB及systemd/Caddy配置备份到/var/backups/af-fuzz-cutover后，停用babynews，Caddy 80/443转新af-fuzz.service(127.0.0.1:8090)。公网healthz和只读摘要已通过；管理员密钥只落远端0600文件，未入库明文/日志/Git。

mlir round13/14：初始同步/存储/架构专项17 passed；v2+sync专项15 passed。round14-focused 39 passed/3.79s，全量 **458 passed / 原5 failed / 80.42s**。首轮额外1项是旧测试仍把v2边界当未知版本，更新为v3后通过，v1 golden不变。

公网同步首次HTTP403 Cloudflare1010，明确产品User-Agent后HTTPS上传15条成功；未关闭TLS验证、未转发凭据到其他站点。系统有systemctl但容器PID1非systemd，因此systemd试启动失败，已改独立POSIX同步daemon。

NPU首轮3例失败原因是覆盖PYTHONPATH漏掉CANN Autofuse；保留原环境路径后3例×CPU/AscendC×3shape均ok。正式1000组诊断runner已启动，使用v2 seed区间0..999，逐seed两个后端三个shape，轻量结果实时同步；未完成前不能报1000通过。日志acceptance-1000.log，数据库acceptance-home/results/results.sqlite。该长验收不是托管NPU租约已完成证明。

round14-final-focused **48 passed / 59.68s**（v1/v2黄金、同步幂等/ACK丢失/断网、备份及view）。公网首次Cloudflare UA限制已使用产品UA修复，自动daemon状态ok/ACK递增，公网摘要已出现真实验收结果。1000组执行仍进行中，不能提前完成goal。

## 19. 网站/设备锁进一步实测

round15-lease.log **27 passed / 3.64s**（新device lease3项、sync7项、v2 golden17项）；round15-device1.json真实AscendC npu:1的v2 seed42三shape ok。物理UUID分别为设备0/1不同值，取torch.npu.get_device_properties而不是猜npu-smi逻辑编号。默认lease只同用户，不宣称跨用户/外部任务隔离。NPU性能仍unsupported。

oracle1真实Playwright public browser probe：URL https://esay.app，Chromium130，筛选/详情/重放命令/移动布局全部通过，page_errors=[]。x86_64/Python3.12五个v2 workload golden与aarch64/Python3.11一致，旧v1 seed42同样不漂移。

## 20. 公网与恢复加固证据

隔离根 `/mnt/workspace/af-fuzz/refactor-d0-Nh5vfp`：round18-focused.log=42 passed；round20-focused.log=28 passed；round20-full.log=478 passed/5原有失败。round19额外出现TaskLedger PRAGMA journal_mode=WAL锁冲突，在新并发负载下捕获；storage/sqlite_utils.py统一有限重试后全量不再出现。后续继续保留并发初始化测试。

round16-outage.log：before(total,ACK)=(157,157)，offline=(173,157)且URLError重试；恢复公网30秒后(total,ACK)=(186,183)，新增结果仍产生，证明上传不阻塞执行且自动补传，不应把不同采样时刻status文件的旧ACK视为丢失。

1000例运行目前使用round17/tools/fuzz_acceptance_1000.py，两张NPU各自独立进程及缓存，统一execute路径。初始旧进程nohup导致SIGINT无效，随后仅对已核验PID366428使用SIGTERM，原进程消失后启动新进程；未保存的在途backend由resume重做，已完成记录不重复计数。最终验收需再核对1000唯一seed、2000唯一seed/backend、6000shape和中心ACK。
