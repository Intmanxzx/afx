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
| `--n-cases` | `100` | 本批次生成并验证的用例总数 |
| `--max-nodes` | `5` | 单个用例计算图节点数的**上限**；每个 case 的真实节点数会在 `[2, max-nodes]` 区间内随机抽取，不是固定等于这个值（至少 2 个节点才有融合的意义） |
| `--method` | `"symbolic-cinit"` | nnsmith 图生成算法，一般不需要改 |
| `--tensor-scale` | `"small"` | `small`/`medium`/`large` 三档，每档同时固定 `--timeout-ms`/`--max-elem-per-tensor`/维度取值区间 3 个值，具体数值见本表后面的"`--tensor-scale` 三档具体数值"表；`small` 等价于改动前的硬编码默认行为 |
| `--timeout-ms` | `None`（跟随 `--tensor-scale` 档位） | 单 case 生成超时（毫秒）。不传时使用 `--tensor-scale` 档位自带的值（见后面的档位数值表），显式传值会覆盖档位值 |
| `--max-elem-per-tensor` | `None`（跟随 `--tensor-scale` 档位） | 单张量最大元素数上限。不传时使用 `--tensor-scale` 档位自带的值（见后面的档位数值表），显式传值会覆盖档位值 |
| `--value-dist-mode` | `"uniform"` | 权重/输入的数值分布档位，四选一：`uniform`（原始 `[3,7)` 均匀分布，默认兼容旧行为）/ `log_scale`（跨 `1e-4~1e4` 量级）/ `llm`（模拟 LLM 激活值分布：主体 `N(0, 0.2²)`，5% 元素随机放大 100~10000 倍，模拟"massive activations"现象）/ `recommendation`（模拟推荐系统归一化特征，`uniform[0,1)`）。具体分布公式见 `docs/af_fuzz_randomization.md` |
| `--seed` | `None`（随机） | 整批的顶层随机种子，固定后同一批次的全部 case 可复现（按进程分片时各分片派生独立子种子，但总体仍确定性） |
| `--target` | `"npu"` | 执行目标后端 |
| `--jobs` | `2` | 按进程切分 `--n-cases`，并行跑 N 个进程。每个 worker 真实占用 NPU 编译/执行资源，偏重 CPU+内存，建议先用低值观察负载再往上调；并发调优细节和实测数据见 `docs/af_fuzz_npu_env.md` 第 5 节 |
| `--compile-threads` | `None`（跟随环境变量 `TORCHINDUCTOR_COMPILE_THREADS`，否则 `4`） | **每个** `--jobs` 进程内部 inductor 并行编译 worker 池大小。inductor 自身默认值是 `min(32, 宿主机 CPU 核数)`，在容器里容易把内存撑爆，所以这里改成了偏保守的默认 `4` |
| `--soc-tier` | `"A3"` | `A3` / `A5` 二选一，决定用哪一代 NPU 的融合算子白名单来约束图生成（见 `ascendc_opset.py`）；在 A5 硬件上验证时要显式传 `A5` |
| `--repro-seed` | `None` | 只跑这一个 `case_seed`，不跑整批（见第 3 节） |
| `--debug` / `--no-debug` | 不传（`None`） | 是否开启调试落盘。只在配合 `--repro-seed` 时生效：不传时 `--repro-seed` 自动当作开了 `--debug`；显式 `--no-debug` 可以在单 case 复现时关掉落盘，只看 pass/fail（见第 3 节） |

`--tensor-scale` 三档具体数值：

| 档位 | `max_elem_per_tensor`（单张量最大元素数） | `concr_ph_dim_rng`（每个维度大小的随机取值区间） | `timeout_ms`（单 case 生成超时） |
|---|---|---|---|
| `small`（默认） | `2^16` = 65536 | `(1, 64)` | `10000`（10s） |
| `medium` | `2^20` ≈ 1,048,576 | `(8, 4096)` | `30000`（30s） |
| `large` | `2^24` ≈ 16,777,216 | `(8, 8192)` | `60000`（60s） |

`concr_ph_dim_rng=(l, r)` 是输入占位符**每一维**大小的随机区间（`random.randint(l, r)`），各维连乘得到总元素数；一旦某一维累乘会超过 `max_elem_per_tensor`，那一维会退化成 `1` 而不是真的生成超大张量。档位越大，单 case 生成的张量可能越大、形状越多样，但耗时和显存/内存占用也越高——批量跑时优先用 `small`，确认要测大 shape 场景时再切 `medium`/`large`。

## 2. 输出结果

### 2.1 输出文件

`--out-dir` 目录下固定产出：

| 文件/目录 | 内容 |
|---|---|
| `summary.json` | 本批次全部 case 的结构化结果（`total`/`passed`/`failed`/`gen_errors` 汇总 + 每条 `CaseResult`） |
| `summary.html` | 可视化报告，字段见下面 2.2 |
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

### 2.2 `summary.html` 字段说明

| 列名 | 对应字段 | 含义 |
|---|---|---|
| `#` | `index` | case 在本批次里的序号（从 0 开始） |
| 状态 | `status` | `pass`（通过验证）/ `fail`（生成成功但验证不一致或执行异常）/ `gen_error`（图生成阶段本身失败，没走到验证） |
| Seed | `seed` | 这个 case 的 `case_seed`，决定它的全部随机内容；用 `--repro-seed <这个值>` 单独复现 |
| 节点数 | `n_nodes` | 生成模型中的计算节点数（不含 `Input` 占位符，`gen_error` case 为 `0`） |
| Kernel数 | `n_kernels` | 这个 case 最终真正生成并在 NPU 上执行的 kernel 总数。**这个数字本身不能单独说明"有没有发生融合"**：`n_nodes` 比 `n_kernels` 大，说明至少有节点被融合进了同一个 kernel；`n_nodes == n_kernels` 通常意味着没有发生融合，但也可能是节点之间本身就没有可融合的依赖关系（比如两条独立并行的分支），不能只凭这一个数字下结论。`gen_error` case 为 `0` |
| 耗时(s) | `duration_s` | 这个 case 从开始生成到验证完成（或失败）的总耗时 |
| Symptom | `symptom` | 只有 `fail` 才有值：`exception`（执行时抛异常）/ `segfault`（进程崩溃）/ `inconsistency`（NPU 和 CPU eager 输出数值不一致）/ `timeout`（执行超时）/ `worse_perf`（性能回退，当前流程不触发） |
| Stage | `stage` | 只有 `fail` 才有值：`compilation`（编译阶段）/ `execution`（执行阶段）/ `verification`（数值校验阶段） |
| 算子集合 | `op_types` | 这个 case 用到的算子类名去重列表（同一种算子出现多次只算一条） |
| Input Shape | `input_shapes` | `{输入张量名: shape}`，`gen_error` case 为空 |
| Output Shape | `output_shapes` | 同上，针对输出张量 |
| 复现路径 | `bug_dir` | 失败用例的 `repro.py` 链接；`pass`/`gen_error` 没有 |

点击表头可排序，顶部可按算子名子串 / 状态筛选。

## 3. 失败用例如何复跑

### 3.1 复跑方式

从 `summary.html`（或 `summary.json` 的 `case_id`/`seed` 字段）拿到失败 case 的 `case_seed`，用 `--repro-seed` 单独重跑：

```bash
autofusedbg af-fuzz --out-dir /tmp/repro --target npu \
  --repro-seed 389609433 --max-nodes 4
```

- 只要生成参数（`--method`/`--max-nodes`/`--timeout-ms`/`--max-elem-per-tensor`/`--target`/`--tensor-scale`/`--value-dist-mode`/`--soc-tier`）和原批次保持一致，`--repro-seed` 就能确定性地重新生成同一张图，不依赖原批次的任何落盘文件。
- 复跑只跑这一个 case，终端会打印：
  ```
  case_id=00000-389609433 status=fail symptom=inconsistency stage=verification
  artifacts: /tmp/repro/case-00000-389609433
  ```
- 也可以直接运行失败目录里现成的 `repro.py`（`bug-<index>-<case_seed>/repro.py`），不需要重新走 `--repro-seed`。

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
