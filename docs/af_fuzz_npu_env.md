# af-fuzz：NPU 环境使用指南

在 NPU pod 上跑 `af-fuzz` 的完整操作说明：怎么跑通、怎么复现单个 case、怎么调并发。pod 是动态分配的（`remote_tool.sh connect` 每次可能给不同实例），照着本文档执行即可，不用重新排查环境问题。

## 1. 最小跑通路径

```bash
pip install git+https://gitcode.com/xingzhixiong/Autofuse-Debugger.git
source autofusedebugger/nnsmith_integration/npu_env.sh # 需将其中的python库路径 和 cann包路径换成用例执行环境对应路径
autofusedbg af-fuzz --out-dir /tmp/af_fuzz_run
```

- 第一条命令 **每次新开 shell / 新连 pod 都要重新 source**（见第 2 节，不 source 会在 import 阶段报错）。
- 第二条命令就是一次完整的默认批次：`--n-cases 100 --max-nodes 5 --method symbolic-cinit --tensor-scale small --value-dist-mode uniform --target npu --jobs 2 --soc-tier A3`。这些默认值已经是最常用、最保守的配置，日常跑批次不需要显式写。
- 只有当你明确要偏离默认值时才加对应参数（比如跑 A5、换输入分布、调并发）。参数含义和取值范围见 `docs/af_fuzz_randomization.md`；并发调优见本文档第 5 节。
- **调用方式只能用 `autofusedbg af-fuzz ...` 或 `python3 -m autofusedebugger af-fuzz ...`**。不要用 `python3 -m autofusedebugger.cli af-fuzz ...` —— `cli.py` 没有 `__main__` 入口，这种调法只会 import 一遍就退出（退出码 0、无输出、`--out-dir` 是空目录），容易被误判为"跑完了但没生成用例"。

跑完后 `--out-dir` 下有 `summary.json`（完整结构化结果）和 `summary.html`（可视化报告，字段含义见 `docs/af_fuzz_randomization.md` 的"结果报告字段说明"）。

## 2. 环境变量：为什么必须 source

`npu_env.sh` 做两件事，缺一个都会在不同的 import 时机报错：

1. **`LD_LIBRARY_PATH` 补 `/opt/buildtools/python-3.12.9/lib`**
   venv 的 `python3` 实际链接的是系统 apt 装的 `libpython3.12.so`，但 venv 里的 `_ctypes.so` 扩展模块是跟 `/opt/buildtools/python-3.12.9` 自带的 `libpython3.12` 一起编译的，两者 ABI 不一致。不设置会在 `import torch`（间接触发 `import ctypes`）时报：
   ```
   ImportError: .../_ctypes.cpython-312-aarch64-linux-gnu.so: undefined symbol: _PyErr_SetLocaleString
   ```

2. **`source ~/Ascend/cann/set_env.sh`**
   `torch_npu` 需要 CANN 自己的运行库（`libascend_hal.so` 等）。`/etc/profile` 里本来就有这一句，但只有**登录 shell** 才会执行 `/etc/profile`——用 `ssh host "命令"` 这种非交互方式连接时不会跑，于是报：
   ```
   ERROR: Failed to load the backend extension: torch_npu.
   ```

## 3. Pod 实例不固定

`remote_tool.sh connect --no-sync` 每次调用可能分配到不同的 pod 实例（不同内网 IP），`/tmp` 下任何东西（装好的依赖、解压的源码、上一轮的 dump 目录）都不保证还在。把"传文件 + 装依赖 + 跑 af-fuzz"放进同一次 `connect` 得到的 session 里原子执行，不要假设分几次 connect 还能复用上一次的产物。

## 4. 复现单个 case

每个 case 有一个唯一的 `case_seed`，`summary.json` 里每条记录都带 `case_id` 字段（格式 `"{index:05d}-{case_seed}"`，和失败用例的 `bug-<index>-<case_seed>` 目录名对应）。只要生成参数（`--method`/`--max-nodes`/`--timeout-ms`/`--max-elem-per-tensor`/`--target`）不变，拿着 `case_seed` 就能确定性地重新生成同一张图，不依赖任何落盘文件：

```bash
autofusedbg af-fuzz --out-dir /tmp/repro --target npu \
  --repro-seed 389609433 --max-nodes 4
```

- `--repro-seed` 指定单个 case 的 `case_seed`（`case_id` 里 `-` 后面的数字），只跑这一个 case，不跑整批。
- 这条路径默认打开 debug dump（等价于加了 `--debug`）：把 `model.pth`/`gir.pkl`/`oracle.pkl`/`repro.py` 落到 `case-<case_id>/` 下，并设置 `TORCH_COMPILE_DEBUG=1`、`TORCH_LOGS=+dynamo,+inductor,+graph_breaks`、`TORCHINDUCTOR_CACHE_DIR`/`TORCH_COMPILE_DEBUG_DIR` 指向该 case 自己的子目录，不会和其它 case 混在一起。传 `--no-debug` 关掉 dump，只看 pass/fail 结果。
- 批量跑（不带 `--repro-seed`）时行为不变：只有失败用例才落盘，debug 环境变量默认不打开，避免整批几十上百个 case 都落一份 inductor debug 目录撑爆磁盘。

## 5. 并发调优

### 5.1 按进程分片，不是按线程

`--jobs N` 把 `--n-cases` 平均分成 N 片，每片一个独立进程（独立派生子种子，整批结果仍是确定性的；独立 case 索引区间，落盘目录不冲突；结束后合并成一份 `summary.json`/`summary.html`）：

```bash
autofusedbg af-fuzz --out-dir /tmp/af_fuzz_big --n-cases 200 --jobs 8 --target npu
```

用多进程而不是多线程，是因为 `activate_autofuse_backend` 和 `inductor_autofuse.py` 的 monkeypatch 会修改**进程级全局状态**（环境变量、`NPUScheduling` 的 patch 标记位、几个模块级诊断列表）——同进程内起多线程会互相踩这些全局状态，多进程则每个进程天然独立。`--repro-seed` 单 case 路径不受 `--jobs` 影响。

### 5.2 两个旋钮，两层并发

- **`--jobs N`**（默认 2）：分片进程数。每个进程各自加载 torch/torch_npu，冷启动约 2GB。
- **`--compile-threads M`**（默认 4，等价于环境变量 `TORCHINDUCTOR_COMPILE_THREADS`）：**每个进程内部** inductor 并行编译 worker 池大小，每个 worker 是一个完整 python 进程（约 0.7~1.5GB）。inductor 自己的默认值是 `min(32, 宿主机 CPU 核数)`——容器里 `os.cpu_count()` 报的是宿主机核数而不是 cgroup 配额，冷编译缓存时会起 32 个 worker，在 32GB 容器里直接 OOM，所以必须显式设置这个参数。

优先级：`--compile-threads` 参数 > 环境变量 `TORCHINDUCTOR_COMPILE_THREADS` > 默认 4。峰值内存粗略估算：`1.3GB(基线) + jobs × (2GB + compile-threads × 1.5GB)`。

A5 实测（32GB cgroup、1 块 NPU、**冷编译缓存**、20 个 case；热缓存时基本不再起编译 worker，内存会低得多）：

| jobs | compile-threads | 耗时 | 内存峰值 |
|---|---|---|---|
| 1 | 4 | 211s | 11.0GB |
| 1 | 8 | 188s | 15.9GB |
| 1 | 32（inductor 默认） | 116s 时 OOM | 32GB |
| 2 | 4 | 81s | 17.4GB |
| 4 | 1 | 147s | 11.7GB |
| 4 | 4 | 74s | 28.1GB |

换到内存更大的机器时可以按上面公式提高两个值；`jobs` 从 2 提到 4 在只有 1 块 NPU 的机器上收益很小（81s → 74s），换成多卡或更大内存机器后需要重新实测。

## 附录：已知 flaky 现象（未定位根因）

第一次在某个 pod 上用 `--seed 1 --n-cases 2` 跑出过 2/2 失败（ascendc 输出几乎全零，CPU eager 输出正常非零值），但换一个 pod 用同样的 `--seed 1 --n-cases 2` 重新跑，2/2 全部通过；随后 `--seed 42 --n-cases 20` 也全部通过。目前判断更像是 NPU 运行时首次初始化/编译缓存相关的偶发问题，不是某个 op/dtype 组合的固定数值 bug，但还没有定位到具体机制。如果后续批量跑出现类似"同一 case 复现不稳定"的情况，优先怀疑这个方向，而不是直接当成 ascendc 的真实 bug 去提单。
