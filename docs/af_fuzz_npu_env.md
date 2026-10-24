# af-fuzz：A3 NPU pod 环境注意事项

记录在 A3 NPU pod 上实际跑通 `af-fuzz` 时踩到的环境问题，方便下次连到新 pod（pod 是动态分配的，`remote_tool.sh connect` 每次可能给不同实例）时直接照做，不用重新排查。

## 1. 启动前必须 source 的环境

```bash
source autofusedebugger/nnsmith_integration/npu_env.sh
```

这个脚本做两件事：

1. `export LD_LIBRARY_PATH=/opt/buildtools/python-3.12.9/lib:$LD_LIBRARY_PATH`
   venv 的 `python3` 可执行文件实际 `ldd` 链接的是系统 apt 装的
   `/usr/lib/aarch64-linux-gnu/libpython3.12.so.1.0`，但 venv 里的
   `_ctypes.so` 扩展模块是跟 `/opt/buildtools/python-3.12.9` 自带的那份
   `libpython3.12` 一起编译的，两者 ABI 不一致，缺符号。不设置这个会在
   `import ctypes`（间接地，`import torch` 也会触发）时报：
   ```
   ImportError: .../_ctypes.cpython-312-aarch64-linux-gnu.so: undefined symbol: _PyErr_SetLocaleString
   ```
2. `source ~/Ascend/cann/set_env.sh`
   `torch_npu` 需要 CANN 自己的运行库（`libascend_hal.so` 等）。`/etc/profile`
   里本来就有这一句，但只有**登录 shell**才会执行 `/etc/profile`；用
   `ssh host "某条命令"` 这种非交互方式连接时不会跑这一段，于是报：
   ```
   ERROR: Failed to load the backend extension: torch_npu. You can disable extension auto-loading with TORCH_DEVICE_BACKEND_AUTOLOAD=0.
   ```

两个问题缺一个都会在不同的 import 时机炸出来，必须都加上。

## 2. 调用方式：必须走真正的入口点

`af-fuzz` 要通过以下方式之一调用：

```bash
autofusedbg af-fuzz --out-dir ... --n-cases ... --target npu
# 或
python3 -m autofusedebugger af-fuzz --out-dir ... --n-cases ... --target npu
```

**不要**用 `python3 -m autofusedebugger.cli af-fuzz ...` ——
`autofusedebugger/cli.py` 没有 `if __name__ == "__main__":` 入口，`-m` 直接
跑这个子模块只会 import 一遍就结束，不会调用 `main()`，表现是退出码 0、
什么输出都没有、`--out-dir` 目录是空的，非常容易被误判为"跑完了但没生成
任何用例"。真正的入口点是顶层包的 `__main__.py`（`python3 -m
autofusedebugger` 能工作）或 `pyproject.toml` 里注册的 console script
(`autofusedbg` / `autofusedebugger`)。

## 3. Pod 实例不固定，不要假设状态会保留

`remote_tool.sh connect --no-sync` 每次调用可能分配到不同的 pod 实例
（不同内网 IP），`/tmp` 下任何东西（装好的 venv 依赖、解压的源码、上一轮
`af-fuzz` 的 dump 目录）都不保证还在。可靠的做法是把"传文件 + 装依赖 +
跑 af-fuzz"全部放进同一次 `connect` 得到的 session 里原子执行，不要假设
分几次 connect 还能复用上一次的产物。

## 4. 用例 ID、单独复现与 debug dump

每个 case 在生成时都有一个唯一的 `case_seed`（`model_gen` 的随机种子），
`summary.json` 里每条 `CaseResult` 都带 `case_id` 字段（格式
`"{index:05d}-{case_seed}"`，和失败用例的 `bug-<index>-<case_seed>` 目录名
对应）。只要生成参数（`--method`/`--max-nodes`/`--timeout-ms`/
`--max-elem-per-tensor`/`--target`）不变，拿着这个 `case_seed` 就能确定性地
重新生成同一张图，不依赖任何落盘文件：

```bash
autofusedbg af-fuzz --out-dir /tmp/repro --target npu \
  --repro-seed 389609433 --max-nodes 4
```

- `--repro-seed` 指定单个 case 的 `case_seed`（即 `case_id` 里 `-` 后面的
  数字，或 `CaseResult.seed`），只跑这一个 case，不跑整批。
- 该路径默认打开 debug dump（等价于加了 `--debug`），无论这个 case 复现
  时是 pass 还是 fail 都会把 `model.pth`/`gir.pkl`/`oracle.pkl`/`repro.py`
  落到 `--out-dir` 下的 `case-<case_id>/` 里；同时设置
  `TORCH_COMPILE_DEBUG=1`、`TORCH_LOGS=+dynamo,+inductor,+graph_breaks`、
  `TORCHINDUCTOR_CACHE_DIR` 指向该 case 目录下的 `inductor_cache/`，
  `TORCH_COMPILE_DEBUG_DIR` 指向 `torch_compile_debug/`，产物都落在这个
  case 自己的目录里，不会和其它 case 混在一起。传 `--no-debug` 可以关掉
  dump，只看 pass/fail 结果。
- 批量跑（不带 `--repro-seed`）时行为不变：只有失败用例才落盘，debug 环境
  变量默认不打开，避免整批几十上百个 case 都落一份 inductor debug 目录
  撑爆磁盘。

## 5. 并发：按进程分片，不是按线程

`af-fuzz` 原来是单进程单线程顺序跑完 `--n-cases`。现在支持
`--jobs N`，按进程把 `n_cases` 平均分成 N 片并行跑，每片有自己独立派生出
的子种子（从顶层 `--seed` 派生，整批结果仍然是确定性的）和独立的 case
索引区间，落盘目录不会冲突，结束后合并成一份 `summary.json`/`summary.html`：

```bash
autofusedbg af-fuzz --out-dir /tmp/af_fuzz_big --n-cases 200 --jobs 8 --target npu
```

用多进程而不是多线程/单进程多线程，是因为 `activate_autofuse_backend` 和
`inductor_autofuse.py` 里的 Autofuse Inductor hook 会修改**进程级全局状态**
（环境变量、给 `NPUScheduling` 打的 monkeypatch 标记位、几个只用
`threading.Lock` 保护的模块级诊断列表）——同进程内起多线程会在这些全局状态
上互相踩，多进程则每个进程天然拿到自己独立的一份，对并发跑多个 NPU
compile/execute 也更安全。`--repro-seed` 单 case 路径不受 `--jobs` 影响。

### 内存与并发配置（`--jobs` 和 `--compile-threads`）

两个旋钮是两层不同的并发，都要按机器内存调：

- `--jobs N`（默认 2）：分片进程数，每个进程各自加载 torch/torch_npu，
  冷启动时单进程约 2GB。
- `--compile-threads M`（默认 4，等价于环境变量 `TORCHINDUCTOR_COMPILE_THREADS`）：
  **每个进程**内部 inductor 并行编译 worker 池的大小。每个 worker 是一个完整的
  python 进程（import torch），约 0.7~1.5GB。inductor 自己的默认值是
  `min(32, 宿主机 CPU 核数)`，容器里 `os.cpu_count()` 报的是宿主机核数而不是
  cgroup 配额，所以冷编译缓存时会起 32 个 worker，在 32GB 容器里直接 OOM。

峰值内存粗略估算：`1.3GB(基线) + jobs × (2GB + compile-threads × 1.5GB)`。
优先级：`--compile-threads` > 环境变量 `TORCHINDUCTOR_COMPILE_THREADS` > 默认 4。

A5 实测（32GB cgroup、1 块 NPU、**冷编译缓存**、20 个 case；热缓存时基本不再起
编译 worker，内存会低得多）：

| jobs | compile-threads | 耗时 | 内存峰值 |
|---|---|---|---|
| 1 | 4 | 211s | 11.0GB |
| 1 | 8 | 188s | 15.9GB |
| 1 | 32（inductor 默认） | 116s 时 OOM | 32GB |
| 2 | 4 | 81s | 17.4GB |
| 4 | 1 | 147s | 11.7GB |
| 4 | 4 | 74s | 28.1GB |

换到内存更大的机器时，可以按上面的公式提高两个值；`jobs` 从 2 提到 4 在只有
1 块 NPU 的机器上收益很小（81s → 74s），换成多卡或更大内存机器后需要重新实测。

## 6. 已观察到的 flaky 现象（记录，未定位根因）

第一次在某个 pod 上用 `--seed 1 --n-cases 2` 跑出过 2/2 失败（ascendc 输出
几乎全零，CPU eager 输出正常非零值），但在另一个 pod 上用同样的
`--seed 1 --n-cases 2` 重新跑，2/2 全部通过；随后 `--seed 42 --n-cases 20`
也全部通过。目前判断更像是 NPU 运行时首次初始化/编译缓存相关的偶发问题，
不是某个 op/dtype 组合的固定数值 bug，但还没有定位到具体机制。如果后续
批量跑出现类似"同一 case 复现不稳定"的失败，优先怀疑这个方向，而不是
直接当成 ascendc 的真实 bug 去提单。
