# Fuzz 目录迁移验证

2026-11-14（America/Phoenix），通过 npussh mlir 验证当前工作区构建的 autofusex wheel。源码中的 NNSmith 支持和客户端管理已移入 autofusex/fuzz/nnsmith 与 autofusex/fuzz/management。

## 环境

- 先加载 $HOME/env.sh，复用项目 torch210_daily 的 Torch/CANN 运行时，在独立 venv 安装本次 wheel。
- Torch 2.10.0+cpu，torch_npu 2.10.0.post7.dev20261015，CANN 9.3.0，Z3 4.12.2。
- NNSmith 安装 pyproject.toml 固定的 9d7f4d16aaeda931e1d79a2f91dbd09d72f86ab3，源码指纹校验通过。
- 实际加载 CANN 9.3.0 安装目录的 compile_adapter.py 和 libaihac_codegen.so，MLIR bridge 检查通过。
- 检查两张卡与已有任务，开始和结束时均无运行中的 NPU 进程。
- 远端验证目录：/mnt/workspace/af-fuzz/layout-validation-20261114。

## 实际执行

生成配置：small、float32、max_nodes=3、dim_max=8、dynamic_variants=3、entropy=3；图包含 aten.relu.default 和 aten.threshold_backward.default。

| 验证 | 设备 | Shape 数 | 结果 |
|---|---|---:|---|
| 静态 staged replay | npu:0 | 1 | AF-MLIR 对 CPU 精度通过 |
| 动态 staged replay | npu:1 | 3 | 全部 Shape 精度通过 |
| 双卡 bounded run，2 workers、每 worker 2 compile threads | npu:0 / npu:1 | 1 + 3 | 4 个 CPU/AF-MLIR 任务完成，无失败或跳过 |

静态 seed：14615016373650975031143068048542536060102709870595。

动态 seed：F0F89YNDXGKQBLWNGDEOLKCR5DV，输入 Shape 为 (3,)、(16,)、(2,)。

四次 AF-MLIR attempt 均记录 candidates=1、codegen_succeeded=1、no_coverage=false 和 AF_CODEGEN_OK。CPU 记录作为基准，correctness=not_checked，不作为自比精度通过计数。

## 上报与其他检查

独立 loopback 测试中心接收真实执行结果，通过 /api/upload 和阶段事实接口完成 8 条 attempt、91 条事实的上传及 ACK；本地 outbox 和事实队列归零。中心记录与本地逐项核对，HTTP 用例查询返回 4 条记录，数据库 integrity_check 通过。测试服务已关闭，临时 token 已删除；未向线上中心上传，也未清理历史数据。

远端架构、管理和基础回归：71 passed。远端 validation-report.json 保存运行时来源、各次执行精度与覆盖、上报 ACK；migration-tests.log 保存测试结果。验证使用安装后的 wheel，而非源目录覆盖导入。

托管 NPU 性能任务被现有调度器的外部负载协调门禁拒绝，因此本次不声明性能测量通过。本次为目录迁移的有限用例验证，不代表全算子覆盖或长期稳定性验证。
