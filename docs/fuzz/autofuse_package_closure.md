# Autofuse Python 包遮蔽修复

2026-11-11（America/Phoenix）在 mlir 定位并修复 `autofuse.compile_adapter` 导入失败。
并非机器没有安装依赖，而是 `PYTHONPATH` 首位的 `autofuse-build-python/autofuse`
只有 `__init__.py` 和 `pyautofuse.so`。Python 选中普通包后，不会继续在后面的同名
包中查找子模块；主进程与 Inductor 编译子进程都因此看不到已安装的 compile_adapter。

## 环境修复

- 在 `/mnt/workspace/af-fuzz/package-closure-1111/runtime/python/site-packages/autofuse`
  创建完整运行包：沿用原 `mlir-runtime/python/site-packages/autofuse` 已部署的 Python
  辅助模块，保留此前选中的 `pyautofuse.so`，不替换为 CANN 自带或后来构建的另一份 binary。
- binary SHA256 不变：`6d5a81a18bb17b4587077ddf18406ec89d95a1fc08123a8d63037c0197168162`。
- `runtime-manifest.json` 记录文件哈希；正式 autofusedbg-npu 入口改为优先使用完整包。
  不修改全局 Python 安装，不修改上游 Autofuse 工作区的未提交代码。
- 私有任务/结果一致性备份和旧入口在同目录 `private/`。暂停 feeder、等待既有任务自然完成，
  切换后按原身份、batch、预算、双卡配置恢复 campaign。历史失败结果不改写。

## 代码防回归

AF 后端在设备初始化前导入编译 adapter 并检查 `build_tiling_obj`、`build_kernel_obj`、
`build_kernel_so`，包不完整时立即报告实际选中的包路径和遮蔽原因，不再等首次编译才失败。
仍保留原有 MLIR bridge 验证，不以 compile_adapter 可导入替代 MLIR 能力证明。

## 验证

- 子进程回归复现“残缺包遮蔽完整包”，检查缺 API 和传递依赖错误。
- mlir 包检查及 dtype 测试：22 passed。
- 主进程、新 Python 子进程均从完整包成功导入 adapter。
- 双卡重放此前相同 seed，120 秒超时不变，CANN 日志开启：
  - E9SQECC3GJ1J8KJBR868L9N2QS，npu:0：静态 Shape 精度 Pass。
  - F9SQECC3GJ1J8KJBR8694VSOGP，npu:1：3 个动态 Shape 精度全部 Pass。
- 隔离验证数据位于 `package-closure-1111/validation/results.sqlite`，没有混入生产上传。
- oracle1 同步 AF 后端的包检查代码，保留其他已部署页面与统计修改。

复测 execution_total：静态 20.996 秒，动态三 Shape 24.546 秒；这是含冷启动及编译的
执行阶段耗时，不是 Kernel 耗时或整个上传链路耗时。
恢复后的 campaign 在 npu:0 / npu:1 均产生了新的 AF-MLIR ok 结果，ACK 已推进到 1730。
本机专项 20 passed；额外架构检查仍有上游 dtype 修复引入的既有失败：
TorchAdapter 导入 autofusedebugger.compare，与旧架构测试约束冲突。本次未扩大范围修改。
