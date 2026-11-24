# 单安装客户端与旧入口清理

本次替代 0.2.5 早期试点保留旧升级入口的方案。旧客户端停止后，通过 pip 重装；不再支持 0.2.4.x 升级器直接安装此版本。

客户端只从调用 afx 的 Python 安装启动。稳定 supervisor 使用 supervisor/v2（协议 3），以本次调用的解释器启动 daemon，不读取 active-client.json 或 active_tool 选择其他 AFX。仍在运行的旧 supervisor 或不同解释器客户端会阻止新启动，要求先停止。supervisor 只保存必要控制代码，不保存完整客户端或执行环境。

注册执行环境保留 Torch/CANN 等依赖及解释器，worker、环境探测、复测进程通过 PYTHONPATH 优先加载当前安装的 AFX。升级不再创建 client-releases 控制和执行 venv。

update 先下载目标版和当前版的 wheels（含依赖），排空任务后由 supervisor 关闭 daemon，在同一 Python 环境执行 pip 安装，再启动并验证。失败时从缓存重新安装上一版。缓存位于 update-artifacts，是安装包，不是可运行的另一套客户端。无法取得回退包时不进入安装阶段。release 保留签名校验，缓存签名 wheels，并通过相同安装机制激活；不再创建 tools/.../venv。

删除旧 fuzz/daemon/runtime 导入包、根模块 daemon run 转发、client 的旧 API 转发、旧设备配对/轮询协议、旧 central 数据模型、空 uploads、platform_agent 及迁移脚本。公开 device 命令先前已删除，内部旧入口也一并删除。

保留 seed 中冻结的历史算子名称，它是用例身份协议。保留原数据读取及路径整理逻辑，身份、token、配置、结果和上传进度不删除。旧 active-client.json、client-releases 和 supervisor/v1 即使留在磁盘也不再参与启动；本次不会自动删除这些目录，因为已注册解释器可能仍引用其依赖。运营重装时应先把执行依赖注册到独立环境，确认无引用后再清理旧安装备份。

重装操作：使用旧版本先停止任务，等待退出，再停止旧 daemon/supervisor；在唯一选定 Python 环境安装新包，使用同一个 data-dir 启动。不要重新 init 已初始化目录。旧进程未退出时，pip 安装不会把常驻旧进程自动变成新版本。

## 验证

- mlir 管理及发布工具测试 113 项通过。
- 干净源码构建 wheel，确认不存在旧 fuzz 包及删除的 client 模块；发布验证器新增拒绝旧 fuzz 包检查，防止 build 缓存污染。
- 隔离 venv 内真实 update --wait：设置失效旧活动指针和 active_tool 后仍从当前解释器启动；pip 安装与重启成功，PID 更新，解释器不变，没有 client-releases 或额外运行环境。
- 注入不存在的目标版本模拟安装失败，真实从 wheel 缓存恢复原版并在同一环境重新启动，操作标记 failed，client_switch=rolled_back。
- 本轮未重新部署生产客户端或服务端；早期 0.2.5 生产试点仍使用旧方案，不代表已经完成此次重装。
