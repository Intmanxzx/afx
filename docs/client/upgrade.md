# 客户端安装与升级

客户端使用单安装模式：daemon 从调用 `afx` 的 Python 环境启动，升级在同一环境安装目标版本。已登记执行环境保留 Torch、torch_npu、CANN 等依赖和解释器，worker 优先加载当前安装的 AFX。

## 更新流程

本机使用 `afx update VERSION`，平台管理员也可在运维页面下发目标版本。升级由 daemon 协调、supervisor 执行：

1. 阻止新任务进入并排空已接纳的用例，保存原 run 和队列。
2. 下载目标版与当前版 wheel 及依赖到 `update-artifacts`，准备安装和回退包；准备失败不进入安装。
3. supervisor 关闭 daemon，在原 Python 环境通过 pip 安装目标包并重新启动。
4. 检查解释器、版本、托管连接和需要恢复的任务；恢复任务还需提供 worker 版本证据。
5. 安装或就绪检查失败时，从缓存重装上一版并恢复运行，操作标记失败。

签名发布通过 `afx release stage/activate` 校验并缓存 wheel，再进入同一安装机制。缓存是安装制品，不是另一套可运行客户端。

supervisor 固定代码保存在数据目录的 `supervisor/v2`，协议版本为 3。旧 supervisor 或不同解释器的客户端仍在运行时，新启动会被拒绝，需先完成停机。普通客户端升级不覆盖稳定 supervisor 快照。

## 旧安装接入

旧版客户端先停止任务并等待退出，再停止旧 daemon/supervisor，在选定的唯一 Python 环境中通过 pip 重装。继续使用原数据目录，保留身份、token、配置、结果和上传进度；已初始化目录不要再次 `init`。0.2.4.x 旧升级器不支持直接安装当前版本。

启动不读取 `active-client.json` 或 `active_tool` 来选择其他 AFX，不创建 `client-releases` 控制/执行虚拟环境。磁盘上已有旧目录不会自动删除，因为登记的解释器可能仍引用其中依赖；确认执行依赖已独立且无引用后再清理。

更新安装包不会自动替换已经驻留的旧 Python 进程。日常更换执行环境后，加载相应环境并使用 `afx restart`。

需要完全停机时，先执行 `afx stop` 并等待任务退出，再执行 `afx daemon stop`。`afx daemon start` 只启动后台管理服务，`afx daemon status` 查询后台状态；这三个命令都支持在末尾传入 `--data-dir DIR`。
