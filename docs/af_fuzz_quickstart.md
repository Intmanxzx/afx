# Fuzz 操作入口

公网结果入口：https://autofuse.online 。总览、测试记录、性能汇总在同一网站中；一条case的后端并排展示。

## 初始化与运行

在已有可用Torch/CANN的NPU环境安装同版本包，管理员离线发放姓名和token：

```sh
autofusedbg fuzz setup --user niuyuhu --token "$FUZZ_TOKEN" --sync-interval 30
autofusedbg fuzz run                 # 持续运行，默认后台
autofusedbg fuzz status
autofusedbg fuzz stop
# 或限定数量：
autofusedbg fuzz run --count 100
```

姓名必须与token对应的管理员记录一致。无需配置服务器、UUID或密钥文件；token内部私有保存，用于自动同步，不输出到结果。命令行token可能短暂出现在进程列表，请使用安全终端。初始化后上传器按默认30秒独立同步，断网保留本地outbox并重试；`--sync-interval`可配置1..86400秒。当前容器无systemd时，重启后需用维护命令恢复上传器，不承诺跨主机重启自动恢复。

每个case必跑CPU和AF-MLIR；`--backend eager-npu`等只增加其他后端。CPU是实际eager精度标杆；AF-MLIR与CPU不一致时独立运行eager-npu复核，一致也算Pass，但保存CPU分歧和误差统计。编译/运行失败不能通过这一规则被掩盖。

资源参数按CPU/cgroup、内存和磁盘自适应保守计算。目前worker串行，不宣称已实现最优并发。未修改共享Torch/CANN/NNSmith。

## 重放与查看

```sh
autofusedbg fuzz replay B000000016
# 定位已知故障时，只执行指定后端：
autofusedbg fuzz replay B000000016 --backend af-mlir --device npu:0
autofusedbg fuzz maintenance snapshot ./results.sqlite ./snapshot.sqlite
autofusedbg fuzz view ./snapshot.sqlite
```

新seed固定10字符，内含协议版本；单seed即可确定图、广播、shape、权重和输入，不依赖缓存或历史库。执行参数不能改变case内容。当前冻结生成协议覆盖小型Add+ReLU/Neg/Mul+Transpose+可选Sum，不是完整算子目录。旧输入不截断以免碰撞。

查看模式不需要Torch、NNSmith、CANN、NPU或Node；前端资源随wheel分发。公开结果快照不含凭据，中心私有灾备含身份与同步元数据，必须私密保存。

## 网站

- 顶部导航；总览图例可切换，测试记录支持任意列模糊搜索。
- 分页大小、列顺序和显隐保存在浏览器。版本列独立且默认隐藏。
- CPU只展示总耗时；总耗时单位秒、1位小数。精度Pass/Failed着色，无边框。
- 功能失败详情显示有界核心日志；精度失败显示误差和二级标准判定。
- 性能仅展示有明确口径的编译/kernel指标，未可靠采集时保持为空，不把首次调用时间当编译时间。

## 服务与能力边界

oracle1使用af-fuzz.service + Caddy，公网80/443，应用loopback8090。服务命令为`autofusedbg fuzz admin serve`。升级前使用`fuzz admin backup`，保留身份、ACK、防重复导入元数据。每日私有备份保留7天，异地备份策略仍需另行配置。部署模板见deploy/fuzz/README.md，完整命令见af_fuzz_cli.md。

AF-MLIR必须加载带MLIR bridge的真实库，严格模式不能静默legacy回退。已验证的隔离运行时为Autofuse c1b46272；默认CANN库不具备该能力。真实测试仍存在Transpose/exec_condition lowering失败，这是测试发现，不用改seed掩盖。

未完成能力：纯CPU编译/设备执行拆分、异进程编译产物移交、准确独立编译/kernel性能、当前CANN下Triton兼容、全机硬资源隔离，以及Windows/macOS原生验收。
