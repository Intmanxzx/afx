# 2026-11-24 服务端重构部署

目标 ssh afx，服务 afx-platform.service。部署源码提交 133b89f，wheel SHA256 为 38ceecc917f4648e414d4559aea74bacc30679ba4186860987904444550bc761。独立环境 /srv/afx-platform/deployments/platform-38ceecc917f4648e。

北京时间 09:48:50 停止旧服务，09:49:05 启动新服务。启动前预置 enabled=true、clients=[]，保持所有执行端接入关闭。未停止远端客户端执行任务，尚未放行任何业务客户端。

私有停机备份：/srv/afx-platform/backups/pre-refactor-20261124T014850Z，包含完整数据目录、旧 systemd 单元及启动器、文件摘要清单。三个 SQLite 数据库 integrity_check 均通过。原部署环境保留。

数据库从 16 自动迁移到 17。对停机备份比较：7 用户、28 token 记录、26 客户端、128234 执行记录一致；secret.key 与既有 token 哈希保持一致。

本机及公网 https://autofuse.online 的健康页和首页正常。bootstrap.js 返回 200。临时短期探针凭据验证本机及公网心跳请求均返回 503 client_paused、Retry-After: 60；探针凭据在 finally 中删除，未替换任何用户原 token。

当前运行新服务，空白名单开启。后续先选试点旧客户端验证真实自动补传，再扩大名单。此次未宣称完成线上旧客户端补传验收。

回滚需先停服务，恢复备份的 service 和 launcher，并使用保留的原部署环境；若需要数据库完整回滚，应先另存当前数据，再从该私有备份恢复整个数据目录，不能混搭各数据库。随后 daemon-reload 并启动原服务。旧服务不识别新白名单配置，回滚时需另行维护接入隔离。
