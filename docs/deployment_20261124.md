# 2026-11-24 服务端重构部署

目标 ssh afx，服务 afx-platform.service。部署源码提交 133b89f，wheel SHA256 为 38ceecc917f4648e414d4559aea74bacc30679ba4186860987904444550bc761。独立环境 /srv/afx-platform/deployments/platform-38ceecc917f4648e。

北京时间 09:48:50 停止旧服务，09:49:05 启动新服务。启动前预置 enabled=true、clients=[]，保持所有执行端接入关闭。未停止远端客户端执行任务，尚未放行任何业务客户端。

私有停机备份：/srv/afx-platform/backups/pre-refactor-20261124T014850Z，包含完整数据目录、旧 systemd 单元及启动器、文件摘要清单。三个 SQLite 数据库 integrity_check 均通过。原部署环境保留。

数据库从 16 自动迁移到 17。对停机备份比较：7 用户、28 token 记录、26 客户端、128234 执行记录一致；secret.key 与既有 token 哈希保持一致。

本机及公网 https://autofuse.online 的健康页和首页正常。bootstrap.js 返回 200。临时短期探针凭据验证本机及公网心跳请求均返回 503 client_paused、Retry-After: 60；探针凭据在 finally 中删除，未替换任何用户原 token。

当前运行新服务，空白名单开启。后续先选试点旧客户端验证真实自动补传，再扩大名单。此次未宣称完成线上旧客户端补传验收。

回滚需先停服务，恢复备份的 service 和 launcher，并使用保留的原部署环境；若需要数据库完整回滚，应先另存当前数据，再从该私有备份恢复整个数据目录，不能混搭各数据库。随后 daemon-reload 并启动原服务。旧服务不识别新白名单配置，回滚时需另行维护接入隔离。

## yuhu 旧客户端试点

仅放行 user_id=yuhu、machine_id=a887d216-bacf-4571-bfb8-90828bf52942（中心 client_id=24）。mlir 本地 source_id 与此一致，运行旧版 0.2.4.1，未升级或重启任务。

放行前执行记录 7594，之后观察到 7822（增加 228），新增数据仅来自 client_id=24，包括维护期间生成的积压记录。心跳继续更新，daemon_state=running。运行任务本地上传游标 6927、当时本地最大 rowid=6930，积压基本追平。11 条本地 rejected 均为 device busy 的环境错误过滤，没有上传格式拒收。旧任务进程已连续运行约 14 小时。

白名单继续开启且只含上述一台客户端；其他客户端未放行。该试点验证了旧版 0.2.4.1 对新服务的真实心跳、自动补传和持续上报。

## 全量恢复接入

yuhu 试点通过后分两批放行：shuer（0.2.4）、zhixiong（0.2.4.1），随后 wenfu、yunpeng（均 0.2.4.1）。五台均恢复心跳；观察窗口内 shuer 新增 294 条、zhixiong 44 条、yunpeng 279 条执行入库。wenfu 状态 stopped，没有新结果，未替用户启动任务；zhixiong 补传后也是 stopped。yuhu 持续上报。

确认五台心跳均在 120 秒内后，将 client-access.json 设置 enabled=false、clients=[]，恢复正常认证接入。恢复前五客户端策略留存于 /srv/afx-platform/backups/rollout-five-client-policy.json。服务 active，NRestarts=0，公网 healthz 正常。客户端未统一升级或重启。
