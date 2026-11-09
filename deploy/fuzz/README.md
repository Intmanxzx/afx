# 单应用 + SQLite 部署

同版本 wheel 同时用于 NPU 端、离线查看和中心。不安装 Torch/NNSmith/Node：

```sh
python3 -m venv venv
venv/bin/pip install --no-deps ./autofusedebugger-0.1.2-py3-none-any.whl
venv/bin/autofusedbg fuzz admin-user ./data/results.sqlite '唯一展示名' ./private/upload.key
venv/bin/autofusedbg fuzz serve ./data/results.sqlite --port 8090
```

key 文件为0600。管理员通过安全渠道离线分发，不在线申请；展示名不鉴权。配置Caddy TLS反代127.0.0.1:8090，不能直接暴露该端口。公开页面只读；上传要求Bearer密钥，上传体最多2MiB/100条。应用最多4个处理线程和12个排队连接，Caddy不记录Authorization。公网数据默认可见，长日志/tensor不上传；不要在短错误/环境字段中放秘密。

## 自动同步

```sh
autofusedbg fuzz sync configure --data-dir ./fuzz-data --endpoint https://esay.app --key-file ./upload.key --interval 30
autofusedbg fuzz sync run --data-dir ./fuzz-data
```

将sync run配置成systemd服务（用户/路径选当前环境；Restart=on-failure、UMask=0077），首次setup后长期运行，不逐批publish。同步独立于测试，故障重试不阻塞执行。结果attempt不可变，新结果新UUID；source/递增revision/本地ACK和中心receipt持久化。重复ACK丢失可幂等重试。每周期最多20批×50条，周期可配置1..86400秒。配置变更下一周期生效。

## 备份、恢复和升级

公开轻量快照：`fuzz snapshot`，不含身份和ACK。
服务器灾备必须：`fuzz server-backup results.sqlite private-backup.sqlite`，包含用户hash/source/receipt。文件必须0600并保存在私有目录。

恢复：停止应用，在新目录放完整备份，以同版本wheel启动；不能把活动主库直接覆盖并保留旧WAL。先在不同端口验证healthz/摘要/已有上传重试，再切TLS upstream。客户端ACK领先于旧备份时拒绝继续，不静默丢失或重编号；必须恢复匹配备份或管理员执行显式重同步流程（自动重置epoch尚未实现）。

升级：先完整backup并保存当前wheel SHA256/代理配置；装入新venv，在备用端口读恢复副本跑回归后切服务，保留旧venv/backup可回退。结果schema目前1；新未知版本拒绝，不自动猜测迁移。任务schema1→2用独立upgrade-queue，不用于中心结果库。

Docker可选：把wheel放本目录，`mkdir data; chown 10001 data; docker compose up --build -d`。固定Python镜像标签和应用版本；Linux host networking仅监听loopback，TLS代理在host。生产建议进一步记录镜像digest。不需要Redis/Postgres。Compose路径需另行实测，不能以本机pip部署成功代替Docker验证。

容器无systemd时使用 `autofusedbg fuzz sync start --data-dir ./fuzz-data`，独立session+flock运行；重复start不会启动第二个上传器。可配置容器entrypoint启动它，但本工具不擅自改用户shell启动文件。`sync configure` 后需首次start或安装系统服务，之后不逐批操作。status读取持久化短状态。公网部署实测使用pip+systemd+Caddy；NPU容器使用POSIX sync daemon，二者不能混称systemd均可用。

## 首次设置与恢复操作

一次完成本地setup和30秒后台上传（不需逐批操作）：

```sh
autofusedbg fuzz setup --data-dir ./fuzz-data --key-file ./upload.key --sync-endpoint https://esay.app --sync-interval 30
autofusedbg fuzz sync status --data-dir ./fuzz-data
```

`sync status`返回当前待传条数/ACK/上次状态，密钥不输出。`sync configure --interval 5 ...`可调整为5秒；现有daemon下一周期读取，不需要重启。页面每30秒刷新，上传30秒意味着最坏通常需再等一个页面周期，不能宣称严格5秒实时。

中心恢复旧备份后如client ACK领先，**确认恢复完成后**在客户端显式执行：

```sh
autofusedbg fuzz sync reset-epoch --data-dir ./fuzz-data
```

该操作持上传锁，保留所有本地结果，创建新source epoch并重传轻量记录；中心按不可变attempt ID+内容+归属去重，不重复统计。不要删除本地尚未完成灾备同步的结果。

离线管理员轮换泄露/到期密钥：`fuzz admin-rotate-key results.sqlite USER_UUID ./new-upload.key`。旧key立即失效，用户名和所有历史归属保留；安全分发新文件并重新configure。密钥丢失时不提供在线申请。

附带`af-fuzz.service`和`af-fuzz-sync.service`模板。前者路径与oracle1实际部署一致；后者需改站点路径。PID1不是systemd的NPU容器请用`sync start`并由容器entrypoint在重启时调用，daemon单实例锁保证幂等；本工具不会修改共享shell配置。

Compose已在oracle1备用端口实际build/up/healthz，私有备份恢复核对37条结果通过；当前公网由pip+systemd承载。固定镜像/应用版本与实际wheel SHA256一并记录于验收文档；不能仅凭版本号判断开发期wheel相同。

## 使用入口

```sh
# 生成并后台执行100个新自包含数字seed，限定当前ETR v2候选范围。
# 建议先运行少量确认环境；不将此命令当作1000例验收证据。
autofusedbg fuzz run --count 100 --backend ascendc --device npu:0 --data-dir ./fuzz-data
# 加 --backend cpu-eager 可为同一seed同时入队独立CPU参考任务。
autofusedbg fuzz status --data-dir ./fuzz-data
# 网站复制出的一个seed即可重新生成相同case；backend/device只改变执行。
autofusedbg fuzz replay SEED --backend ascendc --device npu:0
# 查看快照无Torch、CANN、NNSmith或Node依赖。
autofusedbg fuzz view ./results.sqlite
```

当前ETR v2范围是小型固定协议：Add和ReLU/Neg/Mul分支、显式Transpose(0,1)、可选Sum规约、三组shape；**不代表Elementwise/Reduce类别下所有算子已覆盖**。普通运行默认后台托管；单queue内仍保守串行worker，不宣称已实现CPU编译/NPU执行解耦流水线。不同设备可使用独立queue并行，物理UUID lease避免同工具同卡竞争。性能目前只提供CPU同步wall-time，NPU严格kernel-profiler测量入口保持拒绝。
