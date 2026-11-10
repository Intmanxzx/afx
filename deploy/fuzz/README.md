# Fuzz 服务部署模板

操作说明、身份发放、备份恢复和升级步骤统一见 [Fuzz 使用说明](../../docs/af_fuzz_usage.md#二云端身份汇总网站)。本目录仅保留部署文件，不维护另一份设计或测试流水账。

- `Dockerfile` / `compose.yaml`：固定 Python 镜像与 wheel 版本；将对应 wheel 放到本目录，升级时同步修改版本。Linux host network 供宿主机 TLS 代理访问 loopback 服务。
- `Caddyfile`：HTTPS 代理；使用前核对域名/DNS、80/443 端口和证书持久化目录。
- systemd 模板：无 Docker 时使用同一个 wheel；根据安装路径调整 Python、数据库及用户。
- 备份脚本及 timer：部署前核对 `AF_FUZZ_ROOT`、`AF_FUZZ_PYTHON` 和备份目录，设置权限并安排异地复制。同机备份不是灾备。

不要把数据库、token、私有备份放入镜像或 Git。Compose 启动前创建数据目录并赋予容器用户 UID 10001 写权限；wheel 需自行构建或从可信 Release 下载。本目录模板不会自动覆盖当前线上安装。
