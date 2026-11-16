# Fuzz 安装和部署

客户端与服务端使用同一个 wheel。下面命令在目标 Linux 主机执行；wheel 和部署脚本可以单独复制到新机器，不要求保留源码 checkout。已有客户端 home 的配置、token、结果和 ACK 保留。脚本不会创建身份或启动 campaign。

## 客户端

准备 Python 3.11、venv/pip、Git（通过仓库安装依赖时）、已安装的 CANN toolkit 和同版本 graph-autofusion run 包。SQLite 3.31+ 由 `ensure_sqlite.sh` 自动检测；缺失时使用系统包管理器安装，预编译 Python 若固定加载旧私有库会自动切到系统库，不需要源码编译。Torch 为当前协议要求的 2.10.0+cpu，并配套 torch_npu。脚本支持安装本地 runtime wheel，也支持复用现有 Python 的 Torch；不会升级共享 runtime。

~~~bash
# 新环境：换成实际匹配架构和 Python ABI 的 wheel 路径。
bash install_client.sh \
  --wheel /packages/autofusex-0.1.3-py3-none-any.whl \
  --cann-root "$HOME/Ascend/cann" --home "$HOME/af-fuzz" \
  --torch-wheel /packages/torch.whl --torch-npu-wheel /packages/torch_npu.whl

# 已有 Torch 环境。
bash install_client.sh \
  --wheel /packages/autofusex-0.1.3-py3-none-any.whl \
  --cann-root "$HOME/Ascend/cann" --home /data/af-fuzz \
  --runtime-python /path/to/torch-venv/bin/python
~~~

需要 GCC runtime 时传 --library-dir /path/to/gcc/lib64；需要其编译器时传 --compiler-bin /path/to/gcc/bin。NNSmith 由 pip 安装固定的 AutofuseX fork 版本。无法访问 GitHub 时，传 --nnsmith-wheel /packages/nnsmith-0.1.1+autofusex.2-py3-none-any.whl 安装提前构建的 fork wheel。

每次创建以 wheel 摘要区分的 venv；同一目录已存在则拒绝覆盖，可用 --venv /new/path 指定新目录。安装后验证 NNSmith fork、全部 368 条 ATen 规则、compile_adapter 和实际加载的 MLIR codegen 库。验证成功后才更新 HOME/client-env.sh 和 HOME/bin/afx。切换前用旧入口停止 campaign 和同步进程，再用新入口启动。

~~~bash
# 新身份只在首次部署时 setup；已有 home 不重新 setup 或更换 token。
/data/af-fuzz/bin/afx fuzz status
/data/af-fuzz/bin/afx fuzz run --dtype-mode all --save \
  --workers 4 --compile-threads 1 --device npu:0 --device npu:1
~~~

不传 count 为无限运行，不传 ops 使用完整算子目录。安装脚本不修改采样参数，其他范围见 fuzz run -h。

NNSmith 的 AscendC 修改维护在独立 fork 仓库中；客户端直接通过 pip 安装固定 commit。运行时仍校验源码指纹和 Z3 4.12.2，避免导入到其他 NNSmith。

## 服务端

准备 Python 3.9+、venv/pip、systemd、已有服务账户及 sudo/root 权限。SQLite 3.31+ 同样由共享脚本自动检测和安装。Debian/Ubuntu 缺 venv 时先安装 python3-venv。PyYAML、jsonschema、cryptography 由 wheel 自动安装；无需 Torch、NNSmith、CANN 或 NPU。

~~~bash
bash install_server.sh \
  --wheel /packages/autofusex-0.1.3-py3-none-any.whl \
  --root /srv/af-fuzz --user fuzz --port 8090 \
  --database /srv/af-fuzz/data/results.sqlite --service af-fuzz
~~~

脚本创建独立 venv，打印实际安装路径和版本；对已有数据库做 SQLite 在线一致性备份，在私有副本验证服务初始化后，更新 systemd drop-in。数据库、身份和已有 systemd 环境配置（例如 write allowlist）保留；新主机创建 base unit。服务只监听 loopback，启动后请求本机 /api/summary，HTTP 200 且 JSON 可读才成功。公网域名、TLS 和防火墙另行配置。

每次部署的备份与旧 unit 配置保存在 ROOT/private/deploy-*，目录权限 700。失败时检查该目录和 journal；回滚时恢复上一 venv 的 unit 配置，不用旧库覆盖新数据。数据库、token 和备份不可进入镜像或 Git；同机备份还需另做异地复制。

## 其他部署方式

- Dockerfile / compose.yaml：根据 wheel 更新版本；创建数据目录并赋予容器用户 UID 10001 写权限。
- Caddyfile：HTTPS 代理；核对域名、80/443 端口和证书持久化目录。
- systemd 模板及备份 timer：根据安装路径调整 Python、数据库、用户和备份目录。

身份、环境准备和备份恢复见 [Fuzz 运维说明](../../docs/fuzz/operations.md)，日常任务见 [Fuzz 使用说明](../../docs/fuzz/fuzz_usage.md)。

开发回归测试另需 Node.js 18+；运行服务使用预构建前端资源，不依赖 Node。请完整复制 scripts 目录，两个安装入口都依赖相邻的 ensure_sqlite.sh。
