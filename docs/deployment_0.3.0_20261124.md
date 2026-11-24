# 0.3.0 服务端部署

2026-11-24 以提交 `0a392147dc91f5d53cd665adbb2bcefd1cb7eeed` 为重构后 0.3.0 版本。本地从 Git 归档的干净源码构建 `autofusex-0.3.0-py3-none-any.whl`，使用发布工具锁定的构建依赖和提交时间作为 `SOURCE_DATE_EPOCH`。

wheel SHA256：`eb20fefd84edffd25bb00bff0d9a9bda6ac807fbc72f0cc47385c3bea1c12bab`。本地文件位于 `dist-0.3.0/`，服务端保留在 `/srv/afx-platform/releases/0.3.0/`。

## 验证

- 本地 wheel 元数据检查、17 项发布工具测试、无模型依赖的独立安装冒烟检查通过。冒烟脚本已改为当前命令和资源，移除对已删除公开 daemon 入口及旧 style.css 的要求。
- npussh mlir 隔离目录 `/tmp/afx-release-030-20261124` 中，服务端、客户端管理、发布和命令路由回归共 242 项通过。测试使用已安装的 0.3.0 及当前源码，加载该机现有的新 SQLite 库以支持 trigram。
- mlir 安装包通过 `afx model run` 在 npu:0、npu:1 分别执行加法模型，均 PASS 1/1、误差为 0，并各生成一个 Autofuse 图。此项为有限冒烟验证，不代表完整硬件覆盖。

## 切换与备份

目标 `ssh afx`，服务 `afx-platform.service`。北京时间 14:54:39 停止旧服务，14:56:02 启动新服务，14:56:04 完成健康检查。

新环境 `/srv/afx-platform/deployments/platform-eb20fefd84edffd2`，包版本及 `afx --version` 均为 0.3.0。

私有停机备份 `/srv/afx-platform/backups/pre-030-20261124T065439Z`，包含完整 data、原 systemd 单元、启动器、SHA256 清单及校验结果。三个数据库的 integrity_check 均为 ok。

候选预检前后 6 用户、27 token、25 客户端和 137090 执行记录逐项一致；数据库 schema 保持 17，secret.key 及 client-access.json 字节一致。启动后再次核对既有 token 的哈希、撤销及有效期信息一致。

本机与公网 healthz、capabilities 正常，bootstrap.js、app.css、app.js、page-ops.js 与本地 wheel 内容逐字节一致。浏览器确认运维页正常加载，无控制台错误。服务 active，NRestarts=0。

观察时执行记录增加至 137178；yunpeng、yuhu、wenfu、zhixiong 均恢复新心跳。shuer 最后心跳 14:49:05，早于本次停服，记录其离线现状，未替用户启停。

## 客户端与回滚

本次没有切换生产客户端，也没有下发升级或启停指令；各客户端版本保持 0.2.4、0.2.4.1 或早期试点 0.2.5。旧版迁移步骤见[0.3.0 发布说明](releases/0.3.0.md)。

如需回滚代码，先停止服务，恢复上述备份中的 afx-platform.service 至 /etc/systemd/system/、afx 启动器至 /srv/afx-platform/bin/，daemon-reload 后启动。旧环境保留，本次没有 schema 迁移，回滚代码时保留当前 data，避免丢失切换后的上传记录。
