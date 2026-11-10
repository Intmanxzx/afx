# Fuzz 可用版操作入口

## 已部署网站

https://autofuse.online 。页面自动刷新30秒；按数字seed筛选后URL可分享，详情可复制单seed replay命令。数据是实际测试结果；历史错误保留，不因重跑成功被覆盖。

## 当前NPU环境

安装只在隔离目录，不修改共享Torch/CANN/NNSmith。可执行包装入口：

```sh
npussh mlir
AF=/mnt/workspace/af-fuzz/refactor-d0-Nh5vfp/bin/autofusedbg-npu
$AF fuzz run --count 10 --backend af-legacy --device npu:0
$AF fuzz status
$AF fuzz replay 340282366920938463463374607431768211498 --backend af-legacy --device npu:0
$AF fuzz sync status --data-dir /mnt/workspace/af-fuzz/refactor-d0-Nh5vfp/acceptance-home
```

包装入口加载既有env.sh、引用共享Torch只读site-packages、设置隔离data home与umask077；应用安装于runtime-venv，上传器用无Torch的light-venv。共享库不安装/替换依赖。直接使用其他data-dir时要对该目录做一次setup。

调整上传周期（默认30秒，范围1..86400秒）：

```sh
$AF fuzz sync configure --data-dir /mnt/workspace/af-fuzz/refactor-d0-Nh5vfp/acceptance-home \
  --endpoint https://autofuse.online \
  --key-file /mnt/workspace/af-fuzz/refactor-d0-Nh5vfp/acceptance-home/upload.key --interval 30
```

密钥已由管理员离线创建，不能把文件内容贴到日志/网页。setup --sync-endpoint可一次完成配置与daemon启动；容器重启后用sync start恢复，当前无systemd、未修改共享登录脚本。

## 重放与查看

`replay SEED`足以确定完整case，backend/device/static-only只选择执行方式或shape子集，不能改变物化内容。当前ETR v2覆盖小型Add+ReLU/Neg/Mul+Transpose+可选Sum，不是全部算子目录。

`autofusedbg fuzz view ./results.sqlite`读取一致性快照，既不运行模型，也不上传。网站中心私有灾备与公开结果快照不同：snapshot只含白名单结果；server-backup包括身份/同步元数据，需要私密保存。

## 服务运维

oracle1：af-fuzz.service + Caddy；应用loopback8090，公网80/443由Caddy终止TLS。babynews已停止/禁用，源码及切换前配置在备份目录保留。每日af-fuzz-backup.timer生成私有备份，保留7天；请另行建立异地备份策略。

升级/恢复/迁移步骤与模板见deploy/fuzz/README.md。当前线上使用venv-round26；原venv保留供回退。无需Docker时同wheel即可；Compose已在备用端口实际验证。

## 明确未承诺的能力

真实编译/执行拆分与异进程编译产物移交、NPU严格性能测量、Triton当前CANN适配、跨用户全机硬资源隔离、Windows/macOS原生运行验收仍未完成。对应入口保守拒绝或报告能力边界，不伪称全功能完成。

## 短seed与后端（本轮更新）

现在可用`autofusedbg fuzz replay B16 --backend af-legacy --device npu:0`；B16无损对应原v2 entropy42。网站、复制命令和seed命令显示短形式；新生成编号最多10字符，内含固定协议版本，无数据库映射。内部数据库保留规范数值用于已有指纹/同步身份，不修改历史数据。

后端入口：cpu（eager CPU）、npu-eager、triton-npu、af-legacy、af-mlir。后三种名称不表示当前安装一定可用；当前MLIR未有安全能力检测，显式unsupported；Triton仍有环境问题。历史ascendc未记录legacy/mlir模式，页面明确提示。

页面时间统一北京时间。点击“原版 HTML · 透视汇总 / 用例列表”使用原有分类、dtype、shape汇总和排序记录表；增加按后端图表及筛选。逐记录链接返回统一详情查看每shape结果。该视图最多5000条，超限请先筛选，未采集的性能数据不填造。

### 当前网站布局（取代上文独立报告入口）

左侧“总览 / 实验记录 / 后端”导航；所有图表、记录和详情在同一套页面与样式中展示，不再打开旧HTML。旧/report书签也进入统一网站。当前线上venv-round29。

默认seed固定10字符：`B000000016`；`fuzz replay B000000016 --backend af-legacy --device npu:0`。更早的短输入和数值输入仍解析同case；历史全128bit随机seed不会为压缩长度而截断。列表中的历史AF模式未记录是数据缺失提示，不是新增执行后端。

### 记录页偏好与测量口径

- 初始化可用 `--user niuyuhu` 指定管理员发放的展示名；公网身份以token对应的管理员记录为准，不可用展示名冒充他人。
- 单框搜索匹配姓名、seed、版本、算子、状态等列；分页大小与列顺序/显隐保存在当前浏览器。
- CPU只显示总耗时（秒，1位小数）。NPU性能仅展示可靠kernel profiler数据；无采集值时默认隐藏。编译耗时未能独立测量时为空，不用prepare/first-call时间替代。

CLI已精简，最新命令表及尚未完成的行为见 `af_fuzz_cli.md`。普通初始化为 `autofusedbg fuzz setup --user NAME --token TOKEN`，默认服务器无需指定。
