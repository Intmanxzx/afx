# Fuzz 空环境部署与兼容清理

本次版本只读取 result schema 2 和当前 E/F seed 协议。客户端与服务端必须一起更新，不能原地打开旧结果库。旧 seed（C–J 内测格式、C2/D2 别名和旧十进制编码）、telemetry v1、旧后端别名均不再接受。任务队列 v1 升级和旧报表库迁移代码已经删除。

## 已移除的历史路径

- 任务队列升级脚本及 maintenance upgrade-queue 命令。
- 旧报表表结构迁移副本、CANN 历史拼写修正和历史 source 推断。
- 到达时间 v1 转换、旧快照临时视图、缺失入库时间时使用客户端时间。
- 旧机器心跳的名称补填、仅为旧不完整环境记录提供的清理函数。
- 旧 seed 解析、旧图描述放宽校验、旧记录不可重放的前端分支。
- cpu-eager、npu-eager、ascendc、triton 的历史后端名称入口。
- telemetry v1，以及 MLIR 子分类依赖旧 debug 日志的猜测路径。MLIR 子分类要求 MLIR Codegen Result 明确证据；缺少证据显示未确认。
- 服务端缺少问题索引时逐页扫描并临时聚合的备用实现。新中心初始化即建立查询模型。

## 保留的当前功能

af-legacy 是当前明确支持的 Autofuse 后端模式。cpu-compile 是 CPU 编译诊断后端，具有实际执行行为和测试覆盖，保留。combined 与 staged 是仍可选择的执行模式，它们的设备互斥检查不能删除。环境 compatibility_profile 用于匹配当前 Python/CANN/Torch 组合。SQLite CTE 适配用于受支持运行环境，不涉及历史业务数据。

物化索引的重建和脏数据回退仍用于当前数据的一致性；快照、恢复、同步 epoch 重置、重复上传回执和删除墓碑仍用于正常恢复和防止数据复活。阶段事实、生命周期和诊断可选，因为当前未完成任务、未采集性能和独立 replay 也会产生这些情况。测试历史、多次执行和复测记录属于产品数据，不是旧版本兼容。

共享内容的 SQL 视图用于关联当前规范化表，不保存第二份旧数据。视图的写入触发器是当前存储接口，不是旧数据迁移。

## 部署顺序

1. 停止所有客户端管理守护进程、fuzz/retest supervisor、worker 和 sync daemon；停止服务端及自动备份任务。核对没有遗留执行进程。
2. 在服务停止后保留一份私有、完整的一致性备份，用于回到对应旧程序；新程序不导入其中的结果。
3. 删除服务端整个持久化数据目录，包括主库、WAL/SHM、runtime/public-statistics 侧库、设备注册、会话、回执、复测、问题领取、修复记录、上传暂存和缓存。Docker 应重新创建专用数据卷，而不只是删除主库。
4. 每个客户端清空实际 data home，包括 results、tasks、manager.sqlite、同步/outbox/ACK、运行产物、日志、配对凭据、缓存和 active-home 指针。重装环境后重新设置本机 trust 和配置；不要恢复旧 manager.sqlite 或 config/source ID。
5. 安装同一新版本，从空目录启动中心，建立用户，重新 enroll/pair 每个客户端。检查空首页、新 token 登录、新设备心跳、少量新结果上传以及重复上传 ACK。
6. 完成验证后再启动持续任务。任何旧客户端都不能保留运行或上传。

## 可选保留用户 token

用户 token 的明文由用户保存；中心只保存 SHA256。停服后，只将旧中心 users 表的 id、display_name、key_hash、active 四列私下复制到新空中心同名表即可保留原 token。不要复制 sources、设备 token、sessions、owners、receipts 或任何结果表。导出文件按凭据材料保管，权限 0600，不输出到日志。新客户端使用原用户 token 重新注册，获得新设备凭据和同步身份。

这是部署时的一次性数据操作，不加入运行时兼容层。若旧 users 表不符合这四列契约，直接创建新用户并重新发 token。本次代码清理没有操作任何线上数据，也没有实际导出 token。
