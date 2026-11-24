# 客户端维护白名单

默认关闭。按 user_id + machine_id 精确匹配现有客户端身份，不要求旧客户端升级。machine_id 是客户端申报值，白名单是运维控制，不替代 token 认证或提供机器防伪认证。

在新服务启动前应用策略（该命令不打开或迁移数据库）：

    afx server --data /srv/afx/data client-access apply /private/maintenance.json
    afx server --data /srv/afx/data client-access show

先禁止全部执行端接入：

    {"enabled": true, "clients": []}

放行指定客户端，使用客户端记录中的真实 machine_id：

    {"enabled": true, "clients": [{"user_id": "alice", "machine_id": "实际机器标识"}]}

全部恢复：

    {"enabled": false, "clients": []}

apply 原子替换数据目录中的 client-access.json，下次请求生效，无需重启。文件缺失表示关闭，损坏时受控接口拒绝接入。该配置不在数据库备份包中，需另行备份。

受控接口包括 /api/client/*（心跳、上传、复测领取和完成），source 任务领取、完成及产物写入。租约完成请求通过持久化任务反查机器身份。拒绝返回 HTTP 503、client_paused、Retry-After: 60，不撤销 token、不发送停止任务指令。当前上传实现遇到 503 不推进游标；线上实际旧版本的自动补传仍需试点验证。

source-cases/source-results 旧协议没有机器标识，白名单开启时暂时拒绝发布。source 作业创建、管理员页面、查询及个人配置接口不在机器门禁范围，因为它们属于用户操作或旧协议没有机器标识。白名单不是整库只读，备份前仍应停止中心服务。

部署顺序：停止旧服务、做一致性备份、预先开启空白名单、启动新服务完成迁移、放行试点确认补传、逐步放行、最终关闭门禁。白名单修改本身不部署或重启服务。
