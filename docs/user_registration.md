# 申请加入与用户管理

公开内容无需登录。需要接入客户端或参与问题处理时，在登录窗口点击“申请加入”，填写用户名。

提交后保存申请进度链接；链接包含领取凭证，有效期为提交后 7 天，请勿分享。浏览器会记住最近一次申请，也可以从“申请加入”页查看。管理员批准后，打开进度页领取 token，明文只展示一次；保存后即可使用 token 登录网页或初始化客户端：

~~~sh
afx init --token YOUR_TOKEN
afx start
~~~

如果领取响应因断网而丢失，请联系管理员重置；系统不会重复返回已签发 token。申请链接遗失、申请过期且账号尚未创建时，可以等待旧申请过期后重新申请；已批准创建账号的，请联系管理员。

## 管理员操作

管理员登录后进入“用户管理”：

- 待审核申请：查看用户名，选择 developer 或 admin，批准或拒绝。默认 developer。
- 已有用户：启用、禁用、调整角色；仅在用户遗失 token 时重置。

访客无需账号。新账号只提供 developer 和 admin 两种角色；历史 viewer 账号保持原状态，不自动提升权限，管理员可按需调整。

已有账号和 token 不因功能上线、角色调整或启停而重新签发。禁用立即拒绝后续鉴权请求，启用后原 token 可继续使用。重置会撤销旧 token，新的明文只在管理员操作成功后展示一次，由管理员私下交给用户。最后一个启用的管理员不能在用户管理页面被禁用或降级。

已初始化的客户端不要再次 init。重置凭据后，停止客户端与 daemon，在其数据目录的私有 client.json 中更新 token，再执行 start；不要将配置或 token 放入公开日志。

## API

响应沿用平台 envelope。以下匿名 POST 接口要求 application/json：

| 接口 | 请求 | 返回 |
|---|---|---|
| /api/registration/apply | user_id | receipt、expires_at |
| /api/registration/status | receipt | user_id、status、expires_at |
| /api/registration/claim | receipt | user_id、token（一次性） |

申请状态包括 pending、approved、rejected、claimed、expired。进度凭证放在页面 URL fragment 中，通过请求体发送，服务端只保存其摘要。不能通过用户名查询申请或领取凭据。

以下接口要求管理员 Bearer token：

| 接口 | 用途 |
|---|---|
| GET /api/admin/users | 待审核申请及已有用户，不包含凭据 |
| POST /api/admin/users/review | id、action（approve/reject）、role（默认 developer） |
| POST /api/admin/users/manage | user_id、action（enable/disable/role/reset-token），role 操作另传 role |

批准不签发 token；申请人领取时原子签发并消费凭证。重置 token 会同时关闭该用户尚未领取的申请。现有用户无需迁移凭据。

公开申请按 IP 每分钟 3 次、全站每分钟 30 次限流（令牌桶），最多保留 1000 条未完成申请；查询与领取按 IP 每分钟 60 次限流。反向代理应覆盖客户端传入的 X-Forwarded-For，只有本地代理的转发地址被信任。管理员审核、用户修改和 token 签发写入审计日志，日志不记录完整凭据。
