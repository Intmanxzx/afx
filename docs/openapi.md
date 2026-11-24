# OpenAPI 接口

服务端使用 FastAPI + Uvicorn，公开提供 GET /openapi.json，无需 token，直接返回
OpenAPI 3.1 JSON，不套用业务接口的 data envelope。

curl -fsS http://127.0.0.1:8100/openapi.json -o openapi.json

启动命令：

afx server --data DIR serve --host 127.0.0.1 --port 8100

路由、路径和查询参数、请求模型及鉴权依赖在实际 FastAPI 接口中声明；OpenAPI
由框架自动生成，不再维护第二份路径清单。新增接口时使用同一注册方式和 Pydantic
模型即可自动更新文档。注册入口在 autofusex/server/server.py，请求模型在
api_models.py。业务函数继续负责数据库状态、租约和幂等性等校验。

错误仍使用平台 envelope 和原有 HTTP 状态码，参数校验错误返回 400。
公开读取可以不带 token；写操作需要 Bearer token，管理员操作检查 admin 角色。
未知扩展请求字段保留，未提供的可选字段不注入业务请求，避免改变幂等哈希。

响应 envelope 已建模；data 中的数据库投影及配置、运行快照等扩展对象仍是开放结构。
自动生成不意味着框架能推断未声明的业务语义。交互式文档页和 Agent 执行流程未启用。

开发验证：安装项目和 test extra 后运行 pytest tests/platform；测试对实际 HTTP
入口、鉴权、参数校验、幂等行为和 OpenAPI 标准及路由覆盖进行检查。
