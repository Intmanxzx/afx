# 原始上传包存储

新上传使用 contents.codec=upload-refs-v1：压缩保存原始字段及 log、detail、case.spec 的不可变内容 ID。引用内容按原始值保存，不依赖 cases 当前版本。get_content 还原规范化 JSON 并校验原始 SHA-256；旧 raw/zlib 内容继续兼容。客户端协议及详情读取格式不变，历史数据不迁移。

备份必须包含 core.db 与 content.db。未来内容垃圾回收必须遍历 upload-refs-v1 的 refs，不能只扫描 core 表中的 content_id。引用缺失、内容损坏或还原哈希不匹配均报错，不返回伪造的部分原始包。

写入新格式后，回退版本必须保留新版 store.py 解码能力；仅回退 ingest.py 可停止新格式写入。不能直接回退到不识别 upload-refs-v1 的旧程序。数据库完整性检查不替代内容还原哈希检查。
