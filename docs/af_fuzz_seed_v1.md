# 数字 seed：一个值完整重放

当前默认入口：

```sh
autofusedbg fuzz replay 42
# 只运行基础样本，不改变图、权重、输入或完整 workload ID
autofusedbg fuzz replay 42 --static-only
# 换后端，仍是同一用例
autofusedbg fuzz replay 42 --backend ascendc
# 不执行模型；标准库即可检查完整用例规格与指纹
autofusedbg fuzz seed 42 --show-spec
```

无需 profile、版本参数、生成配置、数据库或原机器。`replay 42` 默认运行同一图/权重的全部三个确定样本；当前默认后端是 cpu-eager，NPU 通过 backend 参数选择，设备选择不改变用例。

## 新契约与破坏性变更

按用户要求全新重构，不兼容先前设计：长格式 `v1:preview-…`、static/dynamic profile、`run --seed-token`、旧 `af-fuzz` 子命令/console entry 均不再支持。数字 seed 42 **不是**旧 batch seed，也不是旧长 token 的别名。旧协议黄金向量已撤下，当前黄金向量为 `tests/fixtures/fuzz/numeric_v1_vectors.json`。

历史 NNSmith 模块/测试暂留用于 D0 故障调查，新的 fuzz CLI 不再调用它们；非 fuzz 调试命令不受此破坏性变更影响。旧诊断文档的命令和指纹属于历史证据，不是当前用法。

## 自包含编码

```
seed = (generation_version - 1) * 2^128 + entropy
```

接受非负整数或规范 ASCII 十进制字符串（不接受符号、空格、前导零）。版本 1 占用 0..2^128-1，42 永远解码为版本 1/entropy 42。更高区间目前明确报 unsupported version，不能回退到版本 1。

后续生成规则改变必须实现新版本，保留**从本轮确定的新数字协议开始**的旧实现和向量。全新重构取消的是废弃设计兼容，不取消未来数字 seed 的长期稳定性要求。UI/JSON 中 seed 用十进制字符串保存，避免 JavaScript 大整数精度丢失。

## 固定生成算法

字段随机源：

```
SHA256(ASCII("autofusedbg/numeric/v1\0") || ASCII(decimal_seed) || 0x00
       || ASCII(field_name) || 0x00 || uint64_big_endian(attempt))
```

digest 是 big-endian uint256；N 个候选使用 rejection sampling，接受 `< 2^256 - 2^256 mod N`，结果 mod N，attempt 从零增加。不使用时间、Python hash、库 RNG、Z3、设备能力或外部文件。

固定因素顺序：dtype(float16/float32)、broadcast_axis(0/1)、fixed_size/base_size(2/3/4)、input_layout(contiguous/transposed)、operation(relu/neg/mul)、reduction(none/axis0/axis1)。

用例为 `x + weight -> relu|neg|mul(weight) -> optional sum(axis)`。weight 在广播轴上恒为 1；基础输入这条轴为 base_size，变体为 base_size+1/+2。三组输入、同一图、同一权重一起构成完整用例；不再存在“静态 seed 和动态 seed 分别生成”。

所有 storage 元素按 `storage/{id}/element/{index}` 取整数 -8..8，再除以 8。规范存储为 little-endian IEEE float16/float32。weight ID=0，样本 ID=1..3。CaseBundle 是 frozen spec + tuple[bytes]；验证存储引用、对齐、view 边界。无需落盘。

当前有界 dyadic 算术可精确表示；独立 scalar reference 版本为 `af-mini-reference-v1-exact-dyadic`。不是通用浮点 oracle。内容 workload ID 只包含规范规格和存储长度/SHA256，不包含执行模式/后端/设备/时序。

## 执行语义

`--static-only` 只选择样本 0，并让编译使用 static shape；其余两个样本依然属于完整 bundle。默认使用 dynamic 编译并遍历所有样本。两种模式都不修改图、权重、基础输入或 workload ID。

后端适配器只处理准备/绑定/执行/同步/输出检查，不改变生成。正确性和性能沿同一循环，逐 shape 分别记录；首次调用为 combined，不把 lazy callable 当编译完成。CPU 可选 wall-time 测量；NPU 严格计时因设备 lease 未实现而明确 unsupported。不同后端使用相同 bundle，设备不支持就报错/unsupported，绝不重抽 seed。

此版仍是有限算子/布局/shape 子集预览：未完成跨架构验收、完整 NPU 因素矩阵、Triton 可用环境、任意 alias/state、编译产物跨进程移交、后台批测/同步/三 OS 验收。不能以本地算法推导代替跨设备/长期版本实测。

## 结果归档（D3 首批）

replay 默认将轻量结果写到 data home/results/results.sqlite，可用 `--database ./results.sqlite` 指定位置或 `--no-save` 显式禁用。只保存 seed/指纹/上下文/每 shape 状态指标和短错误，不保存 CaseBundle/tensor。

```sh
autofusedbg fuzz replay 42 --database ./results.sqlite
autofusedbg fuzz results ./results.sqlite --limit 50
autofusedbg fuzz snapshot ./results.sqlite ./snapshot.sqlite
python -m autofusedebugger.fuzz results ./snapshot.sqlite
autofusedbg fuzz view ./snapshot.sqlite
```

查询、快照与 view 不需要 Torch/NNSmith/CANN/NPU/Node。view 默认在当前电脑打开浏览器，只监听本机随机端口；可用 --no-browser 禁止自动打开、--port 指定端口，Ctrl+C 退出。前端随同一 wheel 分发，不需另装前端。请先将一致性快照复制到个人电脑，再运行 view；这不是个人 NPU 常驻查看服务。只打开可信来源的结果文件；Windows/macOS 的实际运行验收仍待 CI。

## 恢复与合并可信结果

```sh
autofusedbg fuzz check-db ./snapshot.sqlite
autofusedbg fuzz restore ./snapshot.sqlite ./restored.sqlite
autofusedbg fuzz import ./snapshot.sqlite ./existing-results.sqlite
```

restore 仅创建新文件，不覆盖；import 要求目标已存在，整批事务，重复 attempt 幂等、冲突拒绝。不是恢复/运行 fuzz 任务，不上传，不导入旧 summary。check-db 校验完整性及轻量记录 hash；它不验证发送者身份，只打开可信来源文件。当前 schema1，未知 schema 明确拒绝，升级不自动重写源文件。

## 前台 CPU 批测（D4 预览）

```sh
autofusedbg fuzz run 42 43 --foreground --backend cpu-eager --backend cpu-compile --data-dir ./fuzz-data
autofusedbg fuzz status --data-dir ./fuzz-data
autofusedbg fuzz stop <task-id> --data-dir ./fuzz-data
autofusedbg fuzz resume <task-id> --data-dir ./fuzz-data
# resume 只重新排队，再显式执行：
autofusedbg fuzz run --foreground --data-dir ./fuzz-data
```

目前仅 POSIX、串行 CPU worker；`--timeout` 每 task 默认120秒。后台托管和 NPU 全局设备租约未完成，所以 `--foreground` 必填且此入口不接受 NPU backend。停止/中断不会自动重跑；结果已保存而未确认的 attempt 在恢复检查后直接完成，不重复实验。run 的 completed 指已落库，不等于正确性通过；用 results/view 看逐后端结果。多 seed/backend 分别入队，容量或后续出错时已接受的任务保留，用 status 检查，不要盲目重交整批。


## 后台 CPU 批测（最新）

```sh
autofusedbg fuzz run 42 43 --backend cpu-eager --data-dir ./fuzz-data
autofusedbg fuzz status --data-dir ./fuzz-data
# 仍可选择终端内等待：
autofusedbg fuzz run --foreground --data-dir ./fuzz-data
# 已有上一阶段schema1操作库时，先停止所有相关supervisor/worker：
autofusedbg fuzz upgrade-queue ./fuzz-data/tasks/tasks.sqlite ./tasks-v1.backup.sqlite
```

本节取代上文“--foreground必填”和部分入队说明：默认后台有限批次，所有新任务原子入队；超容量/非法规格不会半批提交。启动失败后已接受的批次可保持pending，用status确认，再不带seed的run，不要盲目重交。启动ACK不是正确性成功。后台任务完成后进程退出，不是常驻个人NPU查看API。仍仅POSIX/串行CPU；NPU托管不接受，查看模式跨平台设计不受影响。schema2操作库保存短reason，结果库仍schema1。升级必须显式新备份，不覆盖；回退前终止相关进程并使用匹配旧版本及备份，不混用活跃数据库和WAL文件。

## 托管资源保护（采样，不是硬隔离）

```sh
autofusedbg fuzz setup --data-dir ./fuzz-data --cpu-workers 1 --compile-threads 2 \
  --memory-budget-bytes 4294967296 --cache-budget-bytes 1073741824
autofusedbg fuzz run 42 --data-dir ./fuzz-data
```

数值仅为示例，请按机器调整。配置过大超出探测容量时拒绝执行；CLI --compile-threads 可下调不可突破已配置上限。cpu_workers目前是上限，实际仍串行1个。Linux可按进程组RSS采样，超过memory预算中断；cache预算针对每attempt私有临时目录，结束清理，不影响用户既有缓存。采样有间隔和遍历开销，可能短暂越限，不是硬配额。直接replay不使用此托管策略。NPU托管/严格性能租约仍待实现。

## v2 Elementwise + Reduce + Transpose

v2数字范围为2^128..2^129-1。数字本身仍自包含，无额外profile/生成参数；v1 seed含义不变。v2在Elementwise后执行显式transpose(0,1)，再按seed决定是否/在哪个轴Reduce。独立分类Transpose，不属于View。输入/权重字节沿v1 entropy映射，输出spec/oracle随新版本冻结。网站复制重放命令无需人工填写版本参数。
