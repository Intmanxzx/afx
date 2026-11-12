# CANN 版本来源与历史分组修复

## 采集约定

CPU 与 NPU 后端统一读取 `$ASCEND_HOME_PATH/compiler/version.info` 的 `Version=`。
不调用 `torch_npu.npu.utils.get_cann_version()`，也不根据 `torch_npu` 是否已导入改变来源。
未选择工具包、文件不可读或没有有效版本时记录 `None`，不回退到 runtime 接口或猜测其他安装目录。

## 历史记录

旧采集器可能将同一版本记录为 `9.3.0-beta.1`（文件）或 `9.3.0.beta1`（runtime）。
报表索引只对严格匹配 `X.Y.Z.betaN` 的旧格式转换成 `X.Y.Z-beta.N`，同时用于
`versions` 与 `environment` 两处分组键。不同 beta 编号、正式版和自定义后缀不会混为一组。
原始 contexts、attempts、shapes、结果哈希和同步 ACK 不变。

新写入自动使用兼容规则。旧库先做私有 SQLite 一致性备份，再显式运行：

```python
from autofusedebugger.fuzz.reporting.index import migrate_cann_versions
print(migrate_cann_versions('/absolute/path/to/results.sqlite'))
```

迁移仅重建旧格式记录的派生索引，事务原子、重复运行不再修改记录。
现有完整 `rebuild()` 也使用相同规则。变更日志会通知增量统计移除旧组并统计新组。
大批迁移后应完成 `IncrementalStatistics.refresh()` 预热；超时保留 checkpoint，继续调用，
避免中心缓存指数退避导致上线后长时间没有总览数据。

## 2026-11-11 验证与部署（America/Phoenix）

- 专项及报表/页面测试：109 passed，覆盖文件来源、已导入/未导入 torch_npu、旧记录合组、幂等、
  不可变数据不变和增量统计一致性。
- 扩展本机 fuzz 测试：380 passed / 26 skipped / 35 failed；未宣称全量通过。
  失败包含 macOS SQLite queue 所有权锁冲突和资源预算相关用例；本次未修改这些模块。
- oracle1 当前 `venv-35fe47c` 仅更新 `fuzz/environment.py` 与 `fuzz/reporting/index.py`，
  保留有效 systemd overrides 和其他并行修改。私有备份与验证脚本位于
  `/home/ubuntu/af-fuzz/private/cann-version-1111/`。
- 生产快照和生产库均修复 4867 条派生索引；逐表哈希确认 attempts、contexts、shapes、
  sources、receipts、users、owners 不变。
- `E7ST2L50UUYL83S9650PYPKRAY` 合并为一组 CPU + AF-MLIR，状态 ok，性能 not_requested。
- `npussh mlir` 当前 runtime-round43 安装同步上述两文件；备份在
  `/mnt/workspace/af-fuzz/cann-version-1111/`。真实导入 torch_npu 前后采集均等于该机
  version.info 的 `9.3.0`，禁用 runtime accessor 后仍正常。没有运行额外 NPU kernel，
  没有中断现有 campaign；已有 worker 会在正常回收后加载新代码。

- 统计侧库预热完成，服务恢复 active；无筛选 `/api/summary` 返回 200。
  浏览器目标搜索列表确认为一条记录、AF-MLIR 功能/精度均 Pass；但 q 搜索的汇总仍有
  `database unavailable` 查询超时，列表请求也可能超时。`seed=` 精确筛选的统计接口返回
  单组 ok。这是尚未修复的筛选查询性能问题，不能把本次版本修复描述为全部页面问题已解决。
