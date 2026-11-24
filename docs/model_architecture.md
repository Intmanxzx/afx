# 模型分析架构

模型能力位于 `autofusex/workloads/model/`，负责已有脚本、工厂入口模型的运行、图采集、精度定位及报告。它既可通过 `afx model run` 直接执行，也可作为固定源码或整网任务由客户端调度。

## 执行与采集

`script_runner.py` 识别框架并选择 adapter；`torch_model.py` 处理 `create_case` 等工厂入口。当前自动执行以 PyTorch 为主，TensorFlow/JAX adapter 保留接口但尚未实现执行与精度采集。`agent.py` 与 `normalization.py` 提供脚本归一化和失败诊断。

PyTorch 路径使用 `torch.compile` 与 `inductor_npu_ext` 的 Autofuse 后端。采集同次编译的 pre-fusion FX 图、真实 AOT 输入、编译 provenance 和融合输出；CPU 与 NPU eager 回放对应图和输入。NPU 融合输出通过编译证据映射到原始图出口，不能按输出序号猜测对应关系。

`inductor_autofuse.py` 与 `autofuse_graph.py` 负责图和运行产物。公共图解析、数值比较和环境采集位于 `autofusex/execution/`，HTML 报告由 `autofusex/viewer/report.py` 生成。具体映射规则见[精度映射](accuracy_mapping.md)，编译链路见[Inductor / AscendC 流程](inductor_ascendc_pipeline.md)。

## 固定源码任务

`workloads/imported/registry.py` 登记不可变源码包；导入阶段校验入口和内容摘要，不运行脚本。`workloads/model/task.py` 使用登记的版本、解释器与任务参数执行模型，获取设备租约，保存实际软件环境、执行结果和制品摘要。进程组、超时、停止和内存限制由 `execution/source_task.py` 管理。

固定源码任务当前使用 `af-legacy` 后端与单个所选设备。`passed`、`accuracy_failed` 和 `execution_failed` 分别表示执行结果；任务调度完成不等于精度通过。中心保存该类任务及报告，独立于 fuzz 小图通过率。

## 离线分析与产物

`afx analyze manifest/compare/compare-three/compare-npy/report` 消费图、manifest 和 tensor 数据，输出 JSON、CSV 或 HTML。GE dump 的 origin 属性属于保留的离线数据适配，不是当前自动模型 runner 的主图来源。缺失映射、文件或不支持格式必须在结果中明确反映。

`afx debug camodel info/run` 检查并调用外部 CANN simulator；具体模型执行和 dump 命令由调用方提供。

当前命令示例见[模型分析使用说明](usage.md)，采集字段见[Manifest 格式](manifest_schema.md)，报告示例见[功能展示](feature_gallery.md)。
