from __future__ import annotations

from pathlib import Path

from autofusedebugger.html import write_report_html


def test_write_report_html_embeds_rows(tmp_path: Path) -> None:
    report = {
        "schema": "autofusedebugger-report/1",
        "rtol": 1e-3,
        "atol": 1e-3,
        "summary": {
            "total": 1,
            "passed": 0,
            "failed": 1,
            "max_abs_error": 0.25,
            "max_rel_error": 0.5,
        },
        "graphs": {
            "cpu": {
                "side": "cpu",
                "model": {"name": "m"},
                "nodes": [{"name": "Add_1", "type": "Add", "outputs": []}],
                "edges": [],
                "fusions": [],
            },
            "npu": {
                "side": "npu",
                "model": {"name": "m"},
                    "nodes": [
                        {
                            "name": "autofused_add_ge_0123456789abcdef0123456789abcdef__ge_2",
                            "type": "AscBackend",
                            "outputs": [],
                            "origin_ops": ["Add_1", "Ge_2"],
                            "unfused_reasons": [
                                {
                                    "category": "lowering_fallback",
                                    "target": "aten.mul.Tensor",
                                    "source_node": "mul_source",
                                    "reason": "performance of non-contiguous stride (16, 2) maybe worse than eager",
                                }
                            ],
                            "origin_subgraph": {
                                "source": "CANN GE PreRunBegin",
                                "nodes": [
                                    {"name": "Add_1", "type": "Add", "outputs": []},
                                    {"name": "Ge_2", "type": "Ge", "outputs": []},
                                ],
                                "edges": [{"src": "Add_1", "dst": "Ge_2", "dst_input": 0}],
                            },
                        }
                    ],
                "edges": [],
                "fusions": [
                    {
                        "fused_op": "autofused_add_ge_0123456789abcdef0123456789abcdef__ge_2",
                        "origin_op": "Add_1",
                        "origin_index": 0,
                    }
                ],
            },
            "autofuse": {
                "graphs": [
                    {
                        "name": "graph_hint_0_B0Y0_S0G0C0",
                        "kernel_name": "autofused_add_ge_0123456789abcdef0123456789abcdef",
                        "variant_name": "ascgraph_AutoFuseAfterOptimize_0",
                        "source": "/tmp/autofused_add_ge_0123456789abcdef0123456789abcdef/ascgraph_AutoFuseAfterOptimize_0.txt",
                        "source_relative": "autofused_add_ge_0123456789abcdef0123456789abcdef/ascgraph_AutoFuseAfterOptimize_0.txt",
                        "nodes": [],
                        "edges": [],
                        "queues": [],
                        "buffers": [],
                    }
                ]
            },
        },
        "results": [
            {
                "passed": False,
                "status": "OK",
                "cpu_op": "Add_1",
                "cpu_index": 0,
                "npu_op": "autofused_add_ge_0123456789abcdef0123456789abcdef__ge_2",
                "npu_index": 0,
                "npu_type": "AscBackend",
                "shape": [2, 2],
                "dtype": "float32",
                "max_abs_error": 0.25,
                "max_rel_error": 0.5,
                "rmse": 0.125,
                "cosine_similarity": 0.99,
                "cann_cosine_similarity": None,
                "cann_unavailable_reason": "NONFINITE_INPUT",
                "mismatched": 1,
                "num_elements": 4,
                "finite_pair_count": 3,
                "max_error_index": [0, 1],
                "nonfinite": {
                    "present": True,
                    "cpu": {"nan": 0, "pos_inf": 0, "neg_inf": 1},
                    "npu": {"nan": 0, "pos_inf": 0, "neg_inf": 1},
                    "aligned": {"nan": 0, "pos_inf": 0, "neg_inf": 1},
                    "different_positions": 0,
                    "first_index": [1, 1],
                    "first_cpu": "-Inf",
                    "first_npu": "-Inf",
                },
            }
        ],
    }
    html_path = tmp_path / "report.html"
    write_report_html(report, html_path)
    content = html_path.read_text(encoding="utf-8")
    assert "Autofuse Accuracy Report" in content
    assert "autofused_add_ge_0123456789abcdef0123456789abcdef__ge_2" in content
    assert "errorChart" in content
    assert "CPU&amp;NPU 精度对比" in content
    assert "npuGraph" in content
    assert "CPU Dump Graph" not in content
    assert "zoom-controls" in content
    assert "highlight-controls" in content
    assert "联动高亮" in content
    assert "detailsPanel" in content
    assert "tensorViewer" in content
    assert "tensorMatrix" in content
    assert "openTensorViewer" in content
    assert "tensor-mark" in content
    assert "查看输出 Tensor 逐元素精度" in content
    assert "tensorRowsAndCols" in content
    assert "绿色=PASS，红色=FAIL" in content
    assert "data-highlight-mode=\"depth1\"" in content
    assert "data-highlight-mode=\"depth2\"" in content
    assert "data-highlight-mode=\"all\"" in content
    assert "FUSED" in content
    assert "graphBack" in content
    assert "fusionDetails" in content
    assert "enterFusionGraph" in content
    assert "showMainGraph" in content
    assert "showFusionDetails" in content
    assert "selection.detailNode = graphContext.parent" in content
    assert "父融合算子的精度和输入输出详情" in content
    assert "graphContext.kind === 'fusion'" in content
    assert "融合子图" in content
    assert "点击蓝色入口进入融合子图" in content
    assert "function enterFusionNode" in content
    assert "pointerFusionEntry" in content
    assert "selectedNode && selectedFusionEntry" in content
    assert "data-fusion-entry=\"true\"" in content
    assert "activateNode(name)" in content
    assert "initialParams.get('view') === 'fusion'" in content
    assert "fusionGraph" not in content
    assert "renderFusionSubgraph" not in content
    assert "fusionSection" not in content
    assert "CANN GE PreRunBegin" in content
    assert "Coverage" in content
    assert "UNMATCHED_CPU_OP" in content
    assert "NO MAP" in content
    assert "NO DUMP" in content
    assert "NPU_OUTPUT_NOT_CAPTURED" in content
    assert "NOT_OBSERVABLE_ON_NPU" in content
    assert "融合内部不可观测" in content
    assert "not_observable" in content
    assert "CPU 对照" in content
    assert "rtol / atol" in content
    assert "逐元素对比容差" in content
    assert "tolerance_source" in content
    assert "fitGraph" in content
    assert "applyGraphView" in content
    assert "pointerdown" in content
    assert "pointermove" in content
    assert "pointerup" in content
    assert "Math.hypot" in content
    assert "setPointerCapture" in content
    assert "target.addEventListener('click'" not in content
    assert "ResizeObserver" in content
    assert "event.ctrlKey" not in content
    assert "fallback.push" not in content
    assert "data-zoom-action=\"fit\"" in content
    assert "svg.setAttribute('viewBox'" in content
    assert "const skipsLevel" in content
    assert "width=\"100%\" height=\"100%\"" in content
    assert "view.x = Math.min(target.clientWidth - edge" in content
    assert "style.transform" not in content
    assert "will-change: transform" not in content
    assert "npuGraphZoom" in content
    assert "graph-node.related" in content
    assert "graph-node.dimmed" in content
    assert "一层输入输出" in content
    assert "两层输入输出" in content
    assert "全部输入输出" in content
    assert "traverse(incoming)" in content
    assert "traverse(outgoing)" in content
    assert "current.depth >= maxDepth" in content
    assert "URLSearchParams" in content
    assert "AutoFuseAfterOptimize" in content
    assert "graphTabs" in content
    assert "graph-tab-group" in content
    assert "graph-kernel-tab" in content
    assert "graph-variant-tab" in content
    assert "linkedAutofuseButton" in content
    assert "查看 AutoFuseAfterOptimize" in content
    assert "linkedAutofuseIndices" in content
    assert "kernelNameFromText" in content
    assert "data-kernel-first" in content
    assert "lifecycleTimeline" in content
    assert 'data-lifecycle-kind="queue"' in content
    assert 'data-lifecycle-kind="buffer"' in content
    assert 'data-lifecycle-kind="all"' in content
    assert "lifecycleFilter" in content
    assert '<details class="autofuse-analytics"' in content
    assert "Queue / Buffer 生命周期" in content
    assert "autofuse-analytics-content" in content
    assert "analytics.open = false" in content
    assert ".autofuse-analytics[open] { flex-basis: 440px; }" in content
    assert "height: var(--lifecycle-height, clamp(260px, 32vh, 340px))" in content
    assert "scrollbar-gutter: stable" in content
    assert ".lifecycle-scroll { height: 0; min-height: 0; flex: 1 1 auto; overflow: auto" in content
    assert "const scrollTarget = selectedResource || firstRelated" in content
    assert "lifecycleResizeHandle" in content
    assert "initLifecycleResize" in content
    assert "startHeight + startY - event.clientY" in content
    assert "expandLifecycleHeight" in content
    assert "lifecycle-expanded" in content
    assert "initialParams.get('lifecycleHeight')" in content
    assert "graph.source_relative || graph.source" not in content
    assert "renderLifecycleTimeline" in content
    assert "queueTimelineRows" in content
    assert "bufferTimelineRows" in content
    assert "producer_nodes" in content
    assert "consumer_nodes" in content
    assert "Resource instance" in content
    assert "topological node order" in content
    assert "selectLifecycleRow" in content
    assert "setLifecycleHover" in content
    assert "updateLifecycleSelection" in content
    assert "centerGraphNode" in content
    assert "lifecycleKind" in content
    assert "lifecycleResource" in content
    assert "initialParams.get('lifecycle') === 'open'" in content
    assert "QUE SHARED" in content
    assert "QUE REUSED" in content
    assert "Queue 节点分类" in content
    assert "精度结果分类" in content
    assert "逐元素 / 整体均通过" in content
    assert "整体 Tensor 失败" in content
    assert "逐元素对比通过" in content
    assert "整体 Tensor 对比通过" in content
    assert "CosineSimilarity（余弦值相似度）" in content
    assert "RelativeEuclideanDistance（欧氏相对距离）" in content
    assert "KullbackLeiblerDivergence（KL散度）" in content
    assert "Precision: 逐元素" in content
    assert "metric-group-row" in content
    assert "逐元素精度对比指标" in content
    assert "整 Tensor 精度对比指标" in content
    assert '<th colspan="10"></th><th colspan="6" class="elementwise-group">逐元素精度对比指标</th><th colspan="3" class="tensor-group">整 Tensor 精度对比指标</th>' in content
    assert content.index('data-key="mismatched"') < content.index('data-key="cann_cosine_similarity"')
    assert content.index('data-key="max_error_index"') < content.index('data-key="cann_cosine_similarity"')
    assert "overflow-x: auto" in content
    assert "table { width: max-content" in content
    assert "tableHorizontalScrollbar" in content
    assert "tableHorizontalScrollbarContent" in content
    assert "initTableHorizontalScrollbar" in content
    assert "scrollbar.scrollLeft = wrap.scrollLeft" in content
    assert "wrap.scrollLeft = scrollbar.scrollLeft" in content
    assert "detailsResizeHandle" in content
    assert "initDetailsResize" in content
    assert "details-wide" in content
    assert "最大有限误差位置" in content
    assert "非有限值统计" in content
    assert "首个非有限值位置" in content
    assert "cannMetricText" in content
    assert "mismatchText" in content
    assert "maxIndexText" in content
    assert "-Inf" in content
    assert "value === undefined || value === null || value === ''" in content
    assert "strict_passed" in content
    assert "cann_passed" in content
    assert ".load-legend[hidden] { display: none; }" in content
    assert "loadLegend.innerHTML = '<span class=\"resource\">Queue 同一 que_id 同色；Buffer 同一 buf_id 同色" in content
    assert "loadLegend.innerHTML = '<span class=\"pass\">逐元素 / 整体均通过" in content
    assert "queue_id=" in content
    assert "buffer_id=" in content
    assert "reuse_id=" in content
    assert "depth=" in content
    assert "const queueClassColors" in content
    assert "const resourcePalettes" in content
    assert "resourceColor('queue', queue.id)" in content
    assert "resourceColor('buffer', buffer.id)" in content
    assert "broadcastCacheConditions" in content
    assert "hasBroadcastCache" in content
    assert "broadcast-cache-mark" in content
    assert "broadcast-cache-label" in content
    assert "lifecycle-node-head${cached ? ' broadcast-cache' : ''}" in content
    assert "Broadcast 条件缓存" in content
    assert "Broadcast cache" in content
    assert "Execution condition" in content
    assert "unfusedReasons" in content
    assert "unfusedReasonText" in content
    assert "unfused-mark" in content
    assert "unfused-label" in content
    assert "NOT FUSED" in content
    assert "未融合原因" in content
    assert "Lowering fallback" in content
    assert "Implicit fallback" in content
    assert "source: ${item.source_node}" in content
    assert content.count("unfusedReasons(node).forEach") == 1
    assert content.index("unfusedReasons(node).forEach") < content.index("grid.innerHTML")
