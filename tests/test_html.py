from __future__ import annotations

import json
from pathlib import Path

from autofusedebugger.html import write_report_html, write_report_html_from_json


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
                        "nodes": [
                            {
                                "id": 0,
                                "name": "graph_hint/data",
                                "type": "Data",
                                "inputs": [],
                                "outputs": [{"place": "GM", "tensor_id": 0}],
                                "raw_fields": [
                                    ".ir_attr.index = i: 0",
                                    ".y: f32",
                                    "    .mem = GM[tensor_id=0, pos=GM]",
                                ],
                            }
                        ],
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
                "cann_unavailable_reason": "FINITE_SUBSET_ONLY",
                "mismatched": 1,
                "num_elements": 4,
                "finite_pair_count": 3,
                "nonfinite_aligned": True,
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
    assert "fusion-mark" in content
    assert "function graphNameLines(value)" in content
    assert "function graphNodeHeight(node)" in content
    assert "levelHeights" in content
    assert "<tspan x=\"${position.x + 10}\"" in content
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
    assert "function rawAutofuseFields(node)" in content
    assert "原始图字段" in content
    assert "raw-node-fields" in content
    assert "inputItems.concat(outputItems, rawFieldItems)" in content
    assert "function autofuseFieldLines(node)" not in content
    assert 'xml:space="preserve"' not in content
    assert ".ir_attr.index = i: 0" in content
    assert ".y: f32" in content
    assert ".mem = GM[tensor_id=0, pos=GM]" in content
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
    assert "const strictResult =" in content
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
    assert ">NF</text>" in content
    assert "未融合原因" in content
    assert "Lowering fallback" in content
    assert "Implicit fallback" in content
    assert "source: ${item.source_node}" in content
    assert content.count("unfusedReasons(node).forEach") == 1
    assert content.index("unfusedReasons(node).forEach") < content.index("grid.innerHTML")


def test_write_report_html_streams_script_safe_json(tmp_path: Path) -> None:
    report = {
        "summary": {"total": 0, "passed": 0, "failed": 0},
        "graphs": {},
        "results": [],
        "text": "</script><script>alert(1)</script>",
    }
    target = tmp_path / "streamed.html"

    write_report_html(report, target)

    content = target.read_text(encoding="utf-8")
    assert "<\\/script>" in content
    assert "</script><script>alert(1)</script>" not in content
    assert '"text":"<\\/script><script>alert(1)<\\/script>"' in content


def test_write_report_html_deduplicates_tensor_payloads(tmp_path: Path) -> None:
    encoded = {
        "encoding": "base64",
        "dtype": "float32",
        "logical_dtype": "float32",
        "shape": [2],
        "data": "AACAPwAAAEA=",
    }
    row = {"elementwise_data": {"shape": [2], "cpu": encoded, "npu": encoded, "strict_mask": encoded}}
    report = {
        "summary": {"total": 1, "passed": 1, "failed": 0},
        "graphs": {},
        "results": [row],
        "comparison_order": ["cpu_npu_autofuse"],
        "comparisons": {
            "cpu_npu_autofuse": {
                "right_graph_key": "npu",
                "summary": {"total": 1, "passed": 1, "failed": 0},
                "results": [row],
                "graphs": {"unused": True},
            }
        },
    }
    target = tmp_path / "deduplicated.html"

    write_report_html(report, target)

    content = target.read_text(encoding="utf-8")
    assert content.count("AACAPwAAAEA=") == 1
    assert '"tensor_ref":"tensor_0"' in content
    assert '"tensor_payloads":{"tensor_0"' in content
    assert '"unused":true' not in content


def test_write_report_html_externalizes_large_tensor_payloads(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AUTOFUSEDEBUGGER_EMBED_TENSOR_BYTES", "0")
    encoded = {
        "encoding": "base64",
        "dtype": "float32",
        "logical_dtype": "float32",
        "shape": [2],
        "data": "AACAPwAAAEA=",
    }
    row = {
        "npu_op": "add",
        "elementwise_data": {"shape": [2], "cpu": encoded, "npu": encoded, "strict_mask": encoded},
    }
    report = {
        "summary": {"total": 1, "passed": 1, "failed": 0},
        "graphs": {},
        "results": [row],
    }
    target = tmp_path / "lazy.html"

    write_report_html(report, target)

    content = target.read_text(encoding="utf-8")
    tensor_script = tmp_path / "lazy_data" / "tensor_0.js"
    assert "AACAPwAAAEA=" not in content
    assert '"external_script":"lazy_data/tensor_0.js"' in content
    assert content.count('"tensor_ref":"tensor_0"') == 3
    assert "loadTensorPayload" in content
    assert "return new Constructor(buffer)" in content
    assert tensor_script.is_file()
    assert "window.__autofuseRegisterTensor" in tensor_script.read_text(encoding="utf-8")
    assert "AACAPwAAAEA=" in tensor_script.read_text(encoding="utf-8")


def test_write_report_html_removes_stale_tensor_data_when_report_has_no_payload(tmp_path: Path) -> None:
    target = tmp_path / "report.html"
    stale_data = tmp_path / "report_data"
    stale_data.mkdir()
    (stale_data / "stale.js").write_text("stale", encoding="utf-8")
    report = {
        "summary": {"total": 1, "passed": 1, "failed": 0},
        "graphs": {},
        "results": [{"npu_op": "add", "strict_passed": True, "cann_passed": True}],
    }

    write_report_html(report, target)

    assert target.is_file()
    assert not stale_data.exists()


def test_write_report_html_includes_topological_first_failure_analysis(tmp_path: Path) -> None:
    report = {
        "summary": {"total": 3, "passed": 1, "failed": 2},
        "graphs": {
            "npu": {
                "nodes": [
                    {"name": "input", "type": "Data"},
                    {"name": "first", "type": "Mul"},
                    {"name": "downstream", "type": "Relu"},
                ],
                "edges": [
                    {"src": "input", "dst": "first"},
                    {"src": "first", "dst": "downstream"},
                ],
            }
        },
        "results": [
            {"npu_op": "first", "compared": True, "strict_passed": False, "cann_passed": False},
            {"npu_op": "downstream", "compared": True, "strict_passed": False, "cann_passed": False},
        ],
    }
    target = tmp_path / "first-failure.html"

    write_report_html(report, target)

    content = target.read_text(encoding="utf-8")
    assert "function rebuildFirstFailureAnalysis()" in content
    assert "const hasFailedInput = new Map();" in content
    assert "precisionNodeStatus(mainNpuNodes.get(predecessor)) === 'fail'" in content
    assert "hasFailedAncestor.get(predecessor)" not in content
    assert "|| hasFailedInput.get(predecessor)" not in content
    assert "所有可比较直接输入" in content
    assert "const hasUnknownInput = new Map();" in content
    assert "const observed = !!hasUnknownInput.get(name);" in content
    assert "hasUnknownAncestor.get(predecessor)" not in content
    assert "FIRST OBSERVED FAIL" in content
    assert "非有限值一致 · 通过" in content
    assert "else if (!cannPassed) return 'fail';" in content
    assert "else if (strictPassed === undefined) hasUnknown = true;" in content
    assert "else if (!strictPassed) hasWarning = true;" in content
    assert "if (hasUnknown) return 'unknown';" in content
    assert "return hasWarning ? 'warning' : 'pass';" in content
    assert "precisionStatus === 'warning'" in content
    assert "首次整体失败" in content
    assert "FUSION FIRST FAIL" in content
    assert "function upstreamFirstFailures(nodeName)" in content
    assert ".graph-node.first-failure > rect" in content
    assert "输入传播路径" in content
    assert "上游失败传播" in content
    assert "当前输出仍未通过，但不能据此判定当前算子新增了精度误差" in content


def test_write_report_html_from_json_streams_source_file(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        '{"summary":{"total":0,"passed":0,"failed":0},"graphs":{},"results":[],"text":"</script>"}',
        encoding="utf-8",
    )
    target = tmp_path / "from-json.html"
    stale_data = tmp_path / "from-json_data"
    stale_data.mkdir()
    (stale_data / "stale.js").write_text("stale", encoding="utf-8")

    write_report_html_from_json(report, target)

    content = target.read_text(encoding="utf-8")
    assert '"text":"<\\/script>"' in content
    assert "HTML report placeholder" not in content
    assert not stale_data.exists()


def test_write_report_html_from_json_externalizes_tensor_payloads(tmp_path: Path) -> None:
    encoded = {
        "encoding": "base64",
        "dtype": "float32",
        "logical_dtype": "float32",
        "shape": [2],
        "data": "AACAPwAAAEA=",
    }
    source = tmp_path / "report.json"
    source.write_text(
        json.dumps(
            {
                "summary": {"total": 1, "passed": 1, "failed": 0},
                "graphs": {},
                "results": [
                    {
                        "elementwise_data": {
                            "shape": [2],
                            "cpu": encoded,
                            "npu": encoded,
                            "strict_mask": encoded,
                        }
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    target = tmp_path / "from-json-lazy.html"

    write_report_html_from_json(source, target)

    content = target.read_text(encoding="utf-8")
    scripts = list((tmp_path / "from-json-lazy_data").glob("tensor_*.js"))
    assert "AACAPwAAAEA=" not in content
    assert len(scripts) == 1
    assert content.count('"tensor_ref":"tensor_0"') == 3
    assert '"external_script":"from-json-lazy_data/tensor_0.js"' in content
