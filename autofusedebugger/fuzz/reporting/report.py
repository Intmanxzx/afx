"""Adapt lightweight results to the existing HTML report, without Torch/NNSmith.

Graph metadata is regenerated from the seed protocol, never from device capability.
No missing kernel/profiler measurements are fabricated. Reports are bounded.
"""
import html
import json
import math
from datetime import datetime, timezone, timedelta
from functools import lru_cache

from ..contracts import Seed
from ..generation import plan
from ..backend_names import display_backend


@lru_cache(maxsize=4096)
def case_metadata(seed):
    spec = plan(seed)
    graph = json.loads(spec.graph_json)
    ops = [node['op'] for node in graph['nodes']]
    if graph['reduce_axis'] is not None:
        ops.append('sum')
    return {'graph': graph, 'weights': [{'name': t.name, 'shape': list(t.shape), 'dtype': t.dtype} for t in spec.weights], 'op_types': ops, 'n_nodes': len(ops),
            'input_shapes': {t.name: list(t.shape) for t in spec.inputs},
            'input_dtypes': {t.name: t.dtype for t in spec.inputs},
            'output_shapes': {t.name: list(t.shape) for t in spec.output_samples[0]},
            'output_dtypes': {t.name: t.dtype for t in spec.output_samples[0]},
            'shape_variants': [{'max_dim': max((max(t.shape, default=0) for t in (*sample,*outputs)), default=0),
                                'max_elements': max((math.prod(t.shape) for t in (*sample,*outputs)), default=0),
                                'input_dtypes': {t.name: t.dtype for t in sample}, 'output_dtypes': {t.name: t.dtype for t in outputs}, 'input_shapes': {t.name: list(t.shape) for t in sample},
                                'output_shapes': {t.name: list(t.shape) for t in outputs}}
                               for sample, outputs in zip(spec.shape_samples, spec.output_samples)],
            'weight_broadcast_axes': graph['nodes'][0]['weight_broadcast_axes']}


def report_data(database, *, seed=None, status=None):
    from ..storage.queries import reading, filters
    from .summary_html import render_summary_html
    clauses, values = filters(seed, status)
    where = ' WHERE ' + ' AND '.join(clauses) if clauses else ''
    with reading(database) as conn:
        conn.execute('BEGIN')
        rows = conn.execute('SELECT * FROM attempts'+where+' ORDER BY rowid DESC LIMIT 5001', values).fetchall()
        if len(rows) > 5000:
            raise ValueError('report exceeds 5000 attempts; filter by seed/status; paginated records remain available')
        cases = []
        for index, row in enumerate(rows):
            shapes = conn.execute('SELECT * FROM shapes WHERE attempt_id=? ORDER BY shape_index', (row['id'],)).fetchall()
            execution = json.loads(conn.execute('SELECT payload FROM contexts WHERE id=?', (row['execution_id'],)).fetchone()[0])
            stages = sorted({s['error_stage'] for s in shapes if s['error_stage']})
            errors = [s['error_summary'] for s in shapes if s['error_summary']]
            indexed = conn.execute("SELECT 1 FROM sqlite_master WHERE name='case_facts'").fetchone()
            fact = conn.execute('SELECT metadata FROM case_facts WHERE workload_id=?',(row['workload_id'],)).fetchone() if indexed else None
            metadata = json.loads(fact[0]) if fact else case_metadata(row['seed'])
            selected = [metadata['shape_variants'][s['shape_index']] for s in shapes if s['shape_index'] < len(metadata['shape_variants'])]
            cases.append(dict(metadata, index=index+1, seed=Seed.parse(row['seed']).short,
                backend=html.escape(display_backend(row['backend'])),
                created_at=datetime.fromisoformat(row['created_at']).astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S'),
                status={'ok':'pass','fail':'fail','error':'gen_error','unsupported':'gen_error'}[row['status']],
                shape_mode='static' if execution.get('static_only') else 'dynamic',
                shape_variants=selected, n_kernels=None,
                duration_s=sum(s['first_call_combined_ns'] or 0 for s in shapes)/1e9,
                symptom=html.escape((row['error_summary'] or '\n'.join(errors))[:500]),
                stage=html.escape(row['error_stage'] or ', '.join(stages)),
                bug_dir=None, attempt_id=row['id']))
    return cases


def render(database, *, seed=None, status=None):
    from .summary_html import render_summary_html
    cases = report_data(database, seed=seed, status=status)
    summary = {'cases': cases, 'total': len(cases),
               'passed': sum(c['status']=='pass' for c in cases),
               'failed': sum(c['status']=='fail' for c in cases),
               'gen_errors': sum(c['status']=='gen_error' for c in cases)}
    page = render_summary_html(summary, title='AF Fuzz · 透视汇总与记录', op_categories={
        'Elementwise': {'add','relu','neg','mul'}, 'Reduce': {'sum'}, 'Transpose': {'transpose'}})
    # Retain old charts, click-through filtering, sorting and shape/dtype pivots.
    page = page.replace('<th data-key="seed">Seed</th>', '<th data-key="seed">Seed</th><th data-key="backend">后端</th>')
    page = page.replace('<td>${c.seed}</td>', '<td>${c.seed}</td><td>${c.backend}</td>')
    page = page.replace('Kernel数', 'Kernel数（未采集）').replace('耗时(s)', '首次调用含编译(s)')
    page = page.replace('复现路径', '重放命令')
    page = page.replace('${c.bug_dir ? `<a href="${c.bug_dir}/repro.py">repro.py</a>` : \'-\'}', 'autofusedbg fuzz replay ${c.seed}')
    page = page.replace('<main>', '<main><p>按实验记录统计（重试保留）；图/shape元数据按固定seed协议重建。未采集的kernel/profiler指标不填造。返回主页面可查看每条记录的逐shape结果与环境。</p>')
    page = page.replace('<th data-key="backend">后端</th>', '<th data-key="backend">后端</th><th data-key="created_at">北京时间</th><th>记录详情</th>')
    page = page.replace('<td>${c.backend}</td>', '<td>${c.backend}</td><td>${c.created_at}</td><td><a href="./#attempt=${c.attempt_id}">逐 shape 详情</a></td>')
    page = page.replace('<div class="chart-grid chart-grid-secondary">', '<div class="chart-card"><h3>按后端</h3><svg id="chartBackend" viewBox="0 0 380 260"></svg><div class="legend" id="legendBackend"></div></div><div class="chart-grid chart-grid-secondary">')
    page = page.replace('function renderStats() {', "function renderStats() {\n  drawStackedBars('chartBackend','legendBackend',groupCounts(CASES,c=>c.backend));")
    page = page.replace('<input id="filterOps"', '<input id="filterBackend" placeholder="按后端筛选"><input id="filterOps"')
    page = page.replace('let rows = CASES.filter(c => {', "let rows = CASES.filter(c => {\n    if (!c.backend.toLowerCase().includes(document.querySelector('#filterBackend').value.toLowerCase())) return false;")
    page = page.replace('render();\n\n//', "document.querySelector('#filterBackend').addEventListener('input',render);\nrender();\n\n//", 1)
    page = page.replace('Gen errors', 'Errors / Unsupported').replace('生成错误', '执行异常/不支持')
    return page.encode()



def content_policy(page):
    """Allow only the exact legacy renderer's inline assets, never unsafe-inline."""
    import base64
    import hashlib
    import re
    policies = []
    for tag, directive in (('script', 'script-src'), ('style', 'style-src')):
        hashes = ["'sha256-"+base64.b64encode(hashlib.sha256(block).digest()).decode()+"'"
                  for block in re.findall(rb'<'+tag.encode()+rb'>(.*?)</'+tag.encode()+rb'>', page, re.S)]
        policies.append(directive+' '+' '.join(hashes))
    return "default-src 'none'; style-src-attr 'unsafe-inline'; img-src data:; frame-ancestors 'none'; base-uri 'none'; " + '; '.join(policies)
