"""Coverage is per selected Shape case, with exclusive numeric bins."""
import json
import shutil
import sqlite3
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from autofusedebugger.fuzz.case_spec import encode
from autofusedebugger.fuzz.contracts import canonical_json
from autofusedebugger.fuzz.reporting.incremental import IncrementalStatistics
from autofusedebugger.fuzz.reporting.statistics import statistics
from autofusedebugger.fuzz.storage import ResultRepository
from fuzz_fixtures import bundle


def save_shapes(repo, sizes, *, backend='af-mlir', mixed=False):
    spec = bundle().spec
    tensor = spec.inputs[0]
    inputs = tuple((replace(tensor, name='x', shape=(n,), stride=(1,), dtype='float32'),
                    replace(tensor, name='y', shape=(1,), stride=(1,), dtype='float32')) for n in sizes)
    outputs = tuple((replace(tensor, name='out', shape=(n,), stride=(1,),
                             dtype='float16' if mixed and i % 2 else 'float32'),)
                    for i, n in enumerate(sizes))
    # Repeated add nodes count only once in Elementwise coverage.
    graph = json.loads(spec.graph_json)
    graph['nodes'] = [{'op': op} for op in ('add', 'add', 'transpose', 'sum')]
    spec = replace(spec, graph_json=canonical_json(graph), inputs=inputs[0],
                   shape_samples=inputs, output_samples=outputs)
    return repo.save({'seed': str(spec.seed), 'workload_id': 'a'*64, 'oracle_id': 'ref/1',
        'backend': backend, 'status': 'ok', 'execution': {'backend': backend, 'case_spec': encode(spec)},
        'shape_results': [{'shape_index': i, 'correctness': 'pass', 'performance': 'not_requested'}
                          for i in range(len(sizes))]})


def counts(result, name):
    return {r['label']: r['count'] for r in result[name]}


@pytest.mark.parametrize('field,sizes,labels', [
    ('dimension', [0,8,9,64,65,512,513,4096,4097,16384,16385,65536,65537,131072,131073],
     ['0–8','9–64','65–512','513–4096','4097–16384','16385–65536','65537–131072','>131072']),
    ('elements', [0,1000,1001,16000,16001,64000,64001,256000,256001,1000000,1000001,16000000,16000001,64000000,64000001],
     ['0–1K','(1K,16K]','(16K,64K]','(64K,256K]','(256K,1M]','(1M,16M]','(16M,64M]','>64M']),
])
def test_exclusive_bins_at_every_boundary_and_retries(tmp_path, field, sizes, labels):
    repo = ResultRepository(tmp_path/'results.sqlite')
    for backend in ('cpu', 'af-mlir', 'af-mlir'):
        save_shapes(repo, sizes, backend=backend)
    result = statistics(repo.path)
    assert counts(result, field) == dict(zip(labels, [2]*7+[1]))
    assert sum(counts(result, field).values()) == sum(counts(result, 'overall').values()) == len(sizes)
    selected = statistics(repo.path, q='WHERE shape_index = 2')
    assert counts(selected, field) == {labels[1]: 1}
    assert IncrementalStatistics(repo.path).refresh() == result


def test_category_and_dtype_unique_per_shape_and_missing_metadata(tmp_path):
    repo = ResultRepository(tmp_path/'results.sqlite')
    save_shapes(repo, [8,64], mixed=True)
    result = statistics(repo.path)
    assert counts(result, 'category') == {'Elementwise':2, 'Reduce':2, 'Transpose':2}
    assert counts(result, 'dtype') == {'float32':2, 'float16':1}
    assert counts(statistics(repo.path, q='WHERE shape_index = 0'), 'dtype') == {'float32':1}
    assert counts(statistics(repo.path, q='WHERE shape_index = 1'), 'dtype') == {'float32':1, 'float16':1}
    with sqlite3.connect(repo.path) as conn:
        conn.execute("UPDATE case_facts SET metadata='{}'")
        conn.execute('DELETE FROM case_dimensions')
    result = statistics(repo.path)
    for field in ('category','dtype','dimension','elements'):
        assert counts(result, field) == {'未采集':2}


def test_shape_sizes_include_outputs_and_use_product_not_sum(tmp_path):
    repo = ResultRepository(tmp_path/'results.sqlite')
    save_shapes(repo, [8])
    with sqlite3.connect(repo.path) as conn:
        metadata = json.loads(conn.execute('SELECT metadata FROM case_facts').fetchone()[0])
        # Exercise the metadata producer independently of statistics.
        from autofusedebugger.fuzz.reporting.report import spec_metadata
        spec = bundle().spec
        t = spec.inputs[0]
        spec = replace(spec, shape_samples=((replace(t, shape=(2,3), stride=(3,1)),),),
                       output_samples=((replace(t, shape=(64,64), stride=(64,1)),),))
        metadata['shape_variants'] = spec_metadata(spec)['shape_variants']
        assert metadata['shape_variants'][0]['max_dim'] == 64
        assert metadata['shape_variants'][0]['max_elements'] == 4096
        conn.execute('UPDATE case_facts SET metadata=?', (json.dumps(metadata),))
    result = statistics(repo.path)
    assert counts(result, 'dimension') == {'9–64':1}
    assert counts(result, 'elements') == {'(1K,16K]':1}


def test_numeric_chart_order_not_lexicographic():
    node = shutil.which('node')
    if not node:
        pytest.skip('node unavailable')
    source = Path('autofusedebugger/fuzz/view/static/charts.js').read_text()
    helpers = source[source.index('const SHAPE_BUCKET_ORDER'):source.index('const legendHidden')]
    subprocess.run([node, '-e', "const assert=require('assert'); const STATUS_KEYS=['gen_failed','gen_error','mlir_failed','fail','pass','unknown'];\n"+helpers+"""
for (const id of ['ShapeDim','ShapeElem']) {
 const rows=SHAPE_BUCKET_ORDER[id].map(key=>({key,pass:1})).reverse();
 assert.deepStrictEqual(orderChartRows(rows,id).map(r=>r.key),SHAPE_BUCKET_ORDER[id]);
 assert.equal(rows.reduce((n,r)=>n+r.pass,0),9);
 const displayed=displayChartRows(rows,id);
 assert.deepStrictEqual(displayed.map(r=>r.pass),rows.map(r=>r.pass));
 if(id==='ShapeDim') assert.deepStrictEqual(displayed.map(r=>r.key),['[0,8]','(8,64]','(64,512]','(512,4K]','(4K,16K]','(16K,64K]','(64K,128K]','>128K','未采集']);
 else assert.deepStrictEqual(displayed.map(r=>r.key),['[0,1K]','(1K,16K]','(16K,64K]','(64K,256K]','(256K,1M]','(1M,16M]','(16M,64M]','>64M','未采集']);
}
"""], check=True, capture_output=True, text=True)
    assert 'return orderChartRows([...groups.values()],id)' in source

    assert 'displayChartRows(convert(result[key],id),id)' in source
    assert '独立区间，K=1024' in Path('autofusedebugger/fuzz/view/static/index.html').read_text()
