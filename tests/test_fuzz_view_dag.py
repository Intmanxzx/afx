"""Exercise the viewer's pure DAG transform without a browser or tensor runtime."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from autofusedebugger.fuzz.reporting.report import spec_metadata
from fuzz_fixtures import bundle


@pytest.fixture
def dag():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js is needed to test the browser DAG transform')
    version = subprocess.check_output([node, '--version'], text=True)
    if int(version.lstrip('v').split('.')[0]) < 16:
        pytest.skip('DAG tests need Node.js 16+; viewer runs in modern browsers')
    source = (Path(__file__).parents[1]/'autofusedebugger/fuzz/view/static/app.js').read_text()
    transform = source[source.index('function caseDag('):source.index('function renderCaseDag(')]

    def run(result):
        script = transform + '\nconsole.log(JSON.stringify(caseDag(' + json.dumps(result) + ')));'
        completed = subprocess.run([node, '-e', script], capture_output=True, text=True, check=True)
        return json.loads(completed.stdout)
    return run


def test_captured_dag_infers_intermediates_and_keeps_outputs(dag):
    meta = spec_metadata(bundle().spec)
    graph = dag({'metadata': meta, 'sample': meta['shape_variants'][0]})
    nodes = {n['name']: n for n in graph['nodes']}
    assert len(nodes) == 10 and len(graph['edges']) == 11
    assert nodes['v6_0']['shape'] == [3, 4, 4, 2, 5]
    assert nodes['v3_0']['shape'] == [3, 4, 4, 5, 2]
    assert nodes['v8_0']['shape'] == []  # scalar, not missing metadata
    assert nodes['v8_0']['isOutput'] and not nodes['v8_0']['inferred']
    assert nodes['v6_0']['inferred']
    assert all(n['dtype'] == 'float32' for n in nodes.values())
    assert all(e['from']['level'] < e['to']['level'] for e in graph['edges'])


def test_dag_uses_selected_shape_and_topological_dependencies(dag):
    meta = {'input_shapes': {'x': [1]}, 'input_dtypes': {'x': 'float32'},
            'graph': {'nodes': [
                {'op': 'relu', 'inputs': ['y'], 'output': 'z'},
                {'op': 'neg', 'inputs': ['x'], 'output': 'y'}], 'outputs': ['z']}}
    graph = dag({'metadata': meta, 'sample': {'input_shapes': {'x': [2, 3]},
                                            'input_dtypes': {'x': 'float16'}}})
    nodes = {n['name']: n for n in graph['nodes']}
    assert nodes['z']['shape'] == [2, 3] and nodes['z']['dtype'] == 'float16'
    assert nodes['z']['level'] == 2


def test_dag_missing_metadata_and_cycles_are_explicit(dag):
    assert dag({}) == {'nodes': [], 'edges': []}
    unknown = dag({'metadata': {'graph': {'nodes': [
        {'op': 'relu', 'inputs': ['missing'], 'output': 'y'}]}}})
    assert len(unknown['nodes']) == 2
    assert all('shape' not in n for n in unknown['nodes'])
    cycle = dag({'metadata': {'graph': {'nodes': [
        {'op': 'relu', 'inputs': ['y'], 'output': 'x'},
        {'op': 'relu', 'inputs': ['x'], 'output': 'y'}]}}})
    assert not cycle['nodes'] and '环' in cycle['error']


def test_dag_does_not_invent_invalid_broadcast_shape(dag):
    graph = dag({'metadata': {'input_shapes': {'a': [2], 'b': [3]},
                             'graph': {'nodes': [{'op': 'add', 'inputs': ['a', 'b'], 'output': 'c'}]}}})
    assert 'shape' not in next(n for n in graph['nodes'] if n['name'] == 'c')


def test_compact_lifecycle_and_stage_labels():
    node=shutil.which('node')
    if not node:pytest.skip('Node needed')
    if int(subprocess.check_output([node,'--version'],text=True).lstrip('v').split('.')[0])<16:
        pytest.skip('viewer tests need Node.js 16+')
    source=(Path(__file__).parents[1]/'autofusedebugger/fuzz/view/static/app.js').read_text()
    helpers=source[source.index('function lifecycleTime('):]
    name=source[source.index('function stageName('):].split('\n')[0]
    script=name+'\n'+helpers+"\nconsole.log(JSON.stringify([lifecycleTime('2026-11-10T12:42:30.803558+00:00'),stageLabel({kind:'reference'}),stageLabel({kind:'compare'}),stageLabel({kind:'first_call_combined',lease:{device_uuid:'u'}})]));"
    out=json.loads(subprocess.check_output([node,'-e',script],text=True))
    assert out==['12:42:30.803','CPU 期望结果','CPU 结果校验','NPU首调(惰性编译/执行/同步)']
    assert '(value/1e9).toFixed(3)' not in source
    assert 'result.tool_efficiency?.note' not in source
