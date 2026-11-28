from pathlib import Path
import shutil
import subprocess

import pytest


def test_discovery_keeps_only_latest_execution_per_backend():
    node = shutil.which("node")
    if not node:
        pytest.skip("node unavailable")
    source = Path("autofusex/server/static/page-discovery.js").read_text()
    start = source.index("export function latestByBackend")
    end = source.index("function expandedTable")
    latest = source[start:end].replace("export ", "")
    script = r'''
const assert=require('node:assert/strict');
''' + latest + r'''
const executions=[
 {backend:'af-mlir',exec_id:30},
 {backend:'eager-npu',exec_id:29},
 {backend:'af-mlir',exec_id:10},
 {backend:'af-legacy',exec_id:9},
 {backend:'eager-npu',exec_id:4},
];
assert.deepEqual(latestByBackend(executions).map(e=>e.exec_id), [30,29,9]);
assert.deepEqual(latestByBackend([]), []);
'''
    subprocess.run([node, "-e", script], check=True, capture_output=True, text=True)
