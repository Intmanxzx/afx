"""Shared visual filters and overview percentages follow the selected result set."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_selected_results_recalculate_every_displayed_pass_rate():
    node = shutil.which('node')
    if not node:
        pytest.skip('node unavailable')
    source = Path('autofusex/server/static/page-overview.js').read_text()
    source = source[source.index('const LANE_COLOR'):]
    catalog = Path('autofusex/server/static/result-filter.js').read_text()
    catalog = catalog[catalog.index('export const RESULT_GROUPS'):catalog.index('let collapsed')]
    source = catalog.replace('export ', '') + source
    script = r'''
const assert = require('node:assert/strict');
class E {
  constructor(tag, attrs, children) {
    this.tag = tag; this.attrs = {...attrs}; this.children = children.flat(Infinity).filter(x => x != null);
    this.style = {}; this.events = {};
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(k, v) { this.attrs[k] = v; }
  addEventListener(k, fn) { this.events[k] = fn; }
}
const el = (tag, attrs, ...children) => new E(tag, attrs, children), svg = el;
const window = {}, DIM_ORDER = ['dtype', 'op_display'];
const DIM_LABEL = {dtype:'数据类型', op_display:'算子'};
const label = x => x, fmt = String, qs = () => '', errorBox = String;
const pct = (a,b) => b ? (100*a/b).toFixed(1)+'%' : '—';
const stackBar = (parts,total) => el('bar', {parts,total});
const groups = [
  {value:'全部', total:100, results:{pass:30, nnsmith:40, mlir:10, aclnn:20}},
  {value:'FP32', total:100, results:{pass:30, nnsmith:40, mlir:10, aclnn:20}}
];
const overviewData = async () => ({dims:['dtype','op_display'], cells:Object.fromEntries(
  ['af-mlir','triton-npu'].map(b => [b, Object.fromEntries(
    ['all','dtype','op_display'].map((d,i) => [d,{dim:d,groups:[groups[i ? 1 : 0]]}]))]))});
function nodes(root, predicate) {
  if (!(root instanceof E)) return [];
  return [...(predicate(root) ? [root] : []), ...root.children.flatMap(c => nodes(c,predicate))];
}
''' + source + r'''
(async () => {
  async function rates(keys, expected) {
    const box = el('section', {});
    await renderBackendTabs(box, ['af-mlir','triton-npu'], '', {}, new Set(keys));
    const donuts = nodes(box, n => n.attrs.class === 'donut-center');
    const rows = nodes(box, n => n.attrs.class === 'hbar');
    assert.equal(donuts.length, 2);
    assert.equal(rows.length, 4);
    assert.deepEqual([...donuts.map(n => n.children[0]), ...rows.map(n => n.children.at(-1).children[0])],
      Array(6).fill(expected));
  }
  assert.equal(selectedResults(new URLSearchParams()).size, DEFAULT_RESULTS.length);
  assert.equal(selectedResults(new URLSearchParams('results=none')).size, 0);
  assert.equal(resultFilterParams(new URLSearchParams()).chips, 'a2a3');
  await rates(['pass','mlir'], '75.0%');
  await rates(['pass','mlir','nnsmith'], '37.5%');
  await rates(['pass','mlir','nnsmith','aclnn'], '30.0%');
  await rates(['mlir','nnsmith','aclnn'], '0.0%');
  await rates([], '—');
  await rates(['pass'], '100.0%');
  assert.equal(groups[0].total, 100);
  assert.equal(groups[0].results.pass, 30);
})().catch(e => { console.error(e); process.exitCode = 1; });
'''
    subprocess.run([node, '-e', script], check=True, capture_output=True, text=True)
