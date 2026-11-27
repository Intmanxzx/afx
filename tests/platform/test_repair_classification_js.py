"""Repair labels share discovery's status widget and use each case's execution."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_repair_issue_and_case_classification_widgets():
    node = shutil.which('node')
    if not node:
        pytest.skip('node unavailable')
    root = Path('autofusex/server/static')
    core = (root / 'core.js').read_text()
    widgets = (root / 'widgets.js').read_text()
    repair = (root / 'page-repair.js').read_text()
    script = r'''
const assert = require('node:assert/strict');
class E {
  constructor(tag, attrs, children) {
    this.tag = tag; this.attrs = attrs; this.children = children.flat(Infinity).filter(v => v != null);
  }
}
const el = (tag, attrs, ...children) => new E(tag, attrs, children);
const text = n => n instanceof E ? n.children.map(text).join('') : String(n);
const label = String, archLabel = String;
const TREATMENTS = [['ignore','忽略'],['low','低级'],['normal','中等'],['high','高级']];
let drawerNodes = [];
const openDrawer = (...nodes) => { drawerNodes = nodes; };
const isAdmin = () => false, STATUS_LABEL = {open:'未领取'}, time = String, fmt = String;
const fixture = {issue_id:'F1', title:'DUPLICATE TITLE', dimension:'function', category:'af_fronted',
  treatment:'normal', status:'open', rounds:[], events:[], arches:[], backend:'af-mlir',
  classifications:[{function_status:'ascgraph_fail',fail_stage:'af_fronted'}], log_summary:'retained error'};
const api = async path => path.endsWith('/F1') ? fixture : {cases:[]};
const subway = () => el('div', {});
''' + core[core.index('export const STAGE_LABEL'):core.index('// Offer current')].replace('export ', '')
    script += widgets[widgets.index('export function statusChip'):].replace('export ', '')
    script += repair[repair.index('function issueClassification'):repair.index('function subway')]
    script += repair[repair.index('export async function issueDrawer'):repair.index('function caseList')].replace('export ', '')
    script += repair[repair.index('function caseList'):]
    script += r'''
const frontend = {function_status:'ascgraph_fail', fail_stage:'af_fronted'};
const compile = {function_status:'compile_fail', fail_stage:'mlir'};
const runtime = {function_status:'runtime_fail', fail_stage:'aclnn'};
for (const category of ['af_fronted','graph_check','lowering']) {
  assert.equal(text(issueClassification({category, classifications:[frontend]})), text(statusChip(frontend)));
}
assert.equal(text(issueClassification({classifications:[compile,runtime]})),
  text(statusChip(compile)) + text(statusChip(runtime)));
assert.equal(text(issueClassification({classifications:[]})), '未知');
const table = caseList([
  {case_key:'one', backend:'af-mlir', npu_arch:'910b', ...compile},
  {case_key:'one', backend:'af-legacy', npu_arch:'910b', ...runtime},
  {case_key:'two', backend:'af-mlir', npu_arch:'910b'}
]);
assert.deepEqual(table.children[0].children[0].children.map(text),
  ['用例','芯片','后端','问题分类','选取原因','标记','复测结果']);
const rows = table.children[1].children;
assert.equal(rows[0].children.length, 7);
assert.equal(text(rows[0].children[3]), text(statusChip(compile)));
assert.equal(text(rows[1].children[3]), text(statusChip(runtime)));
assert.equal(text(rows[2].children[3]), '—');
for (const [dimension, expected, cls] of [['function','构图失败 · AF前端','bad'],
    ['accuracy','精度失败','warn'], ['performance','性能问题','bad']]) {
  const cell = issueCell({...fixture, dimension});
  const link = cell.children[0];
  const summary = link.children[0];
  assert.equal(cell.attrs.class, 'issue-cell');
  assert.equal(link.tag, 'a');
  assert.equal(link.attrs['aria-label'], '查看问题 F1 详情');
  assert.equal(typeof link.attrs.onclick, 'function');
  assert.equal(text(summary.children[0]), 'F1');
  assert.equal(text(summary.children[1]), '中等');
  assert.equal(summary.children[1].attrs.class, 'chip priority-normal');
  assert.equal(text(summary.children[2]), expected);
  assert(text(summary.children[2]).includes(expected));
  if (dimension !== 'function') assert.equal(summary.children[2].attrs.class, 'chip ' + cls);
}
(async () => {
  await issueDrawer('F1');
  const rendered = drawerNodes.map(text).join('');
  assert(!rendered.includes('DUPLICATE TITLE'));
  assert.equal(rendered.split('retained error').length - 1, 1);
})().catch(error => {console.error(error); process.exitCode = 1;});
'''
    result = subprocess.run([node, '-e', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
