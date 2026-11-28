"""Panel actions update every filter and survive debounced navigation."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_panel_actions_select_clear_and_restore_defaults():
    node = shutil.which("node")
    if not node:
        pytest.skip("node unavailable")
    root = Path("autofusex/server/static")
    core = (root / "core.js").read_text()
    constants = "\n".join(line.replace("export ", "") for line in core.splitlines()
                          if line.startswith(("export const BACKEND_LABEL", "export const STATUS_LABEL")))
    source = (root / "result-filter.js").read_text().split("\n", 1)[1].replace("export ", "")
    repair = (root / "page-repair.js").read_text()
    repair = repair[repair.index("PAGES.repair ="):repair.index("export async function issueDrawer")]
    script = r'''
const assert = require('node:assert/strict');
let timers = new Map(), nextTimer = 0;
let now = 0;
const setTimeout = (fn, delay) => { timers.set(++nextTimer, {fn, due:now + delay}); return nextTimer; };
const clearTimeout = id => timers.delete(id);
const advance = ms => { now += ms; for (const [id, timer] of [...timers]) if (timer.due <= now) { timers.delete(id); timer.fn(); } };
const flush = () => advance(1000);
const matchMedia = () => ({matches:false});
const archOptions = async () => ['910b', '950', 'custom-chip'];
class E {
  constructor(tag, attrs, children) {
    this.tag = tag; this.attrs = {...attrs}; this.children = []; this.scrollTop = 0;
    this.classList = {toggle() {}}; this.isConnected = true; this.append(...children.flat(Infinity));
  }
  append(...children) {
    this.children.push(...children);
    children.forEach(c => { if (c instanceof E) c.parent = this; });
  }
  replaceWith(fresh) {
    this.parent.children[this.parent.children.indexOf(this)] = fresh;
    fresh.parent = this.parent; this.isConnected = false;
  }
  setAttribute(k, v) { this.attrs[k] = v; }
  replaceChildren(...children) { this.children = []; this.append(...children); }
  addEventListener() {}
  querySelectorAll(selector) { return nodes(this, n => n.attrs.class === selector.slice(1)); }
}
const el = (tag, attrs, ...children) => new E(tag, attrs, children);
const nodes = (root, predicate) => root instanceof E ?
  [...(predicate(root) ? [root] : []), ...root.children.flatMap(c => nodes(c, predicate))] : [];
const text = n => n instanceof E ? n.children.map(text).join('') : String(n);
let pagers = [];
const PAGES = {}, go = () => {}, pagination = (...args) => { pagers.push(args); return el('nav', {}); };
const qs = values => '?' + new URLSearchParams(Object.entries(values).filter(([,v]) => v != null));
let requested;
const api = async path => { requested = new URL(path, 'http://localhost').searchParams;
  return {issues:[], page:1, total_pages:0, total:0}; };
''' + constants + '\n' + source + '\n' + repair + r'''
(async () => {
  for (const dimension of [null, 'function', 'accuracy', 'performance']) {
    const repair = dimension !== null;
    const initial = new URLSearchParams({q:'op:add', backends:'cpu', chips:'a5', results:'mlir',
      ...(repair ? {dimension, scope:'all', categories:'Reduce', treatment:'ignore', status:'closed'} : {})});
    let route = Object.fromEntries(initial), calls = 0;
    const host = el('main', {}, resultFilter(initial, patch => {
      calls++;
      for (const [k, v] of Object.entries(patch)) v == null ? delete route[k] : route[k] = v;
    }, [], {repair}));
    const panel = () => host.children[0];
    const click = name => {
      const found = nodes(panel().children[1].children[0], n => n.tag === 'button' && text(n) === name);
      assert.equal(found.length, 1, name); found[0].attrs.onclick();
    };
    await Promise.resolve();
    assert.equal(panel().children[1].children[0].attrs['aria-label'], '全部筛选操作');
    click('全部清空');
    assert(nodes(panel(), n => n.attrs['aria-pressed'] === 'false').length > 10);
    // Click again before the debounce expires. The
    // asynchronously discovered chip must still participate in select-all.
    click('全部选择');
    flush();
    assert.equal(calls, 1);
    assert.equal(route.chips, 'a2a3,a5,custom-chip');
    assert.equal(route.backends.split(',').length, Object.keys(BACKEND_LABEL).length);
    assert.equal(route.results.split(',').length, repair ? RESULT_ITEMS.length - 3 : RESULT_ITEMS.length);
    if (repair) {
      assert.equal(route.scope, 'all');
      assert.equal(route.categories.split(',').length, ISSUE_CATEGORIES.length);
      assert.equal(route.treatment, 'ignore,low,normal,high');
      assert.equal(route.status.split(',').length, Object.keys(STATUS_LABEL).length);
    }
    click('全部清空'); flush();
    for (const key of ['backends','chips','results', ...(repair ? ['categories','treatment','status'] : [])]) {
      assert.equal(route[key], 'none', key);
    }
    click('恢复默认'); flush();
    assert.deepEqual(route, {q:'op:add', ...(repair ? {dimension} : {})});
    const fields = panel().querySelectorAll('.visual-filter-field');
    assert.deepEqual(fields[0].children[1].children.map(n => n.attrs['aria-pressed']),
      Object.keys(BACKEND_LABEL).map(key => DEFAULT_BACKENDS.includes(key) ? 'true' : 'false'));
    assert.equal(resultFilterParams(new URLSearchParams()).selected_backends, 'af-legacy,af-mlir,triton-npu');
    assert.equal(resultFilterParams(new URLSearchParams({backends:'eager-npu'})).selected_backends, 'eager-npu');
    assert.equal(fields[1].children[1].children[0].attrs['aria-pressed'], 'true');
    assert.equal(fields[1].children[1].children[1].attrs['aria-pressed'], repair ? 'true' : 'false');
    if (repair) {
      const categories = fields.find(n => n.attrs['aria-label'] === '算子类别');
      assert(categories.children[1].children.every(n => n.attrs['aria-pressed'] === 'true'));
      const statuses = fields.find(n => n.attrs['aria-label'] === '修复状态');
      assert.deepEqual(statuses.children[1].children.map(n => n.attrs['aria-pressed']),
        ['true','true','true','true','true','false']);
      const closed = statuses.children[1].children[5];
      closed.attrs.onclick();
      assert(panel());
      if (dimension === 'function') assert.deepEqual(nodes(panel(), n =>
        n.attrs['data-result'] && n.attrs['aria-pressed'] === 'true').map(n => n.attrs['data-result']),
        ['mlir','fused_kernel']);
      const treatment = fields.find(n => n.attrs['aria-label'] === '处理标签');
      assert.deepEqual(treatment.children[1].children.map(n => n.attrs['aria-pressed']),
        ['false','true','true','true']);
      assert.equal(nodes(panel(), n => n.tag === 'button' && text(n) === '全部 issue').length, 0);
    }
    flush();
    const stablePanel = panel();
    const stableButtons = nodes(panel(), n => n.tag === 'button');
    panel().scrollTop = 137;
    const action = (title, name) => {
      const group = nodes(panel(), n => n.attrs['aria-label'] === title + '筛选操作')[0];
      nodes(group, n => n.tag === 'button' && text(n) === name)[0].attrs.onclick();
    };
    // Clear, wait, then select an item: the debounce restarts and keeps DOM/scroll.
    const before = calls;
    action('后端', '全部清空');
    advance(FILTER_DELAY_MS - 1);
    assert.equal(calls, before);
    fields[0].children[1].children[0].attrs.onclick();
    advance(FILTER_DELAY_MS - 1);
    assert.equal(calls, before);
    advance(1);
    assert.equal(calls, before + 1);
    assert.equal(route.backends, Object.keys(BACKEND_LABEL)[0]);
    assert.equal(panel(), stablePanel);
    assert.equal(panel().scrollTop, 137);
    assert.deepEqual(nodes(panel(), n => n.tag === 'button'), stableButtons);
    action('后端', '恢复默认'); flush();
    assert.equal(route.backends, undefined);
    // Every category's actions must leave the other categories alone.
    for (const field of fields) {
      const title = field.attrs['aria-label'];
      action(title, '全部清空');
      assert(field.children[1].children.every(n => n.attrs['aria-pressed'] === 'false'));
      action(title, '全部选择');
      assert(field.children[1].children.every(n => n.attrs['aria-pressed'] === 'true'));
      action(title, '恢复默认');
    }
    if (!repair || dimension === 'function') {
      click('全部清空');
      const mlir = nodes(panel(), n => n.attrs['data-result'] === 'mlir')[0];
      mlir.attrs.onclick(); flush();
      assert.equal(route.results, 'mlir');
      action('编译失败', '全部选择');
      action('构图失败', '全部清空');
      action('编译失败', '恢复默认'); flush();
      assert.deepEqual(route.results.split(',').sort(), RESULT_GROUPS[1][1].map(([key]) => key).filter(key =>
        (repair ? REPAIR_DEFAULT_RESULTS : DEFAULT_RESULTS).includes(key)).sort());
      click('恢复默认');
      mlir.attrs.onclick(); flush();
      assert(!route.results.split(',').includes('mlir'));
      action('结果与阶段', '全部清空'); flush();
      assert.equal(route.results, 'none');
      action('结果与阶段', '恢复默认'); flush();
      assert.equal(route.results, undefined);
    }
    const beforeCancel = calls;
    action('后端', '全部清空');
    assert.equal(isFilterChangePending(), true);
    cancelFilterChange(); flush();
    assert.equal(calls, beforeCancel);
    assert.equal(isFilterChangePending(), false);
    await Promise.resolve();
  }
  await PAGES.repair(el('main', {}), new URLSearchParams());
  assert.equal(pagers.length, 2);
  assert.deepEqual(pagers[0].slice(0, 3), pagers[1].slice(0, 3));
  assert.equal(requested.get('results'), 'mlir,fused_kernel');
  assert.equal(requested.get('selected_backends'), 'af-legacy,af-mlir,triton-npu');
  assert.equal(requested.get('status'), 'open,claimed,verifying,regressing,summarizing');
  assert.equal(requested.get('categories'), null);
  assert.equal(requested.get('treatment'), 'low,normal,high');
  for (const selection of ['none', 'mlir,schedule']) {
    await PAGES.repair(el('main', {}), new URLSearchParams({results:selection,status:'closed'}));
    assert.equal(requested.get('results'), selection);
    assert.equal(requested.get('status'), 'closed');
  }
  await PAGES.repair(el('main', {}), new URLSearchParams({dimension:'accuracy'}));
  assert.equal(requested.get('results'), null);
  assert.equal(requested.get('status'), 'open,claimed,verifying,regressing,summarizing');
})().catch(error => { console.error(error); process.exitCode = 1; });
'''
    result = subprocess.run([node, "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
