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
const setTimeout = fn => { timers.set(++nextTimer, fn); return nextTimer; };
const clearTimeout = id => timers.delete(id);
const flush = () => { for (const fn of timers.values()) fn(); timers.clear(); };
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
const PAGES = {}, go = () => {}, pagination = () => el('div', {});
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
      const found = nodes(panel(), n => n.tag === 'button' && text(n) === name);
      assert.equal(found.length, 1, name); found[0].attrs.onclick();
    };
    await Promise.resolve();
    assert.equal(panel().children[1].children[0].attrs['aria-label'], '全部筛选操作');
    click('全部清空');
    assert(nodes(panel(), n => n.attrs['aria-pressed'] === 'false').length > 10);
    // Recreate the panel and click again before the debounce expires. The
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
    assert(fields[0].children[1].children.every(n => n.attrs['aria-pressed'] === 'true'));
    assert.equal(fields[1].children[1].children[0].attrs['aria-pressed'], 'true');
    assert.equal(fields[1].children[1].children[1].attrs['aria-pressed'], repair ? 'true' : 'false');
    if (repair) {
      const categories = fields.find(n => text(n.children[0]) === '算子类别');
      assert(categories.children[1].children.every(n => n.attrs['aria-pressed'] === 'true'));
      const statuses = fields.find(n => text(n.children[0]) === '修复状态');
      assert.deepEqual(statuses.children[1].children.map(n => n.attrs['aria-pressed']),
        ['true','true','true','true','true','false']);
      const closed = statuses.children[1].children[5];
      closed.attrs.onclick();
      assert(panel());
      if (dimension === 'function') assert.deepEqual(nodes(panel(), n =>
        n.attrs['data-result'] && n.attrs['aria-pressed'] === 'true').map(n => n.attrs['data-result']),
        ['mlir','fused_kernel']);
      const treatment = fields.find(n => text(n.children[0]) === '处理标签');
      assert.deepEqual(treatment.children[1].children.map(n => n.attrs['aria-pressed']),
        ['false','true','true','true']);
      assert.equal(nodes(panel(), n => n.tag === 'button' && text(n) === '全部 issue').length, 0);
    }
    await Promise.resolve();
  }
  await PAGES.repair(el('main', {}), new URLSearchParams());
  assert.equal(requested.get('results'), 'mlir,fused_kernel');
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
