"""Overview MLIR series, legend order and persistent default selections."""
from pathlib import Path
import shutil
import subprocess
import pytest


def test_overview_series_and_legend_defaults():
    node=shutil.which('node')
    if not node:pytest.skip('node unavailable')
    source=Path('autofusedebugger/fuzz/view/static/charts.js').read_text()
    script=r'''
const assert=require('node:assert/strict');
class E {constructor(){this.children=[];this.style={};this.dataset={};this.textContent='';this.events={};}append(...x){this.children.push(...x)}replaceChildren(...x){this.children=x}setAttribute(k,v){this[k]=v}addEventListener(k,f){this.events[k]=f}}
const nodes=new Map();const document={documentElement:{},querySelector:()=>new E(),createElement:()=>new E(),createTextNode:t=>({textContent:t}),getElementById(id){if(!nodes.has(id))nodes.set(id,new E());return nodes.get(id)}};
const getComputedStyle=()=>({getPropertyValue:k=>k});const window={};
''' + source + r'''
const plots=new Map();drawDonut=(id,legend,segments)=>plots.set(id,segments);drawStackedBars=(id,legend,rows)=>plots.set(id,rows);
const rows=['gen_failed','error','mlir_failed','fail','ok','unknown'].map((status,i)=>({label:'全部',status,count:i+1}));
const data=Object.fromEntries(['overall','backend','category','dtype','dimension','elements'].map(k=>[k,rows]));
const updateStatistics=window.updateStatistics;
updateStatistics(data);
assert.equal(STATUS_COLORS.gen_failed,'--viz-graph-error');
assert.deepEqual(STATUS_KEYS,['gen_failed','gen_error','mlir_failed','fail','pass','unknown']);
for(const id of ['Overall','Backend','Category','Dtype','ShapeDim','ShapeElem']){
 const buttons=nodes.get('legend'+id).children;
 assert.deepEqual(buttons.map(b=>b.dataset.status),STATUS_KEYS);
 assert.deepEqual(buttons.map(b=>b['aria-pressed']),['false','false','true','true','true','false']);
}
assert.deepEqual(plots.get('chartOverall').map(s=>s.value),[0,0,3,4,5,0]);
assert.equal(plots.get('chartBackend')[0].mlir_failed,3);
nodes.get('legendOverall').children[1].events.click();
assert.deepEqual(plots.get('chartOverall').map(s=>s.value),[0,2,3,4,5,0]);
updateStatistics(data);assert.equal(nodes.get('legendOverall').children[1]['aria-pressed'],'true');
assert.equal(nodes.get('legendBackend').children[1]['aria-pressed'],'false');
'''
    subprocess.run([node,'-e',script],check=True,capture_output=True,text=True)
    html=Path('autofusedebugger/fuzz/view/static/index.html').read_text()
    assert 'failure-path-breakdown' not in html
    assert html.index('id="errors"') < html.index('id="mlir-errors"') < html.index('id="fail"') < html.index('id="ok"')


def test_summary_cards_split_function_and_mlir_counts():
    node=shutil.which('node')
    if not node:pytest.skip('node unavailable')
    source=Path('autofusedebugger/fuzz/view/static/app.js').read_text()
    start=source.index("  $('mlir-errors').textContent=")
    end=source.index('\n  return null;',start)
    script="""
const assert=require('node:assert/strict');const nodes=new Map();
const $=id=>{if(!nodes.has(id))nodes.set(id,{});return nodes.get(id)};
const summary={total:5000,by_status:{error:3000,ok:1500,fail:200,gen_failed:300},mlir_failed:364};
"""+source[start:end]+"""
assert.equal($('errors').textContent,(2636).toLocaleString());
assert.equal($('mlir-errors').textContent,(364).toLocaleString());
assert.equal($('total').textContent,(5000).toLocaleString());
"""
    subprocess.run([node,'-e',script],check=True,capture_output=True,text=True)
    css=Path('autofusedebugger/fuzz/view/static/style.css').read_text()
    assert '.cards .kpi-mlir span,.cards .kpi-mlir strong{color:var(--viz-mlir-error)}' in css
    assert '#errors{color:var(--viz-function-error)}' in css
    assert '.cards .kpi-fail span,.cards .kpi-fail strong{color:var(--viz-function-error)}' in css


def test_graph_failure_uses_shared_dark_gray_token():
    css = Path('autofusedebugger/fuzz/view/static/style.css').read_text()
    assert '--viz-graph-error:#4b5563;' in css
    assert '#gen-errors{color:var(--viz-graph-error)}' in css
    assert '.cards .kpi-graph span,.cards .kpi-graph strong{color:var(--viz-graph-error)}' in css
    assert '.outcome-button.graph-failed{background:#f3f4f6;border-color:#d1d5db;color:var(--viz-graph-error)}' in css


def test_kpi_navigation_applies_visible_filter_and_resets_pagination():
    import re
    import json
    node = shutil.which('node')
    if not node:
        pytest.skip('node unavailable')
    source = Path('autofusedebugger/fuzz/view/static/app.js').read_text()
    html = Path('autofusedebugger/fuzz/view/static/index.html').read_text()
    queries = re.findall(r'<button type="button" class="kpi [^"]+" data-case-query="([^"]*)">', html)
    assert len(queries) == 6
    assert queries == ['', "WHERE status = 'gen_failed'",
        "WHERE status = 'function_failed' AND failure_path != 'mlir_failed'",
        "WHERE status = 'function_failed' AND failure_path = 'mlir_failed'",
        "WHERE status = 'accuracy_failed'", "WHERE status = 'Pass'"]
    start = source.index('function applyFilters(')
    end = source.index("$('reset-filters')", start)
    script = r'''const assert=require('node:assert/strict');
const queries=''' + json.dumps(queries) + r''';
const cards=queries.map(q=>({dataset:{caseQuery:q},addEventListener(k,f){this[k]=f}}));
const document={querySelectorAll:()=>cards};
const nodes=new Map();const $=id=>{if(!nodes.has(id))nodes.set(id,{replaceChildren(){this.cleared=true}});return nodes.get(id)};
const form={elements:{q:{value:'old filter'}},addEventListener(){}};
const location={pathname:'/',hash:'#overview'};
const history={replaceState(a,b,url){this.url=url;location.hash=url.slice(url.indexOf('#'))}};
const parameters=()=>new URLSearchParams(form.elements.q.value?{q:form.elements.q.value}:{});
let applied, page=5,cursors=[0,123],next=123,generation=0,detailGeneration=0,lastData='old',lastRows=[{}];
const pendingRequests=new Map();let aborted=0,navigations=0,refreshes=0;
const hideFieldTooltip=()=>{};const navigate=()=>{navigations++};const refreshVisible=()=>{refreshes++};
''' + source[start:end] + r'''
for(const [i,card] of cards.entries()){
 page=5;cursors=[0,123];next=123;
 pendingRequests.set('old',{controller:{abort(){aborted++}}});
 card.click();assert.equal(form.elements.q.value,queries[i]);
 assert.equal(location.hash,'#records');assert.equal(page,0);assert.deepEqual(cursors,[0]);assert.equal(next,null);
 assert.equal(applied.get('q')||'',queries[i]);assert.equal($('rows').cleared,true);
 assert.equal($('next').disabled,true);assert.equal($('previous').disabled,true);
 assert.equal(new URL(history.url,'https://example.test').searchParams.get('q')||'',queries[i]);
}
assert.equal(aborted,6);assert.equal(navigations,6);assert.equal(refreshes,6);
'''
    subprocess.run([node, '-e', script], check=True, capture_output=True, text=True)
    assert "const summaryTask=api('api/summary')" in source


def test_metric_and_record_hover_use_light_blue_without_overriding_status_colors():
    css = Path('autofusedebugger/fuzz/view/static/style.css').read_text()
    selector = '.cards>button.kpi:hover:not(:disabled),\n.case-table button:hover:not(:disabled)'
    rule = css.split(selector + '{', 1)[1].split('}', 1)[0]
    assert '--hover-blue:#eaf4ff;' in css
    assert 'background:var(--hover-blue)' in rule
    assert 'border-color:var(--hover-blue-border)' in rule
    assert '\n  color:' not in rule
    assert '.outcome-button:hover{text-decoration:none}' in css
