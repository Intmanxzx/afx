"""Shape-case UI acceptance, including DAG details and full-value tooltips."""
import json,sys,threading,re
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
from autofusedebugger.fuzz.storage import ResultRepository
from autofusedebugger.fuzz.view.server import create_server
from autofusedebugger.fuzz.generation import workload_id
from autofusedebugger.fuzz.case_spec import encode
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tests"))
from fuzz_fixtures import bundle, three_shapes_spec
root=Path(sys.argv[1]);root.mkdir(exist_ok=True);repo=ResultRepository(root/'results.sqlite')
if not repo.path.exists():
 for backend in ['cpu','af-mlir','triton-npu']:
  repo.save({'seed':str(2*2**128+42),'workload_id':workload_id(bundle()),'oracle_id':'ref/1','backend':backend,'status':'ok' if backend=='cpu' else 'error',
   'execution':{'case_spec':encode(three_shapes_spec()),'backend':backend,'batch_id':'probe','diagnostics':json.dumps({'version':1,'error':'error: UnsupportedOperation: '+('long compiler detail '*15),'shapes':{'1':{'error':'selected shape failure'}}}),'telemetry':json.dumps({'version':1,'stages':{'execution_total':9000000000},'shapes':{str(i):{'first_call_combined':1100000000+i} for i in range(3)}})},
   'shape_results':[{'shape_index':i,'correctness':'error' if i==1 and backend!='cpu' else 'pass','performance':'not_requested'} for i in range(3)]},environment={'device_name':'tester','cann':'9.3'})
with create_server(repo.path) as server:
 t=threading.Thread(target=server.serve_forever);t.start()
 try:
  with sync_playwright() as p:
   browser=p.chromium.launch();page=browser.new_page(viewport={'width':1800,'height':1100});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
   page.goto(server.viewer_url+'#records',wait_until='networkidle')
   assert page.locator('.workspace>header').count()==0
   assert page.locator('.topbar #refresh-time').count()==1
   assert page.locator('.cards').bounding_box()['y']-page.locator('.topbar').bounding_box()['height']<32
   expect(page.locator('#total')).to_have_text('3');expect(page.locator('#rows tr')).to_have_count(3)
   assert page.locator('.cards>div>span').all_inner_texts()==['测试记录','功能Failed','精度Failed','Pass']
   assert page.locator('.cards small').count()==0
   for identity,color in [('errors','rgb(211, 79, 67)'),('fail','rgb(185, 133, 0)'),('ok','rgb(20, 125, 104)')]:
    assert page.locator('#'+identity).evaluate('(e)=>getComputedStyle(e).color')==color
   assert page.locator('.log-length').evaluate('(e)=>getComputedStyle(e).flexDirection')=='row'
   assert page.locator('th').evaluate_all('(els)=>els.every(e=>getComputedStyle(e).textAlign==="left")')
   log_input=page.locator('.log-length input')
   log_input.fill('60');log_input.press('Enter')
   expect(log_input).to_have_value('60')
   assert page.evaluate('localStorage.getItem("af-fuzz.log-length")')=='60'
   assert len(page.locator('[data-column="af-mlir:logs"] .error-preview').first.inner_text())<=63
   log_input.fill('999');log_input.press('Enter');expect(log_input).to_have_value('500')
   page.locator('nav [data-page=overview]').click()
   assert page.locator('#legendOverall button').all_inner_texts()==['Pass','精度Failed','功能Failed']
   expect(page.locator('#legendOverall [data-status=gen_error]')).to_have_attribute('aria-pressed','false')
   expect(page.locator('#chartOverall .seg[data-key=gen_error]')).to_have_count(0)
   page.locator('#legendOverall [data-status=gen_error]').click()
   page.locator('#chartOverall .seg[data-key=gen_error]').dispatch_event('pointerenter')
   expect(page.locator('#vizTooltip')).to_contain_text('功能Failed')
   assert 'gen_error' not in page.locator('#vizTooltip').inner_text()
   toggle=page.locator('#legendOverall [data-status=gen_error]');toggle.click()
   expect(toggle).to_have_attribute('aria-pressed','false');toggle.click()
   page.evaluate("updateStatistics({...latestStatistics,overall:[{label:'全部',status:'unknown',count:1}]})")
   expect(page.locator('#legendOverall [data-status=unknown]')).to_have_text('Unkown')
   page.reload(wait_until='networkidle')
   page.locator('nav [data-page=records]').click()
   expect(page.locator('[data-column="af-mlir:function"] .failed')).to_have_text('功能Failed')
   page.evaluate('''()=>{const items=structuredClone(lastRows);const a=items[0].backends['af-mlir'][0];a.correctness_status='fail';a.measurement_status='error';renderRecords(items);}''')
   expect(page.locator('[data-column="af-mlir:accuracy"] .accuracy-failed')).to_have_text('精度Failed')
   expect(page.locator('[data-column="af-mlir:kernel"] .failed')).to_have_text('性能Failed')
   page.evaluate('renderRecords(lastRows)')
   expect(page.locator('#page-title')).to_have_text('功能&精度')
   expect(page.locator('nav [data-page=records]')).to_contain_text('功能&精度')
   expect(page.locator('nav [data-page=performance]')).to_have_text('性能分析')
   assert 'af-mlir 总耗时' in page.locator('#case-columns').inner_text()
   expect(page.locator('#case-columns [data-column="af-mlir:total"]')).to_be_visible()
   page.evaluate("const noTiming=structuredClone(lastRows);for(const row of noTiming)for(const attempt of row.backends['af-mlir'])attempt.total_ns=null;renderRecords(noTiming)")
   expect(page.locator('#case-columns [data-column="af-mlir:total"]')).to_be_visible()
   page.evaluate("columnPreferences.visibility['af-mlir:total']=false;saveColumns()")
   expect(page.locator('#case-columns [data-column="af-mlir:total"]')).to_have_count(0)
   page.evaluate("columnPreferences={order:[],visibility:{}};saveColumns();renderRecords(lastRows)")
   expect(page.locator('#case-columns [data-column="af-mlir:total"]')).to_be_visible()
   page.locator('nav [data-page=performance]').click()
   expect(page.locator('#page-title')).to_have_text('性能分析')
   expect(page.locator('#metric-empty')).to_have_text('暂无独立编译耗时或 Kernel 性能数据')
   page.locator('nav [data-page=records]').click()
   headers=page.locator('#case-columns').inner_text()
   for title in ['输入 Shape / Dtype','静态 / 动态','triton-npu 功能','triton-npu 运行日志摘要']:assert title in headers
   assert 'x1[1,4,4,2,5,fp32], x2[1,1,1,1,1,fp32]' in page.locator('#rows [data-column=inputs]').first.inner_text()
   page.locator('#rows [data-column=ops] .case-ops').first.hover()
   expect(page.locator('#field-tooltip')).to_have_text('relu · amin · add · relu · mul · amax · transpose · add')
   page.locator('#rows [data-column=inputs]').first.hover()
   expect(page.locator('#field-tooltip')).to_contain_text('x2[1,1,1,1,1,fp32]')
   assert 'add ·' in page.locator('#rows [data-column=ops]').first.inner_text()
   assert '→' not in page.locator('#rows [data-column=ops]').first.inner_text()
   assert page.locator('#rows [data-column=inputs]').nth(0).inner_text()!=page.locator('#rows [data-column=inputs]').nth(1).inner_text()
   page.locator('.log-length input').fill('40');page.locator('.log-length input').press('Tab')
   button=page.locator('#rows tr').nth(1).locator('[data-column="af-mlir:logs"] .error-preview')
   assert len(button.inner_text())<=43
   button.hover();expect(page.locator('#field-tooltip')).to_contain_text('long compiler detail '*15)
   button.click();expect(page).to_have_url(re.compile(r'#attempt=.*&shape=1$'));expect(page.locator('#ascir-detail')).to_contain_text('selected shape failure',timeout=15000)
   expect(page.locator('#failure-details')).to_contain_text('UnsupportedOperation')
   expect(page.locator('#shape-detail')).to_have_count(0)
   assert 'shape=1' in page.url
   expect(page.locator('#case-meta')).to_contain_text('S1')
   expect(page.locator('#provenance')).to_contain_text('CANN')
   assert not page.locator('#detail-content').is_visible()
   assert '当前用例 S0' not in page.locator('#timing-detail').inner_text()
   page.reload(wait_until='networkidle');expect(page.locator('#shape-detail')).to_have_count(0)
   expect(page.locator('#case-meta .dag-node')).to_have_count(10)
   expect(page.locator('#case-meta .dag-edge')).to_have_count(11)
   assert page.locator('#detail h3').all_inner_texts()==['用例节点 DAG','ASCIR 表达的图结构','报错日志摘要','用例执行耗时','运行环境']
   expect(page.locator('#detail details:visible')).to_have_count(0)
   assert page.locator('#failure-details').inner_text().count('UnsupportedOperation')==1
   expect(page.locator('#timing-detail')).to_contain_text('9000.000 ms')
   page.locator('.dag-node').first.hover();expect(page.locator('#field-tooltip')).to_contain_text('float32')
   page.keyboard.press('Escape');expect(page.locator('#field-tooltip')).not_to_be_visible()
   page.screenshot(path=str(root/'detail.png'),full_page=True)
   page.set_viewport_size({'width':390,'height':844})
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
   expect(page.locator('.dag-scroll')).to_be_visible()
   page.screenshot(path=str(root/'detail-mobile.png'),full_page=True)
   page.set_viewport_size({'width':1800,'height':1100})
   page.locator('#close-detail').click();expect(page.locator('.log-length input')).to_have_value('40')
   for query,count in [('pass',2),('Failed',1),("WHERE backend = 'triton-npu' AND function_status = 'Failed'",1)]:
    page.locator('[name=q]').fill(query);page.locator('#filters button[type=submit]').click()
    expect(page.locator('#rows tr')).to_have_count(count);expect(page.locator('#total')).to_have_text(str(count))
    assert not page.locator('#chart-message').inner_text()
   page.locator('#reset-filters').click();expect(page.locator('#rows tr')).to_have_count(3)
   assert page.locator('nav a').count()==3
   assert page.locator('[data-column=dtypes]').count()==0
   page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
   assert not errors,errors;print('shape rows, input metadata, graph, log limit, selected detail, triton columns and mobile passed');browser.close()
 finally:server.shutdown();t.join()
