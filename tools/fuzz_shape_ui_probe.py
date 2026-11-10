"""Remote-only shape-case UI acceptance, including optional backend columns."""
import json,sys,threading,re
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
from autofusedebugger.fuzz.storage import ResultRepository
from autofusedebugger.fuzz.view.server import create_server
from autofusedebugger.fuzz.generation import materialize,workload_id
root=Path(sys.argv[1]);root.mkdir(exist_ok=True);repo=ResultRepository(root/'results.sqlite')
if not repo.path.exists():
 for backend in ['cpu','af-mlir','triton-npu']:
  repo.save({'seed':str(2**128+42),'workload_id':workload_id(materialize(str(2**128+42))),'oracle_id':'ref/1','backend':backend,'status':'ok' if backend=='cpu' else 'error',
   'execution':{'backend':backend,'batch_id':'probe','diagnostics':json.dumps({'version':1,'error':'error: UnsupportedOperation: '+('long compiler detail '*15),'shapes':{'1':{'error':'selected shape failure'}}}),'telemetry':json.dumps({'version':1,'stages':{'execution_total':9000000000},'shapes':{str(i):{'first_call_combined':1100000000+i} for i in range(3)}})},
   'shape_results':[{'shape_index':i,'correctness':'error' if i==1 and backend!='cpu' else 'pass','performance':'not_requested'} for i in range(3)]},environment={'device_name':'tester','cann':'9.3'})
with create_server(repo.path) as server:
 t=threading.Thread(target=server.serve_forever);t.start()
 try:
  with sync_playwright() as p:
   browser=p.chromium.launch();page=browser.new_page(viewport={'width':1800,'height':1100});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
   page.goto(server.viewer_url+'#records',wait_until='networkidle')
   expect(page.locator('#total')).to_have_text('3');expect(page.locator('#rows tr')).to_have_count(3)
   headers=page.locator('#case-columns').inner_text()
   for title in ['输入 Shape / Dtype','静态 / 动态','triton-npu 功能','triton-npu 日志']:assert title in headers
   assert 'x1[2,3,fp32], x2[2,1,fp32]' in page.locator('#rows [data-column=inputs]').first.inner_text()
   assert 'add ·' in page.locator('#rows [data-column=ops]').first.inner_text()
   assert '→' not in page.locator('#rows [data-column=ops]').first.inner_text()
   assert page.locator('#rows [data-column=inputs]').nth(0).inner_text()!=page.locator('#rows [data-column=inputs]').nth(1).inner_text()
   page.locator('.log-length input').fill('40');page.locator('.log-length input').press('Tab')
   button=page.locator('[data-column="af-mlir:logs"] .error-preview').first
   assert len(button.inner_text())<=43 and button.inner_text().endswith('...')
   button.click();expect(page).to_have_url(re.compile(r'#attempt=.*&shape=1$'));expect(page.locator('#failure-details')).to_contain_text('selected shape failure',timeout=15000)
   expect(page.locator('#shape-detail tr')).to_have_count(5)
   assert 'shape=1' in page.url
   expect(page.locator('#case-meta')).to_contain_text('S1')
   expect(page.locator('#provenance')).to_contain_text('CANN')
   assert not page.locator('#detail-content').is_visible()
   assert '当前用例 S0' not in page.locator('#timing-detail').inner_text()
   page.reload(wait_until='networkidle');expect(page.locator('#shape-detail tr')).to_have_count(5)
   page.screenshot(path=str(root/'detail.png'),full_page=True)
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
