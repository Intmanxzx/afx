"""Small remote-only UI acceptance, including stable DOM during refresh."""
import json,sys,threading
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
from autofusedebugger.fuzz.storage import ResultRepository
from autofusedebugger.fuzz.generation import materialize,workload_id
from autofusedebugger.fuzz.view.server import create_server
root=Path(sys.argv[1]);root.mkdir(exist_ok=True);repo=ResultRepository(root/'results.sqlite')
if not repo.path.exists():
    for i in range(12):
        seed=str(2**128+i)
        for backend in ['cpu','npu-eager']:
            result={'seed':seed,'workload_id':workload_id(materialize(seed)),'oracle_id':'ref/1','backend':backend,'status':'error' if i==0 else 'fail' if i==1 else 'ok',
              'execution':{'backend':backend,'telemetry':json.dumps({'version':1,'stages':{'execution_total':1234567890},'shapes':{}})},
              'shape_results':[{'shape_index':0,'correctness':'error' if i==0 else 'fail' if i==1 else 'pass','performance':'not_requested'}]}
            if i==0 and backend=='npu-eager':
                result['execution']['diagnostics']=json.dumps({'version':1,'error':'compiler error: unsupported Transpose','shapes':{}})
            if i==1 and backend=='npu-eager':
                result['execution']['diagnostics']=json.dumps({'version':1,'shapes':{'0':{'cpu_pass':False,'secondary_used':True,'secondary_pass':False,'max_abs':0.25,'mismatches':2,'elements':32,'basis':'cpu+eager-npu'}}})
            repo.save(result,environment={'device_name':'niuyuhu' if i%2 else 'tester','user_id':str(i%2),'cann':'9.3'})
with create_server(repo.path) as server:
    thread=threading.Thread(target=server.serve_forever);thread.start()
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch();page=browser.new_page(viewport={'width':1440,'height':1000});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(server.viewer_url+'#records',wait_until='networkidle')
            expect(page.locator('#rows tr')).to_have_count(12)
            expect(page.locator('#total')).to_have_text('12')
            expect(page.locator('#ok')).to_have_text('10')
            expect(page.locator('#errors')).to_have_text('1')
            page.get_by_role('button',name='查看日志',exact=True).last.click()
            expect(page.locator('#failure-details')).to_contain_text('unsupported Transpose')
            assert page.locator('#failure-details').bounding_box()['y'] < page.locator('#timing-detail').bounding_box()['y']
            page.locator('#close-detail').click()
            headers=page.locator('#case-columns').inner_text()
            assert 'cpu 精度' not in headers and 'cpu 性能' not in headers and '设备' not in headers and '性能' not in headers
            assert 'eager-npu 精度' in headers and 'cpu 总耗时' in headers
            assert '1.2 s' in page.locator('#rows').inner_text()
            expect(page.locator('#rows .pass').first).to_have_text('Pass')
            assert page.locator('#rows .pass').first.evaluate('e=>getComputedStyle(e).borderWidth')=='0px'
            page.locator('#page-size').select_option('10');expect(page.locator('#rows tr')).to_have_count(10)
            page.locator('#next').click();expect(page.locator('#rows tr')).to_have_count(2)
            page.locator('#previous').click();expect(page.locator('#rows tr')).to_have_count(10)
            page.locator('[name=q]').fill('NIU');page.locator('button[type=submit]').click();expect(page.locator('#rows tr')).to_have_count(6)
            expect(page.locator('#total')).to_have_text('6')
            page.locator('#reset-filters').click();expect(page.locator('#rows tr')).to_have_count(10)
            page.evaluate('window.rowBefore=document.querySelector("#rows tr");window.mutations=0;new MutationObserver(()=>window.mutations++).observe(document.querySelector("#rows"),{subtree:true,childList:true});')
            page.evaluate('load(true)');page.wait_for_function('document.querySelector("#message").textContent===""')
            assert page.evaluate('rowBefore===document.querySelector("#rows tr") && mutations===0')
            before=page.locator('#rows').bounding_box()['y'];page.locator('#column-picker>summary').click();assert abs(page.locator('#rows').bounding_box()['y']-before)<1
            page.locator('#no-columns').click();expect(page.locator('#case-columns th')).to_have_count(0)
            page.reload(wait_until='networkidle');expect(page.locator('#case-columns th')).to_have_count(0)
            page.locator('#column-picker>summary').click();page.locator('#all-columns').click();expect(page.locator('#case-columns [data-column=cann]')).to_be_visible()
            page.locator('#reset-columns').click();page.locator('h1').click();assert not page.locator('#column-picker').evaluate('e=>e.open')
            page.screenshot(path=str(root/'records.png'),full_page=True)
            # Details use the same UI for native functional failures and precision metrics.
            import sqlite3
            with sqlite3.connect(repo.path) as conn:
                failure=conn.execute("SELECT id FROM attempts WHERE status='error' AND backend='npu-eager' LIMIT 1").fetchone()[0]
                accuracy=conn.execute("SELECT id FROM attempts WHERE status='fail' AND backend='npu-eager' LIMIT 1").fetchone()[0]
            page.evaluate('(id)=>details(id)',failure)
            expect(page.locator('#failure-details')).to_contain_text('unsupported Transpose')
            page.evaluate('(id)=>details(id)',accuracy)
            expect(page.locator('#failure-details')).to_contain_text('最大绝对误差')
            expect(page.locator('#failure-details')).to_contain_text('0.25')
            page.locator('nav [data-page=overview]').click()
            legend=page.locator('#legendOverall button[data-status=gen_error]');legend.click();expect(legend).to_have_attribute('aria-pressed','false')
            assert '90.9%' in page.locator('#legendOverall').inner_text()
            page.evaluate('loadCharts()');expect(page.locator('#legendOverall button[data-status=gen_error]')).to_have_attribute('aria-pressed','false')
            page.locator('nav [data-page=performance]').click();expect(page.locator('#metric-empty')).to_be_visible()
            page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            assert not errors,errors;browser.close();print('12 cases / 24 attempts: columns, search, pagination, refresh DOM, popup, legend, mobile passed')
    finally:server.shutdown();thread.join()
