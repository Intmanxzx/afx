"""Remote-only small-fixture visual regression; no NPU jobs or production mutation."""
import json
from pathlib import Path
import sys
import threading
from playwright.sync_api import sync_playwright, expect
from autofusedebugger.fuzz.storage import ResultRepository
from autofusedebugger.fuzz.generation import materialize, workload_id
from autofusedebugger.fuzz.view.server import create_server

root=Path(sys.argv[1]);root.mkdir(exist_ok=True)
repo=ResultRepository(root/'layout.sqlite')
if not repo.path.exists():
    for index in range(12):
        seed=str(2**128+index)
        for backend in ('cpu','npu-eager'):
            repo.save({'seed':seed,'workload_id':workload_id(materialize(seed)),'oracle_id':'ref/1','backend':backend,
                'status':'fail' if index==2 and backend=='npu-eager' else 'ok',
                'execution':{'backend':backend,'device':'cpu' if backend=='cpu' else 'npu:0','static_only':index%3==0,'batch_id':'small-layout-check',
                             'telemetry':json.dumps({'version':1,'stages':{'generation':1000,'execution_total':2394258000},'shapes':{}})},
                'evaluation':{'performance':index%2==0},'shape_results':[{'shape_index':0,'correctness':'pass','performance':'measured' if index%2==0 else 'not_requested','mean_wall_ns':1000 if index%2==0 else None}]},
                environment={'device_name':'MLIR NPU' if index%2 else '测试设备 · 很长的设备名字也不应越界',
                             'installation_id':'baba3078-6137-47e6-aac3-1c9ebda1e71d','user_id':'64899d98-b1dd-466d-bd89-82cf86296446',
                             'cann':'9.3.0','torch':'2.10.0+cpu','torch_npu':'2.10.0.post7.dev20261015','autofuse_binary':'d'*64})

def geometry(page):
    issues=page.locator('#rows').evaluate('''body => {
      const issues=[];
      for(const td of body.querySelectorAll('td')){
        const box=td.getBoundingClientRect();
        for(const el of td.querySelectorAll('button,small,strong,summary,dd,.config-label,.result-duration')){
          if(!el.getClientRects().length)continue;
          const r=el.getBoundingClientRect();
          if(r.right>box.right+1 || r.left<box.left-1 || el.scrollWidth>el.clientWidth+1)issues.push(el.className+':'+el.textContent);
        }
      }
      return issues;
    }''')
    assert not issues,issues
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')

with create_server(repo.path) as server:
    thread=threading.Thread(target=server.serve_forever);thread.start()
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch();page=browser.new_page(viewport={'width':1440,'height':1000});errors=[]
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(server.viewer_url+'#records',wait_until='networkidle')
            expect(page.locator('#rows tr')).to_have_count(12)
            expect(page.locator('#case-columns th')).to_have_count(10)
            expect(page.locator('#case-columns')).not_to_contain_text('af-mlir')
            expect(page.locator('#case-columns [data-column=user_id]')).to_be_visible()
            assert '精度通过' in page.locator('#rows').inner_text()
            assert '已测量' in page.locator('#rows').inner_text() and '未测量' in page.locator('#rows').inner_text()
            page.locator('#rows .user-filter').first.click()
            expect(page.locator('[name=user_id]')).to_have_value('64899d98-b1dd-466d-bd89-82cf86296446')
            expect(page.locator('#total')).to_have_text('24')
            page.locator('[name=user_id]').fill('nobody');page.locator('button[type=submit]').click()
            expect(page.locator('#total')).to_have_text('0');expect(page.locator('#rows tr')).to_have_count(0)
            page.locator('#reset-filters').click();expect(page.locator('#rows tr')).to_have_count(12)
            geometry(page)
            page.screenshot(path=str(root/'records-desktop.png'),full_page=True)
            expect(page.locator('.topbar nav')).to_be_visible()
            for text in ['结果自动同步','北京时间 · UTC+8','时间均为北京时间']:assert text not in page.locator('body').inner_text()
            expect(page.locator('#case-columns [data-column=cann]')).to_have_count(0)
            page.locator('#column-picker > summary').click()
            page.locator('#column-options input[data-column=cann]').check()
            page.get_by_role('button',name='CANN上移',exact=True).click()
            expected=page.locator('#case-columns th').all_text_contents()
            page.reload(wait_until='networkidle')
            expect(page.locator('#case-columns [data-column=cann]')).to_be_visible()
            assert page.locator('#case-columns th').all_text_contents()==expected
            page.locator('#column-picker > summary').click()
            for key in ['torch','torch_npu','autofuse_binary','source_id']:
                page.locator('#column-options input[data-column='+key+']').check()
            page.locator('#column-picker > summary').click();geometry(page)
            page.screenshot(path=str(root/'records-expanded.png'),full_page=True)
            page.set_viewport_size({'width':390,'height':844});geometry(page)
            page.screenshot(path=str(root/'records-mobile.png'),full_page=True)
            page.locator('#rows .outcome-button').first.click();expect(page.locator('#timing-detail table')).to_be_visible()
            assert not errors,errors
            browser.close()
        print(json.dumps({'cases':12,'attempts':24,'geometry':'no overlap or text overflow','desktop':1440,'mobile':390,'errors':errors}))
    finally:server.shutdown();thread.join()
