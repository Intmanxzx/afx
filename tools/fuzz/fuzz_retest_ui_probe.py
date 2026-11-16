"""Local HTTPS browser acceptance with disposable identities and result fixtures."""
import json
from pathlib import Path
import ssl
import subprocess
import sys
import threading
import uuid

from playwright.sync_api import sync_playwright, expect

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tests/fuzz'))
from test_fuzz_retests import setup as setup_fixture
from autofusex.fuzz.central.server import create_server
from autofusex.fuzz.central import retests
from test_fuzz_retest_completion import upload

root=Path(sys.argv[1]).resolve();root.mkdir(parents=True,exist_ok=False,mode=0o700)
fixture=setup_fixture.__wrapped__(root)
database,alice,bob,attempt,client,payload=fixture
if "--structured" in sys.argv[2:]:
    from autofusex.fuzz.reporting.materialized import rebuild
    rebuild(database)
server=create_server(database,0,cache_interval=.2)
subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-keyout',str(root/'key.pem'),
    '-out',str(root/'cert.pem'),'-days','1','-subj','/CN=localhost'],check=True,capture_output=True)
context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);context.load_cert_chain(root/'cert.pem',root/'key.pem')
server.socket=context.wrap_socket(server.socket,server_side=True)
thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
try:
    with sync_playwright() as p:
        browser=p.chromium.launch()
        page=browser.new_page(ignore_https_errors=True,viewport={'width':1600,'height':1000})
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(f'https://localhost:{server.server_port}/#records',wait_until='networkidle')
        expect(page.locator('#rows tr')).to_have_count(3,timeout=35000)
        expect(page.get_by_role('button',name='批量复测',exact=True)).to_be_hidden()
        expect(page.locator('.retest-row-control')).to_have_count(0)
        page.get_by_role('button',name='登录',exact=True).click()
        page.locator('input[name=token]').fill(bob['token'])
        page.locator('dialog[open]').get_by_role('button',name='登录',exact=True).click()
        expect(page.get_by_role('button',name='退出',exact=True)).to_be_visible()
        expect(page.locator('.retest-row-control')).to_have_count(3)
        page.get_by_role('button',name='选择当前页',exact=True).click()
        page.get_by_role('button',name='批量复测',exact=True).click()
        expect(page.get_by_role('button',name='确认提交',exact=True)).to_be_enabled()
        page.get_by_role('button',name='确认提交',exact=True).click()
        expect(page.locator('dialog[open]').get_by_role('heading',name='我的复测')).to_be_visible()
        batches=retests.batches(database,bob['user_id'])['items']
        assert len(batches)==1 and batches[0]['states']=={'queued':1}
        job=retests.claim(database,bob['user_id'],dict(client_id=client,limit=1))['items'][0]
        retests.complete(database,bob['user_id'],upload(fixture,root,job))
        server.public_cache.invalidate()
        page.locator('dialog[open]').get_by_role('button',name='关闭',exact=True).click()
        expect(page.locator('.history-toggle')).to_have_count(3,timeout=25000)
        expect(page.locator('#rows [data-column="source_label"]').first).to_have_text('复测')
        page.locator('.history-toggle').first.click()
        expect(page.locator('.case-history-row:not(.history-footer)')).to_have_count(1)
        expect(page.locator('.history-toggle').first).to_have_attribute('aria-expanded','true')
        page.screenshot(path=str(root/'expanded-history.png'),full_page=True)
        page.locator('.history-toggle').first.click()
        expect(page.locator('.case-history-row')).to_have_count(0)
        expect(page.locator('#rows .accuracy-failed')).to_have_count(0)
        page.locator('#rows tr').first.locator('[data-column="af-mlir:accuracy"] button').click()
        expect(page.locator('#retest-history')).to_be_visible()
        page.screenshot(path=str(root/'retest-ui.png'),full_page=True)
        page.get_by_role('button',name='退出',exact=True).click()
        expect(page.locator('.retest-row-control')).to_have_count(0)
        # Ordinary reports from any user participate in the same history. Check
        # bounded expansion after >20 rounds and filtering only current rows.
        import copy
        for i in range(22):
            ordinary=copy.deepcopy(job);ordinary['lease_id']=str(uuid.uuid4())
            upload(fixture,root,ordinary,('pass','fail','pass') if i==21 else ('pass','pass','pass'))
        server.public_cache.invalidate()
        page.goto(f'https://localhost:{server.server_port}/#records',wait_until='networkidle')
        expect(page.locator('.history-toggle').first).to_contain_text('24',timeout=25000)
        page.locator('.history-toggle').first.click()
        expect(page.locator('.case-history-row:not(.history-footer)')).to_have_count(20)
        page.get_by_role('button',name='加载更多',exact=True).click()
        expect(page.locator('.case-history-row:not(.history-footer)')).to_have_count(23)
        expect(page.locator('.history-footer')).to_have_text('已显示全部历史')
        page.locator('#filters input[name="q"]').fill("WHERE status='accuracy_failed'")
        page.locator('#filters').evaluate('(form)=>form.requestSubmit()')
        expect(page.locator('#rows tr:not(.history-footer)')).to_have_count(1)
        page.locator('.history-toggle').first.click()
        expect(page.locator('.case-history-row:not(.history-footer)')).to_have_count(20)
        assert not errors,errors
        browser.close()
        print(json.dumps({'login':True,'selection':3,'merged_tasks':1,'automatic_refresh':True,'logout':True,'js_errors':errors}))
finally:
    server.shutdown();server.server_close();thread.join()
