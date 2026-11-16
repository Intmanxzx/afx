"""Docker-only real Chromium UI check; ephemeral identity and database."""
import json
from pathlib import Path
import tempfile
import threading
import subprocess
import ssl
from playwright.sync_api import sync_playwright
from autofusex.fuzz.server.central.repository import create_user
from autofusex.fuzz.server.central.server import create_server
from autofusex.fuzz.daemon.management.central import ControlStore

with tempfile.TemporaryDirectory() as directory:
    database=Path(directory)/'central.sqlite';user=create_user(database,'Browser Fixture')
    server=create_server(database,0)
    cert=Path(directory)/'cert.pem';key=Path(directory)/'key.pem'
    subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-keyout',str(key),'-out',str(cert),'-days','1','-subj','/CN=localhost'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    tls=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);tls.load_cert_chain(cert,key)
    server.socket=tls.wrap_socket(server.socket,server_side=True)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    store=ControlStore(database);pair=store.pairing('Docker browser device')
    url='https://127.0.0.1:'+str(server.server_port)
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True)
            context=browser.new_context(ignore_https_errors=True)
            page=context.new_page();errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
            page.goto(url+'/manage')
            page.get_by_role('button',name='登录',exact=True).click()
            page.get_by_role('textbox',name='Token',exact=True).fill(user['token'])
            page.locator('dialog').get_by_role('button',name='登录',exact=True).click()
            page.wait_for_function("() => document.body.classList.contains('authenticated')")
            page.locator('#code').fill(pair['code']);page.get_by_role('button',name='确认绑定').click()
            page.wait_for_function("() => document.querySelector('#notice').textContent.includes('绑定已批准')")
            store.exchange(pair['id'],pair['secret'])
            page.get_by_role('button',name='刷新',exact=True).click()
            page.wait_for_function("() => document.querySelector('#devices').options.length === 1")
            page.get_by_role('button',name='任务表单',exact=True).click()
            page.wait_for_selector('#schema-form .schema-field')
            page.get_by_role('button',name='表单 → 编辑器').click()
            page.wait_for_function("() => document.querySelector('#spec').value.includes('fuzz_task')")
            page.locator('#template-name').fill('Browser template');page.get_by_role('button',name='保存新修订').click()
            page.wait_for_function("() => document.querySelector('#templates').options.length === 2")
            page.locator('#kind').select_option('environment.inspect');page.locator('#spec').fill('{}')
            page.get_by_role('button',name='提交操作',exact=True).click()
            page.wait_for_function("() => document.querySelector('#operations').textContent.includes('environment.inspect')")
            page.get_by_role('button',name='审计记录',exact=True).click()
            page.wait_for_function("() => document.querySelector('#history-view').textContent.includes('submit')")
            page.screenshot(path='/tmp/managed-client-ui.png',full_page=True)
            assert not errors,errors
            print(json.dumps(dict(browser='chromium',login=True,pairing=True,forms=True,templates=True,operations=True,audit=True,console_errors=errors)))
            browser.close()
    finally:
        server.shutdown();server.server_close();thread.join(timeout=5)
