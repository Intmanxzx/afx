"""Remote browser acceptance for the unified site."""
import json, re, sys, threading
from playwright.sync_api import sync_playwright, expect
from autofusedebugger.fuzz.view.server import create_server

def check(page, base, output):
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(base,wait_until='networkidle')
    expect(page.locator('h1')).to_have_text('总览')
    expect(page.locator('#chartCategory .seg').first).to_be_visible()
    assert 'Transpose' in page.locator('#chartCategory').text_content()
    assert '打开原版' not in page.locator('body').inner_text()
    page.screenshot(path=output+'-overview.png',full_page=True)
    page.locator('nav [data-page=records]').click()
    expect(page.locator('h1')).to_have_text('实验记录')
    expect(page.locator('#rows tr').first).to_be_visible()
    assert all(re.fullmatch('[AB][0-9A-Z]{9}',v) for v in page.locator('#rows td:first-child').all_inner_texts())
    text=page.locator('#rows').inner_text()
    assert 'cpu-eager' not in text and 'ascendc' not in text
    page.locator('[name=seed]').fill('B000000016');page.locator('button[type=submit]').click()
    expect(page.locator('#rows tr')).to_have_count(2)
    page.locator('#rows button').first.click()
    expect(page.locator('#command')).to_have_value('autofusedbg fuzz replay B000000016')
    expect(page.locator('#shape-detail tr')).to_have_count(4)
    page.screenshot(path=output+'-detail.png',full_page=True)
    page.locator('#close-detail').click()
    page.locator('nav [data-page=backends]').click()
    expect(page.locator('.backend-grid article')).to_have_count(5)
    page.locator('nav [data-page=overview]').click()
    page.set_viewport_size({'width':390,'height':844})
    page.screenshot(path=output+'-mobile.png',full_page=True)
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    assert not errors,errors
    print(json.dumps({'checks':'navigation,charts,fixed10seed,backend_names,filter,replay,detail,mobile','errors':errors}))

with sync_playwright() as p:
    browser=p.chromium.launch();page=browser.new_page(viewport={'width':1440,'height':1000})
    if sys.argv[1].startswith('https://'):check(page,sys.argv[1],sys.argv[2])
    else:
        with create_server(sys.argv[1]) as server:
            thread=threading.Thread(target=server.serve_forever);thread.start()
            try:check(page,server.viewer_url,sys.argv[2])
            finally:server.shutdown();thread.join()
    browser.close()
