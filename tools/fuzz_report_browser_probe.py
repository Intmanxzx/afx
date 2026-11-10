"""Remote-only browser acceptance for short seeds, Beijing time and old HTML."""
import json
import re
import sys
import threading
from pathlib import Path
from playwright.sync_api import sync_playwright, expect
from autofusedebugger.fuzz.view.server import create_server


def check(page, base):
    errors=[]; page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto(base+'?seed=B16',wait_until='networkidle')
    expect(page.locator('#rows tr').first.locator('td').first).to_have_text('B16')
    text=page.locator('#rows tr').first.locator('td').nth(3).inner_text()
    assert re.fullmatch(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',text),text
    page.locator('#rows button').first.click()
    expect(page.locator('#command')).to_have_value('autofusedbg fuzz replay B16')
    expect(page.locator('#shape-detail tr')).to_have_count(4)
    page.goto(base+'report',wait_until='networkidle')
    expect(page.locator('#chartCategory .seg').first).to_be_visible()
    assert 'Transpose' in page.locator('#chartCategory').text_content()
    expect(page.locator('#chartBackend .seg').first).to_be_visible()
    page.locator('#tabBtnTable').click()
    expect(page.locator('#caseTable tbody tr').first).to_be_visible()
    page.locator('#filterBackend').fill('cpu')
    assert all('cpu' in text for text in page.locator('#caseTable tbody tr td:nth-child(4)').all_inner_texts())
    page.locator('#caseTable tbody a').first.click()
    expect(page.locator('#shape-detail tr')).to_have_count(4)
    assert not errors,errors
    return {'beijing_time_sample':text,'page_errors':errors,'checks':'short_seed,replay,shape_detail,legacy_charts,backend_pivot,filters,detail_link'}


if __name__=='__main__':
    with sync_playwright() as p:
        browser=p.chromium.launch()
        page=browser.new_page(viewport={'width':1280,'height':900})
        if sys.argv[1].startswith('https://'):
            print(json.dumps(check(page,sys.argv[1])))
        else:
            with create_server(sys.argv[1]) as server:
                thread=threading.Thread(target=server.serve_forever);thread.start()
                try:print(json.dumps(check(page,server.viewer_url)))
                finally:server.shutdown();thread.join()
        page.screenshot(path=sys.argv[2],full_page=True)
        browser.close()
