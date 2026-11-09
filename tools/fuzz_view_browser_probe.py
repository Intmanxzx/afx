"""Optional remote browser QA; requires isolated Playwright and Chromium.

Run from installed wheel environment with path to 100k-row scale database.
Screenshots stay in the supplied remote output directory.
"""
import json
from pathlib import Path
import sys
import threading

from playwright.sync_api import sync_playwright, expect
from autofusedebugger.fuzz.view.server import create_server


def main(database, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    with create_server(database) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch()
                page = browser.new_page(viewport={'width': 1280, 'height': 900})
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(server.viewer_url)
                expect(page.locator('#rows tr')).to_have_count(50)
                expect(page.locator('#total')).to_have_text('100,000')
                page.screenshot(path=str(output/'desktop.png'), full_page=True)
                page.locator('#next').click()
                expect(page.locator('#rows tr').first.locator('td').first).to_have_text('50')
                # Editing a filter without submitting must not change pagination.
                page.locator('[name=seed]').fill('42')
                page.locator('#next').click()
                expect(page.locator('#rows tr').first.locator('td').first).to_have_text('100')
                page.locator('button[type=submit]').click()
                expect(page.locator('#rows tr')).to_have_count(1)
                page.locator('#rows button').click()
                expect(page.locator('#command')).to_have_value('autofusedbg fuzz replay 42')
                expect(page.locator('#detail-content')).to_contain_text('shape_results')
                page.locator('#copy').click()
                expect(page.locator('#detail-message')).to_contain_text('复制')
                page.locator('[name=seed]').fill('999999999')
                page.locator('button[type=submit]').click()
                expect(page.locator('#rows tr')).to_have_count(0)
                expect(page.locator('#message')).to_contain_text('没有符合条件')
                page.locator('[name=seed]').fill('42; echo unsafe')
                page.locator('button[type=submit]').click()
                expect(page.locator('#message')).to_contain_text('invalid query')
                page.locator('#reset-filters').click()
                expect(page.locator('#rows tr')).to_have_count(50)
                page.set_viewport_size({'width': 390, 'height': 844})
                page.screenshot(path=str(output/'mobile.png'), full_page=True)
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                # Browser-level XSS/large numeric seed checks with intercepted API data.
                hostile = '<img src=x onerror="window.injected=true">'
                seed = str(2**128-1)
                def poisoned(route):
                    route.fulfill(json={'items': [{'id': '00000000-0000-0000-0000-000000000001',
                        'seed': seed, 'backend': hostile, 'status': 'ok', 'created_at': 'test'}],
                        'next_cursor': None})
                page.route('**/api/results?*', poisoned)
                page.reload()
                expect(page.locator('#rows tr')).to_have_count(1)
                expect(page.locator('#rows td').nth(0)).to_have_text(seed)
                expect(page.locator('#rows td').nth(1)).to_have_text(hostile)
                assert page.locator('#rows img').count() == 0
                assert page.evaluate('window.injected === undefined')
                assert not errors, errors
                print(json.dumps({'browser': browser.version, 'checks': 'paging, applied filters, seed precision, detail, copy, empty, invalid, reset, mobile overflow, XSS-as-text', 'page_errors': errors}))
                browser.close()
        finally:
            server.shutdown(); thread.join(timeout=5)


if __name__ == '__main__':
    main(*sys.argv[1:])
