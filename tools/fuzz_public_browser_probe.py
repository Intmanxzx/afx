"""Remote Playwright acceptance of the deployed public website."""
import json
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

output=Path(sys.argv[1]);output.mkdir(parents=True,exist_ok=True)
with sync_playwright() as p:
    browser=p.chromium.launch()
    page=browser.new_page(viewport={'width':1280,'height':900})
    errors=[];page.on('pageerror',lambda exc:errors.append(str(exc)))
    page.goto('https://esay.app',wait_until='networkidle')
    expect(page.locator('h1')).to_have_text('团队结果')
    expect(page.locator('#rows tr').first).to_be_visible()
    seed=sys.argv[2] if len(sys.argv)>2 else str(2**128)  # use a completed pair
    page.locator('[name=seed]').fill(seed)
    page.locator('button[type=submit]').click()
    expect(page.locator('#rows tr')).to_have_count(2)
    page.locator('#rows button').first.click()
    expect(page.locator('#command')).to_have_value('autofusedbg fuzz replay '+seed)
    expect(page.locator('#detail-content')).to_contain_text('shape_results')
    page.locator('#compare').click()
    page.get_by_role('button', name='比较', exact=True).click()
    expect(page.locator('#comparison-content')).to_contain_text('same_workload')
    assert '?seed=' in page.url
    page.reload(wait_until='networkidle')
    expect(page.locator('#rows tr')).to_have_count(2)
    page.locator('#rows button').first.click()
    page.screenshot(path=str(output/'public-desktop.png'),full_page=True)
    page.set_viewport_size({'width':390,'height':844})
    page.screenshot(path=str(output/'public-mobile.png'),full_page=True)
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    assert not errors,errors
    print(json.dumps({'url':page.url,'seed':seed,'browser':browser.version,'errors':errors,'filter_detail_mobile':'pass'}))
    browser.close()
