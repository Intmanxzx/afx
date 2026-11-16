import threading,tempfile,json,sys
from pathlib import Path
from playwright.sync_api import sync_playwright
from autofusex.fuzz.server.central.server import create_server
root=Path(tempfile.mkdtemp());s=create_server(root/'db',0);threading.Thread(target=s.serve_forever,daemon=True).start()
try:
 with sync_playwright() as p:
  b=p.chromium.launch();page=b.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  for width in [1440,1024,390]:
   page.set_viewport_size(dict(width=width,height=900));page.goto(f'http://127.0.0.1:{s.server_port}/');page.wait_for_timeout(500)
   assert page.title()=='Autofuse一站式工作平台'
   assert page.locator('.brand-text').inner_text()=='Autofuse一站式工作平台'
   boxes=[page.locator(sel).bounding_box() for sel in ['nav','.topbar-actions','#refresh-time','#session-controls']]
   assert boxes[2]['x']+boxes[2]['width']<=boxes[3]['x']
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
   page.locator('#session-controls button').click()
   d=page.locator('dialog[open]');assert '自己的 Token' not in d.inner_text();assert '仅用于验证身份' not in d.inner_text()
   cancel=d.get_by_role('button',name='取消').bounding_box();login=d.get_by_role('button',name='登录',exact=True).bounding_box()
   assert abs(cancel['y']-login['y'])<1 and cancel['x']<login['x']
   page.screenshot(path=f'/tmp/branding-{width}.png')
   d.get_by_role('button',name='取消').click()
  response=page.request.get(f'http://127.0.0.1:{s.server_port}/favicon.svg');assert response.status==200 and 'image/svg+xml' in response.headers['content-type']
  assert not errors,errors
  print(json.dumps(dict(widths=[1440,1024,390],dialog=True,title=True,icon=True,js_errors=errors)))
  b.close()
finally:s.shutdown();s.server_close()
