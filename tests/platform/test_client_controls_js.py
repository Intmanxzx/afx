from pathlib import Path
import shutil
import subprocess
import pytest


def test_three_controls_and_independent_toggle():
    node = shutil.which('node')
    if not node:
        pytest.skip('node unavailable')
    source = Path('autofusex/server/static/page-ops.js').read_text()
    source = source[source.index('function clientControls('):source.index('// One bar per hour')]
    script = r'''
const assert = require('assert').strict;
const el = (tag, attrs, ...children) => ({tag, attrs, children: children.flat().filter(x => x != null), append(...items) { this.children.push(...items); }});
const me = {}; const isAdmin = () => true; const time = () => '';
let calls = []; const api = async (path, body) => calls.push(body.body);
const rkey = () => 'key'; const render = () => {};
const buttons = root => root.children.flatMap(x => typeof x === 'object' ? (x.tag === 'button' ? [x] : buttons(x)) : []);
const texts = root => root.children.flatMap(x => typeof x === 'object' ? texts(x) : [x]);
''' + source + r'''
(async () => {
const client = {client_id:1, online:true, mine:true, afx_version:'1', workload_states:{fuzz:{state:'running'}, retest:{state:'running'}}};
let bs = buttons(clientControls(client, '2'));
assert.deepEqual(bs.map(x => x.children[0]), ['停fuzz','停复测','升级']);
await bs[0].attrs.onclick({currentTarget:{}});
assert.equal(calls[0].workload, 'fuzz'); assert.equal(calls[0].state, 'stopped');
client.workload_controls = {fuzz:{state:'stopped'}};
bs = buttons(clientControls(client, '2'));
assert.deepEqual(bs.map(x => x.children[0]), ['启动fuzz','停复测','升级']);
client.daemon_state = 'stopped'; client.daemon_state_at = Date.now();
client.workload_states.fuzz.state = 'stopped';
const labels = texts(clientControls(client, '2'));
assert(labels.includes('Fuzz：已停止')); assert(labels.includes('复测：已开启'));
})();
'''
    result = subprocess.run([node, '-e', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
