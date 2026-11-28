from test_platform_ops import world
from autofusex.server import ops
from autofusex.server.issues import IssueError
import pytest


def test_independent_commands_and_exact_ack(world):
    store, user, admin = world
    body = dict(machine_id='independent', workload_states={
        'fuzz': dict(state='running'), 'retest': dict(state='running')})
    cid = ops.heartbeat(store, user, body)['client_id']
    commands = {}
    for name in ('fuzz', 'retest'):
        request = dict(client_id=cid, workload=name, state='stopped', request_key=name)
        commands[name] = ops.request_daemon_state(store, user, request.copy())
        assert ops.request_daemon_state(store, user, request.copy()) == commands[name]
    beat = ops.heartbeat(store, user, body)
    assert set(beat['workload_controls']) == {'fuzz', 'retest'}
    body['workload_states']['fuzz'] = dict(state='stopped', request_id='stale')
    assert len(ops.heartbeat(store, user, body)['workload_controls']) == 2
    body['workload_states']['fuzz']['request_id'] = commands['fuzz']['request_id']
    assert set(ops.heartbeat(store, user, body)['workload_controls']) == {'retest'}
    row = next(c for c in ops.summary(store)['clients'] if c['client_id'] == cid)
    assert row['workload_states']['fuzz']['state'] == 'stopped'
    assert row['workload_states']['retest']['state'] == 'running'


def test_old_clients_reject_independent_controls(world):
    store, user, _ = world
    cid = ops.heartbeat(store, user, dict(machine_id='old-client'))['client_id']
    with pytest.raises(IssueError):
        ops.request_daemon_state(store, user, dict(client_id=cid, workload='fuzz', state='stopped', request_key='old'))
