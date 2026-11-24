"""Submit platform retests to the active run; never launch another executor."""
import json
import uuid
from dataclasses import asdict
from pathlib import Path

from autofusex.client.store import Store
from autofusex.client.artifacts import atomic
from autofusex.contracts.execution import ExecutionConfig, EvaluationConfig
from autofusex.workloads.fuzz.contracts import Seed
from autofusex.workloads.fuzz.scheduling.ledger import TaskLedger
from autofusex.execution.scheduling.budgets import read_policy
from autofusex.client.paths import retest_directory


def submit(home, task):
    store = Store(home)
    record_dir = retest_directory(store.home)
    record = record_dir / ('task-' + str(int(task['task_id'])) + '.json')
    if record.exists():
        saved = json.loads(record.read_text())
        if saved['lease_id'] == task['lease_id']:
            return saved if saved.get('delivered') else admit(record, saved)
    with store.connect() as db:
        runs = list(db.execute("SELECT id FROM runs WHERE state='running'"))
    if len(runs) != 1:
        raise ValueError('retest requires one active run')
    directory = store.home / 'runs' / runs[0]['id']
    if json.loads((directory / 'control.json').read_text())['action'] != 'run':
        raise ValueError('run is stopping or reconfiguring')
    prepared = json.loads((directory / 'spec.json').read_text())
    spec = prepared['spec']
    ledger = TaskLedger(directory / 'tasks.sqlite')
    budget = read_policy(ledger)
    if budget is None:
        raise ValueError('run is not ready')
    binding = json.loads((directory / 'tasks.sqlite.results.json').read_text())
    seed = Seed.parse(task['case_key'])
    ex = spec['execution']; perf = spec['evaluation']['performance']
    evaluation = EvaluationConfig(performance=perf['enabled'], warmup_steps=perf['warmup_steps'], measured_steps=perf['measured_steps'])
    batch = 'platform-retest:' + str(task['task_id']) + ':' + task['lease_id']
    tasks = []
    for backend in task['backends']:
        if backend == 'cpu':
            continue
        identity = str(uuid.uuid5(uuid.NAMESPACE_URL, batch + ':' + backend))
        execution = ExecutionConfig(backend, ex['devices'][0], budget.compile_threads,
            scheduling='staged', static_only=ex['static_only'] or not seed.dynamic,
            batch_id=batch, save_results=True)
        tasks.append((seed, execution, evaluation, identity))
    if not tasks:
        raise ValueError('retest has no executable backend')
    saved = dict(lease_id=task['lease_id'], queue=str(ledger.path), database=binding['database'],
                 task_ids=[t[3] for t in tasks], devices=ex['devices'],
                 specifications=[(str(t[0]), asdict(t[1]), asdict(t[2]), t[3]) for t in tasks])
    atomic(record, saved)
    return admit(record, saved)


def admit(record, saved):
    ledger = TaskLedger(Path(saved['queue']))
    # Only the manager admits retests. A persisted record survives a crash before enqueue.
    with ledger._connection() as db:
        existing = {r[0] for r in db.execute('SELECT id FROM tasks')}
    tasks = [(seed, ExecutionConfig(**ex), EvaluationConfig(**ev), identity)
             for seed, ex, ev, identity in saved['specifications'] if identity not in existing]
    if tasks:
        ledger.enqueue_many(tasks, priority=100, devices=saved['devices'])
    return saved


def completed(saved):
    ledger = TaskLedger(Path(saved['queue']))
    rows = [ledger.get(identity) for identity in saved['task_ids']]
    if not all(row['state'] == 'completed' for row in rows):
        return []
    return [row['attempt_id'] for row in rows]


def synchronize(client, home):
    """Coordinate remote leases and receipts without starting execution processes."""
    from autofusex.client.upgrade_drain import BUSY
    if Store(home).get('client_switch', {}).get('state') in BUSY:
        return 0
    from autofusex.client.cleanup import access
    with access(home):
        return _synchronize(client, home)


def _synchronize(client, home):
    from autofusex.client.platform_transport import installed_packages, Mapper, upload_new
    from autofusex.client.daemon import request
    installed = installed_packages(client.environment())
    if not installed or not client.devices:
        return 0
    env = client.identity()
    tasks = client.api.call('POST', '/api/client/retests/claim', {
        'request_key': 'claim-' + str(uuid.uuid4()),
        'machine_id': env.get('installation_id') or __import__('socket').gethostname(),
        'npu_arch': env.get('device_model') or 'unknown',
        'installed': installed, 'limit': len(client.devices)})['tasks']
    finished = 0
    for task in tasks:
        if task['case_source'] != 'nnsmith':
            continue
        saved = request(home, {'action': 'platform_retest', 'task': task})
        if saved.get('delivered'):
            continue
        ids = completed(saved)
        if not ids:
            continue
        mapper = Mapper(saved['database'])
        try:
            statuses = upload_new(client.api, mapper, client.state, client.counters,
                key_prefix='task' + str(task['task_id']), attempt_ids=ids,
                run_override={'run_uuid': task['run_uuid'][:64], 'kind': task['run_kind']})
        finally:
            mapper.conn.close()
        uploaded = [i for i in ids if statuses.get(i) in {'accepted', 'duplicate'}]
        if len(uploaded) != len(ids):
            continue
        client.api.call('POST', f"/api/client/retests/{task['task_id']}/complete", {
            'request_key': f"complete-{task['task_id']}-{task['lease_id'][:8]}",
            'lease_id': task['lease_id'], 'exec_uuids': uploaded, 'installed': installed})
        saved['delivered'] = True
        atomic(Path(home) / 'retest' / ('task-' + str(task['task_id']) + '.json'), saved)
        finished += 1
    return finished
