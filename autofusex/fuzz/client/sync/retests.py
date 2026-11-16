"""Default bounded retest consumer, using the existing private task ledger.

The independent uploader owns this loop. No Torch imports on the network thread;
all model work runs through the existing staged scheduler and physical leases.
"""
from contextlib import closing
from dataclasses import asdict
import json
from pathlib import Path
import socket
import threading
import time
import uuid
from urllib.error import HTTPError
from urllib.request import Request, build_opener

from .client import NoRedirect
from ..contracts import ExecutionConfig, EvaluationConfig, Seed
from ..storage.repository import _connect
from ..storage.records import canonical
from ..scheduling.ledger import TaskLedger

LIMIT = 8


def initialize(ledger):
    with ledger._connection() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS remote_retests_v1 (
            lease_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, payload TEXT NOT NULL,
            finished INTEGER NOT NULL DEFAULT 0)''')


def request(home, route, payload, opener=None):
    config=json.loads((Path(home)/'sync.json').read_text())
    token=Path(config['key_file']).read_text().strip()
    req=Request(config['endpoint']+'/api/retest/'+route,canonical(payload).encode(),
                headers={'Authorization':'Bearer '+token,'Content-Type':'application/json',
                         'User-Agent':'AutofuseX/0.1.2 (+https://esay.app)'})
    with (opener or build_opener(NoRedirect).open)(req,timeout=15) as response:
        raw=response.read(262145)
    if len(raw)>262144:
        raise ValueError('control response too large')
    value=json.loads(raw)
    if not isinstance(value,dict):
        raise ValueError('invalid control response')
    return value


def accept(ledger, item, devices, budget):
    """Persist lease before enqueue, so a crash can safely replay local admission."""
    lease=str(uuid.UUID(item['lease_id']))
    task=str(uuid.UUID(item['task_id']))
    spec=item['spec']
    Seed.parse(spec['seed'])
    from ..backends import BACKEND_NAMES
    if spec['backend'] != 'af-mlir' or type(spec['execution']['static_only']) is not bool:
        raise ValueError('unsupported retest specification')
    evaluation=EvaluationConfig(**spec['evaluation'])
    device='cpu' if spec['backend']=='cpu' else devices[0]
    config=ExecutionConfig(spec['backend'],device,budget.compile_threads,scheduling='staged',
        static_only=spec['execution']['static_only'],batch_id=lease,save_results=True)
    from ..scheduling.supervisor import _validate_task
    _validate_task({'seed':spec['seed'],'execution':canonical(asdict(config)), 'evaluation':canonical(asdict(evaluation))})
    local_id=str(uuid.uuid5(uuid.UUID(lease),'backend'))
    with ledger._connection() as conn:
        prior=conn.execute('SELECT payload FROM remote_retests_v1 WHERE lease_id=?',(lease,)).fetchone()
        if prior and json.loads(prior[0]) != item:
            raise ValueError('lease specification changed')
        conn.execute('INSERT OR IGNORE INTO remote_retests_v1(lease_id,task_id,payload) VALUES (?,?,?)',
                     (lease,task,canonical(item)))
    # Stable identity makes persistence-before-enqueue recoverable.
    from dataclasses import replace
    cpu_id=str(uuid.uuid5(uuid.UUID(lease),'cpu'))
    ledger.enqueue_many([(spec['seed'],replace(config,backend='cpu',device='cpu'),evaluation,cpu_id),
                         (spec['seed'],config,evaluation,local_id)])
    if device != 'cpu':
        from ..scheduling.placement import initialize as init_placement
        with ledger._connection() as conn:
            init_placement(conn)
            conn.execute('INSERT OR IGNORE INTO task_devices VALUES (?,?)',(local_id,canonical(devices)))
    return local_id


def expected_workload(ledger, tasks):
    """Scheduler hook: only local lease-bound tasks acquire a fingerprint guard."""
    with ledger._connection() as conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='remote_retests_v1'").fetchone():
            return None
        hashes=set()
        for task in tasks:
            lease=json.loads(task['execution']).get('batch_id')
            row=conn.execute('SELECT payload FROM remote_retests_v1 WHERE lease_id=?',(lease,)).fetchone()
            if row:
                hashes.add(json.loads(row[0])['spec']['workload_id'])
        if len(hashes)>1:
            raise ValueError('conflicting retest workload identities')
        return next(iter(hashes),None)


def _uploaded(database, attempt):
    with closing(_connect(database)) as conn:
        return conn.execute('''SELECT 1 FROM sync_outbox WHERE attempt_id=?
            AND revision<=(SELECT ack FROM sync_identity WHERE id=1)''',(attempt,)).fetchone() is not None


class Consumer:
    def __init__(self,home,database,*,opener=None,devices=None):
        self.home=Path(home)
        self.database=Path(database)
        self.ledger=TaskLedger(self.home/'tasks/tasks.sqlite')
        self.opener=opener
        self.devices=devices
        self.registered=None

    def api(self,route,payload):
        return request(self.home,route,payload,self.opener)

    def tick(self):
        initialize(self.ledger)
        from ..scheduling.budgets import read_policy, Budget
        from ..config import load_config
        budget=read_policy(self.ledger) or Budget.configured(load_config(self.home))
        if self.devices is None:
            from ..scheduling.devices import select_devices
            # An already-running pool knows only its startup device set. Reuse
            # that set instead of assigning newly discovered cards to it.
            from ..scheduling.hosting import status
            with self.ledger._connection() as conn:
                campaign=conn.execute("SELECT 1 FROM sqlite_master WHERE name='campaign'").fetchone()
                policy=conn.execute('SELECT payload FROM campaign WHERE id=1').fetchone() if campaign else None
            assigned=json.loads(policy[0]).get('devices') if policy else None
            self.devices=assigned or select_devices()
        with self.ledger._connection() as conn:
            standalone=conn.execute("SELECT 1 FROM tasks WHERE state IN ('pending','running','stop_requested') AND json_extract(execution,'$.scheduling')!='staged' LIMIT 1").fetchone()
        if standalone:
            return {'state':'waiting_for_combined_queue'}
        # Sync source, not installation identity: completion checks its actual
        # uploader provenance. Register even before any local result exists.
        from .outbox import pending
        client=pending(self.database,limit=1)['source_id']
        if self.registered is None or time.monotonic()-self.registered > 30:
            from ..runtime import machine_identity
            self.api('register',dict(client_id=client,machine_id=machine_identity(self.home),
                name=socket.gethostname()[:80],backends=['cpu','eager-npu','triton-npu','af-legacy','af-mlir']))
            self.registered=time.monotonic()
        with self.ledger._connection() as conn:
            active=[dict(r) for r in conn.execute('SELECT * FROM remote_retests_v1 WHERE finished=0 LIMIT 8')]
        if active:
            from ..scheduling.hosting import status
            if status(self.ledger).get('ownership')=='unowned':
                from ..scheduling.ownership import own_queue
                from ..scheduling.supervisor import _reconcile
                try:
                    with own_queue(self.ledger):
                        _reconcile(self.ledger,self.database)
                except ValueError:
                    pass  # A supervisor may have taken ownership since status().
        for remote in active:
            item=json.loads(remote['payload']);lease=remote['lease_id']
            local_id=str(uuid.uuid5(uuid.UUID(lease),'backend'))
            try:
                local=self.ledger.get(local_id)
            except ValueError:
                local=None
            local_state=local['state'] if local else 'pending'
            state='running' if local_state in {'running','completed','stop_requested'} else 'leased'
            try:
                renewal=self.api('renew',dict(client_id=client,lease_id=lease,state=state))
            except HTTPError as exc:
                if exc.code not in {400,401,403}:raise
                if local:
                    self.ledger.stop(local_id)
                    self.ledger.stop(str(uuid.uuid5(uuid.UUID(lease),'cpu')))
                # Stop local work before retiring an expired lease. Late completed
                # evidence can still be uploaded and attributed by complete().
                if local_state in {'running','stop_requested'}:continue
                if local and local.get('attempt_id') and _uploaded(self.database,local['attempt_id']):
                    self.api('complete',dict(client_id=client,lease_id=lease,attempt_id=local['attempt_id']))
                self._finish(lease)
                continue
            if renewal['state'] in {'cancel_requested','cancelled','interrupted','completed'}:
                if local:
                    self.ledger.stop(local_id)
                    self.ledger.stop(str(uuid.uuid5(uuid.UUID(lease),'cpu')))
                if local_state in {'running','stop_requested'}:continue
                if renewal['state']=='cancel_requested':
                    self.api('renew',dict(client_id=client,lease_id=lease,state='cancelled'))
                self._finish(lease)
                continue
            if local is None:
                accept(self.ledger,item,self.devices,budget)
            elif local_state=='completed' and _uploaded(self.database,local['attempt_id']):
                peer=self.ledger.get(str(uuid.uuid5(uuid.UUID(lease),'cpu')))
                if peer['state']!='completed' or not _uploaded(self.database,peer['attempt_id']):
                    continue
                self.api('complete',dict(client_id=client,lease_id=lease,attempt_id=local['attempt_id']))
                self._finish(lease)
            elif local_state in {'interrupted','cancelled'}:
                self.api('renew',dict(client_id=client,lease_id=lease,state='interrupted'))
                self._finish(lease)
        with self.ledger._connection() as conn:
            capacity=conn.execute('SELECT max_active FROM queue_config WHERE id=1').fetchone()[0]
            queued=conn.execute("SELECT count(*) FROM tasks WHERE state NOT IN ('completed','cancelled')").fetchone()[0]
            inflight=conn.execute('SELECT count(*) FROM remote_retests_v1 WHERE finished=0').fetchone()[0]
        room=min(LIMIT-inflight,max(0,(capacity-queued)//2))
        if room:
            answer=self.api('claim',dict(client_id=client,limit=room))
            if not isinstance(answer.get('items'),list) or len(answer['items'])>room:
                raise ValueError('invalid claim size')
            for item in answer['items']:
                accept(self.ledger,item,self.devices,budget)
        # An existing supervisor consumes the same ledger. Never start a second
        # resource pool. A finite idle drain preserves an existing campaign.
        from ..scheduling.hosting import status, launch
        if status(self.ledger).get('ownership')=='unowned':
            with self.ledger._connection() as conn:
                pending_task=conn.execute("SELECT 1 FROM tasks t JOIN remote_retests_v1 r ON json_extract(t.execution,'$.batch_id')=r.lease_id WHERE t.state='pending' AND r.finished=0 LIMIT 1").fetchone()
            if pending_task:
                launch(self.ledger,self.database,[],budget=budget,devices=self.devices,preserve_campaign=True)
        return {'inflight':inflight}

    def _finish(self,lease):
        with self.ledger._connection() as conn:
            conn.execute('UPDATE remote_retests_v1 SET finished=1 WHERE lease_id=?',(lease,))


def start(home,database):
    def run():
        consumer=Consumer(home,database)
        while True:
            try:
                result=consumer.tick()
                state={'state':'ok',**result}
            except Exception as exc:
                state={'state':'retrying','error':type(exc).__name__}
            from ..runtime import atomic
            atomic(Path(home)/'retest-status.json',dict(state,time=time.time()))
            time.sleep(10)
    thread=threading.Thread(target=run,name='retest-consumer',daemon=True)
    thread.start()
    return thread
