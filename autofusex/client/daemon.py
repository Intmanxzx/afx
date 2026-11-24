"""Small persistent management service with local-only IPC and outbound control."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import fcntl
import json
import os
from pathlib import Path
import socket
import sqlite3
import socketserver
import subprocess
import threading
import time
import hashlib

from autofusex.contracts.configuration import canonical, quantity
from autofusex.client.store import Store
from autofusex.client.operations import validate_operation, execute


from autofusex.contracts.ipc import _socket_path, request


class Manager:
    def __init__(self,home):
        self.store=Store(home);self.stop=threading.Event();self.processes={};self.pool=ThreadPoolExecutor(max_workers=1)
        self.active=None;self.control_lock=threading.Lock()

    def dispatch(self,body):
        action=body.get('action')
        if action in {'start_session','replay','fuzz_replay','platform_retest'} and self.store.get('client_switch', {}).get('state') in {'draining','staging','requested','stage_failed'}:
            raise ValueError('client upgrade blocks task admission')
        if action=='resume_upgrade':
            from autofusex.client.tasks import start
            switch=self.store.get('client_switch', {})
            results=[]
            from autofusex.client.cleanup import access
            with access(self.store.home):
                for identity in switch.get('old_runs', []):
                    prepared=json.loads((self.store.home/'runs'/identity/'spec.json').read_text())
                    results.append(start(self.store,prepared['spec'],'upgrade-resume',self.processes,resume_run=identity))
            return dict(runs=results)
        if action=='drain_upgrade':
            from autofusex.client.upgrade_drain import begin
            return dict(runs=begin(self.store, self.processes))
        if action=='start_session':
            from autofusex.client.local_session import start
            return start(self, dump=body.get('dump'))
        if action=='platform_retest':
            from autofusex.client.retest_queue import submit
            return submit(self.store.home, body['task'])
        if action=='fuzz_replay':
            from autofusex.client.replay import submit_fuzz
            return submit_fuzz(self.store, body['options'], self.processes)
        if action=='replay':
            from autofusex.client.replay import submit
            return submit(self.store, body['seeds'], self.processes, dump=body.get('dump',False))
        if action=='status': return self.snapshot()
        if action=='stop_session':
            from autofusex.client.tasks import action as task_action
            for identity in list(self.processes):
                task_action(self.store,identity,'stop',self.processes)
            self.store.put('shutdown_when_idle',False)
            return dict(stopping=True)
        if action=='shutdown':
            if self.active or self.processes: raise ValueError('finish active operations and tasks before stopping daemon')
            self.store.put('normal_shutdown',True)
            self.stop.set();return dict(stopping=True)
        if action=='submit':
            payload=validate_operation(body['kind'],body['payload'])
            return self.store.submit(body['kind'],payload,body['request_key'])
        if action=='operation': return self.store.operation(body['id'])
        if action=='events': return self.store.events(body['id'],int(body.get('after',0)))
        if action=='query': return self.query(body.get('query'), body.get('value'))
        raise ValueError('unknown local action')

    def query(self, query, value=None):
        if query == 'environments':
            with self.store.connect() as db:
                return {'items': [dict(id=r['id'], name=r['name'], state=r['state']) for r in db.execute('SELECT id,name,state FROM environments ORDER BY name')]}
        if query == 'environment':
            with self.store.connect() as db:
                rows=db.execute('SELECT manifest FROM environments WHERE id=? OR name=?',(value,value)).fetchall()
            if len(rows)!=1: raise LookupError('environment not found or ambiguous')
            return json.loads(rows[0][0])
        if query == 'runs':
            with self.store.connect() as db:
                rows=db.execute('SELECT id,state,pid,result FROM runs ORDER BY rowid DESC LIMIT 100').fetchall()
            items=[]
            for r in rows:
                item=dict(id=r['id'],state=r['state'],pid=r['pid'],result=json.loads(r['result']) if r['result'] else None)
                log=self.store.home/'runs'/r['id']/'execution.log'
                if log.exists(): item['log']=str(log)
                items.append(item)
            return {'items':items}
        if query == 'run':
            with self.store.connect() as db:
                row=db.execute('SELECT * FROM runs WHERE id=?',(value,)).fetchone()
            if not row: raise LookupError('run not found')
            result=dict(id=row['id'],state=row['state'],pid=row['pid'],spec=json.loads(row['spec']),result=json.loads(row['result']) if row['result'] else None)
            log=self.store.home/'runs'/row['id']/'execution.log'
            if log.exists(): result['log']=str(log)
            return result
        if query == 'tools':
            requested=value
            root=self.store.home/'tools'; items=[]
            if root.exists():
                for manifest in root.glob('*/ready.json'):
                    release=json.loads(manifest.read_text())
                    items.append(dict(id=release.get('id'), path=str(manifest.parent), state='staged'))
            if requested is not None:
                items=[item for item in items if item['id']==requested]
            return {'items':items}
        raise ValueError('unknown local query')

    def snapshot(self):
        with self.store.connect() as db:
            runs=[dict(id=r['id'],state=r['state'],summary=json.loads(r['result']) if r['result'] else {}) for r in db.execute('SELECT id,state,result FROM runs ORDER BY rowid DESC LIMIT 20')]
            environments=[dict(id=r['id'],name=r['name']) for r in db.execute("SELECT id,name FROM environments WHERE state='ready' ORDER BY name LIMIT 20")]
        for run in runs:
            summary=run.pop('summary', {})
            if run['state'] not in {'completed','failed','stopped','cancelled','interrupted'}:
                continue
            if summary.get('cleanup'):
                run.update(cleanup=summary['cleanup'],result_count=summary.get('result_count',0))
                continue
            path=self.store.home/'runs'/run['id']/'results.sqlite'
            if not path.exists():
                run['result_count']=0
                continue
            try:
                with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=.1)) as results:
                    run['result_count']=results.execute('SELECT count(*) FROM attempts').fetchone()[0]
            except sqlite3.Error:
                pass
        return dict(protocol=1, pid=os.getpid(), agent_python=os.sys.executable, agent_version=__import__('importlib.metadata', fromlist=['version']).version('autofusex'), paired=False,config=self.store.get('config_state',{}),
                    active_operation=self.active[0] if self.active else None,runs=runs,environments=environments,cleanup=self.store.get('cleanup_state',{}))

    def tick(self):
        from autofusex.client.tasks import refresh
        refresh(self.store,self.processes)
        from autofusex.client.session_control import reconcile
        reconcile(self)
        from autofusex.client.platform_retests import tick as retest_tick
        retest_tick(self)
        if self.store.get('shutdown_when_idle') and not self.processes and not self.active:
            self.store.put('shutdown_when_idle',False)
            self.store.put('normal_shutdown',True)
            self.stop.set()
            return
        if self.active and self.active[1].done():
            identity,future=self.active
            try:
                result = future.result()
                staged = result and result.get('state') == 'staged' and self.store.operation(identity)['kind'] == 'client.update'
                self.store.state(identity, 'activating' if staged else 'succeeded', result)
            except Exception as exc:
                # Exception text may contain credentials/URLs from libraries. Only expose safe class/code.
                cancelled=self.store.get('cancel:'+identity,False)
                state='needs_attention' if cancelled else 'failed'
                message='Cancelled during execution; inspect partial installation before retrying' if cancelled else 'Operation failed; inspect local policy and prerequisites'
                self.store.state(identity,state,dict(code=type(exc).__name__,message=message))
            self.active=None
        with self.store.connect() as db:
            controls=db.execute("SELECT id FROM operations WHERE state='queued' AND kind IN ('task.action','operation.cancel') ORDER BY created LIMIT 8").fetchall()
        for row in controls:
            operation=self.store.operation(row['id'])
            try:
                self.store.state(operation['id'],'running')
                self.store.state(operation['id'],'succeeded',execute(self.store,operation,self.processes))
            except Exception as exc:self.store.state(operation['id'],'failed',dict(code=type(exc).__name__))
        if not self.active:
            for operation in self.store.pending():
                if self.store.get('client_switch', {}).get('state') in {'draining', 'staging', 'requested', 'activating'}:
                    continue
                session = self.store.get('session_control', {})
                if operation['kind'] == 'task.submit' and session.get('state') in {'stopping', 'stopped'}:
                    continue
                if operation['kind']=='task.submit' and self.processes:
                    continue
                self.store.state(operation['id'],'running')
                self.active=(operation['id'],self.pool.submit(self.perform,operation))
                break

    def perform(self,operation):
        from autofusex.client.cancellation import bind
        bind(lambda:self.store.get('cancel:'+operation['id'],False))
        try:return execute(self.store,operation,self.processes)
        finally:bind(lambda:False)

    def synchronize(self):
        from autofusex.client.cleanup import access
        with access(self.store.home):
            return self._synchronize()

    def _synchronize(self):
        client_path=self.store.home/'client.json'
        if client_path.exists():
            client=json.loads(client_path.read_text())
            if client.get('mode')=='local': return
            if client.get('mode')=='managed':
                from autofusex.client.platform_sync import synchronize
                return synchronize(self, client)

    def run(self):
        store=self.store
        with (store.home/'manager.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            store.recover()
            with store.connect() as db:
                db.execute("UPDATE runs SET state='needs_attention' WHERE state IN ('preparing','running','pausing','paused','draining','cancelling')")
            sockpath=_socket_path(store.home);sockpath.unlink(missing_ok=True)
            if sockpath != store.home/'manager.sock':
                (store.home/'manager.sock.path').write_text(str(sockpath))
            manager=self
            class Handler(socketserver.StreamRequestHandler):
                def handle(self):
                    self.request.settimeout(5)
                    try:
                        raw=self.rfile.readline(65537)
                        if len(raw)>65536: raise ValueError('request too large')
                        value=manager.dispatch(json.loads(raw));result=dict(ok=True,data=value)
                    except Exception as exc: result=dict(ok=False,error=type(exc).__name__+': '+str(exc)[:300])
                    self.wfile.write(canonical(result).encode()+b'\n')
            server=socketserver.UnixStreamServer(str(sockpath),Handler);os.chmod(sockpath,0o600);server.timeout=.1
            network=ThreadPoolExecutor(max_workers=1);future=None;next_sync=0
            maintenance=ThreadPoolExecutor(max_workers=1,thread_name_prefix="retention");cleaning=None;next_cleanup=0;cleanup_policy=None;next_cleanup_config=0
            try:
                while not self.stop.is_set():
                    server.handle_request();self.tick()
                    from autofusex.client.cleanup import settings, sweep
                    if time.monotonic() >= next_cleanup_config:
                        try: configured = settings(store)
                        except Exception: configured = None
                        if configured != cleanup_policy:
                            cleanup_policy = configured
                            next_cleanup = 0
                        next_cleanup_config = time.monotonic()+10
                    if cleaning and cleaning.done():
                        try: cleaning.result()
                        except Exception as exc: store.put('cleanup_state',dict(state='error',error=type(exc).__name__))
                        cleaning=None
                    if cleaning is None and time.monotonic() >= next_cleanup:
                        cleaning=maintenance.submit(sweep,store)
                        next_cleanup=time.monotonic()+3600
                    if future and future.done():
                        try: future.result();store.put('connection',dict(state='ok',time=time.time()))
                        except Exception as exc: store.put('connection',dict(state='retrying',error=type(exc).__name__,time=time.time()))
                        future=None
                    if future is None and time.monotonic()>=next_sync:
                        future=network.submit(self.synchronize)
                        interval=quantity(store.get('effective_policy',{}).get('sync',{}).get('interval','10s'))
                        next_sync=time.monotonic()+interval
            finally:
                from autofusex.client.platform_retests import stop as stop_retests
                stop_retests(self)
                server.server_close();sockpath.unlink(missing_ok=True)
                network.shutdown(wait=True);self.pool.shutdown(wait=True);maintenance.shutdown(wait=True)
