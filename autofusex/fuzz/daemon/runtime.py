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
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.parse import urlsplit

from .management.config import canonical, quantity
from .management.store import Store, TERMINAL
from .management.operations import validate_operation, execute


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        raise ValueError('control endpoint redirects are forbidden')


def remote(endpoint,route,payload,token=None):
    parsed=urlsplit(endpoint)
    if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('server endpoint must be credential-free HTTPS')
    headers={'Content-Type':'application/json'}
    if token: headers['Authorization']='Bearer '+token
    request=Request(endpoint.rstrip('/')+'/api/manage/'+route,canonical(payload).encode(),headers=headers)
    with build_opener(NoRedirect).open(request,timeout=15) as response:
        raw=response.read(65537)
    if len(raw)>65536: raise ValueError('oversized server response')
    return json.loads(raw)


def _socket_path(home):
    home = Path(home)
    default = home / 'manager.sock'
    if len(str(default).encode()) <= 100:
        return default
    return Path('/tmp') / ('afx-manager-' + hashlib.sha256(str(home).encode()).hexdigest()[:16] + '.sock')

def request(home,payload):
    sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);sock.settimeout(20)
    try:
        sock.connect(str(_socket_path(home)));sock.sendall(canonical(payload).encode()+b'\n')
        with sock.makefile('rb') as stream: raw=stream.readline(131073)
        if len(raw)>131072: raise ValueError('oversized local response')
        result=json.loads(raw)
        if not result.get('ok'): raise ValueError(result.get('error','local operation failed'))
        return result['data']
    finally: sock.close()


class Manager:
    def __init__(self,home):
        self.store=Store(home);self.stop=threading.Event();self.processes={};self.pool=ThreadPoolExecutor(max_workers=1)
        self.active=None;self.control_lock=threading.Lock()

    def dispatch(self,body):
        action=body.get('action')
        if action=='start_session':
            from .management.local_session import start
            return start(self)
        if action=='replay':
            from .management.replay import submit
            return submit(self.store, body['seeds'], self.processes)
        if action=='status': return self.snapshot()
        if action=='stop_session':
            from .management.tasks import action as task_action
            for identity in list(self.processes):
                task_action(self.store,identity,'stop',self.processes)
            self.store.put('shutdown_when_idle',True)
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
        if action=='effective': return self.store.get('effective_policy',{})
        if action=='query': return self.query(body.get('query'), body.get('value'))
        if action=='revoke':
            credential=self.store.get('credential')
            if not credential: raise ValueError('device is not paired')
            result=remote(credential['endpoint'],'revoke',{},credential['token'])
            self.store.put('credential',None)
            return dict(result,local_credential='removed')
        if action=='pair':
            if not body.get('endpoint','').startswith('https://'):
                raise ValueError('pair endpoint must use HTTPS')
            result=remote(body['endpoint'],'pair',dict(name=body['name']))
            self.store.put('pairing',dict(endpoint=body['endpoint'],**result))
            return {k:v for k,v in result.items() if k!='secret'}
        if action=='enroll':
            result=remote(body['endpoint'],'enroll',dict(name=body['name'],token=body['token']))
            self.store.put('credential',dict(endpoint=body['endpoint'],device_id=result['device_id'],token=result['token']))
            self.store.put('pairing',None)
            return dict(device_id=result['device_id'],state='paired')
        if action=='trust':
            # Local socket only. Server operations cannot modify this authority.
            from .artifacts import atomic
            if not isinstance(body['policy'],dict): raise ValueError('local trust object required')
            atomic(self.store.home/'trust.json',body['policy']);return dict(saved=True)
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
            runs=[dict(id=r['id'],state=r['state']) for r in db.execute('SELECT id,state FROM runs ORDER BY rowid DESC LIMIT 20')]
            environments=[dict(id=r['id'],name=r['name']) for r in db.execute("SELECT id,name FROM environments WHERE state='ready' ORDER BY name LIMIT 20")]
        for run in runs:
            if run['state'] not in {'completed','failed','stopped','cancelled','interrupted'}:
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
        return dict(protocol=1,paired=bool(self.store.get('credential')),config=self.store.get('config_state',{}),
                    active_operation=self.active[0] if self.active else None,runs=runs,environments=environments)

    def tick(self):
        from .management.tasks import refresh
        refresh(self.store,self.processes)
        if self.store.get('shutdown_when_idle') and not self.processes and not self.active:
            self.store.put('shutdown_when_idle',False)
            self.store.put('normal_shutdown',True)
            self.stop.set()
            return
        if self.active and self.active[1].done():
            identity,future=self.active
            try: self.store.state(identity,'succeeded',future.result())
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
                session = self.store.get('session_control', {})
                if operation['kind'] == 'task.submit' and session.get('state') in {'stopping', 'stopped'}:
                    continue
                if operation['kind']=='task.submit' and self.processes:
                    continue
                self.store.state(operation['id'],'running')
                self.active=(operation['id'],self.pool.submit(self.perform,operation))
                break

    def perform(self,operation):
        from .management.cancellation import bind
        bind(lambda:self.store.get('cancel:'+operation['id'],False))
        try:return execute(self.store,operation,self.processes)
        finally:bind(lambda:False)

    def synchronize(self):
        client_path=self.store.home/'client.json'
        if client_path.exists():
            client=json.loads(client_path.read_text())
            if client.get('mode')=='local': return
            if client.get('mode')=='managed':
                # Platform facts and heartbeat run from the active client
                # release. The bootstrap supervisor must not own this schema:
                # installing a new client therefore updates all future facts.
                active = self.store.home / 'active-client.json'
                python = None
                if active.exists():
                    try:
                        candidate = json.loads(active.read_text()).get('python')
                        if isinstance(candidate, str) and Path(candidate).is_file():
                            python = candidate
                    except (OSError, ValueError, TypeError):
                        pass
                if python is None:
                    python = os.sys.executable
                command = [python, '-m', 'autofusex.fuzz.daemon.management.platform_agent',
                           '--data-dir', str(self.store.home)]
                subprocess.run(command, check=True, timeout=120,
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, text=True)
                return
        pairing=self.store.get('pairing');credential=self.store.get('credential')
        if pairing and not credential:
            result=remote(pairing['endpoint'],'exchange',dict(id=pairing['id'],secret=pairing['secret']))
            if result['state']=='paired':
                credential=dict(endpoint=pairing['endpoint'],device_id=result['device_id'],token=result['token'])
                self.store.put('credential',credential);self.store.put('pairing',None)
        if not credential: return
        # Persisted mapping/outbox survives lost acknowledgements and restart.
        mappings=self.store.get('remote_operations',{})
        for identity,mapping in list(mappings.items()):
            operation=self.store.operation(identity)
            if operation['state'] not in TERMINAL: continue
            result=operation['result']
            # Never upload credentials, host environment, full specs or arbitrary installer output.
            allowed={'id','plan_id','run_id','state','hash','config_hash','applied','scope','existing_tasks_unchanged','code','message','build_id','verification','npu_verified','release_id','epoch','python','architecture','environments'}
            safe={k:v for k,v in (result or {}).items() if k in allowed}
            remote(credential['endpoint'],'ack',dict(id=identity,generation=mapping['generation'],state=operation['state'],result=safe),credential['token'])
            del mappings[identity];self.store.put('remote_operations',mappings)
        from .management.uploads import upload_runs
        upload_runs(self.store,credential)
        response=remote(credential['endpoint'],'poll',dict(status=self.snapshot()),credential['token'])
        revision=response['revision'];state=self.store.get('config_state',{})
        if revision and revision!=state.get('revision'):
            operation=self.store.submit('config.apply',validate_operation('config.apply',response['config']),'remote-config:'+str(revision))
            self.store.put('config_state',dict(revision=revision,state='received',operation_id=operation['id']))
        state=self.store.get('config_state',{})
        if state.get('operation_id'):
            operation=self.store.operation(state['operation_id'])
            state['state']='applied' if operation['state']=='succeeded' else 'rejected' if operation['state'] in TERMINAL else operation['state']
            self.store.put('config_state',state)
        for item in response['operations']:
            if item['expires']<time.time(): continue
            self.store.submit(item['kind'],validate_operation(item['kind'],item['payload']), 'remote:'+item['id'],item['id'])
            mappings[item['id']]=dict(generation=item['generation'])
        self.store.put('remote_operations',mappings)

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
            try:
                while not self.stop.is_set():
                    server.handle_request();self.tick()
                    if future and future.done():
                        try: future.result();store.put('connection',dict(state='ok',time=time.time()))
                        except Exception as exc: store.put('connection',dict(state='retrying',error=type(exc).__name__,time=time.time()))
                        future=None
                    if future is None and time.monotonic()>=next_sync:
                        future=network.submit(self.synchronize)
                        interval=quantity(store.get('effective_policy',{}).get('sync',{}).get('interval','10s'))
                        next_sync=time.monotonic()+interval
            finally:
                server.server_close();sockpath.unlink(missing_ok=True)
                network.shutdown(wait=True);self.pool.shutdown(wait=True)
