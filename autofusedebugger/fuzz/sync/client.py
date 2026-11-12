"""Independent, retryable HTTPS uploader; no execution dependencies."""
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen, build_opener, HTTPRedirectHandler

from .outbox import pending, acknowledge


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        raise ValueError('upload redirects forbidden; configure canonical HTTPS endpoint')


def configure(home, endpoint, key_file, interval=30):
    home=Path(home).expanduser().resolve();key_file=Path(key_file).expanduser().resolve()
    url=urlsplit(endpoint)
    if url.scheme!='https' or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError('sync requires a canonical HTTPS endpoint')
    if not isinstance(interval,(int,float)) or not math.isfinite(interval) or not 1<=interval<=86400:
        raise ValueError('sync interval must be 1..86400 seconds')
    if not key_file.is_file() or not 20<=key_file.stat().st_size<=256:
        raise ValueError('administrator key file unavailable or invalid')
    if os.name=='posix' and key_file.stat().st_mode&0o077:
        raise ValueError('administrator key file must be private (0600)')
    home.mkdir(parents=True,exist_ok=True)
    config={'version':1,'endpoint':endpoint.rstrip('/'),'key_file':str(key_file),'interval':interval}
    import tempfile
    fd,name=tempfile.mkstemp(prefix='.sync-config-',dir=home)
    try:
        with os.fdopen(fd,'w') as out:
            json.dump(config,out);out.flush();os.fsync(out.fileno())
        os.replace(name,home/'sync.json')
    finally:
        Path(name).unlink(missing_ok=True)
    return {'configured':True,'interval':interval,'endpoint':config['endpoint']}


def _once(home, database, *, opener=None):
    home=Path(home); config=json.loads((home/'sync.json').read_text())
    batch=pending(database)
    if not batch['items']:
        from .facts import upload
        return dict(pending=0,ack=batch['after'],**upload(home,database,opener))
    key=Path(config['key_file']).read_text().strip()
    body=json.dumps(batch,separators=(',',':'),allow_nan=False).encode()
    if len(body)>2*1024*1024:raise ValueError('sync batch exceeds size bound')
    request=Request(config['endpoint']+'/api/upload',body,headers={'Content-Type':'application/json','Authorization':'Bearer '+key,'User-Agent':'AutofuseDebugger/0.1.2 (+https://esay.app)'})
    open_request=opener or build_opener(NoRedirect).open
    with open_request(request,timeout=15) as response:
        data=response.read(8193)
        if len(data)>8192:raise ValueError('oversized ACK')
        ack=json.loads(data)
    # Never let server skip local pending revisions not in this request.
    if ack.get('source_id')!=batch['source_id'] or type(ack.get('ack')) is not int or ack['ack']!=batch['items'][-1]['revision']:
        raise ValueError('unexpected sync ACK')
    acknowledge(database,batch['source_id'],ack['ack'])
    return {'uploaded':len(batch['items']),'ack':ack['ack']}


def once(home, database, *, opener=None):
    import fcntl
    home=Path(home)
    with (home/'sync-upload.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _once(home,database,opener=opener)


def run(home,database, _fd=None):
    import fcntl
    home=Path(home).resolve();home.mkdir(parents=True,exist_ok=True)
    with (os.fdopen(_fd,'a') if _fd is not None else (home/'sync.lock').open('a')) as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('sync daemon already running') from None
        from ..runtime import start_heartbeat
        start_heartbeat(home)
        from .retests import start as start_retests
        start_retests(home,database)
        while True:
            interval=30
            try:
                config=json.loads((home/'sync.json').read_text());interval=config['interval']
                # Bounded drain each interval; retry independent of tests.
                for _ in range(20):
                    result=once(home,database)
                    if not (result.get('uploaded') or result.get('fact_uploaded')):break
                state={'state':'ok','time':time.time(),**result}
            except Exception as exc:
                state={'state':'retrying','time':time.time(),'error':type(exc).__name__}
            from tempfile import NamedTemporaryFile
            with NamedTemporaryFile(mode='w',dir=home,delete=False) as out:
                json.dump(state,out);name=out.name
            os.replace(name,home/'sync-status.json')
            time.sleep(max(1,min(float(interval),86400)))


def start(home, database):
    """Portable POSIX-container fallback, no systemd assumption; single flock daemon."""
    import subprocess
    import sys
    import fcntl
    home=Path(home).resolve()
    if not (home/'sync.json').is_file():
        raise ValueError('configure administrator endpoint/key before starting sync')
    with (home/'sync.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return {'state':'already_running'}
        # Transfer lock across launch to prevent duplicate daemons/start races.
        child=subprocess.Popen([sys.executable,'-m','autofusedebugger.fuzz.sync.daemon',
            str(home),str(Path(database).resolve()),str(lock.fileno())],
            pass_fds=(lock.fileno(),),start_new_session=True,stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        import threading
        threading.Thread(target=child.wait,daemon=True).start()
        return {'state':'started','pid':child.pid,'interval':json.loads((home/'sync.json').read_text())['interval']}


def reset(home, database):
    """Administrative explicit resynchronization, serialized against daemon uploads."""
    import fcntl
    from .outbox import reset_epoch
    with (Path(home)/'sync-upload.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return reset_epoch(database)


def status(home, database):
    """Local-only status; never expose administrator credentials."""
    import sqlite3
    result = {'state': 'not_started', 'pending': 0, 'ack': 0}
    home = Path(home)
    state = home/'sync-status.json'
    if state.exists():
        result.update(json.loads(state.read_text()))
    if Path(database).exists():
        with sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro', uri=True) as conn:
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name='sync_identity'").fetchone():
                source, ack = conn.execute('SELECT source_id,ack FROM sync_identity WHERE id=1').fetchone()
                total = conn.execute('SELECT count(*) FROM attempts').fetchone()[0]
                acknowledged = conn.execute('SELECT count(*) FROM sync_outbox WHERE revision<=?', (ack,)).fetchone()[0]
                result.update(source_id=source, ack=ack, pending=total-acknowledged)
    if (home/'sync.json').exists():
        config = json.loads((home/'sync.json').read_text())
        result.update(endpoint=config['endpoint'], interval=config['interval'])
    return result


def verify_identity(home, *, opener=None):
    """Validate administrator-issued credentials without changing local results."""
    config = json.loads((Path(home)/'sync.json').read_text())
    key = Path(config['key_file']).read_text().strip()
    request = Request(config['endpoint']+'/api/identity', headers={
        'Authorization': 'Bearer '+key, 'User-Agent': 'AutofuseDebugger/0.1.2 (+https://esay.app)'})
    with (opener or build_opener(NoRedirect).open)(request, timeout=15) as response:
        data = response.read(4097)
        if len(data) > 4096:
            raise ValueError('oversized identity response')
        identity = json.loads(data)
    import uuid
    uuid.UUID(identity['user_id'])
    if not isinstance(identity['display_name'], str) or not 1 <= len(identity['display_name']) <= 80:
        raise ValueError('invalid administrator identity')
    return identity
