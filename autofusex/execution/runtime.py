"""Bounded local live snapshots for the platform client to publish.

No Torch imports or network on the scheduler thread. Only current snapshots are
sent; offline observations are never replayed as fresh activity.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import uuid


def machine_identity(home):
    for name in ('/sys/class/dmi/id/product_uuid','/etc/machine-id'):
        try:
            value=Path(name).read_text().strip()
            if value and set(value.lower().replace('-','')) not in ({'0'},{'f'}):
                return hashlib.sha256(('autofuse-host-v1:'+value).encode()).hexdigest()
        except OSError:pass
    path=Path(home)/'machine-id'
    try:
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as out:out.write(str(uuid.uuid4()))
    except FileExistsError:pass
    return path.read_text().strip()


def boot_id():
    try:return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except OSError:return 'unknown'


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    fd,name=tempfile.mkstemp(prefix='.runtime-',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as out:json.dump(value,out)
        os.replace(name,path)
    finally:Path(name).unlink(missing_ok=True)


class Reporter:
    def __init__(self,home,devices):
        self.home=Path(home);self.identity=str(uuid.uuid4());self.path=self.home/'runtime'/f'{self.identity}.json'
        self.devices=devices;self.start=time.monotonic();self.last=0;self.sequence=0
        self.completed=0;self.success=0;self.case_errors=0;self.backends=set()
        # Publish run snapshots only when the platform client is configured to report them.
        self.enabled=(self.home/'platform-client.json').is_file()
        self.versions={}
        if self.enabled:
            from autofusex.execution.environment import collect
            try:self.versions={k:str(v) for k,v in collect(self.home).items() if k in {'package_version','autofuse_binary','cann','torch','torch_npu'} and v}
            except Exception:pass

    def record(self,result,backend):
        if backend in {'cpu'}:return
        self.backends.add(backend)
        shapes=result.get('shape_results',[])
        if not shapes:self.case_errors+=1
        for row in shapes:
            correctness=row.get('correctness');performance=row.get('performance')
            if correctness not in {'pass','fail','error'} and performance not in {'measured','error'}:continue
            self.completed+=1
            self.success+=int(correctness not in {'fail','error'} and performance!='error')

    def publish(self,cases,stages,pool,queued,summary,*,state='running',force=False):
        if not self.enabled:return
        now=time.monotonic()
        if not force and now-self.last<5:return
        self.last=now;self.sequence+=1
        busy={x.get('device_uuid') for x in pool.leases.values()}
        counts={}
        for s in stages.values():
            if s.state=='running':counts[s.kind.value]=counts.get(s.kind.value,0)+1
        for c in cases.values():
            for task in c.tasks:self.backends.add(json.loads(task['execution'])['backend'])
        value=dict(run_id=self.identity,sequence=self.sequence,elapsed=now-self.start,state=state,
                   devices=[dict(uuid=d.uuid,model=d.model,logical=d.logical,busy=d.uuid in busy and state=='running') for d in self.devices],
                   stages=counts if state=='running' else {},queued=queued if state=='running' else 0,active_cases=len(cases) if state=='running' else 0,
                   completed=self.completed,success=self.success,case_errors=self.case_errors,record_failed=summary.get('record_failed',0),
                   backend=sorted(self.backends-{'cpu'}),versions=self.versions,
                   workers=dict(spawned=pool.stats['spawned'],reused=pool.stats['reused'],recycled=dict(pool.stats['recycled'])))
        try:atomic(self.path,dict(monotonic=now,boot=boot_id(),sample=value))
        except OSError:pass  # Observability must never abort execution.
