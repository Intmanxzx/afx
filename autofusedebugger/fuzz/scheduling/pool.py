"""Atomic multi-resource admission and bounded, isolated persistent processes.

Reservations are conservative admission budgets; /proc and runtime samples are
additional guards, not cgroup hard isolation. Idle processes count toward RSS.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import uuid

from .stage_worker import send, receive


@dataclass
class Device:
    logical: str
    uuid: str
    capacity: int
    model: str = "unrecorded"


def discover(devices):
    if not devices:
        return []
    code = '''import json,torch,torch_npu,sys
out=[]
for name in json.loads(sys.argv[1]):
 p=torch.npu.get_device_properties(name)
 free,total=torch.npu.mem_get_info(name)
 out.append({'logical':name,'uuid':str(p.uuid),'capacity':int(free),'model':str(p.name)})
print(json.dumps(out))'''
    proc = subprocess.run([sys.executable, '-c', code, json.dumps(devices)], capture_output=True, text=True, timeout=45)
    if proc.returncode:
        raise ValueError('physical NPU resource discovery failed')
    for line in reversed(proc.stdout.splitlines()):
        try:
            rows = json.loads(line)
            result = [Device(**r) for r in rows]
            if len({d.uuid for d in result}) != len(result):
                raise ValueError('logical devices alias one physical NPU')
            return result
        except (json.JSONDecodeError, TypeError, KeyError):
            continue
    raise ValueError('NPU discovery returned no inventory')


class Pool:
    def __init__(self, budget, directory, inherited_fd, devices, *, max_tasks=16):
        from ..resources import resource_inventory
        inventory = resource_inventory(directory)
        self.budget = budget
        self.directory = Path(directory)
        self.fd = inherited_fd
        self.devices = devices
        self.cpu_capacity = budget.cpu_workers * budget.compile_threads
        self.memory_capacity = budget.memory_budget_bytes or max(512*1024**2, (inventory['effective_memory_bytes'] or 8*1024**3)//2)
        self.cache_capacity = budget.cache_budget_bytes or min(2*1024**3, inventory['disk_free_bytes']//4)
        self.max_tasks = max_tasks
        self.workers = []
        self.leases = {}
        self.executor = ThreadPoolExecutor(max_workers=budget.cpu_workers, thread_name_prefix='stage-io')
        import threading
        self.lock = threading.Lock()
        self.cancelled = set()
        self.closing = False

    def reserve(self, stage):
        r = stage.requirements
        if r.cpu_tokens>self.cpu_capacity or r.memory_bytes>self.memory_capacity or r.cache_bytes>self.cache_capacity:
            raise ValueError('stage resource request exceeds total capacity')
        if r.devices and not any(d.logical in r.devices and d.capacity>=r.device_bytes for d in self.devices):
            raise ValueError('no compatible NPU has required memory capacity')
        if len(self.leases) >= self.budget.cpu_workers:
            return None
        if sum(x['cpu_tokens'] for x in self.leases.values())+r.cpu_tokens > self.cpu_capacity:
            return None
        if sum(x['memory_bytes'] for x in self.leases.values())+r.memory_bytes > self.memory_capacity:
            return None
        if sum(x['cache_bytes'] for x in self.leases.values())+r.cache_bytes > self.cache_capacity:
            return None
        busy = {x['device_uuid'] for x in self.leases.values()}
        device = next((d for d in self.devices if d.logical in r.devices and d.uuid not in busy and d.capacity >= r.device_bytes), None)
        if r.devices and device is None:
            return None
        lease = {'cpu_tokens': r.cpu_tokens, 'memory_bytes': r.memory_bytes, 'cache_bytes': r.cache_bytes,
                 'device_uuid': device.uuid if device else None, 'device': device.logical if device else 'cpu',
                 'device_bytes': r.device_bytes, 'acquired': time.monotonic_ns()}
        self.leases[stage.identity] = lease
        return lease

    def _retire(self, worker):
        from .supervisor import _terminate_group
        try:
            worker['socket'].close()
            _terminate_group(worker['child'])
            return worker['tail'].finish()
        finally:
            worker['scratch'].cleanup()

    def _worker(self, key):
        for worker in self.workers:
            if not worker['busy'] and worker['key']==key and worker['count']<self.max_tasks and worker['child'].poll() is None:
                return worker
        # Evict idle incompatible workers rather than growing a pool per backend.
        for worker in list(self.workers) if len(self.workers)>=self.budget.cpu_workers else []:
            if not worker['busy']:
                self._retire(worker)
                self.workers.remove(worker)
                break
        scratch = tempfile.TemporaryDirectory(prefix='runtime-', dir=self.directory)
        a, b = socket.socketpair()
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   TORCHINDUCTOR_COMPILE_THREADS=str(self.budget.compile_threads),
                   TMPDIR=scratch.name, TORCHINDUCTOR_CACHE_DIR=scratch.name+'/inductor', TRITON_CACHE_DIR=scratch.name+'/triton')
        child = subprocess.Popen([sys.executable,'-m','autofusedebugger.fuzz.scheduling.stage_worker',str(b.fileno())],
            pass_fds=(self.fd,b.fileno()),start_new_session=True,stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
        b.close()
        from ..core_logs import ProcessTail
        worker = {'key':key,'child':child,'socket':a,'scratch':scratch,'tail':ProcessTail(child.stdout),
                  'busy':False,'count':0,'identity':str(uuid.uuid4()),'rss_floor':None}
        self.workers.append(worker)
        return worker

    def submit(self, stage, job, timeout):
        # Runs in an I/O thread: startup, socket IO, termination, log cleanup
        # cannot block the supervisor's timeout/cancellation loop.
        lease = self.leases[stage.identity]
        key = (job.get('config',{}).get('backend','cpu') if job['kind']=='first_call_combined' else 'cpu',
               lease['device'], job.get('config',{}).get('compile_threads', self.budget.compile_threads))
        return self.executor.submit(self._run, stage.identity, key, job, timeout)

    def _run(self, identity, key, job, timeout):
        # Only the coordinator mutates admission. Worker allocation is serialized
        # separately from IO so CPU and NPU branches may execute concurrently.
        import threading
        if not hasattr(self,'lock'):
            raise RuntimeError('pool lock not initialized')
        worker = None
        try:
            with self.lock:
                if self.closing or identity in self.cancelled:
                    raise RuntimeError('stage cancelled before worker dispatch')
                worker = self._worker(key)
                worker['busy'] = True
                worker['stage'] = identity
            if self.closing or identity in self.cancelled:
                raise RuntimeError('stage cancelled during worker startup')
            worker['socket'].settimeout(timeout)
            send(worker['socket'],job)
            response = receive(worker['socket'])
            response['worker_id'] = worker['identity']
            worker['count'] += 1
            from .budgets import group_rss, scratch_size
            rss = group_rss(worker['child'].pid)
            response['rss_bytes'] = rss
            worker['rss_floor'] = worker['rss_floor'] or rss
            device_resident = response.get('value',{}).get('device_memory_bytes') or 0
            # Only a successfully cleaned adapter may opt into runtime reuse.
            # Missing capability reports retain the conservative isolation policy.
            isolated_session = (job['kind']=='first_call_combined' and
                                not response.get('value',{}).get('persistent_runtime', False))
            case_skipped = response.get('value', {}).get('case_skipped', False)
            recycle = (isolated_session or (not response['ok'] and job['kind'] != 'generate') or
                       (not case_skipped and response.get('value',{}).get('result',{}).get('status') in {'error','unsupported'}) or worker['count']>=self.max_tasks or
                       rss > worker['rss_floor']+256*1024**2 or device_resident > 16*1024**2 or
                       scratch_size(worker['scratch'].name) > self.cache_capacity//max(1,self.budget.cpu_workers))
            with self.lock:
                if recycle:
                    self._retire(worker); self.workers.remove(worker)
                else:
                    worker['busy'] = False
            return response
        except BaseException as exc:
            log = ''
            if worker is not None:
                with self.lock:
                    if worker in self.workers:
                        log = self._retire(worker); self.workers.remove(worker)
            if job['kind']=='record' and job.get('save'):
                from .supervisor import _has_result
                if _has_result(Path(job['database']),job['task']):
                    return {'ok':True,'value':{'saved':True,'status':job['result']['status']},'finished':time.monotonic_ns()}
            if log and isinstance(exc, Exception):
                from ..failure_details import redact
                message = redact(f'{type(exc).__name__}: {exc}')[:300]
                raise RuntimeError(message + '\nWorker log:\n' + log[:1200]) from None
            raise

    def cancel(self, identity):
        # SIGKILL the entire group; I/O thread reaps it and publishes completion.
        import signal
        self.cancelled.add(identity)
        for worker in list(self.workers):
            if worker.get('stage')==identity and worker['busy']:
                try: os.killpg(worker['child'].pid,signal.SIGKILL)
                except ProcessLookupError: pass

    def release(self, identity):
        lease = self.leases.pop(identity)
        self.cancelled.discard(identity)
        return dict(lease,released=time.monotonic_ns())

    def close(self):
        self.closing = True
        for identity in list(self.leases): self.cancel(identity)
        self.executor.shutdown(wait=True)
        for worker in list(self.workers): self._retire(worker)
        self.workers.clear()
