"""Isolated real-NPU retest API probe; never use production credentials/homes.

Run in the configured mlir runtime: python tools/fuzz/fuzz_retest_e2e.py NEW_DIRECTORY
Creates private users/database, replays D16, then validates cross-user retests on
both logical devices using the same scheduler. It does not stop existing work.
"""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.request import build_opener, ProxyHandler

from autofusex.fuzz.central.server import create_server
from autofusex.fuzz.central.repository import create_user
from autofusex.fuzz.central import retests
from autofusex.fuzz.config import setup
from autofusex.fuzz.storage.repository import _connect
from autofusex.fuzz.storage.queries import ResultQueries
from autofusex.fuzz.sync.client import once
from autofusex.fuzz.sync.retests import Consumer
from autofusex.fuzz.sync.outbox import pending


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory');parser.add_argument('--structured',action='store_true');args=parser.parse_args()
    root=Path(args.directory).resolve();root.mkdir(mode=0o700,parents=True,exist_ok=False)
    database=root/'central.sqlite'
    users=[create_user(database,name) for name in ('Original','Retester')]
    if args.structured:
        from autofusex.fuzz.reporting.materialized import migrate
        migrate(database)
    server=create_server(database,0,cache_interval=.5)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    endpoint=f'http://127.0.0.1:{server.server_port}'
    opener=build_opener(ProxyHandler({})).open
    def home(name,user):
        path=root/name;path.mkdir(mode=0o700)
        key=path/'upload.token';key.write_text(user['token']);key.chmod(0o600)
        setup(path,display_name=user['display_name'],user_id=user['user_id'],key_file=key,
              cpu_workers=1,compile_threads=1,resource_policy='manual')
        # Loopback HTTP only in this fixture, never configure production insecurely.
        (path/'sync.json').write_text(json.dumps(dict(endpoint=endpoint,key_file=str(key),interval=1)))
        return path,path/'results/results.sqlite'
    report=[]
    try:
        original,archive=home('original',users[0])
        # Competing work keeps its physical lease; retry only our own probe.
        for trial in range(6):
            archive=original/f'baseline-{trial}.sqlite'
            with (root/f'baseline-{trial}.log').open('w') as log:
                env=dict(os.environ,AUTOFUSEX_HOME=str(original))
                completed=subprocess.run([sys.executable,'-m','autofusex.fuzz','replay','D00000000G',
                    '--backend','af-mlir','--device',f'npu:{trial%2}','--save','--database',str(archive),'--compile-threads','1'],
                    env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
            if not completed.returncode:break
            with closing(_connect(archive)) as conn:
                error=conn.execute('SELECT error_summary FROM attempts ORDER BY rowid DESC LIMIT 1').fetchone()
            if not error or 'device busy' not in error[0]:
                break  # An actual compiler failure is valid replay evidence.
            time.sleep(3)
        else:
            raise RuntimeError('baseline device remained busy; no existing task was stopped')
        once(original,archive,opener=opener)
        original_attempt=next(r['id'] for r in ResultQueries(archive).page()['items'] if r['backend']=='af-mlir')
        original_detail=ResultQueries(archive).detail(original_attempt)
        indices=[s['shape_index'] for s in original_detail['shape_results']] or [-1]
        for device in ('npu:0','npu:1'):
            worker,results=home('worker-'+device.replace(':','-'),users[1])
            consumer=Consumer(worker,results,opener=opener,devices=[device])
            consumer.tick()  # Default registration before any local result exists.
            client=pending(results)['source_id']
            for retry in range(6):
                answer=retests.create(database,users[1]['user_id'],dict(client_id=client,
                    request_key=str(__import__('uuid').uuid4()),repetitions=1,
                    targets=[dict(attempt_id=original_attempt,shape_index=i) for i in indices]))
                assert answer['task_count']==1
                deadline=time.monotonic()+240
                while time.monotonic()<deadline:
                    consumer.tick()
                    once(worker,results,opener=opener)
                    details=retests.detail(database,users[1]['user_id'],answer['batch_id'])
                    state=details['items'][0]['state']
                    if state in {'completed','interrupted','cancelled'}:break
                    time.sleep(2)
                if state=='completed':break
                failed=details['items'][0]['attempt_id']
                if not failed or 'device busy' not in ResultQueries(database).detail(failed)['error_summary']:
                    raise RuntimeError(f'{device} failed for non-contention reason: {state}')
                time.sleep(3)
            else:
                raise RuntimeError(f'{device} remained busy; existing tasks were preserved')
            attempt=details['items'][0]['attempt_id']
            detail=ResultQueries(database).detail(attempt)
            assert detail['workload_id']==original_detail['workload_id']
            assert detail['backend']=='af-mlir'
            assert [s['shape_index'] for s in detail['shape_results']]==[s['shape_index'] for s in original_detail['shape_results']]
            assert detail['contexts']['execution']['device']==device
            rows=ResultQueries(database).case_page()['items']
            assert all(r['backends']['af-mlir'][0]['id']==attempt for r in rows)
            assert all(r['history_count']>=2 for r in rows)
            assert all('cpu' in r['backends'] for r in rows if r['shape_index']>=0)
            report.append(dict(device=device,shape_count=len(detail['shape_results']),results=[s['correctness'] for s in detail['shape_results']],status=state,
                               workload_id=detail['workload_id'],task_count=answer['task_count']))
        with closing(_connect(database)) as conn:
            assert conn.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        (root/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
    finally:
        server.shutdown();server.server_close();thread.join()


if __name__=='__main__':main()
