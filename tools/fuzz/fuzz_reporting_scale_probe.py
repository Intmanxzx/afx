"""Synthetic SQL query capacity only: does not measure ingestion or NPU execution."""
import json
from pathlib import Path
import sqlite3
import sys
import time
import uuid
from autofusex.fuzz.client.storage.repository import _initialize
from autofusex.fuzz.client.generation.sampling import SamplingConfig
sampling=SamplingConfig()
from autofusex.fuzz.client.storage.queries import ResultQueries
from autofusex.fuzz.server.reporting.statistics import statistics

rows=int(sys.argv[2]) if len(sys.argv)>2 else 100000
path=Path(sys.argv[1])
with path.open('xb'):pass
with sqlite3.connect(path) as conn:
    _initialize(conn)
    for kind in ('environment','execution','evaluation'):
        conn.execute('INSERT INTO contexts VALUES (?,?,?)',(kind,kind,'{}'))
    for i in range(rows):
        identity=str(uuid.UUID(int=i+1));work=f'{i//2:064x}';group=work
        conn.execute('INSERT OR IGNORE INTO case_facts VALUES (?,?,?,?,?,?,?)',(work,str(sampling.seed(i//2,True)),'{}',2,64,4096,1))
        conn.execute('INSERT OR IGNORE INTO case_dimensions VALUES (?,?,?)',(work,'category','Elementwise'))
        conn.execute('INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(identity,str(sampling.seed(i//2,True)),work,'ref/1','cpu' if i%2 else 'af-mlir','ok','2026-11-09T00:00:00Z','execution','evaluation','environment',1000,'','','b'*64))
        conn.execute('INSERT INTO attempt_facts VALUES (?,?,?,?,?,?,?,?,?,?,?)',(identity,group,'source','device','owner','batch','{}','{}','{}',1000,'partial_recorded_only'))
        conn.execute('INSERT INTO stage_metrics VALUES (?,?,?,?)',(identity,-1,'prepare',1000))
        conn.execute('INSERT INTO shapes VALUES (?,?,?,?,?,?,?,?,?,?)',(identity,0,'pass','not_requested',1000,None,None,None,'',''))
from autofusex.fuzz.server.reporting.materialized import rebuild
start=time.perf_counter(); rebuild(path)
q=ResultQueries(path);results={'rows':rows,'build_seconds':time.perf_counter()-start,'bytes':path.stat().st_size}
for name,call in [('cases',q.case_page),('statistics',lambda:statistics(path)),('seed',lambda:q.case_page(seed=str(sampling.seed(42,True))))]:
    start=time.perf_counter();value=call();elapsed=time.perf_counter()-start
    results[name]={'seconds':elapsed,'response_bytes':len(json.dumps(value))}
    print(json.dumps({name:results[name]}),flush=True)
    assert elapsed<3
print(json.dumps(results,indent=2))
