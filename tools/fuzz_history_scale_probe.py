"""Disposable 2,000-round query benchmark (no runtime/production writes)."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from test_fuzz_global_rounds import save
from autofusedebugger.fuzz.storage.repository import ResultRepository,_connect
from autofusedebugger.fuzz.storage.queries import ResultQueries
from autofusedebugger.fuzz.reporting.statistics import statistics

path=Path(sys.argv[1]);assert not path.exists()
repo=ResultRepository(path)
base=save(repo,'batch','alice','af-mlir',('pass',))
with closing(_connect(path,writable=True)) as conn,conn:
    # Single-backend explicit matrix for synthetic fixture, 2,000 repeated rounds.
    ex=json.loads(conn.execute('SELECT execution FROM attempt_facts').fetchone()[0]);ex['expected_backends']='["af-mlir"]'
    conn.execute('UPDATE attempt_facts SET execution=?',(json.dumps(ex),))
    for table,key in [('attempts','id'),('attempt_facts','attempt_id'),('shapes','attempt_id')]:
        columns=[r[1] for r in conn.execute('PRAGMA table_info('+table+')')]
        selection=','.join('?' if c==key else '"'+c+'"' for c in columns)
        conn.executemany('INSERT INTO '+table+' SELECT '+selection+' FROM '+table+' WHERE '+key+'=?',[(str(__import__('uuid').uuid5(__import__('uuid').UUID(base),str(i))),base) for i in range(1999)])
q=ResultQueries(path);report={}
for name,call in [('summary',q.case_summary),('page',q.case_page),('statistics',lambda:statistics(path,_timeout=30))]:
    started=time.monotonic();answer=call();report[name]=round(time.monotonic()-started,3)
row=q.case_page()['items'][0];assert row['history_count']==2000
seen=set();after=row['cursor']
started=time.monotonic()
while True:
    page=q.case_page(seed=row['seed'],shape=0,after=after,limit=200,_history=True)
    for item in page['items']:
        assert item['record_id'] not in seen;seen.add(item['record_id'])
    if page['next_cursor'] is None:break
    after=page['next_cursor']
assert len(seen)==1999
report.update(rounds=2000,history_rows=len(seen),history_seconds=round(time.monotonic()-started,3))
print(json.dumps(report))
