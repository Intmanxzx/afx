"""Fixed cold-worker C16/D16 probe; run in an isolated CPU/NPU environment.

Usage: python tools/fuzz/fuzz_stage_baseline_probe.py NEW_OUTPUT_DIR WORKERS BACKEND
Never uses production homes or uploader credentials. Run old/new source from
separate directories with identical runtime paths; each invocation starts fresh
workers, not a warm-session benchmark. Wall spans are not kernel measurements.
"""
import json,sys,time
from pathlib import Path
from autofusex.fuzz.contracts import ExecutionConfig,EvaluationConfig
from autofusex.fuzz.scheduling.ledger import TaskLedger
from autofusex.fuzz.scheduling.budgets import Budget,write_policy
from autofusex.fuzz.scheduling import supervisor
from autofusex.fuzz.storage.queries import ResultQueries
root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False)
workers=int(sys.argv[2]);backend=sys.argv[3]
ledger=TaskLedger(root/'tasks');db=root/'results'
for seed,device in [('C16','npu:0'),('D16','npu:1'),('C16','npu:0'),('D16','npu:1')]:
 ledger.enqueue(seed,ExecutionConfig(backend,device,1,static_only=seed.startswith('C')),EvaluationConfig())
write_policy(ledger,Budget(cpu_workers=workers,compile_threads=1))
intervals=[];starts={}
original=supervisor.subprocess.Popen;terminate=supervisor._terminate_group
def launch(*a,**kw):
 p=original(*a,**kw);starts[p.pid]=time.monotonic();return p
def finish(p):
 terminate(p);intervals.append([starts[p.pid],time.monotonic()])
supervisor.subprocess.Popen=launch;supervisor._terminate_group=finish
started=time.monotonic();summary=supervisor.drain(ledger,db,timeout=120);elapsed=time.monotonic()-started
q=ResultQueries(db);details=[q.detail(r['id']) for r in q.page()['items']]
report={'workers':workers,'backend':backend,'elapsed_seconds':elapsed,'summary':summary,'intervals':intervals,'attempts':[{'seed':d['seed_short'],'status':d['status'],'shapes':len(d['shape_results']),'device':d['contexts']['execution']['device'],'error':d.get('error_summary','')[:150]} for d in details]}
report['successful_shapes']=sum(len(d['shape_results']) for d in details if d['status']=='ok')
report['completed_shapes']=sum(len(d['shape_results']) for d in details)
(root/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
assert summary['completed']==4 and summary['interrupted']==0
if backend=='eager-npu':assert report['successful_shapes']==8
