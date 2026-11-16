from autofusex.fuzz.generation.sampling import SamplingConfig
import json,sys,time,sqlite3
from pathlib import Path
from autofusex.fuzz.contracts import ExecutionConfig,EvaluationConfig
from autofusex.fuzz.scheduling.ledger import TaskLedger
from autofusex.fuzz.scheduling.budgets import Budget,write_policy
from autofusex.fuzz.scheduling.supervisor import drain
from autofusex.fuzz.storage.queries import ResultQueries
root=Path(sys.argv[1]);root.mkdir(exist_ok=False)
mode=sys.argv[2];workers=int(sys.argv[3]);reuse=int(sys.argv[4]);backend=sys.argv[5]
l=TaskLedger(root/'tasks');write_policy(l,Budget(cpu_workers=workers,compile_threads=1,worker_max_tasks=reuse))
for n,seed in enumerate([SamplingConfig.profile('small',max_nodes=3,dtype_mode='float32').seed(42).short,SamplingConfig.profile('small',max_nodes=3,dtype_mode='float32').seed(42,True).short,SamplingConfig.profile('small',max_nodes=3,dtype_mode='float32').seed(42).short,SamplingConfig.profile('small',max_nodes=3,dtype_mode='float32').seed(42,True).short]):
 for b,d in [('cpu','cpu'),(backend,'npu:'+str(n%2))]:
  l.enqueue(seed,ExecutionConfig(b,d,1,scheduling=mode,static_only=seed.startswith('C'),batch_id='bench-'+str(n)),EvaluationConfig())
a=time.monotonic();summary=drain(l,root/'results',timeout=120);duration=time.monotonic()-a
q=ResultQueries(root/'results');rows=[q.detail(r['id']) for r in q.page()['items']]
report={'scheduling':mode,'workers':workers,'worker_max_tasks':reuse,'backend':backend,'wall_seconds':duration,'summary':summary,'attempts':[{'backend':r['backend'],'seed':r['seed_short'],'status':r['status'],'shapes':len(r['shape_results'])} for r in rows]}
try:
 from autofusex.fuzz.reporting.statistics import statistics
 report['windows']=statistics(root/'results')['tool_efficiency']['windows']
except KeyError:pass
(root/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
assert summary['completed']==8
if backend=='eager-npu':assert all(r['status']=='ok' for r in rows)
