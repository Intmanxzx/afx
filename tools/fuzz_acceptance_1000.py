"""Remote acceptance runner: frozen ETR v2 seeds, unified execution, light results.

No generating-only count: each seed executes all three samples on each selected
backend. This diagnostic runner is not a replacement managed execution engine.
"""
import argparse
import gc
import json
from pathlib import Path
import time

from autofusedebugger.fuzz.contracts import ExecutionConfig, EvaluationConfig
from autofusedebugger.fuzz.generation import materialize, workload_id
from autofusedebugger.fuzz.generation.categories import categories
from autofusedebugger.fuzz.backends import create_backend
from autofusedebugger.fuzz.execution import execute
from autofusedebugger.fuzz.storage import ResultRepository


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('database');parser.add_argument('--count',type=int,default=1000)
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--backends',nargs='+',default=['cpu-eager','ascendc'])
    parser.add_argument('--device',default='npu:0')
    args=parser.parse_args()
    repo=ResultRepository(args.database)
    counts={}; started=time.monotonic()
    for index in range(args.start,args.start+args.count):
        seed=str(2**128+index); bundle=materialize(seed)
        for name in args.backends:
            config=ExecutionConfig(name,args.device if name in {'ascendc','triton'} else 'cpu',1)
            result=execute(bundle,create_backend(config),EvaluationConfig())
            identity=repo.save(result,environment={'machine':__import__('platform').machine(),'python':__import__('platform').python_version(),'package_version':'0.1.2'})
            counts[name+'/'+result['status']]=counts.get(name+'/'+result['status'],0)+1
            print(json.dumps({'index':index,'seed':seed,'backend':name,'status':result['status'],
                              'shapes':len(result['shape_results']),'categories':categories(bundle.spec),
                              'attempt_id':identity,'workload_id':workload_id(bundle),
                              'elapsed':round(time.monotonic()-started,2)}),flush=True)
            import torch
            torch._dynamo.reset()
            gc.collect()
    print(json.dumps({'completed_seeds':args.count,'counts':counts,'elapsed':time.monotonic()-started}),flush=True)


if __name__=='__main__':main()
