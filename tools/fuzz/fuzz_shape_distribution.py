"""Reproducible axis-size audit; never assumes solver solutions are uniform."""
import argparse
from collections import Counter, defaultdict
import json
import random
from pathlib import Path
from autofusex.fuzz.client.contracts import Seed
from autofusex.fuzz.client.generation import plan


def audit(count, versions=(11,12), config=None):
    from autofusex.fuzz.client.generation.sampling import SamplingConfig
    config = config or SamplingConfig()
    rng=random.Random(20261110)
    hist=defaultdict(Counter); accepted=Counter(); errors=Counter(); records=[]
    for _ in range(count):
        entropy=rng.randrange(36**9)
        for version in versions:
            seed=config.seed(entropy % 2**32, dynamic=version==12)
            try:spec=plan(seed)
            except Exception as exc:
                errors[f'{version}:{type(exc).__name__}']+=1
                continue
            transpose=any(n['op']=='transpose' for n in json.loads(spec.graph_json)['nodes'])
            accepted[f'{version}:all']+=1
            if not transpose:accepted[f'{version}:no_transpose']+=1
            for sample,inputs in enumerate(spec.shape_samples):
                for position,t in enumerate(inputs):
                    records.append({'seed':str(seed),'version':version,'sample':sample,
                                    'input':position,'shape':t.shape,'transpose':transpose,'nodes':len(json.loads(spec.graph_json)['nodes']),
                                    'dtype':t.dtype,'elements':__import__('math').prod(t.shape)})
                    for axis,size in enumerate(t.shape):
                        for population in (['all'] if transpose else ['all','no_transpose']):
                            hist[f'v{version}/{population}/S{sample}/input{position}/rank{len(t.shape)}/axis{axis}'][size]+=1
                            hist[f'v{version}/{population}/S{sample}/all_axes'][size]+=1
    return dict(pairs=count,accepted=dict(accepted),errors=dict(errors),
                histograms={k:dict(sorted(v.items())) for k,v in sorted(hist.items())},records=records)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--count',type=int,default=256)
    parser.add_argument('--output',required=True)
    parser.add_argument('--versions', nargs='+', type=int, default=[11,12], choices=[11,12])
    args=parser.parse_args()
    if not 1<=args.count<=10000:parser.error('count must be 1..10000')
    result=audit(args.count, tuple(args.versions));Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='records' and k!='histograms'}))
    print(json.dumps({k:v for k,v in result['histograms'].items() if k.endswith('/all_axes')},sort_keys=True))
