"""Bounded scalar diagnostics; never persist tensors or unbounded compiler logs."""
import json
import math
import re
import traceback

FIELDS={'cpu_pass','secondary_used','secondary_pass','max_abs','mean_abs','max_rel','mismatches','elements','nonfinite','error','basis'}


def redact(text):
    return re.sub(r'(?i)(bearer\s+|(?:token|secret|password|api_key)\s*[=:]\s*)[^\s,;]+',r'\1[REDACTED]',text)


def excerpt(exc):
    return redact(''.join(traceback.format_exception(type(exc),exc,exc.__traceback__)))[-1600:]


def validate(raw):
    if not isinstance(raw,str) or len(raw)>8000:raise ValueError('diagnostics size exceeded')
    data=json.loads(raw)
    if not isinstance(data,dict) or set(data)-{'version','error','shapes'} or data.get('version')!=1:raise ValueError('invalid diagnostics')
    if 'error' in data and (not isinstance(data['error'],str) or len(data['error'])>1600):raise ValueError('invalid core log')
    shapes=data.get('shapes',{})
    if not isinstance(shapes,dict) or len(shapes)>16:raise ValueError('diagnostic shape limit')
    for index,value in shapes.items():
        if not index.isdecimal() or not isinstance(value,dict) or set(value)-FIELDS:raise ValueError('invalid diagnostic fields')
        for key,v in value.items():
            if key in {'error','basis'}:
                if not isinstance(v,str) or len(v)>1600:raise ValueError('invalid diagnostic text')
            elif not isinstance(v,(int,float,bool)) or not math.isfinite(v):raise ValueError('invalid diagnostic metric')
    return json.dumps(data,sort_keys=True,separators=(',',':'))
