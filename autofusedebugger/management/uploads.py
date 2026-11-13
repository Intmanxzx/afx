"""Replayable result/fact upload using per-device scoped credentials."""
import json
from pathlib import Path
from urllib.request import Request,build_opener
from .daemon import NoRedirect
from .config import canonical


def upload_runs(store,credential):
    from ..fuzz.sync import outbox,facts
    with store.connect() as db:
        runs=db.execute('SELECT id,spec FROM runs ORDER BY rowid DESC LIMIT 100').fetchall()
    for run in runs:
        prepared=json.loads(run['spec'])
        if not prepared['spec']['results']['upload']:continue
        path=store.home/'runs'/run['id']/'results.sqlite'
        if not path.exists():continue
        batch=outbox.pending(path,limit=10)
        kind='upload'
        if not batch['items']:
            batch=facts.pending(path);kind='facts'
            if not batch or not batch['items']:continue
        raw=canonical(batch).encode()
        if len(raw)>2*1024*1024:raise ValueError('result upload exceeds bound')
        req=Request(credential['endpoint'].rstrip('/')+'/api/'+kind,raw,headers={
            'Content-Type':'application/json','Authorization':'Bearer '+credential['token']})
        with build_opener(NoRedirect).open(req,timeout=15) as response:
            data=response.read(8193)
        if len(data)>8192:raise ValueError('oversized result acknowledgement')
        ack=json.loads(data)
        if kind=='upload':
            if ack.get('source_id')!=batch['source_id'] or ack.get('ack')!=batch['items'][-1]['revision']:
                raise ValueError('unexpected upload acknowledgement')
            outbox.acknowledge(path,ack['source_id'],ack['ack'])
        else:facts.acknowledge(path,batch,ack)
