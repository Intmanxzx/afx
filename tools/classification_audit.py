"""Read-only classification upgrade preview. No Store initialization/migrations."""
import argparse
import collections
import json
import sqlite3
import zlib
from pathlib import Path
from autofusex.server.failure_attribution import classify, VERSION

def audit(directory):
    base=Path(directory).resolve()
    db=sqlite3.connect(f'file:{base / "core.db"}?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    content=sqlite3.connect(f'file:{base / "content.db"}?mode=ro',uri=True)
    def read(identity):
        row=content.execute('SELECT codec,data FROM contents WHERE content_id=?',(identity,)).fetchone() if identity else None
        return (zlib.decompress(row[1]) if row[0]=='zlib' else row[1]) if row else None
    transitions=collections.Counter(); changed=0; total=0; examples=[]
    for r in db.execute('SELECT * FROM executions'):
        total+=1
        old=(r['function_status'],r['fail_stage'])
        if r['function_status']=='pass': new=old
        else:
            new=classify(backend=r['backend'],declared_status=r['function_status'],declared_stage=r['fail_stage'],
                         summary=r['log_summary'],log=read(r['log_content_id']),detail=read(r['detail_content_id']))[:2]
        transitions[(r['backend'],*old,*new)]+=1
        if old!=new:
            changed+=1
            if len(examples)<20:examples.append({'exec_id':r['exec_id'],'old':old,'new':new})
    db.close();content.close()
    return {'classifier_version':VERSION,'executions':total,'changed':changed,'transitions':[{'key':k,'count':v} for k,v in transitions.items()],'examples':examples}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('data_dir');args=parser.parse_args()
    print(json.dumps(audit(args.data_dir),ensure_ascii=False,indent=2))
