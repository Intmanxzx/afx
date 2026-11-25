"""Offline classification refresh. Stop server writes and back up before apply."""
import argparse,json
from autofusex.server.store import Store
from autofusex.server.reclassification import refresh,rollback
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('data_dir');p.add_argument('--batch',required=True)
g=p.add_mutually_exclusive_group();g.add_argument('--apply',action='store_true');g.add_argument('--rollback',action='store_true')
p.add_argument('--offline',action='store_true',help='Confirm server writes are stopped or this is an isolated snapshot')
a=p.parse_args()
if not a.offline:p.error('--offline is required: Store may migrate schema even for preview')
store=Store(a.data_dir)
print(json.dumps({'rolled_back':rollback(store,a.batch)} if a.rollback else {'changes':refresh(store,a.batch,apply=a.apply)},ensure_ascii=False))
