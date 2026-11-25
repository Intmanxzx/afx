"""Preview/export or apply historical unknown-module retest markers offline."""
import argparse
import csv
import json
from pathlib import Path

from autofusex.server.historical_retest import plan, apply
from autofusex.server.reclassification import rollback
from autofusex.server.store import Store


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('data_dir')
    p.add_argument('--batch', required=True)
    p.add_argument('--offline', action='store_true', help='Server writes are stopped, or data is an isolated snapshot')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--preview', metavar='MANIFEST', help='Write manifest and current-head replay CSV')
    group.add_argument('--apply', metavar='MANIFEST', help='Apply exactly a reviewed manifest')
    group.add_argument('--rollback', action='store_true')
    a = p.parse_args()
    if not a.offline:
        p.error('--offline required: Store can migrate schema')
    if not (a.preview or a.apply or a.rollback):
        p.error('choose --preview, --apply or --rollback')
    store = Store(a.data_dir)
    if a.rollback:
        print(json.dumps({'rolled_back': rollback(store, a.batch)}))
    elif a.apply:
        manifest = json.loads(Path(a.apply).read_text())
        if manifest['batch'] != a.batch:
            p.error('manifest batch mismatch')
        print(json.dumps({'marked': apply(store, a.batch, manifest['changes'])}))
    else:
        changes = plan(store.reader())
        path = Path(a.preview)
        path.write_text(json.dumps({'batch': a.batch, 'changes': changes}, ensure_ascii=False, indent=2))
        fields = ['case_key', 'backend', 'npu_arch', 'exec_uuid', 'reason']
        heads = [c for c in changes if c['is_head']]
        with path.with_suffix('.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(heads)
        print(json.dumps({'records': len(changes), 'current_heads': len(heads), 'manifest': str(path)}))


if __name__ == '__main__':
    main()
