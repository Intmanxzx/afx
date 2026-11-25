"""Preview or apply AFIR compiler diagnostic corrections on offline data."""
import argparse
import json
import sqlite3
from pathlib import Path

from autofusex.server.reclassification import mlir_plan, apply_mlir, rollback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--offline', action='store_true')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--preview', type=Path, metavar='MANIFEST')
    action.add_argument('--apply', type=Path, metavar='MANIFEST')
    action.add_argument('--rollback', action='store_true')
    args = parser.parse_args()
    if not args.offline:
        parser.error('--offline required: use stopped-server data or an isolated consistent snapshot')
    if args.preview:
        base = args.data_dir.resolve()
        conn = sqlite3.connect(f'file:{base / "core.db"}?mode=ro', uri=True)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute('ATTACH DATABASE ? AS content', (f'file:{base / "content.db"}?mode=ro',))
            conn.execute('PRAGMA query_only=ON')
            conn.execute('BEGIN')
            changes = mlir_plan(conn)
        finally:
            conn.close()
        args.preview.write_text(json.dumps({'batch': args.batch, 'changes': changes},
                                           ensure_ascii=False, indent=2))
        result = {'records': len(changes), 'current_heads': sum(c['is_head'] for c in changes),
                  'manifest': str(args.preview)}
    else:
        from autofusex.server.store import Store
        store = Store(args.data_dir)
        if args.rollback:
            result = {'rolled_back': rollback(store, args.batch)}
        else:
            manifest = json.loads(args.apply.read_text())
            if manifest['batch'] != args.batch:
                parser.error('manifest batch mismatch')
            result = {'corrected': apply_mlir(store, args.batch, manifest['changes'])}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
