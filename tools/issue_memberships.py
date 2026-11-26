"""Backfill missing current issue links offline; default is rollback-only preview."""
import argparse
import json

from autofusex.server.issue_memberships import backfill
from autofusex.server.store import Store


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_dir')
    parser.add_argument('--offline', action='store_true', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(backfill(Store(args.data_dir), apply=args.apply,
        progress=lambda n, total, counts: print(json.dumps(dict(scanned=n,total=total,counts=counts)), flush=True)),
        ensure_ascii=False))
