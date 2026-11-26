"""Rebuild issues on an offline data directory; default is a rollback-only preview.

Back up all databases consistently before --apply. Store may migrate schema.
"""
import argparse
import json
from autofusex.server.store import Store
from autofusex.server.issue_rebuild import rebuild

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_dir')
    parser.add_argument('--offline', action='store_true', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(rebuild(Store(args.data_dir), apply=args.apply), ensure_ascii=False))
