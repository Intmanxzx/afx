"""Remove primary-category partitioning without resetting issue repair history.

Stop server writes and back up all databases before --apply.
"""
import argparse
import json

from autofusex.server.issue_signatures import migrate
from autofusex.server.store import Store


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_dir')
    parser.add_argument('--offline', action='store_true', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(migrate(Store(args.data_dir), apply=args.apply), ensure_ascii=False))
