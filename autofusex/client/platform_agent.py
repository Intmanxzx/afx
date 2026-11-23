"""One-shot platform agent executed from the active client release.

The bootstrap supervisor launches this module in the selected client Python so
heartbeat, inventory, configuration and future platform facts are controlled
by the replaceable client release rather than by the supervisor process.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from autofusex.client.store import Store


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="afx-platform-agent")
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args(argv)
    home = Path(args.data_dir).expanduser().resolve()
    client_path = home / "client.json"
    if not client_path.exists():
        return 0
    client = json.loads(client_path.read_text(encoding="utf-8"))
    if client.get("mode") != "managed":
        return 0
    # Compatibility entry point only: never pretend that a new one-shot
    # helper has upgraded an old resident manager.
    from autofusex.client.daemon import request
    request(home, {'action': 'status'})
    store = Store(home)
    if store.get('supervisor_protocol') != 2:
        raise RuntimeError('explicit control migration required')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
