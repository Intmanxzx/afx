"""Standalone result synchronization CLI."""
import argparse


def parser(*, retest=False):
    client = argparse.ArgumentParser(prog="afx retest" if retest else "afx sync", description="Synchronize local execution results")
    client.add_argument("action", choices=("run", "once") if retest else ("setup", "run", "once", "status"))
    client.add_argument("--endpoint", required=True, help="platform URL, e.g. https://autofuse.online")
    client.add_argument("--token-file", required=True, help="file holding the client token (chmod 600)")
    client.add_argument("--results", required=True, help="local results.sqlite written by afx fuzz --save")
    client.add_argument("--state", help="uploader state file (default: next to --results)")
    client.add_argument("--device", action="append", help="NPU for retest tasks, repeatable (npu:0)")
    client.set_defaults(retest=retest)
    client.add_argument("--interval", type=float, default=30)

    return client


def main(argv=None, *, retest=False):
    from autofusex.client.platform_transport import main as run
    args = parser(retest=retest).parse_args(argv)
    if retest and args.action == "once":
        args.action = "retest-once"
    return run(args)
