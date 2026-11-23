"""Managed-run result upload is retired along with the old central sync protocol.

This used to replay each managed run's local result/fact outbox to the old
central server's /api/upload and /api/facts routes. That server (and its
outbox/ACK protocol) no longer exists; platform v3 clients upload independently
via `afx platform client run` instead of through this per-run channel. Kept as
a no-op so `daemon.runtime.Manager.synchronize()` can still call it safely.
"""


def upload_runs(store, credential):
    pass
