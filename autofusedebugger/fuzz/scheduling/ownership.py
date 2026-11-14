"""POSIX queue ownership; inherited by workers so parent death cannot steal work.

This is a queue supervisor lock, NOT a device lease. No PID/time-based stealing.
"""
from contextlib import contextmanager
import os


@contextmanager
def own_queue(ledger):
    if os.name != 'posix':
        raise ValueError('managed execution currently requires POSIX; offline view remains portable')
    import fcntl
    ledger.page(limit=1)  # Initialize/validate before opening the stable DB inode.
    fd = os.open(ledger.path, os.O_RDONLY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('queue still owned by a supervisor or worker; refusing takeover') from None
        # Do not explicitly LOCK_UN: inherited descriptors must retain ownership
        # after supervisor death until all surviving managed workers have exited.
        yield fd
    finally:
        os.close(fd)
