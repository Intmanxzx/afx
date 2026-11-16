"""Thread-scoped cancellation token for owned installer/build subprocesses."""
import threading

_context=threading.local()


def bind(check):
    _context.check=check


def requested():
    return bool(getattr(_context,'check',lambda:False)())
