"""Stable control plane for the local Fuzz client.

This package deliberately exposes only lifecycle, IPC, watchdog and update
facades.  Heavy execution dependencies remain in :mod:`autofusex.fuzz.client.client`.
"""
def request(*args, **kwargs):
    from .runtime import request as _request
    return _request(*args, **kwargs)

def supervise(*args, **kwargs):
    from .supervisor import run
    return run(*args, **kwargs)

def remote(*args, **kwargs):
    from .runtime import remote as _remote
    return _remote(*args, **kwargs)

def __getattr__(name):
    if name == 'Manager':
        from .runtime import Manager
        return Manager
    raise AttributeError(name)

__all__ = ['Manager', 'request', 'supervise', 'remote']
