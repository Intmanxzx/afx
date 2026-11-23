"""AFX client control services; execution dependencies are loaded by workers."""
from autofusex.contracts.ipc import request


def supervise(*args, **kwargs):
    from autofusex.client.supervisor import run
    return run(*args, **kwargs)


def remote(*args, **kwargs):
    from autofusex.client.daemon import remote as call
    return call(*args, **kwargs)


def __getattr__(name):
    if name == 'Manager':
        from autofusex.client.daemon import Manager
        return Manager
    raise AttributeError(name)


__all__ = ['Manager', 'request', 'supervise', 'remote']
