"""Central server facade; no client execution runtime is imported here."""
def serve(*args, **kwargs):
    from .central.server import serve as _serve
    return _serve(*args, **kwargs)

__all__ = ['serve']
