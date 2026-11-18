"""Fuzz result reporting/view server package; no client execution runtime is imported here."""
def serve(*args, **kwargs):
    from .view.server import serve as _serve
    return _serve(*args, **kwargs)

__all__ = ['serve']
