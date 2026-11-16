"""Client execution plane facade.

Generation, backend execution, scheduling, local result persistence and sync
are client concerns; the daemon controls this plane only through protocols.
"""
from .generation import materialize, materialize_spec

__all__ = ['materialize', 'materialize_spec']
