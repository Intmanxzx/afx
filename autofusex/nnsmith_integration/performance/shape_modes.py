"""Resolve a seed-stable shape mode before case generation or backend dispatch."""
from dataclasses import replace


def resolve_generation(config, seed):
    if config.mixed_shapes:
        # Stable under worker count, completion order, and single-seed replay.
        return replace(config, dynamic_shapes=bool(seed & 1), mixed_shapes=False)
    return config
