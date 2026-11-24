"""Use the invoking installation's AFX with registered runtime dependencies."""
import os
from pathlib import Path


def execution_env(env):
    result = dict(env)
    root = str(Path(__file__).resolve().parents[2])
    result['PYTHONPATH'] = os.pathsep.join([root, *filter(None, result.get('PYTHONPATH', '').split(os.pathsep))])
    return result
