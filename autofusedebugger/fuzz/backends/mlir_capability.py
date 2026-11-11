"""Fail closed unless the loaded codegen library contains the MLIR bridge.

The adapter accepts only the frozen elementwise/reduce/transpose graph. It cannot
contain Cube, the one strict-mode path allowed to return kNotSupported. Strict
--mlir_codegen_config=mlir rejects all other unsupported/failed lowering paths.
This is a Linux diagnostic gate, not a general backend capability API.
"""
from pathlib import Path
import subprocess
from .base import UnsupportedBackend


def verify_runtime():
    try:
        import autofuse.pyautofuse  # noqa: F401; resolve the actual runtime first
        libraries = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                     if 'libaihac_codegen.so' in line}
        if len(libraries) != 1:
            raise ValueError('expected exactly one loaded codegen library')
        library = libraries.pop()
        symbols = subprocess.run(['nm', '-D', '--defined-only', library],
                                 capture_output=True, text=True, timeout=15, check=True).stdout
        if 'RunMlirCodegen' not in symbols:
            raise ValueError('loaded codegen was built without the MLIR bridge')
        return library
    except Exception as exc:
        raise UnsupportedBackend('af-mlir runtime verification failed: ' + str(exc)) from exc
