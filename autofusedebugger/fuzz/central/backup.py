"""Private whole-server consistent backup, including auth hashes and ACK state.

Unlike public result snapshots, this file is sensitive and must remain private.
"""
from contextlib import closing
import os
from pathlib import Path
import sqlite3
import tempfile
from ..storage.repository import _connect


def backup(source,destination):
    destination=Path(destination).expanduser().resolve()
    if destination.exists():raise ValueError('backup destination must be new')
    destination.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.central-backup-',dir=destination.parent);os.close(fd)
    try:
        with closing(_connect(source)) as src,closing(sqlite3.connect(name)) as dst:
            src.backup(dst)
            if dst.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('backup integrity failure')
            for table in ('users','sources','receipts','owners'):
                dst.execute(f'SELECT 1 FROM {table} LIMIT 1')
        with open(name,'rb') as stream:os.fsync(stream.fileno())
        os.link(name,destination)
        return {'backup':str(destination),'sensitive':True}
    finally:
        Path(name).unlink(missing_ok=True)
