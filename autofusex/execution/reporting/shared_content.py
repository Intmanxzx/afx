"""Intern derived text without duplicating its bytes in a UNIQUE index."""
from hashlib import sha256


def intern_text(conn, table, column, value):
    if (table, column) not in {('display_text_v2', 'content'),
                              ('case_preview_v2', 'payload')}:
        raise ValueError('unsupported content store')
    fingerprint = sha256(value.encode('utf-8')).digest()
    conn.execute(f'INSERT INTO {table}(digest,{column}) VALUES (?,?) '
                 'ON CONFLICT(digest) DO NOTHING', (fingerprint, value))
    identity, stored = conn.execute(
        f'SELECT id,{column} FROM {table} WHERE digest=?', (fingerprint,)).fetchone()
    if stored != value:
        raise ValueError('shared content digest collision or corruption')
    return identity
