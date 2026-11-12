"""Browser sessions. Long-lived upload tokens never leave the login exchange."""
from contextlib import closing
import hashlib
import secrets
import time

from ..storage.repository import _connect
from .repository import authenticate

TTL = 12 * 60 * 60
COOKIE = '__Host-af_session'


def initialize(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS browser_sessions_v1 (
        digest TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
        key_hash TEXT NOT NULL, csrf TEXT NOT NULL, expires REAL NOT NULL)''')
    conn.execute('CREATE INDEX IF NOT EXISTS browser_sessions_expiry ON browser_sessions_v1(expires)')


def issue(path, token):
    if not isinstance(token, str) or not 20 <= len(token) <= 256:
        raise PermissionError('invalid credential')
    secret = secrets.token_urlsafe(32)
    csrf = secrets.token_urlsafe(32)
    with closing(_connect(path, writable=True)) as conn, conn:
        user = authenticate(conn, token)
        conn.execute('DELETE FROM browser_sessions_v1 WHERE expires<=?', (time.time(),))
        # Bound active sessions per identity; old browser sessions expire first.
        conn.execute('''DELETE FROM browser_sessions_v1 WHERE digest IN
            (SELECT digest FROM browser_sessions_v1 WHERE user_id=? ORDER BY expires DESC LIMIT -1 OFFSET 19)''', (user,))
        conn.execute('INSERT INTO browser_sessions_v1 VALUES (?,?,?,?,?)',
                     (_hash(secret), user, _hash(token), csrf, time.time() + TTL))
    return secret


def _hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def identify(path, secret, csrf=None):
    if not isinstance(secret, str) or not 20 <= len(secret) <= 128:
        raise PermissionError('session required')
    with closing(_connect(path)) as conn:
        row = conn.execute('''SELECT u.id,u.display_name,s.csrf FROM browser_sessions_v1 s
            JOIN users u ON u.id=s.user_id AND u.active=1 AND u.key_hash=s.key_hash
            WHERE s.digest=? AND s.expires>?''', (_hash(secret), time.time())).fetchone()
    if row is None or (csrf is not None and not secrets.compare_digest(row[2].encode(), csrf.encode())):
        raise PermissionError('invalid session')
    return dict(user_id=row[0], display_name=row[1], csrf=row[2])


def revoke(path, secret):
    with closing(_connect(path, writable=True)) as conn, conn:
        conn.execute('DELETE FROM browser_sessions_v1 WHERE digest=?', (_hash(secret),))


def cookie(secret, *, clear=False):
    return f'{COOKIE}={secret}; Path=/; HttpOnly; Secure; SameSite=Strict; Max-Age={0 if clear else TTL}'
