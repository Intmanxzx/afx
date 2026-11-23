"""Offline-issued users and bearer tokens (design 4.2)."""
from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass

from autofusex.server.store import Store, now_ms

ROLES = ("admin", "developer")
TOKEN_PREFIX = "afx_"
_USER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
# Only record last_used_at when it moved by more than this, to avoid a write per request.
_TOUCH_INTERVAL_MS = 60_000


class AuthError(Exception):
    pass


@dataclass(frozen=True)
class Principal:
    user_id: str
    role: str
    token_id: int

    @property
    def anonymous(self) -> bool:
        return self.token_id == 0

    def require_write(self) -> None:
        """Any logged-in user except a read-only viewer may change data."""
        if self.anonymous:
            raise AuthError("login required")
        if self.role == "viewer":
            raise AuthError("viewer role is read-only")

    def require_admin(self) -> None:
        if self.anonymous:
            raise AuthError("login required")
        if self.role != "admin":
            raise AuthError("admin role required")


ANONYMOUS = Principal("anonymous", "viewer", 0)


def _hash(store: Store, token: str) -> bytes:
    return hmac.new(store.secret, token.encode(), hashlib.sha256).digest()


def add_user(store: Store, user_id: str, role: str, display_name: str | None = None) -> None:
    if not _USER_RE.match(user_id):
        raise ValueError("user_id must be 1-64 chars of letters, digits, _ . -")
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    with store.write() as conn:
        if conn.execute("SELECT 1 FROM users WHERE user_id=?", (user_id,)).fetchone():
            raise ValueError(f"user {user_id} exists")
        conn.execute("INSERT INTO users(user_id, display_name, role, created_at) VALUES(?,?,?,?)",
                     (user_id, display_name, role, now_ms()))
        _audit(conn, "cli", "user.add", f"{user_id} role={role}")


def set_user_status(store: Store, user_id: str, status: str) -> None:
    if status not in ("active", "disabled"):
        raise ValueError("status must be active or disabled")
    with store.write() as conn:
        if conn.execute("UPDATE users SET status=? WHERE user_id=?", (status, user_id)).rowcount != 1:
            raise ValueError(f"no user {user_id}")
        _audit(conn, "cli", "user.status", f"{user_id} {status}")


def issue_token(store: Store, user_id: str, expires_days: int | None = None) -> str:
    """Create the user's token, revoking any earlier one. The plaintext is returned once and never stored."""
    token = TOKEN_PREFIX + base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
    created = now_ms()
    expires = created + expires_days * 86_400_000 if expires_days else None
    with store.write() as conn:
        row = conn.execute("SELECT role FROM users WHERE user_id=?", (user_id,)).fetchone()
        if row is None:
            raise ValueError(f"no user {user_id}")
        conn.execute("UPDATE user_tokens SET revoked_at=? WHERE user_id=? AND revoked_at IS NULL", (created, user_id))
        conn.execute("INSERT INTO user_tokens(user_id, token_prefix, token_hash, expires_at, created_at) "
                     "VALUES(?,?,?,?,?)", (user_id, token[:12], _hash(store, token), expires, created))
        _audit(conn, "cli", "token.issue", f"{user_id} prefix={token[:12]}")
    return token


def revoke_token(store: Store, token_prefix: str) -> int:
    with store.write() as conn:
        count = conn.execute("UPDATE user_tokens SET revoked_at=? WHERE token_prefix=? AND revoked_at IS NULL",
                             (now_ms(), token_prefix)).rowcount
        _audit(conn, "cli", "token.revoke", f"prefix={token_prefix} count={count}")
    return count


def list_tokens(store: Store, user_id: str | None = None):
    sql = ("SELECT token_id, user_id, token_prefix, expires_at, revoked_at, created_at, "
           "last_used_at FROM user_tokens")
    args = ()
    if user_id:
        sql += " WHERE user_id=?"
        args = (user_id,)
    return [dict(row) for row in store.reader().execute(sql + " ORDER BY token_id", args)]


def authenticate(store: Store, header: str | None) -> Principal:
    if not header or not header.startswith("Bearer "):
        raise AuthError("missing bearer token")
    token = header[7:].strip()
    if not token.startswith(TOKEN_PREFIX) or len(token) > 128:
        raise AuthError("invalid token")
    row = store.reader().execute(
        "SELECT t.token_id, t.user_id, t.expires_at, t.revoked_at, t.last_used_at, "
        "u.role, u.status FROM user_tokens t JOIN users u USING(user_id) WHERE t.token_hash=?",
        (_hash(store, token),)).fetchone()
    now = now_ms()
    if row is None or row["revoked_at"] is not None or row["status"] != "active":
        raise AuthError("invalid token")
    if row["expires_at"] is not None and row["expires_at"] <= now:
        raise AuthError("token expired")
    if row["last_used_at"] is None or now - row["last_used_at"] > _TOUCH_INTERVAL_MS:
        with store.write() as conn:
            conn.execute("UPDATE user_tokens SET last_used_at=? WHERE token_id=?", (now, row["token_id"]))
            conn.execute("UPDATE users SET last_seen_at=? WHERE user_id=?", (now, row["user_id"]))
    return Principal(row["user_id"], row["role"], row["token_id"])


def _audit(conn, actor: str, action: str, detail: str) -> None:
    conn.execute("INSERT INTO audit_log(at, actor, action, detail) VALUES(?,?,?,?)",
                 (now_ms(), actor, action, detail))
