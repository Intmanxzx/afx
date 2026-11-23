"""Approval-based enrollment. Only credential digests are persisted."""
import re
import secrets
from autofusex.server.auth import _hash, _audit, _USER_RE, ROLES, AuthError
from autofusex.server.store import now_ms

TTL = 7 * 86400 * 1000

class AccountError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def _body(body):
    if not isinstance(body, dict):
        raise AccountError('expected an object')
    return body


def apply(store, body):
    body = _body(body)
    user = body.get('user_id')
    if not isinstance(user, str) or not _USER_RE.fullmatch(user):
        raise AccountError('用户名须为 1–64 位字母、数字、点、下划线或短横线，以字母或数字开头')
    now = now_ms()
    receipt = secrets.token_urlsafe(32)
    with store.write() as db:
        db.execute("UPDATE registration_requests SET status='expired' WHERE status IN ('pending','approved') AND expires_at<=?", (now,))
        if db.execute('SELECT 1 FROM users WHERE user_id=?', (user,)).fetchone() or db.execute(
            "SELECT 1 FROM registration_requests WHERE user_id=? AND status IN ('pending','approved')", (user,)).fetchone():
            raise AccountError('该用户名已有账号或待处理申请', 409)
        if db.execute("SELECT count(*) FROM registration_requests WHERE status IN ('pending','approved')").fetchone()[0] >= 1000:
            raise AccountError('申请队列已满，请稍后再试', 429)
        # Retain a bounded amount of anonymous application text.
        db.execute("DELETE FROM registration_requests WHERE expires_at<?", (now - 30 * 86400000,))
        db.execute("INSERT INTO registration_requests(user_id,receipt_hash,status,created_at,expires_at) VALUES(?,?,'pending',?,?)",
                   (user, _hash(store, 'application:' + receipt), now, now + TTL))
    return dict(receipt=receipt, expires_at=now + TTL)


def _request(db, store, body):
    receipt = _body(body).get('receipt')
    if not isinstance(receipt, str) or not re.fullmatch(r'[A-Za-z0-9_-]{43}', receipt):
        raise AccountError('申请凭证无效', 404)
    row = db.execute('SELECT * FROM registration_requests WHERE receipt_hash=?', (_hash(store, 'application:' + receipt),)).fetchone()
    if row is None:
        raise AccountError('申请凭证无效', 404)
    return row


def status(store, body):
    row = _request(store.reader(), store, body)
    state = row['status']
    if state in ('pending', 'approved') and row['expires_at'] <= now_ms():
        state = 'expired'
    return dict(user_id=row['user_id'], status=state, expires_at=row['expires_at'])


def _issue(db, store, user, actor):
    token = 'afx_' + secrets.token_urlsafe(32)
    now = now_ms()
    db.execute('UPDATE user_tokens SET revoked_at=? WHERE user_id=? AND revoked_at IS NULL', (now, user))
    db.execute('INSERT INTO user_tokens(user_id,token_prefix,token_hash,created_at) VALUES(?,?,?,?)',
               (user, token[:12], _hash(store, token), now))
    _audit(db, actor, 'token.issue', user)
    return token


def claim(store, body):
    with store.write() as db:
        row = _request(db, store, body)
        if row['expires_at'] <= now_ms() or row['status'] != 'approved':
            raise AccountError('申请尚未批准、已领取或已过期', 409)
        user = db.execute('SELECT status FROM users WHERE user_id=?', (row['user_id'],)).fetchone()
        if not user or user['status'] != 'active':
            raise AccountError('账号已禁用', 403)
        # An administrator reset invalidates the original claim path.
        if db.execute('SELECT 1 FROM user_tokens WHERE user_id=?', (row['user_id'],)).fetchone():
            raise AccountError('凭据已由管理员签发，请联系管理员', 409)
        token = _issue(db, store, row['user_id'], row['user_id'])
        db.execute("UPDATE registration_requests SET status='claimed' WHERE id=?", (row['id'],))
    return dict(token=token, user_id=row['user_id'])


def listing(store, principal):
    principal.require_admin()
    with store.write() as db:
        db.execute("UPDATE registration_requests SET status='expired' WHERE status IN ('pending','approved') AND expires_at<=?", (now_ms(),))
        requests = [dict(r) for r in db.execute("SELECT id,user_id,status,created_at,expires_at FROM registration_requests WHERE status='pending' ORDER BY id")]
        users = [dict(r) for r in db.execute('SELECT user_id,role,status,created_at,last_seen_at FROM users ORDER BY created_at DESC')]
    return dict(requests=requests, users=users)


def review(store, principal, body):
    principal.require_admin()
    body = _body(body)
    action, role = body.get('action'), body.get('role', 'developer')
    if action not in ('approve', 'reject') or role not in ROLES or type(body.get('id')) is not int:
        raise AccountError('无效的审核参数')
    with store.write() as db:
        row = db.execute('SELECT * FROM registration_requests WHERE id=?', (body['id'],)).fetchone()
        target = 'approved' if action == 'approve' else 'rejected'
        if row and row['status'] == target:
            return dict(status=target)
        if not row or row['status'] != 'pending' or row['expires_at'] <= now_ms():
            raise AccountError('申请已处理或已过期', 409)
        if action == 'approve':
            if db.execute('SELECT 1 FROM users WHERE user_id=?', (row['user_id'],)).fetchone():
                raise AccountError('用户名已存在', 409)
            db.execute('INSERT INTO users(user_id,role,created_at) VALUES(?,?,?)', (row['user_id'], role, now_ms()))
        db.execute('UPDATE registration_requests SET status=?,reviewed_by=? WHERE id=?', (target, principal.user_id, row['id']))
        _audit(db, principal.user_id, 'application.' + action, row['user_id'])
    return dict(status=target)


def manage(store, principal, body):
    principal.require_admin()
    body = _body(body)
    user, action = body.get('user_id'), body.get('action')
    if not isinstance(user, str) or action not in ('enable', 'disable', 'role', 'reset-token'):
        raise AccountError('无效的用户操作')
    with store.write() as db:
        row = db.execute('SELECT * FROM users WHERE user_id=?', (user,)).fetchone()
        if not row:
            raise AccountError('用户不存在', 404)
        role = body.get('role') if action == 'role' else row['role']
        if action == 'role' and role not in ROLES:
            raise AccountError('角色必须是 developer 或 admin')
        if row['role'] == 'admin' and row['status'] == 'active' and (action == 'disable' or role != 'admin'):
            if db.execute("SELECT count(*) FROM users WHERE role='admin' AND status='active'").fetchone()[0] <= 1:
                raise AccountError('不能禁用或降级最后一个管理员', 409)
        if action == 'reset-token':
            token = _issue(db, store, user, principal.user_id)
            db.execute("UPDATE registration_requests SET status='claimed' WHERE user_id=? AND status='approved'", (user,))
            return dict(token=token, user_id=user)
        if action == 'role':
            db.execute('UPDATE users SET role=? WHERE user_id=?', (role, user))
        else:
            db.execute('UPDATE users SET status=? WHERE user_id=?', ('active' if action == 'enable' else 'disabled', user))
        _audit(db, principal.user_id, 'user.' + action, user + ((' role=' + role) if action == 'role' else ''))
    return dict(user_id=user)
