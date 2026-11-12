"""Startup-only identity allowlist for an isolated validation window.

This is admission control, not authentication. Existing ownership/lease checks
still apply. No public endpoint can modify this policy.
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import uuid

ENVIRONMENT = 'AF_FUZZ_WRITE_ALLOWLIST'


class MaintenanceBlocked(Exception):
    pass


def _identity(value):
    if not isinstance(value, str):
        raise ValueError('allowlist identities must be UUID strings')
    return str(uuid.UUID(value))


@dataclass(frozen=True)
class WriteAllowlist:
    enabled: bool = False
    users: frozenset = frozenset()
    clients: frozenset = frozenset()

    @classmethod
    def load(cls, filename):
        if filename is None:
            return cls()
        # Missing, oversized or malformed configuration must never fail open.
        with Path(filename).open('rb') as stream:
            raw = stream.read(65537)
        if len(raw) > 65536:
            raise ValueError('allowlist configuration too large')
        value = json.loads(raw)
        if (not isinstance(value, dict) or set(value) != {'enabled', 'users', 'clients'}
                or type(value['enabled']) is not bool
                or not isinstance(value['users'], list) or not isinstance(value['clients'], list)):
            raise ValueError('invalid write allowlist configuration')
        users = frozenset(_identity(user) for user in value['users'])
        clients = set()
        for pair in value['clients']:
            if not isinstance(pair, dict) or set(pair) != {'user_id', 'client_id'}:
                raise ValueError('invalid allowlist client binding')
            user, client = _identity(pair['user_id']), _identity(pair['client_id'])
            if user not in users:
                raise ValueError('client user must be in allowed users')
            clients.add((user, client))
        return cls(value['enabled'], users, frozenset(clients))

    @classmethod
    def environment(cls):
        return cls.load(os.environ.get(ENVIRONMENT))

    def browser(self, user):
        if self.enabled and user not in self.users:
            raise MaintenanceBlocked()

    def client(self, user, client):
        if self.enabled:
            try:
                client = _identity(client)
            except (ValueError, TypeError, AttributeError):
                raise MaintenanceBlocked() from None
            if (user, client) not in self.clients:
                raise MaintenanceBlocked()


def reject(handler):
    handler.send(503, {'error': '验证维护中，当前身份暂不可写入，请稍后重试',
                       'code': 'maintenance_write_blocked'}, headers={'Retry-After': '60'})
