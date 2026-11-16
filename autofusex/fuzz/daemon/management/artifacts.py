"""Bounded, pinned-address HTTPS downloads and explicit local execution authority."""
import hashlib
import http.client
import ipaddress
import json
import os
from pathlib import Path
import socket
import ssl
import subprocess
import time
from urllib.parse import urlsplit, urljoin

from .config import canonical, fingerprint


def atomic(path, value):
    import tempfile
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.write-')
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self, host, address, port):
        super().__init__(host, port, timeout=30, context=ssl.create_default_context())
        self.address = address

    def connect(self):
        sock = socket.create_connection((self.address, self.port), self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


def download(url, expected, directory, *, allowed_hosts=(), max_bytes=8*1024**3):
    if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('trusted SHA-256 is required')
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = directory / expected
    if target.exists() and sha256(target) == expected:
        return target
    import tempfile
    fd, name = tempfile.mkstemp(dir=directory, prefix='.download-')
    try:
        with os.fdopen(fd, 'wb') as output:
            for redirect in range(6):
                parsed = urlsplit(url)
                if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
                    raise ValueError('artifact URL must be credential-free HTTPS')
                host = parsed.hostname
                addresses = {item[4][0] for item in socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)}
                if not addresses:
                    raise ValueError('download host has no addresses')
                for address in addresses:
                    ip = ipaddress.ip_address(address)
                    if ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified or (not ip.is_global and host not in allowed_hosts):
                        raise ValueError('download destination blocked by local policy')
                conn = PinnedHTTPS(host, sorted(addresses)[0], parsed.port or 443)
                try:
                    conn.request('GET', (parsed.path or '/') + ('?' + parsed.query if parsed.query else ''), headers={'User-Agent':'AutofuseX-manager/1'})
                    response = conn.getresponse()
                    if response.status in {301,302,303,307,308}:
                        url = urljoin(url, response.getheader('Location', ''))
                        continue
                    if response.status != 200:
                        raise ValueError('artifact download returned HTTP %s' % response.status)
                    length = response.getheader('Content-Length')
                    if length and int(length) > max_bytes:
                        raise ValueError('artifact exceeds download budget')
                    total = 0
                    while True:
                        from .cancellation import requested
                        if requested(): raise InterruptedError('download cancelled')
                        block = response.read(1024*1024)
                        if not block: break
                        total += len(block)
                        if total > max_bytes:
                            raise ValueError('artifact exceeds download budget')
                        output.write(block)
                    output.flush(); os.fsync(output.fileno())
                    break
                finally:
                    conn.close()
            else:
                raise ValueError('too many download redirects')
        if sha256(name) != expected:
            raise ValueError('artifact SHA-256 mismatch')
        os.replace(name, target)
        return target
    finally:
        Path(name).unlink(missing_ok=True)


def trusted(home):
    path = Path(home)/'trust.json'
    if not path.exists():
        return dict(profiles={}, compatibility={}, approved_artifacts=[], allowed_hosts=[])
    if path.stat().st_mode & 0o077:
        raise PermissionError('trust.json must be private (0600)')
    return json.loads(path.read_text())


def profile(home, name, kind):
    value = trusted(home).get('profiles', {}).get(name)
    if not isinstance(value, dict) or value.get('kind') != kind:
        raise PermissionError('profile is not locally authorized: '+name)
    return value


def confined(path, roots):
    path = Path(path).expanduser().resolve()
    for root in roots:
        root = Path(root).expanduser().resolve()
        if path == root or root in path.parents:
            return path
    raise PermissionError('path outside locally authorized roots')


def command(argv, *, cwd=None, env=None, timeout=3600, cancel=None):
    """Never shell-expand user input. Do not expose installer output containing secrets."""
    if not isinstance(argv, list) or not argv or any(not isinstance(a,str) or '\x00' in a for a in argv):
        raise ValueError('invalid trusted argv')
    import signal
    process = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    from .cancellation import requested
    cancel = cancel or requested
    started = time.monotonic()
    try:
        while process.poll() is None:
            if time.monotonic()-started > timeout or (cancel and cancel()):
                raise TimeoutError('subprocess cancelled or timed out')
            time.sleep(.1)
        if process.returncode:
            raise RuntimeError('trusted subprocess exited with code %s' % process.returncode)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()


def expand(argv, values):
    # Only profiles trusted by the local owner can define executable commands.
    return [part.format_map(values) for part in argv]
