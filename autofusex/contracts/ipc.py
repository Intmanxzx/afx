"""Local control transport shared by the stable supervisor and client."""
import hashlib
import json
from pathlib import Path
import socket

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _socket_path(home):
    home = Path(home)
    default = home / 'manager.sock'
    if len(str(default).encode()) <= 100:
        return default
    return Path('/tmp') / ('afx-manager-' + hashlib.sha256(str(home).encode()).hexdigest()[:16] + '.sock')

def request(home,payload):
    sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);sock.settimeout(20)
    try:
        sock.connect(str(_socket_path(home)));sock.sendall(canonical(payload).encode()+b'\n')
        with sock.makefile('rb') as stream: raw=stream.readline(131073)
        if len(raw)>131072: raise ValueError('oversized local response')
        result=json.loads(raw)
        if not result.get('ok'): raise ValueError(result.get('error','local operation failed'))
        return result['data']
    finally: sock.close()
