import hashlib
import json
from pathlib import Path
import pytest
from autofusex.client import artifacts


def test_local_trust_and_path_confinement(tmp_path):
    artifacts.atomic(tmp_path/'trust.json',{'profiles':{'installer':{'kind':'cann-installer'}}})
    assert artifacts.profile(tmp_path,'installer','cann-installer')
    with pytest.raises(PermissionError):artifacts.profile(tmp_path,'installer','build')
    with pytest.raises(PermissionError):artifacts.confined(tmp_path/'../outside',[tmp_path])
    (tmp_path/'link').symlink_to('/tmp')
    with pytest.raises(PermissionError):artifacts.confined(tmp_path/'link/file',[tmp_path])
    (tmp_path/'trust.json').chmod(0o644)
    with pytest.raises(PermissionError):artifacts.trusted(tmp_path)


@pytest.mark.parametrize('url',['http://example.com/x','https://user:secret@example.com/x','file:///tmp/x','https://127.0.0.1/x','https://169.254.169.254/latest'])
def test_unsafe_download_rejected(tmp_path,url):
    with pytest.raises(ValueError):artifacts.download(url,'0'*64,tmp_path)
    assert list(tmp_path.iterdir())==[]


def test_pinned_download_integrity_and_redirect(tmp_path,monkeypatch):
    payload=b'test artifact';expected=hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(artifacts.socket,'getaddrinfo',lambda *a,**k:[(2,1,6,'',('93.184.216.34',443))])
    class Response:
        status=200
        def __init__(self):self.done=False
        def getheader(self,key,default=None):return str(len(payload)) if key=='Content-Length' else default
        def read(self,n):
            if self.done:return b''
            self.done=True;return payload
    class Connection:
        def __init__(self,host,address,port):assert address=='93.184.216.34'
        def request(self,*a,**k):pass
        def getresponse(self):return Response()
        def close(self):pass
    monkeypatch.setattr(artifacts,'PinnedHTTPS',Connection)
    path=artifacts.download('https://example.com/a',expected,tmp_path)
    assert path.read_bytes()==payload
    with pytest.raises(ValueError):artifacts.download('https://example.com/a','0'*64,tmp_path)
    assert not list(tmp_path.glob('.download-*'))


def test_command_failure_and_timeout():
    import sys
    with pytest.raises(RuntimeError):artifacts.command([sys.executable,'-c','raise SystemExit(7)'])
    with pytest.raises(TimeoutError):artifacts.command([sys.executable,'-c','import time;time.sleep(30)'],timeout=.1)
