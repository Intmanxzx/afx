"""Release safety tests use local artifacts and mocked transport, never real credentials."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

pytestmark = pytest.mark.skipif(sys.version_info < (3, 11), reason='release tooling requires Python 3.11+')
if sys.version_info >= (3, 11):
    spec = importlib.util.spec_from_file_location('release_wheel', Path(__file__).parents[1]/'tools/release_wheel.py')
    release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release)


@pytest.fixture
def artifact(tmp_path):
    (tmp_path/'autofusex').mkdir()
    (tmp_path/'autofusex/__init__.py').write_text('__version__ = "0.1.2"\n')
    (tmp_path/'pyproject.toml').write_text('[project]\nname="autofusex"\nversion="0.1.2"\n')
    def git(*args):
        subprocess.run(['git','-C',str(tmp_path),*args], check=True, capture_output=True)
    git('init');git('add','.');git('-c','user.name=Test','-c','user.email=test@example.invalid','commit','-m','fixture');git('tag','v0.1.2')
    dist=tmp_path/'dist';dist.mkdir();wheel=dist/'autofusex-0.1.2-py3-none-any.whl'
    with zipfile.ZipFile(wheel,'w') as archive:
        archive.writestr('autofusex-0.1.2.dist-info/METADATA','Name: autofusex\nVersion: 0.1.2\n')
        archive.writestr('autofusex-0.1.2.dist-info/entry_points.txt', '[console_scripts]\nafx = autofusex.cli:main\n')
        archive.writestr('afx/__init__.py', 'from autofusex import __version__\n')
        archive.writestr('afx/__main__.py', 'from autofusex.cli import main\nraise SystemExit(main())\n')
        archive.writestr('autofusex/__init__.py','__version__ = "0.1.2"\n')
        for asset in ['index.html','app.js','charts.js']:
            archive.writestr('autofusex/viewer/static/'+asset,'content')
    return tmp_path, wheel


def test_four_segment_version_uses_normalized_wheel_metadata():
    assert release.canonical_version('0.2.2.001') == '0.2.2.1'
    with pytest.raises(ValueError):
        release.canonical_version('0.2.x')


def test_plan_is_offline_and_version_checked(artifact, monkeypatch, capsys):
    root, wheel = artifact
    monkeypatch.chdir(root)
    monkeypatch.setattr(release,'request',lambda *a,**k: pytest.fail('dry run used network'))
    release.main(['--tag','v0.1.2','--repository','owner/repo'])
    assert 'Plan only' in capsys.readouterr().out
    with pytest.raises(ValueError,match='tag must match'):
        release.validate(root,'v0.1.3',wheel.parent)
    (root/'pyproject.toml').write_text((root/'pyproject.toml').read_text()+'# dirty\n')
    with pytest.raises(ValueError,match='dirty'):
        release.validate(root,'v0.1.2',wheel.parent)


def test_nnsmith_patches_are_not_bundled(artifact):
    root, wheel = artifact
    with zipfile.ZipFile(wheel, 'a') as archive:
        archive.writestr('autofusex/workloads/fuzz/nnsmith/patches/old.patch', 'obsolete')
    with pytest.raises(ValueError, match='obsolete bundled NNSmith patches'):
        release.validate(root, 'v0.1.2', wheel.parent)


@pytest.mark.parametrize('suffix, message', [
    ('entry_points.txt', 'afx console'),
    ('afx/__main__.py', 'python -m afx'),
])
def test_afx_entry_points_are_required(artifact, suffix, message):
    root, wheel = artifact
    with zipfile.ZipFile(wheel) as archive:
        contents = {name: archive.read(name) for name in archive.namelist() if not name.endswith(suffix)}
    with zipfile.ZipFile(wheel, 'w') as archive:
        for name, content in contents.items():
            archive.writestr(name, content)
    with pytest.raises(ValueError, match=message):
        release.validate(root, 'v0.1.2', wheel.parent)


def test_pypi_hash_idempotency(artifact,monkeypatch):
    _,wheel=artifact
    def missing(url, **kwargs):
        assert url == 'https://pypi.org/pypi/autofusex/0.1.2/json'
        return None
    monkeypatch.setattr(release,'request',missing)
    assert release.pypi_present(wheel,'0.1.2') is False
    data={'urls':[{'filename':wheel.name,'digests':{'sha256':release.digest(wheel.read_bytes())}}]}
    monkeypatch.setattr(release,'request',lambda *a,**k: json.dumps(data).encode())
    assert release.pypi_present(wheel,'0.1.2') is True
    data['urls'][0]['digests']['sha256']='wrong'
    with pytest.raises(ValueError,match='different bytes'):
        release.pypi_present(wheel,'0.1.2')


def test_release_create_and_upload_does_not_forward_token(artifact,monkeypatch):
    _,wheel=artifact;calls=[]
    def request(url,**kw):
        calls.append((url,kw))
        if '/tags/' in url:
            if len(calls)==1:return None
            return json.dumps({'assets':[{'name':wheel.name,'browser_download_url':'https://storage.example/wheel'}]}).encode()
        if url=='https://storage.example/wheel':return wheel.read_bytes()
        if '/upload_url' in url:return json.dumps({'url':'https://storage.example/upload','headers':{'X-Signed':'abc'}}).encode()
        if kw.get('method')=='POST':return b'{"assets":[]}'
        assert url=='https://storage.example/upload'
        assert kw['method']=='PUT' and kw['headers']=={'X-Signed':'abc'}
        assert kw['data']==wheel.read_bytes()
        return b''
    monkeypatch.setattr(release,'request',request)
    release.publish_release(wheel,'v0.1.2','owner/repo','secret')
    assert len(calls)==6


def test_existing_release_never_overwritten(artifact,monkeypatch):
    _,wheel=artifact
    def request(url,**kw):
        assert kw.get('method','GET')=='GET'
        if '/tags/' in url:
            return json.dumps({'assets':[{'name':wheel.name,'browser_download_url':'https://storage.example/wheel'}]}).encode()
        return wheel.read_bytes()
    monkeypatch.setattr(release,'request',request)
    release.publish_release(wheel,'v0.1.2','owner/repo','secret')
    monkeypatch.setattr(release,'request',lambda url,**kw: request(url,**kw) if '/tags/' in url else b'wrong')
    with pytest.raises(ValueError,match='refusing overwrite'):
        release.publish_release(wheel,'v0.1.2','owner/repo','secret')


def test_http_error_does_not_leak_secret(monkeypatch):
    def fail(*a,**k):
        raise release.HTTPError('https://example/?token=secret',401,'secret',{},None)
    monkeypatch.setattr(release,'urlopen',fail)
    with pytest.raises(RuntimeError) as error:
        release.request('https://example/?token=secret')
    assert 'secret' not in str(error.value)
    with pytest.raises(ValueError,match='HTTPS'):
        release.request('http://example')


def test_missing_viewer_asset_rejected(artifact):
    root, wheel = artifact
    with zipfile.ZipFile(wheel) as archive:
        contents = {name: archive.read(name) for name in archive.namelist() if not name.endswith('/charts.js')}
    with zipfile.ZipFile(wheel, 'w') as archive:
        for name, content in contents.items():
            archive.writestr(name, content)
    with pytest.raises(ValueError, match='viewer assets'):
        release.validate(root, 'v0.1.2', wheel.parent)


def test_publish_requires_both_credentials_before_network(artifact, monkeypatch):
    root, _ = artifact
    monkeypatch.chdir(root)
    monkeypatch.delenv('GITCODE_RELEASE_TOKEN', raising=False)
    monkeypatch.delenv('PYPI_API_TOKEN', raising=False)
    monkeypatch.setattr(release, 'request', lambda *a, **k: pytest.fail('unexpected network'))
    with pytest.raises(ValueError, match='credentials'):
        release.main(['--tag', 'v0.1.2', '--repository', 'owner/repo', '--publish'])


def test_github_publish_requires_its_own_credentials(artifact, monkeypatch):
    root, _ = artifact
    monkeypatch.chdir(root)
    monkeypatch.delenv('GITHUB_TOKEN', raising=False)
    monkeypatch.setenv('GITCODE_RELEASE_TOKEN', 'irrelevant')
    monkeypatch.delenv('PYPI_API_TOKEN', raising=False)
    monkeypatch.setattr(release, 'request', lambda *a, **k: pytest.fail('unexpected network'))
    with pytest.raises(ValueError, match='github'):
        release.main(['--tag', 'v0.1.2', '--repository', 'owner/repo', '--host', 'github', '--publish'])


def test_github_plan_names_its_host(artifact, monkeypatch, capsys):
    root, _ = artifact
    monkeypatch.chdir(root)
    monkeypatch.setattr(release, 'request', lambda *a, **k: pytest.fail('dry run used network'))
    release.main(['--tag', 'v0.1.2', '--repository', 'owner/repo', '--host', 'github'])
    assert 'github Release' in capsys.readouterr().out


def test_github_release_create_and_upload(artifact, monkeypatch):
    _, wheel = artifact
    calls = []
    def request(url, **kw):
        calls.append((url, kw))
        if url.endswith('/releases/tags/v0.1.2'):
            if len(calls) == 1:
                return None
            return json.dumps({'id': 7, 'assets': [{'name': wheel.name, 'url': 'https://api.github.com/asset/1'}]}).encode()
        if url == 'https://api.github.com/repos/owner/repo/releases':
            assert kw['method'] == 'POST'
            return json.dumps({'id': 7, 'assets': []}).encode()
        if url.startswith('https://uploads.github.com/repos/owner/repo/releases/7/assets'):
            assert kw['method'] == 'POST' and kw['data'] == wheel.read_bytes()
            assert kw['headers']['Content-Type'] == 'application/octet-stream'
            return b'{}'
        if url == 'https://api.github.com/asset/1':
            assert kw['headers']['Accept'] == 'application/octet-stream'
            return wheel.read_bytes()
        raise AssertionError(url)
    monkeypatch.setattr(release, 'request', request)
    release.publish_github_release(wheel, 'v0.1.2', 'owner/repo', 'secret')
    # tag lookup (missing) -> create -> upload -> tag lookup (verify) -> asset download
    assert len(calls) == 5


def test_github_existing_release_never_overwritten(artifact, monkeypatch):
    _, wheel = artifact
    asset = {'name': wheel.name, 'url': 'https://api.github.com/asset/1'}
    monkeypatch.setattr(release, 'request', lambda url, **kw: json.dumps({'id': 7, 'assets': [asset]}).encode()
                        if '/releases/tags/' in url else wheel.read_bytes())
    release.publish_github_release(wheel, 'v0.1.2', 'owner/repo', 'secret')
    monkeypatch.setattr(release, 'request', lambda url, **kw: json.dumps({'id': 7, 'assets': [asset]}).encode()
                        if '/releases/tags/' in url else b'wrong')
    with pytest.raises(ValueError, match='refusing overwrite'):
        release.publish_github_release(wheel, 'v0.1.2', 'owner/repo', 'secret')


def test_github_token_is_never_placed_in_the_url(artifact, monkeypatch):
    _, wheel = artifact
    seen = []
    def request(url, **kw):
        seen.append((url, kw.get('headers') or {}))
        if url.endswith('/releases/tags/v0.1.2'):
            return json.dumps({'id': 7, 'assets': [{'name': wheel.name, 'url': 'https://api.github.com/asset/1'}]}).encode()
        return wheel.read_bytes()
    monkeypatch.setattr(release, 'request', request)
    release.publish_github_release(wheel, 'v0.1.2', 'owner/repo', 'topsecret')
    for url, headers in seen:
        assert 'topsecret' not in url
        assert headers['Authorization'] == 'Bearer topsecret'
