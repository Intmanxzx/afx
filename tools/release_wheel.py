"""Validate a tagged wheel; publishing is opt-in. Requires Python 3.11+.

--host picks which forge receives the Release attachment. Releases are cut from
the GitHub mirror, so the GitHub workflow passes --host github; the GitCode
default keeps older local invocations working.
"""
import argparse
import configparser
import email
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import tomllib
from urllib.error import HTTPError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen
import zipfile

API = 'https://api.gitcode.com/api/v5'
GITHUB_API = 'https://api.github.com'
GITHUB_UPLOADS = 'https://uploads.github.com'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def request(url, *, method='GET', data=None, headers=None, missing=False):
    # Never log URLs, response bodies or exception text: these can contain credentials.
    if urlparse(url).scheme != 'https':
        raise ValueError('release transport requires HTTPS')
    try:
        with urlopen(Request(url, data=data, headers=headers or {}, method=method), timeout=120) as response:
            return response.read()
    except HTTPError as exc:
        if missing and exc.code == 404:
            return None
        raise RuntimeError(f'release HTTP request failed ({exc.code})') from None
    except Exception:
        raise RuntimeError('release network request failed') from None


def validate(root, tag, dist):
    project = tomllib.loads((root / 'pyproject.toml').read_text())['project']
    version = project['version']
    if project['name'] != 'autofusex':
        raise ValueError('project name must be autofusex')
    if tag != 'v' + version or not re.fullmatch(r'\d+\.\d+\.\d+', version):
        raise ValueError('tag must match the stable project version vX.Y.Z')
    init = (root / 'autofusex/__init__.py').read_text()
    if not re.search(r'__version__\s*=\s*[\'"]' + re.escape(version) + r'[\'"]', init):
        raise ValueError('package __version__ differs from project version')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.DEVNULL).decode().strip()
    if git('status', '--porcelain', '--untracked-files=no'):
        raise ValueError('tracked working tree is dirty')
    if git('rev-parse', 'HEAD') != git('rev-parse', f'refs/tags/{tag}^{{commit}}'):
        raise ValueError('tag does not point to HEAD')
    wheels = list(dist.glob('*.whl'))
    if len(wheels) != 1:
        raise ValueError('dist must contain exactly one wheel')
    wheel = wheels[0]
    if wheel.name != f'autofusex-{version}-py3-none-any.whl':
        raise ValueError('unexpected wheel filename')
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata = [name for name in names if name.endswith('.dist-info/METADATA')]
        if len(metadata) != 1:
            raise ValueError('invalid wheel metadata')
        info = email.message_from_bytes(archive.read(metadata[0]))
        if info['Name'] != project['name'] or info['Version'] != version:
            raise ValueError('wheel metadata differs from project')
        entry_path = metadata[0].removesuffix('METADATA') + 'entry_points.txt'
        entries = configparser.ConfigParser()
        if entry_path not in names:
            raise ValueError('wheel lacks afx console entry point')
        entries.read_string(archive.read(entry_path).decode())
        if not entries.has_section('console_scripts') or dict(entries['console_scripts']) != {'afx': 'autofusex.cli:main'}:
            raise ValueError('wheel must expose only the afx console command')
        if not {'afx/__init__.py', 'afx/__main__.py'} <= set(names):
            raise ValueError('wheel lacks python -m afx entry point')
        for asset in ('index.html', 'app.js', 'charts.js'):
            if f'autofusex/fuzz/server/view/static/{asset}' not in names:
                raise ValueError('wheel lacks viewer assets')
        if any(name.startswith('autofusex/fuzz/client/nnsmith/patches/') for name in names):
            raise ValueError('wheel contains obsolete bundled NNSmith patches')
        if archive.read('autofusex/__init__.py').decode() != init:
            raise ValueError('wheel package version source differs')
    return wheel, version


def pypi_present(wheel, version):
    raw = request(f'https://pypi.org/pypi/autofusex/{version}/json', missing=True)
    if raw is None:
        return False
    for entry in json.loads(raw)['urls']:
        if entry['filename'] == wheel.name:
            if entry['digests']['sha256'] != digest(wheel.read_bytes()):
                raise ValueError('PyPI already contains different bytes for this wheel')
            return True
    raise ValueError('PyPI version already exists without the expected wheel; use a new version')


def publish_release(wheel, tag, repository, token):
    if not re.fullmatch(r'[\w.-]+/[\w.-]+', repository):
        raise ValueError('repository must be owner/name')
    base = f'{API}/repos/{repository}/releases'
    def api(path='', *, body=None, missing=False, extra=None):
        url = base + path + '?' + urlencode({'access_token': token, **(extra or {})})
        raw = request(url, method='POST' if body is not None else 'GET',
                      data=json.dumps(body).encode() if body is not None else None,
                      headers={'Content-Type': 'application/json'}, missing=missing)
        return json.loads(raw) if raw is not None else None
    release = api('/tags/' + quote(tag, safe=''), missing=True)
    if release is None:
        release = api(body={'tag_name': tag, 'name': tag,
                            'body': f'{wheel.name}\nSHA256: {digest(wheel.read_bytes())}'})
    assets = release.get('assets', [])
    if isinstance(assets, dict):
        assets = assets.get('links', [])
    for asset in assets:
        if asset.get('name') == wheel.name:
            url = asset.get('browser_download_url') or asset.get('direct_asset_url') or asset.get('url')
            if not url or digest(request(url)) != digest(wheel.read_bytes()):
                raise ValueError('existing Release asset cannot be verified or differs; refusing overwrite')
            return
    upload = api('/' + quote(tag, safe='') + '/upload_url', extra={'file_name': wheel.name})
    request(upload['url'], method='PUT', data=wheel.read_bytes(), headers=upload['headers'])
    for attempt in range(3):
        current = api("/tags/" + quote(tag, safe=""))
        assets = current.get("assets", [])
        if isinstance(assets, dict):
            assets = assets.get("links", [])
        for asset in assets:
            if asset.get("name") == wheel.name:
                url = asset.get("browser_download_url") or asset.get("direct_asset_url") or asset.get("url")
                if url and digest(request(url)) == digest(wheel.read_bytes()):
                    return
                raise ValueError("uploaded Release asset digest mismatch")
        if attempt < 2:
            time.sleep(2)
    raise RuntimeError("Release upload not visible; PyPI not attempted; retry identical wheel")


def github_request(path, token, *, method='GET', data=None, extra_headers=None, missing=False, base=GITHUB_API):
    # Never log URLs, response bodies or exception text: these can contain credentials.
    headers = {
        'Accept': 'application/vnd.github+json',
        'Authorization': f'Bearer {token}',
        'X-GitHub-Api-Version': '2022-11-28',
        'User-Agent': 'autofusex-release',
    }
    headers.update(extra_headers or {})
    raw = request(base + path, method=method, data=data, headers=headers, missing=missing)
    return json.loads(raw) if raw is not None else None


def github_assets(release):
    return release.get('assets') or []


def github_asset_bytes(asset, token):
    # The API URL plus an octet-stream Accept returns the asset for private and
    # public repositories alike; browser_download_url needs a session otherwise.
    return request(asset['url'], headers={
        'Accept': 'application/octet-stream',
        'Authorization': f'Bearer {token}',
        'User-Agent': 'autofusex-release',
    })


def publish_github_release(wheel, tag, repository, token):
    if not re.fullmatch(r'[\w.-]+/[\w.-]+', repository):
        raise ValueError('repository must be owner/name')
    matches = github_request(f'/repos/{repository}/releases/tags/{quote(tag, safe="")}', token, missing=True)
    if matches is None:
        matches = github_request(f'/repos/{repository}/releases', token, method='POST',
                                 data=json.dumps({'tag_name': tag, 'name': tag,
                                                  'body': f'{wheel.name}\nSHA256: {digest(wheel.read_bytes())}'}).encode(),
                                 extra_headers={'Content-Type': 'application/json'})
    for asset in github_assets(matches):
        if asset.get('name') == wheel.name:
            if digest(github_asset_bytes(asset, token)) != digest(wheel.read_bytes()):
                raise ValueError('existing Release asset differs; refusing overwrite')
            return
    upload = (f'/repos/{repository}/releases/{matches["id"]}/assets'
              f'?name={quote(wheel.name, safe="")}')
    github_request(upload, token, method='POST', data=wheel.read_bytes(), base=GITHUB_UPLOADS,
                   extra_headers={'Content-Type': 'application/octet-stream'})
    for attempt in range(3):
        current = github_request(f'/repos/{repository}/releases/tags/{quote(tag, safe="")}', token)
        for asset in github_assets(current):
            if asset.get('name') == wheel.name:
                if digest(github_asset_bytes(asset, token)) == digest(wheel.read_bytes()):
                    return
                raise ValueError('uploaded Release asset digest mismatch')
        if attempt < 2:
            time.sleep(2)
    raise RuntimeError('Release upload not visible; PyPI not attempted; retry identical wheel')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--repository', required=True)
    parser.add_argument('--dist', type=Path, default=Path('dist'))
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--host', choices=('gitcode', 'github'), default='gitcode',
                        help='which forge receives the Release attachment')
    args = parser.parse_args(argv)
    wheel, version = validate(Path.cwd(), args.tag, args.dist)
    if not re.fullmatch(r'[\w.-]+/[\w.-]+', args.repository):
        raise ValueError('repository must be owner/name')
    print(f'Validated {wheel.name}; SHA256 {digest(wheel.read_bytes())}')
    if not args.publish:
        print(f'Plan only: {args.host} Release attachment, then PyPI. No network writes.')
        return
    pypi = os.environ.get('PYPI_API_TOKEN')
    if args.host == 'github':
        forge_token = os.environ.get('GITHUB_TOKEN')
        publish = lambda: publish_github_release(wheel, args.tag, args.repository, forge_token)
    else:
        forge_token = os.environ.get('GITCODE_RELEASE_TOKEN')
        publish = lambda: publish_release(wheel, args.tag, args.repository, forge_token)
    if not forge_token or not pypi:
        raise ValueError(f'both {args.host} and PyPI credentials are required')
    present = pypi_present(wheel, version)  # detect version conflicts before any writes
    publish()
    if not present:
        env = dict(os.environ, TWINE_USERNAME='__token__', TWINE_PASSWORD=pypi,
                   TWINE_NON_INTERACTIVE='1')
        # Capture and discard third-party failure output; never leak credentials.
        result = subprocess.run([sys.executable, '-m', 'twine', 'upload',
                                 '--repository-url', 'https://upload.pypi.org/legacy/', str(wheel)],
                                env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('PyPI upload failed; Release may already exist; retry the identical wheel')
    print('Release attachment and PyPI wheel published or verified unchanged.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Only our controlled messages are safe; arbitrary SDK/API text is not.
        safe = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else type(exc).__name__
        print(f'Release aborted: {safe}', file=sys.stderr)
        sys.exit(1)
