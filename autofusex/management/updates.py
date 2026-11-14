"""Signed, isolated tool releases; activation is owned by an external launcher."""
import base64
import json
from pathlib import Path
import sys
import uuid
from .artifacts import atomic, command, download, trusted
from .config import canonical, fingerprint


def verify_manifest(home, envelope):
    if not isinstance(envelope,dict) or set(envelope)!={'manifest','signature','key_id'}:
        raise ValueError('signed manifest envelope required')
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.exceptions import InvalidSignature
    trust=trusted(home)
    key=trust.get('release_keys',{}).get(envelope['key_id'])
    if not key: raise PermissionError('untrusted release signing key')
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(key)).verify(base64.b64decode(envelope['signature'],validate=True),canonical(envelope['manifest']).encode())
    except (ValueError,InvalidSignature) as exc:
        raise PermissionError('invalid release signature') from exc
    manifest=envelope['manifest']
    if set(manifest)!={'release_id','protocol','manager_schema','wheels'} or manifest['protocol']!=1 or manifest['manager_schema']!=1:
        raise ValueError('release incompatible with management protocol/database')
    if not isinstance(manifest['release_id'],str) or not isinstance(manifest['wheels'],list) or not 1<=len(manifest['wheels'])<=64:
        raise ValueError('invalid release manifest')
    if manifest['release_id'] in trust.get('revoked_releases',[]): raise PermissionError('release revoked')
    return manifest


def stage(store,envelope,operation_id):
    manifest=verify_manifest(store.home,envelope)
    identity=fingerprint(manifest);directory=store.home/'tools'/identity
    if (directory/'ready.json').exists():return dict(release_id=identity,state='staged')
    directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    wheels=directory/'wheels';wheels.mkdir(exist_ok=True)
    import shutil
    requirements=[]
    trust=trusted(store.home)
    for item in manifest['wheels']:
        if set(item)!={'name','version','filename','url','sha256'}:raise ValueError('invalid release wheel')
        filename=item['filename']
        if Path(filename).name!=filename or not filename.endswith('.whl'):raise ValueError('invalid wheel filename')
        artifact=download(item['url'],item['sha256'],store.home/'artifacts',allowed_hosts=trust.get('allowed_hosts',[]))
        shutil.copy2(artifact,wheels/filename)
        requirements.append(f"{item['name']}=={item['version']} --hash=sha256:{item['sha256']}")
    if not any(item['name'].replace('-','_')=='autofusex' for item in manifest['wheels']):raise ValueError('tool wheel missing')
    lock=directory/'requirements.lock';lock.write_text('\n'.join(requirements)+'\n')
    command([sys.executable,'-m','venv',str(directory/'venv')])
    python=directory/'venv/bin/python'
    command([str(python),'-m','pip','install','--no-index','--find-links',str(wheels),'--require-hashes','-r',str(lock)],timeout=1800)
    command([str(python),'-c','import autofusex.management.daemon; import autofusex.management.central'])
    atomic(directory/'ready.json',dict(envelope=envelope,python=str(python),id=identity))
    store.event(operation_id,'Signed tool release staged; activation requires managed launcher and idle device')
    return dict(release_id=identity,state='staged')


def activate(store,release_id,processes):
    if not store.get('launcher_enabled'):raise ValueError('daemon must be started through the managed launcher')
    if processes:raise ValueError('drain running tasks before activating a release')
    if len(release_id)!=64 or any(c not in '0123456789abcdef' for c in release_id):raise ValueError('invalid release ID')
    path=store.home/'tools'/release_id/'ready.json'
    value=json.loads(path.read_text())
    verify_manifest(store.home,value['envelope'])
    if value['id']!=release_id:raise ValueError('release identity mismatch')
    previous=store.get('active_tool')
    store.put('tool_switch',dict(target=release_id,previous=previous,state='requested'))
    return dict(release_id=release_id,state='waiting_launcher')
