import base64
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
from autofusex.client.artifacts import atomic
from autofusex.contracts.configuration import canonical
from autofusex.client.updates import verify_manifest,activate
from autofusex.client.store import Store


def signed(tmp_path, wheels=None):
    key=Ed25519PrivateKey.generate()
    public=key.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw).hex()
    atomic(tmp_path/'trust.json',dict(release_keys={'test':public}))
    manifest=dict(release_id='0.2.0',protocol=1,manager_schema=1,wheels=[{}] if wheels is None else wheels)
    envelope=dict(manifest=manifest,key_id='test',signature=base64.b64encode(key.sign(canonical(manifest).encode())).decode())
    return envelope


def test_stage_accepts_signed_autofusex_wheel(tmp_path, monkeypatch):
    from autofusex.client import updates
    artifact = tmp_path/'artifact.whl'
    artifact.write_bytes(b'wheel')
    envelope = signed(tmp_path, [dict(name='autofusex', version='0.1.2',
        filename='autofusex-0.1.2-py3-none-any.whl', url='https://example.invalid/autofusex.whl', sha256='a'*64)])
    monkeypatch.setattr(updates, 'download', lambda *args, **kwargs: artifact)
    commands = []
    monkeypatch.setattr(updates, 'command', lambda args, **kwargs: commands.append(args))
    store = Store(tmp_path)
    monkeypatch.setattr(store, 'event', lambda *args: None)
    result = updates.stage(store, envelope, 'test')
    directory = tmp_path/'tools'/result['release_id']
    assert result['state'] == 'staged'
    assert (directory/'requirements.lock').read_text().startswith('autofusex==0.1.2 --hash=sha256:')
    assert (directory/'ready.json').is_file()
    assert any('--require-hashes' in args for args in commands)


def test_release_signature_and_version_gate(tmp_path):
    envelope=signed(tmp_path)
    assert verify_manifest(tmp_path,envelope)['release_id']=='0.2.0'
    envelope['manifest']['release_id']='tampered'
    with pytest.raises(PermissionError):verify_manifest(tmp_path,envelope)


def test_untrusted_key_and_idle_gate(tmp_path):
    envelope=signed(tmp_path);envelope['key_id']='unknown'
    with pytest.raises(PermissionError):verify_manifest(tmp_path,envelope)
    store=Store(tmp_path)
    with pytest.raises(ValueError):activate(store,'0'*64,{})
    store.put('launcher_enabled',True)
    with pytest.raises(ValueError):activate(store,'0'*64,{'run':object()})


def test_signed_activation_uses_single_install_update(tmp_path, monkeypatch):
    from autofusex.client import client_updates
    from autofusex.contracts.serialization import fingerprint
    import hashlib
    envelope = signed(tmp_path, [dict(name='autofusex', version='0.2.5', filename='test.whl', sha256=hashlib.sha256(b'wheel').hexdigest())])
    identity = fingerprint(envelope['manifest'])
    store = Store(tmp_path)
    store.put('launcher_enabled', True)
    store.put('supervisor_protocol', 3)
    atomic(tmp_path/'tools'/identity/'ready.json', dict(id=identity,envelope=envelope))
    wheels=tmp_path/'tools'/identity/'wheels';wheels.mkdir()
    (wheels/'test.whl').write_bytes(b'wheel')
    seen=[]
    monkeypatch.setattr(client_updates, 'stage', lambda *args, **kwargs: seen.append(args))
    assert activate(store, identity, {})['state']=='waiting_launcher'
    assert seen[0][1]['version']=='0.2.5'
    assert not (tmp_path/'active-client.json').exists()
