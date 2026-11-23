import base64
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
from autofusex.fuzz.daemon.management.artifacts import atomic
from autofusex.contracts.configuration import canonical
from autofusex.fuzz.daemon.management.updates import verify_manifest,activate
from autofusex.fuzz.daemon.management.store import Store


def signed(tmp_path, wheels=None):
    key=Ed25519PrivateKey.generate()
    public=key.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw).hex()
    atomic(tmp_path/'trust.json',dict(release_keys={'test':public}))
    manifest=dict(release_id='0.2.0',protocol=1,manager_schema=1,wheels=[{}] if wheels is None else wheels)
    envelope=dict(manifest=manifest,key_id='test',signature=base64.b64encode(key.sign(canonical(manifest).encode())).decode())
    return envelope


def test_stage_accepts_signed_autofusex_wheel(tmp_path, monkeypatch):
    from autofusex.fuzz.daemon.management import updates
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


def test_launcher_failed_release_rolls_back_without_database_restore(tmp_path):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import time
    from autofusex.fuzz.daemon.runtime import request
    home=tmp_path/'home';store=Store(home)
    envelope=signed(home)
    identity='a'*64;directory=home/'tools'/identity;directory.mkdir(parents=True)
    broken=directory/'broken-python';broken.write_text('#!/bin/sh\nexit 42\n');broken.chmod(0o700)
    atomic(directory/'ready.json',dict(id=identity,envelope=envelope,python=str(broken)))
    child=subprocess.Popen([sys.executable,'-m','autofusex','daemon','supervise','--data-dir',str(home)],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    try:
        for _ in range(100):
            try:request(home,dict(action='status'));break
            except OSError:time.sleep(.05)
        else:raise AssertionError('launcher did not start')
        store.put('preserve-test',dict(identity='same',outbox=['pending-result']))
        assert activate(store,identity,{})['state']=='waiting_launcher'
        for _ in range(200):
            switch=store.get('tool_switch')
            if switch and switch['state']=='rolled_back':
                try:request(home,dict(action='status'));break
                except OSError:pass
            if child.poll() is not None:raise AssertionError(child.communicate())
            time.sleep(.05)
        else:raise AssertionError('rollback not observed')
        assert store.get('active_tool') is None
        assert store.get('preserve-test')['outbox']==['pending-result']
        request(home,dict(action='shutdown'));assert child.wait(timeout=10)==0
    finally:
        if child.poll() is None:
            try:request(home,dict(action='shutdown'))
            except (ValueError,OSError):pass
            child.terminate();child.wait(timeout=10)
