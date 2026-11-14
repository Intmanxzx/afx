"""Real venv/pip and fake RUN execution inside Docker; no CANN/NNSmith install."""
import hashlib
import platform
import sys
import zipfile
from pathlib import Path
from autofusex.management.store import Store
from autofusex.management.artifacts import atomic,sha256
from autofusex.management import environments


def wheel(path):
    with zipfile.ZipFile(path,'w') as out:
        out.writestr('managed_fixture/__init__.py','VALUE = 42\n')
        out.writestr('managed_fixture-1.0.dist-info/METADATA','Metadata-Version: 2.1\nName: managed-fixture\nVersion: 1.0\n')
        out.writestr('managed_fixture-1.0.dist-info/WHEEL','Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
        out.writestr('managed_fixture-1.0.dist-info/RECORD','')


def test_real_venv_pip_and_fake_run_install(tmp_path,monkeypatch):
    store=Store(tmp_path/'home')
    fake=tmp_path/'fake.run';fake.write_text('#!/bin/sh\nset -eu\nmkdir -p "$1"\nprintf fixture > "$1/installed.txt"\n')
    whl=tmp_path/'managed_fixture-1.0-py3-none-any.whl';wheel(whl)
    packages={'torch':'fixture','torch_npu':'fixture','nnsmith':'fixture'}
    trust=dict(profiles={'fixture':dict(kind='cann-installer',package_kinds=['toolkit'],isolated_install=True,
              argv=['/bin/sh','{artifact}','{prefix}'],verify_argv=[sys.executable,'-c','import pathlib,sys;assert pathlib.Path(sys.argv[1],"installed.txt").read_text()=="fixture"','{prefix}'])},
              approved_artifacts=[sha256(fake)],install_roots=[str(tmp_path/'Ascend')],
              compatibility={'fixture':dict(python='3.11',packages=packages,architecture=platform.machine(),python_executable=sys.executable,required_free_bytes=1,
                 wheels=[dict(name='managed-fixture',version='1.0',filename=whl.name,url='https://fixture.invalid/fixture.whl',sha256=sha256(whl))],
                 verify_code='import managed_fixture; assert managed_fixture.VALUE == 42')})
    atomic(store.home/'trust.json',trust)
    def local_transport(url,expected,*args,**kwargs):
        path=whl if url.endswith('.whl') else fake
        assert sha256(path)==expected
        return path
    monkeypatch.setattr(environments,'download',local_transport)
    spec=dict(schema_version=1,kind='environment',name='fixture',python='3.11',compatibility_profile='fixture',packages=packages,
              cann=dict(source='download',install_root=str(tmp_path/'Ascend'),packages=[dict(kind='toolkit',url='https://fixture.invalid/fake.run',sha256=sha256(fake),adapter='fixture')]))
    plan=environments.plan(store,spec)
    result=environments.apply(store,plan['id'],'operation')
    assert result['npu_verified'] is False
    assert Path(result['python']).is_file()
    assert (Path(result['cann_path'])/'installed.txt').read_text()=='fixture'
    assert environments.resolve(store,'fixture')['id']==result['id']
    assert not (Path.home()/'Ascend'/'installed.txt').exists()
