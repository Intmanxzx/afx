import json
import platform
import sys
from pathlib import Path
import pytest
from autofusedebugger.management.store import Store
from autofusedebugger.management.artifacts import atomic
from autofusedebugger.management import environments


def setup(tmp_path):
    store=Store(tmp_path/'home');source=tmp_path/'source';source.mkdir()
    (source/'dirty.txt').write_text('uncommitted local work')
    script=source/'build.py'
    script.write_text('import pathlib,sys;pathlib.Path(sys.argv[1],"test.run").write_text("fixture-run")')
    trust={'profiles':{'dev':{'kind':'build','source_roots':[str(source)],'argv':[sys.executable,str(script),'{output}'],
            'snapshot_inputs':['build.py','dirty.txt'],'outputs':[{'path':'test.run','kind':'toolkit','adapter':'fake'}]},
            'fake':{'kind':'cann-installer','package_kinds':['toolkit'],'isolated_install':True,'argv':['true'],'verify_argv':['true']}},
           'compatibility':{'test':{'python':'3.11','architecture':platform.machine(),'packages':{'torch':'test','torch_npu':'test','nnsmith':'test'},
                            'python_executable':sys.executable,'required_free_bytes':1,'wheels':[]}},
           'install_roots':[str(tmp_path/'Ascend')],'existing_roots':[str(tmp_path/'Ascend')],'approved_artifacts':[]}
    atomic(store.home/'trust.json',trust)
    return store,source,trust


def test_build_uses_fresh_output_preserves_source(tmp_path):
    store,source,trust=setup(tmp_path)
    result=environments.build_run(store,dict(profile='dev',source_dir=str(source)), 'op')
    assert result['packages'][0]['sha256']
    assert (source/'dirty.txt').read_text()=='uncommitted local work'
    assert Path(result['packages'][0]['path']).is_relative_to(store.home/'builds')
    again=environments.build_run(store,dict(profile='dev',source_dir=str(source)), 'op')
    assert again['build_id']!=result['build_id']


def test_build_confines_sources_and_products(tmp_path):
    store,source,trust=setup(tmp_path)
    with pytest.raises(PermissionError):environments.build_run(store,dict(profile='dev',source_dir='/'), 'op')
    trust['profiles']['dev']['outputs'][0]['path']='../../not-output.run'
    atomic(store.home/'trust.json',trust)
    with pytest.raises(PermissionError):environments.build_run(store,dict(profile='dev',source_dir=str(source)), 'op')


def test_snapshot_and_missing_manifest(tmp_path):
    store,source,trust=setup(tmp_path)
    result=environments.build_run(store,dict(profile='dev',source_dir=str(source),mode='snapshot'), 'op')
    assert Path(result['source'])!=source
    assert (Path(result['source'])/'dirty.txt').read_text()=='uncommitted local work'
    trust['profiles']['dev']['snapshot_inputs']=[];atomic(store.home/'trust.json',trust)
    with pytest.raises(ValueError):environments.build_run(store,dict(profile='dev',source_dir=str(source),mode='snapshot'), 'op')


def test_environment_plan_requires_local_authority(tmp_path):
    store,source,trust=setup(tmp_path)
    spec=dict(schema_version=1,kind='environment',name='dev',python='3.11',compatibility_profile='test',
              packages=trust['compatibility']['test']['packages'],cann=dict(source='build',install_root=str(tmp_path/'Ascend'),build=dict(source_dir=str(source),profile='dev')))
    result=environments.plan(store,spec)
    assert result['status']=='planned'
    trust['compatibility']['test']['packages']['torch']='changed';atomic(store.home/'trust.json',trust)
    with pytest.raises(ValueError):environments.apply(store,result['id'],'operation')


def test_install_failure_never_ready(tmp_path,monkeypatch):
    store,source,trust=setup(tmp_path)
    spec=dict(schema_version=1,kind='environment',name='dev',python='3.11',compatibility_profile='test',
              packages=trust['compatibility']['test']['packages'],cann=dict(source='build',install_root=str(tmp_path/'Ascend'),build=dict(source_dir=str(source),profile='dev')))
    result=environments.plan(store,spec)
    original=environments.command
    def fail(argv,**kwargs):
        if 'venv' in argv:raise RuntimeError('fixture failure')
        return original(argv,**kwargs)
    monkeypatch.setattr(environments,'command',fail)
    with pytest.raises(RuntimeError):environments.apply(store,result['id'],'operation')
    with store.connect() as db:
        assert db.execute('SELECT state FROM environments').fetchone()[0]=='needs_attention'
