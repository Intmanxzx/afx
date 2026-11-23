import pytest
from autofusex.workloads.imported.registry import register, inspect, list_cases


def test_registration_is_immutable_and_does_not_execute(tmp_path):
    source = tmp_path / 'case.py'
    source.write_text('raise RuntimeError("must not execute at import")')
    home = tmp_path / 'home'
    original = register(home, source, entry='create_case')
    assert register(home, source, entry='create_case') == original
    source.write_text('print("new revision")')
    changed = register(home, source, entry='create_case')
    assert changed['revision'] != original['revision']
    assert inspect(home, original['case_key']) == original
    assert len(list_cases(home)) == 2
    saved = home / 'cases' / original['revision'] / 'source/case.py'
    saved.write_text('tampered')
    with pytest.raises(ValueError, match='checksum'):
        inspect(home, original['case_key'])


def test_package_rejects_escaping_entry_and_symlink(tmp_path):
    source = tmp_path / 'package'; source.mkdir()
    (tmp_path / 'outside.py').write_text('')
    with pytest.raises(ValueError): register(tmp_path/'home', source, script='../outside.py')
    (source/'case.py').write_text('')
    (source/'link').symlink_to(tmp_path/'outside.py')
    with pytest.raises(ValueError, match='symlink'): register(tmp_path/'home', source)


def test_fixed_task_contract_has_no_seed_dependency(tmp_path):
    import subprocess, sys
    code = '''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.startswith('autofusex.workloads.fuzz') or name.split('.')[0] in {'torch','nnsmith','numpy'}: raise AssertionError(name)
sys.meta_path.insert(0,Block())
from autofusex.client.configuration import validate
value=validate(dict(schema_version=1,kind='imported_task',environment='local',workload=dict(case_key='I:'+'a'*64,revision='a'*64)))
assert value['execution']['backends']==['af-legacy']
assert 'seed' not in value['workload']
policy=validate(dict(schema_version=1,kind='client_policy'))
assert policy['resources']['max_workers']==1
'''
    result = subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr


def test_portable_case_roundtrip_and_path_validation(tmp_path):
    from autofusex.workloads.imported.registry import export, receive
    from autofusex.contracts.source_cases import validate
    source=tmp_path/'case.py';source.write_text('print("portable")')
    case=register(tmp_path/'one',source)
    bundle=export(tmp_path/'one',case['case_key'])
    assert receive(tmp_path/'two',bundle)==case
    bundle['files']['case.py']='invalid base64'
    with pytest.raises(ValueError):validate(bundle['case'],bundle['files'])
