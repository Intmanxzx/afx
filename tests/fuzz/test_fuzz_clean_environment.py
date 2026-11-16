"""Fresh-only boundaries and identity-only restoration."""
from contextlib import closing
import json
import sqlite3
import pytest
from autofusex.fuzz.contracts import Seed
from autofusex.fuzz.storage.repository import _connect, ResultRepository
from autofusex.fuzz.central.repository import create_user, initialize, authenticate

@pytest.mark.parametrize('token', ['C16','D16','C2-16','D2-16',str(2*2**128+42)])
def test_old_seeds_rejected_before_any_result_write(tmp_path, token):
    with pytest.raises(ValueError): Seed.parse(token)
    repo=ResultRepository(tmp_path/'results')
    with pytest.raises(ValueError):
        repo.save(dict(seed=token,workload_id='a'*64,oracle_id='test',backend='cpu',status='ok'))
    assert not repo.path.exists()

def test_old_database_is_never_opened_or_upgraded(tmp_path):
    path=tmp_path/'results'
    with closing(_connect(path,writable=True)) as conn,conn:
        conn.execute('PRAGMA user_version=1')
    before=path.read_bytes()
    for writable in (False,True):
        with pytest.raises(ValueError,match='unsupported result schema'):
            _connect(path,writable=writable)
    assert path.read_bytes()==before

def test_only_user_rows_preserve_token_in_empty_center(tmp_path):
    old,new=tmp_path/'old',tmp_path/'new'
    user=create_user(old,'Alice')
    initialize(new)
    with closing(_connect(old)) as source, closing(_connect(new,writable=True)) as target,target:
        target.executemany('INSERT INTO users VALUES (?,?,?,?)',source.execute('SELECT id,display_name,key_hash,active FROM users'))
        assert authenticate(target,user['token'])==user['user_id']
        for table in ('attempts','sources','receipts','owners','deleted_attempts'):
            assert target.execute('SELECT count(*) FROM '+table).fetchone()[0]==0

def test_old_telemetry_and_backend_aliases_rejected():
    from autofusex.fuzz.telemetry import validate
    from autofusex.fuzz.backends import create_backend
    from autofusex.fuzz.backends.base import UnsupportedBackend
    from autofusex.fuzz.contracts import ExecutionConfig
    with pytest.raises(ValueError): validate(json.dumps(dict(version=1,stages={},shapes={})))
    for backend in ('cpu-eager','npu-eager','ascendc','triton'):
        with pytest.raises(UnsupportedBackend): create_backend(ExecutionConfig(backend,'cpu'))
