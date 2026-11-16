import copy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import uuid

import pytest

from autofusex.fuzz.client.storage import ResultRepository


def result():
    return {'seed':'14615016373650978276328604632813839116929039597610','workload_id':'a'*64,'oracle_id':'ref/1','backend':'test','status':'ok',
            'execution':{'backend':'test','device':'cpu','static_only':False},
            'evaluation':{'correctness':True,'performance':False},
            'shape_results':[{'shape_index':0,'correctness':'pass','performance':'not_requested'}]}


def test_round_trip_idempotency_and_conflict(tmp_path):
    repo = ResultRepository(tmp_path/'results.sqlite')
    identity = str(uuid.uuid4())
    assert repo.save(result(), attempt_id=identity) == identity
    assert repo.save(result(), attempt_id=identity) == identity
    assert len(repo.list()['items']) == 1
    changed = result(); changed['status'] = 'fail'
    with pytest.raises(ValueError, match='different content'):
        repo.save(changed, attempt_id=identity)
    assert repo.detail(identity)['shape_results'][0]['correctness'] == 'pass'
    with sqlite3.connect(repo.path) as conn:
        assert conn.execute('SELECT count(*) FROM contexts').fetchone()[0] == 3


def test_only_whitelisted_lightweight_fields_are_archived(tmp_path):
    repo = ResultRepository(tmp_path/'results.sqlite')
    value = result(); value.update(tensor='SECRET_TENSOR'*1000, raw_profiler='SECRET_PROFILER', full_log='SECRET_LOG')
    value['execution']['upload_key'] = 'SECRET_KEY'
    value['shape_results'][0]['error_summary'] = 'x'*800
    repo.save(value, environment={'python':'3.11','API_KEY':'SECRET_ENV'})
    with sqlite3.connect(repo.path) as conn:
        dump = '\n'.join(conn.iterdump())
    assert 'SECRET_' not in dump
    assert len(repo.detail(repo.list()['items'][0]['id'])['shape_results'][0]['error_summary']) == 500


def test_invalid_records_do_not_create_or_partially_write_db(tmp_path):
    repo = ResultRepository(tmp_path/'results.sqlite')
    value = result(); value['shape_results'] *= 2
    with pytest.raises(ValueError, match='duplicate'):
        repo.save(value)
    assert not repo.path.exists()
    repo.save(result())
    value = result(); value['shape_results'][0]['mean_wall_ns'] = float('nan')
    with pytest.raises(ValueError):
        repo.save(value)
    assert len(repo.list()['items']) == 1


def test_transaction_rolls_back_contexts_and_attempt_on_shape_failure(tmp_path):
    repo = ResultRepository(tmp_path/'results.sqlite'); repo.save(result())
    with sqlite3.connect(repo.path) as conn:
        conn.execute("CREATE TRIGGER reject_shape BEFORE INSERT ON shapes BEGIN SELECT RAISE(ABORT,'injected'); END")
    value = result(); value['execution']['device'] = 'new-device'
    with pytest.raises(sqlite3.DatabaseError):
        repo.save(value)
    with sqlite3.connect(repo.path) as conn:
        assert conn.execute('SELECT count(*) FROM attempts').fetchone()[0] == 1
        assert conn.execute('SELECT count(*) FROM contexts').fetchone()[0] == 3


def test_bounded_pagination_seed_filter_and_missing_detail(tmp_path):
    repo = ResultRepository(tmp_path/'results.sqlite')
    for _ in range(7): repo.save(result())
    first = repo.list(limit=3); second = repo.list(limit=3, after=first['next_cursor'])
    assert not {x['id'] for x in first['items']} & {x['id'] for x in second['items']}
    assert repo.list(seed='14615016373650978276328604632813839116929039597611')['items'] == []
    with pytest.raises(ValueError): repo.list(limit=201)
    with pytest.raises(ValueError): repo.detail('missing')


def test_snapshot_includes_committed_wal_excludes_secrets_and_uncommitted_changes(tmp_path):
    repo = ResultRepository(tmp_path/'results.sqlite'); identity = repo.save(result())
    with sqlite3.connect(repo.path) as writer:
        writer.execute('PRAGMA wal_autocheckpoint=0')
        writer.execute('CREATE TABLE secrets (value TEXT)')
        writer.execute("INSERT INTO secrets VALUES ('DO_NOT_EXPORT')")
        writer.commit()
        writer.execute("UPDATE attempts SET status='fail'")  # still uncommitted
        destination = tmp_path/'snapshot.sqlite'
        repo.snapshot(destination)
        writer.rollback()
    snapshot = ResultRepository(destination)
    assert snapshot.detail(identity)['status'] == 'ok'
    with sqlite3.connect(destination) as conn:
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='secrets'").fetchone() is None
        assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    assert 'DO_NOT_EXPORT' not in destination.read_bytes().decode('latin1')
    with pytest.raises(ValueError): repo.snapshot(destination)
    with pytest.raises(ValueError): repo.snapshot(repo.path)
    assert not list(tmp_path.glob('.fuzz-snapshot-*'))


def test_unknown_corrupt_or_unrelated_database_rejected_without_mutation(tmp_path):
    repo = ResultRepository(tmp_path/'results.sqlite'); repo.save(result())
    with sqlite3.connect(repo.path) as conn:
        conn.execute('PRAGMA user_version=99')
    with pytest.raises(ValueError, match='schema'): repo.save(result())
    with pytest.raises(ValueError, match='schema'): repo.list()
    unrelated = tmp_path/'other.sqlite'
    with sqlite3.connect(unrelated) as conn: conn.execute('CREATE TABLE personal (value TEXT)')
    before = unrelated.read_bytes()
    with pytest.raises(ValueError): ResultRepository(unrelated).save(result())
    assert unrelated.read_bytes() == before
    bad = tmp_path/'corrupt.sqlite'; bad.write_bytes(b'not SQLite')
    with pytest.raises(sqlite3.DatabaseError): ResultRepository(bad).list()


def test_read_only_view_does_not_write_snapshot(tmp_path):
    repo = ResultRepository(tmp_path/'results.sqlite'); repo.save(result())
    destination = tmp_path/'snapshot.sqlite'; repo.snapshot(destination)
    before = destination.read_bytes()
    assert ResultRepository(destination).list()['items']
    assert destination.read_bytes() == before


def test_storage_import_and_cli_query_need_no_execution_dependencies(tmp_path):
    repo = ResultRepository(tmp_path/'results.sqlite'); repo.save(result())
    code = '''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'torch','numpy','nnsmith','torch_npu'}: raise AssertionError(name)
sys.meta_path.insert(0,Block())
from autofusex.fuzz.client.cli import main
assert main(['maintenance', 'results',sys.argv[1]])==0
'''
    out = subprocess.run([sys.executable,'-c',code,str(repo.path)],capture_output=True,text=True)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)['items'][0]['seed'] == '14615016373650978276328604632813839116929039597610'


def test_concurrent_first_writers_publish_only_complete_schema(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    repo = ResultRepository(tmp_path/'results.sqlite')
    with ThreadPoolExecutor(max_workers=4) as pool:
        identities = list(pool.map(lambda _: repo.save(result()), range(8)))
    assert len(set(identities)) == 8
    assert len(repo.list()['items']) == 8
    assert not list(tmp_path.glob('.fuzz-init-*'))
