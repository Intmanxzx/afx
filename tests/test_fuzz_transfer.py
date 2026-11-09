import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import uuid

import pytest

from autofusedebugger.fuzz.storage import ResultRepository
from autofusedebugger.fuzz.storage.transfer import check_database, merge, restore


def record(seed='42'):
    return {'seed': seed, 'workload_id': 'a'*64, 'oracle_id': 'ref/1',
            'backend': 'test', 'status': 'ok',
            'execution': {'backend': 'test', 'device': 'cpu'},
            'shape_results': [{'shape_index': i, 'correctness': 'pass',
                               'performance': 'not_requested'} for i in (2, 0, 1)]}


@pytest.fixture
def source(tmp_path):
    repo = ResultRepository(tmp_path/'source.sqlite')
    repo.save(record())
    return repo


def rows(path):
    with sqlite3.connect(path) as conn:
        return {table: conn.execute(f'SELECT * FROM {table} ORDER BY rowid').fetchall()
                for table in ('contexts', 'attempts', 'shapes')}


def test_restore_merge_roundtrip_and_exact_idempotence(source, tmp_path):
    snapshot = Path(source.snapshot(tmp_path/'snapshot.sqlite'))
    before = snapshot.read_bytes()
    destination = tmp_path/'restored.sqlite'
    assert restore(snapshot, destination)['added'] == 1
    assert rows(destination) == rows(snapshot)
    assert check_database(destination)['attempts'] == 1
    assert merge(snapshot, destination) == {'added': 0, 'skipped': 1}
    assert snapshot.read_bytes() == before
    # Same seed, distinct attempt, not silently deduplicated by seed.
    source.save(record())
    assert merge(source.path, destination) == {'added': 1, 'skipped': 1}
    assert check_database(destination)['attempts'] == 2


def test_restore_excludes_private_tables_and_unreferenced_contexts(source, tmp_path):
    with sqlite3.connect(source.path) as conn:
        conn.execute('CREATE TABLE secrets(value TEXT)')
        conn.execute("INSERT INTO secrets VALUES ('secret')")
        conn.execute("INSERT INTO contexts VALUES ('unreferenced', 'environment', '{}')")
    target = tmp_path/'target.sqlite'
    restore(source.path, target)
    with sqlite3.connect(target) as conn:
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='secrets'").fetchone() is None
        assert conn.execute('SELECT count(*) FROM contexts').fetchone()[0] == 3


@pytest.mark.parametrize('mutation', [
    "UPDATE attempts SET result_hash='forged'",
    "UPDATE attempts SET seed='042'",
    "UPDATE attempts SET created_at='yesterday'",
    "UPDATE attempts SET status='other'",
    "UPDATE attempts SET error_summary='changed'",
    "UPDATE shapes SET correctness='fail'",
    "UPDATE contexts SET payload='{}' WHERE kind='execution'",
    "UPDATE contexts SET kind='environment' WHERE kind='execution'",
    "DELETE FROM contexts WHERE kind='execution'",
])
def test_corrupt_contract_rejected_without_publication(source, tmp_path, mutation):
    with sqlite3.connect(source.path) as conn:
        conn.execute(mutation)
    with pytest.raises((ValueError, sqlite3.Error)):
        restore(source.path, tmp_path/'target.sqlite')
    assert not (tmp_path/'target.sqlite').exists()
    assert not list(tmp_path.glob('.fuzz-restore-*'))


def test_conflict_rolls_back_prior_inserts(source, tmp_path):
    # Insert a new row before the conflicting one to exercise transaction rollback.
    other = ResultRepository(tmp_path/'other.sqlite')
    other.save(record('43'))
    identity = source.list()['items'][0]['id']
    other.save(record('99'), attempt_id=identity)
    before = rows(source.path)
    with pytest.raises(ValueError, match='conflict'):
        merge(other.path, source.path)
    assert rows(source.path) == before


def test_tampered_target_hash_is_not_accepted_as_idempotent(source, tmp_path):
    target = tmp_path/'target.sqlite'
    restore(source.path, target)
    with sqlite3.connect(target) as conn:
        conn.execute("UPDATE attempts SET backend='tampered'")
    with pytest.raises(ValueError, match='fingerprint'):
        merge(source.path, target)


def test_disk_write_failure_rolls_back_and_cleans(source, tmp_path, monkeypatch):
    from autofusedebugger.fuzz.storage import transfer
    original = transfer._copy
    def fail(src, dst):
        original(src, dst)
        raise sqlite3.OperationalError('injected disk full')
    monkeypatch.setattr(transfer, '_copy', fail)
    target = tmp_path/'target.sqlite'
    with pytest.raises(sqlite3.OperationalError, match='disk full'):
        restore(source.path, target)
    assert not target.exists() and not list(tmp_path.glob('.fuzz-restore-*'))
    existing = ResultRepository(tmp_path/'existing.sqlite')
    existing.save(record('999'))
    before = rows(existing.path)
    with pytest.raises(sqlite3.OperationalError):
        merge(source.path, existing.path)
    assert rows(existing.path) == before


def test_atomic_publication_failure_and_no_overwrite(source, tmp_path, monkeypatch):
    from autofusedebugger.fuzz.storage import transfer
    target = tmp_path/'target.sqlite'
    def raced_link(src, dst):
        target.write_bytes(b'another writer won')
        raise FileExistsError(str(dst))
    monkeypatch.setattr(transfer.os, 'link', raced_link)
    with pytest.raises(FileExistsError):
        restore(source.path, target)
    assert target.read_bytes() == b'another writer won'
    assert not list(tmp_path.glob('.fuzz-restore-*'))
    with pytest.raises(ValueError, match='new file'):
        restore(source.path, target)


@pytest.mark.parametrize('version', [0, 2, 999])
def test_unknown_schema_never_migrated_or_rewritten(source, version):
    with sqlite3.connect(source.path) as conn:
        conn.execute(f'PRAGMA user_version={version}')
    before = source.path.read_bytes()
    with pytest.raises(ValueError, match='unsupported result schema'):
        check_database(source.path)
    assert source.path.read_bytes() == before


def test_missing_merge_destination_and_same_source_are_rejected(source, tmp_path):
    target = tmp_path/'missing.sqlite'
    with pytest.raises(sqlite3.Error):
        merge(source.path, target)
    assert not target.exists()
    with pytest.raises(ValueError, match='differ'):
        merge(source.path, source.path)


def test_destination_trigger_never_executes(source, tmp_path):
    target = tmp_path/'target.sqlite'
    restore(source.path, target)
    with sqlite3.connect(target) as conn:
        conn.executescript('CREATE TABLE effects(value); CREATE TRIGGER surprise AFTER INSERT ON attempts BEGIN INSERT INTO effects VALUES (1); END;')
    source.save(record('44'))
    with pytest.raises(ValueError, match='triggers'):
        merge(source.path, target)
    with sqlite3.connect(target) as conn:
        assert conn.execute('SELECT * FROM effects').fetchall() == []


def test_transfer_cli_without_execution_dependencies(source, tmp_path):
    code = '''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'torch','numpy','nnsmith','torch_npu'}: raise AssertionError(name)
sys.meta_path.insert(0,Block())
from autofusedebugger.fuzz.cli import main
assert main(['restore',sys.argv[1],sys.argv[2]])==0
assert main(['import',sys.argv[1],sys.argv[2]])==0
assert main(['check-db',sys.argv[2]])==0
'''
    result = subprocess.run([sys.executable, '-c', code, str(source.path), str(tmp_path/'new.sqlite')],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '"skipped": 1' in result.stdout


def test_killed_import_rolls_back_transaction(source, tmp_path):
    # Kill after actual SQL writes, before commit; SQLite must recover on reopen.
    target = tmp_path/'target.sqlite'
    restore(source.path, target)
    source.save(record('43'))
    before = rows(target)
    code = '''
import os,sys
from autofusedebugger.fuzz.storage import transfer
original=transfer._copy
def crash(src,dst):
 original(src,dst)
 os._exit(77)
transfer._copy=crash
transfer.merge(sys.argv[1],sys.argv[2])
'''
    child = subprocess.run([sys.executable, '-c', code, str(source.path), str(target)])
    assert child.returncode == 77
    assert rows(target) == before
    assert merge(source.path, target) == {'added': 1, 'skipped': 1}


def test_restore_wal_ignores_uncommitted_source(source, tmp_path):
    with sqlite3.connect(source.path) as writer:
        writer.execute("UPDATE attempts SET status='fail'")
        restore(source.path, tmp_path/'target.sqlite')
        writer.rollback()
    assert check_database(tmp_path/'target.sqlite')['attempts'] == 1
    assert ResultRepository(tmp_path/'target.sqlite').list()['items'][0]['status'] == 'ok'
