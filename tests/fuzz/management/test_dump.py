"""Explicit compiler dump policy across CLI, IPC and worker reuse."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from autofusex.client.modern_cli import parser,run,_default_task
from autofusex.contracts.execution import ExecutionConfig
from autofusex.workloads.fuzz.scheduling.supervisor import _debug_env


def test_dump_contract_and_inherited_environment(tmp_path):
    assert ExecutionConfig('cpu','cpu').dump is False
    assert _default_task()['execution']['dump'] is False
    with pytest.raises(ValueError):ExecutionConfig('cpu','cpu',dump=1)
    inherited=dict(TORCH_COMPILE_DEBUG='1',TORCH_COMPILE_DEBUG_DIR='/old',
        DUMP_GE_GRAPH='2',DUMP_GRAPH_LEVEL='3',DUMP_GRAPH_PATH='/old',
        AUTOFUSE_DFX_FLAGS='--mlir_codegen_config=fallback;--codegen_compile_debug=true;--debug_dir=/old;--keep=1')
    off=_debug_env(inherited,tmp_path,None,'af-mlir')
    assert off['TORCH_COMPILE_DEBUG']=='0'
    assert '--codegen_compile_debug=false' in off['AUTOFUSE_DFX_FLAGS']
    assert '--mlir_codegen_config=fallback' in off['AUTOFUSE_DFX_FLAGS']
    assert '--keep=1' in off['AUTOFUSE_DFX_FLAGS']
    assert '--debug_dir=' not in off['AUTOFUSE_DFX_FLAGS']
    assert all(key not in off for key in ('DUMP_GE_GRAPH','DUMP_GRAPH_LEVEL','DUMP_GRAPH_PATH','TORCH_COMPILE_DEBUG_DIR'))
    on=_debug_env(inherited,tmp_path,tmp_path/'debug','af-legacy',True)
    assert on['TORCH_COMPILE_DEBUG']=='1'
    assert '--codegen_compile_debug=true' in on['AUTOFUSE_DFX_FLAGS']
    assert '--debug_dir='+str(tmp_path/'debug') in on['AUTOFUSE_DFX_FLAGS']
    assert inherited['TORCH_COMPILE_DEBUG_DIR']=='/old'


@pytest.mark.parametrize('command',['start','restart','replay'])
def test_modern_cli_help_and_flag(command):
    argv=[command]+(['seed'] if command=='replay' else [])
    assert parser().parse_args([*argv,'--dump']).dump is True
    assert not parser().parse_args(argv).dump


@pytest.mark.parametrize('enabled',[False,True])
def test_start_flag_crosses_ipc_without_changing_profile(tmp_path,monkeypatch,enabled):
    import autofusex.client.modern_cli as cli
    import autofusex.client.daemon as daemon
    prefix=['--data-dir',str(tmp_path)]
    run([*prefix,'init','--local'])
    original=(tmp_path/'task.json').read_text()
    monkeypatch.setattr(cli,'_legacy',lambda argv: {})
    calls=[]
    monkeypatch.setattr(daemon,'request',lambda home,body:calls.append(body) or {})
    run([*prefix,'start',*(['--dump'] if enabled else [])])
    assert calls[-1]==dict(action='start_session',dump=True if enabled else None)
    assert (tmp_path/'task.json').read_text()==original


def test_session_override_is_ephemeral(tmp_path,monkeypatch):
    from autofusex.client.local_session import start
    from autofusex.client.store import Store
    import autofusex.client.tasks as tasks
    store=Store(tmp_path)
    spec=_default_task();spec['results']['upload']=False
    spec['execution']['devices']=['npu:0']
    (tmp_path/'task.json').write_text(json.dumps(spec))
    (tmp_path/'client.json').write_text(json.dumps(dict(mode='local')))
    captured=[]
    monkeypatch.setattr(tasks,'start',lambda store,spec,*args:captured.append(spec) or dict(run_id='x'))
    manager=SimpleNamespace(store=store,processes={})
    start(manager,dump=True)
    assert captured[-1]['execution']['dump'] is True
    assert json.loads((tmp_path/'task.json').read_text())['execution']['dump'] is False
    start(manager)
    assert captured[-1]['execution']['dump'] is False


def test_pool_separates_dump_modes_and_creates_no_default_debug_tree(tmp_path,monkeypatch):
    from autofusex.workloads.fuzz.scheduling.pool import Pool
    from autofusex.execution.scheduling.budgets import Budget
    monkeypatch.setenv('AFX_TASK_HOME',str(tmp_path))
    pool=Pool(Budget(cpu_workers=1),tmp_path,0,[],max_tasks=10)
    try:
        off=pool._worker(('af-mlir','npu:0',1,False))
        assert off['debug_lease'] is None and not (tmp_path/'debug').exists()
        on=pool._worker(('af-mlir','npu:0',1,True))
        assert on is not off and on['debug_lease'] is not None
        assert list((tmp_path/'debug/autofuse_compile').glob('worker-*'))
        off_again=pool._worker(('af-mlir','npu:0',1,False))
        assert off_again is not on and off_again['debug_lease'] is None
    finally:pool.close()


def test_replay_flag_applies_to_active_and_queued_runs(tmp_path):
    import sqlite3
    from autofusex.client.replay import submit
    from autofusex.client.store import Store
    from autofusex.execution.scheduling.budgets import Budget,write_policy
    from autofusex.workloads.fuzz.scheduling.ledger import TaskLedger
    from autofusex.workloads.fuzz.generation.sampling import SamplingConfig
    store=Store(tmp_path);directory=tmp_path/'runs/one';directory.mkdir(parents=True)
    spec=_default_task();spec['execution'].update(backends=['cpu'],dump=True)
    prepared=dict(spec=spec,environment=dict(id='local'))
    with store.connect() as db:db.execute('INSERT INTO runs VALUES (?,?,?,?,?)',('one',json.dumps(prepared),'running',0,None))
    seed=str(SamplingConfig().seed(7))
    with sqlite3.connect(directory/'results.sqlite') as db:
        db.execute('CREATE TABLE attempts(seed TEXT)');db.execute('INSERT INTO attempts VALUES (?)',(seed,))
    ledger=TaskLedger(directory/'tasks.sqlite');write_policy(ledger,Budget())
    for enabled in (False,True):
        result=submit(store,[seed],{'one':object()},dump=enabled)
        assert json.loads(ledger.get(result['task_ids'][0])['execution'])['dump'] is enabled
        queued=submit(store,[seed],{},dump=enabled)
        assert store.operation(queued['operation_ids'][0])['payload']['execution']['dump'] is enabled


def test_wire_compatibility_hides_dump_from_older_clients():
    from autofusex.server.task_config import for_client
    result=dict(config=_default_task(),revision='old',source='personal')
    assert 'dump' not in for_client(result,'cleanup-v1')['config']['execution']
    assert for_client(result,'cleanup-v1,dump-v1')['config']['execution']['dump'] is False
    assert 'dump' in result['config']['execution']


@pytest.mark.parametrize('dump',[False,True])
@pytest.mark.parametrize('scheduling',['combined','staged'])
def test_real_compilation_respects_dump_and_records_mode(tmp_path,monkeypatch,dump,scheduling):
    import os
    if __import__('sys').platform!='linux':pytest.skip('execution validation uses Linux')
    pytest.importorskip('torch')
    from autofusex.contracts.execution import EvaluationConfig
    from autofusex.workloads.fuzz.generation.sampling import SamplingConfig
    from autofusex.workloads.fuzz.scheduling.ledger import TaskLedger
    from autofusex.workloads.fuzz.scheduling.supervisor import drain
    from autofusex.execution.scheduling.budgets import Budget,write_policy
    from autofusex.execution.storage.queries import ResultQueries
    home=tmp_path/'home';queue=home/'tasks/tasks.sqlite'
    queue.parent.mkdir(parents=True)
    from autofusex.client.paths import setup
    setup(home)
    monkeypatch.setenv('AFX_TASK_HOME',str(home))
    monkeypatch.setenv('TORCH_COMPILE_DEBUG','1') # default off must override inherited on
    monkeypatch.setenv('TORCH_COMPILE_DEBUG_DIR',str(tmp_path/'unexpected'))
    ledger=TaskLedger(queue)
    seed=SamplingConfig.profile('small',max_nodes=1,dtype_mode='float32').seed(3)
    identity=ledger.enqueue(str(seed),ExecutionConfig('cpu-compile','cpu',1,scheduling=scheduling,static_only=True,dump=dump),EvaluationConfig())
    write_policy(ledger,Budget(cpu_workers=1,compile_threads=1))
    database=home/'results/results.sqlite'
    result=drain(ledger,database,timeout=120)
    row=ledger.get(identity)
    assert result['completed']==1 and row['state']=='completed'
    saved=ResultQueries(database).detail(row['attempt_id'])
    assert saved['status']=='ok',saved
    assert saved['contexts']['execution']['dump'] is dump
    files=[p for p in (home/'debug').rglob('*') if p.is_file() and p.name!='.active.lock']
    assert bool(files) is dump
    assert not (tmp_path/'unexpected').exists()


def test_fuzz_and_task_cli_expose_dump():
    import argparse
    from autofusex.workloads.fuzz.scheduling.cli import register,_CONTINUE_CONFLICTS
    from autofusex.workloads.fuzz.cli import replay_parser
    from autofusex.client.cli import parser as task_parser
    root=argparse.ArgumentParser();sub=root.add_subparsers(dest='command');register(sub,sub)
    assert root.parse_args(['run','--dump']).dump is True
    assert root.parse_args(['run']).dump is False
    assert replay_parser().parse_args(['seed','--dump']).dump is True
    assert task_parser().parse_args(['task','submit','task.yaml','--dump']).dump is True
    assert task_parser().parse_args(['task','replay','seed','--from-run','run','--dump']).dump is True
    assert _CONTINUE_CONFLICTS['dump'] is False
