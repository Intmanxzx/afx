import io
import json
import uuid
import pytest
from autofusedebugger.fuzz.cli import main
from autofusedebugger.fuzz.onboarding import initialize_user


def test_top_level_is_small(capsys):
    with pytest.raises(SystemExit) as exc:main(['--help'])
    assert exc.value.code==0
    text=capsys.readouterr().out
    assert all(title in text for title in ['客户端','云端','本地查看','维护'])
    assert all(command in text for command in ['setup --token','run [SEED','replay SEED','admin user','view RESULTS'])
    assert 'admin-user' not in text and 'check-db' not in text


def test_token_setup_keeps_secret_private(tmp_path):
    token='test-token-not-a-real-secret-12345'
    identity={'user_id':str(uuid.uuid4()),'display_name':'niuyuhu'}
    def opener(request,**kw):return io.BytesIO(json.dumps(identity).encode())
    result=initialize_user(tmp_path,token,opener=opener,start_sync=False)
    assert token not in json.dumps(result)
    config=json.loads((tmp_path/'config.json').read_text())
    assert token not in json.dumps(config)
    assert config['identity']['user_id']==identity['user_id']
    assert (tmp_path/'upload.token').stat().st_mode & 0o077==0
    assert result['resources']['compile_threads']>=1
    assert result['user']=='niuyuhu'
    assert (tmp_path/'upload.token').read_text().strip()==token


def test_setup_help_has_only_person_token_and_interval(capsys):
    with pytest.raises(SystemExit):main(['setup','--help'])
    text=capsys.readouterr().out
    assert '--user' not in text and '--token ' in text
    for removed in ['--user-id','--key-file','--sync-endpoint','--cpu-workers','--compile-threads','--memory-budget-bytes','--cache-budget-bytes']:
        assert removed not in text


def test_stop_defaults_to_all_tasks(tmp_path,monkeypatch):
    from autofusedebugger.fuzz.scheduling.ledger import TaskLedger
    from autofusedebugger.fuzz.contracts import ExecutionConfig,EvaluationConfig
    monkeypatch.setenv('AUTOFUSEDBG_HOME',str(tmp_path))
    ledger=TaskLedger(tmp_path/'tasks/tasks.sqlite')
    ids=[ledger.enqueue(str(2*2**128+i),ExecutionConfig('cpu','cpu',1),EvaluationConfig()) for i in range(3)]
    assert main(['stop'])==0
    assert all(ledger.get(i)['state']=='cancelled' for i in ids)


def test_admin_rotation_is_by_name(capsys):
    with pytest.raises(SystemExit):main(['admin','rotate-token','--help'])
    text=capsys.readouterr().out
    assert 'user_id' not in text


def test_admin_issues_token_once_without_file(tmp_path,capsys):
    from autofusedebugger.fuzz.central.repository import authenticate
    from autofusedebugger.fuzz.storage.queries import reading
    db=tmp_path/'server.sqlite'
    assert main(['admin','user',str(db),'niuyuhu'])==0
    result=json.loads(capsys.readouterr().out)
    assert result['display_name']=='niuyuhu' and len(result['token'])>=40
    with reading(db) as conn:
        assert authenticate(conn,result['token'])==result['user_id']
        assert conn.execute('SELECT key_hash FROM users').fetchone()[0]!=result['token']
    assert not list(tmp_path.glob('*.token'))


def test_admin_rotates_token_without_file(tmp_path,capsys):
    from autofusedebugger.fuzz.central.repository import create_user,authenticate
    from autofusedebugger.fuzz.storage.queries import reading
    db=tmp_path/'server.sqlite';old=create_user(db,'niuyuhu')
    assert main(['admin','rotate-token',str(db),'niuyuhu'])==0
    new=json.loads(capsys.readouterr().out)
    with reading(db) as conn:
        assert authenticate(conn,new['token'])==old['user_id']
        with pytest.raises(PermissionError):authenticate(conn,old['token'])


@pytest.mark.parametrize('args', [
    ['setup'], ['run'], ['replay'], ['status'], ['stop'], ['view'],
    ['admin', 'user'], ['admin', 'rotate-token'], ['admin', 'serve'], ['admin', 'backup'],
    ['maintenance'], ['maintenance', 'snapshot'], ['maintenance', 'restore'],
    ['maintenance', 'sync'],
])
def test_documented_command_help_is_reachable(args, capsys):
    with pytest.raises(SystemExit) as exc:
        main([*args, '-h'])
    assert exc.value.code == 0
    assert 'usage:' in capsys.readouterr().out


def test_short_flags_and_explicit_save(capsys):
    from autofusedebugger.fuzz.cli import replay_parser
    args = replay_parser().parse_args(['B16', '--perf'])
    assert args.performance and not args.save
    assert replay_parser().parse_args(['B16', '--save']).save
    with pytest.raises(SystemExit): main(['run', '-h'])
    text = capsys.readouterr().out
    assert '--fg' in text and '--perf' in text and '--save' in text
    assert '--foreground' not in text and '--performance' not in text
    assert 'continuous until fuzz stop' in ' '.join(text.split())


def test_sync_uses_remembered_home(tmp_path, monkeypatch, capsys):
    from autofusedebugger.fuzz import config
    monkeypatch.delenv('AUTOFUSEDBG_HOME', raising=False)
    monkeypatch.setattr(config, 'default_home', lambda: tmp_path/'default')
    custom = tmp_path/'custom'
    config.remember_home(custom)
    assert config.data_home() == custom
    monkeypatch.setattr('autofusedebugger.fuzz.sync.client.status', lambda home, database: {'directory': str(home)})
    assert main(['maintenance', 'sync', 'status']) == 0
    assert json.loads(capsys.readouterr().out)['directory'] == str(custom)
    monkeypatch.setenv('AUTOFUSEDBG_HOME', str(tmp_path/'env'))
    assert config.data_home() == tmp_path/'env'
    assert config.data_home(tmp_path/'explicit') == tmp_path/'explicit'


def test_run_defaults_to_infinite_unsaved_and_save_is_per_task(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('AUTOFUSEDBG_HOME', str(tmp_path))
    observed = []
    def launch(ledger, database, specifications, **kwargs):
        observed.append(kwargs)
        return {'state': 'test'}
    monkeypatch.setattr('autofusedebugger.fuzz.scheduling.hosting.launch', launch)
    assert main(['run', '--fg']) == 0
    configs, evaluation, count, devices = observed[-1]['campaign']
    assert count is None and observed[-1]['foreground']
    assert all(not c.save_results for c in configs)
    assert main(['run', '--count', '3', '--save']) == 0
    configs, _, count, devices = observed[-1]['campaign']
    assert count == 3 and all(c.save_results for c in configs)
