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
    assert '{replay,maintenance,admin,setup,view,run,status,stop}' in text
    assert 'admin-user' not in text and 'check-db' not in text


def test_token_setup_keeps_secret_private(tmp_path):
    token='test-token-not-a-real-secret-12345'
    identity={'user_id':str(uuid.uuid4()),'display_name':'niuyuhu'}
    def opener(request,**kw):return io.BytesIO(json.dumps(identity).encode())
    result=initialize_user(tmp_path,'niuyuhu',token,opener=opener,start_sync=False)
    assert token not in json.dumps(result)
    config=json.loads((tmp_path/'config.json').read_text())
    assert token not in json.dumps(config)
    assert config['identity']['user_id']==identity['user_id']
    assert (tmp_path/'upload.token').stat().st_mode & 0o077==0
    assert result['resources']['compile_threads']>=1
    with pytest.raises(ValueError,match='does not match'):
        initialize_user(tmp_path,'other',token,opener=opener,start_sync=False)
    assert (tmp_path/'upload.token').read_text().strip()==token


def test_setup_help_has_only_person_token_and_interval(capsys):
    with pytest.raises(SystemExit):main(['setup','--help'])
    text=capsys.readouterr().out
    assert '--user ' in text and '--token ' in text
    for removed in ['--user-id','--key-file','--sync-endpoint','--cpu-workers','--compile-threads','--memory-budget-bytes','--cache-budget-bytes']:
        assert removed not in text


def test_stop_defaults_to_all_tasks(tmp_path,monkeypatch):
    from autofusedebugger.fuzz.scheduling.ledger import TaskLedger
    from autofusedebugger.fuzz.contracts import ExecutionConfig,EvaluationConfig
    monkeypatch.setenv('AUTOFUSEDBG_HOME',str(tmp_path))
    ledger=TaskLedger(tmp_path/'tasks/tasks.sqlite')
    ids=[ledger.enqueue(str(i),ExecutionConfig('cpu','cpu',1),EvaluationConfig()) for i in range(3)]
    assert main(['stop'])==0
    assert all(ledger.get(i)['state']=='cancelled' for i in ids)


def test_admin_rotation_is_by_name(capsys):
    with pytest.raises(SystemExit):main(['admin','rotate-token','--help'])
    text=capsys.readouterr().out
    assert 'user_id' not in text
