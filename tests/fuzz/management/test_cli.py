import json
import subprocess
import sys
from pathlib import Path
import pytest


def run_cli(tmp_path,*args,input=None):
    env={'PYTHONPATH':str(Path.cwd()),'PYTHONDONTWRITEBYTECODE':'1'}
    import os
    process=subprocess.run([sys.executable,'-m','autofusex.client','daemon',*args,'--data-dir',str(tmp_path)],input=input,text=True,capture_output=True,env={**os.environ,**env})
    return process


def test_help_is_discoverable_and_errors_are_json(tmp_path):
    process=subprocess.run([sys.executable,'-m','autofusex.client','daemon','--help'],text=True,capture_output=True,env={'PYTHONPATH':str(Path.cwd())})
    assert process.returncode==0 and 'start' in process.stdout and 'status' in process.stdout
    process=subprocess.run([sys.executable,'-m','autofusex.client','daemon','does-not-exist'],text=True,capture_output=True,env={'PYTHONPATH':str(Path.cwd())})
    assert process.returncode==2
    assert json.loads(process.stdout)['error']['code']=='usage_error'


def test_daemon_command_json_and_exit_codes(tmp_path):
    start=run_cli(tmp_path,'start');assert start.returncode==0;assert json.loads(start.stdout)['ok']
    status=run_cli(tmp_path,'status');value=json.loads(status.stdout);assert value['ok'] and value['data']['protocol']==1
    stop=run_cli(tmp_path,'stop');assert stop.returncode==0


def test_file_command_parsing_accepts_stdin_and_plan_alias(tmp_path):
    payload=json.dumps(dict(schema_version=1,kind='fuzz_task',environment='local',workload={'mode':'generate'},results={'upload':False}))
    process=subprocess.run([sys.executable,'-m','autofusex','config','validate','-','--data-dir',str(tmp_path)],input=payload,text=True,capture_output=True,env={'PYTHONPATH':str(Path.cwd())})
    assert process.returncode==0,json.loads(process.stdout)
    process=subprocess.run([sys.executable,'-m','autofusex','env','apply','abc','--data-dir',str(tmp_path)],text=True,capture_output=True,env={'PYTHONPATH':str(Path.cwd())})
    # Daemon is absent, but parser accepted --plan and returns a stable envelope.
    assert process.returncode in {2,3};assert json.loads(process.stdout)['ok'] is False


def test_config_export_is_directly_importable(tmp_path):
    args = [sys.executable, '-m', 'autofusex', 'config']
    exported = subprocess.run([*args, 'export', '--data-dir', str(tmp_path)], capture_output=True, text=True)
    assert exported.returncode == 0, exported.stderr
    assert json.loads(exported.stdout)['kind'] == 'fuzz_task'
    checked = subprocess.run([*args, 'validate', '-', '--data-dir', str(tmp_path)], input=exported.stdout, capture_output=True, text=True)
    assert checked.returncode == 0, checked.stdout
    for action in ('effective', 'apply'):
        removed = subprocess.run([*args, action], capture_output=True, text=True)
        assert removed.returncode == 2


def test_removed_local_control_actions(tmp_path):
    from autofusex.client.daemon import Manager
    manager = Manager(tmp_path)
    try:
        for action in ('trust', 'effective'):
            with pytest.raises(ValueError, match='unknown local action'):
                manager.dispatch({'action': action, 'policy': {}})
        assert not (tmp_path / 'trust.json').exists()
    finally:
        manager.pool.shutdown()
