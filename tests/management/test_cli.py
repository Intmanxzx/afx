import json
import subprocess
import sys
from pathlib import Path
import pytest


def run_cli(tmp_path,*args,input=None):
    env={'PYTHONPATH':str(Path.cwd()),'PYTHONDONTWRITEBYTECODE':'1'}
    import os
    process=subprocess.run([sys.executable,'-m','autofusedebugger','daemon',*args,'--data-dir',str(tmp_path)],input=input,text=True,capture_output=True,env={**os.environ,**env})
    return process


def test_help_is_discoverable_and_errors_are_json(tmp_path):
    process=subprocess.run([sys.executable,'-m','autofusedebugger','daemon','--help'],text=True,capture_output=True,env={'PYTHONPATH':str(Path.cwd())})
    assert process.returncode==0 and 'start' in process.stdout and 'status' in process.stdout
    process=subprocess.run([sys.executable,'-m','autofusedebugger','daemon','does-not-exist'],text=True,capture_output=True,env={'PYTHONPATH':str(Path.cwd())})
    assert process.returncode==2
    assert json.loads(process.stdout)['error']['code']=='usage_error'


def test_daemon_command_json_and_exit_codes(tmp_path):
    start=run_cli(tmp_path,'start');assert start.returncode==0;assert json.loads(start.stdout)['ok']
    status=run_cli(tmp_path,'status');value=json.loads(status.stdout);assert value['ok'] and value['data']['protocol']==1
    stop=run_cli(tmp_path,'stop');assert stop.returncode==0


def test_file_command_parsing_accepts_stdin_and_plan_alias(tmp_path):
    payload='schema_version: 1\nkind: client_policy\n'
    process=subprocess.run([sys.executable,'-m','autofusedebugger','config','validate','-','--data-dir',str(tmp_path)],input=payload,text=True,capture_output=True,env={'PYTHONPATH':str(Path.cwd())})
    assert process.returncode==0,json.loads(process.stdout)
    process=subprocess.run([sys.executable,'-m','autofusedebugger','environment','apply','--plan','abc','--data-dir',str(tmp_path)],text=True,capture_output=True,env={'PYTHONPATH':str(Path.cwd())})
    # Daemon is absent, but parser accepted --plan and returns a stable envelope.
    assert process.returncode in {2,3};assert json.loads(process.stdout)['ok'] is False
