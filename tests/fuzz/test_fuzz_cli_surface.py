from autofusex.fuzz.client.generation.sampling import SamplingConfig
import json
import pytest
from autofusex.fuzz.client.cli import main


def test_top_level_is_small(capsys):
    with pytest.raises(SystemExit) as exc:main(['--help'])
    assert exc.value.code==0
    text=capsys.readouterr().out
    assert all(title in text for title in ['客户端','本地查看','维护'])
    assert all(command in text for command in ['run [SEED','replay SEED','view RESULTS'])
    assert 'setup --token' not in text and 'admin user' not in text
    assert 'admin-user' not in text and 'check-db' not in text


def test_stop_defaults_to_all_tasks(tmp_path,monkeypatch):
    from autofusex.fuzz.client.scheduling.ledger import TaskLedger
    from autofusex.fuzz.client.contracts import ExecutionConfig,EvaluationConfig
    monkeypatch.setenv('AUTOFUSEX_HOME',str(tmp_path))
    ledger=TaskLedger(tmp_path/'tasks/tasks.sqlite')
    ids=[ledger.enqueue(str(SamplingConfig.profile('small',max_nodes=8,dtype_mode='float32').seed(i).value),ExecutionConfig('cpu','cpu',1),EvaluationConfig()) for i in range(3)]
    assert main(['stop'])==0
    assert all(ledger.get(i)['state']=='cancelled' for i in ids)


@pytest.mark.parametrize('args', [
    ['run'], ['replay'], ['status'], ['stop'], ['view'],
    ['maintenance'], ['maintenance', 'snapshot'], ['maintenance', 'restore'],
])
def test_documented_command_help_is_reachable(args, capsys):
    with pytest.raises(SystemExit) as exc:
        main([*args, '-h'])
    assert exc.value.code == 0
    assert 'usage:' in capsys.readouterr().out


def test_short_flags_and_explicit_save(capsys):
    from autofusex.fuzz.client.cli import replay_parser
    args = replay_parser().parse_args(['B16', '--perf'])
    assert args.performance and not args.save
    assert replay_parser().parse_args(['B16', '--save']).save
    with pytest.raises(SystemExit): main(['run', '-h'])
    text = capsys.readouterr().out
    assert '--fg' in text and '--perf' in text and '--save' in text
    assert '--foreground' not in text and '--performance' not in text
    assert 'continuous until fuzz stop' in ' '.join(text.split())


def test_run_defaults_to_infinite_unsaved_and_save_is_per_task(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr('autofusex.fuzz.client.generation.runtime.source_evidence', lambda: {})
    monkeypatch.setenv('AUTOFUSEX_HOME', str(tmp_path))
    monkeypatch.setattr('autofusex.fuzz.client.scheduling.devices.select_devices', lambda _: ['npu:0'])
    observed = []
    def launch(ledger, database, specifications, **kwargs):
        observed.append(kwargs)
        return {'state': 'test'}
    monkeypatch.setattr('autofusex.fuzz.client.scheduling.hosting.launch', launch)
    assert main(['run', '--fg']) == 0
    configs, evaluation, count, devices, excluded, sampling, ops, budget = observed[-1]['campaign']
    assert ops is None and budget['worker_max_tasks'] == 16
    assert sampling['dim_dist_mode'] == 'half_normal'
    assert sampling['dim_max'] == 131072
    assert count is None and observed[-1]['foreground']
    assert all(not c.save_results for c in configs)
    assert main(['run', '--count', '3', '--save']) == 0
    configs, _, count, devices, excluded, sampling, ops, budget = observed[-1]['campaign']
    assert count == 3 and all(c.save_results for c in configs)


def test_continue_reuses_the_saved_campaign_without_retyping_flags(tmp_path, monkeypatch, capsys):
    # Regression for a real incident: after a host restart nobody could recall the exact
    # generation/backend/device/budget flags a multi-hour campaign had been started with,
    # and retyping from memory would have silently fallen back to CLI defaults for anything
    # forgotten. --continue must reconstruct everything from what was actually persisted.
    from autofusex.fuzz.client.scheduling.campaign import initialize as initialize_campaign
    monkeypatch.setattr('autofusex.fuzz.client.generation.runtime.source_evidence', lambda: {})
    monkeypatch.setenv('AUTOFUSEX_HOME', str(tmp_path))
    monkeypatch.setattr('autofusex.fuzz.client.scheduling.devices.select_devices', lambda _: ['npu:0', 'npu:1'])
    observed = []
    def launch(ledger, database, specifications, **kwargs):
        if kwargs.get('campaign') is not None:
            initialize_campaign(ledger, *kwargs['campaign'])
        observed.append(kwargs)
        return {'state': 'test'}
    monkeypatch.setattr('autofusex.fuzz.client.scheduling.hosting.launch', launch)
    assert main(['run', '--fg', '--backend', 'af-legacy', '--tensor-scale', 'small', '--worker-max-tasks', '4']) == 0
    assert 'note: replacing' not in capsys.readouterr().err  # nothing saved yet on the first launch
    first = observed[-1]['campaign']
    assert main(['run', '--continue', '--fg']) == 0
    assert observed[-1]['devices'] == ['npu:0', 'npu:1']
    second = observed[-1]['campaign']
    assert [c.backend for c in second[0]] == [c.backend for c in first[0]]
    assert [c.compile_threads for c in second[0]] == [c.compile_threads for c in first[0]]
    assert second[0][0].batch_id != first[0][0].batch_id  # relaunch still gets its own grouping id
    assert second[5] == first[5]  # sampling profile ('small') preserved exactly
    assert second[7] == first[7] == {**first[7], 'worker_max_tasks': 4}  # budget preserved
    # Replacing a still-live saved campaign without --continue is flagged, not silent.
    assert main(['run', '--fg', '--backend', 'af-mlir']) == 0
    assert 'note: replacing the saved campaign' in capsys.readouterr().err


def test_continue_without_a_saved_campaign_is_a_clear_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('AUTOFUSEX_HOME', str(tmp_path))
    assert main(['run', '--continue']) == 2
    assert 'no saved campaign' in capsys.readouterr().err


def test_continue_rejects_flags_it_would_otherwise_silently_ignore(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('AUTOFUSEX_HOME', str(tmp_path))
    assert main(['run', '--continue', '--backend', 'af-mlir']) == 2
    assert 'backend' in capsys.readouterr().err
    assert main(['run', '--continue', 'C16']) == 2
    assert 'seeds' in capsys.readouterr().err


def test_run_resource_policy_recomputes_and_honors_overrides(tmp_path, monkeypatch, capsys):
    from autofusex.fuzz.client.config import setup, load_config
    from autofusex.fuzz.client import resources
    monkeypatch.setenv('AUTOFUSEX_HOME', str(tmp_path))
    monkeypatch.setattr('autofusex.fuzz.client.generation.runtime.source_evidence', lambda: {})
    monkeypatch.setattr('autofusex.fuzz.client.scheduling.devices.select_devices', lambda _: ['npu:0','npu:1'])
    cores = [36]
    monkeypatch.setattr(resources, 'resource_inventory', lambda _: {
        'cpu_worker_ceiling': cores[0], 'effective_memory_bytes': 200*1024**3})
    setup(tmp_path, cpu_workers=6, compile_threads=6, resource_policy='auto')
    original = load_config(tmp_path)
    observed = []
    def launch(ledger, database, specifications, **kwargs):
        observed.append(kwargs['budget'])
        return {'state': 'test'}
    monkeypatch.setattr('autofusex.fuzz.client.scheduling.hosting.launch', launch)
    assert main(['run']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['resource_plan']['selected'] == {'cpu_workers':4,'compile_threads':8}
    cores[0] = 8
    assert main(['run']) == 0
    assert json.loads(capsys.readouterr().out)['resource_plan']['selected'] == {'cpu_workers':4,'compile_threads':2}
    assert main(['run','--fixed-resources']) == 0
    assert 'resource_plan' not in json.loads(capsys.readouterr().out)
    assert (observed[-1].cpu_workers, observed[-1].compile_threads) == (6, 6)
    assert main(['run','--workers','2','--compile-threads','3']) == 0
    capsys.readouterr()
    assert (observed[-1].cpu_workers, observed[-1].compile_threads) == (2,3)
    assert load_config(tmp_path) == original  # Launch never rewrites explicit saved values.


def test_foreground_plan_is_emitted_before_launch(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('AUTOFUSEX_HOME', str(tmp_path))
    monkeypatch.setattr('autofusex.fuzz.client.generation.runtime.source_evidence', lambda: {})
    monkeypatch.setattr('autofusex.fuzz.client.scheduling.devices.select_devices', lambda _: ['npu:0'])
    def launch(*args, **kwargs):
        plan = json.loads(capsys.readouterr().err)['resource_plan']
        assert plan['version'] == 3
        assert plan['selected']['cpu_workers'] == kwargs['budget'].cpu_workers
        assert all(c.compile_threads == kwargs['budget'].compile_threads for c in kwargs['campaign'][0])
        return {'state': 'test'}
    monkeypatch.setattr('autofusex.fuzz.client.scheduling.hosting.launch', launch)
    assert main(['run', '--fg']) == 0
    assert 'resource_plan' in json.loads(capsys.readouterr().out)


@pytest.mark.parametrize('flags', [
    ['--auto-resources', '--workers', '2'],
    ['--auto-resources', '--compile-threads', '2'],
])
def test_auto_explicit_override_conflicts_never_launch(tmp_path, monkeypatch, capsys, flags):
    monkeypatch.setenv('AUTOFUSEX_HOME', str(tmp_path))
    monkeypatch.setattr('autofusex.fuzz.client.generation.runtime.source_evidence', lambda: {})
    monkeypatch.setattr('autofusex.fuzz.client.scheduling.devices.select_devices', lambda _: ['npu:0'])
    monkeypatch.setattr('autofusex.fuzz.client.scheduling.hosting.launch', lambda *a, **kw: pytest.fail('launched'))
    assert main(['run', *flags]) == 2
    assert 'cannot be combined' in capsys.readouterr().err


def test_auto_fixed_flags_are_mutually_exclusive():
    with pytest.raises(SystemExit) as exc:
        main(['run', '--auto-resources', '--fixed-resources'])
    assert exc.value.code == 2
