import json
from dataclasses import asdict
from types import SimpleNamespace

from autofusex.client.retest_queue import skipped_evidence
from autofusex.contracts.coverage import environment_key
from autofusex.contracts.execution import ExecutionConfig
from autofusex.execution.storage.lifecycle import stamp
from autofusex.execution.storage.stage_facts import save_case
from autofusex.workloads.fuzz.scheduling.stages import Case, StageTask, Kind


def test_skip_audit_survives_until_receipt_and_matches_task(tmp_path):
    database = tmp_path / 'results.sqlite'
    evidence = dict(backend='af-legacy', environment={'torch_npu': 'post7'},
                    autofuse_coverage=dict(candidates=0, installed=True, no_coverage=True),
                    error='norm: not in lowering whitelist')
    task = dict(id='local-task', attempt_id='attempt',
                execution=json.dumps(asdict(ExecutionConfig('af-legacy', 'npu:0', 1))))
    case = Case('case', 'C16', stamp(), [task], str(tmp_path))
    case.skip_reason = 'no_autofuse_coverage'
    case.skip_finalized = True
    stage = StageTask(Kind.COMBINED, 'case')
    stage.output = {'result': {'coverage_evidence': evidence}}
    case.stages[stage.identity] = stage
    save_case(database, case, {}, 'window', stamp(), SimpleNamespace())
    assert skipped_evidence(dict(database=str(database), task_ids=['local-task'])) == [evidence]
    assert skipped_evidence(dict(database=str(database), task_ids=['other-task'])) == []


def test_environment_identity_excludes_host_but_includes_local_build_and_flags():
    environment = dict(torch='2.10', torch_npu='post7', torch_npu_build='a'*64,
                       cann='9.2', autofuse_binary='b'*64)
    assert environment_key(environment) == environment_key(dict(environment, installation_id='other'))
    for field in ('torch', 'torch_npu', 'torch_npu_build', 'cann', 'autofuse_binary', 'compiler_flags'):
        assert environment_key(environment) != environment_key(dict(environment, **{field: 'changed'}))
    assert environment_key(dict(environment, torch_npu_build=None)) is None
