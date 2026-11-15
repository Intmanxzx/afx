from autofusex.fuzz.generation.sampling import SamplingConfig
CURRENT_SEED = str(SamplingConfig.profile('small', max_nodes=1, dtype_mode='float32').seed(3))
"""Current production CLI -> supervisor -> NNSmith -> execution -> storage tests."""
import json
import subprocess
import sys
from pathlib import Path
import pytest


def test_no_template_implementation_or_fallback():
    from autofusex.fuzz import generation
    assert not Path(generation.__file__).with_name('v1.py').exists()
    assert not Path(generation.__file__).with_name('v2.py').exists()
    for seed in ('A16','B16','42'):
        with pytest.raises(ValueError):
            generation.materialize(seed)


def test_cloud_import_and_metadata_without_nnsmith(tmp_path):
    code='''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'torch','torch_npu','numpy','nnsmith','z3'}: raise AssertionError(name)
sys.meta_path.insert(0,Block())
from autofusex.fuzz.storage import ResultRepository
from autofusex.fuzz.view.server import create_server
from autofusex.fuzz.central.server import create_server as central
from autofusex.fuzz.cli import main
assert main(['maintenance','doctor'])==0
'''
    subprocess.run([sys.executable,'-c',code],check=True)


def test_run_campaign_and_replay_use_nnsmith(tmp_path, monkeypatch, capsys):
    pytest.importorskip('nnsmith');pytest.importorskip('torch')
    from autofusex.fuzz.cli import main
    from autofusex.fuzz.scheduling import hosting
    from autofusex.fuzz.scheduling.campaign import initialize,refill
    from autofusex.fuzz.generation import materialize
    from autofusex.fuzz.contracts import Seed
    from autofusex.fuzz.storage.queries import ResultQueries
    monkeypatch.setenv('AUTOFUSEX_HOME',str(tmp_path))
    monkeypatch.setattr('autofusex.fuzz.scheduling.campaign.secrets.randbelow',lambda _:42)
    def launch(ledger,database,specifications,**kw):
        initialize(ledger,*kw['campaign']);refill(ledger)
        tasks=ledger.page()['items']
        assert len(tasks)==2
        for task in tasks:
            seed=Seed.parse(task['seed']);assert seed.short.startswith(('E','F')) and Seed.parse(seed.short)==seed
            assert json.loads(materialize(seed).spec.graph_json)['generator']=='nnsmith.graph_gen.model_gen'
        return {'verified':True}
    monkeypatch.setattr(hosting,'launch',launch)
    assert main(['run','--count','1','--save','--fg'])==0
    capsys.readouterr()
    assert main(['replay',CURRENT_SEED,'--backend','cpu','--save','--scheduling','combined'])==0
    result=json.loads(capsys.readouterr().out)
    assert result['status']=='ok'
    assert json.loads(json.loads(result['execution']['case_spec'])['graph_json'])['generator']=='nnsmith.graph_gen.model_gen'
    # CPU-only attempts are saved but intentionally absent from AF case summaries.
    assert ResultQueries(tmp_path/'results/results.sqlite').page()['items']


def test_real_supervisor_worker_uses_nnsmith(tmp_path):
    pytest.importorskip('nnsmith');pytest.importorskip('torch')
    from autofusex.fuzz.scheduling.ledger import TaskLedger
    from autofusex.fuzz.scheduling.supervisor import drain
    from autofusex.fuzz.contracts import ExecutionConfig,EvaluationConfig
    from autofusex.fuzz.storage import ResultRepository
    ledger=TaskLedger(tmp_path/'tasks.sqlite');database=tmp_path/'results.sqlite'
    identity=ledger.enqueue(CURRENT_SEED,ExecutionConfig('cpu','cpu',1),EvaluationConfig())
    assert drain(ledger,database)['completed']==1
    result=ResultRepository(database).detail(ledger.get(identity)['attempt_id'])
    assert result['status']=='ok'
    assert 'nnsmith.graph_gen.model_gen' in result['contexts']['execution']['case_spec']


@pytest.mark.parametrize('value', ['A16','A000000016','B16','B000000016','42',42,str(2**128+42),2**128+42])
def test_no_ab_seed_compatibility(value):
    from autofusex.fuzz.contracts import Seed
    with pytest.raises(ValueError): Seed.parse(value)


def test_only_current_seed_roundtrip():
    from autofusex.fuzz.contracts import Seed
    seed=Seed.parse('C000000016')
    assert Seed.parse(str(seed))==seed
    assert seed.short=='C000000016'
