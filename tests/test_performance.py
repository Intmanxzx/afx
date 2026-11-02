from dataclasses import asdict
import json
from pathlib import Path

import pytest

from autofusedebugger.nnsmith_integration.performance.backends import BACKENDS
from autofusedebugger.nnsmith_integration.performance.report import comparable_speedup, write_reports
from autofusedebugger.nnsmith_integration.performance.schema import BackendResult, ProfileConfig
from autofusedebugger.nnsmith_integration.performance.statistics import (
    ProfileDataError, find_op_statistic, read_op_statistic,
)


def test_profiler_totals_include_all_types_once(tmp_path):
    p=tmp_path/'op_statistic.csv'
    p.write_text('\ufeffOP Type,Task Type,Occurrences,Total Time(us),Average Time(us)\nAdd,AI_CORE,100,1200,12\nOther,AI_CPU,200,300,1.5\n')
    result=read_op_statistic(p,100)
    assert result['total_kernel_us']==1500
    assert result['mean_kernel_us']==15
    assert len(result['op_statistics'])==2


@pytest.mark.parametrize('unit,value,expected',[('ms','1.5',1500),('ns','1500',1.5),('μs','1.5',1.5)])
def test_profiler_units(tmp_path,unit,value,expected):
    p=tmp_path/'op_statistic.csv'
    p.write_text(f'OP Type,Total Time({unit})\nAdd,{value}\n')
    assert read_op_statistic(p,1)['mean_kernel_us']==expected


@pytest.mark.parametrize('value',['NaN','Infinity','-1','bad',''])
def test_invalid_duration_does_not_become_zero(tmp_path,value):
    p=tmp_path/'op_statistic.csv'
    p.write_text(f'OP Type,Total Time(us)\nAdd,{value}\n')
    with pytest.raises(ProfileDataError):read_op_statistic(p,100)


def test_average_column_is_not_accepted_as_total(tmp_path):
    p=tmp_path/'op_statistic.csv'
    p.write_text('OP Type,Average Time(us)\nAdd,15\n')
    with pytest.raises(ProfileDataError):read_op_statistic(p,100)


def test_profiler_file_selection_avoids_duplicate_raw_csv(tmp_path):
    raw=tmp_path/'raw'/'op_statistic.csv';raw.parent.mkdir();raw.touch()
    final=tmp_path/'capture'/'ASCEND_PROFILER_OUTPUT'/'op_statistic.csv';final.parent.mkdir(parents=True);final.touch()
    assert find_op_statistic(tmp_path)==final
    other=tmp_path/'capture2'/'ASCEND_PROFILER_OUTPUT'/'op_statistic.csv';other.parent.mkdir(parents=True);other.touch()
    with pytest.raises(ProfileDataError):find_op_statistic(tmp_path)


def test_triton_removes_inherited_ascendc_before_process_start(tmp_path):
    inherited={'TORCHINDUCTOR_NPU_BACKEND':'ascendc','PYTHONPATH':'/cann/python','TORCH_COMPILE_DEBUG':'1'}
    env=BACKENDS['triton-ascend'].environment(inherited,tmp_path/'triton')
    assert 'TORCHINDUCTOR_NPU_BACKEND' not in env
    assert 'options' not in BACKENDS['triton-ascend'].compile_kwargs()
    assert 'TORCH_COMPILE_DEBUG' not in env
    assert env['PYTHONPATH']=='/cann/python'
    assert inherited['TORCHINDUCTOR_NPU_BACKEND']=='ascendc'
    assert BACKENDS['ascendc'].environment(inherited,tmp_path/'asc')['TORCHINDUCTOR_NPU_BACKEND']=='ascendc'


def measurements():
    return {'ascendc':asdict(BackendResult('ascendc',status='ok',correctness='pass',fingerprint='same',mean_kernel_us=10)),
            'triton-ascend':asdict(BackendResult('triton-ascend',status='ok',correctness='pass',fingerprint='same',mean_kernel_us=20))}


def test_speedup_requires_matching_cases_and_tracks_accuracy_separately():
    results=measurements()
    assert comparable_speedup(results,'same')==2
    results['triton-ascend']['fingerprint']='different'
    assert comparable_speedup(results,'same') is None
    results['triton-ascend']['fingerprint']='same'
    results['ascendc']['correctness']='fail'
    assert comparable_speedup(results,'same') == 2


def test_report_integrates_performance_tabs_and_accuracy(tmp_path):
    results=measurements()
    summary={'total':1,'comparable':1,'errors':0,'profile':asdict(ProfileConfig()),'cases':[{'seed':42,'index':0,'status':'compared','fingerprint':'same','backends':results,'ascendc_speedup':2}]}
    write_reports(summary,tmp_path)
    page=(tmp_path/'summary.html').read_text()
    assert '性能对比汇总' in page and 'tabBtnStats' in page and 'tabBtnTable' in page
    assert 'Kernel耗时(μs/步)' in page and 'worker 总墙钟时间' in page
    assert '"performance_status": "pass"' in page
    assert page.count('class="chart-card"') == 12


@pytest.mark.parametrize('a,t,expected', [(10,10,'pass'),(10,20,'pass'),(20,10,'fail'),(0,10,'error'),(float('nan'),10,'error'),(10,None,'error')])
def test_performance_verdict(a,t,expected):
    from autofusedebugger.nnsmith_integration.performance.metrics import performance_status
    results=measurements()
    results['ascendc']['mean_kernel_us']=a
    results['triton-ascend']['mean_kernel_us']=t
    assert performance_status({'fingerprint':'same','backends':results}) == expected


def test_csv_kernel_count_normalized_per_step(tmp_path):
    p=tmp_path/'op_statistic.csv'
    p.write_text('OP Type,Count,Total Time(us)\nAdd,100,1200\nCopy,200,300\n')
    stats=read_op_statistic(p,100)
    assert stats['total_kernel_count']==300 and stats['mean_kernel_count']==3
    p.write_text('OP Type,Total Time(us)\nAdd,100\n')
    assert read_op_statistic(p,100)['mean_kernel_count'] is None


@pytest.mark.parametrize('count',['-1','1.5','NaN','bad',''])
def test_invalid_kernel_counts(tmp_path,count):
    p=tmp_path/'op_statistic.csv'
    p.write_text(f'OP Type,Count,Total Time(us)\nAdd,{count},100\n')
    with pytest.raises(ProfileDataError):read_op_statistic(p,100)


def test_performance_report_keeps_accuracy_separate_and_escapes_script(tmp_path):
    from autofusedebugger.nnsmith_integration.performance.report import summary_for_html
    results=measurements()
    results['ascendc']['correctness']='fail'
    case={'seed':42,'index':0,'fingerprint':'same','backends':results,'error':'</script><script>alert(1)</script>'}
    summary={'total':1,'cases':[case]}
    adapted=summary_for_html(summary)
    assert adapted['cases'][0]['status']=='fail'
    assert adapted['performance_cases'][0]['performance_status']=='pass'
    write_reports(summary,tmp_path)
    assert '</script><script>alert(1)</script>' not in (tmp_path/'summary.html').read_text()


def test_profile_config_rejects_empty_measurement():
    with pytest.raises(ValueError):ProfileConfig(steps=0)
    with pytest.raises(ValueError):ProfileConfig(warmup_steps=0)
    with pytest.raises(ValueError):ProfileConfig(timeout_s=float('nan'))
    with pytest.raises(ValueError):ProfileConfig(timeout_s=float('inf'))


def test_perf_cli_uses_same_generation_flags():
    from autofusedebugger.cli import build_parser
    args=build_parser().parse_args(['af-fuzz','--out-dir','/tmp/test','--compare-performance','--repro-seed','42','--profile-steps','20'])
    assert args.compare_performance and args.repro_seed==42 and args.profile_steps==20


def test_backend_timeout_terminates_only_its_worker_group(tmp_path,monkeypatch):
    import subprocess
    from autofusedebugger.nnsmith_integration.performance import runner
    calls={}
    class Process:
        pid=4321
        count=0
        def wait(self,timeout=None):
            self.count+=1
            if self.count==1:raise subprocess.TimeoutExpired('worker',timeout)
            return -15
    def popen(command,**kwargs):
        calls.update(kwargs)
        return Process()
    signals=[]
    monkeypatch.setattr(runner.subprocess,'Popen',popen)
    monkeypatch.setattr(runner.os,'killpg',lambda pid,sig:signals.append((pid,sig)))
    result=runner.run_backend(spec=BACKENDS['triton-ascend'],request={},output_dir=tmp_path/'run',cache_dir=tmp_path/'cache',profile=ProfileConfig(timeout_s=0.01))
    assert result.status=='timeout' and result.mean_kernel_us is None
    assert calls['start_new_session'] is True
    assert 'TORCHINDUCTOR_NPU_BACKEND' not in calls['env']
    assert signals[0][0]==4321


def test_session_pairs_same_case_and_alternates_backend_order(tmp_path,monkeypatch):
    import nnsmith.materialize as materialize
    from nnsmith.abstract.op import Add
    from autofusedebugger.nnsmith_integration import ascendc_backend,ascendc_opset,generation
    from autofusedebugger.nnsmith_integration.performance import session
    from types import SimpleNamespace
    monkeypatch.setattr(materialize.Model,'init',lambda *args,**kwargs:SimpleNamespace(add_seed_setter=lambda:None))
    monkeypatch.setattr(ascendc_backend,'AscendC',lambda **kwargs:object())
    monkeypatch.setattr(ascendc_opset,'probe_ascendc_opset',lambda *args,**kwargs:[Add])
    monkeypatch.setattr(generation,'generate_case',lambda **kwargs:object())
    monkeypatch.setattr(session,'case_fingerprint',lambda case:'same')
    monkeypatch.setattr(session,'describe_case',lambda case:{'n_nodes':3})
    calls=[]
    def run_backend(**kwargs):
        request=kwargs['request']; name=kwargs['spec'].name
        calls.append((request['seed'],name,request['expected_fingerprint']))
        return BackendResult(name,status='ok',correctness='pass',fingerprint='same',mean_kernel_us=10 if name=='ascendc' else 20)
    monkeypatch.setattr(session,'run_backend',run_backend)
    summary=session.run_performance_session(out_dir=tmp_path,generation=generation.GenerationConfig(),profile=ProfileConfig(),n_cases=2,seed=42)
    assert [c[1] for c in calls]==['ascendc','triton-ascend','triton-ascend','ascendc']
    assert calls[0][0]==calls[1][0] and calls[2][0]==calls[3][0]
    assert summary['comparable']==2 and summary['errors']==0
    assert all(c['ascendc_speedup']==2 for c in summary['cases'])


def test_capture_synchronizes_final_step_before_saving(tmp_path,monkeypatch):
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace
    from autofusedebugger.nnsmith_integration.performance import profiling
    events=[]
    class Capture:
        def __enter__(self):events.append('enter');return self
        def __exit__(self,*args):events.append('exit')
        def step(self):events.append('step')
    settings={}
    def schedule(**kwargs):settings.update(kwargs);return object()
    profiler=SimpleNamespace(
        _ExperimentalConfig=lambda **kwargs:object(),
        ExportType=SimpleNamespace(Text='text'),ProfilerLevel=SimpleNamespace(Level2=2),
        AiCMetrics=SimpleNamespace(PipeUtilization='pipe'),
        ProfilerActivity=SimpleNamespace(CPU='cpu',NPU='npu'),
        schedule=schedule,profile=lambda **kwargs:Capture(),tensorboard_trace_handler=lambda path:object(),
    )
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(no_grad=nullcontext,npu=SimpleNamespace(synchronize=lambda device:events.append('sync'))))
    monkeypatch.setitem(sys.modules,'torch_npu',SimpleNamespace(profiler=profiler))
    monkeypatch.setattr(profiling,'find_op_statistic',lambda path:path/'op_statistic.csv')
    monkeypatch.setattr(profiling,'read_op_statistic',lambda path,steps:{'measured_steps':steps})
    result=profiling.profile_kernels(lambda *args:events.append('kernel'),[],device='npu:0',output_dir=tmp_path/'profile',config=ProfileConfig(steps=3))
    assert result['measured_steps']==3
    assert settings=={'wait':0,'warmup':0,'active':3,'repeat':1}
    assert events==['sync','enter','kernel','step','kernel','step','kernel','sync','step','exit']
