"""Execute an immutable model or fixed case using the existing model runners."""
import json
from pathlib import Path
import sys
import time
import traceback
import uuid


def acquire_device(device):
    from autofusex.execution.backends.device_lease import DeviceLease
    import torch, torch_npu
    return DeviceLease(str(torch.npu.get_device_properties(device).uuid)).acquire(timeout=60)


def run(prepared):
    from autofusex.client.artifacts import atomic
    from autofusex.workloads.imported.registry import inspect, script_path
    import os
    directory = Path(prepared['directory']); home = Path(os.environ['AFX_TASK_HOME'])
    spec = prepared['spec']; workload = spec['workload']
    record = inspect(home, workload['case_key'], workload['revision'])
    script = script_path(home, record)
    output = directory / 'artifacts'
    start = time.time_ns()
    result = dict(schema_version=1, execution_id=str(uuid.uuid4()), case=record,
                  workload=record['spec']['workload'], started_at=start//1000000,
                  environment=prepared['environment']['id'], status='running', artifacts=[])
    atomic(directory/'state.json', {'state':'running'})
    lease = None
    previous_path = list(sys.path)
    previous_cwd = Path.cwd()
    try:
        sys.path.insert(0, str(script.parent))
        os.chdir(script.parent)
        from autofusex.execution.environment import collect
        result['software'] = collect(home)
        lease = acquire_device(spec['execution']['devices'][0])
        kwargs = dict(script=script, out_dir=output, device=spec['execution']['devices'][0],
                      with_cpu=workload.get('with_cpu', False),
                      rtol=workload.get('rtol'), atol=workload.get('atol'))
        if record['spec']['entry']:
            from autofusex.workloads.model.torch_model import run_torch_model
            summary = run_torch_model(entry=record['spec']['entry'], fullgraph=workload.get('fullgraph'), dynamic=workload.get('dynamic'), **kwargs)
        else:
            from autofusex.workloads.model.script_runner import run_framework_script
            summary = run_framework_script(script_args=tuple(workload.get('script_args', [])),
                fullgraph=workload.get('fullgraph',False), dynamic=workload.get('dynamic',False), **kwargs)
        result.update(status='passed' if summary['passed'] else 'accuracy_failed', summary=summary)
    except Exception as exc:
        result.update(status='execution_failed', error=type(exc).__name__, message=str(exc)[:2000])
        traceback.print_exc()
    finally:
        if lease is not None: lease.close()
        sys.path[:] = previous_path
        os.chdir(previous_cwd)
    result['elapsed_ns'] = time.time_ns()-start
    if output.exists():
        import hashlib
        for path in sorted(output.rglob('*')):
            if path.is_file() and not path.is_symlink():
                result['artifacts'].append(dict(path=path.relative_to(directory).as_posix(),
                    size=path.stat().st_size, sha256=__import__('autofusex.client.artifacts',fromlist=['sha256']).sha256(path)))
    if spec['results']['save']:
        atomic(directory/'result.json', result)
    atomic(directory/'state.json', dict(state='completed' if result['status'] in {'passed','accuracy_failed'} else 'failed', result=result))
    return 0 if result['status'] in {'passed','accuracy_failed'} else 1


if __name__ == '__main__':
    raise SystemExit(run(json.loads(Path(sys.argv[1]).read_text())))
