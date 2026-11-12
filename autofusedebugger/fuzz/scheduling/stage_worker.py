"""Isolated reusable stage worker, JSON over a private inherited socket.

Backend sessions never cross workers. All dynamic shapes run in one invocation
of the shared execute loop. Only CPU output artifacts survive the invocation.
"""
import gc
import json
import os
from pathlib import Path
import socket
import struct
import sys
import time

LIMIT = 2 * 1024 * 1024


def receive(sock):
    def exact(n):
        data = bytearray()
        while len(data) < n:
            chunk = sock.recv(n-len(data))
            if not chunk:
                raise EOFError()
            data.extend(chunk)
        return bytes(data)
    size = struct.unpack('!I', exact(4))[0]
    if size > LIMIT:
        raise ValueError('stage message exceeds bound')
    return json.loads(exact(size))


def send(sock, value):
    raw = json.dumps(value, allow_nan=False).encode()
    if len(raw) > LIMIT:
        raise ValueError('stage response exceeds bound')
    sock.sendall(struct.pack('!I', len(raw))+raw)


def perform(job):
    from .artifacts import bundle_from, write_arrays, read_arrays, write_bundle
    from ..contracts import ExecutionConfig, EvaluationConfig
    from ..backends import create_backend
    from ..execution import execute
    kind = job['kind']
    if kind == 'generate':
        from ..generation import materialize, workload_id
        from ..case_spec import encode
        bundle = materialize(job['seed'])
        if set(job.get('exclude_ops',[])) & {n['op'] for n in json.loads(bundle.spec.graph_json)['nodes']}:
            return {'excluded': True}
        return write_bundle(job['directory'],bundle)
    if kind == 'record':
        from ..storage import ResultRepository
        from ..storage.lifecycle import append
        from ..environment import collect
        result = job['result']
        if job['save']:
            ResultRepository(job['database']).save(result, environment=dict(collect(Path(job['home'])),**job.get('provenance',{})), attempt_id=job['attempt_id'])
            for name, observed in job['events'].items():
                append(job['database'], job['attempt_id'], name, observed)
        return {'saved': job['save'], 'status': result['status']}
    config = ExecutionConfig(**job['config'])
    import torch
    torch.set_num_threads(config.compile_threads)
    evaluation = EvaluationConfig(**job['evaluation'])
    adapter = create_backend(config)
    adapter.validated_shared_case = True
    if kind == 'compare':
        import torch
        adapter.torch = torch
        actual = read_arrays(job['directory'], job['outputs'])
        expected = read_arrays(job['directory'], job['reference'])
        result = job['result']
        telemetry = json.loads(result['execution']['telemetry'])
        diagnostic = json.loads(result['execution']['diagnostics'])
        for row in result['shape_results']:
            index = row['shape_index']
            if row.get('error_stage') or index >= len(actual):
                continue
            start = time.monotonic_ns()
            if evaluation.correctness:
                passed = adapter.compare(actual[index], expected[index])
                row['correctness'] = 'pass' if passed else 'fail'
                if not passed:
                    diagnostic['shapes'][str(index)] = dict(adapter.diagnose_accuracy(actual[index], expected[index]), cpu_pass=False, secondary_used=False, basis='cpu')
            metrics = telemetry['shapes'][str(index)]
            metrics['correctness'] = time.monotonic_ns()-start
            metrics['shape_processing'] = time.monotonic_ns()-job['shape_starts'][str(index)]
        if result['shape_results']:
            result['status'] = 'error' if any(r.get('error_stage') for r in result['shape_results']) else ('fail' if any(r['correctness']=='fail' for r in result['shape_results']) else 'ok')
        result['execution']['telemetry'] = json.dumps(telemetry)
        result['execution']['diagnostics'] = json.dumps(diagnostic)
        return {'result': result}
    bundle = bundle_from(job['bundle'],job['directory'])
    outputs, starts = [], {}
    persistent_runtime = adapter.capabilities.persistent_runtime
    from ..core_logs import capture_tail
    # The reference is evaluated exactly once. Its CPU row is not_checked:
    # executing the same eager graph twice is not independent validation.
    # Target backends consume this artifact in the compare stage.
    with capture_tail(forward=True) as log:
        result = execute(bundle, adapter, evaluation, deferred=True, outputs=outputs, shape_starts=starts)
    if result['status'] in {'error','unsupported'} and log():
        diag = json.loads(result['execution']['diagnostics']); diag['error'] = log()
        result['execution']['diagnostics'] = json.dumps(diag)
    artifact = write_arrays(Path(job['directory'])/job['output_file'], outputs)
    # execute finally closes the physical lease only after all .cpu() transfers.
    # Drop callable/tensors and compiler state before publishing availability.
    del outputs, adapter, bundle
    import torch
    if hasattr(torch, '_dynamo'):
        torch._dynamo.reset()
    gc.collect()
    device_memory = None
    if config.device != 'cpu' and torch.npu.is_initialized():
        torch.npu.synchronize(config.device)
        # execute() restores the caller's device; clear the leased card's cache.
        with torch.npu.device(config.device):
            torch.npu.empty_cache()
        device_memory = int(torch.npu.memory_allocated(config.device))
        peak=int(torch.npu.max_memory_allocated(config.device))
        if job.get('device_budget_bytes') and peak>job['device_budget_bytes']:
            raise ValueError('device memory budget exceeded')
    return {'result': result, 'outputs': artifact, 'shape_starts': starts, 'device_memory_bytes': device_memory,
            'persistent_runtime': persistent_runtime}


def main(fd):
    sock = socket.socket(fileno=int(fd))
    count = 0
    while True:
        try:
            job = receive(sock)
        except EOFError:
            return
        started = time.monotonic_ns()
        try:
            result = perform(job)
            response = {'ok': True, 'value': result}
        except Exception as exc:
            from ..failure_details import excerpt
            response = {'ok': False, 'error': excerpt(exc)}
        response.update(started=started, finished=time.monotonic_ns(), worker_pid=os.getpid(), temperature='cold' if count==0 else 'warm')
        count += 1
        send(sock, response)


if __name__ == '__main__':
    main(sys.argv[1])
