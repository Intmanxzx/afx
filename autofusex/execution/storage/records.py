"""Whitelist boundary: never archive tensors, arbitrary environment or raw logs."""
import hashlib
from autofusex.workloads.fuzz.limits import MAX_CONTEXT_BYTES
import json
import re


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('ascii')).hexdigest()


def text(value, limit):
    if not isinstance(value, str):
        raise ValueError('expected text result field')
    return value[:limit]


def integer(value):
    if value is not None and (type(value) is not int or not 0 <= value < 2**63):
        raise ValueError('invalid result metric')
    return value


def _context(raw, fields):
    if not isinstance(raw, dict):
        raise ValueError('context must be an object')
    result = {}
    for key in fields:
        if key not in raw:
            continue
        value = raw[key]
        if not isinstance(value, (str, int, bool)) and value is not None:
            raise ValueError('context must contain scalar values')
        result[key] = value[:256] if isinstance(value, str) else value
    return result


def normalize(result, environment):
    seed = result['seed']
    if not isinstance(seed, str) or len(seed) > 128 or not re.fullmatch(r'0|[1-9][0-9]*', seed):
        raise ValueError('result seed must be canonical decimal text')
    from autofusex.workloads.fuzz.contracts import Seed
    Seed.parse(seed)
    workload = result['workload_id']
    if not isinstance(workload, str) or not re.fullmatch(r'[0-9a-f]{64}', workload):
        raise ValueError('invalid workload fingerprint')
    if result['status'] not in {'ok', 'fail', 'error', 'unsupported'}:
        raise ValueError('invalid attempt status')
    contexts = {
        'execution': _context(result.get('execution', {}), ('backend','device','compile_threads','scheduling','static_only','batch_id','save_results','dump','round_id','capture_version')),
        'evaluation': _context(result.get('evaluation', {}), ('correctness','performance','warmup_steps','measured_steps')),
        'environment': _context(environment, ('python','platform','machine','torch','torch_npu','triton','package_version','cann','autofuse_commit','autofuse_binary','device_name','user_id','installation_id','device_uuid','device_model')),
    }
    expected = result.get('execution', {}).get('expected_backends')
    if expected is not None:
        backends = json.loads(expected) if isinstance(expected, str) else None
        if (not isinstance(backends, list) or not backends or len(backends) > 16 or
                any(not isinstance(b, str) or not re.fullmatch(r'[a-z0-9-]{1,64}', b) for b in backends)):
            raise ValueError('invalid expected backends')
        contexts['execution']['expected_backends'] = canonical(sorted(set(backends)))
    case_spec = result.get('execution', {}).get('case_spec')
    if case_spec is not None:
        from autofusex.workloads.fuzz.case_spec import decode, encode
        spec = decode(case_spec)
        if str(spec.seed) != seed: raise ValueError('case spec seed mismatch')
        contexts['execution']['case_spec'] = encode(spec)
    telemetry = result.get('execution', {}).get('telemetry')
    if telemetry is not None:
        from autofusex.execution.telemetry import validate
        contexts['execution']['telemetry'] = validate(telemetry)
    diagnostics=result.get('execution',{}).get('diagnostics')
    if diagnostics is not None:
        from autofusex.execution.failure_details import validate as validate_diagnostics
        contexts['execution']['diagnostics']=validate_diagnostics(diagnostics)
    if len(canonical(contexts['execution']))>MAX_CONTEXT_BYTES:raise ValueError('execution context too large')
    row = {key: text(result[key], limit) for key, limit in
           (('seed',128),('workload_id',64),('oracle_id',128),('backend',64),('status',32))}
    row.update(prepare_ns=integer(result.get('prepare_ns')),
               error_stage=text(result.get('error_stage', ''),64),
               error_summary=text(result.get('error_summary', ''),500))
    shapes, seen = [], set()
    raw_shapes = result.get('shape_results', [])
    if not isinstance(raw_shapes, list) or len(raw_shapes) > 1024:
        raise ValueError('invalid or excessive shape results')
    for raw in raw_shapes:
        index = integer(raw['shape_index'])
        if index is None or index >= 1024 or index in seen:
            raise ValueError('duplicate or absent shape index')
        seen.add(index)
        if raw['correctness'] not in {'pass','fail','error','not_checked'} or raw['performance'] not in {'measured','not_requested','error','skipped'}:
            raise ValueError('invalid shape status')
        shape = {key: raw[key] for key in ('shape_index','correctness','performance')}
        for key in ('first_call_combined_ns','mean_wall_ns','measured_steps'):
            shape[key] = integer(raw.get(key))
        for key, limit in (('metric',128),('error_stage',64),('error_summary',500)):
            shape[key] = text(raw.get(key, ''), limit)
        shapes.append(shape)
    return {'attempt': row, 'contexts': contexts, 'shapes': shapes}
