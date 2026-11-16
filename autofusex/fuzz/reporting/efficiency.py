"""Finite-window throughput and non-additive processing latency statistics."""
import json
import math
from ..storage.lifecycle import read, elapsed


def percentile(values,p):
    if not values:return None
    ordered=sorted(values)
    return ordered[max(0,math.ceil(len(ordered)*p)-1)]


def union(intervals):
    total=0;end=None
    for a,b in sorted(intervals):
        if b<a:continue
        total+=max(0,b-max(a,end if end is not None else a));end=max(b,end or b)
    return total


def statistics(conn,selected_ids=None, *, use_cached=False, window_id=None):
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='stage_windows_v1'").fetchone():return []
    output=[]
    windows = conn.execute('SELECT window_id,payload FROM stage_windows_v1 ORDER BY rowid DESC LIMIT 100') if window_id is None else conn.execute('SELECT window_id,payload FROM stage_windows_v1 WHERE window_id=?', (window_id,))
    for window_id,raw in windows:
        window=json.loads(raw);end=window.get('finished');start=window['start']
        if not end or end['domain']!=start['domain']:continue
        elapsed_ns=end['monotonic_ns']-start['monotonic_ns']
        if elapsed_ns<=0:continue
        # A finished local window can arrive before all of its result uploads.
        incomplete = {r[0] for r in conn.execute("""
            SELECT f.attempt_id FROM stage_attempts_v1 x
            JOIN attempt_facts f ON f.attempt_id=x.attempt_id
            WHERE x.window_id=? AND EXISTS (
                SELECT 1 FROM attempt_facts peer,
                     json_each(json_extract(peer.execution,'$.expected_backends')) expected
                WHERE peer.group_id=f.group_id AND NOT EXISTS (
                    SELECT 1 FROM attempt_facts done JOIN attempts a ON a.id=done.attempt_id
                    WHERE done.group_id=f.group_id AND a.backend=expected.value))
            """, (window_id,))}
        if use_cached and selected_ids is None and not incomplete:
            cached=conn.execute('SELECT payload FROM stage_window_aggregates_v1 WHERE window_id=?',(window_id,)).fetchone()
            if cached:
                output.extend(json.loads(cached[0]));continue
        if not conn.execute('SELECT 1 FROM stage_facts_v1 WHERE window_id=?',(window_id,)).fetchone():
            cached=conn.execute('SELECT payload FROM stage_window_aggregates_v1 WHERE window_id=?',(window_id,)).fetchone()
            if cached and selected_ids is None and not incomplete:output.extend(json.loads(cached[0]))
            continue
        attempts=conn.execute('SELECT a.id,a.backend,a.status,a.workload_id,a.evaluation_id,a.environment_id,a.execution_id FROM stage_attempts_v1 x JOIN attempts a ON a.id=x.attempt_id WHERE x.window_id=?',(window_id,)).fetchall()
        by_backend={}
        for identity,backend,status,workload,ev_id,env_id,ex_id in attempts:
            if backend in {'cpu'} or identity in incomplete:continue
            if selected_ids is not None and identity not in selected_ids:continue
            ev=json.loads(conn.execute('SELECT payload FROM contexts WHERE id=?',(ev_id,)).fetchone()[0])
            ex=json.loads(conn.execute('SELECT payload FROM contexts WHERE id=?',(ex_id,)).fetchone()[0])
            # Environment context hash keeps software/hardware populations apart.
            own_facts=[json.loads(r[0]) for r in conn.execute('SELECT payload FROM stage_facts_v1 WHERE attempt_id=?',(identity,))]
            temperature=next((f.get('temperature','uncollected') for f in own_facts if f['kind']=='first_call_combined'),'uncollected')
            env=json.loads(conn.execute('SELECT payload FROM contexts WHERE id=?',(env_id,)).fetchone()[0])
            key=(backend,env_id,'static' if ex.get('static_only') else 'dynamic',temperature,status)
            r=by_backend.setdefault(key,dict(backend=backend,environment_id=env_id,mode=key[2],temperature=temperature,status=status,hardware=env.get('device_model'),software={k:env.get(k) for k in ('torch','torch_npu','cann','autofuse_binary','package_version')},cache_state='reset_per_case',completed=0,success=0,failed=0,case_errors=0,attempts=0,unique=set(),processing=[],e2e=[],cold=0,warm=0,retry_attempts=0))
            r['attempts']+=1
            shape_rows=conn.execute('SELECT shape_index,correctness,performance FROM shapes WHERE attempt_id=?',(identity,)).fetchall()
            if not shape_rows:r['case_errors']+=1
            telemetry=json.loads(ex.get('telemetry','{}'))
            for index,correctness,performance in shape_rows:
                if correctness not in {'pass','fail','error'} and performance not in {'measured','error'}:continue
                r['completed']+=1;r['unique'].add((workload,index))
                success=(not ev.get('correctness',True) or correctness=='pass') and (not ev.get('performance',False) or performance=='measured')
                r['success']+=int(success);r['failed']+=int(not success)
                processing=telemetry.get('shapes',{}).get(str(index),{}).get('shape_processing')
                if processing is not None:r['processing'].append(processing)
                e2e=elapsed(read(conn,identity),'admitted','persisted')
                if e2e is not None:r['e2e'].append(e2e)
            for fact, in conn.execute('SELECT payload FROM stage_facts_v1 WHERE attempt_id=?',(identity,)):
                f=json.loads(fact)
                if f['kind']=='record':r['retry_attempts']+=int(f.get('retry_index',0)>0)
                if f['kind']=='first_call_combined' and f.get('temperature') in {'cold','warm'}:r[f['temperature']]+=1
        facts=[json.loads(r[0]) for r in conn.execute('SELECT payload FROM stage_facts_v1 WHERE window_id=?',(window_id,))]
        token_ns=0;device_intervals={d:[] for d in window['devices']}
        for f in facts:
            lease=f.get('lease',{})
            if not lease:continue
            a=max(start['monotonic_ns'],lease['acquired']);b=min(end['monotonic_ns'],lease['released'])
            token_ns+=max(0,b-a)*lease['cpu_tokens']
            if lease.get('device_uuid') in device_intervals:device_intervals[lease['device_uuid']].append((a,b))
        for r in by_backend.values():
            for name in ('processing','e2e'):
                values=r.pop(name);r[name+'_samples']=len(values);r[name+'_p50_ns']=percentile(values,.5);r[name+'_p95_ns']=percentile(values,.95)
            r['unique_shapes']=len(r.pop('unique'))
            r.update(window_id=window_id,interrupted_attempts=window.get('interrupted_attempts',0),cancelled_attempts=window.get('cancelled_attempts',0),record_failed_attempts=window.get('record_failed_attempts',0),window_seconds=elapsed_ns/1e9,
                     success_shapes_per_min=r['success']*60e9/elapsed_ns,completed_shapes_per_min=r['completed']*60e9/elapsed_ns,
                     amortized_seconds_per_shape=elapsed_ns/1e9/r['completed'] if r['completed'] else None,
                     cpu_pool_utilization=token_ns/(elapsed_ns*window['cpu_tokens']),
                     npu_slot_utilization={d:union(xs)/elapsed_ns for d,xs in device_intervals.items()},
                     measured_cpu_utilization=None,measured_aicore_utilization=None,measured_memory_utilization=None)
            output.append(r)
    return output
