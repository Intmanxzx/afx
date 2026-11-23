"""Register immutable fixed cases and submit them through client task admission."""
import argparse
import json
from autofusex.client.paths import data_home
from autofusex.workloads.imported import registry


def main(argv=None):
    parser = argparse.ArgumentParser(prog='afx case')
    parser.add_argument('--data-dir')
    sub = parser.add_subparsers(dest='command', required=True)
    add = sub.add_parser('import', help='copy a Python script or source package into the local registry')
    add.add_argument('path'); add.add_argument('--script'); add.add_argument('--entry')
    add.add_argument('--name'); add.add_argument('--source', choices=('manual','baize_torch','baize_tf'), default='manual')
    add.add_argument('--workload', choices=('imported','model'), default='imported')
    sub.add_parser('list')
    show = sub.add_parser('inspect'); show.add_argument('case_key')
    run = sub.add_parser('run'); run.add_argument('case_key')
    run.add_argument('--environment', default='local'); run.add_argument('--device', action='append')
    run.add_argument('--submit', action='store_true', help='submit to the configured central server');
    run.add_argument('--upload', action='store_true'); run.add_argument('--with-cpu', action='store_true')
    run.add_argument('--fullgraph', action='store_true', default=None); run.add_argument('--dynamic', action='store_true', default=None)
    run.add_argument('--rtol', type=float); run.add_argument('--atol', type=float)
    run.add_argument('--script-arg', action='append', default=[])
    import sys
    argv=list(sys.argv[1:] if argv is None else argv)
    if '--data-dir' in argv:
        index=argv.index('--data-dir')
        if index+1<len(argv):argv=['--data-dir',argv[index+1],*argv[:index],*argv[index+2:]]
    args = parser.parse_args(argv); home = data_home(args.data_dir)
    try:
        if args.command == 'import':
            result = registry.register(home, args.path, script=args.script, entry=args.entry, name=args.name,
                                       origin=args.source, workload=args.workload)
        elif args.command == 'list': result = registry.list_cases(home)
        else:
            result = registry.inspect(home, args.case_key)
            if args.command == 'run':
                from autofusex.client.modern_cli import _legacy
                from autofusex.contracts.ipc import request
                spec = dict(schema_version=1, kind=result['spec']['workload']+'_task', environment=args.environment,
                    workload=dict(case_key=result['case_key'], revision=result['revision'], with_cpu=args.with_cpu),
                    execution=dict(devices=args.device or ['all']), results=dict(save=True, upload=args.upload))
                spec['workload']['script_args'] = args.script_arg
                for key in ('rtol', 'atol', 'fullgraph', 'dynamic'):
                    if getattr(args, key) is not None: spec['workload'][key] = getattr(args, key)
                if args.submit:
                    from autofusex.client.platform_transport import API
                    import uuid
                    config=json.loads((home/'client.json').read_text())
                    if config.get('mode')!='managed':raise ValueError('central submission requires a managed client')
                    api=API(config['endpoint'],config['token'])
                    api.call('POST','/api/source-cases',registry.export(home,args.case_key))
                    result=api.call('POST','/api/source-jobs',dict(request_key=str(uuid.uuid4()),spec=spec))
                    print(json.dumps(result,indent=2));return 0
                from autofusex.client.store import Store
                from autofusex.client.local_session import _local_manifest
                from autofusex.client.artifacts import atomic
                from autofusex.execution.resources import resource_inventory
                from autofusex.client.configuration import validate
                store = Store(home)
                if not (home/'config.json').exists():
                    import uuid
                    atomic(home/'config.json',dict(version=1,source_id=str(uuid.uuid4()),cpu_workers=1,resource_policy='auto'))
                if args.environment == 'local':
                    with store.connect() as db:
                        db.execute("INSERT OR IGNORE INTO environments VALUES (?,?,?,'ready')",
                                   ('local','local',json.dumps(_local_manifest())))
                limits = dict(allowed_devices=args.device or ['all'], max_workers=1,
                              max_compile_threads=resource_inventory(home)['cpu_worker_ceiling'] or 1)
                if not (home/'trust.json').exists():
                    atomic(home/'trust.json', {'resource_limits': limits})
                if not store.get('effective_policy'):
                    store.put('effective_policy', validate(dict(schema_version=1,kind='client_policy',resources=limits)))
                _legacy(['daemon','start','--data-dir',str(home)])
                import uuid
                result = request(home, dict(action='submit', kind='task.submit', payload=spec, request_key=str(uuid.uuid4())))
        print(json.dumps(result, indent=2)); return 0
    except (ValueError, OSError) as exc:
        import sys
        print(f'ERROR: {exc}', file=sys.stderr); return 2
