"""Versioned environment preparation; no installation into an active environment."""
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import uuid

from .artifacts import atomic, command, confined, download, expand, profile, sha256, trusted
from .config import canonical, fingerprint, validate


def plan(store, spec):
    spec = validate(spec)
    if spec['kind'] != 'environment': raise ValueError('environment required')
    trust = trusted(store.home)
    combination = trust.get('compatibility', {}).get(spec['compatibility_profile'])
    if not combination:
        raise PermissionError('compatibility profile must be authorized locally')
    if combination.get('python') != spec['python'] or combination.get('packages') != spec['packages']:
        raise ValueError('Python/package combination does not match trusted profile')
    if combination.get('architecture') != platform.machine():
        raise ValueError('architecture does not match trusted profile')
    cann = spec['cann']
    if cann['source'] == 'existing':
        path = confined(cann['path'], trust.get('existing_roots', []))
        if not path.is_dir(): raise ValueError('CANN directory does not exist')
    else:
        root = Path(cann.get('install_root', '~/Ascend')).expanduser().resolve()
        confined(root, trust.get('install_roots', [str(Path.home()/'Ascend')]))
        if cann['source'] == 'download':
            for package in cann['packages']:
                adapter = profile(store.home, package['adapter'], 'cann-installer')
                if package['sha256'] not in trust.get('approved_artifacts', []):
                    raise PermissionError('RUN artifact hash requires local owner approval')
                if package['kind'] not in adapter.get('package_kinds', []):
                    raise ValueError('package type not supported by adapter')
        else:
            build = cann['build']; recipe = profile(store.home, build['profile'], 'build')
            confined(build['source_dir'], recipe.get('source_roots', []))
    python = combination.get('python_executable', sys.executable)
    if not Path(python).is_file(): raise ValueError('supported Python executable is missing')
    if shutil.disk_usage(store.home).free < combination.get('required_free_bytes', 2*1024**3):
        raise ValueError('insufficient disk reserve for environment')
    identity = str(uuid.uuid4())
    value = dict(id=identity, plan_id=identity, spec=spec, trust_hash=fingerprint(trust), status='planned')
    store.put('plan:'+identity, value)
    return value


def build_run(store, spec, operation_id):
    import fcntl
    recipe = profile(store.home, spec['profile'], 'build')
    source = confined(spec['source_dir'], recipe.get('source_roots', []))
    build_id = str(uuid.uuid4()); directory = store.home/'builds'/build_id
    directory.mkdir(parents=True, mode=0o700)
    lock_dir = store.home/'locks'; lock_dir.mkdir(exist_ok=True)
    with (lock_dir/('build-'+fingerprint(str(source)))).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from .provenance import evidence
        inputs = recipe.get('snapshot_inputs', [])
        before = evidence(source, inputs)
        if spec.get('mode', 'incremental') == 'snapshot':
            # Only explicit profile inputs are snapshotted; never copy arbitrary home secrets.
            snapshot = directory/'source'; snapshot.mkdir()
            for relative in recipe.get('snapshot_inputs', []):
                item = confined(source/relative, [source])
                if Path(relative).is_absolute() or '..' in Path(relative).parts:
                    raise ValueError('snapshot input must be source-relative')
                target = snapshot/relative; target.parent.mkdir(parents=True, exist_ok=True)
                if item.is_dir():
                    if any(p.is_symlink() for p in item.rglob('*')):
                        raise ValueError('snapshot symlinks require explicit materialization')
                    shutil.copytree(item, target)
                elif item.is_file(): shutil.copy2(item, target)
            if not recipe.get('snapshot_inputs'): raise ValueError('profile has no snapshot input manifest')
            source = snapshot
        before = evidence(source, inputs)
        output = directory/'output'; output.mkdir()
        values = dict(source=str(source), output=str(output), jobs=str(spec.get('jobs',1)))
        store.event(operation_id, 'Building locally authorized source profile')
        command(expand(recipe['argv'], values), cwd=source, timeout=recipe.get('timeout',3600))
        packages = []
        for expected in recipe['outputs']:
            artifact = confined(output/expected['path'], [output])
            if not artifact.is_file(): raise ValueError('build did not produce declared RUN artifact')
            digest = sha256(artifact)
            packages.append(dict(kind=expected['kind'], adapter=expected['adapter'], path=str(artifact), sha256=digest))
        after = evidence(source, inputs)
        result = dict(build_id=build_id, source=str(source), mode=spec.get('mode','incremental'),
                      source_before=before, source_after=after,
                      reproducible=bool(inputs) and spec.get('mode')=='snapshot' and before['input_hash']==after['input_hash'],
                      profile_hash=fingerprint(recipe), packages=packages)
        atomic(directory/'manifest.json', result)
        return result


def apply(store, plan_id, operation_id):
    value = store.get('plan:'+plan_id)
    if not value: raise LookupError('environment plan not found')
    trust = trusted(store.home)
    if value['trust_hash'] != fingerprint(trust): raise ValueError('local trust changed; create a new plan')
    spec = value['spec']; cann = spec['cann']; combo = trust['compatibility'][spec['compatibility_profile']]
    identity = str(uuid.uuid4()); directory = store.home/'environments'/identity
    directory.mkdir(parents=True, mode=0o700)
    with store.connect() as db:
        db.execute('INSERT INTO environments VALUES (?,?,?,?)', (identity, spec['name'], canonical(dict(spec=spec)), 'installing'))
    try:
        build = None
        if cann['source'] == 'existing':
            cann_path = confined(cann['path'], trust.get('existing_roots', []))
        else:
            root = Path(cann.get('install_root','~/Ascend')).expanduser().resolve()
            confined(root, trust.get('install_roots',[str(Path.home()/'Ascend')]))
            cann_path = root/('autofusex-'+identity)
            cann_path.mkdir(parents=True, mode=0o700)
            if cann['source'] == 'build':
                build = build_run(store, cann['build'], operation_id); packages = build['packages']
            else: packages = cann['packages']
            for package in packages:
                adapter = profile(store.home, package['adapter'], 'cann-installer')
                if not adapter.get('isolated_install'):
                    raise PermissionError('only verified isolated-install adapters are supported')
                if package.get('path'):
                    artifact = Path(package['path'])
                else:
                    store.event(operation_id, 'Downloading verified CANN artifact')
                    artifact = download(package['url'], package['sha256'], store.home/'artifacts', allowed_hosts=trust.get('allowed_hosts',[]))
                if sha256(artifact) != package['sha256']: raise ValueError('artifact changed before installation')
                values = dict(artifact=str(artifact), prefix=str(cann_path))
                store.event(operation_id, 'Running authorized user-space CANN installer')
                command(expand(adapter['argv'], values), timeout=adapter.get('timeout',3600))
                command(expand(adapter['verify_argv'], values), timeout=120)
        python = combo.get('python_executable', sys.executable)
        venv = directory/'venv'
        store.event(operation_id, 'Preparing isolated Python environment')
        command([python, '-m', 'venv', str(venv)])
        executable = venv/'bin/python'
        wheel_dir = directory/'wheels'; wheel_dir.mkdir()
        requirements = []
        for wheel in combo['wheels']:
            artifact = download(wheel['url'],wheel['sha256'],store.home/'artifacts',allowed_hosts=trust.get('allowed_hosts',[]))
            filename = wheel['filename']
            if Path(filename).name != filename or not filename.endswith('.whl'): raise ValueError('invalid wheel filename')
            shutil.copy2(artifact, wheel_dir/filename)
            requirements.append(f"{wheel['name']}=={wheel['version']} --hash=sha256:{wheel['sha256']}")
        requirements_path = directory/'requirements.lock'
        requirements_path.write_text('\n'.join(requirements)+'\n')
        command([str(executable), '-m', 'pip', 'install', '--no-index', '--find-links', str(wheel_dir),
                 '--require-hashes', '-r', str(requirements_path)], timeout=1800)
        env = execution_env(cann_path, combo)
        store.event(operation_id, 'Validating installed imports (no NPU workload)')
        versions={wheel['name']:wheel['version'] for wheel in combo['wheels']}
        command([str(executable), '-c', 'import importlib.metadata as m,json,sys; expected=json.loads(sys.argv[1]); assert all(m.version(k)==v for k,v in expected.items())',canonical(versions)],env=env,timeout=120)
        verify_code = combo.get('verify_code', 'import torch, torch_npu; from autofusex.fuzz.generation.nnsmith_v3 import source_evidence; source_evidence()')
        command([str(executable), '-c', verify_code], env=env, timeout=120)
        manifest = dict(id=identity, name=spec['name'], spec=spec, python=str(executable), cann_path=str(cann_path),
                        env=env, build=build, verification='imports_only', npu_verified=False,
                        compatibility_hash=fingerprint(combo))
        # Store only the explicit environment additions, not the host's secrets.
        manifest['env'] = {k:v for k,v in env.items() if k in combo.get('env',{}) or k == 'ASCEND_HOME_PATH'}
        atomic(directory/'manifest.json',manifest)
        with store.connect() as db:
            db.execute('UPDATE environments SET manifest=?,state=? WHERE id=?',(canonical(manifest),'ready',identity))
        return manifest
    except Exception:
        with store.connect() as db: db.execute("UPDATE environments SET state='needs_attention' WHERE id=?",(identity,))
        raise


def execution_env(cann_path, combo):
    env = os.environ.copy()
    env['ASCEND_HOME_PATH'] = str(cann_path)
    for key, value in combo.get('env',{}).items():
        env[key] = value.replace('{cann}',str(cann_path))
    return env


def resolve(store, name):
    with store.connect() as db:
        rows = db.execute("SELECT manifest FROM environments WHERE state='ready' AND (id=? OR name=?)",(name,name)).fetchall()
    if len(rows) != 1: raise ValueError('environment missing or ambiguous; use exact environment ID')
    return json.loads(rows[0][0])
