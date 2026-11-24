"""Isolated retest runner using the registered execution environment."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from autofusex.client.paths import retest_directory


def stop(manager):
    child = getattr(manager, '_retest_process', None)
    if child is not None:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
        manager._retest_process = None


def tick(manager):
    store = manager.store
    retest_dir = retest_directory(store.home)
    path = store.home / 'client.json'
    if not path.exists():
        return
    client = json.loads(path.read_text())
    command = store.home / 'daemon-control.json'
    desired = json.loads(command.read_text()).get('state') if command.exists() else None
    enabled = (client.get('mode') == 'managed' and client.get('platform', {}).get('retest', False)
               and bool(manager.processes) and desired != 'stopped'
               and store.get('client_switch', {}).get('state') not in {'draining', 'staging', 'requested', 'activating', 'stage_failed'})
    if not enabled:
        stop(manager)
        return
    child = getattr(manager, '_retest_process', None)
    if child is not None:
        if child.poll() is None:
            return
        store.put('retest_runner', {'exit_code': child.returncode, 'at': time.time()})
        manager._retest_process = None
    if time.monotonic() < getattr(manager, '_retest_next', 0):
        return
    manager._retest_next = time.monotonic() + 30
    with store.connect() as db:
        row = db.execute("SELECT manifest FROM environments WHERE id='local' AND state='ready'").fetchone()
    if row is None:
        return
    manifest = json.loads(row['manifest'])
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.update(manifest['env'])
    env['AUTOFUSEX_HOME'] = str(store.home)
    env['AFX_TASK_HOME'] = str(store.home)
    with (retest_dir / 'runner.log').open('ab') as log:
        manager._retest_process = subprocess.Popen(
            [manifest['python'], '-c',
             "import runpy,sys; path=sys.argv.pop(1); sys.path.pop(0); runpy.run_path(path,run_name='__main__')",
             str(Path(__file__).resolve()), str(store.home)],
            env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)


class JournalAPI:
    """Persist claims so a runner restart resumes the same server tasks."""
    def __init__(self, api, path):
        self.api, self.path = api, path

    def call(self, method, route, body=None, **kwargs):
        from autofusex.client.artifacts import atomic
        from autofusex.client.platform_transport import APIError
        if route == '/api/client/retests/claim':
            saved = json.loads(self.path.read_text()) if self.path.exists() else {}
            if saved.get('tasks'):
                return {'tasks': saved['tasks']}
            request = saved.get('request') or body
            atomic(self.path, {'request': request})
            result = self.api.call(method, route, request, **kwargs)
            atomic(self.path, {'tasks': result['tasks']})
            return result
        if route.startswith('/api/client/retests/') and route.endswith('/complete'):
            try:
                result = self.api.call(method, route, body, **kwargs)
            except APIError as exc:
                if exc.code != 'lease_lost':
                    raise
                result = {'state': 'lease_lost'}
            saved = json.loads(self.path.read_text()) if self.path.exists() else {}
            task_id = int(route.split('/')[-2])
            atomic(self.path, {'tasks': [t for t in saved.get('tasks', []) if t['task_id'] != task_id]})
            return result
        return self.api.call(method, route, body, **kwargs)


def run(home):
    import fcntl
    import logging
    from autofusex.client.platform_transport import API, Client
    from autofusex.execution.environment import collect
    from autofusex.execution.scheduling.devices import device_inventory_info
    logging.basicConfig(level=logging.INFO)
    home = Path(home)
    retest_dir = retest_directory(home)
    with (retest_dir / 'runner.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        config = json.loads((home / 'client.json').read_text())
        task = json.loads((home / 'task.json').read_text())
        devices = task['execution']['devices']
        available = device_inventory_info()
        if 'all' in devices:
            devices = available['devices']
        trust = json.loads((home / 'trust.json').read_text())
        allowed = trust.get('resource_limits', {}).get('allowed_devices', [])
        devices = [d for d in devices if d in allowed or 'all' in allowed]
        api = JournalAPI(API(config['endpoint'], config['token'], timeout=15), retest_dir/'claims.json')
        from autofusex.client.paths import results_database
        client = Client(api, str(results_database(home)), retest_dir/'upload.sqlite', devices=devices)
        client.env = collect(home)
        client.env['device_model'] = available['model']
        # Do not borrow an identity from an unrelated historical results database.
        client.identity = lambda: dict(client.env)
        try:
            from autofusex.client.retest_queue import synchronize
            print(json.dumps({'finished': synchronize(client, home)}), flush=True)
        finally:
            client.state.conn.close()


if __name__ == '__main__':
    import sys
    run(sys.argv[1])
