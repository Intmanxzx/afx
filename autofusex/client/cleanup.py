"""Background retention for an allowlist of client-owned directories.

No configured paths, no symlink traversal, and no deletion of resumable runs.
Shared readers cover replay, admission and uploads; exclusive detach is brief.
"""
from contextlib import contextmanager, closing, ExitStack
from functools import wraps
import fcntl
import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import sys
import time
import uuid

from autofusex.contracts.retention import policy
from autofusex.contracts.configuration import canonical

TERMINAL = {'completed', 'failed', 'stopped', 'cancelled'}


@contextmanager
def access(home, *, exclusive=False):
    with (Path(home) / 'retention.lock').open('a') as lock:
        fcntl.flock(lock, (fcntl.LOCK_EX | fcntl.LOCK_NB) if exclusive else fcntl.LOCK_SH)
        yield


def reader(function):
    @wraps(function)
    def wrapped(store, *args, **kwargs):
        with access(store.home):
            return function(store, *args, **kwargs)
    return wrapped


def settings(store):
    path = store.home / 'task.json'
    config = json.loads(path.read_text()) if path.exists() else {}
    return policy(config.get('cleanup', {}))


def _tree(path):
    """Yield lstat entries, refusing trees containing any symlink."""
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        raise ValueError('symlink in retained tree')
    yield path, info
    if stat.S_ISDIR(info.st_mode):
        with os.scandir(path) as entries:
            for entry in entries:
                yield from _tree(Path(entry.path))


def _measure(path):
    newest = size = 0
    for _, info in _tree(path):
        newest = max(newest, info.st_mtime)
        if stat.S_ISREG(info.st_mode):
            size += info.st_size
    return newest, size


def _read_db(path):
    if path.is_symlink():
        raise ValueError('symlink database')
    return closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=.1))


def _uploaded(directory, spec):
    if not spec['results']['upload']:
        return True
    if spec['kind'] != 'fuzz_task':
        receipt = directory / 'source-upload.json'
        return receipt.is_file() and bool(json.loads(receipt.read_text()).get('job_id'))
    database = directory / 'results.sqlite'
    if not database.exists():
        # Missing archives cannot prove that all results were delivered.
        return False
    with _read_db(database) as db:
        maximum = db.execute('SELECT coalesce(max(rowid),0) FROM attempts').fetchone()[0]
    if maximum == 0:
        return True
    with _read_db(directory / 'upload.sqlite') as db:
        cursor = db.execute("SELECT value FROM progress WHERE key='fuzz:cursor'").fetchone()
        rejected = db.execute("SELECT 1 FROM rejected WHERE reason IS NULL OR reason NOT LIKE 'environment:%' LIMIT 1").fetchone()
    return bool(cursor and cursor[0] >= maximum and not rejected)


def _protected(store, identity):
    switch = store.get('client_switch', {})
    if switch.get('state') not in {None, 'succeeded', 'completed', 'idle'} and identity in switch.get('old_runs', []):
        return True
    source = store.get('source_job')
    if source:
        operation = store.operation(source['operation_id'])
        if (operation.get('result') or {}).get('run_id') == identity:
            return True
    directory = store.home / 'runs' / identity
    for record in (store.home / 'retest').glob('task-*.json'):
        saved = json.loads(record.read_text())
        if not saved.get('delivered') and any(Path(saved[key]).parent == directory for key in ('queue', 'database')):
            return True
    return False


@contextmanager
def _queue_idle(directory):
    path = directory / 'tasks.sqlite'
    if not path.exists():
        yield
        return
    lockpath = Path(str(path) + '.owner.lock') if sys.platform == 'darwin' else path
    fd = os.open(lockpath, os.O_RDWR | os.O_CREAT, 0o600) if sys.platform == 'darwin' else os.open(lockpath, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def _runs(store, days, now, report):
    root = store.home / 'runs'
    if not days or not root.exists():
        return
    if root.is_symlink():
        raise ValueError('symlink runs root')
    with store.connect() as db:
        rows = list(db.execute('SELECT * FROM runs'))
    for row in rows:
        identity = row['id']
        if Path(identity).name != identity or identity in {'.', '..'}:
            continue
        directory = root / identity
        if not directory.exists():
            continue
        try:
            if row['state'] not in TERMINAL:
                report['runs_skipped'] += 1
                continue
            previous = json.loads(row['result']) if row['result'] else {}
            ended = previous.get('ended_at')
            if ended is None:
                # Legacy runs get one conservative observation timestamp.
                previous['ended_at'] = now
                with store.connect() as db:
                    db.execute('UPDATE runs SET result=? WHERE id=? AND state=? AND result IS ?', (canonical(previous),identity,row['state'],row['result']))
                continue
            if ended >= now - days * 86400:
                continue
            _, size = _measure(directory)
            # Readers never wait on recursive deletion: detach under the lock,
            # then delete the detached tree outside it in this background thread.
            with access(store.home, exclusive=True), _queue_idle(directory):
                with store.connect() as db:
                    current = db.execute('SELECT * FROM runs WHERE id=?', (identity,)).fetchone()
                if current['state'] not in TERMINAL or _protected(store, identity):
                    report['runs_skipped'] += 1
                    continue
                spec = json.loads(current['spec'])['spec']
                if not _uploaded(directory, spec):
                    report['runs_skipped'] += 1
                    continue
                summary = json.loads(current['result']) if current['result'] else {}
                if summary.get('ended_at', now) >= now - days * 86400:
                    continue
                if (directory / 'results.sqlite').exists():
                    with _read_db(directory / 'results.sqlite') as db:
                        summary['result_count'] = db.execute('SELECT count(*) FROM attempts').fetchone()[0]
                trash = root / ('.cleanup-' + identity)
                if trash.exists():
                    raise ValueError('unfinished cleanup')
                # Journal before rename makes a crash between detach and delete recoverable.
                summary['cleanup'] = dict(state='deleting', at=now, bytes=size)
                with store.connect() as db:
                    db.execute('UPDATE runs SET result=? WHERE id=?', (canonical(summary), identity))
                directory.rename(trash)
            shutil.rmtree(trash)
            summary['cleanup']['state'] = 'cleaned'
            with store.connect() as db:
                db.execute('UPDATE runs SET result=? WHERE id=?', (canonical(summary), identity))
            report['runs_removed'] += 1
            report['bytes_removed'] += size
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
            report['runs_skipped'] += 1
    # Only our durable journal authorizes finishing an interrupted deletion.
    for row in rows:
        summary = json.loads(row['result']) if row['result'] else {}
        if summary.get('cleanup', {}).get('state') != 'deleting':
            continue
        identity = row['id']
        if Path(identity).name != identity or identity in {'.', '..'}:
            continue
        trash = root / ('.cleanup-' + identity)
        if (root / identity).exists():
            continue
        if trash.exists():
            _measure(trash)
            shutil.rmtree(trash)
        summary['cleanup']['state'] = 'cleaned'
        with store.connect() as db:
            db.execute('UPDATE runs SET result=? WHERE id=?', (canonical(summary), identity))


def debug_worker(home):
    """A lease inherited by workers protects compiler dumps after parent death."""
    from autofusex.client.paths import debug_directory
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    with access(home):
        directory = debug_directory(home) / ('worker-' + str(uuid.uuid4()))
        directory.mkdir(mode=0o700)
        lease = (directory / '.active.lock').open('a')
        fcntl.flock(lease, fcntl.LOCK_EX)
    return directory, lease


def _debug(store, days, now, report):
    root = store.home / 'debug'
    if not days or not root.exists():
        return
    if root.is_symlink():
        raise ValueError('symlink debug root')
    # Legacy dumps have no leases. Never touch them while any old task owns the
    # managed/standalone queue; new per-worker trees can still be reclaimed.
    with store.connect() as db:
        busy = bool(db.execute("SELECT 1 FROM runs WHERE state NOT IN ('completed','failed','stopped','cancelled') LIMIT 1").fetchone())
    def remove(path, size, files):
        if path.is_dir(): shutil.rmtree(path)
        else: path.unlink()
        report['debug_files_removed'] += files
        report['bytes_removed'] += size
    for group in root.iterdir():
        if group.is_symlink() or not group.is_dir():
            continue
        for directory in group.iterdir():
            if directory.is_symlink():
                continue
            try:
                leasepath = directory / '.active.lock' if directory.is_dir() else None
                with ExitStack() as stack:
                    leased = bool(leasepath and leasepath.is_file() and not leasepath.is_symlink())
                    if leased:
                        lease = stack.enter_context(leasepath.open('r'))
                        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    elif busy:
                        continue
                    newest = size = files = 0
                    for _, info in _tree(directory):
                        newest = max(newest, info.st_mtime)
                        if stat.S_ISREG(info.st_mode):
                            size += info.st_size
                            files += 1
                    if newest >= now - days * 86400:
                        continue
                    if leased:
                        remove(directory, size, files)
                        continue
                if not busy:
                    # Legacy trees have no worker lease. Detach while task
                    # admission and queue takeover are fenced, then delete
                    # outside the fence so IPC remains responsive.
                    with access(store.home, exclusive=True), _queue_idle(store.home / 'tasks'):
                        with (store.home / 'managed-task.lock').open('a') as lock:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            # A worker may have finished acquiring its lease
                            # since the initial scan. Recheck under the fence.
                            if leasepath and leasepath.exists():
                                continue
                            trash = group / ('.cleanup-' + str(uuid.uuid4()))
                            directory.rename(trash)
                    remove(trash, size, files)
            except BlockingIOError:
                report['debug_active'] += 1
            except (OSError, ValueError):
                report['debug_skipped'] += 1


def sweep(store, now=None):
    now = time.time() if now is None else now
    report = dict(at=now, state='ok', runs_removed=0, runs_skipped=0,
                  debug_files_removed=0, debug_active=0, debug_skipped=0, bytes_removed=0)
    try:
        configured = settings(store)
        report['policy'] = configured
        _debug(store, configured['debug_days'], now, report)
        _runs(store, configured['runs_days'], now, report)
    except Exception as exc:
        # Paths and exception text can contain private data.
        report.update(state='error', error=type(exc).__name__)
    store.put('cleanup_state', report)
    return report
