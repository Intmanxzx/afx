"""Persistent device processes drawing tasks from a parent-owned pool."""
import multiprocessing
from queue import Empty
import traceback


def _worker(index, config, tasks, results, initialize, execute):
    try:
        runtime = initialize(index, config)
        results.put(('ready', index, None))
        while True:
            task = tasks.get()
            if task is None:
                return
            results.put(('result', index, execute(runtime, task, config)))
    except BaseException:
        results.put(('error', index, traceback.format_exc()))
        raise


def dynamic_results(tasks, *, workers, config, initialize, execute):
    """Assign one task per ready worker; no worker owns a private backlog."""
    if workers < 1:
        raise ValueError('workers must be positive')
    ctx = multiprocessing.get_context('spawn')
    results = ctx.Queue()
    queues = [ctx.Queue(maxsize=1) for _ in range(workers)]
    processes, active = [], set()
    iterator = iter(tasks)
    try:
        for i, queue in enumerate(queues):
            process = ctx.Process(target=_worker, args=(i,config,queue,results,initialize,execute))
            process.start()
            processes.append(process)
            active.add(i)
        while active:
            try:
                kind, i, payload = results.get(timeout=.5)
            except Empty:
                dead = [i for i in active if processes[i].exitcode is not None]
                if dead:
                    raise RuntimeError(f'Fuzz workers exited before returning results: {dead}')
                continue
            if kind == 'error':
                raise RuntimeError(f'Fuzz worker {i} failed: {payload}')
            if kind == 'result':
                yield payload
            try:
                task = next(iterator)
            except StopIteration:
                queues[i].put(None)
                active.remove(i)
            else:
                queues[i].put(task)
        for process in processes:
            process.join(timeout=10)
            if process.exitcode != 0:
                raise RuntimeError(f'Fuzz worker exited with {process.exitcode}')
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=20)
            if process.is_alive():
                process.kill()
                process.join()
        for queue in queues + [results]:
            queue.cancel_join_thread()
            queue.close()
