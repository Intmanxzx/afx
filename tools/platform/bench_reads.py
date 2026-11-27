"""Measure named GET scenarios without printing credentials or response bodies."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import statistics
import time
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--scenarios', required=True)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--rate', type=float, default=0, help='mixed requests/second; 0 runs each scenario separately')
    parser.add_argument('--duration', type=int, default=30)
    args = parser.parse_args()
    if not 1 <= args.runs <= 100 or not 1 <= args.concurrency <= 16:
        parser.error('runs must be 1..100; concurrency must be 1..16')
    if not 0 <= args.rate <= 100 or not 1 <= args.duration <= 300:
        parser.error('rate must be 0..100; duration must be 1..300 seconds')
    if not args.base_url.startswith(('http://', 'https://')):
        parser.error('base URL must use HTTP or HTTPS')
    with open(args.scenarios) as source:
        scenarios = json.load(source)
    if not isinstance(scenarios, dict) or not scenarios or any(
            not isinstance(path, str) or not path.startswith(('/api/', '/healthz'))
            or '#' in path for path in scenarios.values()):
        parser.error('scenarios must map names to local /api/ or /healthz GET paths')
    headers = {'Accept': 'application/json'}
    token = os.environ.get('AFX_BENCH_TOKEN')
    if token:
        headers['Authorization'] = 'Bearer ' + token

    def measure(path):
        start = time.perf_counter()
        try:
            request = urllib.request.Request(args.base_url.rstrip('/') + path, headers=headers)
            with urllib.request.build_opener(NoRedirect).open(request, timeout=30) as response:
                body = response.read()
                status = response.status
            elapsed = (time.perf_counter() - start) * 1000
            payload = json.loads(body)
            if not isinstance(payload, dict) or payload.get('ok') is False:
                return {'error': 'unsuccessful JSON response', 'status': status}
            return {'ms': round(elapsed, 3), 'bytes': len(body), 'status': status}
        except Exception as exc:
            return {'error': type(exc).__name__, 'status': getattr(exc, 'code', None)}

    if args.rate:
        paths = list(scenarios.values())
        warmup = [measure(path) for path in paths]
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            jobs = []
            def scheduled(path, due):
                result = measure(path)
                result['scheduled_ms'] = round((time.perf_counter() - due) * 1000, 3)
                return result
            for i in range(int(args.rate * args.duration)):
                due = start + i / args.rate
                time.sleep(max(0, due - time.perf_counter()))
                jobs.append(pool.submit(scheduled, paths[i % len(paths)], due))
            samples = [job.result() for job in jobs]
        elapsed = time.perf_counter() - start
        times = sorted(s['scheduled_ms'] for s in samples)
        errors = sum('error' in s for s in samples)
        print(json.dumps({'target_rps': args.rate, 'achieved_rps': len(samples) / elapsed,
                          'requests': len(samples), 'errors': errors, 'warmup': warmup,
                          'p50_ms': statistics.median(times),
                          'p95_ms': times[math.ceil(len(times) * .95) - 1],
                          'p99_ms': times[math.ceil(len(times) * .99) - 1]}, indent=2))
        return int(errors > 0 or any('error' in s for s in warmup))
    report = {'runs': args.runs, 'concurrency': args.concurrency, 'scenarios': {}}
    failed = False
    for name, path in scenarios.items():
        first = measure(path)
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            samples = list(pool.map(measure, [path] * args.runs))
        elapsed = (time.perf_counter() - start) * 1000
        times = sorted(s['ms'] for s in samples if 'ms' in s)
        failed |= 'error' in first or len(times) != len(samples)
        report['scenarios'][name] = {
            'first_request': first, 'samples': samples, 'batch_ms': round(elapsed, 3),
            'p50_ms': statistics.median(times) if times else None,
            'p95_ms': times[math.ceil(len(times) * .95) - 1] if times else None}
    print(json.dumps(report, indent=2))
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
