"""Isolated storage microbenchmark; not an ingest or NPU throughput result."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from autofusex.fuzz.reporting.shared_content import intern_text

p = argparse.ArgumentParser()
p.add_argument('output', type=Path)
p.add_argument('--rows', type=int, default=10000)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
results = []
for unique in (False, True):
    for layout in ('copied', 'text_unique', 'digest_unique'):
        path = a.output / f'{layout}-{unique}.sqlite'
        with sqlite3.connect(path) as c:
            c.execute('CREATE TABLE refs(id INTEGER PRIMARY KEY, content_id INTEGER)')
            if layout == 'digest_unique':
                c.execute('CREATE TABLE display_text_v2(id INTEGER PRIMARY KEY, digest BLOB UNIQUE, content TEXT)')
            else:
                c.execute('CREATE TABLE display_text_v2(id INTEGER PRIMARY KEY, content TEXT'+(' UNIQUE' if layout == 'text_unique' else '')+')')
            start = time.perf_counter()
            for n in range(a.rows):
                # Realistic low-compressibility payload; unique/shared runs use same size.
                salt = n if unique else n % 10
                value = ''.join(hashlib.sha256(f'{salt}:{k}'.encode()).hexdigest() for k in range(64))
                if layout == 'digest_unique':
                    key = intern_text(c, 'display_text_v2', 'content', value)
                elif layout == 'text_unique':
                    c.execute('INSERT OR IGNORE INTO display_text_v2(content) VALUES (?)', (value,))
                    key = c.execute('SELECT id FROM display_text_v2 WHERE content=?', (value,)).fetchone()[0]
                else:
                    key = c.execute('INSERT INTO display_text_v2(content) VALUES (?)', (value,)).lastrowid
                c.execute('INSERT INTO refs VALUES (?,?)', (n, key))
            c.commit()
            elapsed = time.perf_counter() - start
            assert c.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
            results.append(dict(layout=layout, unique=unique, rows=a.rows, bytes=path.stat().st_size, seconds=elapsed))
(a.output / 'metrics.json').write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
