import sqlite3
import pytest
from autofusex.fuzz.reporting.cases import compatible_ctes, rollup, reuse_latest


@pytest.mark.parametrize('version', [(3, 31, 1), (3, 35, 0), (3, 45, 1)])
def test_cte_version_selection_preserves_query_results(monkeypatch, version):
    monkeypatch.setattr(sqlite3, 'sqlite_version_info', version)
    sql = compatible_ctes('WITH value AS MATERIALIZED (SELECT 7 AS n) SELECT n FROM value')
    assert ('MATERIALIZED' in sql) == (version >= (3, 35, 0))
    if version < (3, 35, 0) or tuple(map(int, sqlite3.sqlite_version.split('.'))) >= (3, 35, 0):
        with sqlite3.connect(':memory:') as conn:
            assert conn.execute(sql).fetchone() == (7,)
    query, _ = rollup()
    assert 'SELECT * FROM selected_latest' in reuse_latest(query)


@pytest.mark.parametrize('dynamic', [False, True])
def test_current_short_seed_uses_exact_search(dynamic):
    from autofusex.fuzz.generation.sampling import SamplingConfig
    from autofusex.fuzz.reporting.search import exact_seed, condition
    seed = SamplingConfig().seed(42, dynamic=dynamic)
    assert exact_seed(seed.short) == str(seed)
    assert condition(seed.short) == ('p.seed=?', [str(seed)])
