import hashlib
import json
import zlib

import pytest

from test_platform import env, item, body, upload
from autofusex.server.store import canonical_json, get_content, put_content, put_raw_upload


def test_upload_preserves_original_and_detail(env):
    store, ingestor, principal = env
    original = item('refs-upload-001', detail={'text': '中文' * 1000}, log=('error' + chr(10)) * 1000,
                    extra_unknown={'preserve': True})
    original['case']['spec'] = {'original': [1, None, False]}
    assert upload(ingestor, principal, body([original]))['accepted'] == 1
    row = store.reader().execute('SELECT * FROM executions').fetchone()
    assert get_content(store.reader(), row['raw_upload_content_id']) == canonical_json(original)
    assert get_content(store.reader(), row['detail_content_id']) == canonical_json(original['detail'])
    stored = store.reader().execute('SELECT * FROM content.contents WHERE content_id=?',
                                    (row['raw_upload_content_id'],)).fetchone()
    assert stored['codec'] == 'upload-refs-v1'
    manifest = json.loads(zlib.decompress(stored['data']))
    assert dict((tuple(p), i) for p, i in manifest['refs'])[('detail',)] == row['detail_content_id']
    # Updating the current case must not mutate the historical original.
    with store.write() as conn:
        conn.execute('UPDATE cases SET spec_content_id=NULL')
    assert get_content(store.reader(), row['raw_upload_content_id']) == canonical_json(original)
    assert hashlib.sha256(canonical_json(original)).digest() == row['content_hash']


@pytest.mark.parametrize('payload', [{}, {'log': '', 'detail': {}, 'case': {'spec': {}}},
                                     {'log': None, 'detail': None, 'case': {'spec': None}},
                                     {'detail': {'refs': [], 'payload': 'literal'}}])
def test_roundtrip_and_dedup(env, payload):
    store, _, _ = env
    raw = canonical_json(payload)
    with store.write() as conn:
        ref = put_raw_upload(conn, raw)
        assert get_content(conn, ref) == raw
        assert put_raw_upload(conn, raw) == ref
        assert put_content(conn, raw) == ref


def test_legacy_and_missing_reference(env):
    store, _, _ = env
    with store.write() as conn:
        legacy = canonical_json({'detail': {'old': True}})
        ref = put_content(conn, legacy)
        assert put_raw_upload(conn, legacy) == ref
        raw = canonical_json({'detail': {'new': True}})
        ref = put_raw_upload(conn, raw)
        manifest = json.loads(zlib.decompress(conn.execute(
            'SELECT data FROM content.contents WHERE content_id=?', (ref,)).fetchone()[0]))
        conn.execute('DELETE FROM content.contents WHERE content_id=?', (manifest['refs'][0][1],))
        with pytest.raises(ValueError, match='missing or invalid'):
            get_content(conn, ref)


def test_corrupt_reference_fails_hash_check(env):
    store, _, _ = env
    with store.write() as conn:
        ref = put_raw_upload(conn, canonical_json({'log': 'original'}))
        row = conn.execute('SELECT data FROM content.contents WHERE content_id=?', (ref,)).fetchone()
        manifest = json.loads(zlib.decompress(row[0]))
        manifest['refs'][0][1] = put_content(conn, b'different')
        conn.execute('UPDATE content.contents SET data=? WHERE content_id=?',
                     (zlib.compress(canonical_json(manifest)), ref))
        with pytest.raises(ValueError, match='integrity'):
            get_content(conn, ref)


def test_chip_count_does_not_read_execution_rows(env):
    from autofusex.server import search
    store, ingestor, principal = env
    upload(ingestor, principal, body([item('chip-count-001'), item('chip-count-002', backend='af-legacy')]))
    conn = store.reader()
    import sqlite3
    def deny_execution_reads(action, table, column, database, trigger):
        return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_READ and table == 'executions' else sqlite3.SQLITE_OK
    conn.set_authorizer(deny_execution_reads)
    try:
        assert search.filtered_case_count(conn, search.visual_query(chips='a2a3')) == 1
        assert search.filtered_case_count(conn, search.visual_query(chips='a5')) == 0
    finally:
        conn.set_authorizer(None)
