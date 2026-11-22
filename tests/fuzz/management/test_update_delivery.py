import importlib.metadata
from autofusex.fuzz.daemon.management.store import Store
from autofusex.fuzz.daemon.management.update_delivery import receive, progress


def test_current_version_skips_install_and_acknowledges(tmp_path):
    s=Store(tmp_path)
    assert receive(s,importlib.metadata.version('autofusex'),'one')=='up_to_date'
    assert not s.pending()
    assert progress(s)['state']=='succeeded'


def test_duplicate_commands_during_upgrade_are_ignored(tmp_path):
    s=Store(tmp_path)
    receive(s,'9.0.0','first')
    for state in ['queued','running','activating']:
        with s.connect() as db:
            row=db.execute('SELECT id FROM operations').fetchone()
        s.state(row['id'],state)
        assert receive(s,'9.0.0','another-'+state)=='upgrading'
        assert receive(s,'10.0.0','different-'+state)=='upgrading'
        assert s.get('platform_update_request')=='first'
        assert progress(s)['state']==state
    with s.connect() as db:assert db.execute('SELECT count(*) FROM operations').fetchone()[0]==1
