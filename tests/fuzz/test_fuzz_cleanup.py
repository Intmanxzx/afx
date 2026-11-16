import uuid
import pytest
from autofusex.fuzz.storage import ResultRepository
from autofusex.fuzz.sync.outbox import pending
from autofusex.fuzz.central.repository import create_user, ingest
from autofusex.fuzz.storage.queries import ResultQueries




def test_full_reset_retains_auth_and_tombstones(tmp_path):
    from autofusex.fuzz.central.cleanup import purge_all_results
    central=tmp_path/'central'; key=tmp_path/'key'; create_user(central,'Tester',key)
    local=ResultRepository(tmp_path/'local')
    local.save({'seed':'14615016373650978276328604632813839116929039597610','workload_id':'a'*64,'oracle_id':'ref/1','backend':'cpu','status':'ok','shape_results':[]},environment={'cann':'9.3'})
    batch=pending(local.path);secret=key.read_text().strip();ingest(central,secret,batch)
    with pytest.raises(ValueError):purge_all_results(central,expected_count=2)
    assert purge_all_results(central,expected_count=1)=={'deleted':1,'remaining':0}
    batch['source_id']=str(uuid.uuid4());ingest(central,secret,batch)
    assert ResultQueries(central).summary()['total']==0


def test_exact_id_cleanup_does_not_delete_peers(tmp_path):
    from autofusex.fuzz.central.cleanup import purge_attempts
    central=tmp_path/'central';key=tmp_path/'key';create_user(central,'Tester',key)
    local=ResultRepository(tmp_path/'local')
    record={'seed':'14615016373650978276328604632813839116929039597610','workload_id':'a'*64,'oracle_id':'ref/1','backend':'cpu','status':'ok','shape_results':[]}
    first=local.save(record);second=local.save(record)
    batch=pending(local.path);secret=key.read_text().strip();ingest(central,secret,batch)
    with pytest.raises(ValueError):purge_attempts(central,[first,str(uuid.uuid4())])
    assert purge_attempts(central,[first])=={'deleted':1,'remaining':1}
    assert ResultQueries(central).detail(second)['id']==second
    batch['source_id']=str(uuid.uuid4());ingest(central,secret,batch)
    assert ResultQueries(central).summary()['total']==1
