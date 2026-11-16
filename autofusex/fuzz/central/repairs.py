"""Fenced repair ownership, atomic retest submission, and evidence-gated summaries."""
import hashlib
import json
import time
import uuid

from . import issue_groups, retests
from ..management.central import Conflict
from ..management.config import canonical

TTL = 1800


def initialize(db):
    statements = '''
    CREATE TABLE IF NOT EXISTS repairs_v1(
      id TEXT PRIMARY KEY,issue_key TEXT NOT NULL,snapshot_hash TEXT NOT NULL,snapshot TEXT NOT NULL,
      user_id TEXT NOT NULL,agent_id TEXT NOT NULL,lease_id TEXT NOT NULL,expires REAL NOT NULL,
      state TEXT NOT NULL,latest_retest TEXT,created REAL NOT NULL,updated REAL NOT NULL);
    CREATE UNIQUE INDEX IF NOT EXISTS repairs_active_issue ON repairs_v1(issue_key) WHERE state!='resolved';
    CREATE TABLE IF NOT EXISTS repair_requests_v1(
      user_id TEXT NOT NULL,request_key TEXT NOT NULL,hash TEXT NOT NULL,response TEXT NOT NULL,
      PRIMARY KEY(user_id,request_key));
    CREATE TABLE IF NOT EXISTS repair_retests_v1(
      id TEXT PRIMARY KEY,repair_id TEXT NOT NULL,batches TEXT NOT NULL,artifact TEXT NOT NULL,created REAL NOT NULL);
    CREATE TABLE IF NOT EXISTS repair_summaries_v1(
      id TEXT PRIMARY KEY,repair_id TEXT UNIQUE NOT NULL,retest_id TEXT NOT NULL,
      background TEXT NOT NULL,solution TEXT NOT NULL,notes TEXT NOT NULL,created REAL NOT NULL);
    CREATE TABLE IF NOT EXISTS repair_events_v1(
      seq INTEGER PRIMARY KEY AUTOINCREMENT,repair_id TEXT NOT NULL,event TEXT NOT NULL,
      user_id TEXT NOT NULL,agent_id TEXT NOT NULL,time REAL NOT NULL);
    '''
    for statement in statements.split(';'):
        if statement.strip(): db.execute(statement)


def text(value, maximum=128):
    if not isinstance(value, str) or not value.strip() or len(value)>maximum:
        raise ValueError('invalid text field')
    return value


def event(db, row, name):
    db.execute('INSERT INTO repair_events_v1(repair_id,event,user_id,agent_id,time) VALUES (?,?,?,?,?)',
               (row['id'],name,row['user_id'],row['agent_id'],time.time()))


def owned(db,user,identity,lease=None):
    row=db.execute('SELECT * FROM repairs_v1 WHERE id=? AND user_id=?',(identity,user)).fetchone()
    if row is None: raise PermissionError('repair not owned')
    if lease is not None and (row['lease_id']!=lease or row['expires']<=time.time()):
        raise Conflict('repair lease expired or superseded; reclaim before continuing')
    return row


def locked_for_agent(db, issue_key):
    if not db.execute("SELECT 1 FROM sqlite_master WHERE name='repairs_v1'").fetchone(): return False
    return bool(db.execute("SELECT 1 FROM repairs_v1 WHERE issue_key=? AND state!='resolved'",(issue_key,)).fetchone())


def verification(db, identity):
    record=db.execute('SELECT * FROM repair_retests_v1 WHERE id=?',(identity,)).fetchone()
    if not record: raise LookupError('retest not found')
    batches=json.loads(record['batches']); states=[]; attempts=[]; passed=True
    for batch in batches:
        for task in db.execute('SELECT * FROM retest_tasks_v1 WHERE batch_id=?',(batch,)):
            states.append(task['state'])
            if task['state']!='completed' or not task['attempt_id']:
                passed=False;continue
            attempt=db.execute('SELECT status FROM attempts WHERE id=?',(task['attempt_id'],)).fetchone()
            accepted=db.execute('SELECT accepted FROM retest_completions_v1 WHERE lease_id=? AND attempt_id=?',
                                (task['lease_id'],task['attempt_id'])).fetchone()
            shapes=db.execute('SELECT shape_index,correctness,performance FROM shapes WHERE attempt_id=?',
                              (task['attempt_id'],)).fetchall()
            required={r[0] for r in db.execute('SELECT shape_index FROM retest_targets_v1 WHERE task_id=? AND shape_index>=0',(task['id'],))}
            spec=json.loads(task['spec'])
            good=bool(attempt and attempt[0]=='ok' and accepted and accepted[0] and shapes and
                      required <= {s['shape_index'] for s in shapes} and
                      all(s['correctness']=='pass' for s in shapes))
            if spec['evaluation']['performance']:
                good=good and all(s['performance']=='measured' for s in shapes)
            passed=passed and good
            attempts.append(task['attempt_id'])
    terminal=bool(states) and all(s in {'completed','cancelled','interrupted'} for s in states)
    return dict(id=identity,batch_ids=batches,terminal=terminal,passed=terminal and passed,
                states={s:states.count(s) for s in set(states)},attempt_ids=attempts,
                artifact=json.loads(record['artifact']))


def detail(db,user,identity,*,persist=True):
    row=owned(db,user,identity)
    result={k:row[k] for k in ('id','issue_key','agent_id','state','expires','created','updated')}
    result['snapshot']=json.loads(row['snapshot'])
    result['retest']=verification(db,row['latest_retest']) if row['latest_retest'] else None
    summary=db.execute('SELECT * FROM repair_summaries_v1 WHERE repair_id=?',(identity,)).fetchone()
    result['summary']=dict(summary) if summary else None
    result['events']=[dict(e) for e in db.execute('SELECT seq,event,agent_id,time FROM repair_events_v1 WHERE repair_id=? ORDER BY seq DESC LIMIT 50',(identity,))]
    if result['state'] != 'resolved' and result['retest'] and result['retest']['terminal']:
        result['state'] = 'verified' if result['retest']['passed'] else 'fixing'
        if persist and row['state'] != result['state']:
            db.execute('UPDATE repairs_v1 SET state=?,updated=? WHERE id=?',(result['state'],time.time(),identity))
            event(db,row,result['state'])
    return result


def mutate(path,user,action,body,identity=None,allowlist=None):
    if not isinstance(body,dict): raise ValueError('object required')
    fields={'claim':({'agent_id','request_key'},{'issue_key'}),
            'renew':({'lease_id','request_key'},set()),
            'release':({'lease_id','request_key'},set()),
            'retest':({'lease_id','request_key','client_id','artifact'},set()),
            'summary':({'lease_id','request_key','background','solution'},{'notes'})}
    if action not in fields: raise LookupError('action not found')
    required,optional=fields[action]
    if not required<=set(body) or set(body)-required-optional: raise ValueError('invalid fields')
    if action=='claim' and 'issue_key' in body:
        import re
        if not isinstance(body['issue_key'],str) or not re.fullmatch('[0-9a-f]{24}',body['issue_key']):
            raise ValueError('invalid issue key')
    key=text(body['request_key']);hashed=hashlib.sha256(canonical(dict(action=action,id=identity,body=body)).encode()).hexdigest()
    # A committed retry needs no current-issue aggregation or writer lock.
    # Keep the transactional check below for concurrently arriving first calls.
    from contextlib import closing
    from ..storage.repository import _connect
    with closing(_connect(path)) as read:
        if read.execute("SELECT 1 FROM sqlite_master WHERE name='repair_requests_v1'").fetchone():
            prior=read.execute('SELECT hash,response FROM repair_requests_v1 WHERE user_id=? AND request_key=?',(user,key)).fetchone()
            if prior:
                if prior[0]!=hashed: raise Conflict('request key conflict')
                return json.loads(prior[1])
    candidates=issue_groups._current(path,include_targets=True) if action=='claim' else None
    with retests.transaction(path) as db:
        initialize(db)
        prior=db.execute('SELECT * FROM repair_requests_v1 WHERE user_id=? AND request_key=?',(user,key)).fetchone()
        if prior:
            if prior['hash']!=hashed: raise Conflict('request key conflict')
            return json.loads(prior['response'])
        if allowlist: allowlist.browser(user)
        if action=='claim':
            agent=text(body['agent_id'])
            # Repeated polling by one agent resumes its single active repair.
            row=db.execute("SELECT * FROM repairs_v1 WHERE user_id=? AND agent_id=? AND state!='resolved' AND expires>?",
                           (user,agent,time.time())).fetchone()
            if row and body.get('issue_key') and row['issue_key']!=body['issue_key']:
                raise Conflict('agent already holds another repair')
            if row is None:
                known={g['issue_key'] for g in candidates}
                for old in db.execute("SELECT * FROM repairs_v1 WHERE state!='resolved' AND expires<=? ORDER BY created",(time.time(),)):
                    if old['issue_key'] not in known:
                        snapshot=json.loads(old['snapshot'])
                        candidates.append(dict(issue_key=old['issue_key'],dimension=snapshot['dimension'],
                                               category=snapshot.get('category','Other'),title=snapshot['title'],
                                               _targets=snapshot['targets'],examples=snapshot['examples']))
                for group in candidates:
                    if body.get('issue_key') and group['issue_key']!=body['issue_key']: continue
                    current=db.execute("SELECT * FROM repairs_v1 WHERE issue_key=? AND state!='resolved'",(group['issue_key'],)).fetchone()
                    if current and current['expires']>time.time(): continue
                    claim=db.execute('SELECT updated FROM issue_claims_v1 WHERE issue_key=?',(group['issue_key'],)).fetchone()
                    if not current and claim and time.time()-claim[0]<=issue_groups.CLAIM_TTL: continue
                    snapshot=dict(issue_key=group['issue_key'],dimension=group['dimension'],category=group['category'],title=group['title'],targets=group['_targets'],examples=group['examples'])
                    snapshot['targets']=sorted(snapshot['targets'],key=lambda target:(target['attempt_id'],target['shape_index']))
                    digest=hashlib.sha256(canonical(snapshot['targets']).encode()).hexdigest()
                    if not current and db.execute('SELECT 1 FROM repairs_v1 WHERE issue_key=? AND snapshot_hash=? AND state=?',(group['issue_key'],digest,'resolved')).fetchone(): continue
                    now=time.time();lease=str(uuid.uuid4())
                    if current:
                        identity=current['id']
                        db.execute('UPDATE repairs_v1 SET user_id=?,agent_id=?,lease_id=?,expires=?,updated=? WHERE id=?',(user,agent,lease,now+TTL,now,identity))
                    else:
                        identity=str(uuid.uuid4())
                        db.execute('INSERT INTO repairs_v1 VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',(identity,group['issue_key'],digest,canonical(snapshot),user,agent,lease,now+TTL,'claimed',None,now,now))
                    name=db.execute('SELECT display_name FROM users WHERE id=?',(user,)).fetchone()[0]
                    db.execute('INSERT OR REPLACE INTO issue_claims_v1 VALUES (?,?,?,?,?,?,?,?,?)',(group['issue_key'],group['dimension'],group['category'],user,name,'claimed',now,now,None))
                    row=owned(db,user,identity);event(db,row,'claimed');break
            response=None if row is None else dict(detail(db,user,row['id']),lease_id=row['lease_id'])
        else:
            row=owned(db,user,identity,text(body['lease_id']))
            if row['state']=='resolved': raise Conflict('repair already resolved')
            now=time.time()
            if action=='renew':
                db.execute('UPDATE repairs_v1 SET expires=?,updated=? WHERE id=?',(now+TTL,now,identity))
                db.execute('UPDATE issue_claims_v1 SET updated=? WHERE issue_key=?',(now,row['issue_key']))
            elif action=='release':
                db.execute('UPDATE repairs_v1 SET expires=0,updated=? WHERE id=?',(now,identity))
                event(db,row,'released')
            elif action=='retest':
                if row['latest_retest'] and not verification(db,row['latest_retest'])['terminal']:
                    raise Conflict('a retest is still active; wait before submitting another')
                if allowlist: allowlist.client(user,body['client_id'])
                artifact=body['artifact']
                if not isinstance(artifact,dict) or set(artifact)!={'sha256','version','installed_at'}: raise ValueError('artifact evidence required')
                import re
                if not isinstance(artifact['sha256'],str) or not re.fullmatch('[0-9a-f]{64}',artifact['sha256']):raise ValueError('invalid artifact hash')
                text(artifact['version']);text(artifact['installed_at'])
                from datetime import datetime
                if datetime.fromisoformat(artifact['installed_at']).tzinfo is None: raise ValueError('installed_at requires timezone')
                targets=json.loads(row['snapshot'])['targets'];base=uuid.uuid4();batches=[]
                for offset in range(0,len(targets),200):
                    result=retests.create(path,user,dict(client_id=body['client_id'],request_key=str(uuid.uuid5(base,str(offset))),targets=targets[offset:offset+200],repetitions=1),connection=db)
                    batches.append(result['batch_id'])
                retest_id=str(base)
                db.execute('INSERT INTO repair_retests_v1 VALUES (?,?,?,?,?)',(retest_id,identity,canonical(batches),canonical(artifact),now))
                db.execute("UPDATE repairs_v1 SET state='retesting',latest_retest=?,updated=? WHERE id=?",(retest_id,now,identity))
                db.execute("UPDATE issue_claims_v1 SET status='retesting',batch_id=?,updated=? WHERE issue_key=?",(canonical(batches),now,row['issue_key']))
                event(db,row,'retest_submitted')
            elif action=='summary':
                if not row['latest_retest'] or not verification(db,row['latest_retest'])['passed']:
                    raise Conflict('latest retest must pass before summary submission')
                background=text(body['background'],12000);solution=text(body['solution'],24000)
                notes=body.get('notes','')
                if not isinstance(notes,str) or len(notes)>12000:raise ValueError('invalid notes')
                summary_id=str(uuid.uuid4())
                db.execute('INSERT INTO repair_summaries_v1 VALUES (?,?,?,?,?,?,?)',(summary_id,identity,row['latest_retest'],background,solution,notes,now))
                for batch in verification(db,row['latest_retest'])['batch_ids']:
                    db.execute('UPDATE retest_result_annotations_v1 SET summary_id=? WHERE task_id IN (SELECT id FROM retest_tasks_v1 WHERE batch_id=?)',(summary_id,batch))
                db.execute("UPDATE repairs_v1 SET state='resolved',updated=? WHERE id=?",(now,identity))
                db.execute('DELETE FROM issue_claims_v1 WHERE issue_key=?',(row['issue_key'],))
                event(db,row,'resolved')
            response=detail(db,user,identity)
        db.execute('INSERT INTO repair_requests_v1 VALUES (?,?,?,?)',(user,key,hashed,canonical(response)))
        return response


def get(path,user,identity):
    from contextlib import closing
    import sqlite3
    from ..storage.repository import _connect
    with closing(_connect(path)) as db:
        db.row_factory=sqlite3.Row
        db.execute('BEGIN')
        return detail(db,user,identity,persist=False)
