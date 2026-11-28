"""Non-destructive, explicit upgrade of feature-partitioned accuracy issues."""
from collections import defaultdict

from autofusex.server import issues
from autofusex.server.store import now_ms


def migrate(store, *, apply=False):
    """Keep IDs and repair evidence; leave concurrent repair leases intact.

    Run with server writes stopped. Preview rolls back the entire transaction.
    Independent repairs and terminal evidence remain available under their IDs.
    """
    with store.write() as conn:
        conn.execute('SAVEPOINT accuracy_signatures')
        groups = defaultdict(list)
        for row in conn.execute("SELECT * FROM issues WHERE dimension='accuracy' AND cluster_generation=2"):
            parts = [part.strip() for part in row['category'].split('·')]
            signature = (issues._digest('accuracy-v3', *parts[:5]) if len(parts) >= 5
                         else issues._digest('accuracy', row['primary_category']))
            groups[signature].append(row)
        report = dict(applied=apply, updated=0, merged=[], deferred=[])
        # Keep an existing canonical signature stable; on the first upgrade prefer
        # the most advanced repair, then the oldest issue.
        ranks = {'summarizing': 0, 'regressing': 1, 'verifying': 2, 'claimed': 3,
                 'open': 4, 'resolved': 5, 'closed': 6}
        migrated = []
        for signature, rows in groups.items():
            if any(r['status'] == 'verifying' or r['frozen_at'] is not None or r['similar_to'] for r in rows):
                report['deferred'].append(dict(issues=[r['issue_id'] for r in rows],
                                               reason='frozen verification group'))
                continue
            candidates = [r for r in rows if not r['duplicate_of']]
            if not candidates:
                continue
            target = min(candidates, key=lambda r: (r['signature'] != signature,
                         ranks[r['status']], r['issue_no']))
            target_id = target['issue_id']
            # Preserve an intentional administrative merge out of this group.
            aliases = {r[0] for r in conn.execute(
                "SELECT issue_id FROM issue_signature_alias WHERE dimension='accuracy' AND backend='' "
                "AND signature IN (%s)" % ','.join('?' for _ in rows),
                [r['signature'] for r in rows])}
            external = aliases - {r['issue_id'] for r in rows}
            if external:
                report['deferred'].append(dict(issues=[r['issue_id'] for r in rows],
                                               reason='existing external merge'))
                continue
            migrated.append((signature, rows))
            for member in rows:
                parts = [part.strip() for part in member['category'].split('·')]
                category = ' · '.join(parts[:5]) if len(parts) >= 5 else member['category']
                conn.execute("UPDATE issues SET category=?,title=? WHERE issue_id=?",
                             (category, '精度失败 · ' + category, member['issue_id']))
            conn.execute("INSERT OR REPLACE INTO issue_signature_alias VALUES('accuracy','',?,?)",
                         (signature, target_id))
            for row in rows:
                if row['issue_id'] == target_id:
                    continue
                if row['duplicate_of']:
                    continue
                if row['status'] in issues.ACTIVE or row['status'] in issues.TERMINAL:
                    report['deferred'].append(dict(issue_id=row['issue_id'], into=target_id,
                                                   reason='existing repair or terminal evidence'))
                    continue
                # Copy every evidence column, including pass/skip and round numbers.
                columns = [r[1] for r in conn.execute('PRAGMA table_info(issue_cases)')]
                others = [c for c in columns if c != 'issue_id']
                names = ','.join(others)
                conn.execute(f"INSERT OR IGNORE INTO issue_cases(issue_id,{names}) "
                             f"SELECT ?,{names} FROM issue_cases WHERE issue_id=? AND role='affected'",
                             (target_id, row['issue_id']))
                conn.execute("UPDATE issue_current SET issue_id=? WHERE issue_id=?",
                             (target_id, row['issue_id']))
                conn.execute("INSERT OR REPLACE INTO issue_signature_alias VALUES('accuracy','',?,?)",
                             (row['signature'], target_id))
                conn.execute("UPDATE issue_signature_alias SET issue_id=? WHERE issue_id=?",
                             (target_id, row['issue_id']))
                issues._cancel_tasks(conn, row['issue_id'])
                issues._set_status(conn, row, 'closed', 'system', 'merged', target_id,
                                   duplicate_of=target_id, close_reason=f'merged into {target_id}',
                                   owner_user=None, agent_id=None, lease_id=None, lease_expires_at=None)
                issues.event(conn, target_id, 'system', 'merged_from', None, None,
                             f"{row['issue_id']} (accuracy signature no longer includes features)")
                report['merged'].append(dict(issue_id=row['issue_id'], into=target_id))
            if target['signature'] != signature:
                conn.execute("INSERT OR REPLACE INTO issue_signature_alias VALUES('accuracy','',?,?)",
                             (target['signature'], target_id))
                conn.execute("UPDATE issues SET signature=? WHERE issue_id=?", (signature, target_id))
                report['updated'] += 1
            conn.execute("UPDATE issues SET case_count=(SELECT COUNT(DISTINCT case_id) FROM issue_cases "
                         "WHERE issue_id=? AND role='affected') WHERE issue_id=?", (target_id, target_id))
        # Signatures are derived metadata. Historical execution/log contents remain intact.
        for signature, rows in migrated:
            for row in rows:
                if row['signature'] != signature:
                    conn.execute('UPDATE executions SET issue_signature=? WHERE issue_signature=?',
                                 (signature, row['signature']))
        if apply:
            conn.execute('INSERT INTO audit_log(at,actor,action,detail) VALUES(?,?,?,?)',
                         (now_ms(), 'offline', 'issues.accuracy_signatures',
                          str({k: len(v) if isinstance(v, list) else v for k, v in report.items()})))
        else:
            conn.execute('ROLLBACK TO accuracy_signatures')
        conn.execute('RELEASE accuracy_signatures')
        return report
