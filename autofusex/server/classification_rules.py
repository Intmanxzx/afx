"""High-confidence, administrator-configured log classification rules."""
import json
import re
from .store import get_content
from .failure_attribution import VERSION

KEY = 'classification.rules'
FIELDS = ('function_status', 'fail_stage', 'failure_owner', 'classifier_version')
DEFAULT_RULES = []

def validate(rules):
    if not isinstance(rules, list) or len(rules) > 200:
        raise ValueError('rules must be a list with at most 200 entries')
    out = []
    for rule in rules:
        if not isinstance(rule, dict): raise ValueError('each rule must be an object')
        rid = str(rule.get('id', '')).strip()
        pattern = str(rule.get('pattern', ''))
        if not re.fullmatch(r'[A-Za-z0-9_.:-]{1,64}', rid): raise ValueError('invalid rule id')
        if not pattern or len(pattern) > 4000: raise ValueError(f'invalid pattern for {rid}')
        try: re.compile(pattern, re.I if rule.get('ignore_case', True) else 0)
        except re.error as exc: raise ValueError(f'invalid regex for {rid}: {exc}') from None
        result = rule.get('result') or {}
        if result.get('function_status') not in {'gen_fail','ascgraph_fail','compile_fail','runtime_fail','build_fail','timeout'}: raise ValueError(f'invalid status for {rid}')
        if result.get('fail_stage') is None or result.get('failure_owner') is None: raise ValueError(f'incomplete result for {rid}')
        out.append({'id': rid, 'enabled': bool(rule.get('enabled', True)), 'pattern': pattern,
                    'ignore_case': bool(rule.get('ignore_case', True)), 'backend': rule.get('backend'),
                    'result': {'function_status': result['function_status'], 'fail_stage': str(result['fail_stage']), 'failure_owner': str(result['failure_owner'])}})
    if len({r['id'] for r in out}) != len(out): raise ValueError('duplicate rule id')
    return out

def load(store):
    row = store.reader().execute('SELECT value FROM platform_settings WHERE key=?', (KEY,)).fetchone()
    if not row: return DEFAULT_RULES.copy()
    try: return validate(json.loads(row['value']))
    except (ValueError, TypeError, json.JSONDecodeError): return DEFAULT_RULES.copy()

def match(rules, backend, text):
    for rule in rules:
        if not rule['enabled'] or (rule.get('backend') and rule['backend'] != backend): continue
        if re.search(rule['pattern'], text or '', re.I if rule.get('ignore_case', True) else 0):
            r = rule['result']; return r['function_status'], r['fail_stage'], r['failure_owner'], VERSION, rule['id']
    return None

def plan(store, rules):
    rules = validate(rules); changes = []
    with store.connect(readonly=True) as conn:
        for row in conn.execute('SELECT * FROM executions WHERE classifier_version IS NULL OR classifier_version<>?', (VERSION,)):
            text = '\n'.join(str(x) for x in (row['log_summary'], get_content(conn, row['log_content_id']) if row['log_content_id'] else None, get_content(conn, row['detail_content_id']) if row['detail_content_id'] else None) if x)
            result = match(rules, row['backend'], text)
            if not result: continue
            after = list(result[:4]); before = [row[k] for k in FIELDS]
            if before != after: changes.append({'exec_id': row['exec_id'], 'exec_uuid': row['exec_uuid'], 'before': before, 'after': after, 'rule_id': result[4]})
    return changes
