"""Compatibility entry point for the public error summary."""
import json
import re

_ERROR = re.compile(r'(?<![A-Za-z])ERROR(?![A-Za-z])')


def first_error(lines):
    if not isinstance(lines, list):
        return None
    for line in lines:
        if isinstance(line, str) and _ERROR.search(line):
            return line.strip()[:500]
    return None


def extract(detail=None, log=None):
    from autofusex.server.error_logs import summary
    return summary(detail, log)


def attach(conn, row):
    from autofusex.server.store import get_content
    ids = conn.execute('SELECT detail_content_id,log_content_id,log_summary FROM executions WHERE exec_id=?', (row['exec_id'],)).fetchone()
    detail = None
    if ids[0] is not None:
        raw = get_content(conn, ids[0])
        if raw:
            try: detail = json.loads(raw)
            except (ValueError, UnicodeError): pass
    log = None
    if ids[1] is not None:
        raw = get_content(conn, ids[1])
        if raw: log = raw.decode(errors='replace')
    from autofusex.server.error_logs import summary
    row['log_summary'] = summary(detail, log, ids[2])
