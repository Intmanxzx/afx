"""Public summary comes only from explicit CANN plog evidence."""
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
    if isinstance(detail, dict):
        summary = first_error(detail.get('plog_errors'))
        if summary:
            return summary
    # Older uploads stored an explicitly labelled plog section in the log.
    if isinstance(log, str) and '# CANN plog (ERROR)' in log:
        section = log.split('# CANN plog (ERROR)', 1)[1].split(chr(10)+chr(10), 1)[0]
        return first_error(section.splitlines())
    return None


def attach(conn, row):
    from autofusex.server.store import get_content
    ids = conn.execute('SELECT detail_content_id,log_content_id FROM executions WHERE exec_id=?', (row['exec_id'],)).fetchone()
    detail = None
    if ids[0] is not None:
        raw = get_content(conn, ids[0])
        if raw:
            try: detail = json.loads(raw)
            except (ValueError, UnicodeError): pass
    summary = extract(detail)
    if summary is None and ids[1] is not None:
        raw = get_content(conn, ids[1])
        if raw: summary = extract(log=raw.decode(errors='replace'))
    row['log_summary'] = summary
