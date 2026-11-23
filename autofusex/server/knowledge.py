"""Knowledge base for repair and fusion agents (design 4.6).

Entries are structured so agents can filter by ops/dtypes/backends before
reading text. Confidence: verified (issue passed verify + regression),
measured (capability profile statistics), proposed (needs admin review).
Search: structured filters + FTS5 trigram for terms of 3+ chars, LIKE for
shorter (e.g. two-character Chinese words). The table stays small, so the
LIKE fallback is cheap.
"""
from __future__ import annotations

import json
import re

from autofusex.server.auth import Principal
from autofusex.server.issues import IssueError, event, idempotent
from autofusex.server.store import Store, now_ms

KINDS = ("bugfix_function", "bugfix_accuracy", "perf_tuning", "fusion_pattern", "capability", "best_practice",
         "pitfall")
STATUSES = ("draft", "published", "deprecated")
_LABEL = re.compile(r"^[^\s,]{1,64}$")
TEXT_LIMITS = {"title": 200, "graph_pattern": 1000, "symptom": 8000, "root_cause": 8000, "solution": 8000,
               "applicability": 4000, "version_range": 128, "fix_ref": 512}
REQUIRED = ("kind", "title", "symptom", "root_cause", "solution")
MAX_BODY = 64 * 1024


def validate(payload) -> dict:
    if not isinstance(payload, dict):
        raise IssueError("invalid", "knowledge must be an object")
    out = {}
    for name in REQUIRED:
        if payload.get(name) in (None, ""):
            raise IssueError("invalid", f"knowledge.{name} is required")
    if payload["kind"] not in KINDS:
        raise IssueError("invalid", f"knowledge.kind must be one of {', '.join(KINDS)}")
    out["kind"] = payload["kind"]
    for name, limit in TEXT_LIMITS.items():
        value = payload.get(name)
        if value is not None and (not isinstance(value, str) or len(value) > limit):
            raise IssueError("invalid", f"knowledge.{name} must be a string of at most {limit} chars")
        out[name] = value
    for name, limit in (("keywords", 20), ("ops", 64), ("dtypes", 16), ("backends", 8)):
        value = payload.get(name) or []
        if not isinstance(value, list) or len(value) > limit \
                or not all(isinstance(v, str) and _LABEL.match(v) for v in value):
            raise IssueError("invalid", f"knowledge.{name} must be a list of at most {limit} labels without "
                                        "spaces or commas")
        out[name] = sorted(set(value))
    body = payload.get("body_md")
    if body is not None and (not isinstance(body, str) or len(body.encode()) > MAX_BODY):
        raise IssueError("invalid", "knowledge.body_md must be a string of at most 64 KiB")
    out["body_md"] = body
    return out


def _wrap(values) -> str | None:
    return "," + ",".join(values) + "," if values else None


def _unwrap(value) -> list:
    return [v for v in (value or "").split(",") if v]


def _compose(fields) -> str:
    parts = [f"# {fields['title']}"]
    for label, name in (("现象", "symptom"), ("根因", "root_cause"), ("解决方案", "solution"),
                        ("适用范围", "applicability"), ("子图模式", "graph_pattern")):
        if fields.get(name):
            parts.append(f"## {label}\n\n{fields[name]}")
    return "\n\n".join(parts)


def create(conn, principal: Principal, fields: dict, *, confidence: str, status: str, agent_id=None,
           issue=None) -> str:
    now = now_ms()
    evidence = None
    if issue is not None:
        rounds = [dict(r) for r in conn.execute(
            "SELECT round_no, kind, fix_version, outcome, pass_count, fail_count, invalid_count FROM issue_rounds "
            "WHERE issue_id=? ORDER BY round_no", (issue["issue_id"],))]
        samples = [r[0] for r in conn.execute(
            "SELECT c.case_key FROM issue_cases ic JOIN cases c ON c.case_id=ic.case_id WHERE ic.issue_id=? "
            "AND ic.role='affected' ORDER BY ic.added_at LIMIT 5", (issue["issue_id"],))]
        evidence = {"issue_id": issue["issue_id"], "dimension": issue["dimension"], "backend": issue["backend"],
                    "case_count": issue["case_count"], "sample_cases": samples, "rounds": rounds,
                    "fix_version": issue["fix_version"], "fix_ref": fields.get("fix_ref") or issue["fix_ref"]}
        if not fields["backends"]:
            fields["backends"] = [issue["backend"]]
    body = fields["body_md"] or _compose(fields)
    cur = conn.execute(
        "INSERT INTO knowledge(knowledge_id, kind, title, keywords, ops, dtypes, backends, graph_pattern, symptom, "
        "root_cause, solution, applicability, evidence, version_range, confidence, status, author_user, agent_id, "
        "body_md, created_at, updated_at) VALUES('',?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (fields["kind"], fields["title"], ",".join(fields["keywords"]), _wrap(fields["ops"]), _wrap(fields["dtypes"]),
         _wrap(fields["backends"]), fields["graph_pattern"], fields["symptom"], fields["root_cause"],
         fields["solution"], fields["applicability"], json.dumps(evidence, ensure_ascii=False) if evidence else None,
         fields["version_range"] or (f">={issue['fix_version']}" if issue is not None and issue["fix_version"]
                                     else None),
         confidence, status, principal.user_id, agent_id, body, now, now))
    knowledge_id = f"K{cur.lastrowid}"
    conn.execute("UPDATE knowledge SET knowledge_id=? WHERE knowledge_no=?", (knowledge_id, cur.lastrowid))
    conn.execute("INSERT INTO knowledge_fts(rowid, title, keywords, body) VALUES(?,?,?,?)",
                 (cur.lastrowid, fields["title"], " ".join(fields["keywords"]), body))
    if issue is not None:
        conn.execute("INSERT INTO knowledge_links VALUES(?, 'issue', ?)", (knowledge_id, issue["issue_id"]))
        for key in evidence["sample_cases"]:
            conn.execute("INSERT OR IGNORE INTO knowledge_links VALUES(?, 'case', ?)", (knowledge_id, key))
    return knowledge_id


def propose(store: Store, principal: Principal, body: dict):
    """Knowledge outside the issue flow (e.g. measured profile findings) starts as a draft."""
    principal.require_write()
    fields = validate(body.get("knowledge"))
    confidence = "measured" if body.get("measured") is True else "proposed"

    def action(conn):
        return {"knowledge_id": create(conn, principal, fields, confidence=confidence, status="draft",
                                       agent_id=body.get("agent_id") if isinstance(body.get("agent_id"), str)
                                       else None)}
    return idempotent(store, principal, "knowledge-propose", body.pop("request_key", None), body, action)


def set_status(store: Store, principal: Principal, knowledge_id: str, body: dict):
    principal.require_admin()
    status = body.get("status")
    if status not in STATUSES:
        raise IssueError("invalid", f"status must be one of {', '.join(STATUSES)}")

    def action(conn):
        if conn.execute("UPDATE knowledge SET status=?, updated_at=? WHERE knowledge_id=?",
                        (status, now_ms(), knowledge_id)).rowcount != 1:
            raise IssueError("not_found", "knowledge not found", 404)
        conn.execute("INSERT INTO audit_log(at, actor, action, detail) VALUES(?,?,?,?)",
                     (now_ms(), principal.user_id, "knowledge.status", f"{knowledge_id} {status}"))
        for row in conn.execute("SELECT target_id FROM knowledge_links WHERE knowledge_id=? AND target_kind='issue'",
                                (knowledge_id,)).fetchall():
            event(conn, row[0], principal.user_id, "knowledge_status", None, None, f"{knowledge_id} {status}")
        return {"knowledge_id": knowledge_id, "status": status}
    return idempotent(store, principal, f"knowledge-status:{knowledge_id}", body.pop("request_key", None), body,
                      action)


_COLUMNS = ("knowledge_no, knowledge_id, kind, title, keywords, ops, dtypes, backends, graph_pattern, symptom, "
            "root_cause, solution, applicability, evidence, version_range, confidence, status, author_user, agent_id, "
            "created_at, updated_at")


def _row(row, *, body=None) -> dict:
    out = dict(row)
    out.pop("knowledge_no", None)
    out["keywords"] = _unwrap(out["keywords"])
    for name in ("ops", "dtypes", "backends"):
        out[name] = _unwrap(out[name])
    out["evidence"] = json.loads(out["evidence"]) if out["evidence"] else None
    if body is not None:
        out["body_md"] = body
    return out


def search(conn, *, q=None, kind=None, op=None, dtype=None, backend=None, confidence=None,
           include_unpublished=False, after=None, limit=20) -> dict:
    if not 1 <= limit <= 100:
        raise IssueError("invalid", "limit must be 1..100")
    where, args = ["k.knowledge_no<?"], [after if after is not None else 1 << 62]
    if not include_unpublished:
        where.append("k.status='published'")
    for column, value in (("kind", kind), ("confidence", confidence)):
        if value is not None:
            where.append(f"k.{column}=?")
            args.append(value)
    for column, value in (("ops", op), ("dtypes", dtype), ("backends", backend)):
        if value is not None:
            if not _LABEL.match(value):
                raise IssueError("invalid", f"{column} filter is invalid")
            where.append(f"k.{column} LIKE ? ESCAPE '\\'")
            args.append("%," + _escape(value) + ",%")
    if q:
        if not isinstance(q, str) or len(q) > 128:
            raise IssueError("invalid", "q must be at most 128 chars")
        terms = q.split()[:5]
        for term in terms:
            term = term[:32]
            if len(term) >= 3:
                where.append("k.knowledge_no IN (SELECT rowid FROM knowledge_fts WHERE knowledge_fts MATCH ?)")
                args.append('"' + term.replace('"', '""') + '"')
            else:
                where.append("(k.title LIKE ? ESCAPE '\\' OR k.keywords LIKE ? ESCAPE '\\' "
                             "OR k.body_md LIKE ? ESCAPE '\\')")
                args.extend(["%" + _escape(term) + "%"] * 3)
    rows = conn.execute(f"SELECT {_COLUMNS} FROM knowledge k WHERE {' AND '.join(where)} "
                        "ORDER BY k.knowledge_no DESC LIMIT ?", args + [limit]).fetchall()
    items = [_row(r) for r in rows]
    cursor = rows[-1]["knowledge_no"] if len(rows) == limit else None
    return {"items": items, "next_cursor": cursor}


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def get(conn, knowledge_id: str, *, include_unpublished=False) -> dict:
    row = conn.execute(f"SELECT {_COLUMNS}, body_md FROM knowledge WHERE knowledge_id=?", (knowledge_id,)).fetchone()
    if row is None or (row["status"] != "published" and not include_unpublished):
        raise IssueError("not_found", "knowledge not found", 404)
    out = _row({k: row[k] for k in row.keys() if k != "body_md"}, body=row["body_md"])
    out["links"] = [dict(r) for r in conn.execute(
        "SELECT target_kind, target_id FROM knowledge_links WHERE knowledge_id=?", (knowledge_id,))]
    return out


def related(conn, issue: dict, limit=5) -> list:
    """Published knowledge that shares the issue's backend, newest verified first."""
    rows = conn.execute(
        f"SELECT {_COLUMNS} FROM knowledge WHERE status='published' AND backends LIKE ? "
        "ORDER BY (kind=?) DESC, (confidence='verified') DESC, knowledge_no DESC LIMIT ?",
        ("%," + _escape(issue["backend"]) + ",%",
         {"function": "bugfix_function", "accuracy": "bugfix_accuracy"}.get(issue["dimension"], "perf_tuning"),
         limit)).fetchall()
    return [{"knowledge_id": r["knowledge_id"], "title": r["title"], "kind": r["kind"],
             "confidence": r["confidence"]} for r in rows]


def export_jsonl(conn):
    """Yield one JSON line per published entry, for external RAG indexing."""
    for row in conn.execute(f"SELECT {_COLUMNS}, body_md FROM knowledge WHERE status='published' "
                            "ORDER BY knowledge_no"):
        item = _row({k: row[k] for k in row.keys() if k != "body_md"}, body=row["body_md"])
        yield json.dumps(item, ensure_ascii=False) + "\n"
