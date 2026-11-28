"""Consistent backups of the platform data directory.

An archive is one tar file holding gzip-compressed SQLite snapshots taken with
the online backup API (safe while the server is writing), secret.key and a
manifest with SHA-256 of every member. stats.db is never backed up: it is
rebuilt from core.db on start.

core.db is snapshotted before content.db, so every content id referenced by the
core snapshot is present in the content snapshot (content is append-only).
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import sqlite3
import tarfile
import tempfile
import time
from pathlib import Path

FORMAT = "afx-platform-backup/1"
_CHUNK = 1 << 20


def _snapshot(source: Path, target: Path) -> None:
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst, pages=4096)
        dst.execute("PRAGMA journal_mode=DELETE")
    finally:
        dst.close()
        src.close()


def _gzip(source: Path, target: Path, *, compresslevel: int = 1) -> str:
    if not 1 <= compresslevel <= 9:
        raise ValueError("compression level must be between 1 and 9")
    digest = hashlib.sha256()
    with open(source, "rb") as fin, gzip.open(target, "wb", compresslevel=compresslevel) as fout:
        while chunk := fin.read(_CHUNK):
            digest.update(chunk)
            fout.write(chunk)
    return digest.hexdigest()


def create(data_dir, out_dir, *, include_content: bool = True, tmp_dir=None,
           compression_level: int = 1) -> Path:
    """Write <out_dir>/afx-platform-<UTC>[-core].tar and return its path."""
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    if not (data_dir / "core.db").exists():
        raise ValueError(f"{data_dir} has no core.db")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    name = f"afx-platform-{stamp}{'' if include_content else '-core'}.tar"
    members = ["core.db"] + (["content.db"] if include_content else [])
    with tempfile.TemporaryDirectory(dir=tmp_dir) as work:
        work = Path(work)
        manifest = {"format": FORMAT, "created_at": stamp, "files": {}}
        for member in members:
            raw = work / member
            _snapshot(data_dir / member, raw)
            if member == "core.db":
                with sqlite3.connect(raw) as conn:
                    manifest["ingest_seq"] = conn.execute(
                        "SELECT value FROM meta WHERE key='ingest_seq'").fetchone()[0]
            manifest["files"][member + ".gz"] = {"sha256": _gzip(
                raw, work / (member + ".gz"), compresslevel=compression_level),
                                                 "raw_size": raw.stat().st_size}
            raw.unlink()
        secret = (data_dir / "secret.key").read_bytes()
        (work / "secret.key").write_bytes(secret)
        manifest["files"]["secret.key"] = {"sha256": hashlib.sha256(secret).hexdigest(), "raw_size": len(secret)}
        partial = out_dir / (name + ".partial")
        fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as handle, tarfile.open(fileobj=handle, mode="w") as tar:
            body = json.dumps(manifest, indent=2).encode()
            info = tarfile.TarInfo("manifest.json")
            info.size, info.mode, info.mtime = len(body), 0o600, int(time.time())
            tar.addfile(info, io.BytesIO(body))
            for member in manifest["files"]:
                tar.add(work / member, arcname=member)
        target = out_dir / name
        os.replace(partial, target)
    return target


def _extract(archive: Path, dest: Path) -> dict:
    with tarfile.open(archive, "r:") as tar:
        names = tar.getnames()
        if "manifest.json" not in names:
            raise ValueError("archive has no manifest.json")
        manifest = json.loads(tar.extractfile("manifest.json").read())
        if manifest.get("format") != FORMAT:
            raise ValueError(f"unsupported backup format {manifest.get('format')}")
        expected = set(manifest["files"]) | {"manifest.json"}
        if set(names) != expected:
            raise ValueError(f"archive members {sorted(names)} do not match manifest")
        for member in manifest["files"]:
            info = tar.getmember(member)
            if not info.isfile() or "/" in member or member.startswith("."):
                raise ValueError(f"unsafe archive member {member}")
            with tar.extractfile(info) as src, open(dest / member, "wb") as out:
                shutil.copyfileobj(src, out, _CHUNK)
    for member, meta in manifest["files"].items():
        path = dest / member
        if member.endswith(".gz"):
            raw = dest / member[:-3]
            digest = hashlib.sha256()
            with gzip.open(path, "rb") as fin, open(raw, "wb") as fout:
                while chunk := fin.read(_CHUNK):
                    digest.update(chunk)
                    fout.write(chunk)
            path.unlink()
        else:
            raw = path
            digest = hashlib.sha256(raw.read_bytes())
        if digest.hexdigest() != meta["sha256"]:
            raise ValueError(f"{member} failed SHA-256 check")
    return manifest


def verify(archive, *, tmp_dir=None) -> dict:
    """Extract to a temporary directory, check hashes and SQLite integrity."""
    with tempfile.TemporaryDirectory(dir=tmp_dir) as work:
        work = Path(work)
        manifest = _extract(Path(archive), work)
        for db in ("core.db", "content.db"):
            if (work / db).exists():
                with sqlite3.connect(work / db) as conn:
                    result = conn.execute("PRAGMA integrity_check").fetchone()[0]
                if result != "ok":
                    raise ValueError(f"{db} integrity_check: {result}")
        return {"created_at": manifest["created_at"], "ingest_seq": manifest.get("ingest_seq"),
                "files": sorted(manifest["files"])}


def restore(archive, data_dir, content_archive=None) -> dict:
    """Restore into an empty or missing data directory; stats.db is rebuilt on next start.

    For a core-only archive pass the latest full archive as content_archive; logs
    written between the two snapshots are reported as missing, never invented.
    """
    data_dir = Path(data_dir)
    if data_dir.exists() and any(data_dir.iterdir()):
        raise ValueError(f"{data_dir} is not empty; restore only into an empty directory")
    data_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(data_dir, 0o700)
    manifest = _extract(Path(archive), data_dir)
    if content_archive is not None and not (data_dir / "content.db").exists():
        with tempfile.TemporaryDirectory(dir=data_dir.parent) as work:
            _extract(Path(content_archive), Path(work))
            if not (Path(work) / "content.db").exists():
                raise ValueError("content archive has no content.db")
            shutil.move(str(Path(work) / "content.db"), data_dir / "content.db")
    for path in data_dir.iterdir():
        os.chmod(path, 0o600)
    with sqlite3.connect(data_dir / "core.db") as conn:
        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("restored core.db failed integrity_check")
    return {"created_at": manifest["created_at"], "content_restored": (data_dir / "content.db").exists()}
