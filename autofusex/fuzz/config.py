"""Local setup only; upload credentials are administrator-issued, never generated."""
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import uuid


def default_home():
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library/Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    return (base / "autofusex").resolve()


def data_home(override=None):
    if override or os.environ.get("AUTOFUSEX_HOME"):
        return Path(override or os.environ["AUTOFUSEX_HOME"]).expanduser().resolve()
    pointer = default_home() / "active-home.json"
    if pointer.exists():
        value = json.loads(pointer.read_text(encoding="utf-8"))
        if value.get("version") != 1 or not isinstance(value.get("home"), str) or not Path(value["home"]).is_absolute():
            raise ValueError("invalid remembered data directory; use --data-dir to recover")
        return Path(value["home"])
    return default_home()


def remember_home(home):
    """Persist a successful CLI setup location; explicit argument/env still wins."""
    directory = default_home()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix=".active-home-", dir=directory)
    try:
        with os.fdopen(fd, "w") as out:
            json.dump({"version": 1, "home": str(Path(home).resolve())}, out)
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, directory / "active-home.json")
        _sync_directory(directory)
    finally:
        Path(name).unlink(missing_ok=True)


def validate_config(config):
    if not isinstance(config, dict) or type(config.get("version")) is not int or config["version"] != 1:
        raise ValueError("unsupported or damaged configuration; refusing to replace it")
    try:
        uuid.UUID(config["source_id"])
    except (KeyError, TypeError, ValueError, AttributeError):
        raise ValueError("source_id must be a UUID; refusing to replace it") from None
    if "cpu_workers" not in config:
        raise ValueError("configuration is missing cpu_workers")
    for name in ("cpu_workers", "compile_threads", "memory_budget_bytes", "cache_budget_bytes"):
        if name in config and (type(config[name]) is not int or config[name] < 1):
            raise ValueError(f"{name} must be a positive integer")
    if config.get("resource_policy", "manual") not in ("auto", "manual"):
        raise ValueError("resource_policy must be auto or manual")
    identity = config.get("identity", {})
    if not isinstance(identity, dict) or not isinstance(config.get("sync", {}), dict):
        raise ValueError("identity and sync configuration must be objects")
    for name in ("user_id", "display_name", "key_file"):
        if name in identity:
            value = identity[name]
            if not isinstance(value, str) or not value.strip() or any(ord(c) < 32 for c in value):
                raise ValueError(f"identity.{name} must be nonempty text without control characters")
    # Unknown fields are preserved for forward-compatible local configuration.
    return config


def load_config(home):
    path = data_home(home) / "config.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError):
        raise ValueError("configuration is not valid UTF-8 JSON") from None
    return validate_config(value)


def identity_status(config):
    """Metadata-only checks. Never read/print the secret or authenticate display names."""
    identity = config.get("identity", {})
    missing = [name for name in ("user_id", "display_name", "key_file") if not identity.get(name)]
    warnings = []
    if identity.get("key_file"):
        try:
            info = Path(identity["key_file"]).stat()
            if not stat.S_ISREG(info.st_mode) or info.st_size == 0:
                warnings.append("administrator key must be a nonempty regular file")
            if os.name == "posix" and info.st_mode & 0o077:
                warnings.append("administrator key permissions allow group/other access; use mode 0600")
        except OSError:
            warnings.append("administrator key file is unavailable")
    return {"configured": not missing, "missing_fields": missing, "warnings": warnings,
            "authentication": "not_checked; display name is not an authentication identity"}


def _sync_directory(directory):
    # Atomic replace is portable. Directory fsync is supported on Linux/macOS,
    # but not all filesystems/Windows; do not turn a successful write into a retry.
    if os.name == "posix":
        try:
            fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass


def setup(home, *, display_name=None, user_id=None, key_file=None, cpu_workers=None,
          compile_threads=None, memory_budget_bytes=None, cache_budget_bytes=None, resource_policy=None):
    home = data_home(home)
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = home / ".setup.lock"
    # Exclusive creation avoids concurrent source ID replacement. Never steal stale locks.
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise FileExistsError("setup lock exists; check for active setup before removing the lock") from None
    temporary = None
    try:
        os.close(fd)
        config = load_config(home)
        if config is None:
            config = {"version": 1, "source_id": str(uuid.uuid4()), "cpu_workers": 1,
                      "compile_threads": 4, "sync": {"state": "not_configured"}}
        for name, value in (("cpu_workers", cpu_workers), ("compile_threads", compile_threads),
                            ("memory_budget_bytes", memory_budget_bytes), ("cache_budget_bytes", cache_budget_bytes)):
            if value is not None:
                config[name] = value
        if resource_policy is not None:
            config["resource_policy"] = resource_policy
        identity = config.setdefault("identity", {})
        for name, value in (("display_name", display_name), ("user_id", user_id)):
            if value is not None:
                identity[name] = value
        if key_file is not None:
            key = Path(key_file).expanduser().resolve()
            if not key.is_file() or key.stat().st_size == 0:
                raise ValueError("administrator key must be a nonempty regular file")
            identity["key_file"] = str(key)  # Never copy or print its contents.
        validate_config(config)
        for directory in ("results", "tmp", "cache"):
            (home / directory).mkdir(exist_ok=True, mode=0o700)
        with tempfile.NamedTemporaryFile(mode="w", dir=home, delete=False, encoding="utf-8") as out:
            temporary = Path(out.name)
            json.dump(config, out, indent=2)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        temporary.replace(home / "config.json")
        _sync_directory(home)
        return {"home": str(home), "source_id": config["source_id"],
                "identity_configured": identity_status(config)["configured"],
                "identity": identity_status(config), "sync": "not_implemented",
                "cpu_workers": config["cpu_workers"], "compile_threads": config.get("compile_threads", 4),
                "budget_enforcement": "managed CPU run: admission + sampled memory/scratch guards; serial worker; replay not governed"}
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
        lock.unlink()
