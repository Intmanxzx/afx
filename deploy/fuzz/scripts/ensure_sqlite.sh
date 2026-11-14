#!/usr/bin/env bash
# Ensure Python's sqlite3 library supports UPSERT without building SQLite locally.
set -euo pipefail
usage() {
  cat <<'HELP'
Usage: ensure_sqlite.sh [PYTHON]
Checks that Python's sqlite3 library is at least 3.31. If needed, installs the
distribution SQLite package and remaps a dynamically loaded private SQLite
library to it. No source build is performed.
HELP
}
[[ ${1:-} == -h || ${1:-} == --help ]] && { usage; exit 0; }
PYTHON_BIN=${1:-python3}
MIN_VERSION=3.31.0

version_ge() {
  "$PYTHON_BIN" -I - "$1" "$MIN_VERSION" <<'PY'
import sys
def version(value):
    return tuple(int(part) for part in value.split('.')[:3])
raise SystemExit(0 if version(sys.argv[1]) >= version(sys.argv[2]) else 1)
PY
}

check_features() {
  "$PYTHON_BIN" -I - <<'PY'
import sqlite3
with sqlite3.connect(':memory:') as conn:
    conn.execute('CREATE TABLE sample(id INTEGER PRIMARY KEY, n INTEGER)')
    for _ in range(2):
        conn.execute('INSERT INTO sample VALUES(1,1) ON CONFLICT(id) DO UPDATE SET n=n+1')
    assert conn.execute('SELECT n, row_number() OVER (), json_extract(?, ?) FROM sample',
                        ('{"ok":1}', '$.ok')).fetchone() == (2, 1, 1)
PY
}

current_version=$("$PYTHON_BIN" -I -c 'import sqlite3; print(sqlite3.sqlite_version)')
if version_ge "$current_version"; then
  check_features
  echo "SQLite ${current_version} is ready."
  exit 0
fi
echo "Python SQLite ${current_version} is older than ${MIN_VERSION}; installing the distribution package."

SUDO=()
if ((EUID != 0)); then
  command -v sudo >/dev/null || { echo 'Root or sudo is required to install SQLite.' >&2; exit 1; }
  SUDO=(sudo)
fi
if command -v apt-get >/dev/null; then
  "${SUDO[@]}" apt-get update -o Acquire::http::Proxy=false -o Acquire::https::Proxy=false
  "${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::http::Proxy=false -o Acquire::https::Proxy=false install -y sqlite3 libsqlite3-0
elif command -v dnf >/dev/null; then
  "${SUDO[@]}" dnf install -y sqlite sqlite-libs
elif command -v yum >/dev/null; then
  "${SUDO[@]}" yum install -y sqlite sqlite-libs
elif command -v zypper >/dev/null; then
  "${SUDO[@]}" zypper --non-interactive install sqlite3 libsqlite3-0
else
  echo 'No supported package manager found. Install SQLite >= 3.31 with the OS package manager.' >&2
  exit 1
fi

current_version=$("$PYTHON_BIN" -I -c 'import sqlite3; print(sqlite3.sqlite_version)')
if version_ge "$current_version"; then
  check_features
  echo "SQLite ${current_version} is ready."
  exit 0
fi

# Some prebuilt Pythons have RUNPATH pointing at a private old SQLite. Prefer
# the newly installed system library without rebuilding the Python extension.
extension=$("$PYTHON_BIN" -I -c 'import _sqlite3; print(_sqlite3.__file__)')
loaded_library=$(ldd "$extension" 2>/dev/null | awk '$1=="libsqlite3.so.0" {print $3; exit}')
[[ -n $loaded_library ]] || {
  echo "Python ${PYTHON_BIN} has a statically linked SQLite ${current_version}; use a distribution Python or Python linked to SQLite >= 3.31." >&2
  exit 1
}
private_dir=$(dirname "$loaded_library")
case $private_dir in
  /lib|/lib/*|/lib64|/usr/lib|/usr/lib/*|/usr/lib64|/usr/lib64/*)
    echo 'Distribution SQLite is too old; use a newer distribution Python/package.' >&2; exit 1;;
esac
system_library=$("$PYTHON_BIN" -I - "$private_dir" <<'PY'
import subprocess, sys
exclude=sys.argv[1]
lines=subprocess.run(['ldconfig','-p'], text=True, capture_output=True, check=True).stdout.splitlines()
for line in lines:
    fields=line.split()
    if len(fields)>=4 and fields[0]=='libsqlite3.so.0' and not fields[-1].startswith(exclude):
        print(fields[-1]); break
PY
)
[[ -n $system_library ]] || { echo 'No system libsqlite3.so.0 found after package installation.' >&2; exit 1; }
system_version=$("$PYTHON_BIN" -I - "$system_library" <<'PY'
import ctypes, sys
library=ctypes.CDLL(sys.argv[1])
library.sqlite3_libversion.restype=ctypes.c_char_p
print(library.sqlite3_libversion().decode())
PY
)
version_ge "$system_version" || {
  echo "Distribution SQLite ${system_version} is still older than ${MIN_VERSION}; use a newer distribution package." >&2
  exit 1
}

if [[ $(readlink -f "$loaded_library") != $(readlink -f "$system_library") ]]; then
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  backup="${loaded_library}.pre-autofuse-sqlite-${stamp}"
  "${SUDO[@]}" cp -a "$loaded_library" "$backup"
  replacement="${loaded_library}.autofuse-new-$$"
  "${SUDO[@]}" ln -s "$system_library" "$replacement"
  "${SUDO[@]}" mv -Tf "$replacement" "$loaded_library"
  echo "Remapped $loaded_library to $system_library; backup: $backup"
fi
if ! current_version=$("$PYTHON_BIN" -I -c 'import sqlite3; print(sqlite3.sqlite_version)') ||
   ! version_ge "$current_version" || ! check_features; then
  if [[ -n ${backup:-} ]]; then
    "${SUDO[@]}" cp -a "$backup" "$replacement"
    "${SUDO[@]}" mv -Tf "$replacement" "$loaded_library"
  fi
  echo "SQLite validation failed; restored the previous library. Inspect $extension with ldd." >&2
  exit 1
fi
echo "SQLite ${current_version} is ready."
