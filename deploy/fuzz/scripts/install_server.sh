#!/usr/bin/env bash
# Linux/systemd deployment. Preserve database, identities and existing service env.
set -euo pipefail
usage() {
  cat <<'HELP'
Usage: install_server.sh --wheel FILE --root DIR [options]
  --python PATH      Python with venv support (default: python3)
  --database PATH    Existing/new database (default: ROOT/data/results.sqlite)
  --port N           Loopback HTTP port (default: 8090)
  --user USER        Service account (default: current user)
  --service NAME     systemd service name (default: af-fuzz)
  --venv DIR         New venv (default: ROOT/deployments/server-WHEEL_SHA256)
Existing databases are backed up consistently in ROOT/private before restart.
Existing systemd environment overrides are retained. No identity/token is created.
HELP
}
WHEEL=; ROOT=; DATABASE=; PORT=8090; USER_NAME=$(id -un); SERVICE=af-fuzz; PYTHON_BIN=python3; VENV=
while (($#)); do
  [[ $1 == -h || $1 == --help ]] && { usage; exit 0; }
  (($#>=2)) || { usage >&2; exit 2; }
  case $1 in
    --wheel) WHEEL=$2;; --root) ROOT=$2;; --database) DATABASE=$2;; --port) PORT=$2;;
    --user) USER_NAME=$2;; --service) SERVICE=$2;; --python) PYTHON_BIN=$2;; --venv) VENV=$2;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
  shift 2
done
[[ -f $WHEEL && $ROOT == /* && $PORT =~ ^[0-9]+$ && $SERVICE =~ ^[a-zA-Z0-9_-]+$ ]] || { usage >&2; exit 2; }
((PORT>0 && PORT<65536)) || exit 2
id "$USER_NAME" >/dev/null
command -v systemctl >/dev/null
SUDO=(); ((EUID==0)) || SUDO=(sudo)
unset PYTHONPATH PYTHONHOME
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bash "$SCRIPT_DIR/ensure_sqlite.sh" "$PYTHON_BIN"
umask 077
WHEEL=$(realpath "$WHEEL"); mkdir -p "$ROOT"; ROOT=$(realpath "$ROOT")
DATABASE="${DATABASE:-$ROOT/data/results.sqlite}"
[[ $DATABASE == /* && $ROOT != *[[:space:]]* && $DATABASE != *[[:space:]]* ]] || { echo 'Use absolute paths without whitespace for systemd deployment.' >&2; exit 2; }
DIGEST=$("$PYTHON_BIN" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest()[:16])' "$WHEEL")
VENV="${VENV:-$ROOT/deployments/server-$DIGEST}"
[[ ! -e $VENV ]] || { echo "Refusing to overwrite existing venv: $VENV" >&2; exit 1; }
"$PYTHON_BIN" -m venv "$VENV"; VENV=$(realpath "$VENV")
"$VENV/bin/python" -m pip install "$WHEEL"
# Permit a separate service account to traverse and execute the isolated venv.
chmod -R a+rX "$VENV"
(cd / && "$VENV/bin/python" -I - <<'PY'
import autofusedebugger, importlib.metadata, yaml, jsonschema, cryptography
print(autofusedebugger.__file__)
print(importlib.metadata.version('autofusedebugger'))
PY
)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
PRIVATE="$ROOT/private/deploy-$STAMP-$DIGEST"
mkdir -p "$PRIVATE"; chmod 700 "$PRIVATE"
if [[ -f $DATABASE ]]; then
  "$VENV/bin/python" -m autofusedebugger.fuzz admin backup "$DATABASE" "$PRIVATE/results.sqlite"
  # Prove the new code can start against a private copy before opening live data.
  "$VENV/bin/python" - "$PRIVATE/results.sqlite" <<'PY'
import sys
from autofusedebugger.fuzz.central.server import create_server
server=create_server(sys.argv[1], port=0)
server.server_close()
PY
fi
systemctl cat "$SERVICE" > "$PRIVATE/service-before.txt" 2>/dev/null || true
FRAGMENT="/etc/systemd/system/$SERVICE.service"
DROPIN="/etc/systemd/system/$SERVICE.service.d/zzzzzz-fuzz-deploy.conf"
if [[ -f $DROPIN ]]; then cp "$DROPIN" "$PRIVATE/previous-dropin.conf"; fi
if ! systemctl cat "$SERVICE" >/dev/null 2>&1; then
  cat > "$PRIVATE/base.service" <<CONF
[Unit]
Description=Autofuse fuzz result server
After=network.target
[Service]
Type=simple
Restart=on-failure
RestartSec=5
UMask=0077
[Install]
WantedBy=multi-user.target
CONF
  "${SUDO[@]}" install -m 644 "$PRIVATE/base.service" "$FRAGMENT"
fi
mkdir -p "$(dirname "$DATABASE")"
"${SUDO[@]}" chown "$USER_NAME" "$ROOT" "$(dirname "$DATABASE")"
"${SUDO[@]}" chmod 755 "$ROOT" "$(dirname "$VENV")"
cat > "$PRIVATE/new-dropin.conf" <<CONF
[Service]
User=$USER_NAME
WorkingDirectory=$ROOT
ExecStart=
ExecStart=$VENV/bin/python -m autofusedebugger.fuzz admin serve $DATABASE --port $PORT
Environment=PYTHONPATH=
Environment=PYTHONHOME=
ReadWritePaths=$ROOT $(dirname "$DATABASE")
CONF
"${SUDO[@]}" mkdir -p "$(dirname "$DROPIN")"
"${SUDO[@]}" install -m 644 "$PRIVATE/new-dropin.conf" "$DROPIN"
"${SUDO[@]}" systemctl daemon-reload
"${SUDO[@]}" systemctl enable "$SERVICE"
"${SUDO[@]}" systemctl restart "$SERVICE"
"$VENV/bin/python" - "$PORT" <<'PY'
import json, sys, time, urllib.request
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for i in range(30):
    try:
        with opener.open('http://127.0.0.1:'+sys.argv[1]+'/api/summary', timeout=10) as response:
            assert response.status==200
            json.load(response)
        break
    except Exception:
        if i==29: raise
        time.sleep(1)
PY
systemctl is-active "$SERVICE"
echo "Server verified: $VENV; private backup: $PRIVATE"
