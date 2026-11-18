#!/usr/bin/env bash
# Install or upgrade the AutofuseX platform server under systemd.
# Creates a new venv per wheel digest, keeps the data directory, restarts, checks /healthz.
set -euo pipefail
usage() {
  cat <<'HELP'
Usage: install_server.sh --wheel FILE --root DIR [--port 8100] [--user USER] [--service afx-platform] [--python python3]
  ROOT/data holds core.db, content.db, stats.db and secret.key (created on first start).
  Users and tokens are issued afterwards with:
    ROOT/bin/afx platform --data ROOT/data user add NAME --role admin
    ROOT/bin/afx platform --data ROOT/data token issue NAME --kind web
HELP
}
WHEEL=; ROOT=; PORT=8100; USER_NAME=$(id -un); SERVICE=afx-platform; PYTHON_BIN=python3
while (($#)); do
  [[ $1 == -h || $1 == --help ]] && { usage; exit 0; }
  (($# >= 2)) || { usage >&2; exit 2; }
  case $1 in
    --wheel) WHEEL=$2;; --root) ROOT=$2;; --port) PORT=$2;; --user) USER_NAME=$2;;
    --service) SERVICE=$2;; --python) PYTHON_BIN=$2;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
  shift 2
done
[[ -f $WHEEL && $ROOT == /* && $PORT =~ ^[0-9]+$ && $SERVICE =~ ^[a-zA-Z0-9_-]+$ ]] || { usage >&2; exit 2; }
SUDO=(); ((EUID == 0)) || SUDO=(sudo)
umask 077
mkdir -p "$ROOT/deployments" "$ROOT/data" "$ROOT/bin"
DIGEST=$(sha256sum "$WHEEL" | cut -c1-16)
VENV="$ROOT/deployments/platform-$DIGEST"
if [[ ! -x $VENV/bin/python ]]; then
  "$PYTHON_BIN" -m venv "$VENV"
  "$VENV/bin/python" -m pip install -q "$WHEEL"
fi
(cd / && "$VENV/bin/python" -I -c 'import autofusex, importlib.metadata as m; print(autofusex.__file__, m.version("autofusex"))')
# Prove the new code opens (and migrates) the data directory before switching the service.
"$VENV/bin/python" -I -c 'import sys; from autofusex.platform.server import create_server; s=create_server(sys.argv[1], port=0); s.server_close()' "$ROOT/data"
cat > "$ROOT/bin/afx" <<EOF
#!/usr/bin/env bash
exec "$VENV/bin/python" -I -m afx "\$@"
EOF
chmod 755 "$ROOT/bin/afx"
UNIT="/etc/systemd/system/$SERVICE.service"
"${SUDO[@]}" tee "$UNIT" >/dev/null <<EOF
[Unit]
Description=AutofuseX platform server
After=network.target
[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$ROOT
ExecStart=$VENV/bin/python -I -m afx platform --data $ROOT/data serve --host 127.0.0.1 --port $PORT
Restart=on-failure
RestartSec=5
UMask=0077
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=$ROOT/data
MemoryMax=600M
[Install]
WantedBy=multi-user.target
EOF
"${SUDO[@]}" systemctl daemon-reload
"${SUDO[@]}" systemctl enable "$SERVICE" >/dev/null
"${SUDO[@]}" systemctl restart "$SERVICE"
for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
    echo "$SERVICE is up on 127.0.0.1:$PORT using $VENV"
    exit 0
  fi
  sleep 1
done
echo "$SERVICE did not become healthy; see: journalctl -u $SERVICE" >&2
exit 1
