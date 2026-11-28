#!/bin/sh
# Back up the platform data directory and push the archive to another machine.
#   backup.sh full   # core.db + content.db + secret.key (daily)
#   backup.sh core   # core.db + secret.key (every 6 hours)
# Configuration comes from /etc/afx-platform/backup.env (see backup.env.example).
set -eu
umask 077

MODE=${1:-full}
case "$MODE" in full|core) ;; *) echo "usage: $0 full|core" >&2; exit 2 ;; esac

ENV_FILE=${AFX_BACKUP_ENV:-/etc/afx-platform/backup.env}
[ -r "$ENV_FILE" ] && . "$ENV_FILE"

: "${AFX_DATA:?AFX_DATA (platform data directory) is required}"
: "${BACKUP_REMOTE_HOST:?BACKUP_REMOTE_HOST (user@host) is required}"
: "${BACKUP_REMOTE_DIR:?BACKUP_REMOTE_DIR (absolute path on the remote) is required}"
AFX_BIN=${AFX_BIN:-afx}
# Staging lives on the system disk, never on the small data disk.
STAGING=${BACKUP_STAGING:-/var/tmp/afx-platform-backup}
KEEP_CORE_DAYS=${BACKUP_KEEP_CORE_DAYS:-7}
KEEP_FULL_DAYS=${BACKUP_KEEP_FULL_DAYS:-14}
COMPRESSION_LEVEL=${BACKUP_COMPRESSION_LEVEL:-1}
SSH_OPTS=${BACKUP_SSH_OPTS:-"-o BatchMode=yes -o ConnectTimeout=20"}

mkdir -p "$STAGING"
export SQLITE_TMPDIR="$STAGING"

FLAG=""
[ "$MODE" = core ] && FLAG="--core-only"

# Remove leftovers of an earlier failed run (only this script's own files).
find "$STAGING" -maxdepth 1 -type f -name 'afx-platform-*.tar*' -mmin +60 -delete

OUT=$("$AFX_BIN" platform --data "$AFX_DATA" backup --out "$STAGING" --tmp "$STAGING" \
  --compression-level "$COMPRESSION_LEVEL" $FLAG)
ARCHIVE=$(printf '%s' "$OUT" | sed -n 's/.*"archive": *"\([^"]*\)".*/\1/p')
[ -n "$ARCHIVE" ] && [ -f "$ARCHIVE" ] || { echo "backup failed: $OUT" >&2; exit 1; }

# Never ship an archive that does not restore cleanly.
"$AFX_BIN" platform --data "$AFX_DATA" verify-backup "$ARCHIVE" --tmp "$STAGING" >/dev/null

NAME=$(basename "$ARCHIVE")
# Upload under a temporary name, then rename, so the remote never holds a partial archive.
# shellcheck disable=SC2086
rsync -a --partial -e "ssh $SSH_OPTS" "$ARCHIVE" "$BACKUP_REMOTE_HOST:$BACKUP_REMOTE_DIR/.$NAME.part"
# shellcheck disable=SC2086
ssh $SSH_OPTS "$BACKUP_REMOTE_HOST" \
  "mv '$BACKUP_REMOTE_DIR/.$NAME.part' '$BACKUP_REMOTE_DIR/$NAME' &&
   find '$BACKUP_REMOTE_DIR' -maxdepth 1 -type f -name 'afx-platform-*-core.tar' -mtime +$KEEP_CORE_DAYS -delete &&
   find '$BACKUP_REMOTE_DIR' -maxdepth 1 -type f -name 'afx-platform-*Z.tar' -mtime +$KEEP_FULL_DAYS -delete"

rm -f "$ARCHIVE"
echo "pushed $NAME to $BACKUP_REMOTE_HOST:$BACKUP_REMOTE_DIR"
