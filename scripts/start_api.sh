#!/usr/bin/env bash
# Start the API for local development.
#
# Stopping an existing instance uses a PID file rather than matching the
# process list: a `pkill -f "uvicorn app.main"` pattern also matches the shell
# that is running the pkill, so the script kills itself. `ss -ltnp` is no help
# either, since it hides other users' PIDs without privileges.
set -euo pipefail

PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
PYTHON="${PYTHON:-python}"
RUNDIR="${RUNDIR:-.run}"
PIDFILE="$RUNDIR/api-$PORT.pid"

mkdir -p "$RUNDIR"

if [ -f "$PIDFILE" ]; then
  OLD_PID="$(cat "$PIDFILE")"
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "stopping existing API on :$PORT (pid $OLD_PID)"
    kill "$OLD_PID" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      kill -0 "$OLD_PID" 2>/dev/null || break
      sleep 0.5
    done
    kill -9 "$OLD_PID" 2>/dev/null || true
  fi
  rm -f "$PIDFILE"
fi

echo "starting API on http://$HOST:$PORT (docs at /docs)"
"$PYTHON" -m uvicorn app.main:app --host "$HOST" --port "$PORT" "$@" &
APP_PID=$!
echo "$APP_PID" > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT
wait "$APP_PID"
