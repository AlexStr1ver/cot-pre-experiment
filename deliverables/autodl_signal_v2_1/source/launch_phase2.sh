#!/usr/bin/env bash
# Detached launch survives ordinary SSH/browser disconnects, not machine shutdown/SIGKILL.
if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    printf 'Run this script with bash; do not source it.\n' >&2
    return 2
fi
set -euo pipefail
ROOT=${1:-/root/autodl-tmp}
LIMIT=${2:-3}
PKG=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
command -v nohup >/dev/null
command -v setsid >/dev/null
EXP=$(mktemp -d "$ROOT/infty_signal_phase2fix_XXXXXXXX")
EXP=$(cd -- "$EXP" && pwd)
nohup setsid bash "$PKG/run_phase2.sh" "$ROOT" "$LIMIT" "$EXP" \
    > "$EXP/launcher.log" 2>&1 < /dev/null &
launcher_pid=$!
printf '%s\n' "$launcher_pid" > "$EXP/launcher.pid"
printf 'OUTPUT_DIRECTORY=%s\nLAUNCHER_PID=%s\nLOG=%s/launcher.log\n' "$EXP" "$launcher_pid" "$EXP"
printf '查看日志：tail -f %q\n' "$EXP/launcher.log"
printf '查看状态：%q %q %q\n' "$ROOT/venvs/infty-official/bin/python" "$PKG/status_phase2.py" "$EXP"
printf '运行结束后下载：%s_return.zip\n' "$EXP"
