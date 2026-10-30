#!/usr/bin/env bash
# Detached launch survives ordinary terminal disconnects, not instance shutdown.
if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    printf 'Run with bash; do not source this script.\n' >&2
    return 2
fi
set -euo pipefail
ROOT=${1:-/root/autodl-tmp}
ANCHOR=${2:-$ROOT/infty_paired_v3_rsiFEcc2}
RESUME=${3:-}
PKG=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
command -v nohup >/dev/null
command -v setsid >/dev/null
EXP=$(mktemp -d "$ROOT/infty_expand_v4_XXXXXXXX")
EXP=$(cd -- "$EXP" && pwd)
nohup setsid bash "$PKG/run_expanded.sh" "$ROOT" "$ANCHOR" "$RESUME" "$EXP" \
    > "$EXP/launcher.log" 2>&1 < /dev/null &
launcher_pid=$!
printf '%s\n' "$launcher_pid" > "$EXP/launcher.pid"
printf 'OUTPUT_DIRECTORY=%s\nLAUNCHER_PID=%s\nMODE=expand\nLOG=%s/launcher.log\n' "$EXP" "$launcher_pid" "$EXP"
printf '查看日志：tail -f %q\n' "$EXP/launcher.log"
STATUS_PY=${INFTY_PAIRED_PYTHON:-$ROOT/venvs/infty-official/bin/python}
if [[ ! -x "$STATUS_PY" ]]; then STATUS_PY=$(command -v python3); fi
printf '查看状态：%q -I %q script status_expanded.py %q\n' "$STATUS_PY" "$PKG/isolated_expand_entry.py" "$EXP"
printf '停止任务：kill -TERM $(cat %q)\n' "$EXP/runner.pid"
printf '运行结束后下载：%s_return.zip\n' "$EXP"
