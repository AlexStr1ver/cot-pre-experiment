#!/usr/bin/env bash
# Execute with bash, never source into an interactive shell.
if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    printf 'Run with bash; do not source this script.\n' >&2
    return 2
fi
set -euo pipefail
ROOT=${1:-/root/autodl-tmp}
MODE=${2:-smoke}
PREVIOUS=${3:-}
PKG=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PY="$ROOT/venvs/infty-official/bin/python"
MODEL="$ROOT/inftythink_day1/models/InftyThink-1.5B"
BASELINE="$ROOT/inftythink_official_check/outputs/native5"
# Fourth argument is a newly created directory supplied by the detached launcher.
EXP=${4:-$(mktemp -d "$ROOT/infty_paired_v3_XXXXXXXX")}
EXP=$(cd -- "$EXP" && pwd)
shopt -s nullglob dotglob
for existing in "$EXP"/*; do
    case $(basename -- "$existing") in
        launcher.log|launcher.pid) ;;
        *) printf 'Refusing to reuse nonempty result directory: %s\n' "$EXP" >&2; exit 2 ;;
    esac
done
shopt -u nullglob dotglob
mkdir "$EXP/.runner_claim"
printf '%s\n' "$$" > "$EXP/runner.pid"
awk '{print $22}' "/proc/$$/stat" > "$EXP/runner.start_ticks"
printf 'OUTPUT_DIRECTORY=%s\nMODE=%s\n' "$EXP" "$MODE"
PACK_PY="$PY"
if [[ ! -x "$PACK_PY" ]]; then PACK_PY=$(command -v python3 || true); fi
ACTIVE_PID=
ACTIVE_TEE=

stop_worker() {
    if [[ -n "$ACTIVE_PID" ]]; then
        # execute_logged starts each worker in its own session/process group.
        kill -TERM -- "-$ACTIVE_PID" 2>/dev/null || true
        wait "$ACTIVE_PID" 2>/dev/null || true
        ACTIVE_PID=
    fi
    if [[ -n "$ACTIVE_TEE" ]]; then
        wait "$ACTIVE_TEE" 2>/dev/null || true
        ACTIVE_TEE=
    fi
}
finish() {
    local task_exit=$?
    trap - EXIT INT TERM
    stop_worker
    printf '%s\n' "$task_exit" > "$EXP/exit_status.txt"
    if [[ -n "$PACK_PY" ]]; then
        "$PACK_PY" -u - "$PKG" "$EXP" <<'PY' || printf 'ARCHIVE_FAILED: use status_paired.py RUN_DIRECTORY --pack after workers stop\n' >&2
import pathlib, sys
sys.path.insert(0, sys.argv[1])
from status_paired import pack_result
print('RETURN_ZIP=' + str(pack_result(pathlib.Path(sys.argv[2]))), flush=True)
PY
    else
        printf 'ARCHIVE_FAILED: no Python; preserved result directory %s\n' "$EXP" >&2
    fi
    printf 'EXPERIMENT_EXIT_STATUS=%s\n' "$task_exit"
    exit "$task_exit"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

execute_logged() {
    local log=$1 worker_exit=0 tee_exit=0 sink
    shift
    # Keep the tee PID so the archive is created only after its final bytes flush.
    exec {sink}> >(tee "$log")
    ACTIVE_TEE=$!
    setsid "$@" >&"$sink" 2>&1 &
    ACTIVE_PID=$!
    exec {sink}>&-
    printf '%s\n' "$ACTIVE_PID" > "$EXP/worker.pid"
    if [[ -r "/proc/$ACTIVE_PID/stat" ]]; then
        awk '{print $22}' "/proc/$ACTIVE_PID/stat" > "$EXP/worker.start_ticks"
    fi
    wait "$ACTIVE_PID" || worker_exit=$?
    ACTIVE_PID=
    wait "$ACTIVE_TEE" || tee_exit=$?
    if [[ $worker_exit == 0 && $tee_exit != 0 ]]; then worker_exit=$tee_exit; fi
    ACTIVE_TEE=
    return "$worker_exit"
}

case "$MODE" in smoke|main|replicate) ;; *) printf 'Mode must be smoke, main or replicate.\n' >&2; exit 2 ;; esac
command -v setsid >/dev/null
cd "$PKG"
sha256sum -c checksums.sha256 2>&1 | tee "$EXP/package_checksums.log"
test -x "$PY"
mkdir "$EXP/code"
shopt -s nullglob
for source in "$PKG"/*.py "$PKG"/*.sh "$PKG"/*.md "$PKG"/*.json "$PKG"/checksums.sha256; do
    [[ ! -f "$source" ]] || cp -- "$source" "$EXP/code/"
done
shopt -u nullglob
cp -a -- "$PKG/evidence_phase2fix" "$EXP/code/"
printf 'PHASE=cpu_tests\n'
execute_logged "$EXP/tests.log" "$PY" -u -m unittest -v test_decoder test_actions test_experiment
args=(--model "$MODEL" --baseline "$BASELINE" --phase2-input "$PKG/evidence_phase2fix"
      --output "$EXP/results" --device cuda --mode "$MODE")
if [[ -n "$PREVIOUS" ]]; then
    PREVIOUS=$(cd -- "$PREVIOUS" && pwd)
    case "$MODE" in
        main) args+=(--resume-from "$PREVIOUS/results") ;;
        replicate) args+=(--prior-run "$PREVIOUS/results") ;;
        smoke) printf 'Smoke mode does not accept PREVIOUS_RUN.\n' >&2; exit 2 ;;
    esac
fi
printf 'PHASE=paired_actions MODE=%s\n' "$MODE"
execute_logged "$EXP/experiment.log" "$PY" -u paired_experiment.py "${args[@]}"
