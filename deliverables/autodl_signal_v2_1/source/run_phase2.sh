#!/usr/bin/env bash
# Execute with bash; do not source into an interactive terminal.
if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    printf 'Run this script with bash; do not source it.\n' >&2
    return 2
fi
set -euo pipefail
ROOT=${1:-/root/autodl-tmp}
LIMIT=${2:-3}
PKG=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PY="$ROOT/venvs/infty-official/bin/python"
MODEL="$ROOT/inftythink_day1/models/InftyThink-1.5B"
NATIVE="$ROOT/inftythink_official_check/outputs/native5"
# Third argument is used by launch_phase2.sh, which creates a fresh directory.
EXP=${3:-$(mktemp -d "$ROOT/infty_signal_phase2fix_XXXXXXXX")}
EXP=$(cd -- "$EXP" && pwd)
shopt -s nullglob dotglob
for existing in "$EXP"/*; do
    case $(basename -- "$existing") in
        launcher.log|launcher.pid) ;;
        *) printf 'Refusing to reuse a nonempty result directory: %s\n' "$EXP" >&2; exit 2 ;;
    esac
done
shopt -u nullglob dotglob
mkdir "$EXP/.runner_claim"  # An existing run cannot be executed a second time.
printf '%s\n' "$$" > "$EXP/runner.pid"
awk '{print $22}' "/proc/$$/stat" > "$EXP/runner.start_ticks"
printf 'OUTPUT_DIRECTORY=%s\n' "$EXP"
PACK_PY="$PY"
if [[ ! -x "$PACK_PY" ]]; then PACK_PY=$(command -v python3 || true); fi

finish() {
    local task_exit=$?
    trap - EXIT INT TERM
    printf '%s\n' "$task_exit" > "$EXP/exit_status.txt"
    if [[ -n "$PACK_PY" ]]; then
        # Use stdlib only. The helper can also repackage a run after SIGKILL/reboot.
        "$PACK_PY" -u - "$PKG" "$EXP" <<'PY' || printf 'ARCHIVE_FAILED: run status_phase2.py RUN_DIRECTORY --pack\n' >&2
import pathlib, sys
sys.path.insert(0, sys.argv[1])
from status_phase2 import pack_result
print('RETURN_ZIP=' + str(pack_result(pathlib.Path(sys.argv[2]))), flush=True)
PY
    else
        printf 'ARCHIVE_FAILED: no Python interpreter available; result files remain in %s\n' "$EXP" >&2
    fi
    printf 'EXPERIMENT_EXIT_STATUS=%s\n' "$task_exit"
    exit "$task_exit"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$PKG"
sha256sum -c checksums.sha256 2>&1 | tee "$EXP/package_checksums.log"
test -x "$PY"
mkdir "$EXP/code"
cp -- prepare_plan.py collect_phase2.py v1_core.py test_prepare_plan.py test_collect_phase2.py \
    run_phase2.sh launch_phase2.sh status_phase2.py checksums.sha256 "$EXP/code/"
printf 'PHASE=cpu_tests\n'
"$PY" -u test_prepare_plan.py 2>&1 | tee "$EXP/plan_tests.log"
"$PY" -u test_collect_phase2.py 2>&1 | tee "$EXP/collector_tests.log"
printf 'PHASE=prepare_frozen_plan\n'
"$PY" -u prepare_plan.py --model "$MODEL" --baseline "$NATIVE" \
    --limit "$LIMIT" --targets 1024,2048,4096 --boundary-window 128 \
    --output "$EXP/plan.json" 2>&1 | tee "$EXP/plan.log"
printf 'PHASE=float32_canary_then_nine_positions\n'
"$PY" -u collect_phase2.py --model "$MODEL" --baseline "$NATIVE" \
    --plan "$EXP/plan.json" --output "$EXP/signals" --device cuda --dtype float32 \
    --probe-tokens 128 --prefill-chunk 128 --canary-chunk 64 \
    2>&1 | tee "$EXP/signals.log"
