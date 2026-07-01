#!/usr/bin/env bash
# Unattended auto-resume wrapper around run_bbh_pilot.py.
#
# run_bbh_pilot.py exits 2 when the Max subscription is exhausted, having
# flushed every completed row to disk and printed the partial-results path.
# This wrapper catches that exit, waits for the usage window to recover, and
# re-invokes with --resume so the sweep finishes without a human in the loop.
#
# Usage:
#   scripts/run_bbh_resumable.sh --subtasks ... --n-per-subtask 33 --arms ...
#
# Env knobs:
#   BACKOFF_SECONDS  seconds to wait after an exhaustion abort (default 1800 = 30m)
#   MAX_RETRIES      max resume attempts before giving up (default 12 ≈ 6h)
#
# Exit codes: 0 = sweep complete; 1 = hard error / retries exhausted.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1   # experiment/ dir

BACKOFF_SECONDS="${BACKOFF_SECONDS:-1800}"
MAX_RETRIES="${MAX_RETRIES:-12}"
PY=(python scripts/run_bbh_pilot.py "$@")

log() { printf '[resumable %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

# Pull the results path from a run's output (both the success line
# "Results written to <path>" and the abort line "Partial results saved to
# <path>" contain it). Last match wins.
extract_path() {
  grep -oE 'results/[0-9TZ_]+_bbh_n[0-9]+\.jsonl' "$1" | tail -1
}

tmplog="$(mktemp)"
trap 'rm -f "$tmplog"' EXIT

attempt=0
resume_path=""
while :; do
  if [ -z "$resume_path" ]; then
    log "starting sweep: ${PY[*]}"
    "${PY[@]}" 2>&1 | tee "$tmplog"
  else
    log "resuming $resume_path (attempt $attempt/$MAX_RETRIES)"
    "${PY[@]}" --resume "$resume_path" 2>&1 | tee "$tmplog"
  fi
  rc=${PIPESTATUS[0]}

  if [ "$rc" -eq 0 ]; then
    log "sweep complete."
    exit 0
  elif [ "$rc" -eq 2 ]; then
    # Usage exhausted — capture the partial-results path and back off.
    p="$(extract_path "$tmplog")"
    [ -n "$p" ] && resume_path="$p"
    if [ -z "$resume_path" ]; then
      log "exhaustion abort but no results path found in output; aborting."
      exit 1
    fi
    attempt=$((attempt + 1))
    if [ "$attempt" -gt "$MAX_RETRIES" ]; then
      log "max retries ($MAX_RETRIES) reached; partial data at $resume_path."
      exit 1
    fi
    log "usage exhausted; partial data at $resume_path. Sleeping ${BACKOFF_SECONDS}s before resume."
    sleep "$BACKOFF_SECONDS"
  else
    log "hard error (exit $rc); not a usage-exhaustion abort. See output above."
    exit 1
  fi
done
