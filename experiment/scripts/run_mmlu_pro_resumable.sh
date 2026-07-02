#!/usr/bin/env bash
# Unattended auto-resume wrapper around run_mmlu_pro_pilot.py.
#
# Usage:
#   scripts/run_mmlu_pro_resumable.sh --categories physics,math --n-per-category 25
#
# Env knobs: BACKOFF_SECONDS (default 1800), MAX_RETRIES (default 12)
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

BACKOFF_SECONDS="${BACKOFF_SECONDS:-1800}"
MAX_RETRIES="${MAX_RETRIES:-12}"
PY=(python scripts/run_mmlu_pro_pilot.py "$@")

log() { printf '[resumable %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

extract_path() {
  grep -oE 'results/[0-9TZ_]+_mmlu_pro_n[0-9]+\.jsonl' "$1" | tail -1
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
