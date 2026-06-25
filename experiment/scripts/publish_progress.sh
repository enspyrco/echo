#!/usr/bin/env bash
# Live-publish loop: regenerate the BBH progress page from the streaming JSONL
# and force-push it to the gh-pages branch every INTERVAL seconds, until the
# tracked run PID exits. Operates entirely inside an isolated gh-pages worktree
# so the main working tree (where the run lives) is never touched.
#
# Usage: publish_progress.sh <run_pid> <results_glob_dir> <pages_worktree> <start_epoch>
set -uo pipefail

RUN_PID="$1"
RESULTS_DIR="$2"
PAGES_DIR="$3"
START_EPOCH="$4"
INTERVAL="${INTERVAL:-300}"  # GitHub Pages soft-limits ~10 builds/hr; 5min = 12/hr
REPO="/Users/nick/git/research/echo"
PY="$REPO/experiment/.venv/bin/python"
GEN="$REPO/experiment/scripts/gen_progress.py"

publish() {
  local f now
  f="$(ls -t "$RESULTS_DIR"/*_bbh_n99.jsonl 2>/dev/null | head -1)"
  [ -z "$f" ] && return 0
  now="$(date +%s)"
  "$PY" "$GEN" "$f" "$PAGES_DIR/index.html" "$START_EPOCH" "$now" >/dev/null 2>&1 || return 0
  touch "$PAGES_DIR/.nojekyll"
  ( cd "$PAGES_DIR" \
    && git add index.html .nojekyll \
    && git -c user.name="Nick Meinhold" -c user.email="langer.robin@gmail.com" \
         commit -q --amend --no-edit \
    && git push -q -f origin gh-pages ) >/dev/null 2>&1
}

# Update while the run is alive, then one final update to flip to COMPLETE.
while kill -0 "$RUN_PID" 2>/dev/null; do
  publish
  sleep "$INTERVAL"
done
publish
echo "publish loop done (run $RUN_PID exited)"
