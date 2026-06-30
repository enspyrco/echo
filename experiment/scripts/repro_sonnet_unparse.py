"""Reproduce the Sonnet unparseable BBH outputs (#1305) and dump raw text.

Loads the specific tasks that failed in the n=30 sweep, runs sonnet-only,
and prints the raw model output + what extract_choice / score_bbh return.
We capture WHY it's unparseable before touching the parser.
"""

from __future__ import annotations

import sys
import time

from benchmarks.bbh import extract_choice, load_bbh, score_bbh
from benchmarks.bbh_arms import arm_sonnet_only

# (subtask, index) pairs that came back unparseable from Sonnet in the n=30/n=99 sweeps.
TARGETS = [
    ("causal_judgement", 3),
    ("causal_judgement", 7),
    ("date_understanding", 5),
]


def main() -> None:
    # Load enough of each subtask to cover the target indices.
    subtasks = sorted({s for s, _ in TARGETS})
    tasks = load_bbh(subtasks, n_per_subtask=10, start=0)
    by_id = {t["task_id"]: t for t in tasks}

    for subtask, idx in TARGETS:
        tid = f"bbh/{subtask}/{idx}"
        task = by_id.get(tid)
        if task is None:
            print(f"!! {tid} not loaded", flush=True)
            continue
        print("=" * 80, flush=True)
        print(f"TASK {tid}  gold={task['gold']}  labels={task.get('choice_labels')}", flush=True)
        t0 = time.time()
        output, sub_calls = arm_sonnet_only(task)
        wall = time.time() - t0
        pred = extract_choice(output)
        passed, detail = score_bbh(output, task)
        print(f"wall={wall:.1f}s  extract_choice={pred!r}  score={passed} ({detail})", flush=True)
        print(f"--- raw output ({len(output)} chars) ---", flush=True)
        print(output, flush=True)
        print(f"--- last 300 chars ---\n{output[-300:]!r}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    sys.exit(main())
