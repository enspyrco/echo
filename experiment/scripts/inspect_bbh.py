#!/usr/bin/env python3
"""Print sample BBH tasks — sanity-check loader without calling Claude."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow imports from experiment/ root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.bbh import PILOT_SUBTASKS, load_bbh, score_bbh


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect BBH tasks loaded from Hugging Face.")
    parser.add_argument("--subtasks", type=str, default=",".join(PILOT_SUBTASKS))
    parser.add_argument("--n", type=int, default=2, help="Examples per subtask")
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    names = [s.strip() for s in args.subtasks.split(",") if s.strip()]
    tasks = load_bbh(names, n_per_subtask=args.n, start=args.start)
    print(f"Loaded {len(tasks)} tasks from {len(names)} subtask(s)\n")

    for t in tasks:
        print("=" * 60)
        print(t["task_id"], "| gold:", t["gold"])
        print("-" * 60)
        print(t["prompt"][:800])
        if len(t["prompt"]) > 800:
            print("... [truncated]")
        # Sanity: gold letter should score as passed.
        ok, detail = score_bbh(f"Answer: {t['gold']}", t)
        print(f"\nscore_bbh(Answer: {t['gold']}): {ok} ({detail})\n")


if __name__ == "__main__":
    main()
