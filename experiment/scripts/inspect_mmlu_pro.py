#!/usr/bin/env python3
"""Print sample MMLU-Pro tasks — sanity-check loader without calling Claude."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.mmlu_pro import PILOT_CATEGORIES, load_mmlu_pro, score_mmlu_pro


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect MMLU-Pro tasks loaded from Hugging Face.",
    )
    parser.add_argument("--categories", type=str, default=",".join(PILOT_CATEGORIES))
    parser.add_argument("--split", type=str, default="test", choices=("test", "validation"))
    parser.add_argument("--n", type=int, default=2, help="Examples per category")
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    names = [c.strip() for c in args.categories.split(",") if c.strip()]
    tasks = load_mmlu_pro(names, split=args.split, n_per_category=args.n, start=args.start)
    print(f"Loaded {len(tasks)} tasks from {len(names)} category(ies)\n")

    for t in tasks:
        print("=" * 60)
        print(t["task_id"], "| gold:", t["gold"], "| options:", len(t["choice_labels"]))
        print("-" * 60)
        print(t["prompt"][:900])
        if len(t["prompt"]) > 900:
            print("... [truncated]")
        ok, detail = score_mmlu_pro(f"Answer: {t['gold']}", t)
        print(f"\nscore_mmlu_pro(Answer: {t['gold']}): {ok} ({detail})\n")


if __name__ == "__main__":
    main()
