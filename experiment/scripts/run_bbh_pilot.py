#!/usr/bin/env python3
"""BBH sweep — BBH personas, judge, oracle; same arm names as HumanEval."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.bbh import PILOT_SUBTASKS, load_bbh, score_bbh
from benchmarks.bbh_arms import BBH_ARMS
from run_pilot import RESULTS_DIR, TaskResult, summarize

BENCHMARK = "bbh"


def run_one_bbh(task: dict, arm_name: str, arm_fn) -> TaskResult:
    import time

    t0 = time.perf_counter()
    try:
        output, sub_calls = arm_fn(task)
    except Exception as e:
        return TaskResult(
            task["task_id"],
            arm_name,
            False,
            f"{type(e).__name__}: {str(e)[:200]}",
            time.perf_counter() - t0,
            0,
        )
    try:
        passed, detail = score_bbh(output, task)
    except Exception as e:
        return TaskResult(
            task["task_id"],
            arm_name,
            False,
            f"scorer {type(e).__name__}: {str(e)[:200]}",
            time.perf_counter() - t0,
            sub_calls,
        )
    return TaskResult(
        task["task_id"], arm_name, passed, detail,
        time.perf_counter() - t0, sub_calls,
    )


def main() -> None:
    default_subtasks = ",".join(PILOT_SUBTASKS)
    default_arms = ",".join(BBH_ARMS)

    parser = argparse.ArgumentParser(description="Echo BBH pilot sweep.")
    parser.add_argument("--subtasks", type=str, default=default_subtasks)
    parser.add_argument("--n-per-subtask", type=int, default=5)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--arms", type=str, default=default_arms)
    args = parser.parse_args()

    subtasks = [s.strip() for s in args.subtasks.split(",") if s.strip()]
    selected_arms = {n: BBH_ARMS[n] for n in args.arms.split(",") if n in BBH_ARMS}
    unknown = [n for n in args.arms.split(",") if n.strip() and n.strip() not in BBH_ARMS]
    if unknown:
        print(f"Unknown arms (skipped): {unknown}", file=sys.stderr)
    if not selected_arms:
        print("No valid arms selected.", file=sys.stderr)
        sys.exit(1)

    tasks = load_bbh(subtasks, n_per_subtask=args.n_per_subtask, start=args.start)

    print(f"BBH sweep: {len(tasks)} tasks × {len(selected_arms)} arms = "
          f"{len(tasks) * len(selected_arms)} runs\n")

    results: list[TaskResult] = []
    for task in tasks:
        print(f"=== {task['task_id']} (gold={task['gold']}) ===")
        for arm_name, arm_fn in selected_arms.items():
            r = run_one_bbh(task, arm_name, arm_fn)
            results.append(r)
            marker = "✓" if r.passed else "✗"
            print(f"  {marker} {arm_name:<18} {r.wall_seconds:>5.1f}s  "
                  f"{r.sub_calls} calls  {r.detail[:50]}")

    print("\n=== aggregate ===")
    print(json.dumps(summarize(results), indent=2))

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"{ts}_bbh_n{len(tasks)}.jsonl"
    with out_path.open("w") as f:
        meta = {
            "_meta": True,
            "benchmark": BENCHMARK,
            "subtasks": subtasks,
            "n_per_subtask": args.n_per_subtask,
            "start": args.start,
            "arms": list(selected_arms.keys()),
        }
        f.write(json.dumps(meta) + "\n")
        for r in results:
            row = asdict(r)
            row["benchmark"] = BENCHMARK
            f.write(json.dumps(row) + "\n")
    print(f"\nResults written to {out_path}")
    print(f"Analyze: python scripts/analyze_sweep.py {out_path}")


if __name__ == "__main__":
    main()
