#!/usr/bin/env python3
"""BBH sweep — BBH personas, judge, oracle; same arm names as HumanEval.

Resilience (added 2026-06-25): results stream to disk one row at a time with a
flush after each, so a process death mid-sweep (usage exhaustion, kill, sleep)
preserves everything completed so far instead of losing the in-memory list.
Genuine Max-usage exhaustion aborts loudly rather than being recorded as a pile
of ``passed=False`` rows (which would silently deflate pass rates — a usage
failure is not a wrong answer). ``--resume <jsonl>`` continues an interrupted
run, skipping ``(task_id, arm)`` pairs already present in the file.
"""

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

# Substrings (case-insensitive) in a failed-call's error that mean the Max
# subscription is exhausted — a stop-and-save signal, NOT a per-task wrong
# answer. Deliberately excludes transient signals (overloaded/529/timeout):
# those stay swallowed as single-task failures so one blip doesn't abort a run.
_USAGE_EXHAUSTION_SIGNATURES = (
    "usage limit",
    "limit reached",
    "limit will reset",
    "rate limit",
    "rate_limit",
    "quota",
    "credit balance",
    "out of credits",
    "no credits",
    "insufficient",
    "429",
)


class UsageLimitReached(RuntimeError):
    """Raised when a model call fails due to Max-subscription exhaustion."""


def _is_usage_exhaustion(text: str) -> bool:
    low = text.lower()
    return any(sig in low for sig in _USAGE_EXHAUSTION_SIGNATURES)


def run_one_bbh(task: dict, arm_name: str, arm_fn) -> TaskResult:
    import time

    t0 = time.perf_counter()
    try:
        output, sub_calls = arm_fn(task)
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)[:200]}"
        # Usage exhaustion is account-level, not task-level: every subsequent
        # call would fail the same way. Abort the sweep loudly so we don't
        # pollute the dataset with hundreds of failure rows that look like
        # wrong answers. Transient errors fall through and are recorded.
        if _is_usage_exhaustion(str(e)):
            raise UsageLimitReached(msg) from e
        return TaskResult(
            task["task_id"], arm_name, False, msg, time.perf_counter() - t0, 0,
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


def _load_prior(path: Path) -> tuple[list[TaskResult], set[tuple[str, str]]]:
    """Read an existing results file: return (prior TaskResults, done pairs)."""
    prior: list[TaskResult] = []
    done: set[tuple[str, str]] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("_meta"):
                continue
            tr = TaskResult(
                row["task_id"], row["arm"], row["passed"],
                row["detail"], row["wall_seconds"], row["sub_calls"],
            )
            prior.append(tr)
            done.add((tr.task_id, tr.arm))
    return prior, done


def main() -> None:
    default_subtasks = ",".join(PILOT_SUBTASKS)
    default_arms = ",".join(BBH_ARMS)

    parser = argparse.ArgumentParser(description="Echo BBH pilot sweep.")
    parser.add_argument("--subtasks", type=str, default=default_subtasks)
    parser.add_argument("--n-per-subtask", type=int, default=5)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--arms", type=str, default=default_arms)
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to an existing results .jsonl to continue; completed "
             "(task_id, arm) pairs are skipped and new rows appended.",
    )
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

    # Resolve output path + prior state.
    results: list[TaskResult] = []
    done: set[tuple[str, str]] = set()
    RESULTS_DIR.mkdir(exist_ok=True)
    if args.resume:
        out_path = Path(args.resume)
        if not out_path.exists():
            print(f"--resume target does not exist: {out_path}", file=sys.stderr)
            sys.exit(1)
        results, done = _load_prior(out_path)
        print(f"Resuming {out_path}: {len(done)} pairs already done.")
        out_f = out_path.open("a")
        new_file = False
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = RESULTS_DIR / f"{ts}_bbh_n{len(tasks)}.jsonl"
        out_f = out_path.open("w")
        new_file = True

    total = len(tasks) * len(selected_arms)
    print(f"BBH sweep: {len(tasks)} tasks × {len(selected_arms)} arms = "
          f"{total} runs ({len(done)} skipped via resume)\n")

    def _write(row: dict) -> None:
        out_f.write(json.dumps(row) + "\n")
        out_f.flush()  # survive process death (OS page cache outlives the proc)

    try:
        if new_file:
            _write({
                "_meta": True,
                "benchmark": BENCHMARK,
                "subtasks": subtasks,
                "n_per_subtask": args.n_per_subtask,
                "start": args.start,
                "arms": list(selected_arms.keys()),
            })

        for task in tasks:
            printed_header = False
            for arm_name, arm_fn in selected_arms.items():
                if (task["task_id"], arm_name) in done:
                    continue
                if not printed_header:
                    print(f"=== {task['task_id']} (gold={task['gold']}) ===")
                    printed_header = True
                try:
                    r = run_one_bbh(task, arm_name, arm_fn)
                except UsageLimitReached as e:
                    out_f.flush()
                    out_f.close()
                    print(f"\n!!! Max usage exhausted on {task['task_id']} / "
                          f"{arm_name}: {e}", file=sys.stderr)
                    print(f"Partial results saved to {out_path} "
                          f"({len(results)} rows completed).", file=sys.stderr)
                    print(f"Resume when usage resets with:\n"
                          f"  python scripts/run_bbh_pilot.py --subtasks {args.subtasks} "
                          f"--n-per-subtask {args.n_per_subtask} --arms {args.arms} "
                          f"--resume {out_path}", file=sys.stderr)
                    sys.exit(2)
                results.append(r)
                row = asdict(r)
                row["benchmark"] = BENCHMARK
                _write(row)
                marker = "✓" if r.passed else "✗"
                print(f"  {marker} {arm_name:<18} {r.wall_seconds:>5.1f}s  "
                      f"{r.sub_calls} calls  {r.detail[:50]}")
    finally:
        if not out_f.closed:
            out_f.close()

    print("\n=== aggregate ===")
    print(json.dumps(summarize(results), indent=2))
    print(f"\nResults written to {out_path}")
    print(f"Analyze: python scripts/analyze_sweep.py {out_path}")


if __name__ == "__main__":
    main()
