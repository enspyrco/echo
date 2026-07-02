#!/usr/bin/env python3
"""MMLU-Pro sweep — same Echo arms as BBH/HumanEval.

Results stream to disk with flush-after-each-row; ``--resume`` skips completed
(task_id, arm) pairs. Max usage exhaustion aborts with exit code 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.bbh_arms import BBH_ARMS
from benchmarks.mmlu_pro import PILOT_CATEGORIES, load_mmlu_pro, score_mmlu_pro
from run_pilot import RESULTS_DIR, TaskResult, summarize

BENCHMARK = "mmlu_pro"

# Reuse BBH arms — task dict shape is identical MCQ (prompt, gold, choice_labels).
MMLU_PRO_ARMS = BBH_ARMS

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


def run_one_mmlu_pro(task: dict, arm_name: str, arm_fn) -> TaskResult:
    import time

    t0 = time.perf_counter()
    try:
        output, sub_calls = arm_fn(task)
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)[:200]}"
        if _is_usage_exhaustion(str(e)):
            raise UsageLimitReached(msg) from e
        return TaskResult(
            task["task_id"], arm_name, False, msg, time.perf_counter() - t0, 0,
        )
    try:
        passed, detail = score_mmlu_pro(output, task)
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
    default_categories = ",".join(PILOT_CATEGORIES)
    default_arms = ",".join(
        ["haiku-only", "sonnet-only", "echo-judge", "echo-oracle"]
    )

    parser = argparse.ArgumentParser(description="Echo MMLU-Pro pilot sweep.")
    parser.add_argument("--categories", type=str, default=default_categories)
    parser.add_argument("--split", type=str, default="test", choices=("test", "validation"))
    parser.add_argument("--n-per-category", type=int, default=5)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--arms", type=str, default=default_arms)
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to an existing results .jsonl to continue.",
    )
    args = parser.parse_args()

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    selected_arms = {n: MMLU_PRO_ARMS[n] for n in args.arms.split(",") if n in MMLU_PRO_ARMS}
    unknown = [n for n in args.arms.split(",") if n.strip() and n.strip() not in MMLU_PRO_ARMS]
    if unknown:
        print(f"Unknown arms (skipped): {unknown}", file=sys.stderr)
    if not selected_arms:
        print("No valid arms selected.", file=sys.stderr)
        sys.exit(1)

    tasks = load_mmlu_pro(
        categories,
        split=args.split,
        n_per_category=args.n_per_category,
        start=args.start,
    )

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
        out_path = RESULTS_DIR / f"{ts}_mmlu_pro_n{len(tasks)}.jsonl"
        out_f = out_path.open("w")
        new_file = True

    total = len(tasks) * len(selected_arms)
    print(f"MMLU-Pro sweep: {len(tasks)} tasks × {len(selected_arms)} arms = "
          f"{total} runs ({len(done)} skipped via resume)\n")

    def _write(row: dict) -> None:
        out_f.write(json.dumps(row) + "\n")
        out_f.flush()

    try:
        if new_file:
            _write({
                "_meta": True,
                "benchmark": BENCHMARK,
                "categories": categories,
                "split": args.split,
                "n_per_category": args.n_per_category,
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
                    r = run_one_mmlu_pro(task, arm_name, arm_fn)
                except UsageLimitReached as e:
                    out_f.flush()
                    out_f.close()
                    print(f"\n!!! Max usage exhausted on {task['task_id']} / "
                          f"{arm_name}: {e}", file=sys.stderr)
                    print(f"Partial results saved to {out_path} "
                          f"({len(results)} rows completed).", file=sys.stderr)
                    print(f"Resume when usage resets with:\n"
                          f"  python scripts/run_mmlu_pro_pilot.py "
                          f"--categories {args.categories} "
                          f"--n-per-category {args.n_per_category} --arms {args.arms} "
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
