#!/usr/bin/env python3
"""Summarize an Echo sweep JSONL — pass rate, cost units, oracle alignment."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cost_units import cost_units, escalated, judge_units_for_arm


def load_records(path: Path) -> tuple[dict | None, list[dict]]:
    meta = None
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("_meta"):
            meta = obj
            continue
        rows.append(obj)
    return meta, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Echo sweep JSONL.")
    parser.add_argument("jsonl", type=Path, help="Path to results/*.jsonl")
    args = parser.parse_args()

    meta, rows = load_records(args.jsonl)
    if not rows:
        print("No result rows found.", file=sys.stderr)
        sys.exit(1)

    if meta:
        print("=== sweep meta ===")
        print(json.dumps({k: v for k, v in meta.items() if k != "_meta"}, indent=2))
        print()

    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)

    oracle_rows = by_arm.get("echo-oracle", [])
    oracle_esc = {r["task_id"]: escalated("echo-oracle", r["sub_calls"]) for r in oracle_rows}

    print(f"=== {args.jsonl.name} ({len(rows)} rows) ===")
    print("Cost units: Haiku persona = 1.0; includes judge API pricing (see cost_units.py)\n")
    print(f"{'arm':<32} {'pass':>10} {'esc%':>8} {'cost':>8} {'align':>8}")
    print("-" * 70)

    for arm in sorted(by_arm):
        rs = by_arm[arm]
        n = len(rs)
        passed = sum(1 for r in rs if r.get("passed"))
        esc = sum(1 for r in rs if escalated(arm, r["sub_calls"]))
        total_cost = sum(cost_units(arm, r["sub_calls"]) for r in rs)

        align = ""
        if oracle_esc and arm != "echo-oracle" and arm.startswith("echo-"):
            same = 0
            for tid, oesc in oracle_esc.items():
                r = next((x for x in rs if x["task_id"] == tid), None)
                if r is None:
                    continue
                if escalated(arm, r["sub_calls"]) == oesc:
                    same += 1
            align = f"{100 * same / len(oracle_esc):.0f}%"

        print(
            f"{arm:<32} {passed}/{n:<7} {100 * esc / n:>6.1f}% "
            f"{total_cost:>8.1f} {align:>8}"
        )

    unparseable = sum(1 for r in rows if r.get("detail") == "unparseable")
    if unparseable:
        print(f"\nunparseable responses: {unparseable}")


if __name__ == "__main__":
    main()
