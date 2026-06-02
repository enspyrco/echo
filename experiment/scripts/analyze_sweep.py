#!/usr/bin/env python3
"""Summarize an Echo sweep JSONL — pass rate, cost units, oracle alignment."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Haiku=1, Sonnet=3 per model call (Echo convention from results README).
HAIKU_UNIT = 1
SONNET_UNIT = 3


def escalated(arm: str, sub_calls: int) -> bool:
    if arm == "echo-judge":
        return sub_calls > 3
    if arm.startswith("echo-"):
        return sub_calls > 2
    return False


def cost_units(arm: str, sub_calls: int) -> int:
    if arm == "haiku-only":
        return sub_calls * HAIKU_UNIT
    if arm == "sonnet-only":
        return sub_calls * SONNET_UNIT
    if arm == "echo-judge":
        return 3 * HAIKU_UNIT + (SONNET_UNIT if sub_calls > 3 else 0)
    if arm.startswith("echo-"):
        base = 2 * HAIKU_UNIT
        return base + (SONNET_UNIT if sub_calls > 2 else 0)
    return sub_calls * HAIKU_UNIT


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

    print(f"=== {args.jsonl.name} ({len(rows)} rows) ===\n")
    print(f"{'arm':<20} {'pass':>10} {'esc%':>8} {'cost':>8} {'align':>8}")
    print("-" * 58)

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
            f"{arm:<20} {passed}/{n:<7} {100 * esc / n:>6.1f}% "
            f"{total_cost:>8} {align:>8}"
        )

    unparseable = sum(1 for r in rows if r.get("detail") == "unparseable")
    if unparseable:
        print(f"\nunparseable responses: {unparseable}")


if __name__ == "__main__":
    main()
