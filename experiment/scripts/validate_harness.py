#!/usr/bin/env python3
"""Pre-flight checks before a BBH sweep — no model API calls."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CANONICAL_ARMS = [
    "haiku-only",
    "sonnet-only",
    "echo-judge-openai",
    "echo-judge-openai-gpt-5.4-mini",
    "echo-judge-gemini-flash",
    "echo-oracle",
]


def check(label: str, ok: bool, detail: str = "", *, required: bool = True) -> bool:
    if ok:
        mark = "ok"
    elif required:
        mark = "FAIL"
    else:
        mark = "warn"
    suffix = f" — {detail}" if detail else ""
    print(f"  [{mark}] {label}{suffix}")
    return ok if required else True


def main() -> int:
    print("=== Echo harness pre-flight ===\n")

    ok = True

    print("Tools (required on server)")
    ok &= check("python", sys.version_info >= (3, 11), sys.version.split()[0])
    check("claude CLI", shutil.which("claude") is not None, required=False)
    if shutil.which("claude"):
        rc = subprocess.run(["claude", "auth", "status"], capture_output=True, text=True)
        check("claude auth", rc.returncode == 0, rc.stdout.strip() or rc.stderr.strip(), required=False)

    print("\nAPI keys (required on server for provider judges)")
    check("OPENAI_API_KEY", bool(os.environ.get("OPENAI_API_KEY")), "needed for echo-judge-openai*", required=False)
    check("GOOGLE_API_KEY", bool(os.environ.get("GOOGLE_API_KEY")), "needed for echo-judge-gemini*", required=False)

    print("\nImports")
    try:
        from benchmarks.bbh import PILOT_SUBTASKS, load_bbh  # noqa: F401
        from benchmarks.bbh_arms import BBH_ARMS  # noqa: F401
        from benchmarks.mmlu_pro import PILOT_CATEGORIES, load_mmlu_pro  # noqa: F401
        ok &= check("benchmark modules", True)
    except Exception as exc:
        ok &= check("benchmark modules", False, str(exc))

    print("\nCanonical BBH arms")
    from benchmarks.bbh_arms import BBH_ARMS

    for arm in CANONICAL_ARMS:
        ok &= check(arm, arm in BBH_ARMS)

    print("\nBBH loader (no API)")
    try:
        import datasets  # noqa: F401
        from benchmarks.bbh import load_bbh

        tasks = load_bbh(["logical_deduction_three_objects"], n_per_subtask=1)
        ok &= check("load 1 BBH task", len(tasks) == 1, tasks[0]["task_id"] if tasks else "")
    except ImportError:
        check("datasets package", False, "pip install datasets>=2.14", required=False)
    except Exception as exc:
        ok &= check("load 1 BBH task", False, str(exc))

    print("\nMMLU-Pro loader (no API)")
    try:
        from benchmarks.mmlu_pro import load_mmlu_pro

        tasks = load_mmlu_pro(["physics"], n_per_category=1)
        ok &= check("load 1 MMLU-Pro task", len(tasks) == 1, tasks[0]["task_id"] if tasks else "")
    except ImportError:
        check("datasets package", False, "pip install datasets>=2.14", required=False)
    except Exception as exc:
        ok &= check("load 1 MMLU-Pro task", False, str(exc))

    print("\nUnit tests")
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    ok &= check("unit tests", result.wasSuccessful(), f"{result.testsRun} tests")

    print()
    if ok:
        print("Harness checks passed (code + tests). Server still needs claude + API keys.")
        return 0
    print("Required harness checks failed. Fix before merging or running a sweep.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
