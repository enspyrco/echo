"""HumanEval dataset loader.

Downloads the official HumanEval JSONL once (~30KB gzipped) and caches it
under ~/.cache/echo/. Returns task dicts with the four fields run_pilot.py
needs: task_id, prompt (signature + docstring), test (canonical test code),
entry_point (the function name).
"""

from __future__ import annotations

import gzip
import json
import urllib.request
from pathlib import Path

HUMANEVAL_URL = (
    "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
)
CACHE_PATH = Path.home() / ".cache" / "echo" / "HumanEval.jsonl"


def _download_if_needed() -> None:
    """Idempotently fetch the HumanEval JSONL and stash it under the cache path.
    The 30 KB gzipped payload decompresses to ~140 KB of JSONL.
    """
    if CACHE_PATH.exists():
        return
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(HUMANEVAL_URL) as resp:  # noqa: S310 — well-known URL
        gz = resp.read()
    CACHE_PATH.write_text(gzip.decompress(gz).decode("utf-8"))


def load_humaneval(n: int | None = None) -> list[dict]:
    """Return the first ``n`` tasks from HumanEval (164 total), or all if n is None.

    Each task is a dict with keys:
      task_id      e.g. "HumanEval/0"
      prompt       function signature + docstring (the model fills in the body)
      canonical_solution  ground-truth implementation (not given to the model)
      test         python source containing a ``check(candidate)`` function
      entry_point  the name of the function to test (e.g. "has_close_elements")
    """
    _download_if_needed()
    tasks: list[dict] = []
    for line in CACHE_PATH.read_text().splitlines():
        line = line.strip()
        if line:
            tasks.append(json.loads(line))
    return tasks[:n] if n is not None else tasks
