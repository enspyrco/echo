"""MMLU-Pro loader and scoring for Echo sweeps.

Data source: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
Uses the ``test`` split by default (12,032 questions, 14 categories, up to 10 options).
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from benchmarks.bbh import (
    _CHOICE_LETTERS,
    _clean_label,
    extract_choice,
    extract_choice_text,
    format_prompt,
    score_bbh,
)

MMLU_PRO_DATASET = "TIGER-Lab/MMLU-Pro"

# Pilot slice — harder domains where Haiku/Sonnet may diverge.
PILOT_CATEGORIES = [
    "physics",
    "math",
    "law",
    "chemistry",
    "philosophy",
]

ALL_CATEGORIES = [
    "biology",
    "business",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
]


def category_slug(category: str) -> str:
    return re.sub(r"\s+", "_", category.strip().lower())


def options_to_choices(options: list[str]) -> dict[str, list[str]]:
    labels = list(_CHOICE_LETTERS[: len(options)])
    return {"label": labels, "text": list(options)}


def normalize_gold(answer: str, choices: dict[str, Any]) -> str:
    letter = _clean_label(answer)
    labels = [_clean_label(label) for label in choices.get("label") or []]
    if letter not in labels:
        raise ValueError(f"Gold {answer!r} not in choice labels {labels}")
    return letter


def score_mmlu_pro(model_output: str, task: dict) -> tuple[bool, str]:
    """Grade an MMLU-Pro response. Task shape matches BBH MCQ tasks."""
    return score_bbh(model_output, task)


def _row_to_task(category: str, row: dict) -> dict:
    options = list(row["options"])
    if not options:
        raise ValueError(f"MMLU-Pro question {row.get('question_id')} has no options")
    choices = options_to_choices(options)
    gold = normalize_gold(str(row["answer"]), choices)
    return {
        "task_id": f"mmlu_pro/{category_slug(category)}/{row['question_id']}",
        "prompt": format_prompt(row["question"], choices),
        "gold": gold,
        "choice_labels": [_clean_label(label) for label in choices["label"]],
        "benchmark": "mmlu_pro",
        "category": category,
        "question_id": row["question_id"],
        "question": row["question"],
        "choices": choices,
    }


def load_mmlu_pro(
    categories: list[str] | None = None,
    *,
    split: str = "test",
    n_per_category: int | None = None,
    start: int = 0,
    n: int | None = None,
) -> list[dict]:
    """Load MMLU-Pro tasks from Hugging Face.

    Args:
        categories: Category names to load (default: PILOT_CATEGORIES).
        split: HF split — ``test`` (eval) or ``validation`` (70 dev items).
        n_per_category: Cap examples per category (applied after ``start`` slice).
        start: Skip first ``start`` examples within each category.
        n: Cap total tasks across all categories (after per-category caps).
    """
    from datasets import load_dataset

    names = categories or PILOT_CATEGORIES
    unknown = [name for name in names if name not in ALL_CATEGORIES]
    if unknown:
        raise ValueError(f"Unknown MMLU-Pro categories: {unknown}")

    ds = load_dataset(MMLU_PRO_DATASET)[split]
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in ds:
        cat = row["category"]
        if cat in names:
            by_category[cat].append(row)

    tasks: list[dict] = []
    for cat in names:
        rows = sorted(by_category[cat], key=lambda r: int(r["question_id"]))
        rows = rows[start:]
        if n_per_category is not None:
            rows = rows[:n_per_category]
        for row in rows:
            tasks.append(_row_to_task(cat, row))
        if n is not None and len(tasks) >= n:
            return tasks[:n]
    if n is not None:
        return tasks[:n]
    return tasks
