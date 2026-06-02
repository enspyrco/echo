"""Big-Bench Hard (BBH) loader and scoring for Echo sweeps.

Data source: https://huggingface.co/datasets/Joschka/big_bench_hard
Each subtask is a separate config; the eval split name matches the config name.
"""

from __future__ import annotations

import re
from typing import Any

BBH_DATASET = "Joschka/big_bench_hard"

# Pilot subset — expand after team confirms scope.
PILOT_SUBTASKS = [
    "logical_deduction_three_objects",
    "causal_judgement",
    "date_understanding",
]

# All BBH subtasks on Joschka/big_bench_hard (excluding few_shot_prompts).
ALL_SUBTASKS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]

_CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _format_choices(choices: dict[str, Any]) -> str:
    """Render BBH choices dict into prompt text."""
    labels = choices.get("label") or []
    texts = choices.get("text") or []
    lines: list[str] = []
    for i, text in enumerate(texts):
        if i < len(labels):
            label = str(labels[i]).strip()
            if not label.endswith(")"):
                label = f"{label})"
        else:
            letter = _CHOICE_LETTERS[i] if i < len(_CHOICE_LETTERS) else str(i)
            label = f"({letter})"
        lines.append(f"{label} {text}")
    return "\n".join(lines)


def format_prompt(question: str, choices: dict[str, Any]) -> str:
    """Build the model-facing prompt for a BBH multiple-choice item."""
    return (
        f"{question.strip()}\n\n"
        f"Choices:\n{_format_choices(choices)}\n\n"
        "Reply with your reasoning, then end with exactly one line:\n"
        "Answer: X\n"
        "where X is the letter of the correct choice (A, B, C, ...)."
    )


def normalize_gold(target: str) -> str:
    """BBH targets are usually a single letter; normalize to uppercase."""
    letter = extract_choice(str(target))
    if letter is None:
        raise ValueError(f"Could not parse gold target: {target!r}")
    return letter


def extract_choice(text: str) -> str | None:
    """Parse a multiple-choice letter from model output.

    Tries explicit patterns first, then the last standalone A–Z near the end.
    Returns uppercase A–Z or None if unparseable.
    """
    if not text or not text.strip():
        return None

    patterns = [
        r"(?im)^\s*answer\s*:\s*\(?\s*([A-Z])\s*\)?\s*\.?\s*$",
        r"(?im)^\s*answer\s*:\s*\(?\s*([A-Z])\s*\)?",
        r"(?im)^\s*\(?\s*([A-Z])\s*\)\s*$",
        r"(?im)correct\s+(?:answer\s+is|choice\s+is)\s*\(?\s*([A-Z])\s*\)?",
        r"\(\s*([A-Z])\s*\)",
    ]
    for pat in patterns:
        matches = re.findall(pat, text)
        if matches:
            return matches[-1].upper()

    tail = "\n".join(text.strip().splitlines()[-5:])
    for pat in (
        r"(?i)\b(?:option|choice|answer)\s+is\s+\(?\s*([A-Z])\s*\)?",
        r"(?i)\b([A-Z])\s*\.?\s*$",
    ):
        m = re.search(pat, tail.strip())
        if m:
            return m.group(1).upper()
    return None


def score_bbh(model_output: str, task: dict) -> tuple[bool, str]:
    """Grade a BBH response against task['gold']."""
    pred = extract_choice(model_output)
    if pred is None:
        return False, "unparseable"
    gold = task["gold"]
    if pred == gold:
        return True, "passed"
    return False, f"expected {gold} got {pred}"


def _row_to_task(subtask: str, index: int, row: dict) -> dict:
    gold = normalize_gold(row["target"])
    return {
        "task_id": f"bbh/{subtask}/{index}",
        "prompt": format_prompt(row["question"], row["choices"]),
        "gold": gold,
        "benchmark": "bbh",
        "subtask": subtask,
        # Echo arms use task["prompt"]; keep raw fields for debugging.
        "question": row["question"],
        "choices": row["choices"],
    }


def load_bbh(
    subtasks: list[str] | None = None,
    *,
    n_per_subtask: int | None = None,
    start: int = 0,
    n: int | None = None,
) -> list[dict]:
    """Load BBH tasks from Hugging Face.

    Args:
        subtasks: Config names to load (default: PILOT_SUBTASKS).
        n_per_subtask: Cap examples per subtask (applied after ``start`` slice).
        start: Skip first ``start`` examples within each subtask.
        n: Cap total tasks across all subtasks (after per-subtask caps).
    """
    from datasets import load_dataset

    names = subtasks or PILOT_SUBTASKS
    tasks: list[dict] = []
    for name in names:
        ds = load_dataset(BBH_DATASET, name, split=name)
        rows = list(ds)[start:]
        if n_per_subtask is not None:
            rows = rows[:n_per_subtask]
        for i, row in enumerate(rows):
            tasks.append(_row_to_task(name, start + i, row))
        if n is not None and len(tasks) >= n:
            return tasks[:n]
    if n is not None:
        return tasks[:n]
    return tasks
