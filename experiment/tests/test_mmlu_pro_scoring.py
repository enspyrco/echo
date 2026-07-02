"""Unit tests for MMLU-Pro loading and scoring (no model calls)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.mmlu_pro import (
    category_slug,
    load_mmlu_pro,
    normalize_gold,
    options_to_choices,
    score_mmlu_pro,
    _row_to_task,
)


class TestOptionsToChoices(unittest.TestCase):
    def test_ten_options(self) -> None:
        choices = options_to_choices(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"])
        self.assertEqual(choices["label"], list("ABCDEFGHIJ"))

    def test_four_options(self) -> None:
        choices = options_to_choices(["one", "two", "three", "four"])
        self.assertEqual(choices["label"], ["A", "B", "C", "D"])


class TestNormalizeGold(unittest.TestCase):
    def test_letter_gold(self) -> None:
        choices = options_to_choices(["x", "y", "z"])
        self.assertEqual(normalize_gold("B", choices), "B")


class TestScoreMmluPro(unittest.TestCase):
    def _task(self, gold: str = "C", n_options: int = 10) -> dict:
        choices = options_to_choices([f"opt{i}" for i in range(n_options)])
        return _row_to_task(
            "physics",
            {
                "question_id": 42,
                "question": "Sample question?",
                "options": choices["text"],
                "answer": gold,
                "answer_index": choices["label"].index(gold),
            },
        )

    def test_pass(self) -> None:
        task = self._task("I", n_options=9)
        ok, detail = score_mmlu_pro("Reasoning...\nAnswer: I", task)
        self.assertTrue(ok)
        self.assertEqual(detail, "passed")

    def test_wrong(self) -> None:
        task = self._task("C")
        ok, detail = score_mmlu_pro("Answer: A", task)
        self.assertFalse(ok)
        self.assertIn("expected C", detail)

    def test_invalid_letter_for_task(self) -> None:
        task = self._task("D", n_options=4)
        ok, detail = score_mmlu_pro("Answer: J", task)
        self.assertFalse(ok)
        self.assertEqual(detail, "unparseable")


class TestCategorySlug(unittest.TestCase):
    def test_spaces(self) -> None:
        self.assertEqual(category_slug("computer science"), "computer_science")


class TestLoadMmluPro(unittest.TestCase):
    def test_load_one_per_category(self) -> None:
        fake_rows = [
            {
                "question_id": 10,
                "question": "Q1?",
                "options": ["a", "b", "c", "d"],
                "answer": "B",
                "answer_index": 1,
                "category": "physics",
            },
            {
                "question_id": 5,
                "question": "Q2?",
                "options": ["a", "b", "c", "d", "e"],
                "answer": "E",
                "answer_index": 4,
                "category": "math",
            },
        ]

        class FakeSplit:
            def __iter__(self):
                return iter(fake_rows)

        class FakeDataset:
            def __getitem__(self, key: str):
                return FakeSplit()

        with patch("datasets.load_dataset", return_value=FakeDataset()):
            tasks = load_mmlu_pro(["physics", "math"], n_per_category=1)

        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["task_id"], "mmlu_pro/physics/10")
        self.assertEqual(tasks[0]["gold"], "B")
        self.assertEqual(tasks[1]["task_id"], "mmlu_pro/math/5")
        self.assertEqual(tasks[1]["gold"], "E")


if __name__ == "__main__":
    unittest.main()
