"""Unit tests for BBH answer parsing and scoring (no model calls)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.bbh import extract_choice, format_prompt, normalize_gold, score_bbh


class TestExtractChoice(unittest.TestCase):
    def test_answer_line(self) -> None:
        self.assertEqual(extract_choice("Reasoning here.\nAnswer: C"), "C")

    def test_answer_with_parens(self) -> None:
        self.assertEqual(extract_choice("Answer: (B)"), "B")

    def test_parenthetical(self) -> None:
        self.assertEqual(extract_choice("Therefore (A) is correct."), "A")

    def test_gold_target(self) -> None:
        self.assertEqual(normalize_gold("A"), "A")

    def test_unparseable(self) -> None:
        self.assertIsNone(extract_choice("I have no idea"))

    def test_last_line_letter(self) -> None:
        self.assertEqual(
            extract_choice("Step by step...\nThe best option is D"),
            "D",
        )


class TestScoreBbh(unittest.TestCase):
    def _task(self, gold: str = "C") -> dict:
        return {
            "task_id": "bbh/test/0",
            "prompt": "dummy",
            "gold": gold,
        }

    def test_pass(self) -> None:
        ok, detail = score_bbh("Answer: C", self._task())
        self.assertTrue(ok)
        self.assertEqual(detail, "passed")

    def test_wrong(self) -> None:
        ok, detail = score_bbh("Answer: A", self._task("C"))
        self.assertFalse(ok)
        self.assertIn("expected C", detail)

    def test_unparseable(self) -> None:
        ok, detail = score_bbh("maybe yes", self._task())
        self.assertFalse(ok)
        self.assertEqual(detail, "unparseable")


class TestFormatPrompt(unittest.TestCase):
    def test_includes_choices(self) -> None:
        p = format_prompt(
            "Who is oldest?",
            {"label": ["A)", "B)"], "text": ["Ann", "Bob"]},
        )
        self.assertIn("Who is oldest?", p)
        self.assertIn("A)", p)
        self.assertIn("Ann", p)
        self.assertIn("Answer: X", p)


if __name__ == "__main__":
    unittest.main()
