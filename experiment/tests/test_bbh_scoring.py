"""Unit tests for BBH answer parsing and scoring (no model calls)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.bbh import (
    extract_answer,
    extract_choice,
    extract_yes_no,
    format_prompt,
    normalize_gold,
    score_bbh,
)


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

    def test_yes_no_prompt_when_no_choices(self) -> None:
        p = format_prompt("Did X cause Y?", None)
        self.assertIn("Did X cause Y?", p)
        self.assertIn("Answer: Yes", p)
        self.assertIn("Answer: No", p)
        self.assertNotIn("Answer: X", p)

    def test_yes_no_prompt_when_empty_choices(self) -> None:
        p = format_prompt("Did X cause Y?", {})
        self.assertIn("Answer: Yes", p)


class TestYesNoSupport(unittest.TestCase):
    def test_extract_yes_no_explicit(self) -> None:
        self.assertEqual(extract_yes_no("Reasoning...\nAnswer: Yes"), "YES")
        self.assertEqual(extract_yes_no("Reasoning...\nAnswer: No"), "NO")

    def test_extract_yes_no_phrase(self) -> None:
        self.assertEqual(extract_yes_no("So the answer is yes"), "YES")
        self.assertEqual(extract_yes_no("Therefore the answer is no"), "NO")

    def test_extract_yes_no_last_line(self) -> None:
        self.assertEqual(extract_yes_no("Long reasoning ...\n...\nYes."), "YES")

    def test_extract_yes_no_unparseable(self) -> None:
        self.assertIsNone(extract_yes_no("Maybe, but possibly"))
        self.assertIsNone(extract_yes_no(""))

    def test_normalize_gold_yes_no(self) -> None:
        self.assertEqual(normalize_gold("Yes"), "YES")
        self.assertEqual(normalize_gold("No"), "NO")
        self.assertEqual(normalize_gold("yes"), "YES")

    def test_normalize_gold_letter_still_works(self) -> None:
        self.assertEqual(normalize_gold("A"), "A")
        self.assertEqual(normalize_gold("(C)"), "C")

    def test_score_yes_no_task(self) -> None:
        task = {"task_id": "t", "prompt": "p", "gold": "YES"}
        ok, _ = score_bbh("Answer: Yes", task)
        self.assertTrue(ok)
        ok, detail = score_bbh("Answer: No", task)
        self.assertFalse(ok)
        self.assertIn("expected YES got NO", detail)

    def test_score_yes_no_ignores_stray_mcq_letters(self) -> None:
        # A Yes/No question whose reasoning mentions "(A)" must not be
        # mis-scored as an MCQ answer.
        task = {"task_id": "t", "prompt": "p", "gold": "YES"}
        ok, _ = score_bbh("Reasoning mentions (A) in passing.\nAnswer: Yes", task)
        self.assertTrue(ok)

    def test_extract_answer_unified(self) -> None:
        self.assertEqual(extract_answer("Answer: B"), "B")
        self.assertEqual(extract_answer("Answer: Yes"), "YES")


if __name__ == "__main__":
    unittest.main()
