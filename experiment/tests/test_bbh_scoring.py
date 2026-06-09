"""Unit tests for BBH answer parsing and scoring (no model calls)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.bbh import extract_choice, format_prompt, normalize_gold, score_bbh
from benchmarks.bbh import _row_to_task, normalize_gold_for_choices


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

    def test_final_answer_sentence_beats_reasoning_option_mentions(self) -> None:
        self.assertEqual(
            extract_choice("I considered (A) and then (B). Therefore the answer is C."),
            "C",
        )

    def test_correct_answer_sentence_beats_reasoning_option_mentions(self) -> None:
        self.assertEqual(
            extract_choice("Option (A) is tempting, but the correct answer is (C)."),
            "C",
        )

    def test_reasoning_option_mentions_alone_are_not_parseable(self) -> None:
        self.assertIsNone(extract_choice("I considered (A), then (B), then (C)."))

    # Regression: an answer stated EARLY followed by trailing reasoning must
    # still be recovered. Tail-only matching dropped these (cage-match #335).
    def test_answer_stated_early_then_six_trailing_lines(self) -> None:
        out = "The answer is C.\n\nl1\nl2\nl3\nl4\nl5\nl6"
        self.assertEqual(extract_choice(out), "C")

    def test_answer_colon_early_then_trailing_lines(self) -> None:
        out = "Answer: C\nthanks\nbye\n.\n-\n="
        self.assertEqual(extract_choice(out), "C")

    # Regression: under re.I, [A-Z] also matches lowercase, so a broad
    # "answer is <word>" pattern must NOT grab the first letter of prose.
    def test_answer_is_lowercase_word_is_not_a_false_letter(self) -> None:
        self.assertIsNone(extract_choice("After analysis, the answer is straightforward."))
        self.assertIsNone(extract_choice("The answer is dependent on the framing."))
        self.assertIsNone(extract_choice("So the final answer is best understood as follows."))


class TestNormalizeGoldForChoices(unittest.TestCase):
    # Regression: choices carrying text without an explicit "label" key must
    # not raise IndexError (cage-match: Kelvin).
    def test_text_choices_without_labels_fall_back_to_positional(self) -> None:
        self.assertEqual(
            normalize_gold_for_choices("No", {"text": ["Yes", "No"]}),
            "B",
        )

    # Regression: a single-letter target must resolve to its own label, not be
    # remapped by a decoy choice whose text is that same letter (cage-match:
    # Carnot).
    def test_single_letter_target_not_remapped_by_decoy_text(self) -> None:
        choices = {"label": ["A", "B", "C"], "text": ["apple", "A", "cat"]}
        self.assertEqual(normalize_gold_for_choices("A", choices), "A")


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

    def test_binary_answer_text_scores_against_synthetic_choices(self) -> None:
        task = _row_to_task(
            "causal_judgement",
            0,
            {"question": "Did X cause Y?", "target": "No"},
        )
        ok, detail = score_bbh("Reasoning...\nAnswer: No", task)
        self.assertTrue(ok)
        self.assertEqual(detail, "passed")

    def test_binary_answer_letter_scores_against_synthetic_choices(self) -> None:
        task = _row_to_task(
            "causal_judgement",
            0,
            {"question": "Did X cause Y?", "target": "No"},
        )
        ok, detail = score_bbh("Answer: B", task)
        self.assertTrue(ok)
        self.assertEqual(detail, "passed")


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
