"""Tests for BBH routing helpers (no API calls)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.bbh import _row_to_task
from benchmarks.bbh_arms import lexical_agree


class TestLexicalAgree(unittest.TestCase):
    def test_same_answer_line(self) -> None:
        self.assertTrue(lexical_agree("Answer: B", "Answer: B"))

    def test_different_letters(self) -> None:
        self.assertFalse(lexical_agree("Answer: A", "Answer: C"))

    def test_same_letter_different_reasoning(self) -> None:
        self.assertTrue(
            lexical_agree("Because foo.\nAnswer: A", "Other path.\nAnswer: A")
        )

    def test_binary_yes_no_text_agrees_with_letter(self) -> None:
        task = _row_to_task(
            "causal_judgement",
            0,
            {"question": "Did X cause Y?", "target": "No"},
        )
        self.assertTrue(
            lexical_agree("Answer: No", "Answer: B", task)
        )


if __name__ == "__main__":
    unittest.main()
