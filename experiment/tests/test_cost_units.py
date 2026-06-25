"""Tests for cross-provider cost unit accounting."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cost_units import (
    HAIKU_PERSONA,
    SONNET_PERSONA,
    cost_units,
    escalated,
    judge_units_for_arm,
)


class TestCostUnits(unittest.TestCase):
    def test_haiku_persona_is_one(self) -> None:
        self.assertAlmostEqual(HAIKU_PERSONA, 1.0, places=5)

    def test_sonnet_persona_is_three(self) -> None:
        self.assertAlmostEqual(SONNET_PERSONA, 3.0, places=5)

    def test_baselines(self) -> None:
        self.assertAlmostEqual(cost_units("haiku-only", 1), 1.0, places=5)
        self.assertAlmostEqual(cost_units("sonnet-only", 1), 3.0, places=5)

    def test_provider_judge_accept_includes_judge_cost(self) -> None:
        judge = judge_units_for_arm("echo-judge-openai-gpt-5.4-mini")
        self.assertGreater(judge, 0.0)
        self.assertAlmostEqual(
            cost_units("echo-judge-openai-gpt-5.4-mini", 3),
            2 * HAIKU_PERSONA + judge,
            places=5,
        )

    def test_provider_judge_escalate_adds_sonnet(self) -> None:
        judge = judge_units_for_arm("echo-judge-gemini-flash")
        self.assertAlmostEqual(
            cost_units("echo-judge-gemini-flash", 4),
            2 * HAIKU_PERSONA + judge + SONNET_PERSONA,
            places=5,
        )

    def test_local_judge_still_free(self) -> None:
        self.assertAlmostEqual(cost_units("echo-small-judge", 2), 2.0, places=5)

    def test_meghana_n30_mini_sweep_math(self) -> None:
        """29 accepts + 1 escalate on n=30 (3.3% escalation)."""
        judge = judge_units_for_arm("echo-judge-openai-gpt-5.4-mini")
        accept = 2 * HAIKU_PERSONA + judge
        escalate = accept + SONNET_PERSONA
        total = 29 * accept + 1 * escalate
        self.assertAlmostEqual(
            total,
            sum(
                cost_units("echo-judge-openai-gpt-5.4-mini", sc)
                for sc in [3] * 29 + [4]
            ),
            places=4,
        )

    def test_escalation_thresholds(self) -> None:
        self.assertFalse(escalated("echo-judge-openai", 3))
        self.assertTrue(escalated("echo-judge-openai", 4))
        self.assertTrue(escalated("echo-oracle", 3))


if __name__ == "__main__":
    unittest.main()
