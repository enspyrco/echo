"""Echo cost units — normalize API spend across providers.

Anchor: one Haiku persona call = 1.0 unit (Claude Haiku 4.5 list pricing).

Per-call cost uses typical token profiles for this harness:
  - Persona / baseline calls: 600 input, 250 output tokens
  - Judge calls (YES/NO agreement): 1200 input, 3 output tokens

List prices sourced from provider docs (June 2026). Update MODEL_PRICING when tiers change.
"""

from __future__ import annotations

from dataclasses import dataclass

# Typical tokens per call shape in Echo sweeps (BBH / HumanEval).
PERSONA_INPUT_TOKENS = 600
PERSONA_OUTPUT_TOKENS = 250
JUDGE_INPUT_TOKENS = 1200
JUDGE_OUTPUT_TOKENS = 3


@dataclass(frozen=True)
class ModelPricing:
    """USD per million tokens (input, output)."""

    input_per_m: float
    output_per_m: float


# Anthropic: https://platform.claude.com/docs/en/about-claude/pricing
# OpenAI: https://developers.openai.com/api/docs/models (GPT-5.x family)
# Google: https://ai.google.dev/gemini-api/docs/pricing
MODEL_PRICING: dict[str, ModelPricing] = {
    "haiku": ModelPricing(1.00, 5.00),
    "sonnet": ModelPricing(3.00, 15.00),
    "gpt-5.5": ModelPricing(5.00, 30.00),
    "gpt-5.4": ModelPricing(2.50, 15.00),
    "gpt-5.4-mini": ModelPricing(0.75, 4.50),
    "gpt-5.4-nano": ModelPricing(0.20, 1.25),
    "gemini-2.5-pro": ModelPricing(1.25, 10.00),
    "gemini-2.5-flash": ModelPricing(0.30, 2.50),
    "gemini-2.5-flash-lite": ModelPricing(0.15, 1.25),
    "local-qwen": ModelPricing(0.0, 0.0),
}

# Map sweep arm names to judge model keys in MODEL_PRICING.
ARM_JUDGE_MODEL: dict[str, str] = {
    "echo-judge": "haiku",
    "echo-judge-openai": "gpt-5.5",
    "echo-judge-openai-gpt-5.4": "gpt-5.4",
    "echo-judge-openai-gpt-5.4-mini": "gpt-5.4-mini",
    "echo-judge-openai-gpt-5.4-nano": "gpt-5.4-nano",
    "echo-judge-gemini-pro": "gemini-2.5-pro",
    "echo-judge-gemini-flash": "gemini-2.5-flash",
    "echo-judge-gemini-flash-lite": "gemini-2.5-flash-lite",
}


def _call_cost_usd(pricing: ModelPricing, input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * pricing.input_per_m + output_tokens * pricing.output_per_m
    ) / 1_000_000


def _persona_units(model_key: str) -> float:
    """Units for one persona-style call; Haiku persona = 1.0."""
    baseline = _call_cost_usd(
        MODEL_PRICING["haiku"], PERSONA_INPUT_TOKENS, PERSONA_OUTPUT_TOKENS
    )
    cost = _call_cost_usd(
        MODEL_PRICING[model_key], PERSONA_INPUT_TOKENS, PERSONA_OUTPUT_TOKENS
    )
    return cost / baseline


def _judge_units(model_key: str) -> float:
    """Units for one judge YES/NO call."""
    baseline = _call_cost_usd(
        MODEL_PRICING["haiku"], PERSONA_INPUT_TOKENS, PERSONA_OUTPUT_TOKENS
    )
    cost = _call_cost_usd(
        MODEL_PRICING[model_key], JUDGE_INPUT_TOKENS, JUDGE_OUTPUT_TOKENS
    )
    return cost / baseline


HAIKU_PERSONA = _persona_units("haiku")
SONNET_PERSONA = _persona_units("sonnet")


def _uses_judge_call(arm: str) -> bool:
    return arm in ARM_JUDGE_MODEL or arm.startswith(
        ("echo-judge-openai", "echo-judge-gemini")
    )


def _judge_model_key(arm: str) -> str | None:
    if arm in ARM_JUDGE_MODEL:
        return ARM_JUDGE_MODEL[arm]
    if arm.startswith("echo-judge-openai"):
        return "gpt-5.5"
    if arm.startswith("echo-judge-gemini"):
        return "gemini-2.5-flash"
    return None


def escalated(arm: str, sub_calls: int) -> bool:
    if _uses_judge_call(arm):
        return sub_calls > 3
    if arm.startswith("echo-"):
        return sub_calls > 2
    return False


def cost_units(arm: str, sub_calls: int) -> float:
    """Total cost units for one task, all providers included."""
    if arm == "haiku-only":
        return sub_calls * HAIKU_PERSONA
    if arm == "sonnet-only":
        return sub_calls * SONNET_PERSONA

    if arm == "echo-oracle" or arm in ("echo-lexical", "echo-ast"):
        if sub_calls <= 2:
            return sub_calls * HAIKU_PERSONA
        return 2 * HAIKU_PERSONA + SONNET_PERSONA

    if arm == "echo-small-judge":
        if sub_calls <= 2:
            return sub_calls * HAIKU_PERSONA
        return 2 * HAIKU_PERSONA + SONNET_PERSONA

    if _uses_judge_call(arm):
        judge_key = _judge_model_key(arm)
        if judge_key is None:
            raise ValueError(f"No judge pricing for arm {arm!r}")
        judge = _judge_units(judge_key)
        if sub_calls <= 3:
            return 2 * HAIKU_PERSONA + judge
        return 2 * HAIKU_PERSONA + judge + SONNET_PERSONA

    if arm.startswith("echo-"):
        if sub_calls <= 2:
            return sub_calls * HAIKU_PERSONA
        return 2 * HAIKU_PERSONA + SONNET_PERSONA

    return sub_calls * HAIKU_PERSONA


def judge_units_for_arm(arm: str) -> float:
    """Expose per-judge-call unit cost for documentation."""
    key = _judge_model_key(arm)
    if key is None:
        raise ValueError(f"No judge model for arm {arm!r}")
    return _judge_units(key)
