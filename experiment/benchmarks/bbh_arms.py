"""BBH routing arms — same Echo logic as run_pilot, MCQ prompts and scoring."""

from __future__ import annotations

import re
import textwrap
from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda, RunnableParallel

from benchmarks.bbh import extract_choice, score_bbh
from chat_claude_code import ChatClaudeCode
from run_pilot import SMALL_JUDGE_BASE_URL, SMALL_JUDGE_MODEL, _HAS_OLLAMA

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None  # type: ignore[misc, assignment]

PERSONA_A = textwrap.dedent("""\
    You are a careful, methodical reasoner. Work through the problem step by step.
    End with exactly one line: Answer: X
    where X is the letter of your chosen option (A, B, C, ...). No text after that line.
""").strip()

PERSONA_B = textwrap.dedent("""\
    You are a pragmatic problem solver. Use the simplest reasoning that fits the evidence.
    End with exactly one line: Answer: X
    where X is the letter of your chosen option (A, B, C, ...). No text after that line.
""").strip()


def call_with_persona(model: ChatClaudeCode, persona: str, task_prompt: str) -> str:
    response = model.invoke([
        SystemMessage(content=persona),
        HumanMessage(content=task_prompt),
    ])
    return response.content


def _haiku_pair(task_prompt: str) -> dict[str, str]:
    haiku = ChatClaudeCode(model="haiku")
    parallel = RunnableParallel(
        a=RunnableLambda(lambda p: call_with_persona(haiku, PERSONA_A, p)),
        b=RunnableLambda(lambda p: call_with_persona(haiku, PERSONA_B, p)),
    )
    return parallel.invoke(task_prompt)


def _normalize_answer_text(text: str) -> str:
    """Lexical signal for BBH: collapse whitespace on full response."""
    return re.sub(r"\s+", " ", text.strip())


def lexical_agree(a: str, b: str) -> bool:
    """Same final choice letter, or identical normalized text."""
    ca, cb = extract_choice(a), extract_choice(b)
    if ca is not None and cb is not None:
        return ca == cb
    return _normalize_answer_text(a) == _normalize_answer_text(b)


def judge_agree(a: str, b: str, task: dict, judge: BaseChatModel | None = None) -> bool:
    if judge is None:
        judge = ChatClaudeCode(model="haiku")
    resp = judge.invoke([
        SystemMessage(content=(
            "You compare two answers to the same multiple-choice question. "
            "Reply ONLY with the single word YES or NO. Nothing else."
        )),
        HumanMessage(content=(
            f"Question and choices:\n{task['prompt']}\n\n"
            f"--- Candidate A ---\n{a}\n\n"
            f"--- Candidate B ---\n{b}\n\n"
            "Do A and B select the same choice letter (same meaning)? Answer YES or NO."
        )),
    ])
    return resp.content.strip().upper().startswith("YES")


def arm_haiku_only(task: dict) -> tuple[str, int]:
    haiku = ChatClaudeCode(model="haiku")
    return call_with_persona(haiku, PERSONA_A, task["prompt"]), 1


def arm_sonnet_only(task: dict) -> tuple[str, int]:
    sonnet = ChatClaudeCode(model="sonnet")
    return call_with_persona(sonnet, PERSONA_A, task["prompt"]), 1


def arm_echo_lexical(task: dict) -> tuple[str, int]:
    pair = _haiku_pair(task["prompt"])
    if lexical_agree(pair["a"], pair["b"]):
        return pair["a"], 2
    sonnet = ChatClaudeCode(model="sonnet")
    return call_with_persona(sonnet, PERSONA_A, task["prompt"]), 3


def arm_echo_judge(task: dict) -> tuple[str, int]:
    pair = _haiku_pair(task["prompt"])
    if judge_agree(pair["a"], pair["b"], task):
        return pair["a"], 3
    sonnet = ChatClaudeCode(model="sonnet")
    return call_with_persona(sonnet, PERSONA_A, task["prompt"]), 4


def arm_echo_small_judge(task: dict) -> tuple[str, int]:
    if not _HAS_OLLAMA or ChatOllama is None:
        raise RuntimeError(
            "echo-small-judge requires langchain-ollama and Ollama with "
            f"{SMALL_JUDGE_MODEL}"
        )
    pair = _haiku_pair(task["prompt"])
    small_judge = ChatOllama(model=SMALL_JUDGE_MODEL, base_url=SMALL_JUDGE_BASE_URL)
    if judge_agree(pair["a"], pair["b"], task, judge=small_judge):
        return pair["a"], 2
    sonnet = ChatClaudeCode(model="sonnet")
    return call_with_persona(sonnet, PERSONA_A, task["prompt"]), 3


def arm_echo_oracle(task: dict) -> tuple[str, int]:
    """Escalate only when both cheap answers are wrong vs gold label."""
    pair = _haiku_pair(task["prompt"])
    a_ok, _ = score_bbh(pair["a"], task)
    b_ok, _ = score_bbh(pair["b"], task)
    if a_ok and b_ok:
        return pair["a"], 2
    if a_ok:
        return pair["a"], 2
    if b_ok:
        return pair["b"], 2
    sonnet = ChatClaudeCode(model="sonnet")
    return call_with_persona(sonnet, PERSONA_A, task["prompt"]), 3


BBH_ARMS: dict[str, Callable[[dict], tuple[str, int]]] = {
    "haiku-only": arm_haiku_only,
    "sonnet-only": arm_sonnet_only,
    "echo-lexical": arm_echo_lexical,
    "echo-judge": arm_echo_judge,
    "echo-small-judge": arm_echo_small_judge,
    "echo-oracle": arm_echo_oracle,
}
