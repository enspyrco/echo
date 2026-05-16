"""End-to-end pilot: one HumanEval task, three routing arms.

Goal of this script is not statistical power — it's to prove the plumbing
works end-to-end. We pick a single canonical HumanEval task, run three
routing arms against it (Haiku-only, Sonnet-only, Echo), and print:

  - what each arm produced
  - whether it passes the test suite
  - rough wall-clock time per arm

If this runs cleanly we have everything needed to scale to N tasks.
"""

from __future__ import annotations

import json
import re
import textwrap
import time
from dataclasses import dataclass
from typing import Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda, RunnableParallel

from chat_claude_code import ChatClaudeCode


# ─────────────────────────────────────────────────────────────────────────────
# One HumanEval task, hardcoded. HumanEval/0 — has_close_elements.
# (Full benchmark integration comes after we've validated the plumbing.)
# ─────────────────────────────────────────────────────────────────────────────

HUMANEVAL_PROMPT = '''from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
'''

HUMANEVAL_TESTS = '''
METADATA = {}


def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True
    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False
'''
ENTRY_POINT = "has_close_elements"


# ─────────────────────────────────────────────────────────────────────────────
# Personas — the two prompt variants whose agreement signal we use for Echo.
# Deliberately different *framings*, not different *capability levels*.
# Same model, just nudged into different self-presentations. The hypothesis is
# that on easy tasks both framings produce equivalent solutions; on hard tasks
# the disagreement is the difficulty signal.
# ─────────────────────────────────────────────────────────────────────────────

PERSONA_A = textwrap.dedent("""\
    You are a careful, methodical programmer. You value correctness over
    cleverness. You think through edge cases before writing code. When you
    return code, you return only the code — no commentary, no markdown
    fences, no explanation. Just the implementation.
""").strip()

PERSONA_B = textwrap.dedent("""\
    You are a pragmatic senior engineer. You write the simplest code that
    correctly solves the problem. You don't over-engineer. When you return
    code, you return only the code — no commentary, no markdown fences,
    no explanation. Just the implementation.
""").strip()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def strip_code_fences(text: str) -> str:
    """Models sometimes wrap output in ```python fences despite instructions.
    Strip them so we can compare and execute the inner code cleanly."""
    fence_match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return text.strip()


def normalize_for_comparison(code: str) -> str:
    """For Echo's agreement check: are the two Haiku outputs the same solution?

    Whitespace-tolerant comparison is the v1 strategy. We strip blank lines,
    collapse whitespace within lines, and ignore comments. This is crude;
    a real version should run both against the test suite and compare
    pass/fail patterns, which is a stronger agreement signal. For the
    pilot, lexical normalization is enough to demonstrate the routing
    decision happens at all.
    """
    code = strip_code_fences(code)
    lines = [re.sub(r"\s+", " ", line.strip()) for line in code.splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    return "\n".join(lines)


def run_tests(implementation: str, prompt: str, tests: str, entry_point: str) -> tuple[bool, str]:
    """Execute the model's implementation against the canonical tests.

    We splice the prompt (which contains the function signature) together
    with the model's body, then run the HumanEval-format test suite. Any
    exception or AssertionError = fail.
    """
    body = strip_code_fences(implementation)
    full_program = prompt + "\n" + body + "\n" + tests + f"\ncheck({entry_point})\n"
    ns: dict = {}
    try:
        exec(full_program, ns)  # noqa: S102 — pilot harness, trusted local code
        return True, "passed"
    except AssertionError as e:
        return False, f"assertion: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


@dataclass
class ArmResult:
    arm: str
    output: str
    passed: bool
    detail: str
    wall_seconds: float
    sub_calls: int  # how many model calls the arm consumed


# ─────────────────────────────────────────────────────────────────────────────
# Routing arms
# ─────────────────────────────────────────────────────────────────────────────

def call_with_persona(model: ChatClaudeCode, persona: str, task_prompt: str) -> str:
    """Single Haiku/Sonnet call with a persona system message."""
    response = model.invoke([
        SystemMessage(content=persona),
        HumanMessage(content=f"Implement the following function:\n\n{task_prompt}"),
    ])
    return response.content


def arm_haiku_only(task_prompt: str) -> tuple[str, int]:
    """Baseline: one Haiku call with no persona scaffolding."""
    haiku = ChatClaudeCode(model="haiku")
    out = call_with_persona(haiku, PERSONA_A, task_prompt)
    return out, 1


def arm_sonnet_only(task_prompt: str) -> tuple[str, int]:
    """Quality baseline: one Sonnet call."""
    sonnet = ChatClaudeCode(model="sonnet")
    out = call_with_persona(sonnet, PERSONA_A, task_prompt)
    return out, 1


def _haiku_pair(task_prompt: str) -> dict[str, str]:
    """Two parallel Haiku calls, one per persona. Returns {a: str, b: str}.

    Shared by all Echo variants. RunnableParallel runs both Haiku calls
    concurrently in worker threads — two parallel Haiku calls cost less
    than one Sonnet call, AND they finish in ~max(a, b) wall time, not
    a + b. That's the cost-economic point of Echo.
    """
    haiku = ChatClaudeCode(model="haiku")
    parallel = RunnableParallel(
        a=RunnableLambda(lambda p: call_with_persona(haiku, PERSONA_A, p)),
        b=RunnableLambda(lambda p: call_with_persona(haiku, PERSONA_B, p)),
    )
    return parallel.invoke(task_prompt)


def arm_echo_lexical(task_prompt: str) -> tuple[str, int]:
    """Echo v1 — agreement signal is whitespace-tolerant string match.

    Too strict for code-gen: two correct implementations almost always
    differ in naming/whitespace and trip the escalation. Kept here as
    the baseline 'naive' signal we compare smarter signals against.
    """
    pair = _haiku_pair(task_prompt)
    if normalize_for_comparison(pair["a"]) == normalize_for_comparison(pair["b"]):
        return pair["a"], 2
    sonnet = ChatClaudeCode(model="sonnet")
    return call_with_persona(sonnet, PERSONA_A, task_prompt), 3


def arm_echo_oracle(task_prompt: str) -> tuple[str, int]:
    """Echo with an ORACLE agreement signal — uses ground-truth test pass/fail.

    NOTE: This is not a production-deployable router. A live router can't
    consult ground-truth tests at routing time — if it could, you wouldn't
    need a router in the first place. This arm exists to characterize the
    UPPER BOUND of what any agreement signal could achieve on this
    benchmark: it's the cost-quality frontier Echo would lie on if its
    agreement signal were perfect. Real signals (AST diff, judge model,
    embedding similarity) get added as separate arms and measured against
    this upper bound.

    Decision logic:
      both pass → accept either (cost: 2 Haiku calls)
      exactly one passes → accept the passing one (cost: 2 Haiku calls)
      both fail → escalate to Sonnet (cost: 2 Haiku + 1 Sonnet)
    """
    pair = _haiku_pair(task_prompt)
    a_passed, _ = run_tests(pair["a"], HUMANEVAL_PROMPT, HUMANEVAL_TESTS, ENTRY_POINT)
    b_passed, _ = run_tests(pair["b"], HUMANEVAL_PROMPT, HUMANEVAL_TESTS, ENTRY_POINT)

    if a_passed and b_passed:
        return pair["a"], 2  # both correct — keep either
    if a_passed:
        return pair["a"], 2  # one correct — keep it
    if b_passed:
        return pair["b"], 2
    # Both failed — Sonnet is our remaining hope.
    sonnet = ChatClaudeCode(model="sonnet")
    return call_with_persona(sonnet, PERSONA_A, task_prompt), 3


# ─────────────────────────────────────────────────────────────────────────────
# Main pilot
# ─────────────────────────────────────────────────────────────────────────────

ARMS: dict[str, Callable[[str], tuple[str, int]]] = {
    "haiku-only": arm_haiku_only,
    "sonnet-only": arm_sonnet_only,
    "echo-lexical": arm_echo_lexical,
    "echo-oracle": arm_echo_oracle,
}


def main() -> None:
    results: list[ArmResult] = []
    for arm_name, arm_fn in ARMS.items():
        print(f"\n=== arm: {arm_name} ===")
        t0 = time.perf_counter()
        try:
            output, sub_calls = arm_fn(HUMANEVAL_PROMPT)
            elapsed = time.perf_counter() - t0
            passed, detail = run_tests(output, HUMANEVAL_PROMPT, HUMANEVAL_TESTS, ENTRY_POINT)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            output, sub_calls, passed, detail = "", 0, False, f"{type(e).__name__}: {e}"

        r = ArmResult(arm_name, output, passed, detail, elapsed, sub_calls)
        results.append(r)
        print(f"  passed: {r.passed}")
        print(f"  wall:   {r.wall_seconds:.1f}s ({r.sub_calls} model calls)")
        print(f"  detail: {r.detail}")
        if not r.passed:
            print("  output (first 20 lines):")
            for line in r.output.splitlines()[:20]:
                print(f"    {line}")

    print("\n=== summary ===")
    print(json.dumps([{
        "arm": r.arm, "passed": r.passed, "wall_seconds": round(r.wall_seconds, 1),
        "sub_calls": r.sub_calls,
    } for r in results], indent=2))


if __name__ == "__main__":
    main()
