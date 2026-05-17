"""Echo pilot harness — runs N HumanEval tasks across all routing strategies.

Strategies (the four arms compared):
  haiku-only    one Haiku call, baseline cheap
  sonnet-only   one Sonnet call, baseline quality
  echo-lexical  two Haiku calls (different personas) + lexical agreement →
                accept if outputs match, otherwise escalate to Sonnet
  echo-oracle   two Haiku calls + GROUND-TRUTH agreement (uses test pass/fail).
                Not deployable; characterizes the upper bound on what any
                agreement signal could achieve.

Outputs:
  - prints a per-task line summary and a final aggregate
  - writes one JSONL line per (task, strategy) to experiment/results/<ts>.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import textwrap
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda, RunnableParallel

from chat_claude_code import ChatClaudeCode
from dataset import load_humaneval

RESULTS_DIR = Path(__file__).parent / "results"


# ─────────────────────────────────────────────────────────────────────────────
# Personas — same model, different self-presentations. The hypothesis is that
# on easy tasks both produce equivalent solutions; on hard tasks the
# disagreement is the difficulty signal.
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
    """Models sometimes wrap output in ```python fences despite instructions."""
    fence = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return fence.group(1).strip() if fence else text.strip()


def normalize_for_comparison(code: str) -> str:
    """Lexical agreement signal: ignore whitespace and comments, collapse rest."""
    code = strip_code_fences(code)
    lines = [re.sub(r"\s+", " ", line.strip()) for line in code.splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    return "\n".join(lines)


TEST_TIMEOUT_SECONDS = 10  # match HumanEval's official evaluator default


def run_tests(implementation: str, task: dict) -> tuple[bool, str]:
    """Run model-generated code against the canonical test suite, with hard timeout.

    Why a subprocess: model-generated code WILL sometimes contain infinite loops
    or pathologically slow algorithms (HumanEval/32 'find_zero' is a known
    offender). Bare ``exec()`` in this process hangs the entire sweep.
    Subprocess gives us a kill signal that actually works.

    This mirrors HumanEval's official evaluator pattern (10s timeout, kill on
    expiry, treat as failure with reason "timeout").
    """
    import subprocess as _sp

    body = strip_code_fences(implementation)
    full = task["prompt"] + "\n" + body + "\n" + task["test"] + f"\ncheck({task['entry_point']})\n"

    try:
        result = _sp.run(
            ["python3", "-c", full],
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT_SECONDS,
            check=False,
        )
    except _sp.TimeoutExpired:
        return False, f"timeout (>{TEST_TIMEOUT_SECONDS}s)"

    if result.returncode == 0:
        return True, "passed"
    # Truncate stderr — model traces can be long, we just want the type/message
    err = (result.stderr or "").strip().splitlines()
    last = err[-1] if err else "<no stderr>"
    return False, last[:160]


def call_with_persona(model: ChatClaudeCode, persona: str, task_prompt: str) -> str:
    response = model.invoke([
        SystemMessage(content=persona),
        HumanMessage(content=f"Implement the following function:\n\n{task_prompt}"),
    ])
    return response.content


# ─────────────────────────────────────────────────────────────────────────────
# Routing strategies (the four "arms" — experimental conditions, not OCI cores)
# ─────────────────────────────────────────────────────────────────────────────

def _haiku_pair(task_prompt: str) -> dict[str, str]:
    haiku = ChatClaudeCode(model="haiku")
    parallel = RunnableParallel(
        a=RunnableLambda(lambda p: call_with_persona(haiku, PERSONA_A, p)),
        b=RunnableLambda(lambda p: call_with_persona(haiku, PERSONA_B, p)),
    )
    return parallel.invoke(task_prompt)


def arm_haiku_only(task: dict) -> tuple[str, int]:
    haiku = ChatClaudeCode(model="haiku")
    return call_with_persona(haiku, PERSONA_A, task["prompt"]), 1


def arm_sonnet_only(task: dict) -> tuple[str, int]:
    sonnet = ChatClaudeCode(model="sonnet")
    return call_with_persona(sonnet, PERSONA_A, task["prompt"]), 1


def arm_echo_lexical(task: dict) -> tuple[str, int]:
    pair = _haiku_pair(task["prompt"])
    if normalize_for_comparison(pair["a"]) == normalize_for_comparison(pair["b"]):
        return pair["a"], 2
    sonnet = ChatClaudeCode(model="sonnet")
    return call_with_persona(sonnet, PERSONA_A, task["prompt"]), 3


def ast_agree(a: str, b: str) -> bool:
    """Structural Python-AST equivalence between two candidate solutions.

    Strips code fences, parses both into AST trees, and compares the
    structure-only dump. ``annotate_fields=False`` strips field labels
    but literal names (variable names, function names) still appear in
    the dump — so two functionally identical solutions using different
    variable names will still disagree. Not as strict as lexical, not
    as forgiving as semantic. Naive baseline before adding alpha-renaming.
    """
    try:
        tree_a = ast.parse(strip_code_fences(a))
        tree_b = ast.parse(strip_code_fences(b))
        return ast.dump(tree_a, annotate_fields=False) == ast.dump(tree_b, annotate_fields=False)
    except SyntaxError:
        # Parse error → can't compare → safe default is "disagree, escalate".
        return False


def judge_agree(a: str, b: str, task: dict) -> bool:
    """Ask a third cheap model (Haiku) whether two solutions are equivalent.

    Cost: one extra Haiku call per task on top of the parallel pair. Only
    pays off if it prevents Sonnet escalations >~33% of the time (the
    cost ratio of judge-call to escalation-saved).

    The judge call is purely YES/NO — we keep the prompt minimal to make
    the answer space deterministic. Anything other than 'YES' counts as
    disagree (defensive default; ambiguous responses → escalate).
    """
    judge = ChatClaudeCode(model="haiku")
    resp = judge.invoke([
        SystemMessage(content=(
            "You compare two candidate Python implementations of a problem. "
            "Reply ONLY with the single word YES or NO. Nothing else."
        )),
        HumanMessage(content=(
            f"Problem:\n{task['prompt']}\n\n"
            f"--- Candidate A ---\n{strip_code_fences(a)}\n\n"
            f"--- Candidate B ---\n{strip_code_fences(b)}\n\n"
            "Do A and B implement the same algorithm and produce the same outputs "
            "on all valid inputs? Answer YES or NO."
        )),
    ])
    return resp.content.strip().upper().startswith("YES")


def arm_echo_ast(task: dict) -> tuple[str, int]:
    """Echo with AST-equivalence agreement signal.

    Cheaper than judge (no extra model call) but blind to naming differences
    and idiom choices. Test of whether catching whitespace+comments alone
    moves the needle vs lexical.
    """
    pair = _haiku_pair(task["prompt"])
    if ast_agree(pair["a"], pair["b"]):
        return pair["a"], 2
    sonnet = ChatClaudeCode(model="sonnet")
    return call_with_persona(sonnet, PERSONA_A, task["prompt"]), 3


def arm_echo_judge(task: dict) -> tuple[str, int]:
    """Echo with judge-model agreement signal.

    Three cheap calls (2 Haiku pair + 1 Haiku judge) instead of 2, but
    the judge call evaluates *semantic* equivalence — it can recognize
    two implementations using different idioms as solving the same problem.
    Watch for same-family bias: Haiku judging Haiku may over-agree.
    """
    pair = _haiku_pair(task["prompt"])
    if judge_agree(pair["a"], pair["b"], task):
        return pair["a"], 3  # 2 pair + 1 judge, no escalation
    sonnet = ChatClaudeCode(model="sonnet")
    return call_with_persona(sonnet, PERSONA_A, task["prompt"]), 4


def arm_echo_oracle(task: dict) -> tuple[str, int]:
    """Oracle agreement signal: ground-truth test pass/fail.

    Decision:
      both pass    → accept either (cost: 2 Haiku calls)
      one passes   → accept the passing one (cost: 2 Haiku calls)
      both fail    → escalate to Sonnet (cost: 2 Haiku + 1 Sonnet)
    """
    pair = _haiku_pair(task["prompt"])
    a_passed, _ = run_tests(pair["a"], task)
    b_passed, _ = run_tests(pair["b"], task)
    if a_passed and b_passed:
        return pair["a"], 2
    if a_passed:
        return pair["a"], 2
    if b_passed:
        return pair["b"], 2
    sonnet = ChatClaudeCode(model="sonnet")
    return call_with_persona(sonnet, PERSONA_A, task["prompt"]), 3


ARMS: dict[str, Callable[[dict], tuple[str, int]]] = {
    "haiku-only": arm_haiku_only,
    "sonnet-only": arm_sonnet_only,
    "echo-lexical": arm_echo_lexical,
    "echo-ast": arm_echo_ast,
    "echo-judge": arm_echo_judge,
    "echo-oracle": arm_echo_oracle,
}


# ─────────────────────────────────────────────────────────────────────────────
# Sweep runner
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    task_id: str
    arm: str
    passed: bool
    detail: str
    wall_seconds: float
    sub_calls: int  # # of model calls the strategy consumed for this task


def run_one(task: dict, arm_name: str, arm_fn: Callable[[dict], tuple[str, int]]) -> TaskResult:
    t0 = time.perf_counter()
    try:
        output, sub_calls = arm_fn(task)
        passed, detail = run_tests(output, task)
    except Exception as e:
        return TaskResult(
            task["task_id"], arm_name, False, f"{type(e).__name__}: {str(e)[:200]}",
            time.perf_counter() - t0, 0,
        )
    return TaskResult(task["task_id"], arm_name, passed, detail, time.perf_counter() - t0, sub_calls)


def summarize(results: list[TaskResult]) -> dict:
    by_arm: dict[str, list[TaskResult]] = {}
    for r in results:
        by_arm.setdefault(r.arm, []).append(r)
    summary = {}
    for arm, rs in by_arm.items():
        n = len(rs)
        passed = sum(1 for r in rs if r.passed)
        # "Escalation rate" for Echo arms: fraction of tasks that used >2 calls
        # (oracle and lexical both escalate when sub_calls jumps from 2 to 3).
        # For non-Echo arms this is just an extra metric that happens to be 0%.
        escalated = sum(1 for r in rs if r.sub_calls > 2)
        summary[arm] = {
            "n": n,
            "pass_rate": round(passed / n, 3) if n else None,
            "escalation_rate": round(escalated / n, 3) if n else None,
            "mean_wall_seconds": round(sum(r.wall_seconds for r in rs) / n, 2) if n else None,
            "total_sub_calls": sum(r.sub_calls for r in rs),
            "mean_sub_calls": round(sum(r.sub_calls for r in rs) / n, 2) if n else None,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Echo pilot sweep over HumanEval.")
    parser.add_argument("--n-tasks", type=int, default=1, help="Number of HumanEval tasks (default 1)")
    parser.add_argument("--start", type=int, default=0, help="Skip the first --start tasks (default 0)")
    parser.add_argument("--arms", type=str, default=",".join(ARMS),
                        help=f"Comma-separated arm names (default all: {','.join(ARMS)})")
    args = parser.parse_args()

    selected_arms = {name: ARMS[name] for name in args.arms.split(",") if name in ARMS}
    all_tasks = load_humaneval()
    tasks = all_tasks[args.start : args.start + args.n_tasks]
    print(f"Sweep: {len(tasks)} tasks × {len(selected_arms)} arms = {len(tasks) * len(selected_arms)} runs\n")

    results: list[TaskResult] = []
    for task in tasks:
        print(f"=== {task['task_id']} ===")
        for arm_name, arm_fn in selected_arms.items():
            r = run_one(task, arm_name, arm_fn)
            results.append(r)
            marker = "✓" if r.passed else "✗"
            print(f"  {marker} {arm_name:<14} {r.wall_seconds:>5.1f}s  {r.sub_calls} calls  {r.detail[:60]}")

    summary = summarize(results)
    print("\n=== aggregate ===")
    print(json.dumps(summary, indent=2))

    # Persist for later inspection. Per-task JSONL is the source of truth;
    # aggregate is the convenience view.
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"{ts}_n{len(tasks)}.jsonl"
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    print(f"\nResults written to {out_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
