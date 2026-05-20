# Echo experiment results

This directory holds per-sweep JSONL records — one line per `(task, routing_strategy)` pair. Each record:

```json
{"task_id": "HumanEval/100", "arm": "haiku-only", "passed": true,
 "detail": "passed", "wall_seconds": 7.1, "sub_calls": 1}
```

## What's been measured so far

### Sweep 1: HumanEval/0-19 (`20260516T232833Z_n20.jsonl`)
First end-to-end run. All 4 routing strategies passed 100% — but this slice is too easy to demonstrate Echo's value because Haiku alone is already sufficient. Used to validate the harness, not to draw research conclusions.

### Sweep 2: HumanEval/100-163 v1 (`20260517T112818Z_n64.jsonl`)
First run on the harder slice. Surfaced two classes of harness bugs:

1. **Output-shape parser miss** — `run_tests` couldn't assemble Sonnet's varied output formats (markdown-fenced, single-backtick spans, body-only `return`, `import`-led full files) into valid Python. 6 sonnet-only "failures" were correctly-answered tasks the parser couldn't run.

2. **Indented-body case** — `.strip()` only removes whitespace from the whole string's edges, leaving multi-line bodies indented on lines 2+. Caused 4 Haiku "failures" on tasks where the model returned a function body without its `def` signature.

Both fixed in commits `25d385b` and `b707a64`.

### Sweep 3: HumanEval/100-163 v2 (`20260518T011927Z_n64.jsonl`)
Re-run with the parser fixes. Pass rates now align with published benchmarks:

| Strategy | Pass | Escalation | Mean wall | Calls/task |
|---|---|---|---|---|
| haiku-only | 63/64 (98%) | n/a | 11.1s | 1.00 |
| sonnet-only | 64/64 (100%) | n/a | 7.0s | 1.00 |
| echo-lexical | 64/64 (100%) | 83% | 18.0s | 2.83 |
| echo-ast | 61/64 (95%) | 81% | 18.0s | 2.81 |
| **echo-judge** | **64/64 (100%)** | **14%** | 28.0s | 3.14 |
| echo-oracle | 64/64 (100%) | 3% | 12.8s | 2.03 |

## Signal-vs-oracle alignment

The core research finding: how closely does each agreement signal track the ground-truth oracle?

| Signal | Routes same as oracle | False escalate | Missed escalate |
|---|---|---|---|
| lexical | 13/64 (20%) | 51 | 0 |
| AST | 12/64 (19%) | 51 | 1 |
| **judge** | **55/64 (86%)** | 8 | 1 |

Lexical/AST are essentially uniform-noise signals at this difficulty. Judge (a third Haiku call asking "are these equivalent?") reaches 86% oracle alignment.

## Cost economics

Assuming Haiku = 1 unit, Sonnet = 3 units, for 64 tasks:

| Strategy | Total cost | Pass rate |
|---|---|---|
| haiku-only | 64u | 98% |
| sonnet-only | 192u | 100% |
| echo-lexical | 64u + 53×3u = 223u | 100% |
| echo-judge | 192u + 9×3u = 219u | 100% |
| **echo-oracle** | **64u×2 + 2×3u = 134u** | 100% |

The oracle establishes the cost-quality upper bound: **30% cheaper than sonnet-only at equal accuracy**. The challenge for the paper is finding a real (no-ground-truth) signal that approaches this frontier. Judge currently lands at sonnet-only's cost, not the oracle's — the next research question is how to get cheaper.

## False-escalation taxonomy

Reframing of an early hypothesis: Claudis (parallel session) suggested tasks where `lexical-agreed-but-failed` would cluster structurally. On HumanEval 100-163, there are **zero such tasks**. The empirical question is the inverse: tasks where `lexical-disagreed-but-oracle-agreed` (51 of 64 tasks). Categorising the 51 by topic:

- string tasks: 19
- list/array tasks: 21
- math/number tasks: 7
- other: 4

Distribution is uniform across task topics — there's no clean category-based prefilter. Code-gen's natural implementation entropy (loop vs comprehension, dict vs class, etc) dominates topical structure. This forecloses the "smarter lexical/AST prefilter" path and motivates pursuing cheaper *semantic-equivalence* judges.

## Open experiments

- **Sweep 4: small-model judge (in progress as of 2026-05-19)** — Adds `echo-small-judge` using local Qwen 2.5 7B-instruct via Ollama. Hypothesis: 7B-instruct is accurate enough to maintain ~75-85% oracle alignment at near-zero call cost (local CPU compute = free relative to Max budget).
- **Semantic persona axes** — Current PERSONA_A / PERSONA_B are stylistic variants ("careful" vs "pragmatic"). Future axes worth testing: `edge-case-hunter` vs `happy-path-implementer`, `strict-spec` vs `creative-interpretation`. These probe whether *semantic* persona variation produces a more useful disagreement signal than *stylistic*.
- **Generality across benchmarks** — Current data is HumanEval-only. BBH (Big-Bench Hard) and MMLU-Pro would test whether the signal-quality patterns hold beyond code generation.

## Reading the JSONL directly

```bash
# How many tasks did each arm pass?
jq -s 'group_by(.arm) | map({arm: .[0].arm, n: length, passed: (map(select(.passed)) | length)})' results.jsonl

# Which tasks escalated for echo-judge?
jq 'select(.arm == "echo-judge" and .sub_calls > 3) | .task_id' results.jsonl

# Per-arm cost (assuming Haiku=1, Sonnet=3 for arms that ever escalate to Sonnet)
jq -s 'group_by(.arm) | map({arm: .[0].arm, calls: (map(.sub_calls) | add)})' results.jsonl
```
