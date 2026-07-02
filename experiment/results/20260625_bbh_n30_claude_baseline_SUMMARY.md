# BBH n=30 — Claude-baseline slice (2026-06-25)

> **SUPERSEDED.** This run used the pre-fix harness (`claude --print` inherited project
> `CLAUDE.md` + hooks). Sonnet answered as the project agent (92–125s latency, 4
> unparseables). Use [`20260630T232105Z_bbh_n30.jsonl`](20260630T232105Z_bbh_n30.jsonl)
> and [`20260701T000922Z_bbh_n99.jsonl`](20260701T000922Z_bbh_n99.jsonl) instead.
> See [`README.md`](README.md#bbh--claude-baselines-clean-harness-1305).

Run on `main` via the Claude Max CLI (zero API spend). Harness = the post-#3
parser (the fix that resolved the flat-0.14 #335 artifact).

- **Subtasks:** logical_deduction_three_objects, causal_judgement, date_understanding (10 each = 30 tasks)
- **Arms:** haiku-only, sonnet-only, echo-judge, echo-oracle
- **Data:** `20260625T013951Z_bbh_n30.jsonl` (this dir)
- **Analyze:** `python scripts/analyze_sweep.py results/20260625T013951Z_bbh_n30.jsonl`

| arm | pass | esc% | cost* | oracle-align | raw calls |
|---|---|---|---|---|---|
| haiku-only | 26/30 (0.867) | 0% | 30 | — | 30 |
| sonnet-only | 25/30 (0.833) | 0% | 90 | — | 30 |
| echo-judge | 27/30 (0.900) | 0% | 90 | 90% | 90 |
| echo-oracle | 28/30 (0.933) | 10% | 69 | — (ceiling) | 63 |

\* **cost** = weighted units (Haiku=1, Sonnet=3), per `analyze_sweep.cost_units`.
Distinct from **raw calls** (the `sub_calls` field): e.g. sonnet-only is 30 raw
calls but 90 cost-units because each Sonnet call weighs 3x a Haiku call.

## Read
- **haiku-only (0.867) > sonnet-only (0.833)** on this slice — the cheap model is at/above ceiling, so blind routing to Sonnet buys nothing. Consistent with the n=15 pilot; this is the known Echo signal, not the #335 parser artifact.
- **echo-judge lands 0.900 at the same cost as sonnet-only (90 units)** while beating its accuracy — and beats haiku-only too.
- **echo-oracle (ceiling) 0.933 at 69 units** shows the headroom a perfect router would capture.

## Caveat (deflates sonnet)
4 unparseable responses, **all Sonnet outputs at 92-125s wall time** (3x sonnet-only,
1x the oracle's escalated Sonnet call). Haiku never produced an unparseable. The
parser is correct — Sonnet genuinely ran long and skipped the final `Answer: X`
line. So sonnet-only's true accuracy is a touch above the measured 0.833.
