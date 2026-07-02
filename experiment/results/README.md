# Echo experiment results

Per-sweep JSONL logs live here — one line per `(task, arm)` pair. Each record:

```json
{"task_id": "HumanEval/100", "arm": "haiku-only", "passed": true,
 "detail": "passed", "wall_seconds": 7.1, "sub_calls": 1}
```

**Canonical HumanEval hard-slice result (sweep 4):** [`20260519T085620Z_n64.jsonl`](20260519T085620Z_n64.jsonl) — tasks 100–163, all seven arms.

Public write-up: [Echo: routing LLM requests cheaply without training a router](https://enspyr.co/blog/echo-cheap-routing-without-a-router) (repo copy: [`../../blog/echo-cheap-routing-without-a-router.md`](../../blog/echo-cheap-routing-without-a-router.md)).

---

## Headline (sweep 4, HumanEval 100–163)

Cost in units where **Haiku = 1**, **Sonnet = 3** per call.

**Escalation** = paid for Sonnet after the two persona calls. In JSONL, that is `sub_calls > 2` for most Echo arms; for **`echo-judge`** use `sub_calls > 3` because the third call is the Haiku judge, not Sonnet.

| Arm | Pass | Escalations | Oracle alignment | Cost (units) |
|-----|------|-------------|------------------|--------------|
| haiku-only | 63/64 | — | — | 64 |
| sonnet-only | 63/64 | — | — | 192 |
| echo-lexical | 64/64 | 55/64 | 16% | 293 |
| echo-ast | 62/64 | 54/64 | 17% | 290 |
| echo-judge | 61/64 | 11/64 | 81% | 225 |
| **echo-small-judge** | **62/64** | **3/64** | **94%** | **137** |
| echo-oracle | 64/64 | 1/64 | (ceiling) | 131 |

**Takeaways:**

- **Cost thesis holds** with a deployable signal: `echo-small-judge` is within ~5% of the oracle cost floor (137 vs 131 units), ~**29% cheaper** than sonnet-only (192 units), with pass rate statistically equal to sonnet-only on this slice.
- **Lexical / AST are noise** at this difficulty — ~85% escalation, total cost **above** sonnet-only.
- **Cross-family local judge (Qwen 2.5 7B via Ollama) beats same-family Haiku judge** on oracle alignment (94% vs 81%) with far fewer escalations (3 vs 11).
- **Caveats:** HumanEval only; n = 64; local judge mean wall time ~86s/task on CPU-only ARM (infra, not API cost). See blog for full discussion.

---

## Sweep history

### Sweep 1: HumanEval/0–19 (`20260516T232833Z_n20.jsonl` — not in repo)

First end-to-end run on the easy slice. All arms passed 100% — too easy to show Echo’s value. Used to validate the harness, not for research conclusions.

### Sweep 2: HumanEval/100–163 v1 (`20260517T112818Z_n64.jsonl`)

First hard-slice run. Surfaced harness bugs:

1. **Output-shape parser miss** — `run_tests` couldn’t assemble Sonnet’s varied output formats into valid Python. Some sonnet-only “failures” were parser misses, not model errors.
2. **Indented-body case** — multi-line bodies kept leading indentation; broke execution.

Fixed in commits `25d385b` and `b707a64`.

### Sweep 3: HumanEval/100–163 v2 (`20260518T011927Z_n64.jsonl`)

Re-run with parser fixes. Established lexical/AST failure and Haiku-judge viability before the local judge arm existed:

| Strategy | Pass | Escalation | Mean wall | Calls/task |
|----------|------|------------|-----------|------------|
| haiku-only | 63/64 (98%) | n/a | 11.1s | 1.00 |
| sonnet-only | 64/64 (100%) | n/a | 7.0s | 1.00 |
| echo-lexical | 64/64 (100%) | 83% | 18.0s | 2.83 |
| echo-ast | 61/64 (95%) | 81% | 18.0s | 2.81 |
| echo-judge | 64/64 (100%) | 14% | 28.0s | 3.14 |
| echo-oracle | 64/64 (100%) | 3% | 12.8s | 2.03 |

Signal-vs-oracle on this file: lexical 20%, AST 19%, Haiku judge **86%**.

### Sweep 4: HumanEval/100–163 + small-model judge (`20260519T085620Z_n64.jsonl`)

Adds **`echo-small-judge`** (local Qwen 2.5 7B-instruct via Ollama). Main table above. This is the sweep cited in the blog and the current paper narrative.

---

## Signal-vs-oracle alignment (sweep 4)

How often each deployable signal makes the same escalate/accept decision as `echo-oracle`:

| Signal | Routes same as oracle | False escalate | Missed escalate |
|--------|----------------------:|---------------:|----------------:|
| lexical | 10/64 (16%) | 54 | 0 |
| AST | 11/64 (17%) | 53 | 0 |
| judge (Haiku) | 52/64 (81%) | 11 | 1 |
| **small-judge (Qwen 7B)** | **60/64 (94%)** | 3 | 1 |

False escalate = paid for Sonnet when oracle would not have. Missed escalate = kept cheap when oracle would have escalated.

---

## False-escalation taxonomy (lexical, sweep 3 era)

On HumanEval 100–163 there are **zero** tasks where lexical agreed but both cheap answers failed. The expensive mistake is the reverse: **lexical-disagreed-but-oracle-agreed** (51/64 in sweep 3). Topic breakdown of those 51 false escalations: 19 string, 21 list/array, 7 math, 4 other — matches the overall topic mix (no useful prefilter). Surface signals fail because implementation entropy dominates; semantic judging is required.

---

## BBH — Claude baselines (clean harness, #1305)

**Harness fix:** Before 2026-06-30, `claude --print` run from inside the repo inherited project `CLAUDE.md` and SessionStart hooks — models answered as the project agent, not the BBH question. That inflated Sonnet latency (40–125s), caused unparseable outputs, and confounded accuracy. Fixed by passing `--setting-sources ""` in `chat_claude_code.py` (keeps Max auth, loads no project settings).

**Invalid (pre-fix):** `20260604T011133Z_bbh_n15.jsonl`, `20260611T230804Z_bbh_n15.jsonl`, `20260625T013951Z_bbh_n30.jsonl`, `20260613T051627Z_bbh_n30.jsonl` (provider judges — needs re-run on clean harness). See [`20260625_bbh_n30_claude_baseline_SUMMARY.md`](20260625_bbh_n30_claude_baseline_SUMMARY.md) (superseded).

### Canonical easy slice — n = 99

[`20260701T000922Z_bbh_n99.jsonl`](20260701T000922Z_bbh_n99.jsonl) — `logical_deduction_three_objects`, `causal_judgement`, `date_understanding` (33 each).

| Arm | Pass | Escalations | Oracle alignment | Cost (units) |
|-----|------|-------------|------------------|--------------|
| haiku-only | 84/99 (0.848) | 0% | — | 99 |
| sonnet-only | 84/99 (0.848) | 0% | — | 297 |
| echo-judge | 86/99 (0.869) | 5.1% | 87% | 278 |
| echo-oracle | 89/99 (0.899) | 10.1% | (ceiling) | 228 |

**Reads:**

- Haiku and Sonnet **tie** on this slice — no accuracy gap to route on.
- Oracle headroom comes entirely from **`causal_judgement`** (baselines 22/33, oracle 27/33) — Haiku and Sonnet miss *different* questions, so routing value is **decorrelated errors**, not “escalate to a smarter model.”
- `echo-judge` captures most of the oracle gain (~3pts below ceiling).

### Pilot re-run — n = 30 (same subtasks)

[`20260630T232105Z_bbh_n30.jsonl`](20260630T232105Z_bbh_n30.jsonl) — 10 per subtask, Claude baselines only. Zero unparseables.

| Arm | Pass | Escalations | Oracle alignment | Cost (units) |
|-----|------|-------------|------------------|--------------|
| haiku-only | 26/30 | 0% | — | 30 |
| sonnet-only | 27/30 | 0% | — | 90 |
| echo-judge | 26/30 | 6.7% | 83% | 85.7 |
| echo-oracle | 28/30 | 10.0% | (ceiling) | 69 |

### Hard subtasks — n = 99 (saturated)

[`20260701T050041Z_bbh_n99.jsonl`](20260701T050041Z_bbh_n99.jsonl) — `logical_deduction_seven_objects`, `tracking_shuffled_objects_seven_objects`, `temporal_sequences` (33 each).

| Arm | Pass | Escalations | Cost (units) |
|-----|------|-------------|--------------|
| haiku-only | 99/99 | 0% | 99 |
| sonnet-only | 99/99 | 0% | 297 |
| echo-judge | 98/99 | 0% | 263 |
| echo-oracle | 99/99 | 0% | 198 |

MCQ BBH is **saturated** for current Claude on these subtasks — no Sonnet-vs-Haiku separation, no meaningful judge headroom. Forward paths: free-form BBH subtasks, MMLU-Pro/GPQA, or reframe Echo as cost/latency routing (“route cheap, lose nothing”).

### Provider judges (pre-fix — pending re-run)

[`20260613T051627Z_bbh_n30.jsonl`](20260613T051627Z_bbh_n30.jsonl) — same 3 subtasks × 10, pre-fix harness. Directional only; Meghana to re-run after clean harness lands on `main`.

| Arm | Pass | Escalations | Oracle alignment | Cost (units) |
|-----|------|-------------|------------------|--------------|
| echo-judge-openai (GPT-5.5) | 27/30 | 10.0% | 80% | 167.8 |
| echo-judge-openai-gpt-5.4-mini | 25/30 | 3.3% | 87% | 77.8 |
| echo-judge-gemini-flash | 17/30 | 3.3% | 87% | 69.0 |
| echo-oracle | 28/30 | 10.0% | (ceiling) | 69.0 |

**Smoke test (wiring only):** [`20260611T133324Z_bbh_n16.jsonl`](20260611T133324Z_bbh_n16.jsonl) — `logical_deduction_three_objects` only, 16/16 pass everywhere, 0% escalation. Too easy for quality claims.

Analyze any BBH sweep:

```bash
python scripts/analyze_sweep.py results/<timestamp>_bbh_n30.jsonl
```

Runbook: [`../scripts/RUN_FOR_NICK.md`](../scripts/RUN_FOR_NICK.md). Resumable sweeps: `scripts/run_bbh_resumable.sh`.

---

## Open work

- **MMLU-Pro canonical sweep** — Nick to run n=125 Claude baselines; Adarsha to analyze.
- **BBH provider-judge re-run** — Meghana, clean harness + OpenAI/Gemini keys on server.
- **MMLU-Pro** — breadth / domain coverage (harness ready; no canonical JSONL yet).
- **Semantic persona axes** — e.g. edge-case-hunter vs happy-path-implementer (current personas are stylistic).
- **Harness** — reproducible sweep metadata, cost in JSONL, `analyze_sweep.py`, resume, BBH adapter (see [`../README.md`](../README.md)).

---

## Reading the JSONL directly

```bash
cd experiment/results

# Pass counts per arm
jq -s 'group_by(.arm) | map({arm: .[0].arm, n: length, passed: (map(select(.passed)) | length)})' 20260519T085620Z_n64.jsonl

# Tasks where echo-small-judge escalated to Sonnet
jq 'select(.arm == "echo-small-judge" and .sub_calls > 2) | .task_id' 20260519T085620Z_n64.jsonl

# Tasks where echo-judge escalated to Sonnet (third call is judge, not Sonnet)
jq 'select(.arm == "echo-judge" and .sub_calls > 3) | .task_id' 20260519T085620Z_n64.jsonl

# Total sub_calls per arm
jq -s 'group_by(.arm) | map({arm: .[0].arm, calls: (map(.sub_calls) | add)})' 20260519T085620Z_n64.jsonl
```

Cost units normalize all provider spend to **Haiku persona call = 1.0 unit** (Claude Haiku 4.5 list pricing). Per-call cost uses typical token profiles: persona calls 600 in / 250 out; judge calls 1200 in / 3 out. Implementation: [`../cost_units.py`](../cost_units.py).

| Model | Input $/M | Output $/M | Judge call (units) |
|-------|-----------|------------|-------------------|
| Haiku (persona) | $1 | $5 | 1.0 (persona) / ~0.73 (judge) |
| Sonnet | $3 | $15 | 3.0 (persona) |
| GPT-5.5 | $5 | $30 | ~3.3 |
| GPT-5.4 | $2.50 | $15 | ~1.6 |
| GPT-5.4-mini | $0.75 | $4.50 | ~0.49 |
| GPT-5.4-nano | $0.20 | $1.25 | ~0.13 |
| Gemini 2.5 Flash | $0.30 | $2.50 | ~0.19 |
| Gemini 2.5 Flash-Lite | $0.15 | $1.25 | ~0.10 |
| Local Qwen (Ollama) | — | — | 0 |
