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

## Open work

- **BBH** — test whether cross-family judging generalizes to hard reasoning (priority over MMLU-Pro for the core hypothesis).
- **MMLU-Pro** — breadth / domain coverage after BBH.
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

Cost units (Haiku=1, Sonnet=3): for Echo arms, `2 + 3 * (sub_calls > 2 ? 1 : 0)` per task when `sub_calls` is 2 or 3; see blog for arm-specific accounting when the judge adds a third cheap call.
