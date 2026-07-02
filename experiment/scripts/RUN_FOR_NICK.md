# Commands for Nick (model sweeps)

Adarsha does not have Claude Max/Pro for `claude login`. Run these on the server when asked.

**Branch:** `integrate/judge-branches` (or `main` after merge)

---

## One-time setup

```bash
cd ~/echo
git fetch origin
git checkout integrate/judge-branches   # or main after PR merge
cd experiment
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]" 2>/dev/null || pip install \
  "langchain-core>=0.3,<0.4" \
  "langchain>=0.3,<0.4" \
  "langchain-ollama>=0.2,<0.4" \
  "langchain-openai>=0.3" \
  "langchain-google-genai>=2.0" \
  "datasets>=2.14"
```

**Prereqs:**

- `claude auth status` shows logged in (Claude Max)
- `OPENAI_API_KEY` set (for `echo-judge-openai*`)
- `GOOGLE_API_KEY` set (for `echo-judge-gemini*`)

**Pre-flight (no API calls except optional claude auth check):**

```bash
python scripts/validate_harness.py
```

---

## Canonical BBH sweep (priority — run when Adarsha asks)

**Goal:** Trustworthy Pareto numbers for the paper. n = 30 tasks (3 subtasks × 10).

**Arms:** baselines + best cross-family judges from Meghana's n=16 pilot + oracle ceiling.

```bash
python scripts/run_bbh_pilot.py \
  --subtasks logical_deduction_three_objects,causal_judgement,date_understanding \
  --n-per-subtask 10 \
  --arms haiku-only,sonnet-only,echo-judge-openai,echo-judge-openai-gpt-5.4-mini,echo-judge-gemini-flash,echo-oracle
```

**Expected:** 30 tasks × 6 arms = **180 model calls** (mostly Haiku; judge arms add OpenAI/Gemini calls).

**Analyze:**

```bash
python scripts/analyze_sweep.py results/<timestamp>_bbh_n30.jsonl
```

**Sanity checks before committing:**

| Check | Healthy signal |
|-------|----------------|
| `haiku-only` pass rate | Below `sonnet-only` (if both 100%, slice may be too easy) |
| `echo-oracle` escalation | > 0% |
| Provider judges vs `echo-judge` | Higher oracle alignment, fewer false escalations |
| `unparseable` count | 0 or explain in commit message |

**Commit results:**

```bash
git add results/<timestamp>_bbh_n30.jsonl
git commit -m "data: canonical BBH sweep n=30 with cross-family judges"
git push
```

Ping Adarsha with the JSONL path. He will update `results/README.md`.

---

## HumanEval sanity (1 task)

Quick check that Claude CLI works:

```bash
python run_pilot.py --n-tasks 1 --start 100 --arms haiku-only,echo-oracle
```

---

## BBH smoke test (small — 15 tasks)

Faster validation after harness changes:

```bash
python scripts/run_bbh_pilot.py \
  --subtasks logical_deduction_three_objects,causal_judgement,date_understanding \
  --n-per-subtask 5 \
  --arms haiku-only,sonnet-only,echo-judge-openai,echo-oracle
```

---

## Full judge comparison (optional — Meghana already ran n=16)

All OpenAI + Gemini judge slots:

```bash
python scripts/run_bbh_pilot.py \
  --subtasks logical_deduction_three_objects,causal_judgement,date_understanding \
  --n-per-subtask 5 \
  --arms echo-judge-openai,echo-judge-openai-gpt-5.4,echo-judge-openai-gpt-5.4-mini,echo-judge-openai-gpt-5.4-nano,echo-judge-gemini-flash,echo-judge-gemini-flash-lite,echo-oracle
```

---

## MMLU-Pro canonical sweep (next — run when Adarsha asks)

**Goal:** First reasoning benchmark where Haiku and Sonnet may diverge. n = 125 (5 categories × 25).

**Arms:** Claude baselines first; add provider judges after clean n=125 lands.

```bash
python scripts/run_mmlu_pro_pilot.py \
  --categories physics,math,law,chemistry,philosophy \
  --n-per-category 25 \
  --arms haiku-only,sonnet-only,echo-judge,echo-oracle
```

**Expected:** 125 tasks × 4 arms = **500 runs** (mostly Haiku; echo-judge adds Haiku judge calls).

**Unattended (auto-resume on usage window):**

```bash
./scripts/run_mmlu_pro_resumable.sh \
  --categories physics,math,law,chemistry,philosophy \
  --n-per-category 25 \
  --arms haiku-only,sonnet-only,echo-judge,echo-oracle
```

**Analyze:**

```bash
python scripts/analyze_sweep.py results/<timestamp>_mmlu_pro_n125.jsonl
```

**Sanity checks:**

| Check | Healthy signal |
|-------|----------------|
| `sonnet-only` vs `haiku-only` | Sonnet ≥ Haiku (if tied, slice may still be easy) |
| `echo-oracle` vs baselines | Oracle pass rate above both |
| `echo-judge` escalation | > 0% on harder domains |
| `unparseable` count | 0 |

**Smoke test (25 tasks, ~1h):**

```bash
python scripts/run_mmlu_pro_pilot.py --n-per-category 5
```

---

## Notes

- BBH uses `benchmarks/bbh_arms.py` (MCQ personas + gold-label oracle).
- Yes/No subtasks (`causal_judgement`) use synthetic A/B choices — scoring is unified.
- `echo-small-judge` needs Ollama + `qwen2.5:7b-instruct-q4_K_M` on the server.
- Provider judge arms: 3 calls = accept (2 Haiku + judge); 4 calls = escalated to Sonnet.
