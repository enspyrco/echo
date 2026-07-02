# Commands for Nick (model sweeps)

Adarsha does not have Claude Max/Pro for `claude login`. Run these on the server when asked.

**Branch:** `feat/mmlu-pro-harness` (or `main` after merge)

---

## One-time setup

```bash
cd ~/echo
git fetch origin
git checkout feat/mmlu-pro-harness   # or main after PR merge
git pull
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

**Pre-flight (no model API calls):**

```bash
python scripts/validate_harness.py
```

**Important:** This branch includes the BBH harness fix (`--setting-sources ""` in
`chat_claude_code.py`). Without it, models inherit project `CLAUDE.md` and BBH/MMLU
accuracy numbers are invalid.

---

## MMLU-Pro canonical sweep (priority — run when Adarsha asks)

**Goal:** First reasoning benchmark where Haiku and Sonnet may diverge. n = 125 (5 categories × 25).

**Arms:** Claude baselines first; add provider judges after clean n=125 lands.

```bash
./scripts/run_mmlu_pro_resumable.sh \
  --categories physics,math,law,chemistry,philosophy \
  --n-per-category 25 \
  --arms haiku-only,sonnet-only,echo-judge,echo-oracle
```

**Expected:** 125 tasks × 4 arms = **500 runs** (mostly Haiku; echo-judge adds Haiku judge calls).

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

**Smoke test (25 tasks):**

```bash
python scripts/run_mmlu_pro_pilot.py --n-per-category 5
```

**Commit results:**

```bash
git add results/<timestamp>_mmlu_pro_n125.jsonl
git commit -m "data(mmlu-pro): canonical n=125 Claude baseline sweep"
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

## BBH — provider-judge re-run (lower priority)

MCQ BBH Claude baselines are done (clean n=99 on this branch). The easy and hard
subtasks are **saturated** for current Claude (Haiku = Sonnet). Remaining BBH work
is re-running **provider judges** on the clean harness:

```bash
python scripts/run_bbh_pilot.py \
  --subtasks logical_deduction_three_objects,causal_judgement,date_understanding \
  --n-per-subtask 10 \
  --arms echo-judge-openai,echo-judge-openai-gpt-5.4-mini,echo-judge-gemini-flash,echo-oracle
```

Pre-fix provider-judge JSONLs on `main` must not be cited — see
[`results/README.md`](../results/README.md).

**BBH smoke test (15 tasks):**

```bash
python scripts/run_bbh_pilot.py \
  --subtasks logical_deduction_three_objects,causal_judgement,date_understanding \
  --n-per-subtask 5 \
  --arms haiku-only,sonnet-only,echo-judge,echo-oracle
```

---

## Notes

- BBH and MMLU-Pro share `benchmarks/bbh_arms.py` (MCQ personas + gold-label oracle).
- Yes/No BBH subtasks (`causal_judgement`) use synthetic A/B choices — scoring is unified.
- MMLU-Pro supports variable option counts (4–10 choices, letters A–J).
- `echo-small-judge` needs Ollama + `qwen2.5:7b-instruct-q4_K_M` on the server.
- Provider judge arms: 3 calls = accept (2 Haiku + judge); 4 calls = escalated to Sonnet.
- Long sweeps auto-resume on Max usage window: `run_bbh_resumable.sh`, `run_mmlu_pro_resumable.sh`.
