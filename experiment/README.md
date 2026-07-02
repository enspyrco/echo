# Echo experiment harness

Python runner for HumanEval routing sweeps. Each **arm** is a strategy (always cheap, always expensive, Echo variants). The harness logs one JSON line per `(task, arm)` under `results/`.

For sweep narratives and headline numbers, see [`results/README.md`](results/README.md). Public summary: [blog post](https://enspyr.co/blog/echo-cheap-routing-without-a-router).

## Prerequisites

| Requirement | Used for |
|-------------|----------|
| **Python 3.11+** | Runtime |
| **`claude` CLI** (Claude Code) | `haiku-only`, `sonnet-only`, and Echo arms that call Haiku/Sonnet via `claude --print` |
| **Internet** (first run) | Downloads HumanEval to `~/.cache/echo/HumanEval.jsonl` |

**Optional** — only for `echo-small-judge`:

- [Ollama](https://ollama.com/) at `http://localhost:11434`
- Model: `qwen2.5:7b-instruct-q4_K_M` (see `SMALL_JUDGE_MODEL` in `run_pilot.py`)

**Optional** — only for OpenAI judge arms:

- `OPENAI_API_KEY` in the environment
- OpenAI judge models: `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano` (see `OPENAI_JUDGE_MODELS` in `run_pilot.py`)

**Optional** — only for Gemini judge arms:

- `GOOGLE_API_KEY` in the environment
- Gemini judge models: `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite` (see `GEMINI_JUDGE_MODELS` in `run_pilot.py`)

```bash
claude --version          # must work before running sweeps
ollama pull qwen2.5:7b-instruct-q4_K_M   # only if using echo-small-judge
```

## Setup

```bash
cd experiment
python3 -m venv .venv
source .venv/bin/activate

pip install "langchain-core>=0.3,<0.4" "langchain>=0.3,<0.4" "langchain-ollama>=0.2,<0.4" "langchain-openai>=0.3" "langchain-google-genai>=2.0"
```

## Quick start (1 task)

Sanity check — one HumanEval task, two arms:

```bash
python run_pilot.py --n-tasks 1 --start 0 --arms haiku-only,echo-oracle
```

You should see per-arm pass/fail lines, an aggregate JSON summary, and a new file:

`results/<UTC-timestamp>_n1.jsonl`

## Harder slice (like published sweeps)

Tasks **100–163** (64 tasks), same range the team used for main results:

```bash
python run_pilot.py --start 100 --n-tasks 64
```

Default runs all arms. Expect long runtime, Claude CLI usage, and provider API usage for the judge arms. Confirm the 1-task run first.

Subset of arms:

```bash
python run_pilot.py --start 100 --n-tasks 5 --arms haiku-only,sonnet-only,echo-small-judge
```

Compare OpenAI judges:

```bash
python run_pilot.py --start 100 --n-tasks 10 --arms haiku-only,sonnet-only,echo-judge-openai,echo-judge-openai-gpt-5.4,echo-judge-openai-gpt-5.4-mini,echo-judge-openai-gpt-5.4-nano
```

Compare Gemini judges:

```bash
python run_pilot.py --start 100 --n-tasks 10 --arms haiku-only,sonnet-only,echo-judge-gemini-pro,echo-judge-gemini-flash,echo-judge-gemini-flash-lite
```

Compare all provider judges:

```bash
python run_pilot.py --start 100 --n-tasks 10 --arms echo-judge-openai,echo-judge-openai-gpt-5.4,echo-judge-openai-gpt-5.4-mini,echo-judge-openai-gpt-5.4-nano,echo-judge-gemini-pro,echo-judge-gemini-flash,echo-judge-gemini-flash-lite
```

## CLI reference

| Flag | Default | Meaning |
|------|---------|---------|
| `--n-tasks N` | `1` | Number of tasks to run |
| `--start K` | `0` | Skip first K tasks in HumanEval (e.g. `100` for harder half) |
| `--arms a,b,...` | all arms | Comma-separated strategy names |

### Arms

| Arm | Description |
|-----|-------------|
| `haiku-only` | One Haiku call (cheap baseline) |
| `sonnet-only` | One Sonnet call (quality baseline) |
| `echo-lexical` | Two Haiku personas; escalate if text matches poorly |
| `echo-ast` | Two Haiku personas; escalate if AST structure differs |
| `echo-judge` | Two Haiku personas; Haiku judges equivalence |
| `echo-small-judge` | Two Haiku personas; local Qwen 7B judge (Ollama) |
| `echo-judge-openai` | Two Haiku personas; GPT-5.5 judges equivalence via OpenAI |
| `echo-judge-openai-gpt-5.4` | Two Haiku personas; GPT-5.4 judges equivalence via OpenAI |
| `echo-judge-openai-gpt-5.4-mini` | Two Haiku personas; GPT-5.4 mini judges equivalence via OpenAI |
| `echo-judge-openai-gpt-5.4-nano` | Two Haiku personas; GPT-5.4 nano judges equivalence via OpenAI |
| `echo-judge-gemini-pro` | Two Haiku personas; Gemini 2.5 Pro judges equivalence |
| `echo-judge-gemini-flash` | Two Haiku personas; Gemini 2.5 Flash judges equivalence |
| `echo-judge-gemini-flash-lite` | Two Haiku personas; Gemini 2.5 Flash-Lite judges equivalence |
| `echo-oracle` | Two Haiku personas; escalate only if both fail tests (upper bound, not deployable) |

## Output format

Each sweep writes `results/<timestamp>_n<tasks>.jsonl`. One line per run:

```json
{"task_id": "HumanEval/100", "arm": "haiku-only", "passed": true, "detail": "passed", "wall_seconds": 7.2, "sub_calls": 1}
```

- **`passed`** — automated HumanEval tests succeeded
- **`sub_calls`** — model calls for that task (Echo escalate → typically 3)

Aggregate metrics:

| Metric | Meaning |
|--------|---------|
| `n` | Number of tasks run for that arm |
| `pass_rate` | Fraction of final selected implementations that passed HumanEval tests |
| `escalation_rate` | Fraction of tasks where the arm called Sonnet after the cheap pair/judge disagreed |
| `mean_wall_seconds` | Average elapsed seconds per task for that arm |
| `total_sub_calls` | Total counted model calls for that arm |
| `mean_sub_calls` | Average counted model calls per task |
| `failures` | Number of tasks that failed tests or hit provider/harness errors |
| `top_failure_details` | Most common failure reasons, useful for spotting provider limits, auth errors, syntax errors, and timeouts |

For provider judge arms, `3 calls` means two Haiku candidates plus one judge call. `4 calls` means the judge disagreed and the arm escalated to Sonnet.

**Cost units** (see [`cost_units.py`](cost_units.py)): Haiku persona call = 1.0; Sonnet persona = 3.0; OpenAI/Gemini judge calls priced from list API rates; local Ollama judge = 0. Run `python scripts/analyze_sweep.py results/<file>.jsonl` for totals.

## View existing results (no run)

Committed JSONL files are under `results/`. Summary and interpretation: [`results/README.md`](results/README.md).

Example with `jq` (optional):

```bash
jq -s 'group_by(.arm) | map({arm: .[0].arm, n: length, passed: (map(select(.passed)) | length)})' results/20260519T085620Z_n64.jsonl
```

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| `claude: command not found` | Install [Claude Code](https://docs.anthropic.com/en/docs/claude-code) and ensure `claude` is on your `PATH` |
| `claude --print failed` | Log in to Claude Code; confirm Max/subscription access |
| `echo-small-judge requires langchain-ollama` | `pip install "langchain-ollama>=0.2,<0.4"` or omit that arm |
| `echo-judge-openai requires langchain-openai` | `pip install "langchain-openai>=0.3"` and set `OPENAI_API_KEY` |
| `echo-judge-gemini requires langchain-google-genai` | `pip install "langchain-google-genai>=2.0"` and set `GOOGLE_API_KEY` |
| Ollama connection errors | Start Ollama; `ollama pull qwen2.5:7b-instruct-q4_K_M`; check `SMALL_JUDGE_BASE_URL` in `run_pilot.py` |
| Very slow runs | Expected — each call spawns `claude --print` (~seconds overhead per call) |

## BBH

Big-Bench Hard loader, scoring, and BBH-specific Echo arms — **no Claude needed** for local tests. Model sweeps require the Claude Code CLI on a machine with Max auth.

**Harness note:** `chat_claude_code.py` passes `--setting-sources ""` so `claude --print` does not inherit project `CLAUDE.md` or hooks (see #1305). Without this, BBH accuracy numbers are invalid.

```bash
cd experiment
source .venv/bin/activate
pip install "datasets>=2.14"

# Pre-flight (tests + arm wiring; no model calls)
python scripts/validate_harness.py

# Unit tests
python -m unittest tests.test_bbh_scoring tests.test_bbh_arms -v

# Print sample tasks
python scripts/inspect_bbh.py --n 1

# Analyze any sweep JSONL (HumanEval or BBH)
python scripts/analyze_sweep.py results/20260519T085620Z_n64.jsonl
```

**Model sweeps:** see [`scripts/RUN_FOR_NICK.md`](scripts/RUN_FOR_NICK.md) — copy-paste commands for Nick.

```bash
python scripts/run_bbh_pilot.py \
  --subtasks logical_deduction_three_objects,causal_judgement,date_understanding \
  --n-per-subtask 5

# Long sweeps (auto-resume on usage window)
./scripts/run_bbh_resumable.sh
```

Canonical results and interpretation: [`results/README.md`](results/README.md#bbh--claude-baselines-clean-harness-1305).

Files: `benchmarks/bbh.py`, `benchmarks/bbh_arms.py`, `scripts/inspect_bbh.py`, `scripts/run_bbh_pilot.py`, `scripts/analyze_sweep.py`, `scripts/run_bbh_resumable.sh`.

Pilot subtasks: `logical_deduction_three_objects`, `causal_judgement`, `date_understanding` (confirm with team).

## MMLU-Pro

Multiple-choice reasoning benchmark (14 domains, up to 10 options per question). Reuses the same MCQ Echo arms as BBH.

```bash
# Pre-flight + unit tests (includes MMLU-Pro loader)
python scripts/validate_harness.py
python -m unittest tests.test_mmlu_pro_scoring -v

# Inspect tasks (downloads HF dataset on first run)
python scripts/inspect_mmlu_pro.py --n 1

# Pilot sweep: 5 categories × 5 questions = 25 tasks (default arms)
python scripts/run_mmlu_pro_pilot.py --n-per-category 5

# Canonical first run (match BBH scale): 5 × 25 = 125 tasks
python scripts/run_mmlu_pro_pilot.py \
  --categories physics,math,law,chemistry,philosophy \
  --n-per-category 25 \
  --arms haiku-only,sonnet-only,echo-judge,echo-oracle

# Long sweeps (auto-resume on usage window)
./scripts/run_mmlu_pro_resumable.sh \
  --categories physics,math,law,chemistry,philosophy \
  --n-per-category 25 \
  --arms haiku-only,sonnet-only,echo-judge,echo-oracle
```

Pilot categories: `physics`, `math`, `law`, `chemistry`, `philosophy`. Data: [`TIGER-Lab/MMLU-Pro`](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) (`test` split).

Files: `benchmarks/mmlu_pro.py`, `scripts/inspect_mmlu_pro.py`, `scripts/run_mmlu_pro_pilot.py`.

## Layout

```
experiment/
  run_pilot.py          # HumanEval sweep runner + routing arms
  dataset.py            # HumanEval loader
  benchmarks/bbh.py     # BBH loader + scoring
  benchmarks/bbh_arms.py # BBH Echo arms (MCQ personas + oracle)
  benchmarks/mmlu_pro.py # MMLU-Pro loader + scoring
  scripts/              # inspect/run pilots, analyze_sweep, RUN_FOR_NICK
  tests/                # test_bbh_*, test_mmlu_pro_scoring
  chat_claude_code.py   # LangChain wrapper around `claude --print`
  results/              # JSONL sweep logs (committed)
  pyproject.toml
```

## Related

- Project overview: [`../README.md`](../README.md)
- Collaborators: [`../COLLABORATORS.md`](../COLLABORATORS.md)
