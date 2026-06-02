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

```bash
claude --version          # must work before running sweeps
ollama pull qwen2.5:7b-instruct-q4_K_M   # only if using echo-small-judge
```

## Setup

```bash
cd experiment
python3 -m venv .venv
source .venv/bin/activate

pip install "langchain-core>=0.3,<0.4" "langchain>=0.3,<0.4" "langchain-ollama>=0.2,<0.4"
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

Default runs **all 7 arms** → 448 model calls. Expect long runtime and Claude CLI usage. Confirm the 1-task run first.

Subset of arms:

```bash
python run_pilot.py --start 100 --n-tasks 5 --arms haiku-only,sonnet-only,echo-small-judge
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
| `echo-oracle` | Two Haiku personas; escalate only if both fail tests (upper bound, not deployable) |

## Output format

Each sweep writes `results/<timestamp>_n<tasks>.jsonl`. One line per run:

```json
{"task_id": "HumanEval/100", "arm": "haiku-only", "passed": true, "detail": "passed", "wall_seconds": 7.2, "sub_calls": 1}
```

- **`passed`** — automated HumanEval tests succeeded
- **`sub_calls`** — model calls for that task (Echo escalate → typically 3)

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
| Ollama connection errors | Start Ollama; `ollama pull qwen2.5:7b-instruct-q4_K_M`; check `SMALL_JUDGE_BASE_URL` in `run_pilot.py` |
| Very slow runs | Expected — each call spawns `claude --print` (~seconds overhead per call) |

## BBH (ready for Nick to run)

Big-Bench Hard loader, scoring, and BBH-specific Echo arms — **no Claude needed** for local tests.

```bash
cd experiment
source .venv/bin/activate
pip install "datasets>=2.14"

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
```

Files: `benchmarks/bbh.py`, `benchmarks/bbh_arms.py`, `scripts/inspect_bbh.py`, `scripts/run_bbh_pilot.py`, `scripts/analyze_sweep.py`.

Pilot subtasks: `logical_deduction_three_objects`, `causal_judgement`, `date_understanding` (confirm with team).

## Layout

```
experiment/
  run_pilot.py          # HumanEval sweep runner + routing arms
  dataset.py            # HumanEval loader
  benchmarks/bbh.py     # BBH loader + scoring
  benchmarks/bbh_arms.py # BBH Echo arms (MCQ personas + oracle)
  scripts/              # inspect_bbh, run_bbh_pilot, analyze_sweep, RUN_FOR_NICK
  tests/                # test_bbh_scoring, test_bbh_arms
  chat_claude_code.py   # LangChain wrapper around `claude --print`
  results/              # JSONL sweep logs (committed)
  pyproject.toml
```

## Related

- Project overview: [`../README.md`](../README.md)
- Collaborators: [`../COLLABORATORS.md`](../COLLABORATORS.md)
