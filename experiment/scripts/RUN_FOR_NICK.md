# Commands for Nick (model sweeps)

Adarsha does not have Claude Max/Pro for `claude login`. Run these on the server when asked.

**Prereqs:** `claude auth status` shows logged in; repo at `~/echo/experiment`; venv active.

```bash
cd ~/echo/experiment
source .venv/bin/activate
git pull
pip install "datasets>=2.14"   # BBH only
```

---

## HumanEval sanity (1 task)

```bash
python run_pilot.py --n-tasks 1 --start 100 --arms haiku-only,echo-oracle
```

---

## BBH pilot (recommended first BBH run)

3 subtasks × 5 questions × 6 arms = 90 model calls. Adjust as needed.

```bash
python scripts/run_bbh_pilot.py \
  --subtasks logical_deduction_three_objects,causal_judgement,date_understanding \
  --n-per-subtask 5 \
  --arms haiku-only,sonnet-only,echo-lexical,echo-judge,echo-small-judge,echo-oracle
```

Analyze:

```bash
python scripts/analyze_sweep.py results/<timestamp>_bbh_n15.jsonl
```

Commit the JSONL under `experiment/results/` or send Adarsha the path.

---

## Notes

- BBH uses `benchmarks/bbh_arms.py` (MCQ personas + gold-label oracle), not HumanEval code prompts.
- `echo-small-judge` needs Ollama + `qwen2.5:7b-instruct-q4_K_M` on the server.
