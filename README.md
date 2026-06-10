# Echo

**Self-consistency as a cost-control mechanism for LLM routing.**

## The idea in one paragraph

Most production LLM apps overpay because they send every request to the same model. The standard fix is a *router*: a learned classifier that decides "easy task, use the cheap model; hard task, use the expensive one." Trained routers work but need labelled training data per task domain, which is the bottleneck for adoption.

Echo proposes a simpler primitive. Call the cheap model twice with two different persona prompts. If the two answers agree, accept the cheap answer. If they disagree, escalate to the expensive model. No classifier, no training data, no calibration. The technique exists in the literature for accuracy gains (self-consistency, Wang et al. 2022), but has not been studied as a *cost* tool. At current Claude pricing, calling Haiku twice ($2/M input) is still cheaper than calling Sonnet once ($3/M input), so the arithmetic works as long as the tier gap is roughly 3x.

## Why this might be a paper

1. **Calibration-free routing.** Every existing router (RouteLLM, FrugalGPT, Hybrid LLM, AutoMix) needs training data. This one needs none.
2. **It has a prior empirical foundation.** A previous experiment we ran (cohort comparison across 10 PR reviews) found same-family-different-persona divergence sits around 4 percentage points, while different-family-same-prompt divergence is 8 to 19 points. That gap is what licenses using persona perturbation as a difficulty signal.
3. **Reframes a known technique.** Self-consistency was proposed to make models more accurate. We argue it can instead make them cheaper, which changes which knobs you turn.
4. **It generalises.** Works across any provider and any model pair where there is a meaningful price tier.

## What the experiment looks like

**Routing arms to compare:**
1. Haiku-only (cheap baseline)
2. Sonnet-only (quality baseline)
3. Trained router (RouteLLM as comparator)
4. **Echo: Haiku-twice-with-personas, escalate on disagreement** (the contribution)
5. Cascade with confidence (cascadeflow as comparator)

**Benchmarks:**
- HumanEval+ (code generation)
- BBH or MMLU-Pro (reasoning)
- A real-world heterogeneous task with ground-truth outcomes (PR review with merge decisions)

**Primary metric:** Pareto frontier of cost-per-task vs accuracy. Success criterion: Echo lands on or above the trained-router frontier without using any training data.

**Ablations:**
- Persona-pair selection sensitivity
- Two-way vs three-way self-consistency
- What happens at the top tier (Sonnet-twice escalating to Opus)
- Behaviour as tier gap shrinks (Sonnet vs Opus is only ~1.7x, not 3x)

## Where the work lives

- **Compute:** OCI free-tier ARM (158.179.17.233, host `nick-mel`), behind Caddy. The harness is I/O-bound (API calls), so ARM is fine.
- **Public replication endpoint:** plan to expose a subset of experiments at a public URL so reviewers can re-run live. This is rare in ML papers and tends to build trust.
- **Repo layout (planned):**
  - `harness/` Python orchestrator that runs routing arms against benchmarks
  - `personas/` the prompt-perturbation pairs we test
  - `benchmarks/` dataset adapters
  - `results/` JSONL logs of every API call, replayable
  - `paper/` LaTeX source

## Honest risks

1. **Haiku might confidently agree with itself when wrong.** If the cheap model has consistent blind spots, persona perturbation will not surface them and Echo collapses to "Haiku with extra steps." Week one of running the harness will tell us. A clean negative result here is also publishable.
2. **Persona-pair selection might dominate the result.** Could turn into a tuning paper rather than a clean-method paper.
3. **Anthropic could shift tier pricing** and break the cost arithmetic. We frame the contribution as a *technique* whose economics depend on tier gap, not as a specific price claim.

## Collaborators

See [COLLABORATORS.md](COLLABORATORS.md). Roles still to be assigned; the immediate questions are who owns the harness build, who owns benchmark curation, and who owns the paper draft.

## Status

**HumanEval hard slice (tasks 100–163, n=64) is complete** through sweep 4, including `echo-small-judge`. Headline: local cross-family judge (Qwen 7B) reaches ~94% oracle alignment at ~137 cost units vs sonnet-only ~192, with pass rates statistically equal on this slice.

- **Run locally:** [`experiment/README.md`](experiment/README.md)
- **Results & sweep history:** [`experiment/results/README.md`](experiment/results/README.md)
- **Blog:** [enspyr.co/blog/echo-cheap-routing-without-a-router](https://enspyr.co/blog/echo-cheap-routing-without-a-router)

**BBH:** Judge branches merged on `integrate/judge-branches` (OpenAI/Gemini judge arms, Yes/No scoring fixes, 25 unit tests). Canonical n=30 server sweep pending — see [`experiment/scripts/RUN_FOR_NICK.md`](experiment/scripts/RUN_FOR_NICK.md).

**Next steps:**

1. Merge `integrate/judge-branches` → run canonical BBH sweep on OCI
2. Pareto analysis + update `experiment/results/README.md`
3. MMLU-Pro after BBH
4. Paper draft / venue

## Background reading

- Wang et al. 2022, *Self-Consistency Improves Chain of Thought Reasoning in Language Models*
- Ong et al. 2024, *RouteLLM*
- Chen et al. 2023, *FrugalGPT*
- Ding et al. 2024, *Hybrid LLM*
- AutoMix, cascadeflow (open source comparators)
