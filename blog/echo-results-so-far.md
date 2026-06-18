---
title: "Echo: results so far"
published: true
description: "Routing LLM requests cheaply without training a router — and the measurement bug that nearly fooled us. A cross-family local judge reaches 94% of the oracle's routing quality at ~29% lower cost than always using the big model."
tags: llm, ml, costoptimization, research
canonical_url: https://enspyr.co/blog/echo-results-so-far
---

# Echo: results so far

*Routing LLM requests cheaply without training a router — and the measurement bug that nearly fooled us.*

By [Nick Meinhold](https://enspyr.co/about#nicholas-meinhold), [Robin Langer](https://enspyr.co/about#robin-langer), [Meghana Ganapa](https://enspyr.co/about#meghana-ganapa), and [Adarsha Aryal](https://enspyr.co/about#adarsha-aryal) · 10 June 2026

> **TL;DR**
> - **The idea:** instead of training a classifier to route easy tasks to a cheap model and hard ones to an expensive model, call the *cheap* model twice with two different personas. If the answers agree, keep the cheap one; if they disagree, escalate. No classifier, no labels.
> - **What works:** on HumanEval's hard slice, a cross-family local judge (Qwen 2.5 7B) reaches **94% of the oracle's routing quality**, is **~29% cheaper than always using Sonnet**, and matches its pass rate.
> - **The honest part:** our first reasoning-benchmark numbers were a *harness bug*, not a result. Finding and fixing it is half this update.

## Most LLM apps overpay

A trivial "reverse this string" and a gnarly multi-step refactor usually go to the same expensive endpoint. The standard fix is a **router**: a learned classifier that decides "easy → cheap model, hard → expensive." RouteLLM, FrugalGPT, Hybrid LLM and AutoMix all do versions of this, and they work — but every one needs labelled training data for *your* task domain. That label-collection step is the adoption bottleneck. You can't drop a trained router into a new product on day one.

We wanted to know how much of the benefit you can get with *none* of the training.

## The idea: let the cheap model check itself

Here's the whole move. Call the cheap model twice on the same task, with two different persona prompts — one a "careful, methodical programmer," the other a "pragmatic senior engineer who writes the simplest thing that works." Then:

- If the two answers **agree**, framing didn't matter — the task was easy. Keep the cheap answer.
- If they **disagree**, the task is sitting on a decision boundary where small perturbations change the output. That's your difficulty signal. Escalate to the expensive model.

The difficulty signal is manufactured at inference time, for free, out of the model's own (in)consistency. It's a reframe of self-consistency (Wang et al., 2022) — but used as a *cost* signal instead of an accuracy one. The arithmetic clears one bar: two cheap calls must cost less than one expensive call. At current Claude pricing, Haiku-twice beats Sonnet-once while the tier gap stays ~3×. It does.

## The catch nobody warns you about: what does "agree" mean?

"If the two answers agree" sounds simple until you implement `agree(a, b)` for code. Two programs can be character-identical, or solve the same problem with a loop vs a comprehension, a dict vs a class, different names, different decomposition — all "agreement" in the sense that matters (same behaviour) and "disagreement" in the sense that's easy to measure (different text).

<details>
<summary>The ladder of agreement signals we tested</summary>

| Signal | What it checks | Extra cost |
|---|---|---|
| **lexical** | normalized text match | free |
| **AST** | Python syntax-tree structure match | free |
| **judge** | a third Haiku call: "are these equivalent?" | +1 cheap call |
| **small-judge** | same question, asked to a *local* Qwen 2.5 7B | ~free (local compute) |
| **oracle** | ground truth: do the answers pass the hidden tests? | not deployable |

The oracle isn't a real strategy — it cheats by looking at test results you'd never have in production. We include it to mark the ceiling. The research question is how close a *deployable* signal gets to it.

</details>

## Results: HumanEval hard slice

All arms over HumanEval 100–163 (the first hundred tasks are too easy to separate arms). Cost in units where **Haiku = 1**, **Sonnet = 3** per call. "Oracle alignment" = how often the signal escalates on exactly the tasks the oracle would.

| Arm | Pass | Escalations | Oracle align | Cost |
|---|---|---|---|---|
| haiku-only | 63/64 | — | — | 64 |
| sonnet-only | 63/64 | — | — | 192 |
| echo-lexical | 64/64 | 55/64 | 16% | 293 |
| echo-ast | 62/64 | 54/64 | 17% | 290 |
| echo-judge (Haiku) | 61/64 | 11/64 | 81% | 225 |
| **echo-small-judge (Qwen 7B)** | **62/64** | **3/64** | **94%** | **137** |
| echo-oracle (ceiling) | 64/64 | 1/64 | — | 131 |

Three things fall out:

- **The cost thesis holds with a deployable signal.** `echo-small-judge` lands within ~5% of the oracle's cost floor (137 vs 131), ~29% cheaper than always-Sonnet, with a pass rate statistically equal to Sonnet on this slice.
- **Free signals are noise here.** Lexical and AST escalate ~85% of the time — they cost *more* than just using Sonnet.
- **The surprise: a cross-family *local* judge beats the same-family one.** Qwen 7B (a different model family, running locally) tracks the oracle better than a Haiku judge (94% vs 81%) with a third the escalations. Independence beats capability for this job.

> Why would a smaller, cheaper, local model judge agreement *better* than Haiku? Our read: a same-family judge shares Haiku's blind spots — it agrees that two Haiku answers match precisely when Haiku is consistently wrong. A different family disagrees out of genuine independence. That's the whole thesis in miniature.

<details>
<summary>Methodology & caveats</summary>

HumanEval only; n = 64; single hard slice. The local-judge mean wall time is ~86s/task on a CPU-only ARM box — that's infrastructure latency, not API cost, and would drop sharply on a GPU. Earlier sweeps surfaced (and fixed) two output-parser bugs in the code harness before these numbers stabilised — a recurring theme (see below). Full per-task JSONL logs and sweep history live in the repo's `experiment/results/`.

</details>

## The plot twist: our first reasoning numbers were a lie

HumanEval is code. To claim Echo generalises, we need reasoning benchmarks — so we ported the harness to BBH (Big-Bench Hard). The n=10 pilot came back looking like this:

| Arm | Pass rate |
|---|---|
| haiku-only | 0.14 |
| sonnet-only | 0.14 |
| echo-judge | 0.12 |

Low, and *suspiciously flat*. The red flag: on reasoning tasks, Sonnet should clearly beat Haiku. Them tying at 0.14 doesn't say "Echo doesn't work" — it says **the measuring instrument is broken.**

So we put the BBH scoring code through an adversarial review — three AI reviewers from different model families, each trying to break it. They found the answer parser was silently corrupting results in *both* directions.

<details>
<summary>The three bugs (and why a silent one is the worst kind)</summary>

- **Tail truncation.** The parser only looked at the last 5 lines of output before searching for the answer. A model that states "The answer is C" early and then keeps explaining had its answer fall outside the window — scored as unparseable, counted as a failure.
- **Case-folding over-match.** A case-insensitive letter pattern matched the first letter of the *next word*: "the answer is **s**traightforward" was parsed as answer "S". This one is bidirectional — it manufactures both false failures (wrong letter) and false passes (lucky letter), silently, because a bogus-but-valid letter is accepted without complaint.
- **Cross-family recency.** "Answer: A … therefore the answer is C" returned A — an early scratch line beat the final answer because the two were caught by different patterns.

All three are fixed, each with a regression test; the scoring suite is green. The lesson: *a measurement apparatus with a silent, bidirectional bias is worse than a noisy one.* The flat 0.14 wasn't just low — it was untrustworthy in an unknown direction.

</details>

**In progress:** the fix is in; the proof is a re-run showing the pass rates *separate*. We won't scale to the full sweep until they do — no point reproducing a (now-fixed) bug at scale.

## What's next

- **Confirm the BBH fix** — re-run the pilot; Sonnet should now beat Haiku.
- **Cross-family judge sweep at n=30** — does the Qwen-beats-Haiku surprise from code hold on reasoning? We've added OpenAI and Gemini judges to widen the matrix (same-family vs cross-family × small vs large).
- **Full BBH sweep, then MMLU-Pro** — statistically meaningful Pareto numbers across benchmarks.
- **The real test** — a heterogeneous real-world task (PR review with merge decisions), where task difficulty actually varies.

If Echo lands on or above a trained router's cost/accuracy frontier with *zero* training data, that's the result worth publishing. If it collapses to "Haiku with extra steps," that's a clean negative — also worth publishing.

## The team

- **[Nick Meinhold](https://enspyr.co/about#nicholas-meinhold)** · Director & Tech Lead — originated the self-consistency-as-cost-signal idea and the experiment design.
- **[Robin Langer](https://enspyr.co/about#robin-langer)** · Agentic Engineer — agentic engineering and research; co-founder of Sawasdee Cellars. ([Semantic Scholar](https://www.semanticscholar.org/author/Robin-Langer/39449928) · [Hugging Face](https://huggingface.co/RobBobin))
- **[Meghana Ganapa](https://enspyr.co/about#meghana-ganapa)** · Agentic AI Engineer — ML/NLP across healthcare and legal domains. On Echo: cross-family judge arms and BBH scoring.
- **[Adarsha Aryal](https://enspyr.co/about#adarsha-aryal)** · Agentic Engineer — Master of Data Science, Monash. On Echo: judge-branch integration, the BBH sweep harness, and run tooling.

---

*Echo is open research at [github.com/enspyrco/echo](https://github.com/enspyrco/echo). Background: Wang et al. 2022 (Self-Consistency), Ong et al. 2024 (RouteLLM), Chen et al. 2023 (FrugalGPT), Ding et al. 2024 (Hybrid LLM). Earlier post: [Echo: routing LLM requests cheaply without training a router](https://enspyr.co/blog/echo-cheap-routing-without-a-router).*
