---
title: "Echo: routing LLM requests cheaply without training a router"
published: true
description: "Call the cheap model twice with different personas. If the two answers agree, keep the cheap answer; if they disagree, escalate. No classifier, no labels. The hard part turns out to be measuring agreement, and the winning signal is a surprise."
tags: llm, ml, costoptimization, research
---

# Echo: routing LLM requests cheaply without training a router

Most production LLM apps overpay, because they send every request to the same model. A trivial "reverse this string" and a gnarly multi-step refactor both go to the same expensive endpoint.

The standard fix is a **router**: a learned classifier that looks at each request and decides "easy, use the cheap model" or "hard, use the expensive one." RouteLLM, FrugalGPT, Hybrid LLM, AutoMix all do versions of this, and they work. But every one of them needs labelled training data for your task domain. That label-collection step is the adoption bottleneck. You can't drop a trained router into a new product on day one.

We wanted to know if you can get most of the benefit with none of the training. This post is what we found.

## The idea: let the cheap model check itself

Here is the whole technique in one move. Instead of training a classifier to predict difficulty, **call the cheap model twice on the same task, with two different persona prompts**. One call is told it's a "careful, methodical programmer who values correctness." The other is told it's a "pragmatic senior engineer who writes the simplest code that works."

Then:

- If the two answers **agree**, the task was easy enough that framing didn't matter. Keep the cheap answer.
- If they **disagree**, the task is sitting on some decision boundary where small perturbations change the output. That's your difficulty signal. Escalate to the expensive model.

No classifier. No labels. No calibration set. The difficulty signal is manufactured at inference time, for free, out of the model's own (in)consistency.

This isn't a brand-new mechanism. Self-consistency (Wang et al., 2022) has been used for years to make models *more accurate* by sampling multiple times and voting. Our reframe is to use the same disagreement as a *cost* signal instead of an accuracy signal. That changes which knob you turn: you're not voting to improve the answer, you're escalating only when voting fails.

The arithmetic has to clear one bar: two cheap calls must cost less than one expensive call. At current Claude pricing, Haiku-twice is still cheaper than Sonnet-once as long as the tier gap is roughly 3x. It is. So the method is economically live.

## The catch nobody warns you about: what does "agree" mean?

This is where the experiment actually lives. "If the two answers agree" sounds simple until you try to implement `agree(a, b)` for code generation. Two programs can be character-identical, or they can solve the same problem with a loop versus a comprehension, a dict versus a class, different variable names, different helper decomposition. All of those are "agreement" in the sense that matters (same behaviour) and "disagreement" in the sense that's easy to measure (different text).

So we built a ladder of agreement signals, from dumb-and-free to smart-and-costly, and measured how well each one tracks the truth:

| Signal | What it checks | Extra cost |
|---|---|---|
| **lexical** | normalized text match (strip whitespace/comments) | free |
| **AST** | Python abstract-syntax-tree structure match | free |
| **judge** | a third Haiku call asked "are these equivalent?" | +1 cheap call |
| **small-judge** | same question, asked to a *local* Qwen 2.5 7B | ~free (local compute) |
| **oracle** | ground-truth: do the answers pass the hidden tests? | not deployable |

The oracle isn't a real strategy. It cheats by looking at the test results, which you'd never have in production. We include it to mark the ceiling: it's the best any agreement signal could possibly do. The research question is how close a *deployable* signal can get to that ceiling.

## Results

We ran all seven arms over HumanEval 100-163 (the harder slice of the standard code-generation benchmark; the first hundred tasks are too easy to separate the arms). Cost is in units where Haiku = 1 and Sonnet = 3.

| Arm | Pass | Escalations | Oracle alignment | Cost (units) |
|---|---|---|---|---|
| haiku-only | 63/64 | -- | -- | 64 |
| sonnet-only | 63/64 | -- | -- | 192 |
| echo-lexical | 64/64 | 55/64 | 16% | 293 |
| echo-ast | 62/64 | 54/64 | 17% | 290 |
| echo-judge | 61/64 | 11/64 | 81% | 225 |
| **echo-small-judge** | **62/64** | **3/64** | **94%** | **~137** |
| echo-oracle | 64/64 | 1/64 | (ceiling) | 131 |

Two things jump out.

### 1. The cost thesis holds with a deployable signal

`echo-small-judge` lands at **137 units, within 5% of the oracle's 131-unit floor**, at 94% oracle alignment and a pass rate statistically equal to sonnet-only. That's **29% cheaper than always using Sonnet, at the same accuracy, with zero training data.**

This is the result the whole project rests on. The oracle proved a cheap-routing frontier *exists*; the small-judge proves you can *reach* it without ground truth.

Notice also how badly the free signals do. Lexical and AST escalate on 85% of tasks and end up *more expensive than just using Sonnet for everything* (293 and 290 units versus 192). They're not signals, they're noise. We'll come back to why.

### 2. The surprise: a weaker, different-family judge beats Claude judging Claude

Look at the two judge rows. The Haiku judge (`echo-judge`) hits 81% oracle alignment. The local Qwen 7B judge (`echo-small-judge`) hits **94%**. A smaller, weaker, cheaper, different-family model makes *better* routing decisions than Claude evaluating Claude.

The likely mechanism is independence. When Haiku judges two Haiku answers, the judge shares the candidates' blind spots: if both candidates are confidently wrong in the same way, a same-family judge is liable to confidently agree that they're equivalent (and skip the escalation that would have caught the error). A different-family model has *different* blind spots, so its mistakes are uncorrelated with the candidates' mistakes. That uncorrelated error is exactly what you want in an adjudicator.

This is a diversity argument, not a capability argument. You don't want your judge to be smart. You want it to be *wrong in different places* than the thing it's judging.

## Why the free signals failed

We expected lexical/AST to be weak. We didn't expect them to be pure noise. So we dug into the 51 tasks where the lexical signal said "disagree" but the oracle said "both answers were actually fine" (false escalations, the expensive kind of mistake).

The hypothesis going in was that these would cluster: maybe string-manipulation tasks, or recursion, some structural category you could pre-filter. They don't cluster at all. Sorted by topic, the 51 false escalations break down as 19 string tasks, 21 list/array tasks, 7 math tasks, 4 other. That's just the underlying topic distribution. **Uniform.**

The takeaway: for code generation, the natural implementation entropy (loop vs comprehension, this idiom vs that one) completely swamps any topical structure. There is no clever surface-level prefilter waiting to be discovered. If you want a signal, you have to actually reason about semantic equivalence, which means you need a model in the loop. That's why the judge arms exist and the free arms don't work.

## Honest caveats

- **One benchmark.** This is HumanEval only. Code generation has unusually high implementation entropy, which is exactly why surface signals fail here; the picture might look different on multiple-choice reasoning (BBH, MMLU-Pro), where "agreement" is easier to define. Generality is the next thing to test.
- **n = 64.** The pass-rate differences (61 vs 62 vs 63 of 64) are well within noise. We're claiming the arms are *statistically equal* on accuracy, and that the *cost* difference is the real result, not that one arm is more correct.
- **Latency.** The local judge ran at 86 seconds mean wall-time per task on a CPU-only ARM box. That's an infrastructure artifact (a GPU erases it); the cost result is what the thesis rests on, and cost is unaffected by how slow the free local call is.
- **Pricing risk.** The whole economic case assumes the tier gap stays around 3x. We frame the contribution as a *technique* whose economics depend on the tier gap, not as a claim about any specific price.

## Where this goes

The clean version of the finding: **you can route LLM requests near-optimally without training a router, by checking the cheap model against itself, as long as the agreement check is done by an independent model rather than the same family.**

Next on the list: test whether the signal-quality ordering holds on non-code benchmarks, probe whether *semantic* persona axes (edge-case-hunter vs happy-path-implementer) beat the *stylistic* ones we used, and push on whether an even cheaper independent judge can hold the 94% alignment. If the cross-family effect generalizes, the practical advice writes itself: don't reach for a bigger judge, reach for a different one.

*Echo is an open research project. The harness, every per-call result as replayable JSONL, and the full four-sweep writeup are in the repo.*
