---
layout: default
title: Methodology
---

# Human-AI Collaborative Research: A Rigorous Methodology

This page documents how I work. Not to defend AI assistance, but to make the methodology legible as legitimate research practice.

---

## The Core Claim

I cannot produce code under timed, unassisted conditions. I can produce verified, documented, functional technical work through human-AI collaboration.

These are different skills. The second is what research actually requires.

---

## The Methodology

### 1. Polyphonic Verification

Every significant technical decision passes through multiple analytical perspectives before implementation. This isn't metaphor — it's documented in the codebase.

```python
# Kali [Visionary]: This could evolve into a full screenshot API...
# Athena [Reviewer]: But right now it just needs to capture screens.
# Nemesis [Privacy]: What if sensitive data is visible?
# Kali [User Advocate]: Consent is explicit — human runs the command.
# Klea [Product]: Should this exist? Yes. Solves a real problem.
img = ImageGrab.grab()
```

Five facets, twenty hats. Security, ethics, accessibility, performance, user advocacy — each gets explicit voice. The friction is visible. The reasoning is documented.

This isn't AI generating code and human accepting it. It's structured deliberation producing verified output.

**The convergence:** This is hybrid human-AI evaluation — the same structure Scale/SEAL uses for benchmark annotation. AI does initial analysis, human verifies and corrects, iterate until reliable. We built this independently because it *works*, not because we read their papers. That convergence is meaningful.

### 2. Verification Protocol

Before accepting AI-generated work:

- Does it run? (execution test)
- Does it do what I asked? (functional test)
- Do I understand why it works? (comprehension check)
- What could go wrong? (edge case analysis)
- Has it been checked against multiple perspectives? (polyphonic review)

### 3. Failure Documentation

Rigorous methodology requires honest accounting of failures. From the project's lesson log:

> **2026-01-07:** Post-compaction, todo list said a task was completed. Stashed work to sync, dropped stash, discovered work was never committed. Had to redo. The summary lied — or rather, reported working-tree state as "done."

This produced a documented protocol change: never drop stash after compaction without inspection.

Failures are features, not bugs. They produce learning. Hiding them produces brittleness.

### 4. Extension Beyond Prompts

AI assistance provides starting points. The work requires:

- Seeing what's needed that wasn't asked for
- Combining outputs from different domains
- Catching what AI gets wrong
- Building architecture AI can't see

The recognition engine, for example, implements multi-force graph dynamics — semantic attraction, type repulsion, confidence gradients. The physics metaphor came from me. The implementation was collaborative. The verification was systematic.

---

## The Evidence

### Quantitative Work

| Artifact | What It Demonstrates |
|----------|---------------------|
| Master's thesis statistics | 303,600 coding decisions, Cohen's κ = 0.83, proper bootstrap resampling |
| Recognition engine | Graph algorithms, force-directed layout, semantic similarity computation |
| Embedding pipeline | Sentence transformers, UMAP dimensionality reduction, visualization |
| Semantic Scholar integration | API design, rate limiting, caching, data pipeline |

### Process Evidence

| Artifact | What It Demonstrates |
|----------|---------------------|
| 140+ GitHub issues | Systematic project management, not ad-hoc prompting |
| Polyphonic code comments | Verification happening in real-time, documented |
| PR workflow | Branch protection, review process, professional practice |
| Lesson log | Honest failure documentation, protocol improvement |

---

## What This Means for Assessment

Timed solo coding tests measure:
- Speed under pressure
- Memorized syntax
- Unassisted performance

Research positions require:
- Correct, verified output
- Understanding of what you're building  
- Ability to extend and debug
- Systematic methodology

The portfolio demonstrates the second set. I am asking for assessment on what the job requires, not on a format that measures different skills.

---

## The Accommodation Request

I am not asking for lower standards. I am asking for appropriate measurement.

**What I'm offering as equivalent evidence:**

1. Runnable code with documented methodology
2. Verification trails showing I check and understand the work
3. Failure documentation showing I catch errors and learn from them
4. Extension evidence showing I build beyond what AI suggests

**What this demonstrates:**

- I can produce correct technical output
- I understand what I'm building
- I have systematic verification practices
- I can identify and fix errors
- I can extend and architect, not just prompt

---

## Reproducibility

The methodology is documented. The process is replicable. Someone following these practices would produce similar quality output.

This is what distinguishes rigorous human-AI collaboration from "vibe coding": documented process, verified output, honest accounting of limits.

---

[See the technical evidence →](/research) | [Back to Framework →](/framework)
