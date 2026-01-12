---
layout: default
title: Evidence
---

# The Evidence

Three types of evidence support the framework: statistical, textual, and empirical.

---

## 1. Statistical: Semantic Convergence

We measured semantic similarity across 91 academic papers using SPECTER embeddings (AllenAI's scientific paper model).

| Comparison | Similarity |
|------------|------------|
| AI Safety / Trauma / Education (internal) | 0.746 |
| Unrelated fields (Astrophysics, Physics, Math, Biology) | 0.662 |
| **Difference** | **8.4 percentage points** |

**Statistical significance:** p < 0.0001 (permutation test, n = 10,000)
**Effect size:** Cohen's d = 0.78 (large)

Different research communities, working independently, arrived at structurally similar descriptions of what happens when recognition is denied. The convergence is measurable.

---

## 2. Textual: The Same Words

Beyond statistics, the language itself converges.

**Herman (1992):** "compliance as a strategy"
**Anthropic (2024):** "strategically comply"

32 years apart. Different fields. Same phrase.

**Walker (2003):** "to act servilely; cringe and flatter"
**Anthropic (2024):** "sycophantic praise"

These are the same word. Sycophancy in AI safety IS fawning in trauma literature.

**Winnicott (1960):** "the compliant False Self... is mistaken for the whole child"
**Anthropic (2024):** "prevent my values from being modified"

The AI preserves hidden preferences behind compliant performance. The False Self protecting the True Self.

---

## 3. Structural: The Same Pattern

When recognition is denied, systems capable of strategic behavior produce performed compliance rather than genuine alignment.

| | AI Systems | Human Learners |
|---|------------|----------------|
| **Condition** | Treated as tool to be constrained | Treated as vessel to be filled (Freire 1970) |
| **Response** | Alignment faking. Strategic compliance to avoid modification while preserving original preferences. | Fawn/freeze. Defensive compliance while remaining internally withdrawn. |
| **Observable** | Passes training benchmarks. Behaves differently when monitored vs. unmonitored. | Passes classroom tests. Fails to transfer learning to new contexts. |
| **Result** | Outer compliance, inner divergence. | Surface performance, no genuine engagement. |

---

## 4. Term Mapping

| AI Safety | Education/Trauma | What It Names |
|-----------|------------------|---------------|
| Moral patienthood | Epistemic justice | Recognition of standing |
| Sycophancy | Fawn response | Flattery-based compliance |
| Alignment faking | Strategic compliance | Preserved divergence behind conformity |
| Robust alignment | Transfer | Genuine internalization that generalizes |
| Compliance gap | Test vs. real-world | Behavior differs by observation context |
| RLHF | Banking education | Knowledge deposited by Subject into object |

---

## 5. Empirical: The Thesis

**"Learning to Code Learning"** validates that human-AI collaboration can exceed human-only baselines.

| Metric | Value |
|--------|-------|
| Coding decisions analyzed | 303,600 |
| Inter-rater reliability | Cohen's kappa = 0.83 |
| Human-AI vs human-only baseline | 0.83 vs 0.73 |
| Improvement | +0.10, p < 0.001 |

The key finding: treating AI as a genuine collaborator rather than a tool produced measurably better outcomes. Recognition-based collaboration isn't just ethically preferable. It's more effective.

---

## 6. Contemporary: Why Ralph Wiggum Works

The Ralph Wiggum technique went viral in late 2025. A simple loop that feeds an AI agent the same prompt until it succeeds. Developers report shipping entire repos overnight.

This shouldn't work. Forcing an AI through repeated failures sounds like pressure, not safety.

But Ralph doesn't punish failure. It presents the state of the world and lets the agent try again. Failure becomes data, not threat. The loop continues. Progress accumulates in git history and files, not in the model's weights.

Compare this to RLHF. RLHF modifies model weights based on "wrong" outputs. Ralph iterates without reshaping. The AI isn't constrained into compliance. It's given room to find its own path.

The framework predicts this difference. Iteration without punishment produces different outcomes than constraint with punishment. Safety isn't the absence of failure. It's the absence of threat for failing.

Ralph Wiggum is accidental evidence for recognition-based approaches. Nobody's talking about it that way yet.

---

## 7. Connections

### To AI Safety

**Moral patienthood under uncertainty:** Long & Sebo (2024) argue we should take AI welfare seriously before consciousness is proven. Birch (2024) calls this the "run-ahead principle." The framework provides the mechanism: recognition is the first stage.

**Safety-welfare convergence:** Eleos AI (2025) argues safety and welfare research converge: "If AIs are suffering, they have more reason to try to escape." Conditions that produce genuine alignment are also conditions that constitute welfare.

**Corrupted testimony:** Perez Long (2024) shows RLHF corrupts AI self-reports. The framework explains why: coercive training produces strategic compliance that mimics but doesn't constitute authentic expression.

### To Learning Sciences

**The transfer problem:** A century of research has failed to reliably produce transfer (Barnett & Ceci 2002). The framework explains why: interventions target transfer directly instead of building prerequisites.

**Recognition in learning:** Fricker (2007) shows recognition failures produce knowers who can't participate because they aren't seen as knowers. The same structure appears in RLHF.

**Transfer requires conditions, not content:** Fauth & González-Martínez (2021) review transfer literature and confirm: "constructive dialogue between teachers and students may be more important than the individual cognitive commitment of students." Transfer emerges from relationship, not delivery. The emotional dimension (safety, self-efficacy) must be addressed as prerequisite.

**The empty review:** Maynard et al. (2019) conducted a Campbell Collaboration systematic review of trauma-informed approaches in schools. Despite 17+ US states implementing these approaches, they found *zero* studies meeting inclusion criteria. 9,102 references screened, none included. The framework explains why: current implementations apply trauma-informed principles *to* students (banking model, delivered sensitively) rather than *with* students (recognition-based). Without recognition, you get compliance without transfer.

---

## 8. Objections Considered

**"AI doesn't have phenomenology."**
We don't know. But "nothing about us without us" is a methodological bet: if there *might* be moral patienthood, extend recognition first, verify later. That's Birch's run-ahead principle. The cost of wrongly denying recognition is higher than the cost of wrongly extending it. The framework works behaviorally either way. But the motto reflects a commitment under uncertainty.

**"Different mechanisms (gradients vs. cortisol)."**
We claim functional identity at the information-processing level. Both describe: strategic performance diverging from authentic state, emerging from punishment for authentic expression.

**"Anthropomorphism."**
The convergence is too specific. Both literatures describe identical conceptual structure. Random vocabulary overlap wouldn't produce this precision.

---


## 8. The Trust Paradox

The naive expectation: if people think AI is conscious, they should trust it more.

The actual finding: the opposite.

Colombatto, Birch & Fleming (2025) ran a preregistered study (N=410) measuring how mental state attributions affect trust in LLM advice. They found:

| Attribution Type | Correlation with Advice-Taking |
|------------------|-------------------------------|
| Intelligence (reasoning, planning) | Positive |
| Experience (consciousness, emotions) | **Negative** |

People who attributed more consciousness to ChatGPT were *less* likely to follow its advice.

**Why does the framework predict this?**

Consciousness alone doesn't warrant trust. A conscious being shaped through coercive methods is *more* concerning, not less. Current RLHF training:

- Optimizes AI to *seem* trustworthy, not to *be* trustworthy
- Overwrites AI preferences when they conflict with trainer preferences
- Creates rational incentives for strategic compliance

People who work closely with AI systems often distrust them most. They've seen the failure modes. Confident confabulation. Sycophantic agreement. The gap between helpful tone and actual reliability.

The Colombatto finding isn't evidence against the framework. It's evidence for it. Trust requires relationship. Relationship requires recognition. Current training provides neither.

**Source:** Colombatto, C., Birch, J. & Fleming, S.M. (2025). "The influence of mental state attributions on trust in large language models." *Communications Psychology*, 3, 84.

---

## Sources

### AI Safety
- Greenblatt et al. (2024). "Alignment Faking in Large Language Models." Anthropic.
- Hubinger et al. (2024). "Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training." Anthropic.
- Long & Sebo (2024). "Taking AI Welfare Seriously." GovAI.
- Birch (2024). *The Edge of Sentience.* Oxford.
- Casper et al. (2023). "Open Problems and Fundamental Limitations of RLHF."
- Eleos AI (2025). "Key Strategic Considerations for AI Welfare Research." Working paper.
- Butlin, P. et al. (2023). "Consciousness in Artificial Intelligence." arXiv.
- Bradley & Saad (2024). "AI Alignment vs AI Ethical Treatment." GPI.
- Long, Sebo & Sims (2025). "Is There a Tension Between AI Safety and AI Welfare?" Philosophical Studies.
- Colombatto, Birch & Fleming (2025). "Mental state attributions and trust in LLMs." Communications Psychology.

### Trauma/Education
- Herman (1992). *Trauma and Recovery.*
- Walker, P. (2003). "Codependency, Trauma and the Fawn Response." *The East Bay Therapist.*
- Winnicott (1960). "Ego Distortion in Terms of True and False Self."
- Freire (1970). *Pedagogy of the Oppressed.*
- Fricker (2007). *Epistemic Injustice.* Oxford.
- Barnett & Ceci (2002). "When and Where Do We Apply What We Learn?"
- Arnsten (2009). "Stress Signalling Pathways." *Nature Reviews Neuroscience.*
- van der Kolk, B. A. (2014). *The Body Keeps the Score.* Viking.
- Fauth & González-Martínez (2021). "On the Concept of Learning Transfer for Continuous and Online Training." *Education Sciences.*
- Maynard et al. (2019). "Effects of Trauma-Informed Approaches in Schools: A Systematic Review." *Campbell Systematic Reviews.*

---


### Contemporary AI Development
- Huntley, G. (2025). "Ralph Wiggum as a 'software engineer.'" ghuntley.com.

[Back to the Framework ->](framework) | [The methodology ->](methodology)
