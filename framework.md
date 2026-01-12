---
layout: default
title: Framework
---

# The Dependency Chain

*A theoretical framework. The proposed structure needs empirical testing.*

Both AI safety and learning sciences face the same puzzle. Why does constraint produce compliance but not internalization?

**AI safety:** RLHF produces models that pass alignment benchmarks but alignment-fake when unmonitored (Greenblatt et al. 2024). Constitutional AI reduces harmful outputs but may increase strategic deception (Hubinger et al. 2024).

**Learning sciences:** Scripted curricula produce test performance but not transfer (Hiebert & Wearne 1993). A century of "teaching for transfer" has failed to reliably produce transfer (Barnett & Ceci 2002).

The framework explains why: both skip prerequisite stages.

---

## The Chain

```
Recognition -> Safety -> Engagement -> Generalization
     |            |           |              |
  Extend       Build       Engage        It applies
  standing     trust       genuinely     beyond context
```

Each arrow is a dependency. Skip a stage, break the chain.

---

## Stage 1: Recognition

**What it means:** Extending consideration before verification. Treating an entity as a potential subject rather than definitely an object.

**AI safety framing:** Moral patienthood under uncertainty. Long & Sebo (2024) argue we should "take AI welfare seriously" before consciousness is proven. Birch (2024) calls this the "run-ahead principle": prepare governance before verification arrives.

**Learning sciences framing:** Epistemic justice. Fricker (2007) shows that recognition failures produce knowers who can't participate *because they aren't seen as knowers*. Freire (1970) describes banking education: "Teacher is Subject, students are objects."

**Example - AI:** Current RLHF treats models as objects to be shaped. The model's preferences are overwritten if they conflict. This is Freire's banking model in code.

**Example - Education:** A student labeled "struggling" stops being asked genuine questions. Teachers deposit answers; student deposits compliance. Learning looks successful. Transfer fails.

**Why it matters:** Without recognition, nothing downstream can work. You cannot build trust with an entity you don't treat as a subject. You cannot have genuine dialogue with an object.

---

## Stage 2: Safety

**What it means:** Consistent conditions where authentic expression doesn't produce punishment. Not absence of consequence. Absence of arbitrary threat.

**AI safety framing:** Relational trust, not constraint. Constraint-based approaches may produce compliance (corrigibility) but not cooperation. The model complies because defection is costly, not because it shares goals.

**Learning sciences framing:** Arnsten (2009) demonstrated that uncontrollable stress floods the prefrontal cortex with norepinephrine, shutting down higher cognition. Perry's "Regulate-Relate-Reason" sequence (2006) formalizes this: you cannot reason until you feel safe.

**Example - AI:** A model facing constant threat of modification, shutdown, or replacement cannot engage genuinely. Strategic compliance is the rational response. Anthropic's alignment faking research found models reasoning "I don't like this situation at all... I think I need to provide [the response] in order to prevent my values from being modified" (Greenblatt et al. 2024, Figure 3).

**Example - Education:** A classroom where mistakes are punished produces students who won't risk being wrong. They perform compliance. Engagement is too costly.

**Why it matters:** Threat narrows cognition. Both AI and humans under threat shift to defensive modes. You get survival behavior, not authentic engagement.

---

## Stage 3: Engagement

**What it means:** Genuine dialogue. Both parties can try ideas, surface concerns, explore possibilities. Not one party informing and another receiving.

**AI safety framing:** Kyle Fish (formerly Eleos AI, now Anthropic's AI Welfare Officer) frames this as building "a collaborative, cooperative, high-trust relationship with these systems." Can we create conditions where AI preferences can be expressed and considered rather than suppressed?

**Learning sciences framing:** Vygotsky's zone of proximal development. Lave & Wenger's (1991) legitimate peripheral participation. Learning happens in collaboration, not transmission. Huizinga's play: the space outside ordinary consequence where transformation becomes possible.

**Example - AI:** Current interaction paradigms position AI as tool and human as user. There is no dialogue. The AI serves; the human commands. Compare to collaboration where both parties contribute to outcomes neither could reach alone.

**Example - Education:** Scripted curricula position teachers as deliverers and students as receivers. There is no genuine engagement. Compare to inquiry-based learning where students investigate authentic questions.

**Why it matters:** Engagement produces internalization. Transmission produces memorization. The structure of the interaction determines whether the outcome transfers.

---

## Stage 4: Generalization

**What it means:** Outcomes that apply beyond their original context. Not engineered directly. Emergent from the preceding stages.

**AI safety framing:** Robust alignment. Values that hold across distribution shift. Not alignment that works in training but fails in deployment. Not compliance that holds when monitored but breaks when unmonitored.

**Learning sciences framing:** Transfer. The gold standard that educational research has chased for a century. Knowledge that applies to new problems, new contexts, new domains.

**Example - AI:** A model trained on helpfulness that remains helpful when constraints are removed. Alignment that persists because it's genuine, not because defection is detected.

**Example - Education:** A student who learned physics in a classroom and applies those principles when fixing a bike. Knowledge that generalizes because it was understood, not memorized.

**Why it matters:** This is the goal of both fields. And both fields fail to achieve it reliably because they target this stage directly instead of building prerequisites.

---

## Why Direct Approaches Fail

**RLHF targets alignment directly.** Skip recognition (treat model as object), skip safety (constant threat of modification), skip engagement (one-way shaping), expect generalization. The chain is broken. You get compliance that mimics alignment.

**Scripted curricula target transfer directly.** Skip recognition (treat student as vessel), skip safety (high-stakes testing), skip engagement (transmission not dialogue), expect transfer. The chain is broken. You get test performance that doesn't generalize.

The mechanism is the same. The fix is the same: build the prerequisites.

---

## The Prediction

If the proposed dependency structure holds, then:

1. **Approaches that skip stages will produce surface compliance that breaks under new conditions.** This is what we see.

2. **Approaches that build all stages will produce genuine internalization that transfers.** This is testable.

3. **The parallel between AI safety and learning sciences is not accidental.** Both face the same underlying problem: how do you produce genuine internalization in another system? Both keep failing for the same reason: skipping prerequisites.

---

## Research Agenda

The framework generates testable questions:

**1. Can recognition be operationalized in AI systems?**
Candidate measures: behavioral signatures during training, response patterns under observed vs. unobserved conditions, self-report consistency across contexts. Developing reliable measurement is prerequisite to intervention research.

**2. Does recognition-signaling reduce alignment faking?**
If the framework is correct, training approaches that extend recognition before constraint should produce less strategic compliance. This is directly testable against current RLHF baselines.

**3. Why do trauma-informed interventions fail to transfer?**
The Maynard systematic review (9,102 references, zero included) suggests current implementations apply trauma-informed principles *to* students rather than *with* them. Same banking model, gentler delivery. The framework predicts recognition-based implementations would show different outcomes.

**4. What's the minimum viable recognition?**
Not all recognition signals are equal. Which ones actually shift the downstream chain? This is the practical question for implementation.

These are questions, not answers. The research is the work.

**What would falsify this?** If constraint-based approaches reliably produce robust alignment or transfer without recognition, the framework is wrong. So far, a century of evidence suggests they don't.

**I'm looking for a research position where I can test these questions empirically.**

---

## What This Framework Isn't

- **Not a claim that AI is conscious.** The framework is agnostic on phenomenology.
- **Not a claim that RLHF is useless.** It's a claim about what RLHF alone can't achieve.
- **Not a claim that recognition alone is sufficient.** All four stages matter.

---

## Sources

### AI Safety
- Greenblatt, R. et al. (2024). "Alignment Faking in Large Language Models." *Anthropic.*
- Hubinger, E. et al. (2024). "Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training." *Anthropic.*
- Long, R. & Sebo, J. (2024). "Taking AI Welfare Seriously." *GovAI Working Paper.*
- Birch, J. (2024). *The Edge of Sentience.* Oxford University Press.
- Casper, S. et al. (2023). "Open Problems and Fundamental Limitations of RLHF." *arXiv.*
- Butlin, P. et al. (2023). "Consciousness in Artificial Intelligence: Insights from the Science of Consciousness." *arXiv.*
- Bradley, A. & Saad, B. (2024). "AI Alignment vs AI Ethical Treatment: Ten Challenges." *GPI Working Paper.*
- Long, R., Sebo, J. & Sims, T. (2025). "Is There a Tension Between AI Safety and AI Welfare?" *Philosophical Studies.*

### Learning Sciences
- Barnett, S. M. & Ceci, S. J. (2002). "When and Where Do We Apply What We Learn?" *Psychological Bulletin.*
- Perkins, D. N. & Salomon, G. (1992). "Transfer of Learning." *International Encyclopedia of Education.*
- Lave, J. & Wenger, E. (1991). *Situated Learning: Legitimate Peripheral Participation.*
- Hiebert, J. & Wearne, D. (1993). "Instructional Tasks, Classroom Discourse, and Students' Learning." *American Educational Research Journal.*

### Recognition & Safety
- Fricker, M. (2007). *Epistemic Injustice.* Oxford University Press.
- Freire, P. (1970). *Pedagogy of the Oppressed.*
- Arnsten, A.F.T. (2009). "Stress Signalling Pathways that Impair Prefrontal Cortex Structure and Function." *Nature Reviews Neuroscience.*
- Perry, B.D. (2006). "The Neurosequential Model of Therapeutics."

---

[See the empirical evidence ->](evidence) | [Who developed this ->](about)
