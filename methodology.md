---
layout: default
title: Methodology
---

# Methodology

## Polyphonic Code Review

Code in this work uses a practice called **polyphonic code comments**—a structured approach to multi-perspective review embedded directly in the codebase.

The idea is simple: do not let one voice dominate. Make multiple perspectives visible, in dialogue with each other.

Each comment identifies a **perspective** and **role**:

```python
# Kali [Visionary]: This could evolve into a full API...
# Athena [Reviewer]: But right now it just needs to work.
# Nemesis [Security]: What if the input is malicious?
# Klea [Accessibility]: Does this work with screen readers?
```

Five perspectives, each with four specialized roles:

| Perspective | Roles | Focus |
|-------------|-------|-------|
| **Kali** | Visionary, User Advocate, i18n, Onboarding | Possibilities, user needs, newcomer experience |
| **Athena** | Reviewer, Tester, Documentation, Future Maintainer | Flaws, edge cases, clarity, maintainability |
| **Vesta** | Builder, Architect, Refactorer, DevOps | Structure, patterns, elegance, deployment |
| **Nemesis** | Destroyer, Security, Privacy, Ethics | Fatal flaws, vulnerabilities, harm potential |
| **Klea** | Accessibility, Performance, Reliability, Product | Inclusion, speed, uptime, "should this exist?" |

Why does this matter? Because marginalized concerns (accessibility, ethics, i18n) are often afterthoughts. Named roles make them first-class citizens. Disagreement is documented. Future readers see the reasoning, not just the code.

It forces consideration. You cannot write `# Nemesis [Security]:` without actually thinking about security.

## Dialogic Drafting

This project was written through human-AI dialogue. The process demonstrates something: draft, feedback, revision, where both parties shape the final output.

The human brings positioning, questions, judgment. The AI brings pattern recognition across domains, capacity to hold complexity, responsiveness.

What emerges is neither human work assisted by AI nor AI work supervised by humans. It is dialogic. Something that neither party could have produced alone.

## Why This Matters for Relational AI Safety

These practices embody the argument:

- **Polyphony** demonstrates that multiple perspectives held in genuine tension produce better thinking than one voice
- **Dialogue** shows that human-AI collaboration can exceed both parties' individual capacities
- **Transparency** makes visible what is usually hidden: the reasoning, the disagreement, the process

Relational AI safety is not about hiding humans' involvement or pretending AI is just a tool. It is about honest, explicit partnership where both parties are recognized.

## The & in Practice

The methodology is the & in action:

Multiple voices AND coherent output. Disagreement AND synthesis. Human judgment AND AI extension. Transparency AND finished work.

Not choosing between them. Holding them together.
