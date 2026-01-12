# Semantic Search - Sample Output

Query: `"recognition and safety in learning"`

```
Query: recognition and safety in learning

1. dissertation/05_Safety_Threat.md
   ## 5.10 Interim Conclusion
   > **SPEC:** Summarizes and transitions.
   > - Safety is constitutive condition for learning that transfers
   > - Evidence converges from multiple domains:
   >   - Neuroscience (PFC, stress response)
   >   - D...
   Score: 0.685

2. vault/00_Framework/Safety.md
   ## [[Recognition]] → Safety → [[Play]] → [[Transfer]]
   Safety is **stage 2** in the ontological entailment chain:
   | What Comes Before | What Safety IS | What Comes After |
   |-------------------|----------------|------------------|
   | [[Recognition]] | The c...
   Score: 0.657

3. dissertation/06_Recognition.md
   # Section 6: Recognition, Legitimacy, and the Epistemic Conditions of Learning
   > **DEPENDS ON:** Section 5 (safety requires stabilization)
   > **FEEDS INTO:** Section 7 (unified ontology)
   Score: 0.649

4. dissertation/Research/TRANSFER_SYNTHESIS_05_Safety.md
   ## Psychological Safety and Trust in Learning
   as learning opportunities, and cultivating respectful, inclusive
   relationships. It often requires recognizing power dynamics and
   explicitly inviting all voices to contribute without fear...
   Score: 0.648

5. dissertation/Research/LITERATURE_LANDSCAPE_ANALYSIS.md
   ## Conclusion: Positioning new synthesis work
   The landscape reveals **islands of sophisticated synthesis** surrounded
   by **unexplored theoretical waters**. Fleming's recognition-transformative
   learning synthesis, Murris's Baradian educational ont...
   Score: 0.641
```

## What This Shows

The search finds documents by **meaning**, not keywords:

- Query mentions "recognition" and "safety" - top results are about the relationship between them
- Result #2 shows the framework structure: Recognition → Safety → Play → Transfer
- Result #4 discusses "psychological safety" even though query said "safety" (semantic match)

This is different from grep/keyword search:
- `grep "recognition and safety"` would find exact phrase matches (probably zero)
- Semantic search understands the *concept* and finds relevant passages

## Another Example

Query: `"why do students forget what they learned"`

```
1. vault/00_Framework/Transfer.md
   Score: 0.612

2. vault/Desirable Difficulties.md
   Score: 0.589

3. dissertation/04_Transfer.md
   Score: 0.571

4. vault/Acquisition Metaphor.md
   Score: 0.543
```

The search understands that "forgetting" relates to transfer failure, desirable difficulties (deliberate challenge aids retention), and the acquisition metaphor (treating knowledge as possession that can be "lost").
