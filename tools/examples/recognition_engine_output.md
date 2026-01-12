# Recognition Engine - Sample Output

```
Loading vault from vault/...
  Loaded 212 notes
  Built 680 alias mappings
  Graph has 212 nodes, 876 edges

Computing semantic embeddings...
  Using cached embeddings from embeddings.pkl

Computing graph positions...
  Computed 212 positions

Detecting recognition gaps...
  Analyzing 22366 pairs...

Detecting bridges...

============================================================
RECOGNITION ENGINE - Analysis Complete
============================================================

Summary:
  Total gaps found: 625
    - instance-category: 305
    - scale-differentiation: 1
    - source-notes: 1
    - unclassified: 318
  Total bridges found: 64

Ontological Structure Detected
(The graph correctly distinguishes these different KINDS)

  Instance <-> Category (305 pairs)
    Communities of Practice
      <-> Gee - Learning by Design
    Communities of Practice
      <-> Paper Status Tracker
    Communities of Practice
      <-> Gathering Understanding

  Individual <-> Collective scale (1 pairs)
    Community Identity
      <-> Projective Identity

  Source <-> Notes about source (1 pairs)
    Lave & Wenger (1991) - Situated Learning
      <-> Lave & Wenger 1991 — Chunk 12 Notes (pp. 110-120)

Potential Recognition Failures
(High semantic similarity + not connected + no pattern)

  1. Communities of Practice
     <-> Home
     Semantic: 0.547
     Gap score: 0.203

  2. Communities of Practice
     <-> Acquisition Metaphor
     Semantic: 0.500
     Gap score: 0.171

  3. Home
     <-> Acquisition Metaphor
     Semantic: 0.624
     Gap score: 0.263

  4. Home
     <-> Agency
     Semantic: 0.568
     Gap score: 0.275

  5. Home
     <-> Desirable Difficulties
     Semantic: 0.516
     Gap score: 0.318

Top Bridges
(Low semantic similarity but connected - creative synthesis)

  1. Mislevy et al. (2003) - A Brief Introduction to Evidence-Centered Design
     -> Commander (EDH)
     Semantic: 0.132
     Bridge score: 0.868

  2. Friday Night Magic (FNM)
     -> Learning
     Semantic: 0.143
     Bridge score: 0.857

  3. Magic: The Gathering
     -> Learning
     Semantic: 0.170
     Bridge score: 0.830

  4. Limited (Format)
     -> Learning
     Semantic: 0.172
     Bridge score: 0.828

  5. Malone (2009) — Dragon Kill Points: The Economics of Power Gamers
     -> Participation Metaphor
     Semantic: 0.172
     Bridge score: 0.828

  6. Dialogue
     -> Relationship
     Semantic: 0.211
     Bridge score: 0.789

  7. Safety
     -> Community
     Semantic: 0.213
     Bridge score: 0.787

  8. Productive Failure
     -> Vygotsky (1933/1967) - Play and Its Role in Mental Development
     Semantic: 0.220
     Bridge score: 0.780

  9. Deckbuilding
     -> Learning and Identity: What Does It Mean to Be a Half-Elf?
     Semantic: 0.223
     Bridge score: 0.777

============================================================
Results saved to recognition_results.json
```

## What This Shows

**Gaps detected**: 625 total, of which 318 are "unclassified" (potential real recognition failures). The rest are expected gaps (e.g., a book shouldn't directly link to every note that cites it).

**Bridges found**: 64 surprising connections. The top bridges connect game design concepts (Magic: The Gathering, Commander format, Deckbuilding) to learning theory. These are the creative cross-domain insights that pure clustering would miss.

**Ontological patterns**: The engine distinguishes between different *kinds* of gaps:
- Instance ↔ Category (a specific paper vs. a general concept)
- Individual ↔ Collective (personal identity vs. community identity)
- Source ↔ Notes (a book vs. reading notes about that book)

This prevents false positives - not every semantic similarity should become a link.
