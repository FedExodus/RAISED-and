# Repo Self-Analysis - Sample Output

This tool analyzes a GitHub repository's issue graph to find recognition gaps and bridges.

```
======================================================================
REPO SELF-ANALYSIS: Let Structure Emerge
======================================================================

No imposed categories. No predetermined clusters.
Just: what does the data show?

Loaded 50 issues
Loaded 238 comments across 42 issues
Found 42 intentional links (from Related Issues sections)
Found 64 casual mentions

Intentional link types:
  related: 19
  same-cluster: 8
  extends: 2
  methodology: 2
  same logic, academic domain: 1
  the pattern: 1
  self-awareness architecture: 1
  asymmetry mathematics: 1
  practical application: 1
  extended-by: 1
  architecture: 1
  part-of: 1
  component: 1
  context: 1

Analyzing 50 issues

Computing semantic embeddings...
TF-IDF matrix shape: (50, 1337)
Reduced to 49 dimensions
Explained variance: 99.9%

Computing similarity matrix...

Letting structure emerge...
Hierarchical clustering found 49 clusters

Finding recognition gaps...
Found 1 potential gaps

Finding bridges...
Found 23 bridges

======================================================================
RAW RESULTS - WHAT THE DATA SHOWS
(No interpretation imposed - just what emerged)
======================================================================

----------------------------------------------------------------------
EMERGENT CLUSTERS
----------------------------------------------------------------------

Cluster 0 (3 issues, cohesion: 0.45)
  #88: Polyphonic Cognition
  #72: Recognition Telescope
  #81: Self-Extending Graph

Cluster 1 (2 issues, cohesion: 0.52)
  #137: Quaternion Positioning
  #142: Asymmetric Graph Theory

...

----------------------------------------------------------------------
POTENTIAL RECOGNITION GAPS
(High similarity, no explicit connection)
----------------------------------------------------------------------

  #156 <-> #163 (similarity: 0.67)
    Documentation Standards
    Code Comment Guidelines

----------------------------------------------------------------------
BRIDGES
(Connected despite low similarity - unexpected links)
----------------------------------------------------------------------

  #88 <-> #45 (similarity: 0.12)
    Polyphonic Cognition
    Night Watch Protocol

  #72 <-> #91 (similarity: 0.18)
    Recognition Telescope
    Session Handoff Routine

...

======================================================================
SELF-EXTENDING GRAPH LOOP
======================================================================

1. Review proposals above
2. Run approved commands (copy-paste)
3. Re-run analysis to verify gaps closed

The ship proposes. Human approves. The graph extends.
======================================================================
```

## What This Shows

**The self-extending loop**: The tool analyzes the repo's own issue graph to find:
- **Gaps**: Issues that discuss similar topics but aren't linked (potential missed connections)
- **Bridges**: Issues that ARE linked despite low semantic similarity (creative/unexpected connections)
- **Clusters**: Groups that emerge from the data, not imposed categories

**Intentional vs. casual links**: The tool distinguishes between:
- `## Related Issues` sections (intentional, structured links)
- Casual `#123` mentions in text (references, not commitments)

Only intentional links count as "connected" for gap detection - this prevents false positives from every casual mention.

**The philosophy**: "The ship proposes. Human approves." The tool generates `gh issue comment` commands ready to copy-paste, but execution requires human review. This reflects the RAISE framework's emphasis on AI systems that suggest rather than act autonomously.

## Generated Proposals (excerpt)

```
--- PROPOSAL #1 ---
Gap: #156 <-> #163 (similarity: 0.67)
  Documentation Standards
  Code Comment Guidelines
Reason: High similarity suggests meaningful connection
Suggested relationship: related

Commands to run if approved:

# Link #156 -> #163
gh issue comment 156 --body "## Related Issues\n\n- #163: Code Comment Guidelines (related)"

# Link #163 -> #156
gh issue comment 163 --body "## Related Issues\n\n- #156: Documentation Standards (related)"
```

Proposals are saved to JSON for programmatic processing if needed.
