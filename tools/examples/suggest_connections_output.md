# Suggest Connections - Sample Output

```
Suggestions written to: connection_suggestions.md
  Top 10 suggestions with gap score >= 0.3

============================================================
CONNECTION SUGGESTIONS PREVIEW
============================================================

  Culture
    <-> Connection: Malaby ↔ Pearce
    Similarity: 0.728, Gap: 0.628

  Community-Building
    <-> Relationship
    Similarity: 0.685, Gap: 0.552

  Language
    <-> Connection: Malaby ↔ Pearce
    Similarity: 0.572, Gap: 0.500

  Identity
    <-> Relationship
    Similarity: 0.632, Gap: 0.497

  Culture
    <-> Homo Ludens: A Renewed Reading
    Similarity: 0.585, Gap: 0.487
```

## Generated Markdown (excerpt)

The tool writes a full markdown file with actionable suggestions:

```markdown
# Connection Suggestions

*Generated 2026-01-11 14:32 by the Recognition Engine*

---

## What This Is

The recognition engine found these pairs of notes that are **semantically similar**
but **not connected** in the vault graph. They might be recognition failures -
ideas that SHOULD be connected but aren't.

Review each suggestion. If the connection makes sense, add it to the frontmatter.

---

## Suggested Connections

### 1. Culture <-> Connection: Malaby ↔ Pearce

**Semantic Similarity:** 0.728
**Gap Score:** 0.628

**Source:** `Culture.md`
**Target:** `Connection - Malaby ↔ Pearce.md`

**Action:** Add to one file's frontmatter:
```yaml
connects-to:
  - target: "[[Connection - Malaby ↔ Pearce]]"
    type: thematic
```

---

### 2. Community-Building <-> Relationship

**Semantic Similarity:** 0.685
**Gap Score:** 0.552

...
```

## What This Shows

The tool transforms raw gap data into **actionable suggestions** with:
- Copy-paste YAML for frontmatter
- Clear similarity/gap scores
- Human-readable format for review

The suggestions go to an "airlock" directory for human review - the system proposes, the human decides.
