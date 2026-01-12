# Polyphonic Tools

Tools demonstrating the RAISE framework's approach to human-AI collaboration. Each tool uses **polyphonic methodology** - five distinct cognitive perspectives commenting on code to make reasoning visible.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run any tool with --help
python polyphony.py --help
python recognition_engine.py --help
python semantic_search.py --help
```

---

## The Tools

### 1. polyphony.py - Multi-Perspective Dialogue Generator
*Lines: 700+ | Dependencies: anthropic (optional)*

Generates structured debate between five distinct cognitive perspectives on any question. Each perspective has a defined epistemic stance (expansion, analysis, construction, destruction, observation). Demonstrates:
- Structured prompt engineering with distinct voice personas
- Graceful fallback to template mode when no API key available
- The thesis claim that insight emerges through friction between perspectives, not consensus

```bash
python polyphony.py "What happens when AI is trained to be helpful above all else?"
python polyphony.py --interactive
python polyphony.py "Question" --save output.md
```

**Requires:** `ANTHROPIC_API_KEY` in environment or `.env` file. Falls back to template mode without key.

---

### 2. recognition_engine.py - Semantic Gap Detection
*Lines: 1000+ | Dependencies: sentence-transformers, networkx*

The thesis made computational. Finds "recognition gaps" by comparing two distance metrics:
1. **Semantic distance** (sentence-transformer embeddings)
2. **Graph distance** (network topology)

High semantic similarity + low connection = recognition failure. Low semantic + high connection = creative bridge. Demonstrates:
- Dual-embedding theory for gap detection
- Local GPU processing (no data leaves machine)
- Graph algorithms: betweenness centrality, clustering coefficient, PageRank
- Ontological pattern filtering to reduce false positives

```bash
python recognition_engine.py --vault /path/to/vault
python recognition_engine.py --vault ./vault --rebuild  # Force rebuild embeddings
python recognition_engine.py --gaps    # Show only gaps
python recognition_engine.py --bridges # Show only bridges
```

**Output:** `recognition_results.json` with gaps, bridges, and statistics.

---

### 3. semantic_search.py - Natural Language Document Search
*Lines: 550+ | Dependencies: sentence-transformers*

Search documents by meaning, not keywords. "How do we handle disagreement" finds relevant docs even if they use different words. Demonstrates:
- Document chunking by headers and size limits
- Embedding-based similarity search
- Incremental index updates (only re-embeds changed files)
- All processing local, GPU-accelerated when available

```bash
python semantic_search.py "how do we handle disagreement"
python semantic_search.py "where do files go"
python semantic_search.py --rebuild  # Rebuild index
python semantic_search.py --info     # Show index stats
```

**Note:** Run from directory containing your markdown files, or modify search paths in the code.

---

### 4. suggest_connections.py - Gap-to-Action Converter
*Lines: 350+ | Dependencies: none (stdlib only)*

Post-processor that converts recognition_engine output into actionable markdown. Generates copy-paste YAML for Obsidian frontmatter. Demonstrates:
- Pipeline architecture (recognition -> suggestion -> human review)
- **Tools suggest, humans decide** - writes to review location, doesn't auto-modify
- Clean separation of detection from intervention

```bash
python suggest_connections.py --input recognition_results.json --output suggestions.md
python suggest_connections.py --top 20          # Top N suggestions
python suggest_connections.py --threshold 0.4   # Min gap score
```

**Output:** Markdown file with copy-paste YAML for Obsidian frontmatter.

---

### 5. repo_self_analysis.py - Self-Improving Knowledge Graph
*Lines: 1100+ | Dependencies: scikit-learn, numpy*

The recognition engine applied to GitHub issues. A knowledge graph analyzing itself to find its own blind spots. Demonstrates:
- TF-IDF embeddings + hierarchical clustering
- Graph construction from "Related Issues" sections
- Gap/bridge detection on issue graph
- Generates `gh issue comment` commands for human approval
- **Recursive thesis application** - the tool embodies what it detects

```bash
# First, export your issues
gh issue list --limit 200 --json number,title,body,labels,state --state all > issues.json

# Then analyze
python repo_self_analysis.py --input issues.json --output results.json
```

**Output:** Clusters, gaps, bridges, and `gh issue comment` commands ready to copy-paste.

---

## Common Elements Across All Tools

1. **Polyphonic code comments** - Five perspectives comment on every section, making reasoning visible
2. **Local processing** - No data transmitted to external services (except optional Claude API for polyphony.py)
3. **Graceful degradation** - Works without GPU, works without API keys, works offline
4. **Human-in-the-loop** - Tools suggest, humans decide

---

## The Polyphonic Methodology

Every tool uses five cognitive facets in code comments:

| Facet | Mode | What They Catch |
|-------|------|-----------------|
| **Kali** | Expansion | Possibilities, user needs, what-ifs |
| **Athena** | Analysis | Flaws, assumptions, missing evidence |
| **Vesta** | Architecture | Structure, implementation, what could break |
| **Nemesis** | Destruction | Security issues, uncomfortable truths |
| **Klea** | Observation | What others miss, the space between |

This makes reasoning visible - not just WHAT the code does, but WHY, and what different perspectives think about it.

---

## Example Outputs

See the `examples/` directory for sample outputs from each tool.

---

## Dependencies

Core:
- `numpy`, `networkx` - Data structures
- `sentence-transformers`, `torch` - Semantic embeddings
- `scikit-learn`, `scipy` - Clustering

Optional:
- `anthropic` - For polyphony.py API mode
- `python-dotenv` - For .env file support

Install all with:
```bash
pip install -r requirements.txt
```

---

## Files to .gitignore

The tools create cache files you may want to ignore:

```gitignore
# Embedding caches
.recognition_index/
*.pkl

# Generated outputs (optional - you may want to keep these)
recognition_results.json
*_suggestions.md
*_results.json
*_proposals.json
```

---

## License

Part of the [RAISE framework](https://github.com/FedExodus/RAISE). MIT License.
