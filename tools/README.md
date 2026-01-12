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

## The Tools

### polyphony.py - Multi-Voice Dialogue Generator

Generates structured debate between five perspectives on any question.

```bash
python polyphony.py "What happens when AI is trained to be helpful above all else?"
python polyphony.py --interactive
python polyphony.py "Question" --save output.md
```

**Requires:** `ANTHROPIC_API_KEY` in environment or `.env` file. Falls back to template mode without key.

---

### recognition_engine.py - Semantic Gap Detection

Finds "recognition gaps" in an Obsidian vault - notes that are semantically similar but not connected in the graph.

```bash
python recognition_engine.py --vault /path/to/vault
python recognition_engine.py --vault ./vault --rebuild  # Force rebuild embeddings
python recognition_engine.py --gaps    # Show only gaps
python recognition_engine.py --bridges # Show only bridges
```

**Output:** `recognition_results.json` with gaps, bridges, and statistics.

---

### suggest_connections.py - Gap-to-Action Converter

Takes recognition_engine output and generates actionable markdown suggestions.

```bash
python suggest_connections.py --input recognition_results.json --output suggestions.md
python suggest_connections.py --top 20          # Top N suggestions
python suggest_connections.py --threshold 0.4   # Min gap score
```

**Output:** Markdown file with copy-paste YAML for Obsidian frontmatter.

---

### semantic_search.py - Natural Language Search

Search documents by meaning, not keywords.

```bash
python semantic_search.py "how do we handle disagreement"
python semantic_search.py --rebuild  # Rebuild index
python semantic_search.py --info     # Show index stats
```

**Note:** Run from directory containing your markdown files, or modify search paths in the code.

---

### repo_self_analysis.py - GitHub Issue Graph Analyzer

Applies recognition gap detection to a GitHub repository's issue graph.

```bash
# First, export your issues
gh issue list --limit 200 --json number,title,body,labels,state --state all > issues.json

# Then analyze
python repo_self_analysis.py --input issues.json --output results.json
```

**Output:** Clusters, gaps, bridges, and `gh issue comment` commands ready to copy-paste.

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

## Example Outputs

See the `examples/` directory for sample outputs from each tool.

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

## License

Part of the [RAISE framework](https://github.com/FedExodus/RAISE). MIT License.
