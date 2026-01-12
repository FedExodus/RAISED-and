#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recognition Engine - The Thesis Made Computational
===================================================

# =============================================================================
# PURPOSE - What is this and why does it exist?
# =============================================================================
#
# Kali ❤️‍🔥 [Visionary]: This is THE THESIS in code. We claim that recognition
#     failures produce alignment failures. This tool DETECTS recognition failures
#     by finding gaps between semantic similarity and graph structure. If two
#     notes are about similar things but aren't connected, that's a gap. If they're
#     connected despite being semantically distant, that's a BRIDGE - creative
#     synthesis across domains.
#
# Athena 🦉 [Reviewer]: The core insight is dual-embedding theory: we compute
#     two different measures of distance and compare them.
#     1. SEMANTIC distance: How similar are the contents? (sentence-transformers)
#     2. RECOGNITION distance: How connected are they in the graph? (network topology)
#     High semantic similarity + low recognition = potential failure
#     Low semantic similarity + high recognition = surprising synthesis
#
# Vesta 🔥 [Architect]: Implementation uses:
#     - sentence-transformers for local GPU embeddings (no API calls)
#     - networkx for graph analysis (betweenness, clustering, pagerank)
#     - Quaternion-like position vectors for multi-dimensional graph position
#     - Ontological pattern detection to filter false positives
#
# Nemesis 💀 [Security]: ALL PROCESSING IS LOCAL. No data leaves your machine.
#     Embeddings are cached to disk but never transmitted. API keys not required.
#     This is important: the vault may contain sensitive research notes.
#
# Klea 👁️ [Product]: Should this exist? Yes. The question "what should be
#     connected but isn't?" is the fundamental question of research synthesis.
#     This tool answers it computationally instead of relying on memory and luck.
#     ...it finds what you didn't know you were missing.
#
# =============================================================================
# WHAT THIS TOOL DEMONSTRATES (for RAISE portfolio)
# =============================================================================
#
# 1. SEMANTIC SIMILARITY: Uses sentence-transformers to embed text into vectors.
#    Demonstrates understanding of modern NLP techniques.
#
# 2. GRAPH ALGORITHMS: Betweenness centrality, clustering coefficient, pagerank.
#    Demonstrates understanding of network analysis.
#
# 3. DUAL-EMBEDDING THEORY: Comparing two different distance metrics to find
#    meaningful gaps. Novel application of the thesis concept.
#
# 4. ONTOLOGICAL PATTERN DETECTION: Filtering false positives by recognizing
#    when semantic similarity is expected but connection isn't (e.g., a source
#    and notes about that source are similar but shouldn't be linked).
#
# 5. QUATERNION-LIKE POSITIONS: Multi-dimensional graph position capturing
#    centrality, asymmetry, local density, and global importance.
# =============================================================================

Usage:
    python tools/recognition_engine.py             # Run analysis
    python tools/recognition_engine.py --rebuild   # Force rebuild embeddings
    python tools/recognition_engine.py --gaps      # Show only gaps
    python tools/recognition_engine.py --bridges   # Show only bridges
    python tools/recognition_engine.py --json      # Output as JSON

Built through human-AI collaboration.
Part of the RAISE framework: https://github.com/FedExodus/RAISE
"""

# =============================================================================
# IMPORTS
# =============================================================================
# Kali ❤️‍🔥 [Onboarding]: Standard library only at top. Heavy deps loaded lazily.
# Athena 🦉 [Future Maintainer]: All imports visible. networkx is the only
#   heavy dep that loads at import time (unavoidable - we need nx.DiGraph).
# Vesta 🔥 [Builder]: sentence_transformers and torch loaded ONLY when needed.
# Nemesis 💀 [Security]: yaml.safe_load only. No arbitrary code execution.
# Klea 👁️ [Accessibility]: Works without GPU - just slower on CPU.

import os
import sys
import re
import json
import pickle
import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import networkx as nx

# =============================================================================
# PLATFORM COMPATIBILITY
# =============================================================================
# Kali ❤️‍🔥 [i18n]: Unicode everywhere! Output includes fancy characters.
# Athena 🦉 [Tester]: Windows terminal needs explicit encoding configuration.
# Vesta 🔥 [DevOps]: Do this ONCE at module load, not per-write.
# Nemesis 💀 [Destroyer]: errors='replace' turns corrupt chars into � not crashes.
# Klea 👁️ [Reliability]: Silent degradation preferred for cosmetic issues.

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# =============================================================================
# CONFIGURATION
# =============================================================================
# Kali ❤️‍🔥 [Visionary]: These paths point to the Obsidian vault where your
#     research lives. The recognition graph is built from this data.
#
# Athena 🦉 [Documentation]: REPO_ROOT is computed from script location.
#     DEFAULT_VAULT can be overridden with --vault flag.
#
# Vesta 🔥 [Architect]: INDEX_DIR stores cached embeddings. Gitignored.
#     Embeddings take ~30s to compute, so caching is important.
#
# Nemesis 💀 [Privacy]: Embeddings are LOCAL. Never transmitted anywhere.
#     The .recognition_index/ directory should be in .gitignore.
#
# Klea 👁️ [Product]: ...configurable thresholds let users tune sensitivity.

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_VAULT = Path("./vault")  # Override with --vault flag
INDEX_DIR = REPO_ROOT / ".recognition_index"
INDEX_FILE = INDEX_DIR / "embeddings.pkl"

# =============================================================================
# THRESHOLDS - Tunable Parameters
# =============================================================================
# Kali ❤️‍🔥 [Visionary]: These control what counts as "similar" or "different".
#     Higher semantic threshold = stricter about what counts as similar.
#     Lower gap threshold = more sensitive to recognition failures.
#
# Athena 🦉 [Documentation]: Based on empirical testing with the vault.
#     0.5 cosine similarity = "fairly similar" for all-MiniLM-L6-v2.
#     0.3 cosine similarity = "fairly different".
#
# Vesta 🔥 [Builder]: Could make these CLI arguments. For now, hardcoded.
#
# Nemesis 💀 [Destroyer]: What if thresholds are wrong? Results will be noisy.
#     But that's information too - tells us the threshold needs adjustment.
#
# Klea 👁️ [Performance]: Lower thresholds = more pairs to check = slower.

SEMANTIC_HIGH_THRESHOLD = 0.5   # Above this = semantically similar
SEMANTIC_LOW_THRESHOLD = 0.3    # Below this = semantically different
RECOGNITION_GAP_THRESHOLD = 0.3  # Above this = recognition gap

# =============================================================================
# DATA CLASSES
# =============================================================================
# Kali ❤️‍🔥 [Visionary]: Structured data makes reasoning explicit. A Gap is
#     a recognition failure; a Bridge is creative synthesis. Names matter.
#
# Athena 🦉 [Documentation]: Dataclasses give us: typed fields, auto __init__,
#     asdict() for JSON serialization. Immutable semantics (we don't mutate).
#
# Vesta 🔥 [Architect]: Two core data types mirror the two findings:
#     - RecognitionGap: Should be connected, isn't
#     - Bridge: Shouldn't be connected (semantically), but is
#
# Nemesis 💀 [Ethics]: The gap_score is multiplicative: high semantic sim
#     TIMES high recognition distance. Both factors must be present.
#
# Klea 👁️ [Product]: ...these become the output. JSON-serializable for piping.

@dataclass
class RecognitionGap:
    """
    A gap between semantic similarity and graph recognition.

    # Kali ❤️‍🔥 [Visionary]: This is a PROBLEM. Two notes talk about similar
    #     things but aren't connected in the graph. The recognition system
    #     has failed to see their relationship.
    #
    # Athena 🦉 [Documentation]: Fields:
    #     - source/target: Note filenames (without .md)
    #     - source_title/target_title: Human-readable titles
    #     - semantic_similarity: Cosine similarity of embeddings (0-1)
    #     - recognition_distance: Normalized quaternion distance (0-1)
    #     - gap_score: semantic_similarity * recognition_distance
    #     - ontological_pattern: If this gap is expected (e.g., source-notes)
    #
    # Nemesis 💀 [Destroyer]: ontological_pattern filters false positives.
    #     Some gaps are CORRECT - e.g., a book and notes about that book
    #     are semantically similar but shouldn't be linked.
    """
    source: str
    target: str
    source_title: str
    target_title: str
    semantic_similarity: float      # Higher = more similar content
    recognition_distance: float     # Higher = less connected in graph
    gap_score: float                # semantic_sim * recognition_dist
    ontological_pattern: Optional[str]  # None = true gap, else = expected

@dataclass
class Bridge:
    """
    A connection across semantic distance - unexpected synthesis.

    # Kali ❤️‍🔥 [Visionary]: This is GOOD. Two notes that seem different are
    #     connected anyway. This represents creative synthesis - seeing
    #     relationships that aren't obvious.
    #
    # Athena 🦉 [Documentation]: Fields:
    #     - source/target: Note filenames
    #     - semantic_similarity: Low (that's why it's a bridge)
    #     - bridge_score: 1 - semantic_similarity (higher = more surprising)
    #
    # Klea 👁️ [Product]: ...bridges are the opposite of gaps. gaps are failures;
    #     bridges are successes. both are interesting.
    """
    source: str
    target: str
    source_title: str
    target_title: str
    semantic_similarity: float  # Low for bridges
    bridge_score: float         # 1 - semantic_similarity

# =============================================================================
# TERMINAL COLORS
# =============================================================================
# Kali ❤️‍🔥 [User Advocate]: Visual feedback makes output readable!
# Athena 🦉 [Documentation]: Standard ANSI escape codes.
# Vesta 🔥 [DevOps]: Works on Windows Terminal, VS Code, Git Bash.
# Nemesis 💀 [Destroyer]: May not work on plain cmd.exe. That's cmd.exe's fault.
# Klea 👁️ [Accessibility]: Consider --no-color flag for accessibility needs.

class Colors:
    """ANSI terminal color codes for pretty output."""
    HEADER = '\033[95m'     # Light magenta
    BLUE = '\033[94m'       # Light blue
    CYAN = '\033[96m'       # Cyan (status messages)
    GREEN = '\033[92m'      # Light green (success/bridges)
    YELLOW = '\033[93m'     # Yellow (warnings/gaps)
    RED = '\033[91m'        # Light red (errors)
    END = '\033[0m'         # Reset to default
    BOLD = '\033[1m'        # Bold text
    DIM = '\033[2m'         # Dim/faint text

# =============================================================================
# Dependency Checking
# =============================================================================

def check_dependencies() -> bool:
    """Check if required packages are installed."""
    # Vesta [Architect]: Early exit if deps missing, with helpful message
    # Klea [Product]: Clear error messages are kindness
    missing = []

    try:
        import sentence_transformers
    except ImportError:
        missing.append("sentence-transformers")

    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        import networkx
    except ImportError:
        missing.append("networkx")

    if missing:
        print(f"{Colors.RED}Missing dependencies: {', '.join(missing)}{Colors.END}")
        print(f"\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    return True

# =============================================================================
# YAML/Markdown Parsing
# =============================================================================

def extract_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter from markdown content."""
    # Vesta [Architect]: Standard YAML frontmatter between --- markers
    # Nemesis [Security]: Safe load, fail gracefully
    if not content.startswith('---'):
        return {}
    try:
        end = content.index('---', 3)
        yaml_content = content[3:end]
        return yaml.safe_load(yaml_content) or {}
    except:
        return {}

def get_title_from_content(content: str) -> Optional[str]:
    """Extract H1 title from markdown."""
    # Kali [Visionary]: First # heading is the title
    match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    return match.group(1) if match else None

def get_body_text(content: str) -> str:
    """Extract body text (after frontmatter) for embedding."""
    # Vesta [Architect]: Skip the YAML, get the content
    if content.startswith('---'):
        try:
            end = content.index('---', 3)
            return content[end+3:].strip()
        except:
            return content
    return content

# =============================================================================
# Vault Loading
# =============================================================================

def load_vault() -> Tuple[Dict, Dict, nx.DiGraph]:
    """
    Load all notes from vault.

    # Kali [Visionary]: Two passes - first load everything, then build edges
    # Athena [Documentation]: Returns notes dict, alias map, and directed graph
    # Vesta [Architect]: The graph IS the recognition structure

    Returns: (notes dict, alias_map, directed graph)
    """
    notes = {}
    alias_map = {}
    graph = nx.DiGraph()

    print(f"{Colors.CYAN}Loading vault from {DEFAULT_VAULT}...{Colors.END}")

    # Pass 1: Load all notes and build alias map
    for md_file in DEFAULT_VAULT.rglob("*.md"):
        if '.obsidian' in str(md_file):
            continue

        rel_path = md_file.relative_to(DEFAULT_VAULT)
        filename = md_file.stem

        try:
            with open(md_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            continue

        fm = extract_frontmatter(content)
        title = fm.get('title') or get_title_from_content(content) or filename
        body = get_body_text(content)

        notes[filename] = {
            'path': str(rel_path),
            'title': title,
            'frontmatter': fm,
            'body': body,
            'connects_to': [],
            'supported_by': [],
        }

        # Build alias map
        # Kali [Visionary]: Multiple ways to reference the same note
        alias_map[filename.lower()] = filename
        alias_map[title.lower()] = filename
        for alias in fm.get('aliases', []):
            if alias:
                alias_map[str(alias).lower()] = filename

        # Add node to graph
        graph.add_node(filename, **notes[filename])

    # Pass 2: Build edges
    # Athena [Documentation]: connects-to and supported-by create the recognition graph
    for filename, note in notes.items():
        fm = note['frontmatter']

        # connects-to edges
        for target in fm.get('connects-to', []):
            target_name = target.get('target') if isinstance(target, dict) else str(target)
            target_name = target_name.strip('[]').strip()
            resolved = alias_map.get(target_name.lower())
            if resolved and resolved != filename:
                note['connects_to'].append(resolved)
                graph.add_edge(filename, resolved, type='connects-to')

        # supported-by edges
        for target in fm.get('supported-by', []):
            target_name = target.get('target') if isinstance(target, dict) else str(target)
            target_name = target_name.strip('[]').strip()
            resolved = alias_map.get(target_name.lower())
            if resolved and resolved != filename:
                note['supported_by'].append(resolved)
                graph.add_edge(filename, resolved, type='supported-by')

    print(f"  Loaded {Colors.GREEN}{len(notes)}{Colors.END} notes")
    print(f"  Built {Colors.GREEN}{len(alias_map)}{Colors.END} alias mappings")
    print(f"  Graph has {Colors.GREEN}{graph.number_of_nodes()}{Colors.END} nodes, {Colors.GREEN}{graph.number_of_edges()}{Colors.END} edges")

    return notes, alias_map, graph

# =============================================================================
# Semantic Embeddings (Local GPU)
# =============================================================================

def compute_semantic_embeddings(notes: Dict, force_rebuild: bool = False) -> Dict[str, np.ndarray]:
    """
    Compute semantic embeddings for all notes using sentence-transformers.

    # Kali [Visionary]: Local GPU power! No API calls!
    # Vesta [Architect]: Caches embeddings to disk so we don't recompute every time
    # Nemesis [Security]: All local. No data leaves your machine.
    """
    from sentence_transformers import SentenceTransformer
    import torch

    INDEX_DIR.mkdir(exist_ok=True)

    # Check if we have cached embeddings
    if INDEX_FILE.exists() and not force_rebuild:
        try:
            with open(INDEX_FILE, 'rb') as f:
                cached = pickle.load(f)
            # Verify cache is valid
            if set(cached.keys()) == set(notes.keys()):
                print(f"  Using cached embeddings from {Colors.CYAN}{INDEX_FILE.name}{Colors.END}")
                return cached
            else:
                print(f"  {Colors.YELLOW}Cache stale, rebuilding...{Colors.END}")
        except Exception as e:
            print(f"  {Colors.YELLOW}Cache corrupted, rebuilding...{Colors.END}")

    # Load model
    # Kali [Visionary]: all-MiniLM-L6-v2 is fast and good
    # Athena [Documentation]: 384-dimensional embeddings, good for semantic similarity
    print(f"  Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Check for CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {Colors.GREEN}{device}{Colors.END}")
    if device == 'cuda':
        print(f"  GPU: {Colors.GREEN}{torch.cuda.get_device_name(0)}{Colors.END}")
    model = model.to(device)

    # Prepare texts
    filenames = list(notes.keys())
    texts = [notes[fn]['body'][:8000] for fn in filenames]  # Truncate for memory

    # Embed
    print(f"  Embedding {len(texts)} documents...")
    embeddings_array = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device
    )

    # Convert to dict
    embeddings = {fn: embeddings_array[i] for i, fn in enumerate(filenames)}

    # Cache
    with open(INDEX_FILE, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"  Cached embeddings to {Colors.CYAN}{INDEX_FILE.name}{Colors.END}")

    return embeddings

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Athena [Documentation]: Standard cosine sim, handles zero vectors
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

# =============================================================================
# GRAPH POSITION (Quaternion-like)
# =============================================================================
# Kali ❤️‍🔥 [Visionary]: This is one of the coolest parts! We represent each
#     node's position in "recognition space" as a 4D vector, like a quaternion.
#     The four dimensions capture different aspects of how a note relates to
#     the rest of the knowledge graph.
#
# Athena 🦉 [Reviewer]: Why quaternion-like? Because we need multiple orthogonal
#     dimensions to capture graph position. Using 4 specific metrics:
#     - Betweenness centrality (structural importance)
#     - Recognition asymmetry (gives vs receives recognition)
#     - Clustering coefficient (local neighborhood density)
#     - PageRank (global importance accounting for direction)
#
# Vesta 🔥 [Architect]: All four metrics are computed by networkx. We just
#     combine them into a single vector per node, then compute distances
#     between vectors to get "recognition distance".
#
# Nemesis 💀 [Ethics]: The asymmetry metric (gives vs receives) captures
#     something important: some notes GIVE recognition (cite others) while
#     some RECEIVE it (are cited). This asymmetry matters for understanding
#     the recognition landscape.
#
# Klea 👁️ [Product]: ...four dimensions might not be enough. but it's a start.

def compute_graph_positions(graph: nx.DiGraph) -> Dict[str, np.ndarray]:
    """
    Compute quaternion-like positions from graph structure.

    # Kali ❤️‍🔥 [Visionary]: Each node gets a 4D position vector representing
    #     its role in the recognition landscape.
    #
    # Athena 🦉 [Documentation]: The four components:
    #     w = betweenness centrality (0-1): How often this node lies on
    #         shortest paths between other nodes. High = structurally important.
    #     i = recognition asymmetry (-1 to +1): (out_degree - in_degree) / total.
    #         Positive = gives more than receives. Negative = receives more.
    #     j = clustering coefficient (0-1): How connected are this node's
    #         neighbors to each other? High = dense local neighborhood.
    #     k = pagerank (0-1): Global importance accounting for who links to you.
    #
    # Vesta 🔥 [Builder]: Returns dict mapping filename -> 4D numpy array.
    #
    # Nemesis 💀 [Security]: Pure computation, no side effects.

    Returns:
        Dict mapping node names to 4D position vectors
    """
    positions = {}

    # Compute graph metrics
    # Kali ❤️‍🔥 [Visionary]: networkx does the heavy lifting!
    # Athena 🦉 [Tester]: These can fail on empty graphs - we handle upstream.
    # Vesta 🔥 [DevOps]: O(n³) for betweenness on dense graphs. Fine for <1000 nodes.
    betweenness = nx.betweenness_centrality(graph)
    clustering = nx.clustering(graph.to_undirected())
    pagerank = nx.pagerank(graph)

    for node in graph.nodes():
        # Compute recognition asymmetry
        out_deg = graph.out_degree(node)
        in_deg = graph.in_degree(node)
        total_deg = out_deg + in_deg + 1  # +1 to avoid division by zero

        # Build quaternion-like position
        # Kali ❤️‍🔥 [Visionary]: Four orthogonal dimensions of graph position!
        w = betweenness.get(node, 0)              # Structural importance
        i = (out_deg - in_deg) / total_deg        # Asymmetry: positive = gives
        j = clustering.get(node, 0)               # Local density
        k = pagerank.get(node, 0)                 # Global importance

        positions[node] = np.array([w, i, j, k])

    return positions

def quaternion_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Compute distance between quaternion-like positions.

    # Athena 🦉 [Reviewer]: Using Euclidean distance for simplicity.
    #     Could use geodesic distance on the 4-sphere for true quaternion
    #     semantics, but Euclidean works fine for our purposes.
    #
    # Vesta 🔥 [Builder]: Simple and fast. O(1) per pair.
    """
    return float(np.linalg.norm(q1 - q2))

# =============================================================================
# ONTOLOGICAL PATTERN DETECTION
# =============================================================================
# Kali ❤️‍🔥 [Visionary]: This is the false positive filter! Sometimes two notes
#     are semantically similar but SHOULDN'T be linked. A book and notes about
#     that book will have high similarity but aren't the same thing.
#
# Athena 🦉 [Reviewer]: We detect four patterns that explain semantic similarity
#     without requiring graph connection:
#     1. source-notes: Original source and reading notes about it
#     2. entity-text: A person and texts featuring that person
#     3. instance-category: Specific example and general category
#     4. scale-differentiation: Individual vs collective scale
#
# Vesta 🔥 [Architect]: Pattern detection uses heuristics on filenames and paths.
#     Not perfect, but catches the most common false positives.
#
# Nemesis 💀 [Ethics]: This is important for HONESTY. Without it, we'd report
#     false gaps and mislead the user. Better to miss some real gaps than
#     flood output with false positives.
#
# Klea 👁️ [Product]: ...these patterns are the thesis in miniature. different
#     KINDS of things can be semantically similar but ontologically distinct.

def detect_ontological_pattern(source: str, target: str, notes: Dict) -> Optional[str]:
    """
    Detect if a gap represents an ontological category difference
    rather than a recognition failure.

    # Kali ❤️‍🔥 [Visionary]: Some things SHOULD be different even if semantically
    #     similar. This function catches those cases.
    #
    # Athena 🦉 [Documentation]: Returns one of four pattern names, or None.
    #     - "source-notes": A source and reading notes about it
    #     - "entity-text": A person and texts featuring that person
    #     - "instance-category": A specific thing and its general category
    #     - "scale-differentiation": Individual vs collective scale
    #
    # Vesta 🔥 [Builder]: Uses string matching heuristics. Fast but imperfect.
    #
    # Nemesis 💀 [Destroyer]: What if we miss patterns? We'll report false gaps.
    #     What if we over-detect patterns? We'll miss real gaps. Tradeoff.
    #     Current tuning errs toward fewer false positives.

    Args:
        source: Source note filename
        target: Target note filename
        notes: Dict of all notes with metadata

    Returns:
        Pattern name if detected, None if this looks like a real gap
    """
    s_lower = source.lower()
    t_lower = target.lower()
    s_title = notes[source]['title'].lower()
    t_title = notes[target]['title'].lower()
    s_path = notes[source].get('path', '').lower()
    t_path = notes[target].get('path', '').lower()

    # Pattern 1: Source <-> Notes about source
    if ('chunk' in s_lower or 'notes' in s_lower) != ('chunk' in t_lower or 'notes' in t_lower):
        for word in t_lower.replace('-', ' ').replace('_', ' ').split():
            if len(word) > 4 and word in s_lower:
                return "source-notes"

    # Pattern 2: Person <-> Interview/Text featuring person
    if 'interview' in s_lower or 'interview' in t_lower:
        if any(word in t_lower for word in s_lower.split() if len(word) > 4):
            return "entity-text"

    # Pattern 3: Instance <-> Category
    if 'literature' in s_path and 'literature' not in t_path:
        return "instance-category"
    if 'literature' in t_path and 'literature' not in s_path:
        return "instance-category"

    # Pattern 4: Scale differentiation
    scale_words = ['community', 'collective', 'group', 'social', 'individual', 'personal']
    s_has_scale = any(w in s_lower for w in scale_words)
    t_has_scale = any(w in t_lower for w in scale_words)
    if s_has_scale != t_has_scale:
        for word in s_lower.replace('-', ' ').replace('_', ' ').split():
            if len(word) > 4 and word in t_lower:
                return "scale-differentiation"

    return None

# =============================================================================
# Gap and Bridge Detection
# =============================================================================

def find_recognition_gaps(
    notes: Dict,
    semantic_embeddings: Dict[str, np.ndarray],
    graph_positions: Dict[str, np.ndarray],
    graph: nx.DiGraph,
) -> Tuple[List[RecognitionGap], Dict[str, List[RecognitionGap]]]:
    """
    Find pairs with high semantic similarity but low recognition connection.

    # Kali [Visionary]: THE CORE OPERATION. This is the thesis.
    # Athena [Documentation]: High semantic sim + not connected = recognition failure
    #                         Unless there's an ontological reason
    """
    gaps = []
    ontological_clusters = defaultdict(list)
    filenames = list(notes.keys())
    n = len(filenames)

    # Normalize quaternion distances
    all_quat_dists = []
    for i in range(n):
        for j in range(i+1, n):
            d = quaternion_distance(graph_positions[filenames[i]], graph_positions[filenames[j]])
            all_quat_dists.append(d)
    max_quat_dist = max(all_quat_dists) if all_quat_dists else 1.0

    # Find gaps
    print(f"  Analyzing {n * (n-1) // 2} pairs...")
    for i in range(n):
        for j in range(i+1, n):
            fn_a, fn_b = filenames[i], filenames[j]

            # Skip if already connected
            if graph.has_edge(fn_a, fn_b) or graph.has_edge(fn_b, fn_a):
                continue

            # Compute distances
            sem_sim = cosine_similarity(semantic_embeddings[fn_a], semantic_embeddings[fn_b])
            quat_dist = quaternion_distance(graph_positions[fn_a], graph_positions[fn_b])
            quat_dist_norm = quat_dist / max_quat_dist if max_quat_dist > 0 else 0

            # Recognition failure: high semantic similarity + high recognition distance
            if sem_sim > SEMANTIC_HIGH_THRESHOLD and quat_dist_norm > RECOGNITION_GAP_THRESHOLD:
                pattern = detect_ontological_pattern(fn_a, fn_b, notes)

                gap = RecognitionGap(
                    source=fn_a,
                    target=fn_b,
                    source_title=notes[fn_a]['title'],
                    target_title=notes[fn_b]['title'],
                    semantic_similarity=sem_sim,
                    recognition_distance=quat_dist_norm,
                    gap_score=sem_sim * quat_dist_norm,
                    ontological_pattern=pattern,
                )
                gaps.append(gap)

                if pattern:
                    ontological_clusters[pattern].append(gap)
                else:
                    ontological_clusters['unclassified'].append(gap)

    # Sort by gap score
    gaps.sort(key=lambda x: x.gap_score, reverse=True)
    return gaps, ontological_clusters

def find_bridges(
    notes: Dict,
    semantic_embeddings: Dict[str, np.ndarray],
    graph: nx.DiGraph,
) -> List[Bridge]:
    """
    Find pairs with LOW semantic similarity but ARE connected.
    These are BRIDGES - unexpected synthesis across different domains.

    # Kali [Visionary]: Bridges are GOOD. They're creative connections.
    # Athena [Documentation]: Low semantic sim + connected = surprising synthesis
    """
    bridges = []

    for source, target in graph.edges():
        if source not in semantic_embeddings or target not in semantic_embeddings:
            continue

        sem_sim = cosine_similarity(semantic_embeddings[source], semantic_embeddings[target])

        # Bridge: low semantic similarity but connected
        if sem_sim < SEMANTIC_LOW_THRESHOLD:
            bridges.append(Bridge(
                source=source,
                target=target,
                source_title=notes[source]['title'],
                target_title=notes[target]['title'],
                semantic_similarity=sem_sim,
                bridge_score=1 - sem_sim,
            ))

    bridges.sort(key=lambda x: x.bridge_score, reverse=True)
    return bridges

# =============================================================================
# Reporting
# =============================================================================

def print_report(
    gaps: List[RecognitionGap],
    bridges: List[Bridge],
    ontological_clusters: Dict[str, List[RecognitionGap]],
    notes: Dict,
    graph: nx.DiGraph,
):
    """Print a beautiful report of findings."""

    # Header
    print()
    print(f"{Colors.BOLD}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}RECOGNITION ENGINE - Analysis Complete{Colors.END}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.END}")

    # Summary stats
    print(f"\n{Colors.BOLD}Summary:{Colors.END}")
    print(f"  Total gaps found: {Colors.YELLOW}{len(gaps)}{Colors.END}")
    for pattern, items in sorted(ontological_clusters.items()):
        color = Colors.DIM if pattern != 'unclassified' else Colors.YELLOW
        print(f"    - {pattern}: {color}{len(items)}{Colors.END}")
    print(f"  Total bridges found: {Colors.GREEN}{len(bridges)}{Colors.END}")

    # Ontological patterns (classified gaps - these are expected)
    pattern_descriptions = {
        'source-notes': "Source <-> Notes about source",
        'entity-text': "Entity <-> Text featuring entity",
        'instance-category': "Instance <-> Category",
        'scale-differentiation': "Individual <-> Collective scale",
    }

    if any(p for p in ontological_clusters if p != 'unclassified'):
        print(f"\n{Colors.BOLD}Ontological Structure Detected{Colors.END}")
        print(f"{Colors.DIM}(The graph correctly distinguishes these different KINDS){Colors.END}")

        for pattern, items in sorted(ontological_clusters.items()):
            if pattern == 'unclassified':
                continue
            desc = pattern_descriptions.get(pattern, pattern)
            print(f"\n  {Colors.CYAN}{desc}{Colors.END} ({len(items)} pairs)")
            for gap in items[:3]:
                print(f"    {gap.source_title}")
                print(f"      <-> {gap.target_title}")

    # Unclassified gaps (potential real failures)
    unclassified = ontological_clusters.get('unclassified', [])
    if unclassified:
        print(f"\n{Colors.BOLD}{Colors.YELLOW}Potential Recognition Failures{Colors.END}")
        print(f"{Colors.DIM}(High semantic similarity + not connected + no pattern){Colors.END}")

        for i, gap in enumerate(unclassified[:10], 1):
            print(f"\n  {Colors.BOLD}{i}. {gap.source_title}{Colors.END}")
            print(f"     <-> {gap.target_title}")
            print(f"     Semantic: {Colors.GREEN}{gap.semantic_similarity:.3f}{Colors.END}")
            print(f"     Gap score: {Colors.YELLOW}{gap.gap_score:.3f}{Colors.END}")

    # Bridges
    if bridges:
        print(f"\n{Colors.BOLD}{Colors.GREEN}Top Bridges{Colors.END}")
        print(f"{Colors.DIM}(Low semantic similarity but connected - creative synthesis){Colors.END}")

        for i, bridge in enumerate(bridges[:10], 1):
            print(f"\n  {Colors.BOLD}{i}. {bridge.source_title}{Colors.END}")
            print(f"     -> {bridge.target_title}")
            print(f"     Semantic: {Colors.DIM}{bridge.semantic_similarity:.3f}{Colors.END}")
            print(f"     Bridge score: {Colors.GREEN}{bridge.bridge_score:.3f}{Colors.END}")

    print()
    print(f"{Colors.BOLD}{'=' * 60}{Colors.END}")

def save_json(
    gaps: List[RecognitionGap],
    bridges: List[Bridge],
    ontological_clusters: Dict[str, List[RecognitionGap]],
    output_path: Path = REPO_ROOT / "tools" / "recognition_results.json"
):
    """Save results as JSON."""
    results = {
        'gaps': [asdict(g) for g in gaps[:50]],
        'bridges': [asdict(b) for b in bridges[:50]],
        'ontological_clusters': {
            k: [asdict(g) for g in v[:10]]
            for k, v in ontological_clusters.items()
        },
        'stats': {
            'total_gaps': len(gaps),
            'unclassified_gaps': len(ontological_clusters.get('unclassified', [])),
            'bridges_found': len(bridges),
            'patterns': {k: len(v) for k, v in ontological_clusters.items() if k != 'unclassified'},
        }
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {Colors.CYAN}{output_path}{Colors.END}")

# =============================================================================
# Main
# =============================================================================

def main():
    args = set(sys.argv[1:])

    if '--help' in args or '-h' in args:
        print(__doc__)
        return

    if not check_dependencies():
        return

    # Load vault
    notes, alias_map, graph = load_vault()

    if not notes:
        print(f"{Colors.RED}No notes found in vault!{Colors.END}")
        return

    # Compute embeddings
    print(f"\n{Colors.CYAN}Computing semantic embeddings...{Colors.END}")
    force_rebuild = '--rebuild' in args
    embeddings = compute_semantic_embeddings(notes, force_rebuild=force_rebuild)

    # Compute graph positions
    print(f"\n{Colors.CYAN}Computing graph positions...{Colors.END}")
    positions = compute_graph_positions(graph)
    print(f"  Computed {Colors.GREEN}{len(positions)}{Colors.END} positions")

    # Find gaps and bridges
    print(f"\n{Colors.CYAN}Detecting recognition gaps...{Colors.END}")
    gaps, ontological_clusters = find_recognition_gaps(notes, embeddings, positions, graph)

    print(f"\n{Colors.CYAN}Detecting bridges...{Colors.END}")
    bridges = find_bridges(notes, embeddings, graph)

    # Output
    if '--json' in args:
        save_json(gaps, bridges, ontological_clusters)
    else:
        # Print report
        show_gaps = '--gaps' in args
        show_bridges = '--bridges' in args

        if not show_gaps and not show_bridges:
            # Show everything
            print_report(gaps, bridges, ontological_clusters, notes, graph)
        else:
            if show_gaps:
                print(f"\n{Colors.BOLD}Recognition Gaps:{Colors.END}")
                for gap in gaps[:20]:
                    print(f"  {gap.source_title} <-> {gap.target_title} ({gap.gap_score:.3f})")
            if show_bridges:
                print(f"\n{Colors.BOLD}Bridges:{Colors.END}")
                for bridge in bridges[:20]:
                    print(f"  {bridge.source_title} -> {bridge.target_title} ({bridge.bridge_score:.3f})")

        # Always save JSON
        save_json(gaps, bridges, ontological_clusters)

if __name__ == "__main__":
    main()
