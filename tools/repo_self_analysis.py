#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Repo Self-Analysis: Let Structure Emerge
=========================================

# =============================================================================
# PURPOSE - What is this and why does it exist?
# =============================================================================
#
# Kali ❤️‍🔥 [Visionary]: THE REPO ANALYZING ITSELF. The thesis made recursive!
#     We use recognition gap detection on GitHub issues to find connections
#     we've missed. A knowledge graph improving itself.
#
# Athena 🦉 [Reviewer]: This is the recognition engine applied to the issue
#     graph rather than the vault. Same principle: semantic similarity vs
#     graph distance reveals gaps and bridges.
#
# Vesta 🔥 [Architect]: Pipeline:
#     1. Export issues: gh issue list -> issues.json
#     2. Build graph from Related Issues sections
#     3. Compute TF-IDF embeddings + hierarchical clustering
#     4. Find gaps (similar but unconnected) and bridges (connected but dissimilar)
#     5. Generate proposals for human approval
#
# Nemesis 💀 [Ethics]: CRITICAL: This generates PROPOSALS, not edits. Human
#     reviews each suggested connection before it's added. AI suggests,
#     human decides. Autonomy preserved.
#
# Klea 👁️ [Product]: Should this exist? Yes. A self-improving knowledge graph
#     is the thesis embodied. We're not just documenting recognition - we're
#     practicing it on ourselves.
#     ...learning to see its own blind spots.
#
# =============================================================================

Usage:
    python repo_self_analysis.py --input issues.json
    python repo_self_analysis.py --input issues.json --output results.json

Built through human-AI collaboration.
Part of the RAISE framework: https://github.com/FedExodus/RAISE
"""

# =============================================================================
# IMPORTS
# =============================================================================
# Kali ❤️‍🔥 [Onboarding]: Standard library + sklearn for clustering.
# Athena 🦉 [Documentation]: json/re for parsing, numpy for matrices, sklearn
#     for TF-IDF and clustering. No exotic dependencies.
# Vesta 🔥 [Builder]: sklearn is the one external dep. pip install scikit-learn
# Nemesis 💀 [Security]: No network calls except to load local JSON files.
# Klea 👁️ [Accessibility]: Works offline after initial issue export.

import sys
import json
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass

# Windows console encoding fix
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# =============================================================================
# DATA LOADING
# =============================================================================
# Kali ❤️‍🔥 [Visionary]: First we need the raw material - the issue graph!
# Athena 🦉 [Documentation]: Issues exported via `gh issue list --json ...`
# Vesta 🔥 [Architect]: Load issues and comments as separate files, merge later.
# Nemesis 💀 [Security]: Local files only. No API calls during analysis.
# Klea 👁️ [Product]: ...the foundation of everything that follows.


def load_issues(path: Path) -> List[Dict]:
    """
    Load issues from JSON export.

    # Kali ❤️‍🔥 [Visionary]: The raw graph nodes - every issue is a thought.
    # Athena 🦉 [Documentation]: Expects output from:
    #     gh issue list --limit 200 --json number,title,body,labels,state --state all
    # Vesta 🔥 [Builder]: Simple JSON load. Returns list of issue dicts.
    """
    with open(path, 'r', encoding='utf-8') as f:
        issues = json.load(f)
    print(f"Loaded {len(issues)} issues")
    return issues


def load_comments(path: Path) -> Dict[int, List[Dict]]:
    """
    Load issue comments from JSON export.

    # Kali ❤️‍🔥 [Visionary]: Comments contain Related Issues too! Don't miss them.
    # Athena 🦉 [Documentation]: Returns {issue_number: [list of comment dicts]}.
    #     Groups by issue number extracted from issue_url.
    # Vesta 🔥 [Builder]: Uses defaultdict for clean grouping.
    # Nemesis 💀 [Destroyer]: What if issue_url field is named differently?
    #     We check multiple possible field names.

    Returns:
        {issue_number: [list of comment dicts]}
    """
    if not path.exists():
        return {}

    with open(path, 'r', encoding='utf-8') as f:
        comments_list = json.load(f)

    # Kali ❤️‍🔥 [User Advocate]: Group by issue number for easy lookup.
    # Athena 🦉 [Reviewer]: Handle different possible field names for URL.
    comments_by_issue = defaultdict(list)
    for comment in comments_list:
        # Extract issue number from the issue URL
        # Format: https://github.com/owner/repo/issues/123
        issue_url = comment.get('issue_url', '') or comment.get('issueUrl', '') or comment.get('issue', {}).get('url', '')
        if issue_url:
            match = re.search(r'/issues/(\d+)', issue_url)
            if match:
                issue_num = int(match.group(1))
                comments_by_issue[issue_num].append(comment)

    total = sum(len(c) for c in comments_by_issue.values())
    print(f"Loaded {total} comments across {len(comments_by_issue)} issues")
    return dict(comments_by_issue)


# =============================================================================
# REFERENCE EXTRACTION
# =============================================================================
# Kali ❤️‍🔥 [Visionary]: Finding connections - casual mentions vs intentional links.
# Athena 🦉 [Documentation]: Two types: #N anywhere (mention) vs ## Related Issues (intentional).
# Vesta 🔥 [Architect]: The distinction matters - we find gaps in INTENTIONAL connections.
# Nemesis 💀 [Ethics]: Casual mentions aren't commitments. Intentional links are.
# Klea 👁️ [Product]: ...this distinction is the thesis applied to metadata.


def extract_issue_references(text: str) -> Set[int]:
    """
    Find all #N references in text.

    # Kali ❤️‍🔥 [Visionary]: Every #123 is a potential connection!
    # Athena 🦉 [Reviewer]: But these are just mentions, not intentional links.
    # Vesta 🔥 [Builder]: Simple regex. Returns set of integers.
    """
    if not text:
        return set()
    # Match #123 patterns (issue references)
    matches = re.findall(r'#(\d+)', text)
    return {int(m) for m in matches}


def extract_related_issues_section(body: str) -> List[Dict]:
    """
    Parse the ## Related Issues section specifically.

    # Kali ❤️‍🔥 [Visionary]: This is where INTENTIONAL connections live!
    #     The author deliberately listed these as related.
    #
    # Athena 🦉 [Documentation]: Expected format:
    #     ## Related Issues
    #     - #81: Self-Extending Graph (extends)
    #     - #88: Polyphonic Cognition (methodology)
    #
    # Vesta 🔥 [Architect]: Regex patterns for different section header styles.
    #     Returns list of {number, description, relationship_type}.
    #
    # Nemesis 💀 [Destroyer]: What if format varies? We try multiple patterns.
    #     But we might miss non-standard formats. That's okay - false negatives
    #     are better than false positives for this use case.

    Returns:
        list of {number, description, relationship_type}
    """
    if not body:
        return []

    related = []

    # Find the Related Issues section
    # Look for ## Related Issues or ## Related or similar
    # Kali ❤️‍🔥 [User Advocate]: Support multiple header styles.
    # Athena 🦉 [Tester]: Each pattern captures the section content until next header.
    patterns = [
        r'##\s*Related\s*Issues?\s*\n(.*?)(?=\n##|\n---|\Z)',
        r'##\s*Related\s*\n(.*?)(?=\n##|\n---|\Z)',
        r'\*\*Related.*?\*\*\s*\n(.*?)(?=\n##|\n---|\n\*\*|\Z)',
    ]

    section_text = None
    for pattern in patterns:
        match = re.search(pattern, body, re.IGNORECASE | re.DOTALL)
        if match:
            section_text = match.group(1)
            break

    if not section_text:
        return []

    # Parse each line in the section
    # Format: - #N: Description (relationship_type)
    # Or: - #N - Description
    # Or just: - #N
    # Kali ❤️‍🔥 [Visionary]: Flexible parsing for different styles!
    # Athena 🦉 [Documentation]: Groups: (1) number, (2) description, (3) rel_type
    line_pattern = r'-\s*#(\d+)(?:\s*[:\-]\s*([^(\n]+))?(?:\(([^)]+)\))?'

    for match in re.finditer(line_pattern, section_text):
        number = int(match.group(1))
        description = (match.group(2) or '').strip()
        rel_type = (match.group(3) or 'related').strip().lower()

        related.append({
            'number': number,
            'description': description,
            'relationship_type': rel_type,
        })

    return related


# =============================================================================
# GRAPH BUILDING
# =============================================================================
# Kali ❤️‍🔥 [Visionary]: From raw issues to a graph we can analyze!
# Athena 🦉 [Documentation]: Nodes = issues, edges = Related Issues links,
#     mentions = casual #N references (tracked but not used for gap detection).
# Vesta 🔥 [Architect]: Returns {nodes, edges, mentions, all_edges} dict.
# Nemesis 💀 [Ethics]: Why separate intentional from casual? Because casual
#     mentions aren't commitments. We only flag missing INTENTIONAL connections.
# Klea 👁️ [Product]: ...the graph is the project's nervous system.


def build_issue_graph(issues: List[Dict], comments: Dict[int, List[Dict]] = None) -> Dict:
    """
    Build graph of issues with:
    - nodes: issue number -> {title, body, labels}
    - edges: from Related Issues sections (intentional links) in bodies AND comments
    - mentions: casual #N references (not in Related Issues section)

    # Kali ❤️‍🔥 [Visionary]: The graph is the structure we'll analyze!
    #
    # Athena 🦉 [Reviewer]: Intentional vs casual distinction is crucial.
    #     Gaps are found in intentional connections only.
    #
    # Vesta 🔥 [Builder]: Processes both issue bodies and comments.
    #
    # Nemesis 💀 [Security]: We're only reading, never modifying the graph here.
    """
    if comments is None:
        comments = {}

    nodes = {}
    edges = []  # Intentional links from Related Issues sections
    mentions = []  # Casual #N references

    # Build nodes
    # Kali ❤️‍🔥 [Visionary]: Every issue is a node in our graph.
    for issue in issues:
        num = issue['number']
        nodes[num] = {
            'number': num,
            'title': issue.get('title', ''),
            'body': issue.get('body', '') or '',
            'labels': [l['name'] for l in issue.get('labels', [])],
            'state': issue.get('state', 'OPEN'),
        }

    # Build edges
    # Kali ❤️‍🔥 [Visionary]: Find all intentional and casual references.
    for issue in issues:
        source = issue['number']
        body = issue.get('body', '') or ''

        # Get intentional links from Related Issues section in body
        related = extract_related_issues_section(body)

        # Also check comments for Related Issues sections
        # Kali ❤️‍🔥 [User Advocate]: Comments can add Related Issues too!
        issue_comments = comments.get(source, [])
        for comment in issue_comments:
            comment_body = comment.get('body', '') or ''
            comment_related = extract_related_issues_section(comment_body)
            related.extend(comment_related)

        related_numbers = {r['number'] for r in related}

        # Add intentional edges
        # Vesta 🔥 [Builder]: Only add if target exists and isn't self-reference.
        for rel in related:
            target = rel['number']
            if target in nodes and target != source:
                edges.append({
                    'source': source,
                    'target': target,
                    'type': rel['relationship_type'],
                    'description': rel['description'],
                    'intentional': True,
                })

        # Get all #N mentions from body
        all_refs = extract_issue_references(body)

        # Also get mentions from comments (but don't double count)
        for comment in issue_comments:
            comment_body = comment.get('body', '') or ''
            all_refs.update(extract_issue_references(comment_body))

        # Mentions are refs NOT in the Related Issues section
        # Athena 🦉 [Reviewer]: Subtract intentional to get casual mentions.
        for target in all_refs:
            if target in nodes and target != source and target not in related_numbers:
                mentions.append({
                    'source': source,
                    'target': target,
                    'type': 'mention',
                    'intentional': False,
                })

    print(f"Found {len(edges)} intentional links (from Related Issues sections)")
    print(f"Found {len(mentions)} casual mentions")

    return {
        'nodes': nodes,
        'edges': edges,
        'mentions': mentions,
        'all_edges': edges + mentions,  # Combined for backward compatibility
    }


# =============================================================================
# SEMANTIC ANALYSIS
# =============================================================================
# Kali ❤️‍🔥 [Visionary]: Now we embed the issues into semantic space!
# Athena 🦉 [Documentation]: TF-IDF + SVD for dimensionality reduction.
#     This is the "semantic" half of the dual-embedding approach.
# Vesta 🔥 [Architect]: sklearn does the heavy lifting.
# Nemesis 💀 [Destroyer]: Why TF-IDF instead of sentence-transformers?
#     Because issues are longer documents and TF-IDF handles that well.
#     Also: fewer dependencies, works offline, faster.
# Klea 👁️ [Performance]: ...TF-IDF is surprisingly effective for this.


def get_issue_text(node: Dict) -> str:
    """
    Combine title and body for embedding.

    # Kali ❤️‍🔥 [Visionary]: Title + body = the full semantic content.
    # Athena 🦉 [Documentation]: Truncates very long bodies to 8000 chars.
    # Vesta 🔥 [Builder]: Simple concatenation.
    # Nemesis 💀 [Destroyer]: 8000 char limit prevents memory issues.
    """
    title = node.get('title', '')
    body = node.get('body', '') or ''
    # Truncate very long bodies
    if len(body) > 8000:
        body = body[:8000]
    return f"{title}\n\n{body}"


def compute_tfidf_embeddings(texts: List[str]) -> np.ndarray:
    """
    Compute TF-IDF embeddings with SVD reduction.

    # Kali ❤️‍🔥 [Visionary]: Turn text into vectors we can compare!
    #
    # Athena 🦉 [Documentation]:
    #     1. TfidfVectorizer creates sparse word frequency matrix
    #     2. TruncatedSVD reduces to 50 dimensions
    #     3. Result: dense vectors capturing semantic content
    #
    # Vesta 🔥 [Architect]: max_features=5000, stop_words removed,
    #     min_df=2 (must appear in 2+ docs), max_df=0.8 (can't be in 80%+ docs).
    #
    # Nemesis 💀 [Destroyer]: What if we don't have enough features?
    #     We fall back to raw TF-IDF. Not ideal but works.
    #
    # Klea 👁️ [Performance]: SVD reduction speeds up similarity computation.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        min_df=2,  # Must appear in at least 2 docs
        max_df=0.8,  # Can't appear in more than 80% of docs
    )

    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    # Reduce to manageable dimensions
    # Athena 🦉 [Reviewer]: n_components must be < min(n_features, n_samples)
    n_components = min(50, tfidf_matrix.shape[1] - 1, len(texts) - 1)
    if n_components < 2:
        print("Warning: Not enough features for SVD, using raw TF-IDF")
        return tfidf_matrix.toarray()

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(tfidf_matrix)
    print(f"Reduced to {reduced.shape[1]} dimensions")
    print(f"Explained variance: {svd.explained_variance_ratio_.sum():.1%}")

    return reduced


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarities.

    # Kali ❤️‍🔥 [Visionary]: How similar is every issue to every other?
    # Athena 🦉 [Documentation]: Returns NxN matrix where M[i,j] is similarity.
    # Vesta 🔥 [Builder]: sklearn's cosine_similarity handles this efficiently.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(embeddings)


# =============================================================================
# CLUSTERING - LET STRUCTURE EMERGE
# =============================================================================
# Kali ❤️‍🔥 [Visionary]: This is where we let the DATA tell us what groups exist!
#     No imposed categories. No predetermined clusters. Just: what emerges?
# Athena 🦉 [Documentation]: Two methods: DBSCAN (density-based) and
#     hierarchical (shows structure at multiple levels).
# Vesta 🔥 [Architect]: Both work on similarity/distance matrices.
# Nemesis 💀 [Destroyer]: Why not just use labels? Because labels are imposed.
#     These clusters are DISCOVERED. That's the thesis.
# Klea 👁️ [Product]: ...the repo learning its own structure.


def cluster_dbscan(similarity_matrix: np.ndarray, eps: float = 0.3, min_samples: int = 2) -> np.ndarray:
    """
    DBSCAN clustering on similarity matrix.
    Converts similarity to distance, finds dense regions.

    # Kali ❤️‍🔥 [Visionary]: DBSCAN finds clusters without being told how many!
    #
    # Athena 🦉 [Documentation]:
    #     - eps: max distance between neighbors (0.3 = 70% similarity)
    #     - min_samples: min points to form a cluster
    #     - Returns -1 for outliers (don't fit any cluster)
    #
    # Vesta 🔥 [Builder]: Converts similarity to distance (1 - sim).
    #
    # Nemesis 💀 [Destroyer]: Outliers (-1) are interesting! They're unique ideas
    #     that don't cluster with anything else.
    """
    from sklearn.cluster import DBSCAN

    # Convert similarity to distance
    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(distance_matrix)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"DBSCAN found {n_clusters} clusters, {n_noise} outliers")

    return labels


def cluster_hierarchical(similarity_matrix: np.ndarray, n_clusters: int = None) -> Tuple[np.ndarray, any]:
    """
    Hierarchical clustering - shows structure at multiple levels.
    If n_clusters is None, we'll determine it from the dendrogram.

    # Kali ❤️‍🔥 [Visionary]: Hierarchical shows LAYERS of structure!
    #     Like zooming in and out on a map.
    #
    # Athena 🦉 [Documentation]:
    #     - Uses average linkage (average distance between clusters)
    #     - If n_clusters not specified, finds natural break point
    #     - Returns labels AND linkage matrix for dendrogram
    #
    # Vesta 🔥 [Architect]: scipy handles the heavy lifting.
    #
    # Nemesis 💀 [Destroyer]: "Natural" break point is heuristic (biggest gap).
    #     Not perfect, but usually reasonable.
    """
    from sklearn.cluster import AgglomerativeClustering
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
    from scipy.spatial.distance import squareform

    # Convert to condensed distance matrix
    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)

    # Make symmetric and convert to condensed form
    # Athena 🦉 [Reviewer]: squareform expects symmetric matrix.
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    condensed = squareform(distance_matrix, checks=False)

    # Hierarchical clustering
    Z = linkage(condensed, method='average')

    if n_clusters is None:
        # Use the gap in the dendrogram to determine clusters
        # Kali ❤️‍🔥 [Visionary]: Find where the biggest jump happens!
        distances = Z[:, 2]
        diffs = np.diff(distances)
        # Find where the biggest jump happens
        if len(diffs) > 0:
            cut_point = np.argmax(diffs) + 1
            threshold = (distances[cut_point] + distances[cut_point-1]) / 2 if cut_point > 0 else distances[0]
            labels = fcluster(Z, t=threshold, criterion='distance') - 1
        else:
            labels = np.zeros(len(similarity_matrix), dtype=int)
    else:
        labels = fcluster(Z, t=n_clusters, criterion='maxclust') - 1

    n_found = len(set(labels))
    print(f"Hierarchical clustering found {n_found} clusters")

    return labels, Z


# =============================================================================
# ANALYSIS - GAPS AND BRIDGES
# =============================================================================
# Kali ❤️‍🔥 [Visionary]: THIS IS THE THESIS! Comparing semantic similarity
#     to graph structure reveals recognition failures.
# Athena 🦉 [Documentation]: Gaps = similar but unconnected (missed connections).
#     Bridges = connected but dissimilar (surprising connections).
# Vesta 🔥 [Architect]: Both use the same similarity matrix, different filters.
# Nemesis 💀 [Ethics]: We're making the repo see its own blind spots.
# Klea 👁️ [Product]: ...self-improvement through self-analysis.


def find_recognition_gaps(
    nodes: Dict[int, Dict],
    similarity_matrix: np.ndarray,
    node_ids: List[int],
    explicit_edges: Set[Tuple[int, int]],
    threshold: float = 0.5
) -> List[Dict]:
    """
    Find pairs with high semantic similarity but no explicit connection.
    These are potential recognition failures.

    # Kali ❤️‍🔥 [Visionary]: HIGH similarity + NO connection = missed recognition!
    #     These are ideas that SHOULD be linked but aren't.
    #
    # Athena 🦉 [Documentation]:
    #     - threshold: minimum similarity to consider (0.5 = moderate)
    #     - explicit_edges: set of (source, target) tuples from Related Issues
    #     - Returns sorted by similarity descending (worst gaps first)
    #
    # Vesta 🔥 [Builder]: O(n²) comparison but n is small (100s of issues).
    #
    # Nemesis 💀 [Destroyer]: What counts as "connected"? We check BOTH directions.
    #     (i,j) OR (j,i) means connected.
    """
    gaps = []
    n = len(node_ids)

    for i in range(n):
        for j in range(i + 1, n):
            sim = similarity_matrix[i, j]
            if sim < threshold:
                continue

            id_i, id_j = node_ids[i], node_ids[j]

            # Check if explicitly connected (either direction)
            is_connected = (id_i, id_j) in explicit_edges or (id_j, id_i) in explicit_edges

            if not is_connected:
                gaps.append({
                    'issue_a': id_i,
                    'issue_b': id_j,
                    'title_a': nodes[id_i]['title'],
                    'title_b': nodes[id_j]['title'],
                    'similarity': float(sim),
                })

    # Sort by similarity (highest gaps first)
    # Kali ❤️‍🔥 [Visionary]: Most similar = most likely missed connection.
    gaps.sort(key=lambda x: x['similarity'], reverse=True)
    return gaps


def find_bridges(
    nodes: Dict[int, Dict],
    similarity_matrix: np.ndarray,
    node_ids: List[int],
    explicit_edges: Set[Tuple[int, int]],
    threshold: float = 0.3
) -> List[Dict]:
    """
    Find pairs that ARE connected but have LOW similarity.
    These are bridges - unexpected connections across domains.

    # Kali ❤️‍🔥 [Visionary]: LOW similarity + CONNECTION = surprising insight!
    #     These are creative leaps, non-obvious connections.
    #
    # Athena 🦉 [Documentation]:
    #     - threshold: maximum similarity to consider (0.3 = low)
    #     - Returns sorted by similarity ascending (most surprising first)
    #
    # Vesta 🔥 [Builder]: Same structure as find_recognition_gaps, inverted logic.
    #
    # Nemesis 💀 [Destroyer]: Bridges are GOOD! They're the creative connections
    #     that wouldn't emerge from pure clustering.
    """
    bridges = []
    n = len(node_ids)

    for i in range(n):
        for j in range(i + 1, n):
            sim = similarity_matrix[i, j]
            if sim > threshold:
                continue

            id_i, id_j = node_ids[i], node_ids[j]

            # Check if explicitly connected
            is_connected = (id_i, id_j) in explicit_edges or (id_j, id_i) in explicit_edges

            if is_connected:
                bridges.append({
                    'issue_a': id_i,
                    'issue_b': id_j,
                    'title_a': nodes[id_i]['title'],
                    'title_b': nodes[id_j]['title'],
                    'similarity': float(sim),
                })

    # Sort by similarity (lowest = most surprising bridges first)
    bridges.sort(key=lambda x: x['similarity'])
    return bridges


def analyze_clusters(
    nodes: Dict[int, Dict],
    labels: np.ndarray,
    node_ids: List[int],
    similarity_matrix: np.ndarray
) -> Dict:
    """
    Analyze the emergent clusters.

    # Kali ❤️‍🔥 [Visionary]: What groups emerged? How cohesive are they?
    #
    # Athena 🦉 [Documentation]: For each cluster:
    #     - List members (issue number + title)
    #     - Compute cohesion (average internal similarity)
    #
    # Vesta 🔥 [Builder]: Returns dict with cluster names as keys.
    #
    # Nemesis 💀 [Destroyer]: Cluster -1 is "OUTLIERS" - issues that don't fit
    #     any cluster. These might be unique or just poorly connected.
    """
    clusters = defaultdict(list)

    for idx, label in enumerate(labels):
        issue_id = node_ids[idx]
        clusters[label].append({
            'number': issue_id,
            'title': nodes[issue_id]['title'],
        })

    # Compute cluster statistics
    # Kali ❤️‍🔥 [Visionary]: Cohesion = how "tight" the cluster is.
    cluster_stats = {}
    for label, members in clusters.items():
        if label == -1:
            name = "OUTLIERS"
        else:
            name = f"Cluster {label}"

        # Get indices for this cluster
        indices = [i for i, l in enumerate(labels) if l == label]

        # Compute internal cohesion (average similarity within cluster)
        # Athena 🦉 [Documentation]: Higher cohesion = tighter cluster.
        if len(indices) > 1:
            sims = []
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    sims.append(similarity_matrix[indices[i], indices[j]])
            cohesion = np.mean(sims) if sims else 0
        else:
            cohesion = 1.0

        cluster_stats[name] = {
            'size': len(members),
            'cohesion': float(cohesion),
            'members': members,
        }

    return cluster_stats


# =============================================================================
# OUTPUT - RAW RESULTS
# =============================================================================
# Kali ❤️‍🔥 [Visionary]: Show the user what the data says!
# Athena 🦉 [Documentation]: Print clusters, gaps, bridges, orphans.
# Vesta 🔥 [Architect]: Terminal output for quick review.
# Nemesis 💀 [Ethics]: "No interpretation imposed" - just raw findings.
# Klea 👁️ [Product]: ...human reads, human decides.


def print_raw_results(
    cluster_stats: Dict,
    gaps: List[Dict],
    bridges: List[Dict],
    orphans: List[Dict]
):
    """
    Print raw results for the user to see.

    # Kali ❤️‍🔥 [Visionary]: Clean, readable output. Let the data speak.
    # Athena 🦉 [Documentation]: Sections: clusters, gaps, bridges, orphans.
    # Vesta 🔥 [Builder]: Formatted terminal output with separators.
    """

    print("\n" + "=" * 70)
    print("RAW RESULTS - WHAT THE DATA SHOWS")
    print("(No interpretation imposed - just what emerged)")
    print("=" * 70)

    # Clusters
    # Kali ❤️‍🔥 [Visionary]: What groups emerged naturally?
    print("\n" + "-" * 70)
    print("EMERGENT CLUSTERS")
    print("-" * 70)

    # Sort by size
    sorted_clusters = sorted(
        cluster_stats.items(),
        key=lambda x: (-x[1]['size'], x[0])
    )

    for name, stats in sorted_clusters:
        print(f"\n{name} ({stats['size']} issues, cohesion: {stats['cohesion']:.2f})")
        for member in stats['members'][:10]:  # Show first 10
            print(f"  #{member['number']}: {member['title'][:60]}")
        if len(stats['members']) > 10:
            print(f"  ... and {len(stats['members']) - 10} more")

    # Gaps (recognition failures)
    # Kali ❤️‍🔥 [Visionary]: The missed connections!
    print("\n" + "-" * 70)
    print("POTENTIAL RECOGNITION GAPS")
    print("(High similarity, no explicit connection)")
    print("-" * 70)

    for gap in gaps[:15]:
        print(f"\n  #{gap['issue_a']} <-> #{gap['issue_b']} (similarity: {gap['similarity']:.2f})")
        print(f"    {gap['title_a'][:50]}")
        print(f"    {gap['title_b'][:50]}")

    if len(gaps) > 15:
        print(f"\n  ... and {len(gaps) - 15} more gaps")

    # Bridges
    # Kali ❤️‍🔥 [Visionary]: The surprising connections!
    print("\n" + "-" * 70)
    print("BRIDGES")
    print("(Connected despite low similarity - unexpected links)")
    print("-" * 70)

    for bridge in bridges[:10]:
        print(f"\n  #{bridge['issue_a']} <-> #{bridge['issue_b']} (similarity: {bridge['similarity']:.2f})")
        print(f"    {bridge['title_a'][:50]}")
        print(f"    {bridge['title_b'][:50]}")

    if len(bridges) > 10:
        print(f"\n  ... and {len(bridges) - 10} more bridges")

    # Orphans
    # Kali ❤️‍🔥 [Visionary]: Ideas that don't fit anywhere.
    if orphans:
        print("\n" + "-" * 70)
        print("ORPHANS / OUTLIERS")
        print("(Don't fit any cluster)")
        print("-" * 70)

        for orphan in orphans:
            print(f"  #{orphan['number']}: {orphan['title'][:60]}")


# =============================================================================
# OUTPUT - PROPOSALS
# =============================================================================
# Kali ❤️‍🔥 [Visionary]: Turn gaps into actionable proposals!
# Athena 🦉 [Documentation]: Generate gh commands ready to copy-paste.
# Vesta 🔥 [Architect]: Each proposal = two commands (link both directions).
# Nemesis 💀 [Ethics]: PROPOSALS, not actions. Human reviews and decides.
# Klea 👁️ [Product]: ...AI suggests, human approves.


def generate_proposals(gaps: List[Dict], nodes: Dict[int, Dict]) -> List[Dict]:
    """
    Generate actionable proposals to fix recognition gaps.

    # Kali ❤️‍🔥 [Visionary]: Each gap becomes a proposal with ready-to-run commands!
    #
    # Athena 🦉 [Documentation]:
    #     - For each gap, suggest adding Related Issues link
    #     - Generate gh commands for copy-paste
    #     - Proposals come in pairs (A->B and B->A)
    #
    # Vesta 🔥 [Builder]: Returns list of proposal dicts.
    #
    # Nemesis 💀 [Ethics]: The commands add COMMENTS with Related Issues,
    #     not direct edits to issue bodies. Non-destructive.

    Returns:
        list of proposals with gh commands ready to execute.
    """
    proposals = []

    for i, gap in enumerate(gaps):
        issue_a = gap['issue_a']
        issue_b = gap['issue_b']
        title_a = gap['title_a']
        title_b = gap['title_b']
        similarity = gap['similarity']

        # Determine likely relationship type based on similarity
        # Kali ❤️‍🔥 [Visionary]: Higher similarity = tighter relationship.
        if similarity > 0.7:
            rel_type = "same-cluster"
            reason = "Very high similarity suggests same workstream"
        elif similarity > 0.5:
            rel_type = "related"
            reason = "High similarity suggests meaningful connection"
        else:
            rel_type = "related"
            reason = "Moderate similarity worth investigating"

        # Generate the comment body
        comment_body = f"""## Related Issues

- #{issue_b}: {title_b[:50]} ({rel_type})

---
*Proposed by repo self-analysis (similarity: {similarity:.2f})*"""

        # Escape for shell
        # Vesta 🔥 [Builder]: Must escape quotes and backticks for bash.
        escaped_body = comment_body.replace('"', '\\"').replace('`', '\\`')

        proposal = {
            'id': i + 1,
            'gap': gap,
            'action': f"Link #{issue_a} -> #{issue_b}",
            'relationship': rel_type,
            'reason': reason,
            'command': f'gh issue comment {issue_a} --body "{escaped_body}"',
            'comment_body': comment_body,
        }
        proposals.append(proposal)

        # Also propose the reverse link
        # Athena 🦉 [Reviewer]: Bidirectional links are stronger.
        reverse_body = f"""## Related Issues

- #{issue_a}: {title_a[:50]} ({rel_type})

---
*Proposed by repo self-analysis (similarity: {similarity:.2f})*"""

        escaped_reverse = reverse_body.replace('"', '\\"').replace('`', '\\`')

        reverse_proposal = {
            'id': i + 1,  # Same ID - they're a pair
            'gap': gap,
            'action': f"Link #{issue_b} -> #{issue_a}",
            'relationship': rel_type,
            'reason': reason,
            'command': f'gh issue comment {issue_b} --body "{escaped_reverse}"',
            'comment_body': reverse_body,
        }
        proposals.append(reverse_proposal)

    return proposals


def print_proposals(proposals: List[Dict]):
    """
    Print proposals for human review.

    # Kali ❤️‍🔥 [Visionary]: Clear, actionable output for the user.
    # Athena 🦉 [Documentation]: Shows each gap, reason, and commands.
    # Vesta 🔥 [Builder]: Commands at the end for easy copy-paste.
    # Nemesis 💀 [Ethics]: "Review each" - not "run all blindly".
    """
    if not proposals:
        print("\n" + "-" * 70)
        print("NO PROPOSALS")
        print("(No gaps detected - graph is well-connected!)")
        print("-" * 70)
        return

    print("\n" + "=" * 70)
    print("PROPOSALS FOR HUMAN APPROVAL")
    print("(Review each, then run approved commands)")
    print("=" * 70)

    seen_pairs = set()
    for p in proposals:
        pair = (min(p['gap']['issue_a'], p['gap']['issue_b']),
                max(p['gap']['issue_a'], p['gap']['issue_b']))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        print(f"\n--- PROPOSAL #{p['id']} ---")
        print(f"Gap: #{p['gap']['issue_a']} <-> #{p['gap']['issue_b']} (similarity: {p['gap']['similarity']:.2f})")
        print(f"  {p['gap']['title_a'][:60]}")
        print(f"  {p['gap']['title_b'][:60]}")
        print(f"Reason: {p['reason']}")
        print(f"Suggested relationship: {p['relationship']}")
        print()
        print("Commands to run if approved:")

    # Print all commands grouped
    # Klea 👁️ [Product]: ...easy copy-paste at the end.
    print("\n" + "-" * 70)
    print("COMMANDS (copy-paste approved ones):")
    print("-" * 70)
    for p in proposals:
        print(f"\n# {p['action']}")
        print(p['command'])


def save_proposals(proposals: List[Dict], output_path: Path):
    """
    Save proposals to JSON for programmatic processing.

    # Kali ❤️‍🔥 [Visionary]: JSON for scripts, terminal output for humans.
    # Vesta 🔥 [Builder]: Simple json.dump with indent.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(proposals, f, indent=2)
    print(f"Proposals saved to {output_path}")


def save_results(
    cluster_stats: Dict,
    gaps: List[Dict],
    bridges: List[Dict],
    similarity_matrix: np.ndarray,
    node_ids: List[int],
    output_path: Path
):
    """
    Save results to JSON for further analysis.

    # Kali ❤️‍🔥 [Visionary]: Persist for later review or visualization.
    # Athena 🦉 [Documentation]: Includes clusters, gaps, bridges, stats.
    # Vesta 🔥 [Builder]: Truncates to top 50 gaps/bridges to keep file manageable.
    """
    results = {
        'clusters': cluster_stats,
        'gaps': gaps[:50],
        'bridges': bridges[:50],
        'node_ids': node_ids,
        'stats': {
            'total_issues': len(node_ids),
            'total_clusters': len([k for k in cluster_stats.keys() if k != 'OUTLIERS']),
            'total_gaps': len(gaps),
            'total_bridges': len(bridges),
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
# Kali ❤️‍🔥 [Visionary]: The full pipeline: load -> embed -> cluster -> analyze -> propose!
# Athena 🦉 [Documentation]: Main orchestrates all the pieces.
# Vesta 🔥 [Architect]: Error handling for missing input files.
# Nemesis 💀 [Ethics]: Ends with reminder: AI proposes, human approves.
# Klea 👁️ [Product]: ...the self-improving graph loop.


def main():
    """
    Main entry point.

    # Kali ❤️‍🔥 [Visionary]: Let structure emerge! The repo analyzing itself.
    #
    # Athena 🦉 [Documentation]: Full pipeline:
    #     1. Load issues and comments from JSON
    #     2. Build graph from Related Issues sections
    #     3. Compute TF-IDF embeddings
    #     4. Hierarchical clustering
    #     5. Find gaps and bridges
    #     6. Generate proposals
    #     7. Save results
    #
    # Vesta 🔥 [Builder]: Output path configurable via --output flag.
    """
    # Parse CLI arguments
    # Vesta 🔥 [Architect]: Simple arg parsing for --input and --output
    args = sys.argv[1:]
    issues_path = Path("./issues.json")  # Default
    output_path = Path("./analysis_results.json")  # Default

    i = 0
    while i < len(args):
        if args[i] == '--input' and i + 1 < len(args):
            issues_path = Path(args[i + 1])
            i += 2
        elif args[i] == '--output' and i + 1 < len(args):
            output_path = Path(args[i + 1])
            i += 2
        elif args[i] in ('--help', '-h'):
            print(__doc__)
            return
        else:
            i += 1

    print("=" * 70)
    print("REPO SELF-ANALYSIS: Let Structure Emerge")
    print("=" * 70)
    print()
    print("No imposed categories. No predetermined clusters.")
    print("Just: what does the data show?")
    print()

    # Load issues
    # Kali ❤️‍🔥 [Visionary]: First we need the raw material.
    if not issues_path.exists():
        print(f"Error: {issues_path} not found")
        print("Run: gh issue list --limit 200 --json number,title,body,labels,state --state all > issues.json")
        return

    issues = load_issues(issues_path)

    # Load comments (optional but recommended for full analysis)
    # Athena 🦉 [Documentation]: Comments file is optional - same name with _comments suffix
    comments_path = issues_path.parent / (issues_path.stem + "_comments.json")
    comments = load_comments(comments_path)
    if not comments:
        print("(No comments loaded - Related Issues in comments won't be detected)")
        print("To include comments, export them to: " + str(comments_path))

    # Build graph (including comments if available)
    graph = build_issue_graph(issues, comments)
    nodes = graph['nodes']
    intentional_edges = graph['edges']
    mentions = graph['mentions']
    all_edges = graph['all_edges']

    # Build edge sets for lookup
    # Kali ❤️‍🔥 [Visionary]: Intentional edges are the "real" connections.
    intentional_edge_set = {(e['source'], e['target']) for e in intentional_edges}
    all_edge_set = {(e['source'], e['target']) for e in all_edges}

    # For gap detection, use intentional edges only
    # Athena 🦉 [Reviewer]: We want to find things that SHOULD be in Related Issues but aren't.
    edge_set = intentional_edge_set

    # Report edge types
    if intentional_edges:
        print("\nIntentional link types:")
        type_counts = defaultdict(int)
        for e in intentional_edges:
            type_counts[e['type']] += 1
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  {t}: {c}")

    # Get ordered list of node IDs
    node_ids = sorted(nodes.keys())
    print(f"Analyzing {len(node_ids)} issues")

    # Get texts for embedding
    texts = [get_issue_text(nodes[nid]) for nid in node_ids]

    # Compute embeddings
    # Kali ❤️‍🔥 [Visionary]: Turn text into vectors!
    print("\nComputing semantic embeddings...")
    embeddings = compute_tfidf_embeddings(texts)

    # Compute similarity matrix
    print("\nComputing similarity matrix...")
    sim_matrix = cosine_similarity_matrix(embeddings)

    # Clustering - let structure emerge
    # Kali ❤️‍🔥 [Visionary]: No imposed categories. Just: what emerges?
    print("\nLetting structure emerge...")

    # Try hierarchical first (shows levels)
    labels, linkage_matrix = cluster_hierarchical(sim_matrix)

    # Analyze clusters
    cluster_stats = analyze_clusters(nodes, labels, node_ids, sim_matrix)

    # Find gaps
    # Kali ❤️‍🔥 [Visionary]: The missed connections!
    print("\nFinding recognition gaps...")
    gaps = find_recognition_gaps(nodes, sim_matrix, node_ids, edge_set, threshold=0.5)
    print(f"Found {len(gaps)} potential gaps")

    # Find bridges
    # Kali ❤️‍🔥 [Visionary]: The surprising connections!
    print("\nFinding bridges...")
    bridges = find_bridges(nodes, sim_matrix, node_ids, edge_set, threshold=0.3)
    print(f"Found {len(bridges)} bridges")

    # Get orphans
    orphans = cluster_stats.get('OUTLIERS', {}).get('members', [])

    # Print raw results
    print_raw_results(cluster_stats, gaps, bridges, orphans)

    # Generate proposals for gaps
    proposals = generate_proposals(gaps, nodes)

    # Print proposals for human review
    print_proposals(proposals)

    # Save for further analysis
    save_results(
        cluster_stats, gaps, bridges,
        sim_matrix, node_ids,
        output_path
    )

    # Save proposals separately
    proposals_path = output_path.parent / (output_path.stem + "_proposals.json")
    if proposals:
        save_proposals(proposals, proposals_path)

    # Kali ❤️‍🔥 [Visionary]: The self-extending loop!
    print("\n" + "=" * 70)
    print("SELF-EXTENDING GRAPH LOOP")
    print("=" * 70)
    print()
    print("1. Review proposals above")
    print("2. Run approved commands (copy-paste)")
    print("3. Re-run analysis to verify gaps closed")
    print()
    print("AI proposes. Human approves. The graph extends.")
    print("=" * 70)


if __name__ == "__main__":
    main()
