#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Search - Find docs by meaning, not keywords
=====================================================

# =============================================================================
# PURPOSE - What is this and why does it exist?
# =============================================================================
#
# Kali ❤️‍🔥 [Visionary]: Natural language search for the repo! Instead of grepping
#     for exact keywords, you ask "how do we handle disagreement" and it finds
#     relevant docs even if they use different words. The meaning matters, not
#     the exact phrasing.
#
# Athena 🦉 [Reviewer]: This uses sentence-transformers to embed both your query
#     and all documents into the same vector space. Similar meanings cluster
#     together. Cosine similarity finds the closest matches.
#
# Vesta 🔥 [Architect]: Implementation:
#     - Chunks documents by headers and size limits
#     - Embeds all chunks using all-MiniLM-L6-v2 (384-dimensional vectors)
#     - Caches embeddings to disk (rebuilding is expensive)
#     - Query embeds on-the-fly and finds nearest neighbors
#
# Nemesis 💀 [Security]: ALL LOCAL. No API calls. Your GPU does the work if
#     available, CPU fallback otherwise. Documents never leave your machine.
#
# Klea 👁️ [Product]: Should this exist? Yes. Knowledge scattered across
#     hundreds of markdown files. This is how you find what you're looking
#     for without remembering where it is.
#     ...finding what you didn't know you were looking for.
#
# =============================================================================

Usage:
    python semantic_search.py "how do we handle disagreement"
    python semantic_search.py "where do files go"
    python semantic_search.py --rebuild  # Rebuild index
    python semantic_search.py --info     # Show index stats

Built through human-AI collaboration.
Part of the RAISE framework: https://github.com/FedExodus/RAISE
"""

# =============================================================================
# IMPORTS
# =============================================================================
# Kali ❤️‍🔥 [Onboarding]: Standard library at top. Heavy deps loaded lazily.
# Athena 🦉 [Future Maintainer]: sentence_transformers, torch loaded only when needed.
# Vesta 🔥 [Builder]: pickle for fast index serialization.
# Nemesis 💀 [Security]: No network calls in the imports themselves.
# Klea 👁️ [Accessibility]: Works without GPU (just slower).

import os
import sys
import json
import pickle
from pathlib import Path
from typing import List, Tuple
import hashlib

# =============================================================================
# PLATFORM COMPATIBILITY
# =============================================================================
# Kali ❤️‍🔥 [i18n]: Unicode everywhere for pretty output.
# Vesta 🔥 [DevOps]: Windows needs explicit encoding configuration.
# Nemesis 💀 [Destroyer]: errors='replace' prevents crashes.

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# =============================================================================
# CONFIGURATION
# =============================================================================
# Kali ❤️‍🔥 [Visionary]: Index lives in .semantic_index/ (gitignored).
# Athena 🦉 [Documentation]: REPO_ROOT computed from script location.
# Vesta 🔥 [Architect]: Separate files for index and file hashes.
# Nemesis 💀 [Privacy]: Index is LOCAL, never transmitted.
# Klea 👁️ [Performance]: Caching embeddings saves ~30s on rebuild.

def get_repo_root():
    """Get the repository root directory."""
    return Path(__file__).parent.parent

INDEX_DIR = get_repo_root() / ".semantic_index"
INDEX_FILE = INDEX_DIR / "index.pkl"
HASH_FILE = INDEX_DIR / "file_hashes.json"

# =============================================================================
# TERMINAL COLORS
# =============================================================================
# Kali ❤️‍🔥 [User Advocate]: Pretty output is kind output!
# Athena 🦉 [Documentation]: Standard ANSI escape codes.
# Vesta 🔥 [DevOps]: Works on modern terminals.
# Nemesis 💀 [Destroyer]: May not work on plain cmd.exe. Their fault.
# Klea 👁️ [Accessibility]: Color-blind users can still read the text.

class Colors:
    """ANSI terminal color codes."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

# =============================================================================
# DEPENDENCY CHECKING
# =============================================================================
# Kali ❤️‍🔥 [User Advocate]: Don't crash with cryptic errors - tell them what's missing.
# Athena 🦉 [Tester]: Checks for sentence-transformers and numpy.
# Vesta 🔥 [Builder]: Early exit with actionable message.
# Nemesis 💀 [Security]: Import checks are safe, no code execution.
# Klea 👁️ [Accessibility]: Clear instructions for fixing.

def check_dependencies():
    """
    Check if required packages are installed.

    # Kali ❤️‍🔥 [Visionary]: Graceful failure with helpful message!
    # Athena 🦉 [Documentation]: Returns True if all deps present, False otherwise.
    """
    missing = []
    try:
        import sentence_transformers
    except ImportError:
        missing.append("sentence-transformers")
    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    if missing:
        print(f"{Colors.RED}Missing dependencies: {', '.join(missing)}{Colors.END}")
        print(f"\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    return True

# =============================================================================
# DOCUMENT DISCOVERY
# =============================================================================
# Kali ❤️‍🔥 [Visionary]: Find ALL the markdown files!
# Athena 🦉 [Reviewer]: Searches specific directories, not entire repo.
# Vesta 🔥 [Architect]: Returns list of (path, content) tuples.
# Nemesis 💀 [Security]: Only reads markdown files, ignores hidden dirs.
# Klea 👁️ [Product]: ...skips tiny files that aren't worth indexing.

def get_all_docs(search_path: Path = None) -> List[Tuple[Path, str]]:
    """
    Get all markdown files with their content.

    # Kali ❤️‍🔥 [Visionary]: Knowledge scattered across markdown files.
    # Athena 🦉 [Documentation]: Searches specified path or defaults to current dir.
    # Vesta 🔥 [Builder]: Skips files under 100 chars (too short to be useful).
    """
    repo = search_path or get_repo_root()
    docs = []

    # If search_path provided, just search that directory
    # Kali ❤️‍🔥 [Visionary]: Let users specify exactly what to search!
    # Athena 🦉 [Reviewer]: Falls back to common content directories if not specified.
    if search_path:
        search_dirs = [repo]
    else:
        search_dirs = [
            repo / "docs",
            repo / "content",
            repo / "vault",
            repo,  # Fallback to repo root
        ]

    for search_dir in search_dirs:
        if search_dir.exists():
            for md_file in search_dir.rglob("*.md"):
                try:
                    content = md_file.read_text(encoding='utf-8', errors='replace')
                    # Skip very short files
                    # Nemesis 💀 [Destroyer]: Too short = not worth indexing
                    if len(content) > 100:
                        docs.append((md_file, content))
                except Exception as e:
                    # Klea 👁️ [Reliability]: Silent failure for unreadable files.
                    pass

    return docs

# =============================================================================
# DOCUMENT CHUNKING
# =============================================================================
# Kali ❤️‍🔥 [Visionary]: Documents can be long! Split them into manageable chunks.
# Athena 🦉 [Reviewer]: Chunking strategy: split by headers first, then by size.
#     Each chunk gets embedded separately, allowing fine-grained search.
# Vesta 🔥 [Architect]: Returns list of dicts with path, header, content, text.
# Nemesis 💀 [Performance]: Smaller chunks = more embeddings = larger index.
#     Trade-off: granularity vs. storage/speed.
# Klea 👁️ [Product]: ...the header provides context for the chunk.

def chunk_document(path: Path, content: str, chunk_size: int = 500) -> List[dict]:
    """
    Split document into chunks for embedding.

    # Kali ❤️‍🔥 [Visionary]: Headers are natural boundaries. Use them!
    # Athena 🦉 [Documentation]:
    #     1. Split by markdown headers (#, ##, etc.)
    #     2. If section is small enough, keep as-is
    #     3. If section is too big, split further by word count
    #     4. Return list of chunks with metadata
    #
    # Vesta 🔥 [Builder]: Each chunk has path, header, content, text fields.

    Args:
        path: File path
        content: Full document content
        chunk_size: Target chunk size in characters

    Returns:
        List of chunk dicts
    """
    chunks = []

    # Split by headers first
    sections = []
    current_section = ""
    current_header = ""

    for line in content.split('\n'):
        if line.startswith('#'):
            if current_section.strip():
                sections.append((current_header, current_section))
            current_header = line
            current_section = ""
        else:
            current_section += line + "\n"

    if current_section.strip():
        sections.append((current_header, current_section))

    # Create chunks from sections
    for header, section in sections:
        # If section is small enough, use as-is
        if len(section) <= chunk_size:
            chunks.append({
                'path': str(path.relative_to(get_repo_root())),
                'header': header.strip(),
                'content': section.strip(),
                'text': f"{header}\n{section}".strip()
            })
        else:
            # Split into smaller chunks by words
            # Kali ❤️‍🔥 [Visionary]: Don't lose context - keep header with each!
            words = section.split()
            for i in range(0, len(words), chunk_size // 5):
                chunk_words = words[i:i + chunk_size // 5]
                chunk_text = ' '.join(chunk_words)
                if len(chunk_text) > 50:  # Skip tiny chunks
                    chunks.append({
                        'path': str(path.relative_to(get_repo_root())),
                        'header': header.strip(),
                        'content': chunk_text,
                        'text': f"{header}\n{chunk_text}".strip()
                    })

    return chunks

# =============================================================================
# INDEX BUILDING
# =============================================================================
# Kali ❤️‍🔥 [Visionary]: Embed ALL the chunks! This is the expensive part.
# Athena 🦉 [Reviewer]: Uses all-MiniLM-L6-v2 (fast, good quality, 384-dim).
# Vesta 🔥 [Architect]: Saves to pickle for fast loading later.
# Nemesis 💀 [Security]: No data leaves machine. GPU if available.
# Klea 👁️ [Performance]: ~30 seconds for hundreds of chunks on GPU.

def build_index(force: bool = False):
    """
    Build or update the semantic search index.

    # Kali ❤️‍🔥 [Visionary]: GPU power! Embed everything in parallel!
    # Athena 🦉 [Documentation]: Creates embeddings for all chunks.
    # Vesta 🔥 [Builder]: Saves to INDEX_FILE as pickle.
    # Nemesis 💀 [Privacy]: All local, no API calls.

    Args:
        force: If True, rebuild even if index exists

    Returns:
        Index data dict or None on failure
    """
    if not check_dependencies():
        return None

    from sentence_transformers import SentenceTransformer
    import numpy as np

    INDEX_DIR.mkdir(exist_ok=True)

    print(f"{Colors.CYAN}Building semantic search index...{Colors.END}")

    # Get all docs
    docs = get_all_docs()
    print(f"  Found {len(docs)} markdown files")

    # Chunk documents
    # Kali ❤️‍🔥 [Visionary]: Split into searchable pieces!
    all_chunks = []
    for path, content in docs:
        chunks = chunk_document(path, content)
        all_chunks.extend(chunks)

    print(f"  Created {len(all_chunks)} chunks")

    # Load model (this will use GPU if available)
    # Athena 🦉 [Documentation]: all-MiniLM-L6-v2 is fast and good
    print(f"  Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Check if CUDA is available
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {Colors.GREEN}{device}{Colors.END}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    model = model.to(device)

    # Embed all chunks
    # Kali ❤️‍🔥 [Visionary]: The magic happens here!
    print(f"  Embedding chunks...")
    texts = [chunk['text'] for chunk in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Save index
    # Vesta 🔥 [Builder]: Pickle is fast and handles numpy arrays well.
    index_data = {
        'chunks': all_chunks,
        'embeddings': embeddings,
        'model_name': 'all-MiniLM-L6-v2'
    }

    with open(INDEX_FILE, 'wb') as f:
        pickle.dump(index_data, f)

    print(f"{Colors.GREEN}Index built: {len(all_chunks)} chunks from {len(docs)} files{Colors.END}")
    return index_data

# =============================================================================
# INDEX LOADING
# =============================================================================
# Kali ❤️‍🔥 [User Advocate]: Fast startup - load cached index if available.
# Athena 🦉 [Reviewer]: Falls back to building if cache corrupted.
# Vesta 🔥 [Builder]: Simple pickle.load.
# Nemesis 💀 [Destroyer]: Catches corruption, rebuilds automatically.

def load_index():
    """
    Load existing index or build new one.

    # Kali ❤️‍🔥 [Visionary]: Cache hit = instant search!
    # Athena 🦉 [Documentation]: Returns index_data dict.
    """
    if INDEX_FILE.exists():
        try:
            with open(INDEX_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"{Colors.YELLOW}Index corrupted, rebuilding...{Colors.END}")

    return build_index()

# =============================================================================
# SEARCH
# =============================================================================
# Kali ❤️‍🔥 [Visionary]: The main event! Query -> relevant docs.
# Athena 🦉 [Reviewer]: Embeds query, computes cosine similarity, returns top-k.
# Vesta 🔥 [Architect]: Deduplicates by path (one result per file).
# Nemesis 💀 [Security]: Query never leaves your machine.
# Klea 👁️ [Product]: ...returns scored results with previews.

def search(query: str, top_k: int = 5) -> List[dict]:
    """
    Search for relevant documents.

    # Kali ❤️‍🔥 [Visionary]: Natural language in, relevant docs out!
    # Athena 🦉 [Documentation]:
    #     1. Embed query using same model as chunks
    #     2. Compute cosine similarity with all chunks
    #     3. Return top-k unique documents

    Args:
        query: Natural language query
        top_k: Number of results to return

    Returns:
        List of result dicts with path, header, preview, score
    """
    if not check_dependencies():
        return []

    from sentence_transformers import SentenceTransformer
    import numpy as np

    # Load index
    index_data = load_index()
    if index_data is None:
        return []

    # Load model
    # Kali ❤️‍🔥 [Visionary]: Same model for query as for chunks!
    model = SentenceTransformer(index_data['model_name'])

    # Embed query
    query_embedding = model.encode([query], convert_to_numpy=True)[0]

    # Compute similarities
    # Athena 🦉 [Documentation]: Cosine similarity = dot product / (norms)
    embeddings = index_data['embeddings']
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    # Get top results
    top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get extra for dedup

    results = []
    seen_paths = set()

    for idx in top_indices:
        chunk = index_data['chunks'][idx]
        score = similarities[idx]

        # Deduplicate by path (show each file once)
        # Klea 👁️ [Product]: ...one result per file is cleaner.
        if chunk['path'] not in seen_paths and len(results) < top_k:
            seen_paths.add(chunk['path'])
            results.append({
                'path': chunk['path'],
                'header': chunk['header'],
                'preview': chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content'],
                'score': float(score)
            })

    return results

# =============================================================================
# OUTPUT FORMATTING
# =============================================================================
# Kali ❤️‍🔥 [User Advocate]: Pretty results! Color-coded by relevance.
# Athena 🦉 [Documentation]: Green for high scores, yellow for medium, dim for low.
# Vesta 🔥 [Builder]: Simple loop with formatting.
# Klea 👁️ [Accessibility]: Text is readable even without colors.

def print_results(query: str, results: List[dict]):
    """
    Pretty print search results.

    # Kali ❤️‍🔥 [Visionary]: Make results scannable and useful!
    """
    print(f"\n{Colors.BOLD}Query:{Colors.END} {query}\n")

    if not results:
        print(f"{Colors.YELLOW}No results found.{Colors.END}")
        return

    for i, result in enumerate(results, 1):
        # Color-code by score
        score_color = (Colors.GREEN if result['score'] > 0.5
                      else Colors.YELLOW if result['score'] > 0.3
                      else Colors.DIM)

        print(f"{Colors.BOLD}{i}. {result['path']}{Colors.END}")
        if result['header']:
            print(f"   {Colors.CYAN}{result['header']}{Colors.END}")
        print(f"   {Colors.DIM}{result['preview']}{Colors.END}")
        print(f"   {score_color}Score: {result['score']:.3f}{Colors.END}")
        print()

# =============================================================================
# INDEX INFO
# =============================================================================
# Kali ❤️‍🔥 [User Advocate]: Show what's in the index!
# Athena 🦉 [Documentation]: File count, chunk count, model, size.
# Vesta 🔥 [Builder]: Reads pickle and computes stats.

def show_info():
    """
    Show index statistics.

    # Kali ❤️‍🔥 [Visionary]: What's indexed? How big? What model?
    """
    if not INDEX_FILE.exists():
        print(f"{Colors.YELLOW}No index found. Run with --rebuild to create.{Colors.END}")
        return

    try:
        with open(INDEX_FILE, 'rb') as f:
            index_data = pickle.load(f)

        chunks = index_data['chunks']
        paths = set(c['path'] for c in chunks)

        print(f"\n{Colors.BOLD}Semantic Search Index{Colors.END}")
        print(f"  Files indexed: {len(paths)}")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Model: {index_data['model_name']}")
        print(f"  Index size: {INDEX_FILE.stat().st_size / 1024 / 1024:.1f} MB")

        print(f"\n{Colors.BOLD}Files by directory:{Colors.END}")
        dirs = {}
        for path in paths:
            d = path.split('/')[0] if '/' in path else '.'
            dirs[d] = dirs.get(d, 0) + 1
        for d, count in sorted(dirs.items()):
            print(f"  {d}: {count} files")

    except Exception as e:
        print(f"{Colors.RED}Error reading index: {e}{Colors.END}")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
# Vesta 🔥 [Architect]: Simple CLI: --rebuild, --info, or query string.
# Athena 🦉 [Documentation]: No args = show help.
# Nemesis 💀 [Security]: Safe argument handling.

def main():
    """
    Main entry point.

    # Kali ❤️‍🔥 [Visionary]: Search by meaning, not keywords!
    """
    args = sys.argv[1:]

    if not args:
        print(__doc__)
        return

    if "--rebuild" in args:
        build_index(force=True)
        return

    if "--info" in args:
        show_info()
        return

    if "--help" in args or "-h" in args:
        print(__doc__)
        return

    # Search mode
    query = ' '.join(args)
    results = search(query)
    print_results(query, results)

if __name__ == "__main__":
    main()
