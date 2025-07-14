#!/usr/bin/env python3
"""
code_index.py: Scan a Git repo, build a DuckDB-backed code index with embeddings and HNSW (cosine similarity),
with CLI commands: index, search, watch (incremental, debounced).

This module provides a complete solution for indexing and searching code repositories using:
- DuckDB for persistent storage of file content and embeddings
- SentenceTransformers for generating semantic embeddings of code files
- HNSWLib for fast approximate nearest neighbor search using cosine similarity
- Watchdog for real-time file system monitoring with debounced updates

The system supports three main operations:
1. index: Scan a repository and build the initial search index
2. search: Query the index for semantically similar code
3. watch: Monitor a repository for changes and incrementally update the index
"""

# Standard library imports for file operations, process management, and utilities
import os
import subprocess
import argparse
import hashlib
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Third-party imports for database, ML, and file monitoring
import duckdb
import numpy as np
from sentence_transformers import SentenceTransformer
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

def get_version():
    """Get the version of turboprop."""
    try:
        from _version import __version__
        return __version__
    except ImportError:
        return "0.1.0"

# Configuration constants - these control the behavior of the indexing system
# Name of the table that stores file content and embeddings
TABLE_NAME = "code_files"
DIMENSIONS = 384               # Embedding dimensions for all-MiniLM-L6-v2 model
# SentenceTransformer model name for generating embeddings
EMBED_MODEL = "all-MiniLM-L6-v2"
# Backward compatibility for tests
DB_PATH = "code_index.duckdb"

# File extensions that we consider to be code files worth indexing
# This covers most major programming languages and common config/markup files
CODE_EXTENSIONS = {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp",
                   ".h", ".cs", ".go", ".rs", ".swift", ".kt", ".m", ".rb",
                   ".php", ".sh", ".html", ".css", ".json", ".yaml", ".yml",
                   ".xml"}


def compute_id(text: str) -> str:
    """
    Generate a unique identifier for file content using SHA-256 hashing.

    This creates a deterministic hash based on the input text, which allows us to:
    - Detect when file content has changed (different hash = updated content)
    - Avoid duplicate entries in the database
    - Create stable IDs that don't change unless the content changes

    Args:
        text: The text content to hash (typically file content)

    Returns:
        A hexadecimal string representation of the SHA-256 hash
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def init_db(repo_path: Path = None):
    """
    Initialize the DuckDB database and create the code_files table if it doesn't exist.

    The table schema includes:
    - id: VARCHAR PRIMARY KEY - unique hash of file path + content
    - path: VARCHAR - absolute path to the file
    - content: TEXT - full text content of the file
    - embedding: FLOAT[768] - vector embedding for similarity search

    Args:
        repo_path: Path to the repository where the database should be stored.
                  If None, uses current directory (for backward compatibility)

    Returns:
        DuckDB connection object ready for use
    """
    # Create database in the repository directory or current directory
    if repo_path is None:
        db_path = Path(DB_PATH)
    else:
        # Create turboprop directory if it doesn't exist
        turboprop_dir = repo_path / "turboprop"
        turboprop_dir.mkdir(exist_ok=True)
        db_path = turboprop_dir / "code_index.duckdb"
    
    con = duckdb.connect(str(db_path))
    con.execute(f"""
      CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id VARCHAR PRIMARY KEY,
        path VARCHAR,
        content TEXT,
        embedding DOUBLE[{DIMENSIONS}]
      )
    """)
    return con


def scan_repo(repo_path: Path, max_bytes: int):
    """
    Scan a Git repository for code files, respecting .gitignore rules.

    This function:
    1. Uses 'git ls-files' to get a list of files ignored by Git
    2. Walks the repository directory tree
    3. Filters files based on:
       - Not being in the Git ignore list
       - Having a code-related file extension
       - Being under the specified size limit

    Args:
        repo_path: Path to the Git repository root
        max_bytes: Maximum file size in bytes to include in indexing

    Returns:
        List of Path objects for files that should be indexed
    """
    # Resolve to absolute path to avoid issues with relative paths
    repo_path = repo_path.resolve()

    try:
        # Get list of files that Git ignores (including .gitignore, .git/info/exclude, etc.)
        result = subprocess.run(
            ["git", "ls-files", "--exclude-standard", "-oi", "--directory"],
            cwd=repo_path, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, check=True
        )
        # Convert the output to a set of resolved absolute paths for fast lookup
        ignored = {str(Path(line.strip()).resolve()) if Path(line.strip()).is_absolute() 
                   else str((repo_path / line.strip()).resolve())
                   for line in result.stdout.splitlines()}
    except subprocess.CalledProcessError:
        # If git command fails (not a git repo, etc.), proceed without ignore list
        ignored = set()

    files = []
    # Walk through all files in the repository
    for root, _, names in os.walk(repo_path):
        for name in names:
            p = Path(root) / name

            # Skip files that are ignored by Git
            if str(p.resolve()) in ignored:
                continue

            # Only include files with code-related extensions
            if p.suffix.lower() not in CODE_EXTENSIONS:
                continue

            try:
                # Skip files that are too large to process efficiently
                if p.stat().st_size > max_bytes:
                    continue
            except OSError:
                # Skip files we can't read (permissions, broken symlinks, etc.)
                continue

            files.append(p)
    return files


def get_existing_file_hashes(con):
    """
    Retrieve existing file hashes from the database for smart re-indexing.
    
    Returns:
        Dict mapping file paths to their content hashes
    """
    try:
        result = con.execute(f"SELECT path, id FROM {TABLE_NAME}").fetchall()
        return {path: hash_id for path, hash_id in result}
    except Exception:
        return {}


def filter_changed_files(files, existing_hashes):
    """
    Filter files to only include those that have changed or are new.
    
    Args:
        files: List of Path objects to check
        existing_hashes: Dict of path -> hash from database
        
    Returns:
        List of Path objects that need to be reprocessed
    """
    changed_files = []
    
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
            current_hash = compute_id(str(path) + text)
            
            # If file is new or content has changed, include it
            if str(path) not in existing_hashes or existing_hashes[str(path)] != current_hash:
                changed_files.append(path)
        except Exception:
            # If we can't read the file, skip it
            continue
    
    return changed_files


def process_single_file(path, embedder):
    """
    Process a single file for embedding generation.
    
    Args:
        path: Path object to process
        embedder: SentenceTransformer model
        
    Returns:
        Tuple of (uid, path_str, text, embedding_list) or None if failed
    """
    try:
        text = path.read_text(encoding="utf-8")
        uid = compute_id(str(path) + text)
        emb = embedder.encode(text, show_progress_bar=False).astype(np.float32)
        return (uid, str(path), text, emb.tolist())
    except Exception:
        return None


def embed_and_store(con, embedder, files, max_workers=None, progress_callback=None):
    """
    Process a list of files by generating embeddings and storing them in the database.
    Now supports parallel processing and progress reporting.

    Args:
        con: DuckDB database connection
        embedder: SentenceTransformer model for generating embeddings
        files: List of Path objects to process
        max_workers: Number of parallel workers (default: CPU count)
        progress_callback: Function to call with progress updates
    """
    if not files:
        return
    
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(files))
    
    rows = []
    completed = 0
    
    if progress_callback:
        progress_callback(0, len(files), "Starting embedding generation...")
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {executor.submit(process_single_file, path, embedder): path 
                         for path in files}
        
        # Collect results as they complete
        for future in as_completed(future_to_path):
            result = future.result()
            if result is not None:
                rows.append(result)
            
            completed += 1
            if progress_callback:
                progress_callback(completed, len(files), f"Processed {completed}/{len(files)} files")
    
    # Insert all rows in a single batch operation for better performance
    if rows:
        con.executemany(
            f"INSERT OR REPLACE INTO {TABLE_NAME} VALUES (?, ?, ?, ?)",
            rows
        )


def build_full_index(con):
    """
    Verify that the database contains embeddings for search operations.

    With DuckDB vector operations, no separate index file is needed.
    This function just validates that embeddings exist in the database.

    Args:
        con: DuckDB database connection

    Returns:
        Number of embeddings in the database, or 0 if none found
    """
    # Check if we have any embeddings in the database
    result = con.execute(
        f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE embedding IS NOT NULL").fetchone()
    return result[0] if result else 0


def search_index(con, embedder, query: str, k: int):
    """
    Search the code index for files semantically similar to the given query.

    This function:
    1. Generates an embedding for the search query
    2. Uses DuckDB's array operations to compute cosine similarity
    3. Returns the top k most similar files with their similarity scores

    Args:
        con: DuckDB database connection
        embedder: SentenceTransformer model for generating query embedding
        query: Text query to search for (e.g., "function to parse JSON")
        k: Number of top results to return

    Returns:
        List of tuples: (file_path, content_snippet, distance_score)
        Distance is cosine distance (lower = more similar)
    """
    # Generate embedding for the search query
    q_emb = embedder.encode(query, show_progress_bar=False).astype(np.float32).tolist()

    # Use DuckDB's array operations for cosine similarity search
    # Cosine similarity = dot(a,b) / (norm(a) * norm(b))
    # Cosine distance = 1 - cosine similarity
    results = con.execute(f"""
        SELECT 
            path,
            substr(content, 1, 300) as snippet,
            1 - (list_dot_product(embedding, $1) / 
                (sqrt(list_dot_product(embedding, embedding)) * sqrt(list_dot_product($1, $1)))) as distance
        FROM {TABLE_NAME}
        WHERE embedding IS NOT NULL
        ORDER BY distance ASC
        LIMIT {k}
    """, [q_emb]).fetchall()

    return results


def embed_and_store_single(con, embedder, path: Path):
    """
    Process a single file for embedding and storage (used for incremental updates).

    This is similar to embed_and_store() but handles just one file, making it
    suitable for real-time updates when files change during watch mode.

    Args:
        con: DuckDB database connection
        embedder: SentenceTransformer model for generating embeddings
        path: Path to the single file to process

    Returns:
        Tuple of (unique_id, embedding) if successful, None if file couldn't be processed
    """
    try:
        # Read the file content
        text = path.read_text(encoding="utf-8")
    except Exception:
        # Return None if file can't be read
        return None

    # Generate unique ID and embedding
    uid = compute_id(str(path) + text)
    emb = embedder.encode(text, show_progress_bar=False).astype(np.float32)

    # First, delete any existing records for this path (handles content changes)
    con.execute(f"DELETE FROM {TABLE_NAME} WHERE path = ?", (str(path),))
    
    # Store in database
    con.execute(
        f"INSERT INTO {TABLE_NAME} VALUES (?, ?, ?, ?)",
        (uid, str(path), text, emb.tolist())
    )

    # Return the ID and embedding for immediate index update
    return uid, emb


class DebouncedHandler(FileSystemEventHandler):
    """
    File system event handler that implements debounced file processing.

    Debouncing prevents excessive processing when files are rapidly modified
    (e.g., during saves, builds, or IDE operations). Instead of processing
    every single file change immediately, this handler waits for a quiet
    period before actually updating the index.

    This is especially important for:
    - Large files that take time to embed
    - Rapid-fire changes during development
    - Build processes that modify many files quickly
    - Auto-save features in editors

    How it works:
    1. When a file changes, start a timer
    2. If the file changes again before the timer expires, restart the timer
    3. Only process the file when the timer actually expires
    4. This ensures we only process the "final" version of rapid changes
    """

    def __init__(self, repo: Path, max_bytes: int, con, embedder, debounce_sec: float):
        """
        Initialize the debounced file handler.

        Args:
            repo: Path to the repository being watched
            max_bytes: Maximum file size to process
            con: DuckDB database connection
            embedder: SentenceTransformer model for embeddings
            debounce_sec: Seconds to wait before processing file changes
        """
        self.repo = repo
        self.max_bytes = max_bytes
        self.con = con
        self.embedder = embedder
        self.debounce = debounce_sec
        # Track active timers for each file to implement debouncing
        self.timers = {}

    def on_any_event(self, event):
        """
        Handle any file system event (create, modify, delete, move).

        This method is called by the watchdog library whenever a file system
        event occurs in the monitored directory tree.

        Args:
            event: FileSystemEvent object containing event details
        """
        p = Path(event.src_path)

        # Ignore directory events and non-code files
        if event.is_directory or p.suffix.lower() not in CODE_EXTENSIONS:
            return

        # Cancel any existing timer for this file (debouncing logic)
        if p in self.timers:
            self.timers[p].cancel()

        # Start a new timer that will process this file after the debounce period
        timer = threading.Timer(
            self.debounce, self._process, args=(event.event_type, p))
        self.timers[p] = timer
        timer.start()

    def _process(self, ev_type: str, p: Path):
        """
        Actually process a file change after the debounce period has elapsed.

        This method is called by the timer thread after the debounce delay.
        It handles the actual work of updating the database and search index.

        Args:
            ev_type: Type of file system event ("created", "modified", "deleted", etc.)
            p: Path to the file that changed
        """
        # Remove the timer from tracking (it's about to fire)
        self.timers.pop(p, None)

        if ev_type in ("created", "modified") and p.exists():
            try:
                # Only process files under the size limit
                if p.stat().st_size <= self.max_bytes:
                    # Update database with new file content and embedding
                    res = embed_and_store_single(self.con, self.embedder, p)
                    if res:
                        uid, emb = res
                        # With DuckDB, no separate index update needed
                        print(f"[debounce] updated {p}")
            except OSError:
                # Ignore files we can't access (permissions, etc.)
                pass
        elif ev_type == "deleted":
            # Remove deleted file from database
            self.con.execute(
                f"DELETE FROM {TABLE_NAME} WHERE path = ?", (str(p),))
            print(f"[debounce] {p} deleted from database")


def watch_mode(repo_path: str, max_mb: float, debounce_sec: float):
    """
    Start watching a repository for file changes and update the index incrementally.

    This function sets up a file system watcher that monitors the repository
    for any changes to code files. When changes are detected, it updates the
    search index incrementally rather than rebuilding from scratch.

    Benefits of watch mode:
    - Near real-time search results as you code
    - Much faster than full reindexing after every change
    - Debounced updates prevent excessive processing during rapid changes
    - Runs continuously until interrupted with Ctrl+C

    Args:
        repo_path: Path to the repository to watch
        max_mb: Maximum file size in megabytes to process
        debounce_sec: Seconds to wait before processing file changes
    """
    # Convert and validate the repository path
    repo = Path(repo_path).resolve()
    max_bytes = int(max_mb * 1024**2)

    # Initialize database and ML model
    con = init_db(repo)
    embedder = SentenceTransformer(EMBED_MODEL)

    # Verify database has embeddings
    embedding_count = build_full_index(con)
    print(f"[watch] found {embedding_count} embeddings in database")

    # Set up the debounced file handler
    handler = DebouncedHandler(
        repo, max_bytes, con, embedder, debounce_sec)

    # Create and configure the file system observer
    observer = Observer()
    observer.schedule(handler, str(repo), recursive=True)

    print(
        f"[watch] watching {repo} (max {max_mb} MB, debounce {debounce_sec}s)")
    observer.start()

    try:
        # Keep the watcher running until user interrupts
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Clean shutdown on Ctrl+C
        observer.stop()
    observer.join()


def reindex_all(repo_path: Path, max_bytes: int, con, embedder, max_workers=None, force_all=False):
    """
    Intelligently reindex a repository by only processing changed files.

    This function provides a smart reindexing workflow that:
    1. Scans repository for all eligible code files
    2. Compares with existing database to find changed/new files
    3. Only processes files that have actually changed
    4. Uses parallel processing for faster embedding generation

    Args:
        repo_path: Path to the repository root directory
        max_bytes: Maximum file size in bytes to process
        con: DuckDB database connection
        embedder: SentenceTransformer model for generating embeddings
        max_workers: Number of parallel workers (default: CPU count)
        force_all: If True, reprocess all files regardless of changes

    Returns:
        Tuple of (total_files_found, files_processed, time_taken)
    """
    start_time = time.time()
    
    # Scan the repository for indexable files
    print("🔍 Scanning repository for code files...")
    files = scan_repo(repo_path, max_bytes)
    
    if not files:
        return 0, 0, 0
    
    # Smart re-indexing: only process changed files
    files_to_process = files
    if not force_all:
        print("🧠 Checking for changed files...")
        existing_hashes = get_existing_file_hashes(con)
        files_to_process = filter_changed_files(files, existing_hashes)
        
        if not files_to_process:
            print("✨ No changes detected - index is up to date!")
            return len(files), 0, time.time() - start_time
        
        print(f"📝 Found {len(files_to_process)} changed files out of {len(files)} total")
    
    # Progress reporting function
    def show_progress(completed, total, message):
        percent = (completed / total) * 100 if total > 0 else 0
        elapsed = time.time() - start_time
        
        if completed > 0:
            rate = completed / elapsed
            eta = (total - completed) / rate if rate > 0 else 0
            print(f"\r🧠 {message} ({percent:.1f}% | {rate:.1f} files/sec | ETA: {eta:.1f}s)", end="", flush=True)
        else:
            print(f"\r🧠 {message}", end="", flush=True)
    
    # Generate embeddings and store them in the database
    embed_and_store(con, embedder, files_to_process, max_workers, show_progress)
    print()  # New line after progress
    
    # Build the search index from all stored embeddings
    build_full_index(con)
    
    elapsed = time.time() - start_time
    return len(files), len(files_to_process), elapsed


def main():
    """
    Main entry point for the code indexing CLI application.

    This function sets up the command-line argument parser and dispatches
    to the appropriate functionality based on user input. The CLI supports
    three main commands:

    1. index: Scan a repository and build a searchable index
    2. search: Query the index for semantically similar code
    3. watch: Monitor a repository for changes and update index incrementally

    Each command has its own set of arguments and options for customization.
    """
    # Set up the main argument parser with enhanced help
    parser = argparse.ArgumentParser(
        prog="turboprop",
        description="""
🚀 Turboprop - Lightning-fast semantic code search & indexing

Transform your codebase into a searchable knowledge base using AI embeddings.
Find code by meaning, not just keywords. Perfect for code exploration,
documentation, and AI-assisted development.

Examples:
  turboprop index .                          # Index current directory
  turboprop search "JWT authentication"     # Find auth-related code
  turboprop watch ~/my-project              # Live-update index
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
💡 Pro Tips:
  • Use natural language: "error handling", "database connection"
  • Respects .gitignore automatically
  • Index persists between sessions (stored in code_index.duckdb)
  • Watch mode keeps index fresh as you code

🔗 More help: https://github.com/glamp/turboprop
        """
    )

    # Add version argument
    parser.add_argument(
        '--version', '-v',
        action='version',
        version=f'turboprop {get_version()}'
    )

    sub = parser.add_subparsers(
        dest="cmd",
        required=True,
        title="commands",
        description="Available operations",
        help="Run 'turboprop <command> --help' for detailed usage"
    )

    # 'index' command: Build initial index from repository
    p_i = sub.add_parser(
        "index",
        help="🔍 Build searchable index from repository",
        description="""
Build a semantic search index for a Git repository.

This scans all code files, generates AI embeddings for semantic understanding,
and creates a fast search index. The process respects .gitignore and only
indexes files with recognized code extensions.

The index is persistent and stored locally, so you only need to run this
once per repository (or when you want a full refresh).
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  turboprop index .                    # Index current directory
  turboprop index /path/to/project     # Index specific project
  turboprop index . --max-mb 2.0      # Allow larger files

Supported file types: .py, .js, .ts, .java, .go, .rs, .cpp, .h, .json, .yaml, etc.
        """
    )
    p_i.add_argument(
        "repo",
        help="Path to the Git repository to index"
    )
    p_i.add_argument(
        "--max-mb",
        type=float,
        default=1.0,
        metavar="SIZE",
        help="Maximum file size in MB to include (default: 1.0). "
             "Larger files are skipped to avoid memory issues."
    )
    p_i.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="NUM",
        help="Number of parallel workers for embedding generation (default: CPU count). "
             "More workers = faster processing but higher memory usage."
    )
    p_i.add_argument(
        "--force-all",
        action="store_true",
        help="Force reprocessing of all files, even if unchanged. "
             "Useful for model updates or index corruption recovery."
    )

    # 'search' command: Query the existing index
    p_s = sub.add_parser(
        "search",
        help="🔎 Search code using natural language",
        description="""
Search your indexed codebase using natural language queries.

This performs semantic search - finding code by meaning rather than exact
keyword matches. Ask questions like "how to parse JSON" or "authentication
middleware" to discover relevant code across your entire repository.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Query Examples:
  "JWT authentication"              → Find auth-related code
  "parse JSON response"             → Discover JSON parsing logic  
  "error handling middleware"       → Locate error handling patterns
  "database connection setup"       → Find DB initialization code
  "function to calculate tax"       → Search for specific functions
  "React component for forms"       → Find form-related components

💡 Tip: Use descriptive phrases rather than single keywords for best results.
        """
    )
    p_s.add_argument(
        "query",
        help="Natural language search query (e.g., 'function to parse JSON')"
    )
    p_s.add_argument(
        "--repo",
        default=".",
        help="Path to the Git repository to search (default: current directory)"
    )
    p_s.add_argument(
        "--k",
        type=int,
        default=5,
        metavar="NUM",
        help="Number of results to return (default: 5, max recommended: 20)"
    )

    # 'watch' command: Monitor repository for changes
    p_w = sub.add_parser(
        "watch",
        help="👀 Monitor repository and update index in real-time",
        description="""
Watch a repository for file changes and incrementally update the search index.

This mode runs continuously, monitoring for file modifications, additions,
and deletions. Changes are processed with intelligent debouncing to avoid
excessive work during rapid edits or build processes.

Perfect for keeping your search index fresh during active development.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Patterns:
  • Development mode: Run in background while coding
  • CI/CD integration: Auto-update index on deployments  
  • Team environments: Keep shared index synchronized

Press Ctrl+C to stop watching.

💡 Pro Tip: Higher debounce values reduce CPU usage during heavy file activity.
        """
    )
    p_w.add_argument(
        "repo",
        help="Path to the Git repository to watch"
    )
    p_w.add_argument(
        "--max-mb",
        type=float,
        default=1.0,
        metavar="SIZE",
        help="Maximum file size in MB to process (default: 1.0)"
    )
    p_w.add_argument(
        "--debounce-sec",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="Seconds to wait before processing changes (default: 5.0). "
             "Higher values reduce CPU usage during rapid file changes."
    )

    # 'mcp' command: Start MCP server
    p_m = sub.add_parser(
        "mcp",
        help="🔗 Start Model Context Protocol (MCP) server",
        description="""
Start a Model Context Protocol server for semantic code search.

This enables Claude and other MCP clients to search and index your codebase
using natural language queries. The server provides tools for indexing,
searching, and monitoring repositories.

Perfect for AI-assisted development workflows.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  turboprop mcp --repository .                     # Start MCP server for current directory
  turboprop mcp --repository /path/to/repo         # Index specific repository
  turboprop mcp --repository . --max-mb 2.0        # Allow larger files
  turboprop mcp --repository . --no-auto-index     # Don't auto-index on startup

💡 Pro Tip: Use with Claude Code or other MCP clients for AI-powered code exploration.
        """
    )
    p_m.add_argument(
        "--repository",
        default=".",
        help="Path to the repository to index and monitor (default: current directory)"
    )
    p_m.add_argument(
        "--max-mb",
        type=float,
        default=1.0,
        metavar="SIZE",
        help="Maximum file size in MB to process (default: 1.0)"
    )
    p_m.add_argument(
        "--debounce-sec",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="Seconds to wait before processing file changes (default: 5.0)"
    )
    p_m.add_argument(
        "--auto-index",
        action="store_true",
        default=True,
        help="Automatically index the repository on startup (default: True)"
    )
    p_m.add_argument(
        "--no-auto-index",
        action="store_false",
        dest="auto_index",
        help="Don't automatically index the repository on startup"
    )
    p_m.add_argument(
        "--auto-watch",
        action="store_true",
        default=True,
        help="Automatically watch for file changes (default: True)"
    )
    p_m.add_argument(
        "--no-auto-watch",
        action="store_false",
        dest="auto_watch",
        help="Don't automatically watch for file changes"
    )

    # Parse command line arguments
    args = parser.parse_args()

    # Add some visual flair for better UX
    print("🚀 Turboprop - Semantic Code Search")
    print("=" * 40)

    # Initialize the ML model (shared across all commands)
    print("⚡ Initializing AI model...")
    try:
        embedder = SentenceTransformer(EMBED_MODEL)
    except Exception as e:
        # Handle MPS/Metal Performance Shaders compatibility issues on Apple Silicon
        if "meta tensor" in str(e) or "to_empty" in str(e):
            print("🔧 Detected MPS compatibility issue, falling back to CPU...")
            import torch
            embedder = SentenceTransformer(EMBED_MODEL, device='cpu')
        else:
            raise e

    # Dispatch to appropriate command handler
    if args.cmd == "index":
        # Build full index from repository
        print(f"\n🔍 Indexing repository: {args.repo}")
        print(f"📁 Max file size: {args.max_mb} MB")
        if args.workers:
            print(f"👥 Using {args.workers} parallel workers")
        else:
            print(f"👥 Using {multiprocessing.cpu_count()} parallel workers (auto-detected)")
        
        if args.force_all:
            print("🔄 Force mode: reprocessing all files")

        repo_path = Path(args.repo).resolve()
        con = init_db(repo_path)

        # Use new smart reindex function
        total_files, processed_files, elapsed = reindex_all(
            repo_path, 
            int(args.max_mb * 1024**2), 
            con, 
            embedder,
            args.workers,
            args.force_all
        )

        if total_files == 0:
            print("❌ No code files found. Make sure you're in a Git repository with code files.")
            return

        print(f"🎉 Indexing complete!")
        print(f"📊 Files found: {total_files}")
        print(f"📊 Files processed: {processed_files}")
        print(f"⏱️  Time elapsed: {elapsed:.2f} seconds")
        
        if processed_files > 0:
            print(f"⚡ Processing rate: {processed_files/elapsed:.1f} files/second")
        
        embedding_count = build_full_index(con)
        print(f"💾 Database saved to: {repo_path / 'turboprop' / 'code_index.duckdb'}")
        print(f"🔍 Total embeddings in index: {embedding_count}")
        print("\n💡 Try searching: turboprop search \"your query here\"")

    elif args.cmd == "search":
        # Search the existing index
        print(f"\n🔎 Searching for: \"{args.query}\"")
        print(f"📊 Returning top {args.k} results...")

        repo_path = Path(args.repo).resolve()
        con = init_db(repo_path)

        try:
            results = search_index(con, embedder, args.query, args.k)
        except Exception as e:
            print(f"❌ Search failed: {e}")
            print("💡 Make sure you've built an index first: turboprop index <repo>")
            return

        if not results:
            print("❌ No results found.")
            print("💡 Try:")
            print("   • Building an index first: turboprop index <repo>")
            print("   • Using different search terms")
            print("   • Making sure your query describes code concepts")
            return

        # Display search results with enhanced formatting
        print(f"\n🎯 Found {len(results)} relevant results:\n")
        for i, (path, snippet, dist) in enumerate(results, 1):
            similarity_pct = (1 - dist) * 100
            print(f"┌─ {i}. {path}")
            print(f"│  📈 Similarity: {similarity_pct:.1f}%")
            print(f"│")
            # Clean up the snippet display
            clean_snippet = snippet.strip()[:200]
            if len(snippet) > 200:
                clean_snippet += "..."
            print(f"│  {clean_snippet}")
            print("└" + "─" * 60)
            print()

    elif args.cmd == "watch":
        # Start continuous monitoring mode
        print(f"\n👀 Starting watch mode for: {args.repo}")
        print(f"📁 Max file size: {args.max_mb} MB")
        print(f"⏱️  Debounce delay: {args.debounce_sec}s")
        print("🔄 Monitoring for changes... (Press Ctrl+C to stop)")

        try:
            watch_mode(args.repo, args.max_mb, args.debounce_sec)
        except KeyboardInterrupt:
            print("\n\n⏹️  Watch mode stopped by user")
        except Exception as e:
            print(f"\n❌ Watch mode failed: {e}")

    elif args.cmd == "mcp":
        # Start MCP server
        print(f"\n🔗 Starting MCP server...")
        print("📞 Invoking MCP server module...")
        
        # Import and run the MCP server with the provided arguments
        try:
            from mcp_server import main as mcp_main
            import sys
            
            # Build arguments for MCP server
            mcp_args = []
            if args.repository:
                mcp_args.extend(["--repository", args.repository])
            
            mcp_args.extend([
                "--max-mb", str(args.max_mb),
                "--debounce-sec", str(args.debounce_sec)
            ])
            
            if not args.auto_index:
                mcp_args.append("--no-auto-index")
            if not args.auto_watch:
                mcp_args.append("--no-auto-watch")
            
            # Override sys.argv for MCP server
            original_argv = sys.argv[:]
            sys.argv = ["turboprop-mcp"] + mcp_args
            
            try:
                mcp_main()
            finally:
                # Restore original argv
                sys.argv = original_argv
                
        except ImportError:
            print("❌ MCP server module not available.")
            print("💡 Make sure you've installed the MCP dependencies: pip install turboprop[mcp]")
            return
        except Exception as e:
            print(f"❌ MCP server failed to start: {e}")
            return

    if args.cmd != "mcp":
        print("\n✨ Done!")


if __name__ == "__main__":
    # Entry point when script is run directly (not imported as module)
    # This allows the script to be used both as a CLI tool and as a library
    main()
