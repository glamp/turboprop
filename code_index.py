#!/usr/bin/env python3
"""
code_index.py: Scan a Git repo, build a DuckDB-backed code index with embeddings
and HNSW (cosine similarity),
with CLI commands: index, search, watch (incremental, debounced).

This module provides a complete solution for indexing and searching code
repositories using:
- DuckDB for persistent storage of file content and embeddings
- SentenceTransformers for generating semantic embeddings of code files
- HNSWLib for fast approximate nearest neighbor search using cosine similarity
- Watchdog for real-time file system monitoring with debounced updates

The system supports three main operations:
1. index: Scan a repository and build the initial search index
2. search: Query the index for semantically similar code
3. watch: Monitor a repository for changes and incrementally update the index
"""

# Force CPU usage for PyTorch before any imports to avoid MPS issues on Apple Silicon
from apple_silicon_compat import setup_apple_silicon_compatibility

setup_apple_silicon_compatibility()

# Imports must be placed after environment variables are set for Apple Silicon MPS
# compatibility
# This ensures PyTorch uses CPU backend instead of problematic MPS backend
import argparse  # noqa: E402
import hashlib  # noqa: E402
import multiprocessing  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from tqdm import tqdm  # noqa: E402
from watchdog.events import FileSystemEventHandler  # noqa: E402
from watchdog.observers import Observer  # noqa: E402

from database_manager import DatabaseManager  # noqa: E402
from embedding_helper import EmbeddingGenerator  # noqa: E402
from logging_config import get_logger  # noqa: E402

# Global database manager instance
_db_manager = None
_db_manager_lock = threading.Lock()

# Logger instance
logger = get_logger(__name__)


def get_version():
    """Get the version of turboprop."""
    try:
        from turboprop._version import __version__

        return __version__
    except ImportError:
        return "0.2.2"


# Import configuration
from config import config

# Backward compatibility for tests
DB_PATH = "code_index.duckdb"


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
    Initialize or get the database manager instance.

    Args:
        repo_path: Path to the repository where the database should be stored.
                  If None, uses current directory (for backward compatibility)

    Returns:
        DatabaseManager instance
    """
    global _db_manager

    with _db_manager_lock:
        if _db_manager is None:
            # Create database in the repository directory or current directory
            if repo_path is None:
                # Use current directory with .turboprop subdirectory
                turboprop_dir = Path.cwd() / ".turboprop"
                turboprop_dir.mkdir(exist_ok=True)
                db_path = turboprop_dir / "code_index.duckdb"
            else:
                # Create .turboprop directory if it doesn't exist
                turboprop_dir = repo_path / ".turboprop"
                turboprop_dir.mkdir(exist_ok=True)
                db_path = turboprop_dir / "code_index.duckdb"

            _db_manager = DatabaseManager(db_path)

            # Initialize table schema with modification time tracking and metadata columns
            _db_manager.execute_with_retry(
                f"""
                CREATE TABLE IF NOT EXISTS {config.database.TABLE_NAME} (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT,
                    embedding DOUBLE[{config.embedding.DIMENSIONS}],
                    last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_mtime TIMESTAMP,
                    file_type VARCHAR,
                    language VARCHAR,
                    size_bytes INTEGER,
                    line_count INTEGER,
                    category VARCHAR
                )
            """
            )

            # Migrate schema to add any missing columns (including legacy and new metadata columns)
            try:
                _db_manager.migrate_schema(config.database.TABLE_NAME)
            except Exception as e:
                logger.warning(f"Schema migration failed, continuing with existing schema: {e}")

        return _db_manager


def scan_repo(repo_path: Path, max_bytes: int):
    """
    Scan a Git repository for code files, respecting .gitignore rules.

    This function uses a more efficient approach:
    1. Gets all tracked files from Git
    2. Gets all untracked files that are not ignored
    3. Filters by code extensions and size

    Args:
        repo_path: Path to the Git repository root
        max_bytes: Maximum file size in bytes to include in indexing

    Returns:
        List of Path objects for files that should be indexed
    """
    # Resolve to absolute path to avoid issues with relative paths
    repo_path = repo_path.resolve()

    files = []

    try:
        # Get all tracked files (already in git)
        tracked_result = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        tracked_files = {
            line.strip() for line in tracked_result.stdout.splitlines() if line.strip()
        }

        # Get all untracked files that are not ignored
        untracked_result = subprocess.run(
            ["git", "ls-files", "--exclude-standard", "--others"],
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        untracked_files = {
            line.strip() for line in untracked_result.stdout.splitlines() if line.strip()
        }

        # Combine tracked and untracked files
        all_files = tracked_files | untracked_files

    except subprocess.CalledProcessError:
        # If git commands fail (not a git repo, etc.), fall back to simple file walk
        all_files = set()
        for root, _, names in os.walk(repo_path):
            for name in names:
                rel_path = Path(root).relative_to(repo_path) / name
                all_files.add(str(rel_path))

    # Process the files
    for file_path_str in all_files:
        p = repo_path / file_path_str

        try:
            # Skip files that are too large to process efficiently
            if p.stat().st_size > max_bytes:
                continue
        except OSError:
            # Skip files we can't read (permissions, broken symlinks, etc.)
            continue

        files.append(p)

    return files


def get_existing_file_hashes(db_manager):
    """
    Retrieve existing file hashes from the database for smart re-indexing.

    Returns:
        Dict mapping file paths to their content hashes
    """
    try:
        result = db_manager.execute_with_retry(f"SELECT path, id FROM {config.database.TABLE_NAME}")
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


def embed_and_store(db_manager, embedder, files, max_workers=None, progress_callback=None):
    """
    Process a list of files by generating embeddings and storing them in the database.
    Uses sequential processing for reliability.

    Args:
        db_manager: DatabaseManager instance
        embedder: EmbeddingGenerator instance for generating embeddings
        files: List of Path objects to process
        max_workers: Number of parallel workers (ignored - always sequential)
        progress_callback: Function to call with progress updates (deprecated, use tqdm)
    """
    if not files:
        return

    rows = []
    failed_files = []

    # Use sequential processing for reliability
    with tqdm(total=len(files), desc="🔍 Generating embeddings", unit="files") as pbar:
        for path in files:
            try:
                text = path.read_text(encoding="utf-8")
                uid = compute_id(str(path) + text)

                emb = embedder.encode(text)
                # Get file modification time
                import datetime

                file_mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime)
                rows.append((uid, str(path), text, emb.tolist(), file_mtime))

            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
                failed_files.append(path)

            pbar.update(1)

            # Maintain backward compatibility with progress_callback
            if progress_callback:
                progress_callback(pbar.n, len(files), f"Processed {pbar.n}/{len(files)} files")

    # Report processing results
    if failed_files:
        print(
            f"⚠️  Failed to process {len(failed_files)} files " f"out of {len(files)} total",
            file=sys.stderr,
        )

    # Insert all rows in a single batch operation for better performance
    if rows:
        operations = [
            (
                (
                    f"INSERT OR REPLACE INTO {config.database.TABLE_NAME} "
                    f"(id, path, content, embedding, file_mtime) VALUES (?, ?, ?, ?, ?)"
                ),
                row,
            )
            for row in rows
        ]
        db_manager.execute_transaction(operations)
        print(f"✅ Successfully processed {len(rows)} files", file=sys.stderr)


def build_full_index(db_manager):
    """
    Verify that the database contains embeddings for search operations.

    With DuckDB vector operations, no separate index file is needed.
    This function just validates that embeddings exist in the database.

    Args:
        db_manager: DatabaseManager instance

    Returns:
        Number of embeddings in the database, or 0 if none found
    """
    # Check if we have any embeddings in the database
    result = db_manager.execute_with_retry(
        f"SELECT COUNT(*) FROM {config.database.TABLE_NAME} WHERE embedding IS NOT NULL"
    )
    return result[0][0] if result and result[0] else 0


def search_index(db_manager, embedder, query: str, k: int):
    """
    Search the code index for files semantically similar to the given query.

    This function:
    1. Generates an embedding for the search query
    2. Uses DuckDB's array operations to compute cosine similarity
    3. Returns the top k most similar files with their similarity scores

    Args:
        db_manager: DatabaseManager instance
        embedder: SentenceTransformer model for generating query embedding
        query: Text query to search for (e.g., "function to parse JSON")
        k: Number of top results to return

    Returns:
        List of tuples: (file_path, content_snippet, distance_score)
        Distance is cosine distance (lower = more similar)
    """
    # Generate embedding for the search query
    q_emb = embedder.encode(query).tolist()

    # Use DuckDB's array operations for cosine similarity search
    # Cosine similarity = dot(a,b) / (norm(a) * norm(b))
    # Cosine distance = 1 - cosine similarity
    with db_manager.get_connection() as con:
        results = con.execute(
            f"""
            SELECT
                path,
                substr(content, 1, 300) as snippet,
                1 - (list_dot_product(embedding, $1) /
                    (sqrt(list_dot_product(embedding, embedding)) *
                     sqrt(list_dot_product($1, $1)))) as distance
            FROM {config.database.TABLE_NAME}
            WHERE embedding IS NOT NULL
            ORDER BY distance ASC
            LIMIT {k}
        """,
            [q_emb],
        ).fetchall()

    return results


def embed_and_store_single(db_manager, embedder, path: Path):
    """
    Process a single file for embedding and storage (used for incremental updates).

    This is similar to embed_and_store() but handles just one file, making it
    suitable for real-time updates when files change during watch mode.

    Args:
        db_manager: DatabaseManager instance
        embedder: SentenceTransformer model for generating embeddings
        path: Path to the single file to process

    Returns:
        Tuple of (unique_id, embedding) if successful,
        None if file couldn't be processed
    """
    try:
        # Read the file content
        text = path.read_text(encoding="utf-8")
    except Exception:
        # Return None if file can't be read
        return None

    # Generate unique ID and embedding
    uid = compute_id(str(path) + text)
    emb = embedder.encode(text)

    # Get file modification time
    import datetime

    file_mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime)

    # Use transaction for database operations
    operations = [
        (f"DELETE FROM {config.database.TABLE_NAME} WHERE path = ?", (str(path),)),
        (
            (
                f"INSERT INTO {config.database.TABLE_NAME} (id, path, content, embedding, "
                f"file_mtime) VALUES (?, ?, ?, ?, ?)"
            ),
            (uid, str(path), text, emb.tolist(), file_mtime),
        ),
    ]
    db_manager.execute_transaction(operations)

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

    def __init__(self, repo: Path, max_bytes: int, db_manager, embedder, debounce_sec: float):
        """
        Initialize the debounced file handler.

        Args:
            repo: Path to the repository being watched
            max_bytes: Maximum file size to process
            db_manager: DatabaseManager instance
            embedder: SentenceTransformer model for embeddings
            debounce_sec: Seconds to wait before processing file changes
        """
        self.repo = repo
        self.max_bytes = max_bytes
        self.db_manager = db_manager
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

        # Ignore directory events
        if event.is_directory:
            return

        # For deletions, we should always process to remove from database
        # For creates/modifies, we'll check if file should be indexed in _process
        # (No extension filtering - index all Git-tracked files)

        # Cancel any existing timer for this file (debouncing logic)
        if p in self.timers:
            self.timers[p].cancel()

        # Start a new timer that will process this file after the debounce period
        timer = threading.Timer(self.debounce, self._process, args=(event.event_type, p))
        self.timers[p] = timer
        timer.start()

    def _should_index_file(self, file_path: Path) -> bool:
        """
        Check if a file should be indexed using git internals to respect .gitignore.

        Args:
            file_path: Path to the file to check

        Returns:
            True if the file should be indexed, False otherwise
        """
        # Index all files that are tracked by Git (no extension filtering)

        # Check if file is too large
        try:
            if file_path.stat().st_size > self.max_bytes:
                return False
        except OSError:
            return False

        # Use git to check if file should be ignored
        try:
            # Get relative path from repo root
            rel_path = file_path.relative_to(self.repo)

            # Check if git would ignore this file
            result = subprocess.run(
                ["git", "check-ignore", str(rel_path)],
                cwd=self.repo,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False
            )
            # If git check-ignore returns 0, the file is ignored
            if result.returncode == 0:
                return False

        except (subprocess.CalledProcessError, ValueError):
            # If git command fails or file is outside repo, use basic checks
            pass

        return True

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
            # Check if this file should be indexed (respects .gitignore)
            if self._should_index_file(p):
                try:
                    # Update database with new file content and embedding
                    res = embed_and_store_single(self.db_manager, self.embedder, p)
                    if res:
                        uid, emb = res
                        # With DuckDB, no separate index update needed
                        logger.debug(f"[debounce] updated {p}")
                except OSError:
                    # Ignore files we can't access (permissions, etc.)
                    pass
        elif ev_type == "deleted":
            # Remove deleted file from database
            self.db_manager.execute_with_retry(
                f"DELETE FROM {config.database.TABLE_NAME} WHERE path = ?", (str(p),)
            )
            logger.debug(f"[debounce] {p} deleted from database")


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
    db_manager = init_db(repo)

    try:
        embedder = EmbeddingGenerator(config.embedding.EMBED_MODEL)
        logger.info("Embedding generator initialized for watch mode")
    except Exception as e:
        logger.error(f"Failed to initialize embedding generator: {e}")
        raise

    # Verify database has embeddings
    embedding_count = build_full_index(db_manager)
    logger.info(f"[watch] found {embedding_count} embeddings in database")

    # Set up the debounced file handler
    handler = DebouncedHandler(repo, max_bytes, db_manager, embedder, debounce_sec)

    # Create and configure the file system observer
    observer = Observer()
    observer.schedule(handler, str(repo), recursive=True)

    print(f"[watch] watching {repo} (max {max_mb} MB, debounce {debounce_sec}s)")
    observer.start()

    try:
        # Keep the watcher running until user interrupts
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Clean shutdown on Ctrl+C
        observer.stop()
    finally:
        observer.join()
        # Clean up database connections
        db_manager.cleanup()


def print_indexed_files_tree(files, repo_path: Path):
    """
    Print a tree-like structure of indexed files, similar to the `tree` command.

    Args:
        files: List of Path objects for indexed files
        repo_path: Path to the repository root
    """
    if not files:
        print("📁 No files indexed.")
        return

    print(f"📂 Indexed files in {repo_path}:")
    print("=" * 60)

    # Convert to relative paths and sort
    relative_files = []
    for file_path in files:
        try:
            rel_path = file_path.relative_to(repo_path)
            relative_files.append(rel_path)
        except ValueError:
            # File is outside repo, use absolute path
            relative_files.append(file_path)

    relative_files.sort()

    # Group files by directory
    directories = {}
    for file_path in relative_files:
        parts = file_path.parts
        if len(parts) == 1:
            # File in root
            directories.setdefault("", []).append(file_path.name)
        else:
            # File in subdirectory
            dir_path = str(Path(*parts[:-1]))
            directories.setdefault(dir_path, []).append(parts[-1])

    # Print the tree structure
    for dir_path in sorted(directories.keys()):
        if dir_path:
            print(f"📁 {dir_path}/")
            prefix = "  "
        else:
            prefix = ""

        files_in_dir = directories[dir_path]
        for i, filename in enumerate(sorted(files_in_dir)):
            is_last = i == len(files_in_dir) - 1
            if dir_path:
                connector = "└── " if is_last else "├── "
                print(f"{prefix}{connector}📄 {filename}")
            else:
                print(f"📄 {filename}")

    print("=" * 60)
    print(f"📊 Total: {len(files)} files indexed")


def remove_orphaned_files(db_manager, current_files):
    """
    Remove database entries for files that no longer exist in the repository.

    Args:
        db_manager: DatabaseManager instance
        current_files: List of Path objects for files currently in the repo

    Returns:
        Number of orphaned entries removed
    """
    # Get all paths currently in the database
    existing_paths = set()
    try:
        result = db_manager.execute_with_retry(f"SELECT path FROM {config.database.TABLE_NAME}")
        existing_paths = {row[0] for row in result}
    except Exception:
        return 0

    # Convert current files to string paths for comparison
    current_paths = {str(path) for path in current_files}

    # Find orphaned paths (in database but not in current files)
    orphaned_paths = existing_paths - current_paths

    if not orphaned_paths:
        return 0

    # Remove orphaned entries
    removed_count = 0
    operations = []
    for path in orphaned_paths:
        operations.append((f"DELETE FROM {config.database.TABLE_NAME} WHERE path = ?", (path,)))
        removed_count += 1

    if operations:
        db_manager.execute_transaction(operations)
        print(f"🗑️  Removed {removed_count} orphaned files from index")

    return removed_count


def get_last_index_time(db_manager):
    """
    Get the timestamp of the most recent indexing operation.

    Args:
        db_manager: DatabaseManager instance

    Returns:
        datetime or None if no files are indexed
    """
    try:
        result = db_manager.execute_with_retry(f"SELECT MAX(last_modified) FROM {config.database.TABLE_NAME}")
        if result and result[0] and result[0][0]:
            return result[0][0]
        return None
    except Exception:
        return None


def get_current_files_safely(repo_path: Path, max_bytes: int):
    """Get current files from repository with error handling."""
    current_files = scan_repo(repo_path, max_bytes)
    if not current_files:
        return None, "No files found in repository"
    return current_files, None


def get_existing_files_from_db(db_manager):
    """Get existing files from database with error handling."""
    try:
        result = db_manager.execute_with_retry(f"SELECT path, file_mtime FROM {config.database.TABLE_NAME}")
        return {row[0]: row[1] for row in result}, None
    except Exception:
        return None, "Database error - need to rebuild index"


def check_file_count_changed(current_files, existing_files):
    """Check if file counts have changed."""
    if len(current_files) != len(existing_files):
        file_diff = len(current_files) - len(existing_files)
        return True, f"File count changed ({file_diff:+d} files)", abs(file_diff)
    return False, None, 0


def find_changed_files(current_files, existing_files):
    """Find files that have been modified or added."""
    changed_files = []
    for file_path in current_files:
        try:
            file_mtime = file_path.stat().st_mtime
            str_path = str(file_path)

            # File is new if not in database
            if str_path not in existing_files:
                changed_files.append(str_path)
                continue

            # Check if file has been modified
            db_mtime = existing_files[str_path]
            if db_mtime is None or file_mtime > db_mtime.timestamp():
                changed_files.append(str_path)

        except OSError:
            # File might have been deleted, count as changed
            changed_files.append(str(file_path))

    return changed_files


def has_repository_changed(repo_path: Path, max_bytes: int, db_manager):
    """
    Check if any files in the repository have changed since the last indexing.

    This function provides a fast way to determine if reindexing is needed by:
    1. Comparing the number of files in the repo vs database
    2. Checking if any files have newer modification times than the last index
    3. Detecting new files that aren't in the database

    Args:
        repo_path: Path to the repository
        max_bytes: Maximum file size to consider
        db_manager: DatabaseManager instance

    Returns:
        tuple: (has_changed: bool, reason: str, changed_count: int)
    """
    try:
        # Get current files in repository
        current_files, error = get_current_files_safely(repo_path, max_bytes)
        if error:
            return False, error, 0

        # Get existing file info from database
        existing_files, error = get_existing_files_from_db(db_manager)
        if error:
            return True, error, len(current_files)

        # Check if file counts differ
        count_changed, reason, count = check_file_count_changed(current_files, existing_files)
        if count_changed:
            return True, reason, count

        # Check modification times
        changed_files = find_changed_files(current_files, existing_files)
        if changed_files:
            return True, "Files modified or added", len(changed_files)

        return False, "Repository is up to date", 0

    except Exception as e:
        return True, f"Error checking repository: {str(e)}", 0


def check_index_freshness(repo_path: Path, max_bytes: int, db_manager):
    """
    Check if the index is fresh and up-to-date.

    This is a comprehensive check that determines if reindexing is needed.

    Args:
        repo_path: Path to the repository
        max_bytes: Maximum file size to consider
        db_manager: DatabaseManager instance

    Returns:
        dict: {
            'is_fresh': bool,
            'reason': str,
            'last_index_time': datetime or None,
            'changed_files': int,
            'total_files': int
        }
    """
    # Check if index exists
    try:
        file_count = db_manager.execute_with_retry(f"SELECT COUNT(*) FROM {config.database.TABLE_NAME}")[0][0]
    except Exception:
        return {
            "is_fresh": False,
            "reason": "No index found",
            "last_index_time": None,
            "changed_files": 0,
            "total_files": 0,
        }

    if file_count == 0:
        return {
            "is_fresh": False,
            "reason": "Index is empty",
            "last_index_time": None,
            "changed_files": 0,
            "total_files": 0,
        }

    # Get last index time
    last_index_time = get_last_index_time(db_manager)

    # Check if repository has changed
    has_changed, reason, changed_count = has_repository_changed(repo_path, max_bytes, db_manager)

    # Count current files
    current_files = scan_repo(repo_path, max_bytes)

    return {
        "is_fresh": not has_changed,
        "reason": reason,
        "last_index_time": last_index_time,
        "changed_files": changed_count,
        "total_files": len(current_files),
    }


def reindex_all(
    repo_path: Path,
    max_bytes: int,
    db_manager,
    embedder,
    max_workers=None,
    force_all=False,
):
    """
    Intelligently reindex a repository by only processing changed files.

    This function provides a smart reindexing workflow that:
    1. Scans repository for all eligible code files
    2. Compares with existing database to find changed/new files
    3. Only processes files that have actually changed
    4. Removes orphaned database entries for deleted files
    5. Uses parallel processing for faster embedding generation

    Args:
        repo_path: Path to the repository root directory
        max_bytes: Maximum file size in bytes to process
        db_manager: DatabaseManager instance
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
        # Even if no files, we should clean up orphaned entries
        removed_count = remove_orphaned_files(db_manager, [])
        return 0, 0, 0

    # Remove orphaned files from database (files that no longer exist in repo)
    removed_count = remove_orphaned_files(db_manager, files)

    # Smart re-indexing: only process changed files
    files_to_process = files
    if not force_all:
        print("🧠 Checking for changed files...")
        existing_hashes = get_existing_file_hashes(db_manager)
        files_to_process = filter_changed_files(files, existing_hashes)

        if not files_to_process and removed_count == 0:
            print("✨ No changes detected - index is up to date!")
            return len(files), 0, time.time() - start_time

        if files_to_process:
            print(f"📝 Found {len(files_to_process)} changed files " f"out of {len(files)} total")

    # Generate embeddings and store them in the database
    if files_to_process:
        # Use the enhanced function from indexing_operations which includes construct extraction
        from indexing_operations import embed_and_store as embed_and_store_enhanced
        embed_and_store_enhanced(db_manager, embedder, files_to_process, extract_constructs=True)

    # Build the search index from all stored embeddings
    build_full_index(db_manager)

    # Print the tree structure of indexed files
    print_indexed_files_tree(files, repo_path)

    elapsed = time.time() - start_time
    return len(files), len(files_to_process), elapsed


def _setup_base_parser() -> argparse.ArgumentParser:
    """Set up the base argument parser."""
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
        """,
    )

    # Add version argument
    parser.add_argument("--version", "-v", action="version", version=f"turboprop {get_version()}")

    return parser

def _setup_index_subcommand(sub):
    """Set up the index subcommand and its arguments."""

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
        """,
    )
    p_i.add_argument("repo", help="Path to the Git repository to index")
    p_i.add_argument(
        "--max-mb",
        type=float,
        default=1.0,
        metavar="SIZE",
        help=(
            "Maximum file size in MB to include (default: 1.0). "
            "Larger files are skipped to avoid memory issues."
        ),
    )
    p_i.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="NUM",
        help="Number of parallel workers for embedding generation "
        "(default: CPU count). "
        "More workers = faster processing but higher memory usage.",
    )
    p_i.add_argument(
        "--force-all",
        action="store_true",
        help="Force reprocessing of all files, even if unchanged. "
        "Useful for model updates or index corruption recovery.",
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
        """,
    )
    p_s.add_argument("query", help="Natural language search query (e.g., 'function to parse JSON')")
    p_s.add_argument(
        "--repo",
        default=".",
        help="Path to the Git repository to search (default: current directory)",
    )
    p_s.add_argument(
        "--k",
        type=int,
        default=5,
        metavar="NUM",
        help="Number of results to return (default: 5, max recommended: 20)",
    )
    p_s.add_argument(
        "--mode",
        choices=["auto", "hybrid", "semantic", "text"],
        default="auto",
        help="Search mode: 'auto' (intelligent routing), 'hybrid' (semantic+text), "
             "'semantic' (embeddings only), 'text' (exact matching only)",
    )
    p_s.add_argument(
        "--enable-advanced",
        action="store_true",
        help="Enable advanced features like regex search, wildcards, and file type filtering"
    )
    p_s.add_argument(
        "--semantic-weight",
        type=float,
        default=0.6,
        metavar="WEIGHT",
        help="Weight for semantic search in hybrid mode (0.0-1.0, default: 0.6)"
    )
    p_s.add_argument(
        "--text-weight",
        type=float,
        default=0.4,
        metavar="WEIGHT",
        help="Weight for text search in hybrid mode (0.0-1.0, default: 0.4)"
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
        """,
    )
    p_w.add_argument("repo", help="Path to the Git repository to watch")
    p_w.add_argument(
        "--max-mb",
        type=float,
        default=1.0,
        metavar="SIZE",
        help="Maximum file size in MB to process (default: 1.0)",
    )
    p_w.add_argument(
        "--debounce-sec",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="Seconds to wait before processing changes (default: 5.0). "
        "Higher values reduce CPU usage during rapid file changes.",
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
  turboprop mcp --repository .                     # Start MCP server for current dir
  turboprop mcp --repository /path/to/repo         # Index specific repository
  turboprop mcp --repository . --max-mb 2.0        # Allow larger files
  turboprop mcp --repository . --no-auto-index     # Don't auto-index on startup

💡 Pro Tip: Use with Claude Code or other MCP clients for AI-powered code exploration.
        """,
    )
    p_m.add_argument(
        "--repository",
        default=".",
        help="Path to the repository to index and monitor (default: current directory)",
    )
    p_m.add_argument(
        "--max-mb",
        type=float,
        default=1.0,
        metavar="SIZE",
        help="Maximum file size in MB to process (default: 1.0)",
    )
    p_m.add_argument(
        "--debounce-sec",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="Seconds to wait before processing file changes (default: 5.0)",
    )
    p_m.add_argument(
        "--auto-index",
        action="store_true",
        default=True,
        help="Automatically index the repository on startup (default: True)",
    )
    p_m.add_argument(
        "--no-auto-index",
        action="store_false",
        dest="auto_index",
        help="Don't automatically index the repository on startup",
    )
    p_m.add_argument(
        "--auto-watch",
        action="store_true",
        default=True,
        help="Automatically watch for file changes (default: True)",
    )
    p_m.add_argument(
        "--no-auto-watch",
        action="store_false",
        dest="auto_watch",
        help="Don't automatically watch for file changes",
    )

    return parser


def initialize_embedder():
    """Initialize the embedding generator with error handling."""
    print("⚡ Initializing AI model...")
    try:
        embedder = EmbeddingGenerator(config.embedding.EMBED_MODEL)
        logger.info("Embedding generator initialized successfully")
        return embedder
    except Exception as e:
        logger.error(f"Failed to initialize embedding generator: {e}")
        raise


def handle_index_command(args, embedder):
    """Handle the index command logic."""
    print(f"\n🔍 Indexing repository: {args.repo}")
    print(f"📁 Max file size: {args.max_mb} MB")
    if args.workers:
        print(f"👥 Using {args.workers} parallel workers")
    else:
        auto_workers = min(4, multiprocessing.cpu_count())
        print(f"👥 Using {auto_workers} parallel workers (auto-detected)")

    if args.force_all:
        print("🔄 Force mode: reprocessing all files")

    repo_path = Path(args.repo).resolve()
    db_manager = init_db(repo_path)
    max_bytes = int(args.max_mb * 1024**2)

    # Check index freshness and provide user feedback
    if not args.force_all:
        print("\n🔍 Checking index freshness...")
        freshness = check_index_freshness(repo_path, max_bytes, db_manager)
        print(f"📊 {freshness['reason']}")
        print(f"📁 Repository files: {freshness['total_files']}")

        if freshness["is_fresh"]:
            print("✨ Index is up-to-date!")
            if freshness["last_index_time"]:
                print(f"📅 Last indexed: {freshness['last_index_time']}")
            print("💡 Use --force-all to reindex anyway")
            return
        elif freshness["changed_files"] > 0:
            print(f"🔄 Found {freshness['changed_files']} changed files")
    else:
        print("\n🔄 Force reindexing all files...")

    # Use new smart reindex function
    total_files, processed_files, elapsed = reindex_all(
        repo_path, max_bytes, db_manager, embedder, args.workers, args.force_all
    )

    if total_files == 0:
        print("❌ No code files found. Make sure you're in a Git repository " "with code files.")
        return

    print("🎉 Indexing complete!")
    print(f"📊 Files found: {total_files}")
    print(f"📊 Files processed: {processed_files}")
    print(f"⏱️  Time elapsed: {elapsed:.2f} seconds")

    if processed_files > 0:
        print(f"⚡ Processing rate: {processed_files/elapsed:.1f} files/second")

    embedding_count = build_full_index(db_manager)
    print(f"💾 Database saved to: {repo_path / '.turboprop' / 'code_index.duckdb'}")
    print(f"🔍 Total embeddings in index: {embedding_count}")
    print('\n💡 Try searching: turboprop search "your query here"')

    # Clean up database connections
    db_manager.cleanup()


def handle_search_command(args, embedder):
    """Handle the search command logic with hybrid search support."""
    print(f'\n🔎 Searching for: "{args.query}"')
    print(f"📊 Mode: {args.mode} | Results: {args.k}")
    if args.mode == "hybrid":
        print(f"⚖️  Weights: semantic={args.semantic_weight}, text={args.text_weight}")

    repo_path = Path(args.repo).resolve()
    db_manager = init_db(repo_path)

    try:
        # Import hybrid search functions
        from search_operations import search_with_intelligent_routing, search_with_hybrid_fusion

        # Create fusion weights if using hybrid mode
        fusion_weights = None
        if args.mode == "hybrid":
            fusion_weights = {
                'semantic_weight': args.semantic_weight,
                'text_weight': args.text_weight,
                'rrf_k': 60,
                'boost_exact_matches': True,
                'exact_match_boost': 1.5
            }

        # Use intelligent routing for auto mode, hybrid fusion for others
        if args.mode == "auto":
            results = search_with_intelligent_routing(
                db_manager, embedder, args.query, args.k, args.enable_advanced
            )
        else:
            results = search_with_hybrid_fusion(
                db_manager, embedder, args.query, args.k, args.mode,
                fusion_weights, enable_query_expansion=True
            )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        print("💡 Make sure you've built an index first: turboprop index <repo>")
        return
    finally:
        # Clean up database connections
        db_manager.cleanup()

    if not results:
        print("❌ No results found.")
        print("💡 Try:")
        print("   • Building an index first: turboprop index <repo>")
        print("   • Using different search terms")
        print("   • Trying a different search mode")
        print("   • Making sure your query describes code concepts")
        return

    # Use hybrid search result formatting for enhanced display
    from search_operations import format_hybrid_search_results
    formatted_results = format_hybrid_search_results(
        results, args.query, show_fusion_details=(args.mode == "hybrid"),
        repo_path=str(repo_path)
    )
    print(formatted_results)


def handle_watch_command(args):
    """Handle the watch command logic."""
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


def handle_mcp_command(args):
    """Handle the MCP command logic."""
    print("\n🔗 Starting MCP server...")
    print("📞 Invoking MCP server module...")

    # Import and run the MCP server with the provided arguments
    try:
        import sys

        from mcp_server import main as mcp_main

        # Build arguments for MCP server
        mcp_args = []
        if args.repository:
            mcp_args.extend(["--repository", args.repository])

        mcp_args.extend(["--max-mb", str(args.max_mb), "--debounce-sec", str(args.debounce_sec)])

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
        print("💡 Make sure you've installed the MCP dependencies: " "pip install turboprop[mcp]")
        return
    except Exception as e:
        print(f"❌ MCP server failed to start: {e}")
        return


def main():
    """Main entry point for the code indexing CLI application."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Add some visual flair for better UX
    print("🚀 Turboprop - Semantic Code Search")
    print("=" * 40)

    # Initialize the ML model (shared across all commands)
    embedder = initialize_embedder()

    # Dispatch to appropriate command handler
    if args.cmd == "index":
        handle_index_command(args, embedder)

    elif args.cmd == "search":
        handle_search_command(args, embedder)

    elif args.cmd == "watch":
        handle_watch_command(args)

    elif args.cmd == "mcp":
        handle_mcp_command(args)

    if args.cmd != "mcp":
        print("\n✨ Done!")


if __name__ == "__main__":
    # Entry point when script is run directly (not imported as module)
    # This allows the script to be used both as a CLI tool and as a library
    main()
