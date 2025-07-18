#!/usr/bin/env python3
"""
indexing_operations.py: Core indexing functionality for the Turboprop code search
system.

This module contains the core functions for building and maintaining the searchable
code index:
- File scanning and processing
- Embedding generation and storage
- Database operations for code indexing
- Index validation and maintenance
"""

import datetime
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from tqdm import tqdm

from database_manager import DatabaseManager
from embedding_helper import EmbeddingGenerator

# Constants
TABLE_NAME = "code_files"
DIMENSIONS = 384
EMBED_MODEL = "all-MiniLM-L6-v2"


def compute_id(text: str) -> str:
    """
    Compute a unique identifier for file content using SHA-256 hash.

    Args:
        text: The text content to hash

    Returns:
        A SHA-256 hash string
    """
    return hashlib.sha256(text.encode()).hexdigest()


def scan_repo(repo_path: Path, max_bytes: int) -> List[Path]:
    """
    Scan a Git repository to find all tracked files within size limits.

    This function uses 'git ls-files' to get all files tracked by Git,
    then filters them by size. It respects .gitignore and only includes
    files that are actually tracked by Git.

    Args:
        repo_path: Path to the Git repository
        max_bytes: Maximum file size in bytes to include

    Returns:
        List of Path objects for files to be indexed

    Raises:
        subprocess.CalledProcessError: If git command fails
        FileNotFoundError: If git is not found
    """
    if not (repo_path / ".git").exists():
        raise ValueError(f"Not a Git repository: {repo_path}")

    # Get all files tracked by git
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )

    # Filter files by size and existence
    files = []
    git_files = result.stdout.strip().split("\n")

    if not git_files or git_files == [""]:
        return files

    # Use tqdm for progress tracking
    with tqdm(total=len(git_files), desc="🔍 Scanning repository", unit="files") as pbar:
        for filename in git_files:
            if not filename:
                continue

            file_path = repo_path / filename

            # Skip if file doesn't exist (could be in .gitignore)
            if not file_path.exists():
                pbar.update(1)
                continue

            # Skip if file is too large
            try:
                if file_path.stat().st_size > max_bytes:
                    pbar.update(1)
                    continue
            except OSError:
                # Skip files that can't be stat'd
                pbar.update(1)
                continue

            files.append(file_path)
            pbar.update(1)

    return files


def get_existing_file_hashes(db_manager: DatabaseManager) -> Dict[str, str]:
    """
    Get existing file hashes from the database.

    Args:
        db_manager: DatabaseManager instance

    Returns:
        Dictionary mapping file paths to their content hashes
    """
    try:
        rows = db_manager.execute_with_retry(f"SELECT path, id FROM {TABLE_NAME}")
        return {row[0]: row[1] for row in rows}
    except Exception:
        return {}


def filter_changed_files(files: List[Path], existing_hashes: Dict[str, str]) -> List[Path]:
    """
    Filter files to only include those that have changed since last indexing.

    Args:
        files: List of file paths to check
        existing_hashes: Dictionary of existing file hashes

    Returns:
        List of files that have changed or are new
    """
    changed_files = []

    for file_path in files:
        try:
            # Compute current hash
            content = file_path.read_text(encoding="utf-8")
            current_hash = compute_id(str(file_path) + content)

            # Check if file is new or changed
            if str(file_path) not in existing_hashes or existing_hashes[str(file_path)] != current_hash:
                changed_files.append(file_path)

        except Exception as e:
            print(f"⚠️  Could not read {file_path}: {e}", file=sys.stderr)
            continue

    return changed_files


def embed_and_store(
    db_manager: DatabaseManager,
    embedder: EmbeddingGenerator,
    files: List[Path],
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
) -> None:
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
                file_mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime)
                rows.append((uid, str(path), text, emb.tolist(), file_mtime))

            except Exception as e:
                print(f"⚠️  Failed to process {path}: {e}", file=sys.stderr)
                failed_files.append(path)

            pbar.update(1)

            # Maintain backward compatibility with progress_callback
            if progress_callback:
                progress_callback(pbar.n, len(files), f"Processed {pbar.n}/{len(files)} files")

    # Report processing results
    if failed_files:
        print(
            f"⚠️  Failed to process {len(failed_files)} files out of " f"{len(files)} total",
            file=sys.stderr,
        )

    # Insert all rows in a single batch operation for better performance
    if rows:
        operations = [
            (
                (
                    f"INSERT OR REPLACE INTO {TABLE_NAME} "
                    f"(id, path, content, embedding, file_mtime) VALUES (?, ?, ?, ?, ?)"
                ),
                row,
            )
            for row in rows
        ]
        db_manager.execute_transaction(operations)
        print(f"✅ Successfully processed {len(rows)} files", file=sys.stderr)


def build_full_index(db_manager: DatabaseManager) -> int:
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
    result = db_manager.execute_with_retry(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE embedding IS NOT NULL")
    return result[0][0] if result and result[0] else 0


def embed_and_store_single(db_manager: DatabaseManager, embedder: EmbeddingGenerator, path: Path) -> bool:
    """
    Process a single file by generating embeddings and storing them in the database.

    Args:
        db_manager: DatabaseManager instance
        embedder: EmbeddingGenerator instance for generating embeddings
        path: Path object for the file to process

    Returns:
        True if successful, False otherwise
    """
    try:
        text = path.read_text(encoding="utf-8")
        uid = compute_id(str(path) + text)

        emb = embedder.encode(text)
        # Get file modification time
        file_mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime)

        # Insert into database
        db_manager.execute_with_retry(
            f"INSERT OR REPLACE INTO {TABLE_NAME} "
            f"(id, path, content, embedding, file_mtime) VALUES (?, ?, ?, ?, ?)",
            (uid, str(path), text, emb.tolist(), file_mtime),
        )
        return True

    except Exception as e:
        print(f"⚠️  Failed to process {path}: {e}", file=sys.stderr)
        return False


def remove_orphaned_files(db_manager: DatabaseManager, current_files: List[Path]) -> int:
    """
    Remove files from the database that no longer exist in the repository.

    Args:
        db_manager: DatabaseManager instance
        current_files: List of files currently in the repository

    Returns:
        Number of orphaned files removed
    """
    # Get all files currently in the database
    db_files = db_manager.execute_with_retry(f"SELECT path FROM {TABLE_NAME}")

    if not db_files:
        return 0

    # Convert current files to set of strings for faster lookup
    current_file_paths = {str(f) for f in current_files}

    # Find orphaned files
    orphaned_files = []
    for (db_path,) in db_files:
        if db_path not in current_file_paths:
            orphaned_files.append(db_path)

    # Remove orphaned files
    if orphaned_files:
        operations = [(f"DELETE FROM {TABLE_NAME} WHERE path = ?", (path,)) for path in orphaned_files]
        db_manager.execute_transaction(operations)
        print(
            f"🗑️  Removed {len(orphaned_files)} orphaned files from index",
            file=sys.stderr,
        )

    return len(orphaned_files)


def get_last_index_time(db_manager: DatabaseManager) -> Optional[datetime.datetime]:
    """
    Get the last index time from the database.

    Args:
        db_manager: DatabaseManager instance

    Returns:
        The last index time, or None if no files are indexed
    """
    try:
        result = db_manager.execute_with_retry(f"SELECT MAX(file_mtime) FROM {TABLE_NAME}")
        if result and result[0] and result[0][0]:
            return result[0][0]
        return None
    except Exception:
        return None


def reindex_all(
    repo_path: Path,
    max_bytes: int,
    db_manager: DatabaseManager,
    embedder: EmbeddingGenerator,
) -> Tuple[int, int]:
    """
    Reindex all files in the repository.

    Args:
        repo_path: Path to the repository
        max_bytes: Maximum file size in bytes
        db_manager: DatabaseManager instance
        embedder: EmbeddingGenerator instance

    Returns:
        Tuple of (files_processed, embeddings_count)
    """
    # Scan repository for files
    files = scan_repo(repo_path, max_bytes)

    # Remove orphaned files
    remove_orphaned_files(db_manager, files)

    # Process all files
    embed_and_store(db_manager, embedder, files)

    # Build/validate index
    embedding_count = build_full_index(db_manager)

    return len(files), embedding_count
