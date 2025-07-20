#!/usr/bin/env python3
"""
file_watching.py: File system monitoring for the Turboprop code search system.

This module contains functionality for monitoring file system changes:
- Real-time file change detection
- Debounced update handling
- Incremental index updates
- File system event processing
"""

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from config import config
from database_manager import DatabaseManager
from embedding_helper import EmbeddingGenerator
from indexing_operations import embed_and_store_single

# Setup logging
logger = logging.getLogger(__name__)


class DebouncedHandler(FileSystemEventHandler):
    """
    A file system event handler that debounces file change events.

    This handler collects file change events and processes them in batches
    after a configurable delay, preventing excessive reindexing when many
    files change rapidly.
    """

    def __init__(
        self,
        repo_path: Path,
        max_bytes: int,
        db_manager: DatabaseManager,
        embedder: EmbeddingGenerator,
        debounce_sec: Optional[float] = None,
    ):
        """
        Initialize the debounced handler.

        Args:
            repo_path: Path to the repository being watched
            max_bytes: Maximum file size to process
            db_manager: DatabaseManager instance
            embedder: EmbeddingGenerator instance
            debounce_sec: Debounce delay in seconds
        """
        super().__init__()
        self.repo_path = repo_path
        self.max_bytes = max_bytes
        self.db_manager = db_manager
        self.embedder = embedder
        self.debounce_sec = debounce_sec or config.file_processing.DEBOUNCE_SECONDS

        # Track pending changes
        self.pending_changes: Set[Path] = set()
        self.timer: Optional[threading.Timer] = None
        self.lock = threading.Lock()

        # Track processed files to avoid duplicate processing
        self.recently_processed: Set[Path] = set()
        self.last_cleanup = time.time()

        logger.info(f"Initialized file watcher for {repo_path} with {debounce_sec}s debounce")

    def _should_index_file(self, file_path: Path) -> bool:
        """
        Check if a file should be indexed based on various criteria.

        Args:
            file_path: Path to the file to check

        Returns:
            True if the file should be indexed, False otherwise
        """
        # Skip if file doesn't exist
        if not file_path.exists():
            return False

        # Skip if file is too large
        try:
            if file_path.stat().st_size > self.max_bytes:
                return False
        except OSError:
            return False

        # Skip if file is not tracked by git
        try:
            import subprocess

            result = subprocess.run(
                ["git", "ls-files", "--error-unmatch", str(file_path)],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=False,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _cleanup_recent_files(self):
        """Clean up the recently processed files set periodically."""
        current_time = time.time()
        if current_time - self.last_cleanup > config.file_processing.CLEANUP_INTERVAL:
            self.recently_processed.clear()
            self.last_cleanup = current_time

    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Skip if file shouldn't be indexed
        if not self._should_index_file(file_path):
            return

        # Skip if recently processed
        if file_path in self.recently_processed:
            return

        with self.lock:
            self.pending_changes.add(file_path)

            # Cancel existing timer and start a new one
            if self.timer:
                self.timer.cancel()

            self.timer = threading.Timer(self.debounce_sec, self._process_pending_changes)
            self.timer.start()

            logger.debug(f"Queued file for processing: {file_path}")

    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Skip if file shouldn't be indexed
        if not self._should_index_file(file_path):
            return

        with self.lock:
            self.pending_changes.add(file_path)

            # Cancel existing timer and start a new one
            if self.timer:
                self.timer.cancel()

            self.timer = threading.Timer(self.debounce_sec, self._process_pending_changes)
            self.timer.start()

            logger.debug(f"Queued new file for processing: {file_path}")

    def on_deleted(self, event):
        """Handle file deletion events."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        with self.lock:
            # Remove from pending changes if it was queued
            self.pending_changes.discard(file_path)

            # Remove from database
            try:
                self.db_manager.execute_with_retry("DELETE FROM code_files WHERE path = ?", (str(file_path),))
                logger.info(f"Removed deleted file from index: {file_path}")
            except Exception as e:
                logger.error(f"Error removing deleted file from index: {e}")

    def _process_pending_changes(self):
        """Process all pending file changes."""
        with self.lock:
            if not self.pending_changes:
                return

            changes_to_process = self.pending_changes.copy()
            self.pending_changes.clear()
            self.timer = None

        # Clean up recently processed files
        self._cleanup_recent_files()

        # Process changes
        successful_count = 0
        failed_count = 0

        for file_path in changes_to_process:
            try:
                # Double-check that file should still be indexed
                if not self._should_index_file(file_path):
                    continue

                # Process the file
                success = embed_and_store_single(self.db_manager, self.embedder, file_path)

                if success:
                    successful_count += 1
                    self.recently_processed.add(file_path)
                    logger.info(f"Updated index for: {file_path}")
                else:
                    failed_count += 1
                    logger.warning(f"Failed to update index for: {file_path}")

            except Exception as e:
                failed_count += 1
                logger.error(f"Error processing {file_path}: {e}")

        # Report results
        if successful_count > 0:
            print(f"🔄 Updated {successful_count} files in index", file=sys.stderr)

        if failed_count > 0:
            print(f"⚠️  Failed to update {failed_count} files", file=sys.stderr)

    def shutdown(self):
        """Shutdown the handler and process any remaining changes."""
        with self.lock:
            if self.timer:
                self.timer.cancel()
                self.timer = None

        # Process any remaining changes
        self._process_pending_changes()
        logger.info("File watcher handler shutdown complete")


def watch_mode(repo_path: str, max_mb: float, debounce_sec: float):
    """
    Monitor a repository for file changes and update the index incrementally.

    This function sets up a file system watcher that monitors the repository
    for changes and updates the search index when files are modified, created,
    or deleted.

    Args:
        repo_path: Path to the repository to watch
        max_mb: Maximum file size in MB to process
        debounce_sec: Debounce delay in seconds for batching changes
    """
    from code_index import init_db
    from embedding_helper import EmbeddingGenerator

    repo_path_obj = Path(repo_path).resolve()
    max_bytes = int(max_mb * 1024 * 1024)

    # Verify repository exists
    if not repo_path_obj.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    if not (repo_path_obj / ".git").exists():
        raise ValueError(f"Not a Git repository: {repo_path}")

    # Initialize database and embedder
    db_manager = init_db(repo_path_obj)
    embedder = EmbeddingGenerator()

    # Create the debounced handler
    handler = DebouncedHandler(repo_path_obj, max_bytes, db_manager, embedder, debounce_sec)

    # Set up the file system observer
    observer = Observer()
    observer.schedule(handler, str(repo_path_obj), recursive=True)

    # Start monitoring
    observer.start()

    try:
        print(
            f"👀 Watching {repo_path} for changes (debounce: {debounce_sec}s)",
            file=sys.stderr,
        )
        print("Press Ctrl+C to stop watching", file=sys.stderr)

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Stopping file watcher...", file=sys.stderr)

    finally:
        observer.stop()
        observer.join()
        handler.shutdown()
        db_manager.close()
        print("✅ File watcher stopped", file=sys.stderr)


def start_background_watcher(
    repo_path: Path,
    max_bytes: int,
    db_manager: DatabaseManager,
    embedder: EmbeddingGenerator,
    debounce_sec: Optional[float] = None,
) -> tuple[Observer, DebouncedHandler]:
    """
    Start a background file watcher that doesn't block the main thread.

    Args:
        repo_path: Path to the repository to watch
        max_bytes: Maximum file size in bytes to process
        db_manager: DatabaseManager instance
        embedder: EmbeddingGenerator instance
        debounce_sec: Debounce delay in seconds

    Returns:
        Tuple of (Observer, DebouncedHandler) for management
    """
    # Create the debounced handler
    effective_debounce = debounce_sec or config.file_processing.DEBOUNCE_SECONDS
    handler = DebouncedHandler(repo_path, max_bytes, db_manager, embedder, effective_debounce)

    # Set up the file system observer
    observer = Observer()
    observer.schedule(handler, str(repo_path), recursive=True)

    # Start monitoring in the background
    observer.start()

    logger.info(f"Started background file watcher for {repo_path}")

    return observer, handler


def stop_background_watcher(observer: Observer, handler: DebouncedHandler):
    """
    Stop a background file watcher.

    Args:
        observer: The Observer instance to stop
        handler: The DebouncedHandler instance to shutdown
    """
    observer.stop()
    observer.join()
    handler.shutdown()

    logger.info("Stopped background file watcher")


def get_watcher_status(observer: Observer) -> Dict[str, Any]:
    """
    Get the status of a file watcher.

    Args:
        observer: The Observer instance to check

    Returns:
        Dictionary with watcher status information
    """
    return {
        "is_alive": observer.is_alive(),
        "watches": len(observer.emitters),
        "daemon": observer.daemon,
    }


def monitor_repository_health(
    repo_path: Path, db_manager: DatabaseManager, check_interval: Optional[int] = None
) -> None:
    """
    Monitor repository health and detect issues.

    Args:
        repo_path: Path to the repository
        db_manager: DatabaseManager instance
        check_interval: Check interval in seconds (defaults to config value)
    """
    effective_check_interval = check_interval or config.file_processing.HEALTH_CHECK_INTERVAL

    while True:
        try:
            # Check if repository still exists
            if not repo_path.exists():
                logger.error(f"Repository path no longer exists: {repo_path}")
                break

            # Check if .git directory still exists
            if not (repo_path / ".git").exists():
                logger.error(f"Repository is no longer a Git repository: {repo_path}")
                break

            # Check database connectivity
            try:
                db_manager.execute_with_retry("SELECT 1")
            except Exception as e:
                logger.error(f"Database connectivity issue: {e}")
                break

            # Log health status
            logger.debug(f"Repository health check passed for {repo_path}")

            # Wait for next check
            time.sleep(effective_check_interval)

        except Exception as e:
            logger.error(f"Error during repository health check: {e}")
            break
