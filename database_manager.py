import fcntl
import signal
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, TextIO

import duckdb

from config import config
from exceptions import DatabaseError, DatabaseLockError
from logging_config import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """
    Thread-safe database connection manager with file locking and proper lifecycle
    management.
    """

    def __init__(
        self,
        db_path: Path,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
    ) -> None:
        self.db_path = db_path
        self.max_retries = max_retries or config.database.MAX_RETRIES
        self.retry_delay = retry_delay or config.database.RETRY_DELAY
        self._lock = threading.RLock()
        self._connections: Dict[int, duckdb.DuckDBPyConnection] = {}
        self._file_lock: Optional[TextIO] = None
        self._lock_file_path = db_path.with_suffix(".lock")

        # Register cleanup handlers only in main thread
        try:
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
        except ValueError:
            # Signal handling only works in main thread, ignore in other threads
            pass

    def _signal_handler(self, signum, frame):
        """Handle termination signals gracefully."""
        self.cleanup()
        sys.exit(0)

    def _acquire_file_lock(self) -> None:
        """Acquire file-based lock to prevent multiple processes from accessing the
        same database."""
        try:
            self._lock_file_path.parent.mkdir(parents=True, exist_ok=True)
            self._file_lock = open(self._lock_file_path, "w")
            fcntl.flock(self._file_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (IOError, OSError) as e:
            if self._file_lock:
                self._file_lock.close()
                self._file_lock = None
            raise RuntimeError(f"Could not acquire database lock: {e}")

    def _release_file_lock(self) -> None:
        """Release file-based lock."""
        if self._file_lock:
            try:
                fcntl.flock(self._file_lock.fileno(), fcntl.LOCK_UN)
                self._file_lock.close()
                self._lock_file_path.unlink(missing_ok=True)
            except (IOError, OSError):
                pass
            finally:
                self._file_lock = None

    def _create_connection(self) -> duckdb.DuckDBPyConnection:
        """Create a new database connection with proper configuration."""
        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create connection - DuckDB uses different configuration than SQLite
        conn = duckdb.connect(str(self.db_path))

        # Configure for better performance and concurrency
        # DuckDB automatically handles WAL-like behavior and concurrency
        conn.execute(f"SET memory_limit = '{config.database.MEMORY_LIMIT}'")
        conn.execute(f"SET threads = {config.database.THREADS}")

        return conn

    def _get_thread_connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create a connection for the current thread."""
        thread_id = threading.get_ident()

        if thread_id not in self._connections:
            self._connections[thread_id] = self._create_connection()

        return self._connections[thread_id]

    @contextmanager
    def get_connection(self):
        """
        Context manager that provides a thread-safe database connection.

        Usage:
            with db_manager.get_connection() as conn:
                result = conn.execute("SELECT * FROM table").fetchall()
        """
        with self._lock:
            if not self._file_lock:
                self._acquire_file_lock()

            conn = self._get_thread_connection()

            try:
                yield conn
            except Exception as e:
                # Log error but don't re-raise to allow cleanup
                logger.error(f"Database operation failed: {e}")
                raise DatabaseError(f"Database operation failed: {e}") from e

    def execute_with_retry(self, query: str, params: Optional[tuple] = None) -> Any:
        """
        Execute a query with retry logic for handling temporary database locks.
        """
        for attempt in range(self.max_retries):
            try:
                with self.get_connection() as conn:
                    if params:
                        return conn.execute(query, params).fetchall()
                    else:
                        return conn.execute(query).fetchall()
            except duckdb.Error as e:
                if "database is locked" in str(e).lower() and attempt < self.max_retries - 1:
                    logger.warning(
                        f"Database locked on attempt {attempt + 1}, retrying in "
                        f"{self.retry_delay * (2**attempt):.1f}s"
                    )
                    time.sleep(self.retry_delay * (2**attempt))  # Exponential backoff
                    continue
                logger.error(f"Database error after {attempt + 1} attempts: {e}")
                raise DatabaseLockError(f"Database is locked after {attempt + 1} attempts") from e
            except Exception as e:
                logger.error(f"Unexpected error during database operation: {e}")
                raise DatabaseError(f"Database operation failed: {e}") from e

        raise RuntimeError(f"Failed to execute query after {self.max_retries} attempts")

    def execute_transaction(self, operations: list) -> Any:
        """
        Execute multiple operations within a single transaction.

        Args:
            operations: List of (query, params) tuples
        """
        with self.get_connection() as conn:
            conn.execute("BEGIN TRANSACTION")
            try:
                results = []
                for query, params in operations:
                    if params:
                        result = conn.execute(query, params).fetchall()
                    else:
                        result = conn.execute(query).fetchall()
                    results.append(result)

                conn.execute("COMMIT")
                return results
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def execute(self, query: str, params: Optional[tuple] = None) -> Any:
        """
        Execute a single query (for backward compatibility with tests).

        Args:
            query: SQL query string
            params: Optional parameters for the query
        """
        with self.get_connection() as conn:
            if params:
                return conn.execute(query, params)
            else:
                return conn.execute(query)

    def executemany(self, query: str, param_list: list) -> None:
        """
        Execute a query multiple times with different parameters.

        Args:
            query: SQL query string
            param_list: List of parameter tuples
        """
        with self.get_connection() as conn:
            conn.executemany(query, param_list)

    def cleanup(self) -> None:
        """Clean up all connections and locks."""
        with self._lock:
            # Close all thread connections
            for conn in self._connections.values():
                try:
                    conn.close()
                except Exception:
                    pass

            self._connections.clear()
            self._release_file_lock()

    def close(self) -> None:
        """Close all connections - compatibility method for tests."""
        self.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
