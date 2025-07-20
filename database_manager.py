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
from exceptions import (
    DatabaseConnectionTimeoutError,
    DatabaseCorruptionError,
    DatabaseDiskSpaceError,
    DatabaseError,
    DatabaseLockError,
    DatabasePermissionError,
    DatabaseTimeoutError,
)
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
        connection_timeout: Optional[float] = None,
        lock_timeout: Optional[float] = None,
    ) -> None:
        self.db_path = db_path
        self.max_retries = max_retries or config.database.MAX_RETRIES
        self.retry_delay = retry_delay or config.database.RETRY_DELAY
        self.connection_timeout = connection_timeout or config.database.CONNECTION_TIMEOUT
        self.lock_timeout = lock_timeout or config.database.LOCK_TIMEOUT
        self._lock = threading.RLock()
        self._connections: Dict[int, duckdb.DuckDBPyConnection] = {}
        self._connection_created_time: Dict[int, float] = {}
        self._connection_last_used: Dict[int, float] = {}
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

            # Try to acquire lock with timeout by using non-blocking mode and retrying
            lock_acquired = False
            start_time = time.time()

            while not lock_acquired and (time.time() - start_time) < self.lock_timeout:
                try:
                    fcntl.flock(self._file_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    lock_acquired = True
                except (IOError, OSError) as e:
                    error_str = str(e)
                    if (
                        "Resource temporarily unavailable" in error_str or
                        "temporarily unavailable" in error_str
                    ):
                        # Lock is held by another process, wait and retry
                        time.sleep(config.database.LOCK_RETRY_INTERVAL)
                        continue
                    else:
                        # Other error, re-raise
                        raise

            if not lock_acquired:
                raise DatabaseTimeoutError(
                    f"Could not acquire database lock within {self.lock_timeout} seconds"
                )

        except (IOError, OSError) as e:
            if self._file_lock:
                self._file_lock.close()
                self._file_lock = None
            if "permission denied" in str(e).lower():
                raise DatabasePermissionError(
                    f"Permission denied when acquiring database lock: {e}"
                )
            elif "no space left" in str(e).lower():
                raise DatabaseDiskSpaceError(
                    f"Insufficient disk space when acquiring database lock: {e}"
                )
            else:
                raise DatabaseLockError(f"Could not acquire database lock: {e}")

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
        try:
            # Ensure database directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Create connection - DuckDB uses different configuration than SQLite
            conn = duckdb.connect(str(self.db_path))

            # Configure for better performance and concurrency
            # DuckDB automatically handles WAL-like behavior and concurrency
            conn.execute(f"SET memory_limit = '{config.database.MEMORY_LIMIT}'")
            conn.execute(f"SET threads = {config.database.THREADS}")

            # Configure timeout settings
            if (hasattr(conn, "execute") and 
                config.database.STATEMENT_TIMEOUT > 0):
                # Note: DuckDB doesn't have a direct statement timeout, but we can use this for logging
                pass

            # Configure temporary directory if specified
            if config.database.TEMP_DIRECTORY:
                conn.execute(f"SET temp_directory = '{config.database.TEMP_DIRECTORY}'")

            # Configure auto-vacuum if enabled
            if config.database.AUTO_VACUUM:
                # DuckDB handles this automatically, but we can add related optimizations
                pass

            return conn
        except PermissionError as e:
            raise DatabasePermissionError(
                f"Permission denied when creating database connection: {e}"
            )
        except OSError as e:
            if "no space left" in str(e).lower():
                raise DatabaseDiskSpaceError(
                    f"Insufficient disk space when creating database connection: {e}"
                )
            else:
                raise DatabaseConnectionTimeoutError(f"Failed to create database connection: {e}")
        except duckdb.Error as e:
            if "corrupt" in str(e).lower():
                raise DatabaseCorruptionError(f"Database corruption detected: {e}")
            else:
                raise DatabaseError(f"Database error when creating connection: {e}")
        except Exception as e:
            raise DatabaseError(f"Unexpected error when creating database connection: {e}")

    def _get_thread_connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create a connection for the current thread with lifecycle management."""
        thread_id = threading.get_ident()
        current_time = time.time()

        # Check if connection exists and is still valid
        if thread_id in self._connections:
            conn = self._connections[thread_id]
            created_time = self._connection_created_time.get(thread_id, 0)
            last_used = self._connection_last_used.get(thread_id, 0)

            # Check connection age
            if current_time - created_time > config.database.CONNECTION_MAX_AGE:
                logger.debug(f"Recreating connection for thread {thread_id}: max age exceeded")
                self._close_thread_connection(thread_id)
            # Check idle timeout
            elif current_time - last_used > config.database.CONNECTION_IDLE_TIMEOUT:
                logger.debug(f"Recreating connection for thread {thread_id}: idle timeout exceeded")
                self._close_thread_connection(thread_id)
            # Check connection health
            elif not self._check_connection_health(conn):
                logger.debug(f"Recreating connection for thread {thread_id}: health check failed")
                self._close_thread_connection(thread_id)
            else:
                # Update last used time and return existing connection
                self._connection_last_used[thread_id] = current_time
                return conn

        # Create new connection if needed
        if thread_id not in self._connections:
            self._connections[thread_id] = self._create_connection()
            self._connection_created_time[thread_id] = current_time
            self._connection_last_used[thread_id] = current_time
            logger.debug(f"Created new connection for thread {thread_id}")

        return self._connections[thread_id]

    def _check_connection_health(self, conn: duckdb.DuckDBPyConnection) -> bool:
        """Check if a connection is healthy and usable."""
        try:
            # Simple health check - execute a basic query
            conn.execute("SELECT 1").fetchone()
            return True
        except Exception as e:
            logger.debug(f"Connection health check failed: {e}")
            return False

    def _close_thread_connection(self, thread_id: int) -> None:
        """Close and remove a specific thread connection."""
        if thread_id in self._connections:
            try:
                self._connections[thread_id].close()
            except Exception as e:
                logger.debug(f"Error closing connection for thread {thread_id}: {e}")

            # Remove from all tracking dictionaries
            self._connections.pop(thread_id, None)
            self._connection_created_time.pop(thread_id, None)
            self._connection_last_used.pop(thread_id, None)

    def cleanup_idle_connections(self) -> int:
        """Clean up idle connections and return the number of connections closed."""
        current_time = time.time()
        closed_count = 0

        with self._lock:
            threads_to_close = []

            for thread_id, last_used in self._connection_last_used.items():
                if current_time - last_used > config.database.CONNECTION_IDLE_TIMEOUT:
                    threads_to_close.append(thread_id)

            for thread_id in threads_to_close:
                self._close_thread_connection(thread_id)
                closed_count += 1

        if closed_count > 0:
            logger.debug(f"Cleaned up {closed_count} idle connections")

        return closed_count

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
            self._connection_created_time.clear()
            self._connection_last_used.clear()
            self._release_file_lock()

    def close(self) -> None:
        """Close all connections - compatibility method for tests."""
        self.cleanup()

    def __enter__(self):
        return self

    def migrate_schema(self, table_name: str) -> None:
        """
        Migrate database schema to add new metadata columns if they don't exist.

        Adds the following columns:
        - file_type VARCHAR - file extension (.py, .js, .md, etc.)
        - language VARCHAR - detected programming language
        - size_bytes INTEGER - file size in bytes
        - line_count INTEGER - number of lines in the file

        Args:
            table_name: Name of the table to migrate
        """
        logger.info(f"Starting schema migration for table {table_name}")

        # Define all columns to add (includes legacy columns for backward compatibility)
        new_columns = {
            'last_modified': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
            'file_mtime': 'TIMESTAMP',
            'file_type': 'VARCHAR',
            'language': 'VARCHAR',
            'size_bytes': 'INTEGER',
            'line_count': 'INTEGER'
        }

        try:
            # Get existing columns using DuckDB's information_schema
            with self.get_connection() as conn:
                result = conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = ? ORDER BY ordinal_position",
                    (table_name,)
                ).fetchall()

                existing_columns = [row[0].lower() for row in result]
                logger.debug(f"Existing columns: {existing_columns}")

                # Check if table exists (if no columns found, table likely doesn't exist)
                if not existing_columns:
                    raise DatabaseError(f"Table {table_name} does not exist or has no columns")

                # Add new columns that don't exist
                columns_added = []
                for column_name, column_type in new_columns.items():
                    if column_name.lower() not in existing_columns:
                        try:
                            alter_query = (
                                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
                            )
                            conn.execute(alter_query)
                            columns_added.append(column_name)
                            logger.info(f"Added column: {column_name} {column_type}")
                        except Exception as e:
                            logger.warning(
                                f"Failed to add column {column_name}: {e}"
                            )
                    else:
                        logger.debug(f"Column {column_name} already exists, skipping")

                if columns_added:
                    logger.info(
                        f"Schema migration completed successfully. Added columns: {columns_added}"
                    )
                else:
                    logger.info("Schema migration completed - no columns needed to be added")
        except Exception as e:
            logger.error(f"Schema migration failed: {e}")
            raise DatabaseError(f"Failed to migrate schema for table {table_name}: {e}") from e

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
