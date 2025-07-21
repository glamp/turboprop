import fcntl
import signal
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

import duckdb

from config import EmbeddingConfig, config
from exceptions import (
    DatabaseConnectionTimeoutError,
    DatabaseCorruptionError,
    DatabaseDiskSpaceError,
    DatabaseError,
    DatabaseLockError,
    DatabaseMigrationError,
    DatabasePermissionError,
    DatabaseTimeoutError,
)
from logging_config import get_logger

logger = get_logger(__name__)

# Index naming pattern constants
INDEX_PREFIX = "idx"
FTS_CONTENT_INDEX_SUFFIX = "content"
CODE_CONSTRUCTS_INDEXES = {
    "file_id": f"{INDEX_PREFIX}_code_constructs_file_id",
    "type": f"{INDEX_PREFIX}_code_constructs_type",
    "name": f"{INDEX_PREFIX}_code_constructs_name",
    "parent": f"{INDEX_PREFIX}_code_constructs_parent",
}
REPOSITORY_CONTEXT_INDEXES = {
    "path": f"{INDEX_PREFIX}_repository_context_path",
    "type": f"{INDEX_PREFIX}_repository_context_type",
    "indexed": f"{INDEX_PREFIX}_repository_context_indexed",
}


class ConnectionManager:
    """Manages database connections with lifecycle management."""

    def __init__(self, db_path: Path, connection_timeout: float):
        self.db_path = db_path
        self.connection_timeout = connection_timeout
        self._connections: Dict[int, duckdb.DuckDBPyConnection] = {}
        self._connection_created_time: Dict[int, float] = {}
        self._connection_last_used: Dict[int, float] = {}

    def get_thread_connection(self) -> duckdb.DuckDBPyConnection:
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
                logger.debug("Recreating connection for thread %s: max age exceeded", thread_id)
                self._close_thread_connection(thread_id)
            # Check idle timeout
            elif current_time - last_used > config.database.CONNECTION_IDLE_TIMEOUT:
                logger.debug("Recreating connection for thread %s: idle timeout exceeded", thread_id)
                self._close_thread_connection(thread_id)
            # Check connection health
            elif not self._check_connection_health(conn):
                logger.debug("Recreating connection for thread %s: health check failed", thread_id)
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
            logger.debug("Created new connection for thread %s", thread_id)

        return self._connections[thread_id]

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
            if hasattr(conn, "execute") and config.database.STATEMENT_TIMEOUT > 0:
                # Note: DuckDB doesn't have a direct statement timeout,
                # but we can use this for logging
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
            raise DatabasePermissionError(f"Permission denied when creating database connection: {e}") from e
        except OSError as e:
            if "no space left" in str(e).lower():
                raise DatabaseDiskSpaceError(f"Insufficient disk space when creating database connection: {e}") from e
            raise DatabaseConnectionTimeoutError(f"Failed to create database connection: {e}") from e
        except duckdb.Error as e:
            if "corrupt" in str(e).lower():
                raise DatabaseCorruptionError(f"Database corruption detected: {e}") from e
            raise DatabaseError(f"Database error when creating connection: {e}") from e
        except Exception as error:
            raise DatabaseError(f"Unexpected error when creating database connection: {error}") from error

    def _check_connection_health(self, conn: duckdb.DuckDBPyConnection) -> bool:
        """Check if a connection is healthy and usable."""
        try:
            # Simple health check - execute a basic query
            conn.execute("SELECT 1").fetchone()
            return True
        except (duckdb.Error, OSError, RuntimeError) as e:
            logger.debug("Connection health check failed: %s", e)
            return False

    def _close_thread_connection(self, thread_id: int) -> None:
        """Close and remove a specific thread connection."""
        if thread_id in self._connections:
            try:
                self._connections[thread_id].close()
            except Exception as e:
                logger.debug("Error closing connection for thread %s: %s", thread_id, e)

            # Remove from all tracking dictionaries
            self._connections.pop(thread_id, None)
            self._connection_created_time.pop(thread_id, None)
            self._connection_last_used.pop(thread_id, None)

    def cleanup_idle_connections(self) -> int:
        """Clean up idle connections and return the number of connections closed."""
        current_time = time.time()
        closed_count = 0

        threads_to_close = []
        for thread_id, last_used in self._connection_last_used.items():
            if current_time - last_used > config.database.CONNECTION_IDLE_TIMEOUT:
                threads_to_close.append(thread_id)

        for thread_id in threads_to_close:
            self._close_thread_connection(thread_id)
            closed_count += 1

        if closed_count > 0:
            logger.debug("Cleaned up %d idle connections", closed_count)

        return closed_count

    def cleanup(self) -> None:
        """Clean up all connections."""
        # Close all thread connections
        for conn in self._connections.values():
            try:
                conn.close()
            except Exception as e:
                logger.debug("Error closing connection during cleanup: %s", e)

        self._connections.clear()
        self._connection_created_time.clear()
        self._connection_last_used.clear()


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
        # Ensure db_path is a Path object
        self.db_path = Path(db_path) if not isinstance(db_path, Path) else db_path
        self.max_retries = max_retries or config.database.MAX_RETRIES
        self.retry_delay = retry_delay or config.database.RETRY_DELAY
        self.lock_timeout = lock_timeout or config.database.LOCK_TIMEOUT
        self._lock = threading.RLock()
        self._file_lock: Optional[TextIO] = None
        self._lock_file_path = self.db_path.with_suffix(".lock")

        # Use ConnectionManager for connection handling
        self.connection_timeout = connection_timeout or config.database.CONNECTION_TIMEOUT
        self._connection_manager = ConnectionManager(self.db_path, self.connection_timeout)

        # Register cleanup handlers only in main thread
        try:
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
        except ValueError as e:
            # Signal handling only works in main thread, ignore in other threads
            logger.debug("Signal handler setup failed (not in main thread): %s", e)

    @property
    def _connections(self) -> Dict[int, Any]:
        """Access to connection manager's connections for testing."""
        return self._connection_manager._connections

    @property
    def _connection_created_time(self) -> Dict[int, float]:
        """Access to connection manager's creation times for testing."""
        return self._connection_manager._connection_created_time

    @property
    def _connection_last_used(self) -> Dict[int, float]:
        """Access to connection manager's last used times for testing."""
        return self._connection_manager._connection_last_used

    def _check_connection_health(self, conn: Any) -> bool:
        """Check if a connection is healthy - delegates to connection manager."""
        return self._connection_manager._check_connection_health(conn)

    def _signal_handler(self, signum: int, frame: Any) -> None:
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
                    if "Resource temporarily unavailable" in error_str or "temporarily unavailable" in error_str:
                        # Lock is held by another process, wait and retry
                        time.sleep(config.database.LOCK_RETRY_INTERVAL)
                        continue
                    else:
                        # Other error, re-raise
                        raise

            if not lock_acquired:
                if self._file_lock:
                    self._file_lock.close()
                    self._file_lock = None
                raise DatabaseTimeoutError(f"Could not acquire database lock within {self.lock_timeout} seconds")

        except (IOError, OSError) as e:
            if self._file_lock:
                self._file_lock.close()
                self._file_lock = None
            if "permission denied" in str(e).lower():
                raise DatabasePermissionError(f"Permission denied when acquiring database lock: {e}")
            elif "no space left" in str(e).lower():
                raise DatabaseDiskSpaceError(f"Insufficient disk space when acquiring database lock: {e}")
            else:
                raise DatabaseLockError(f"Could not acquire database lock: {e}")

    def _release_file_lock(self) -> None:
        """Release file-based lock."""
        if self._file_lock:
            try:
                fcntl.flock(self._file_lock.fileno(), fcntl.LOCK_UN)
                self._file_lock.close()
                self._lock_file_path.unlink(missing_ok=True)
            except (IOError, OSError) as e:
                logger.debug("Error cleaning up lock file: %s", e)
            finally:
                self._file_lock = None

    def cleanup_idle_connections(self) -> int:
        """Clean up idle connections and return the number of connections closed."""
        with self._lock:
            return self._connection_manager.cleanup_idle_connections()

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

            conn = self._connection_manager.get_thread_connection()

            try:
                yield conn
            except (
                DatabaseError,
                DatabaseMigrationError,
                DatabaseConnectionTimeoutError,
                DatabaseCorruptionError,
                DatabaseDiskSpaceError,
                DatabaseLockError,
                DatabasePermissionError,
                DatabaseTimeoutError,
            ) as e:
                # These are already properly typed database exceptions, let them bubble up
                logger.error("Database operation failed: %s", e)
                raise
            except Exception as e:
                # Only wrap generic exceptions that aren't already database-specific
                logger.error("Database operation failed: %s", e)
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
                    return conn.execute(query).fetchall()
            except duckdb.Error as e:
                if "database is locked" in str(e).lower() and attempt < self.max_retries - 1:
                    logger.warning(
                        "Database locked on attempt %d, retrying in %.1fs",
                        attempt + 1,
                        self.retry_delay * (2**attempt),
                    )
                    time.sleep(self.retry_delay * (2**attempt))  # Exponential backoff
                    continue
                logger.error("Database error after %d attempts: %s", attempt + 1, e)
                raise DatabaseLockError(f"Database is locked after {attempt + 1} attempts") from e
            except Exception as e:
                logger.error("Unexpected error during database operation: %s", e)
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
            except Exception as e:
                logger.error("Transaction failed, rolling back: %s", e)
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
            self._connection_manager.cleanup()
            self._release_file_lock()

    def close(self) -> None:
        """Close all connections - compatibility method for tests."""
        self.cleanup()

    def __enter__(self) -> "DatabaseManager":
        return self

    def migrate_schema(self, table_name: str) -> None:
        """
        Migrate database schema to add new metadata columns if they don't exist.

        Adds the following columns:
        - file_type VARCHAR - file extension (.py, .js, .md, etc.)
        - language VARCHAR - detected programming language
        - size_bytes INTEGER - file size in bytes
        - line_count INTEGER - number of lines in the file
        - category VARCHAR - file category (source/configuration/documentation/build/etc.)

        Args:
            table_name: Name of the table to migrate
        """
        logger.info("Starting schema migration for table %s", table_name)

        # Define all columns to add (includes legacy columns for backward compatibility)
        new_columns = {
            "last_modified": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            "file_mtime": "TIMESTAMP",
            "file_type": "VARCHAR",
            "language": "VARCHAR",
            "size_bytes": "INTEGER",
            "line_count": "INTEGER",
            "category": "VARCHAR",
        }

        try:
            # Get existing columns using DuckDB's information_schema
            with self.get_connection() as conn:
                result = conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = ? ORDER BY ordinal_position",
                    (table_name,),
                ).fetchall()

                existing_columns = [row[0].lower() for row in result]
                logger.debug("Existing columns: %s", existing_columns)

                # Check if table exists (if no columns found, table likely doesn't exist)
                if not existing_columns:
                    raise DatabaseError(f"Table {table_name} does not exist or has no columns")

                # Add new columns that don't exist
                columns_added = []
                for column_name, column_type in new_columns.items():
                    if column_name.lower() not in existing_columns:
                        try:
                            alter_query = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
                            conn.execute(alter_query)
                            columns_added.append(column_name)
                            logger.info("Added column: %s %s", column_name, column_type)
                        except (duckdb.Error, OSError) as e:
                            logger.warning("Failed to add column %s: %s", column_name, e)
                    else:
                        logger.debug("Column %s already exists, skipping", column_name)

                if columns_added:
                    logger.info("Schema migration completed successfully. Added columns: %s", columns_added)
                else:
                    logger.info("Schema migration completed - no columns needed to be added")
        except Exception as e:
            logger.error("Schema migration failed: %s", e)
            raise DatabaseError(f"Failed to migrate schema for table {table_name}: {e}") from e

    def create_constructs_table(self) -> None:
        """
        Create the code_constructs table for storing extracted programming constructs.

        This table stores individual programming constructs (functions, classes, variables, etc.)
        with their own embeddings for more granular search capabilities.
        """
        logger.info("Creating code_constructs table")

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS code_constructs (
            id VARCHAR PRIMARY KEY,
            file_id VARCHAR NOT NULL,
            construct_type VARCHAR NOT NULL,
            name VARCHAR NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            signature TEXT,
            docstring TEXT,
            parent_construct_id VARCHAR,
            embedding DOUBLE[{EmbeddingConfig.DIMENSIONS}],
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """

        create_indexes_sql = [
            f"CREATE INDEX IF NOT EXISTS {CODE_CONSTRUCTS_INDEXES['file_id']} ON code_constructs(file_id)",
            f"CREATE INDEX IF NOT EXISTS {CODE_CONSTRUCTS_INDEXES['type']} ON code_constructs(construct_type)",
            f"CREATE INDEX IF NOT EXISTS {CODE_CONSTRUCTS_INDEXES['name']} ON code_constructs(name)",
            f"CREATE INDEX IF NOT EXISTS {CODE_CONSTRUCTS_INDEXES['parent']} ON code_constructs(parent_construct_id)",
        ]

        try:
            with self.get_connection() as conn:
                # Create the table
                conn.execute(create_table_sql)

                # Create indexes
                for index_sql in create_indexes_sql:
                    conn.execute(index_sql)

                logger.info("Successfully created code_constructs table with indexes")

        except Exception as e:
            logger.error("Failed to create code_constructs table: %s", e)
            raise DatabaseError(f"Failed to create code_constructs table: {e}") from e

    def _ensure_fts_extension_loaded(self, conn: duckdb.DuckDBPyConnection) -> bool:
        """
        Ensure DuckDB FTS extension is installed and loaded.

        Returns:
            bool: True if FTS extension is available, False otherwise
        """
        try:
            # Install and load FTS extension
            conn.execute("INSTALL fts")
            conn.execute("LOAD fts")
            logger.info("DuckDB FTS extension loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to load DuckDB FTS extension: {e}")
            return False

    def _load_fts_extension(self, conn: duckdb.DuckDBPyConnection) -> bool:
        """
        Alias for _ensure_fts_extension_loaded for backward compatibility.
        Returns:
            bool: True if FTS extension is available, False otherwise
        """
        return self._ensure_fts_extension_loaded(conn)

    def _drop_existing_fts_table(self, conn: duckdb.DuckDBPyConnection, fts_table_name: str) -> None:
        """Drop existing FTS table if it exists."""
        try:
            conn.execute(f"DROP TABLE IF EXISTS {fts_table_name}")
        except duckdb.Error:
            pass

    def _create_fts_index_pragma(self, conn: duckdb.DuckDBPyConnection, fts_table_name: str, table_name: str) -> bool:
        """
        Try to create FTS index using PRAGMA syntax.

        Returns:
            True if successful, False if failed
        """
        fts_create_sql = f"PRAGMA create_fts_index('{fts_table_name}', '{table_name}', 'content')"

        try:
            conn.execute(fts_create_sql)
            logger.info("Successfully created FTS index %s", fts_table_name)
            return True
        except duckdb.Error as e:
            logger.warning("PRAGMA FTS creation failed: %s. Trying alternative approach.", e)
            return False

    def _create_alternative_fts_table(
        self, conn: duckdb.DuckDBPyConnection, fts_table_name: str, table_name: str
    ) -> None:
        """Create alternative FTS table and indexes when PRAGMA approach fails.

        Args:
            conn: DuckDB connection object to execute SQL commands
            fts_table_name: Name for the new FTS table to be created
            table_name: Source table name to copy data from

        Raises:
            DatabasePermissionError: When permission is denied for table/index creation
            DatabaseDiskSpaceError: When insufficient disk space for operations
            DatabaseCorruptionError: When database corruption is detected
            DatabaseLockError: When database is locked during operations
            DatabaseError: For other database-related errors
        """
        try:
            # Create the FTS table
            self._create_fts_table_structure(conn, fts_table_name, table_name)
            logger.info("Created alternative FTS table %s", fts_table_name)

            # Try to create an index (non-critical if it fails)
            self._create_fts_content_index(conn, fts_table_name)

        except duckdb.Error as e:
            self._handle_fts_table_creation_error(e, "alternative FTS table")

    def _create_fts_table_structure(
        self, conn: duckdb.DuckDBPyConnection, fts_table_name: str, table_name: str
    ) -> None:
        """Create the basic FTS table structure."""
        fts_create_alt = f"""
        CREATE TABLE IF NOT EXISTS {fts_table_name} AS
        SELECT id, path, content
        FROM {table_name}
        WHERE content IS NOT NULL
        """
        conn.execute(fts_create_alt)

    def _create_fts_content_index(self, conn: duckdb.DuckDBPyConnection, fts_table_name: str) -> None:
        """Create content index on FTS table, with graceful fallback if it fails."""
        try:
            index_name = f"{INDEX_PREFIX}_{fts_table_name}_{FTS_CONTENT_INDEX_SUFFIX}"
            index_sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {fts_table_name} (content)"
            conn.execute(index_sql)
            logger.info("Created basic content index on alternative FTS table %s", fts_table_name)
        except duckdb.Error as e:
            self._handle_index_creation_error(e, fts_table_name)

    def _handle_index_creation_error(self, error: duckdb.Error, fts_table_name: str) -> None:
        """Handle errors during index creation with appropriate logging."""
        error_msg = str(error).lower()
        if "permission" in error_msg or "denied" in error_msg:
            logger.warning("Permission denied when creating content index on fallback table: %s", error)
        elif "space" in error_msg or "disk" in error_msg:
            logger.warning("Insufficient disk space when creating content index on fallback table: %s", error)
        elif "lock" in error_msg:
            logger.warning("Database locked when creating content index on fallback table: %s", error)
        else:
            logger.warning("Failed to create content index on fallback table: %s", error)
        logger.info("Alternative FTS table %s created without index", fts_table_name)

    def _handle_fts_table_creation_error(self, error: duckdb.Error, operation: str) -> None:
        """Handle database errors during FTS table creation and raise appropriate exceptions."""
        error_msg = str(error).lower()
        if "permission" in error_msg or "denied" in error_msg:
            logger.error("Permission denied when creating %s: %s", operation, error)
            raise DatabasePermissionError(f"Permission denied when creating {operation}: {error}") from error
        elif "space" in error_msg or "disk" in error_msg:
            logger.error("Insufficient disk space when creating %s: %s", operation, error)
            raise DatabaseDiskSpaceError(f"Insufficient disk space when creating {operation}: {error}") from error
        elif "corrupt" in error_msg:
            logger.error("Database corruption detected when creating %s: %s", operation, error)
            raise DatabaseCorruptionError(f"Database corruption detected when creating {operation}: {error}") from error
        elif "lock" in error_msg:
            logger.error("Database locked when creating %s: %s", operation, error)
            raise DatabaseLockError(f"Database locked when creating {operation}: {error}") from error
        else:
            logger.error("Failed to create %s: %s", operation, error)
            raise DatabaseError(f"Failed to create {operation}: {error}") from error

    def create_fts_index(self, table_name: str = "code_files") -> bool:
        """
        Create FTS index using DuckDB FTS extension.

        Args:
            table_name: Base table name for FTS

        Returns:
            bool: True if FTS index was created successfully
        """
        logger.info("Creating DuckDB FTS index for table %s", table_name)
        fts_table_name = f"{table_name}_fts"

        try:
            with self.get_connection() as conn:
                if not self._ensure_fts_extension_loaded(conn):
                    return False

                # Drop existing FTS table if it exists
                self._drop_existing_fts_table(conn, fts_table_name)

                # Try PRAGMA approach first using separate method
                if self._create_fts_index_pragma(conn, fts_table_name, table_name):
                    logger.info(f"Created DuckDB FTS index using PRAGMA for table: {table_name}")
                else:
                    # Use alternative approach if PRAGMA fails
                    self._create_alternative_fts_table(conn, fts_table_name, table_name)
                    logger.info(f"Created alternative FTS table: {fts_table_name}")

                logger.info(f"DuckDB FTS setup completed for table {table_name}")
                return True

        except Exception as e:
            logger.error(f"Failed to create DuckDB FTS index: {e}")
            # Fallback to alternative FTS table for compatibility
            return self._create_alternative_fts_fallback(table_name)

    def _create_alternative_fts_fallback(self, table_name: str) -> bool:
        """Create fallback FTS table when DuckDB FTS5 is not available."""
        logger.info("Creating fallback FTS table for %s", table_name)
        fts_table_name = f"{table_name}_fts"

        try:
            with self.get_connection() as conn:
                self._create_alternative_fts_table(conn, fts_table_name, table_name)
                return True
        except Exception as e:
            logger.error("Failed to create fallback FTS table: %s", e)
            return False

    def rebuild_fts_index(self, table_name: str = "code_files") -> None:
        """
        Rebuild the full-text search index to incorporate new content.

        Args:
            table_name: Name of the table to rebuild FTS index for
        """
        logger.info("Rebuilding FTS index for table %s", table_name)

        try:
            with self.get_connection() as conn:
                # For DuckDB FTS, we need to drop and recreate the index
                # First, try to drop the existing FTS index
                try:
                    conn.execute(f"PRAGMA drop_fts_index('{table_name}')")
                    logger.debug(f"Dropped existing FTS index for {table_name}")
                except Exception as drop_e:
                    logger.debug(f"Could not drop FTS index (may not exist): {drop_e}")

                # Recreate the FTS index (this will include all current data)
                if not self._ensure_fts_extension_loaded(conn):
                    logger.warning("FTS extension not available for rebuild")
                    return

                # Recreate with current data
                pragma_sql = f"PRAGMA create_fts_index('{table_name}', 'id', 'content', 'path', overwrite=1)"
                try:
                    conn.execute(pragma_sql)
                    logger.info(f"Successfully rebuilt DuckDB FTS index for {table_name}")
                except Exception as create_e:
                    logger.warning(f"FTS index recreation failed: {create_e}, trying alternative approach")

                    # Fallback to alternative table approach
                    fts_table_name = f"{table_name}_fts"
                    try:
                        conn.execute(f"DROP TABLE IF EXISTS {fts_table_name}")
                        # Recreate alternative FTS table
                        self._create_alternative_fts_table(conn, fts_table_name, table_name)
                        logger.info(f"Successfully rebuilt alternative FTS table {fts_table_name}")
                    except Exception as fallback_e:
                        logger.error(f"Failed to rebuild alternative FTS table: {fallback_e}")

        except Exception as e:
            logger.error("Failed to rebuild FTS index: %s", e)

    def is_fts_available(self, table_name: str = "code_files") -> bool:
        """
        Check if FTS is available and working.

        Args:
            table_name: Base table name to check FTS for

        Returns:
            bool: True if FTS is available and functional
        """
        try:
            with self.get_connection() as conn:
                # First, check if alternative FTS table exists
                fts_table_name = f"{table_name}_fts"
                try:
                    conn.execute(f"SELECT COUNT(*) FROM {fts_table_name} LIMIT 1")
                    logger.debug(f"Alternative FTS table available for {table_name}")
                    return True
                except Exception:
                    # Alternative FTS table doesn't exist, try DuckDB FTS schema
                    pass

                # Test if DuckDB FTS schema exists and is functional
                fts_schema = f"fts_main_{table_name}"
                try:
                    # Try to use match_bm25 function
                    conn.execute(f"SELECT {fts_schema}.match_bm25('{table_name}', 'test') LIMIT 1")
                    logger.debug(f"DuckDB FTS is available for table {table_name}")
                    return True
                except Exception:
                    # FTS schema/function not available
                    pass

        except Exception:
            pass

        logger.debug(f"FTS not available for table {table_name}")
        return False

    def _execute_fts_search_duckdb(
        self, conn: duckdb.DuckDBPyConnection, query: str, limit: int, table_name: str = "code_files"
    ) -> List[tuple]:
        """
        Execute full-text search using DuckDB FTS extension.

        Args:
            conn: DuckDB connection
            query: Search query string
            limit: Maximum number of results
            table_name: Base table name

        Returns:
            List of search results with FTS scores
        """
        try:
            # Try DuckDB native FTS first
            results = self._try_native_fts_search(conn, query, limit, table_name)
            if results:
                return results

            # Fall back to alternative FTS table search
            return self._try_alternative_fts_search(conn, query, limit, table_name)

        except Exception as e:
            logger.warning(f"DuckDB FTS search failed: {e}")
            return []

    def _try_native_fts_search(
        self, conn: duckdb.DuckDBPyConnection, query: str, limit: int, table_name: str
    ) -> List[tuple]:
        """Try native DuckDB FTS search using match_bm25 function."""
        try:
            fts_schema = f"fts_main_{table_name}"

            # Use DuckDB's match_bm25 function with correct syntax
            fts_query = f"""
            SELECT cf.id, cf.path, cf.content,
                   {fts_schema}.match_bm25(cf.id, ?) as fts_score
            FROM {table_name} cf
            WHERE {fts_schema}.match_bm25(cf.id, ?) IS NOT NULL
               AND {fts_schema}.match_bm25(cf.id, ?) > 0
            ORDER BY fts_score DESC
            LIMIT ?
            """

            results = conn.execute(fts_query, [query, query, query, limit]).fetchall()
            logger.debug(f"DuckDB FTS search for '{query}' returned {len(results)} results")
            return results

        except Exception as fts_error:
            logger.debug(f"DuckDB FTS search failed: {fts_error}. Trying alternative FTS table.")
            return []

    def _try_alternative_fts_search(
        self, conn: duckdb.DuckDBPyConnection, query: str, limit: int, table_name: str
    ) -> List[tuple]:
        """Try alternative FTS search using custom FTS table with LIKE queries."""
        fts_table_name = f"{table_name}_fts"

        alt_fts_query = f"""
        SELECT id, path, content,
               CASE
                   WHEN LOWER(content) LIKE ? THEN 1.0
                   WHEN LOWER(path) LIKE ? THEN 0.8
                   WHEN LOWER(content) LIKE ? THEN 0.6
                   ELSE 0.4
               END as fts_score
        FROM {fts_table_name}
        WHERE LOWER(content) LIKE ? OR LOWER(path) LIKE ? OR LOWER(content) LIKE ?
        ORDER BY fts_score DESC
        LIMIT ?
        """

        # Create search patterns
        exact_pattern = f"%{query.lower()}%"
        fuzzy_pattern = f"%{query.lower().replace(' ', '%')}%"

        results = conn.execute(
            alt_fts_query,
            [exact_pattern, exact_pattern, fuzzy_pattern, exact_pattern, exact_pattern, fuzzy_pattern, limit],
        ).fetchall()

        logger.debug(f"Alternative FTS search for '{query}' returned {len(results)} results")
        return results

    def _execute_like_search(
        self, conn: duckdb.DuckDBPyConnection, query: str, limit: int, table_name: str = "code_files"
    ) -> List[tuple]:
        """
        Execute fallback LIKE-based text search.

        Args:
            conn: DuckDB connection
            query: Search query string
            limit: Maximum number of results
            table_name: Base table name

        Returns:
            List of search results with relevance scores
        """
        try:
            search_sql = f"""
            SELECT
                id,
                path,
                content,
                CASE
                    WHEN content ILIKE ? THEN 1.0
                    WHEN path ILIKE ? THEN 0.8
                    ELSE 0.5
                END as relevance_score
            FROM {table_name}
            WHERE content ILIKE ? OR path ILIKE ?
            ORDER BY relevance_score DESC, path
            LIMIT ?
            """

            # Create LIKE patterns
            exact_pattern = f"%{query}%"

            results = conn.execute(
                search_sql, [exact_pattern, exact_pattern, exact_pattern, exact_pattern, limit]
            ).fetchall()

            logger.debug(f"LIKE search for '{query}' returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"LIKE search failed: {e}")
            return []

    def search_full_text(
        self,
        query: str,
        table_name: str = "code_files",
        limit: int = 10,
        enable_stemming: bool = True,
        enable_fuzzy: bool = False,
        fuzzy_distance: int = 2,
    ) -> list:
        """
        Perform full-text search on file content.

        Supports:
        - Exact phrase matching with quotes: "exact phrase"
        - Boolean operators: AND, OR, NOT
        - Wildcard matching: term*
        - Fuzzy matching: term~ (if enabled)

        Args:
            query: Search query with optional Boolean operators and phrases
            table_name: Table to search in
            limit: Maximum number of results to return
            enable_stemming: Whether to use word stemming for better matching
            enable_fuzzy: Whether to enable fuzzy matching for typos
            fuzzy_distance: Maximum edit distance for fuzzy matching

        Returns:
            List of tuples (id, path, content, relevance_score)
        """
        try:
            with self.get_connection() as conn:
                # Try DuckDB FTS first if available
                if self.is_fts_available(table_name):
                    logger.debug("Using DuckDB FTS for search")
                    fts_results = self._execute_fts_search_duckdb(conn, query, limit, table_name)

                    if fts_results:
                        logger.debug(f"Found {len(fts_results)} results using DuckDB FTS")
                        return fts_results

                # Fallback to LIKE-based text search
                logger.debug("Falling back to LIKE-based text search")
                return self._execute_like_search(conn, query, limit, table_name)

        except Exception as e:
            logger.error("Full-text search failed for query '%s': %s", query, e)
            return []

    def _process_fts_query(self, query: str, enable_fuzzy: bool, fuzzy_distance: int) -> str:
        """
        Process a search query to handle Boolean operators and special syntax.

        Args:
            query: Raw search query
            enable_fuzzy: Whether fuzzy matching is enabled
            fuzzy_distance: Maximum edit distance for fuzzy matching

        Returns:
            Processed query string suitable for FTS
        """
        # Handle quoted phrases - preserve exact matching
        import re

        processed = query.strip()

        # Extract quoted phrases
        quoted_phrases = re.findall(r'"([^"]*)"', processed)
        for phrase in quoted_phrases:
            # Replace quoted phrases with placeholder to preserve them
            placeholder = f"__PHRASE_{len(quoted_phrases)}__"
            processed = processed.replace(f'"{phrase}"', placeholder)

        # Handle Boolean operators (convert to appropriate syntax)
        processed = processed.replace(" AND ", " & ")
        processed = processed.replace(" OR ", " | ")
        processed = processed.replace(" NOT ", " -")

        # Handle wildcards (already supported with *)

        # Handle fuzzy matching if enabled
        if enable_fuzzy and "~" not in processed:
            # Add fuzzy matching to individual terms
            words = processed.split()
            fuzzy_words = []
            for word in words:
                if word not in ["&", "|", "-"] and not word.startswith("__PHRASE_"):
                    fuzzy_words.append(f"{word}~{fuzzy_distance}")
                else:
                    fuzzy_words.append(word)
            processed = " ".join(fuzzy_words)

        # Restore quoted phrases
        for i, phrase in enumerate(quoted_phrases):
            placeholder = f"__PHRASE_{i+1}__"
            processed = processed.replace(placeholder, f'"{phrase}"')

        return processed

    def migrate_fts_to_duckdb(self, table_name: str = "code_files") -> bool:
        """
        Migrate existing FTS setup to DuckDB-compatible version.

        Args:
            table_name: Base table name to migrate FTS for

        Returns:
            bool: True if migration was successful
        """
        logger.info("Migrating FTS to DuckDB-compatible version for table %s", table_name)

        try:
            with self.get_connection() as conn:
                fts_table_name = f"{table_name}_fts"

                # Drop old PostgreSQL-style FTS artifacts if they exist
                tables_to_drop = [fts_table_name]  # Old FTS table names

                for table in tables_to_drop:
                    try:
                        conn.execute(f"DROP TABLE IF EXISTS {table}")
                        logger.debug(f"Dropped old FTS table: {table}")
                    except Exception as e:
                        logger.debug(f"Could not drop table {table}: {e}")
                        pass  # Table might not exist

                # Create new DuckDB FTS index
                return self.create_fts_index(table_name)

        except Exception as e:
            logger.error(f"FTS migration failed: {e}")
            return False

    def hybrid_search_with_fallback(self, query: str, table_name: str = "code_files", limit: int = 10) -> list:
        """
        Hybrid search with automatic fallback when FTS is unavailable.

        Args:
            query: Search query string
            table_name: Base table name
            limit: Maximum number of results

        Returns:
            List of search results
        """
        try:
            with self.get_connection() as conn:
                if self.is_fts_available(table_name):
                    logger.debug("Using FTS-enabled hybrid search")
                    return self._execute_fts_search_duckdb(conn, query, limit, table_name)
                else:
                    logger.info("FTS not available, using LIKE search")
                    return self._execute_like_search(conn, query, limit, table_name)

        except Exception as e:
            logger.error(f"Hybrid search with fallback failed: {e}")
            return []

    def search_by_file_type_fts(self, query: str, file_extensions: list, limit: int = 10) -> list:
        """
        Perform full-text search filtered by file types.

        Args:
            query: Search query
            file_extensions: List of file extensions to filter by (e.g., ['.py', '.js'])
            limit: Maximum number of results

        Returns:
            List of search results filtered by file type
        """
        try:
            with self.get_connection() as conn:
                # Create extension filter
                extension_conditions = []
                params = []

                for ext in file_extensions:
                    extension_conditions.append("f.path ILIKE ?")
                    params.append(f"%{ext}")

                extension_filter = " OR ".join(extension_conditions)

                # Add query parameters
                query_pattern = f"%{query}%"
                params.extend([query_pattern, query_pattern, limit])

                search_sql = f"""
                SELECT
                    f.id,
                    f.path,
                    f.content,
                    1.0 as relevance_score
                FROM code_files_fts f
                WHERE ({extension_filter})
                  AND (f.content ILIKE ? OR f.path ILIKE ?)
                ORDER BY f.path
                LIMIT ?
                """

                results = conn.execute(search_sql, params).fetchall()
                logger.debug("File type FTS search returned %d results", len(results))
                return results

        except Exception as e:
            logger.error("File type FTS search failed: %s", e)
            return []

    def search_regex(self, pattern: str, limit: int = 10) -> list:
        """
        Perform regex search on file content.

        Args:
            pattern: Regular expression pattern
            limit: Maximum number of results

        Returns:
            List of search results matching the regex pattern
        """
        try:
            with self.get_connection() as conn:
                # DuckDB supports regex with regexp_matches function
                search_sql = """
                SELECT
                    id,
                    path,
                    content,
                    1.0 as relevance_score
                FROM code_files
                WHERE regexp_matches(content, ?, 'gm')
                ORDER BY path
                LIMIT ?
                """

                results = conn.execute(search_sql, (pattern, limit)).fetchall()
                logger.debug("Regex search for pattern '%s' returned %d results", pattern, len(results))
                return results

        except Exception as e:
            logger.error("Regex search failed for pattern '%s': %s", pattern, e)
            return []

    def store_construct(self, construct, file_id: str, construct_id: str, embedding: list) -> None:
        """
        Store a code construct in the database.

        Args:
            construct: CodeConstruct instance to store
            file_id: ID of the file containing this construct
            construct_id: Unique ID for this construct
            embedding: Embedding vector for the construct
        """
        try:
            with self.get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO code_constructs
                    (id, file_id, construct_type, name, start_line, end_line,
                     signature, docstring, parent_construct_id, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        construct_id,
                        file_id,
                        construct.construct_type,
                        construct.name,
                        construct.start_line,
                        construct.end_line,
                        construct.signature,
                        construct.docstring,
                        construct.parent_construct_id,
                        embedding,
                    ),
                )
                logger.debug("Stored construct %s for file %s", construct.name, file_id)

        except Exception as e:
            logger.error("Failed to store construct %s: %s", construct.name, e)
            raise DatabaseError(f"Failed to store construct: {e}") from e

    def get_constructs_by_file(self, file_id: str) -> list:
        """
        Get all constructs for a specific file.

        Args:
            file_id: ID of the file to get constructs for

        Returns:
            List of construct records
        """
        try:
            with self.get_connection() as conn:
                result = conn.execute(
                    "SELECT * FROM code_constructs WHERE file_id = ? ORDER BY start_line", (file_id,)
                ).fetchall()
                return result

        except (duckdb.Error, DatabaseError) as e:
            logger.error("Failed to get constructs for file %s: %s", file_id, e)
            return []

    def remove_constructs_for_file(self, file_id: str) -> int:
        """
        Remove all constructs associated with a file.

        Args:
            file_id: ID of the file whose constructs should be removed

        Returns:
            Number of constructs removed
        """
        try:
            with self.get_connection() as conn:
                # First, count the constructs that will be removed
                count_result = conn.execute(
                    "SELECT COUNT(*) FROM code_constructs WHERE file_id = ?", (file_id,)
                ).fetchone()
                construct_count = count_result[0] if count_result else 0

                # Then delete them
                conn.execute("DELETE FROM code_constructs WHERE file_id = ?", (file_id,))
                return construct_count

        except (duckdb.Error, DatabaseError) as e:
            logger.error("Failed to remove constructs for file %s: %s", file_id, e)
            return 0

    def get_construct_count(self) -> int:
        """
        Get the total number of constructs in the database.

        Returns:
            Number of constructs stored
        """
        try:
            result = self.execute_with_retry("SELECT COUNT(*) FROM code_constructs")
            return result[0][0] if result and result[0] else 0
        except (duckdb.Error, DatabaseError) as e:
            logger.error("Failed to get construct count: %s", e)
            return 0

    def create_repository_context_table(self) -> None:
        """
        Create the repository_context table for storing repository-level metadata.

        This table stores git information, project type, dependencies, and other
        repository-level context that helps with code understanding.
        """
        logger.info("Creating repository_context table")

        create_table_sql = """
        CREATE TABLE IF NOT EXISTS repository_context (
            repository_id VARCHAR PRIMARY KEY,
            repository_path VARCHAR NOT NULL,
            git_branch VARCHAR,
            git_commit VARCHAR,
            git_remote_url VARCHAR,
            project_type VARCHAR,
            dependencies JSON,
            package_managers JSON,
            indexed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """

        create_indexes_sql = [
            f"CREATE INDEX IF NOT EXISTS {REPOSITORY_CONTEXT_INDEXES['path']} ON repository_context(repository_path)",
            f"CREATE INDEX IF NOT EXISTS {REPOSITORY_CONTEXT_INDEXES['type']} ON repository_context(project_type)",
            f"CREATE INDEX IF NOT EXISTS {REPOSITORY_CONTEXT_INDEXES['indexed']} ON repository_context(indexed_at)",
        ]

        try:
            with self.get_connection() as conn:
                # Create the table
                conn.execute(create_table_sql)

                # Create indexes
                for index_sql in create_indexes_sql:
                    conn.execute(index_sql)

                logger.info("Successfully created repository_context table with indexes")

        except Exception as e:
            logger.error("Failed to create repository_context table: %s", e)
            raise DatabaseError(f"Failed to create repository_context table: {e}") from e

    def store_repository_context(self, context) -> None:
        """
        Store repository context information in the database.

        Args:
            context: RepositoryContext instance to store
        """
        try:
            context_dict = context.to_dict()
            with self.get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO repository_context
                    (repository_id, repository_path, git_branch, git_commit, git_remote_url,
                     project_type, dependencies, package_managers, indexed_at, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        context_dict["repository_id"],
                        context_dict["repository_path"],
                        context_dict["git_branch"],
                        context_dict["git_commit"],
                        context_dict["git_remote_url"],
                        context_dict["project_type"],
                        context_dict["dependencies"],
                        context_dict["package_managers"],
                        context_dict["indexed_at"],
                        context_dict["created_at"],
                    ),
                )
                logger.debug("Stored repository context for %s", context.repository_path)

        except Exception as e:
            logger.error("Failed to store repository context: %s", e)
            raise DatabaseError(f"Failed to store repository context: {e}") from e

    def get_repository_context(self, repository_id: str) -> Optional[dict]:
        """
        Get repository context by repository ID.

        Args:
            repository_id: ID of the repository

        Returns:
            Dictionary with repository context, or None if not found
        """
        try:
            with self.get_connection() as conn:
                result = conn.execute(
                    "SELECT * FROM repository_context WHERE repository_id = ?", (repository_id,)
                ).fetchone()

                if result:
                    # Convert result to dictionary
                    columns = [desc[0] for desc in conn.description]
                    return dict(zip(columns, result))
                return None

        except (duckdb.Error, DatabaseError) as e:
            logger.error("Failed to get repository context for %s: %s", repository_id, e)
            return None

    def get_repository_context_by_path(self, repository_path: str) -> Optional[dict]:
        """
        Get repository context by repository path.

        Args:
            repository_path: Path of the repository

        Returns:
            Dictionary with repository context, or None if not found
        """
        try:
            with self.get_connection() as conn:
                result = conn.execute(
                    "SELECT * FROM repository_context WHERE repository_path = ?", (repository_path,)
                ).fetchone()

                if result:
                    # Convert result to dictionary
                    columns = [desc[0] for desc in conn.description]
                    return dict(zip(columns, result))
                return None

        except (duckdb.Error, DatabaseError) as e:
            logger.error("Failed to get repository context for path %s: %s", repository_path, e)
            return None

    def update_repository_context_indexed_time(self, repository_id: str) -> None:
        """
        Update the indexed_at timestamp for a repository.

        Args:
            repository_id: ID of the repository to update
        """
        try:
            with self.get_connection() as conn:
                conn.execute(
                    "UPDATE repository_context SET indexed_at = CURRENT_TIMESTAMP WHERE repository_id = ?",
                    (repository_id,),
                )
                logger.debug("Updated indexed time for repository %s", repository_id)

        except (duckdb.Error, DatabaseError) as e:
            logger.error("Failed to update repository context indexed time: %s", e)

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[Any]) -> None:
        self.cleanup()

    # MCP Tool-specific operations

    def create_mcp_tool_tables(self) -> None:
        """
        Create all MCP tool-related tables if they don't exist.

        This method creates the core MCP tool schema including:
        - mcp_tools: Core tool metadata
        - tool_parameters: Tool parameter definitions
        - tool_examples: Usage examples and patterns
        - tool_relationships: Tool interconnections
        - All necessary indexes for efficient querying
        """
        logger.info("Creating MCP tool tables")

        from mcp_tool_schema import MCPToolSchema

        try:
            with self.get_connection() as conn:
                # Create all tables
                table_creation_sql = [
                    MCPToolSchema.get_schema_version_table_sql(),
                    MCPToolSchema.get_mcp_tools_table_sql(),
                    MCPToolSchema.get_tool_parameters_table_sql(),
                    MCPToolSchema.get_tool_examples_table_sql(),
                    MCPToolSchema.get_tool_relationships_table_sql(),
                ]

                for sql in table_creation_sql:
                    conn.execute(sql)

                # Create all indexes
                for index_sql in MCPToolSchema.get_index_definitions():
                    conn.execute(index_sql)

                logger.info("Successfully created all MCP tool tables and indexes")

        except Exception as e:
            logger.error("Failed to create MCP tool tables: %s", e)
            raise DatabaseError(f"Failed to create MCP tool tables: {e}") from e

    def store_mcp_tool(
        self,
        tool_id: str,
        name: str,
        description: str = None,
        tool_type: str = None,
        provider: str = None,
        version: str = None,
        category: str = None,
        embedding: list = None,
        metadata_json: str = None,
        is_active: bool = True,
    ) -> None:
        """
        Store an MCP tool in the database.

        Args:
            tool_id: Unique tool identifier
            name: Tool display name
            description: Tool description
            tool_type: Type of tool ('system', 'custom', 'third_party')
            provider: Tool provider/source
            version: Tool version
            category: Tool category
            embedding: 384-dimension embedding vector
            metadata_json: Additional metadata as JSON string
            is_active: Whether the tool is active
        """
        try:
            with self.get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO mcp_tools
                    (id, name, description, tool_type, provider, version, category,
                     embedding, metadata_json, is_active, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    tool_type = EXCLUDED.tool_type,
                    provider = EXCLUDED.provider,
                    version = EXCLUDED.version,
                    category = EXCLUDED.category,
                    embedding = EXCLUDED.embedding,
                    metadata_json = EXCLUDED.metadata_json,
                    is_active = EXCLUDED.is_active,
                    last_updated = ?
                    """,
                    (
                        tool_id,
                        name,
                        description,
                        tool_type,
                        provider,
                        version,
                        category,
                        embedding,
                        metadata_json,
                        is_active,
                        datetime.now(),
                        datetime.now(),
                    ),
                )
                logger.debug("Stored MCP tool: %s", tool_id)

        except Exception as e:
            logger.error("Failed to store MCP tool %s: %s", tool_id, e)
            raise DatabaseError(f"Failed to store MCP tool: {e}") from e

    def store_tool_parameter(
        self,
        parameter_id: str,
        tool_id: str,
        parameter_name: str,
        parameter_type: str = None,
        is_required: bool = False,
        description: str = None,
        default_value: str = None,
        schema_json: str = None,
        embedding: list = None,
    ) -> None:
        """
        Store a tool parameter in the database.

        Args:
            parameter_id: Unique parameter identifier
            tool_id: ID of the tool this parameter belongs to
            parameter_name: Name of the parameter
            parameter_type: Type of parameter ('string', 'number', 'boolean', etc.)
            is_required: Whether the parameter is required
            description: Parameter description
            default_value: Default value for the parameter
            schema_json: Full JSON schema for the parameter
            embedding: 384-dimension embedding vector
        """
        try:
            with self.get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO tool_parameters
                    (id, tool_id, parameter_name, parameter_type, is_required,
                     description, default_value, schema_json, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (id) DO UPDATE SET
                    parameter_name = EXCLUDED.parameter_name,
                    parameter_type = EXCLUDED.parameter_type,
                    is_required = EXCLUDED.is_required,
                    description = EXCLUDED.description,
                    default_value = EXCLUDED.default_value,
                    schema_json = EXCLUDED.schema_json,
                    embedding = EXCLUDED.embedding
                    """,
                    (
                        parameter_id,
                        tool_id,
                        parameter_name,
                        parameter_type,
                        is_required,
                        description,
                        default_value,
                        schema_json,
                        embedding,
                    ),
                )
                logger.debug("Stored tool parameter: %s for tool %s", parameter_name, tool_id)

        except Exception as e:
            logger.error("Failed to store tool parameter %s: %s", parameter_name, e)
            raise DatabaseError(f"Failed to store tool parameter: {e}") from e

    def store_tool_example(
        self,
        example_id: str,
        tool_id: str,
        use_case: str = None,
        example_call: str = None,
        expected_output: str = None,
        context: str = None,
        embedding: list = None,
        effectiveness_score: float = 0.0,
    ) -> None:
        """
        Store a tool usage example in the database.

        Args:
            example_id: Unique example identifier
            tool_id: ID of the tool this example belongs to
            use_case: Brief description of the use case
            example_call: Example tool invocation
            expected_output: Expected response/output
            context: When to use this pattern
            embedding: 384-dimension embedding vector
            effectiveness_score: Score indicating example effectiveness (0.0-1.0)
        """
        try:
            with self.get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO tool_examples
                    (id, tool_id, use_case, example_call, expected_output,
                     context, embedding, effectiveness_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (id) DO UPDATE SET
                    use_case = EXCLUDED.use_case,
                    example_call = EXCLUDED.example_call,
                    expected_output = EXCLUDED.expected_output,
                    context = EXCLUDED.context,
                    embedding = EXCLUDED.embedding,
                    effectiveness_score = EXCLUDED.effectiveness_score
                    """,
                    (
                        example_id,
                        tool_id,
                        use_case,
                        example_call,
                        expected_output,
                        context,
                        embedding,
                        effectiveness_score,
                    ),
                )
                logger.debug("Stored tool example: %s for tool %s", example_id, tool_id)

        except Exception as e:
            logger.error("Failed to store tool example %s: %s", example_id, e)
            raise DatabaseError(f"Failed to store tool example: {e}") from e

    def store_tool_relationship(
        self,
        relationship_id: str,
        tool_a_id: str,
        tool_b_id: str,
        relationship_type: str,
        strength: float = 0.0,
        description: str = None,
    ) -> None:
        """
        Store a relationship between two tools.

        Args:
            relationship_id: Unique relationship identifier
            tool_a_id: ID of the first tool
            tool_b_id: ID of the second tool
            relationship_type: Type of relationship ('alternative', 'complement', 'prerequisite')
            strength: Relationship strength (0.0-1.0)
            description: Description of the relationship
        """
        try:
            with self.get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO tool_relationships
                    (id, tool_a_id, tool_b_id, relationship_type, strength, description)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT (tool_a_id, tool_b_id, relationship_type) DO UPDATE SET
                    id = EXCLUDED.id,
                    strength = EXCLUDED.strength,
                    description = EXCLUDED.description
                    """,
                    (relationship_id, tool_a_id, tool_b_id, relationship_type, strength, description),
                )
                logger.debug("Stored tool relationship: %s between %s and %s", relationship_type, tool_a_id, tool_b_id)

        except Exception as e:
            logger.error("Failed to store tool relationship %s: %s", relationship_id, e)
            raise DatabaseError(f"Failed to store tool relationship: {e}") from e

    def get_mcp_tool(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an MCP tool by ID.

        Args:
            tool_id: ID of the tool to retrieve

        Returns:
            Dictionary with tool data, or None if not found
        """
        try:
            with self.get_connection() as conn:
                result = conn.execute("SELECT * FROM mcp_tools WHERE id = ?", (tool_id,)).fetchone()

                if result:
                    columns = [desc[0] for desc in conn.description]
                    return dict(zip(columns, result))
                return None

        except Exception as e:
            logger.error("Failed to get MCP tool %s: %s", tool_id, e)
            return None

    def get_tool_parameters(self, tool_id: str) -> List[Dict[str, Any]]:
        """
        Get all parameters for a tool.

        Args:
            tool_id: ID of the tool

        Returns:
            List of parameter dictionaries
        """
        try:
            with self.get_connection() as conn:
                results = conn.execute(
                    "SELECT * FROM tool_parameters WHERE tool_id = ? ORDER BY parameter_name", (tool_id,)
                ).fetchall()

                if results:
                    columns = [desc[0] for desc in conn.description]
                    return [dict(zip(columns, row)) for row in results]
                return []

        except Exception as e:
            logger.error("Failed to get tool parameters for %s: %s", tool_id, e)
            return []

    def get_tool_examples(self, tool_id: str) -> List[Dict[str, Any]]:
        """
        Get all examples for a tool.

        Args:
            tool_id: ID of the tool

        Returns:
            List of example dictionaries
        """
        try:
            with self.get_connection() as conn:
                results = conn.execute(
                    "SELECT * FROM tool_examples WHERE tool_id = ? ORDER BY effectiveness_score DESC", (tool_id,)
                ).fetchall()

                if results:
                    columns = [desc[0] for desc in conn.description]
                    return [dict(zip(columns, row)) for row in results]
                return []

        except Exception as e:
            logger.error("Failed to get tool examples for %s: %s", tool_id, e)
            return []

    def get_related_tools(self, tool_id: str, relationship_type: str = None) -> List[Dict[str, Any]]:
        """
        Get tools related to a given tool.

        Args:
            tool_id: ID of the tool to find relationships for
            relationship_type: Optional filter by relationship type

        Returns:
            List of related tool dictionaries with relationship info
        """
        try:
            with self.get_connection() as conn:
                base_query = """
                    SELECT mt.*, tr.relationship_type, tr.strength, tr.description as rel_description
                    FROM tool_relationships tr
                    JOIN mcp_tools mt ON (tr.tool_b_id = mt.id OR tr.tool_a_id = mt.id)
                    WHERE (tr.tool_a_id = ? OR tr.tool_b_id = ?) AND mt.id != ?
                """

                params = [tool_id, tool_id, tool_id]

                if relationship_type:
                    base_query += " AND tr.relationship_type = ?"
                    params.append(relationship_type)

                base_query += " ORDER BY tr.strength DESC"

                results = conn.execute(base_query, params).fetchall()

                if results:
                    columns = [desc[0] for desc in conn.description]
                    return [dict(zip(columns, row)) for row in results]
                return []

        except Exception as e:
            logger.error("Failed to get related tools for %s: %s", tool_id, e)
            return []

    def _build_embedding_search_query(self, table_name: str, filters: Dict[str, Any], limit: int) -> tuple[str, list]:
        """
        Build SQL query and parameters for embedding-based search.

        Args:
            table_name: Table to search in
            filters: Dictionary of filter conditions
            limit: Result limit

        Returns:
            Tuple of (query_string, parameters_list)
        """
        base_query = f"""
            SELECT *,
                   list_dot_product(embedding, ?) as similarity_score
            FROM {table_name}
            WHERE embedding IS NOT NULL
        """

        params = [filters.pop("query_embedding")]  # Always first parameter

        # Add dynamic filters
        for column, value in filters.items():
            if value is not None:
                base_query += f" AND {column} = ?"
                params.append(value)

        base_query += " ORDER BY similarity_score DESC LIMIT ?"
        params.append(limit)

        return base_query, params

    def search_mcp_tools_by_embedding(
        self,
        query_embedding: list,
        limit: int = 10,
        category: str = None,
        tool_type: str = None,
        is_active: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search MCP tools using semantic similarity.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results to return
            category: Optional filter by category
            tool_type: Optional filter by tool type
            is_active: Whether to include only active tools

        Returns:
            List of tool dictionaries with similarity scores
        """
        try:
            filters = {
                "query_embedding": query_embedding,
                "is_active": is_active,
                "category": category,
                "tool_type": tool_type,
            }

            query, params = self._build_embedding_search_query("mcp_tools", filters, limit)

            with self.get_connection() as conn:
                results = conn.execute(query, params).fetchall()

                if results:
                    columns = [desc[0] for desc in conn.description]
                    return [dict(zip(columns, row)) for row in results]
                return []

        except Exception as e:
            logger.error("Failed to search MCP tools by embedding: %s", e)
            return []

    def search_tool_parameters_by_embedding(
        self, query_embedding: list, limit: int = 10, tool_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        Search tool parameters using semantic similarity.

        Args:
            query_embedding: 384-dimension query embedding
            limit: Maximum number of results to return
            tool_id: Optional filter by specific tool

        Returns:
            List of parameter dictionaries with similarity scores
        """
        try:
            with self.get_connection() as conn:
                base_query = """
                    SELECT tp.*, mt.name as tool_name,
                           list_dot_product(tp.embedding, ?) as similarity_score
                    FROM tool_parameters tp
                    JOIN mcp_tools mt ON tp.tool_id = mt.id
                    WHERE tp.embedding IS NOT NULL
                """

                params = [query_embedding]

                if tool_id:
                    base_query += " AND tp.tool_id = ?"
                    params.append(tool_id)

                base_query += " ORDER BY similarity_score DESC LIMIT ?"
                params.append(limit)

                results = conn.execute(base_query, params).fetchall()

                if results:
                    columns = [desc[0] for desc in conn.description]
                    return [dict(zip(columns, row)) for row in results]
                return []

        except Exception as e:
            logger.error("Failed to search tool parameters by embedding: %s", e)
            return []

    def get_mcp_tool_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about MCP tools in the database.

        Returns:
            Dictionary with various statistics
        """
        try:
            with self.get_connection() as conn:
                stats = {}

                # Total tool count
                result = conn.execute("SELECT COUNT(*) FROM mcp_tools").fetchone()
                stats["total_tools"] = result[0] if result else 0

                # Active tool count
                result = conn.execute("SELECT COUNT(*) FROM mcp_tools WHERE is_active = true").fetchone()
                stats["active_tools"] = result[0] if result else 0

                # Tool count by type
                type_results = conn.execute("SELECT tool_type, COUNT(*) FROM mcp_tools GROUP BY tool_type").fetchall()
                stats["tools_by_type"] = {row[0] or "unknown": row[1] for row in type_results}

                # Tool count by category
                category_results = conn.execute("SELECT category, COUNT(*) FROM mcp_tools GROUP BY category").fetchall()
                stats["tools_by_category"] = {row[0] or "unknown": row[1] for row in category_results}

                # Parameter count
                result = conn.execute("SELECT COUNT(*) FROM tool_parameters").fetchone()
                stats["total_parameters"] = result[0] if result else 0

                # Example count
                result = conn.execute("SELECT COUNT(*) FROM tool_examples").fetchone()
                stats["total_examples"] = result[0] if result else 0

                # Relationship count
                result = conn.execute("SELECT COUNT(*) FROM tool_relationships").fetchone()
                stats["total_relationships"] = result[0] if result else 0

                # Tools with embeddings
                result = conn.execute("SELECT COUNT(*) FROM mcp_tools WHERE embedding IS NOT NULL").fetchone()
                stats["tools_with_embeddings"] = result[0] if result else 0

                return stats

        except Exception as e:
            logger.error("Failed to get MCP tool statistics: %s", e)
            return {}

    def remove_mcp_tool(self, tool_id: str) -> bool:
        """
        Remove an MCP tool and all its related data.

        Args:
            tool_id: ID of the tool to remove

        Returns:
            True if tool was removed, False if not found
        """
        # Use execute_transaction for proper transaction handling
        operations = [
            ("DELETE FROM tool_parameters WHERE tool_id = ?", (tool_id,)),
            ("DELETE FROM tool_examples WHERE tool_id = ?", (tool_id,)),
            ("DELETE FROM tool_relationships WHERE tool_a_id = ? OR tool_b_id = ?", (tool_id, tool_id)),
            ("DELETE FROM mcp_tools WHERE id = ?", (tool_id,)),
        ]

        try:
            # Check if tool exists first
            with self.get_connection() as conn:
                result = conn.execute("SELECT COUNT(*) FROM mcp_tools WHERE id = ?", (tool_id,)).fetchone()

                if not result or result[0] == 0:
                    logger.warning("Tool %s not found for removal", tool_id)
                    return False

            # Execute all deletion operations in a transaction
            self.execute_transaction(operations)
            logger.info("Removed MCP tool: %s", tool_id)
            return True

        except Exception as e:
            logger.error("Failed to remove MCP tool %s: %s", tool_id, e)
            return False
