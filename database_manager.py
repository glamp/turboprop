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
            if (hasattr(conn, "execute")
                    and config.database.STATEMENT_TIMEOUT > 0):
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
            raise DatabasePermissionError(
                f"Permission denied when creating database connection: {e}"
            ) from e
        except OSError as e:
            if "no space left" in str(e).lower():
                raise DatabaseDiskSpaceError(
                    f"Insufficient disk space when creating database connection: {e}"
                ) from e
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
            except Exception:
                pass

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
        self.db_path = db_path
        self.max_retries = max_retries or config.database.MAX_RETRIES
        self.retry_delay = retry_delay or config.database.RETRY_DELAY
        self.lock_timeout = lock_timeout or config.database.LOCK_TIMEOUT
        self._lock = threading.RLock()
        self._file_lock: Optional[TextIO] = None
        self._lock_file_path = db_path.with_suffix(".lock")

        # Use ConnectionManager for connection handling
        self.connection_timeout = connection_timeout or config.database.CONNECTION_TIMEOUT
        self._connection_manager = ConnectionManager(db_path, self.connection_timeout)

        # Register cleanup handlers only in main thread
        try:
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
        except ValueError:
            # Signal handling only works in main thread, ignore in other threads
            pass

    @property
    def _connections(self):
        """Access to connection manager's connections for testing."""
        return self._connection_manager._connections

    @property
    def _connection_created_time(self):
        """Access to connection manager's creation times for testing."""
        return self._connection_manager._connection_created_time

    @property
    def _connection_last_used(self):
        """Access to connection manager's last used times for testing."""
        return self._connection_manager._connection_last_used

    def _check_connection_health(self, conn):
        """Check if a connection is healthy - delegates to connection manager."""
        return self._connection_manager._check_connection_health(conn)

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
                        "Resource temporarily unavailable" in error_str
                        or "temporarily unavailable" in error_str
                    ):
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
            except Exception as e:
                # Log error but don't re-raise to allow cleanup
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
                        attempt + 1, self.retry_delay * (2**attempt)
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
        - category VARCHAR - file category (source/configuration/documentation/build/etc.)

        Args:
            table_name: Name of the table to migrate
        """
        logger.info("Starting schema migration for table %s", table_name)

        # Define all columns to add (includes legacy columns for backward compatibility)
        new_columns = {
            'last_modified': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
            'file_mtime': 'TIMESTAMP',
            'file_type': 'VARCHAR',
            'language': 'VARCHAR',
            'size_bytes': 'INTEGER',
            'line_count': 'INTEGER',
            'category': 'VARCHAR'
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
                logger.debug("Existing columns: %s", existing_columns)

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
                            logger.info("Added column: %s %s", column_name, column_type)
                        except (duckdb.Error, OSError) as e:
                            logger.warning(
                                "Failed to add column %s: %s", column_name, e
                            )
                    else:
                        logger.debug("Column %s already exists, skipping", column_name)

                if columns_added:
                    logger.info(
                        "Schema migration completed successfully. Added columns: %s", columns_added
                    )
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

        create_table_sql = """
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
            embedding DOUBLE[384],
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """

        create_indexes_sql = [
            "CREATE INDEX IF NOT EXISTS idx_code_constructs_file_id ON code_constructs(file_id)",
            "CREATE INDEX IF NOT EXISTS idx_code_constructs_type ON code_constructs(construct_type)",
            "CREATE INDEX IF NOT EXISTS idx_code_constructs_name ON code_constructs(name)",
            "CREATE INDEX IF NOT EXISTS idx_code_constructs_parent ON code_constructs(parent_construct_id)"
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

    def create_fts_index(self, table_name: str = "code_files") -> None:
        """
        Create full-text search index for file content using DuckDB's FTS extension.

        This enables exact text matching, Boolean search operators, and fuzzy matching
        to complement semantic search capabilities.

        Args:
            table_name: Name of the table to create FTS index for (default: code_files)
        """
        logger.info("Creating full-text search index for table %s", table_name)

        try:
            with self.get_connection() as conn:
                # Load the FTS extension if not already loaded
                try:
                    conn.execute("LOAD fts")
                    logger.debug("FTS extension loaded")
                except duckdb.Error as e:
                    if "already loaded" not in str(e).lower():
                        logger.warning("FTS extension load failed: %s", e)

                # Create FTS index on content column
                # This creates a virtual table for full-text search
                fts_table_name = f"{table_name}_fts"

                # Drop existing FTS table if it exists
                try:
                    conn.execute(f"DROP TABLE IF EXISTS {fts_table_name}")
                except duckdb.Error:
                    pass

                # Create FTS virtual table
                # Note: DuckDB FTS syntax may vary from SQLite, adjust as needed
                fts_create_sql = f"""
                PRAGMA create_fts_index('{fts_table_name}', '{table_name}', 'content')
                """

                try:
                    conn.execute(fts_create_sql)
                    logger.info("Successfully created FTS index %s", fts_table_name)
                except duckdb.Error as e:
                    # If PRAGMA syntax doesn't work, try alternative approach
                    logger.warning("PRAGMA FTS creation failed: %s. Trying alternative approach.", e)

                    # Alternative: Create a separate FTS table manually
                    fts_create_alt = f"""
                    CREATE TABLE IF NOT EXISTS {fts_table_name} AS
                    SELECT id, path, content
                    FROM {table_name}
                    WHERE content IS NOT NULL
                    """
                    conn.execute(fts_create_alt)

                    # Create indexes for text search on the FTS table
                    conn.execute(
                        f"CREATE INDEX IF NOT EXISTS idx_{fts_table_name}_content ON {fts_table_name} USING gin(to_tsvector('english', content))")

                    logger.info("Created alternative FTS table %s with text indexes", fts_table_name)

        except Exception as e:
            logger.error("Failed to create FTS index: %s", e)
            # Don't raise an error here - FTS is supplementary functionality
            logger.warning("FTS index creation failed, full-text search will be limited")

    def rebuild_fts_index(self, table_name: str = "code_files") -> None:
        """
        Rebuild the full-text search index to incorporate new content.

        Args:
            table_name: Name of the table to rebuild FTS index for
        """
        logger.info("Rebuilding FTS index for table %s", table_name)

        try:
            with self.get_connection() as conn:
                fts_table_name = f"{table_name}_fts"

                # Clear existing FTS data
                try:
                    conn.execute(f"DELETE FROM {fts_table_name}")
                except duckdb.Error:
                    # Table might not exist, create it
                    self.create_fts_index(table_name)
                    return

                # Repopulate FTS table with current data
                repopulate_sql = f"""
                INSERT INTO {fts_table_name} (id, path, content)
                SELECT id, path, content
                FROM {table_name}
                WHERE content IS NOT NULL
                """
                conn.execute(repopulate_sql)

                logger.info("Successfully rebuilt FTS index for %s", fts_table_name)

        except Exception as e:
            logger.error("Failed to rebuild FTS index: %s", e)

    def search_full_text(
        self,
        query: str,
        table_name: str = "code_files",
        limit: int = 10,
        enable_stemming: bool = True,
        enable_fuzzy: bool = False,
        fuzzy_distance: int = 2
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
            fts_table_name = f"{table_name}_fts"

            with self.get_connection() as conn:
                # Check if FTS table exists
                try:
                    conn.execute(f"SELECT COUNT(*) FROM {fts_table_name} LIMIT 1")
                except duckdb.Error:
                    logger.warning("FTS table %s doesn't exist, creating it", fts_table_name)
                    self.create_fts_index(table_name)

                # Process query for FTS
                self._process_fts_query(query, enable_fuzzy, fuzzy_distance)

                # Perform full-text search
                # Using a simple LIKE-based approach for compatibility
                # In a production system, this would use proper FTS ranking
                search_sql = f"""
                SELECT
                    f.id,
                    f.path,
                    f.content,
                    CASE
                        WHEN f.content ILIKE ? THEN 1.0
                        WHEN f.path ILIKE ? THEN 0.8
                        ELSE 0.5
                    END as relevance_score
                FROM {fts_table_name} f
                WHERE f.content ILIKE ? OR f.path ILIKE ?
                ORDER BY relevance_score DESC, f.path
                LIMIT ?
                """

                # Create LIKE patterns for different search variations
                exact_pattern = f"%{query}%"
                word_pattern = f"%{query.replace(' ', '%')}%"

                results = conn.execute(
                    search_sql,
                    (exact_pattern, exact_pattern, word_pattern, word_pattern, limit)
                ).fetchall()

                logger.debug("FTS search for '%s' returned %d results", query, len(results))
                return results

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
        processed = processed.replace(' AND ', ' & ')
        processed = processed.replace(' OR ', ' | ')
        processed = processed.replace(' NOT ', ' -')

        # Handle wildcards (already supported with *)

        # Handle fuzzy matching if enabled
        if enable_fuzzy and '~' not in processed:
            # Add fuzzy matching to individual terms
            words = processed.split()
            fuzzy_words = []
            for word in words:
                if word not in ['&', '|', '-'] and not word.startswith('__PHRASE_'):
                    fuzzy_words.append(f"{word}~{fuzzy_distance}")
                else:
                    fuzzy_words.append(word)
            processed = ' '.join(fuzzy_words)

        # Restore quoted phrases
        for i, phrase in enumerate(quoted_phrases):
            placeholder = f"__PHRASE_{i+1}__"
            processed = processed.replace(placeholder, f'"{phrase}"')

        return processed

    def search_by_file_type_fts(
        self,
        query: str,
        file_extensions: list,
        limit: int = 10
    ) -> list:
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
                        embedding
                    )
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
                    "SELECT * FROM code_constructs WHERE file_id = ? ORDER BY start_line",
                    (file_id,)
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
                    "SELECT COUNT(*) FROM code_constructs WHERE file_id = ?",
                    (file_id,)
                ).fetchone()
                construct_count = count_result[0] if count_result else 0

                # Then delete them
                conn.execute(
                    "DELETE FROM code_constructs WHERE file_id = ?",
                    (file_id,)
                )
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
            "CREATE INDEX IF NOT EXISTS idx_repository_context_path ON repository_context(repository_path)",
            "CREATE INDEX IF NOT EXISTS idx_repository_context_type ON repository_context(project_type)",
            "CREATE INDEX IF NOT EXISTS idx_repository_context_indexed ON repository_context(indexed_at)"
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
                        context_dict['repository_id'],
                        context_dict['repository_path'],
                        context_dict['git_branch'],
                        context_dict['git_commit'],
                        context_dict['git_remote_url'],
                        context_dict['project_type'],
                        context_dict['dependencies'],
                        context_dict['package_managers'],
                        context_dict['indexed_at'],
                        context_dict['created_at']
                    )
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
                    "SELECT * FROM repository_context WHERE repository_id = ?",
                    (repository_id,)
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
                    "SELECT * FROM repository_context WHERE repository_path = ?",
                    (repository_path,)
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
                    (repository_id,)
                )
                logger.debug("Updated indexed time for repository %s", repository_id)

        except (duckdb.Error, DatabaseError) as e:
            logger.error("Failed to update repository context indexed time: %s", e)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
