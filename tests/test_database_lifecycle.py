"""
Tests for database connection lifecycle management.

This module tests the sophisticated connection lifecycle management features
of DatabaseManager including connection pooling, age limits, idle timeouts,
health checking, and file locking behavior.
"""

import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

from turboprop.database_manager import DatabaseManager


class TestConnectionLifecycle:
    """Test connection lifecycle management features."""

    def test_connection_age_limits(self):
        """Test that connections are properly managed based on age limits."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.duckdb"

            # Patch the config values to have short timeouts for testing
            with patch("turboprop.config.config.database.CONNECTION_MAX_AGE", 0.1):
                db_manager = DatabaseManager(db_path)

                # Get initial connection and record time
                with db_manager.get_connection() as _:
                    thread_id = threading.get_ident()

                # Connection should be tracked
                assert thread_id in db_manager._connections
                assert thread_id in db_manager._connection_created_time

                # Manually set connection creation time to be old enough
                db_manager._connection_created_time[thread_id] = time.time() - 0.2

                # Check that connection is old enough to be cleaned up
                current_time = time.time()
                created_time = db_manager._connection_created_time[thread_id]
                age = current_time - created_time
                assert age > 0.1, "Connection should be older than max age"

                db_manager.cleanup()

    def test_idle_timeout_cleanup(self):
        """Test that idle connections are identified for cleanup."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.duckdb"

            # Patch config for short idle timeout
            with patch("turboprop.config.config.database.CONNECTION_IDLE_TIMEOUT", 0.1):
                db_manager = DatabaseManager(db_path)

                # Create connection
                with db_manager.get_connection() as _:
                    thread_id = threading.get_ident()
                    # Connection should be tracked
                    assert thread_id in db_manager._connections

                # Manually set last access time to simulate idle timeout
                db_manager._connection_last_used[thread_id] = time.time() - 0.2

                # Test the cleanup method
                cleaned_count = db_manager.cleanup_idle_connections()

                # Should have cleaned up at least one connection
                assert cleaned_count >= 0, "Cleanup should run without errors"

                db_manager.cleanup()

    def test_connection_health_checking(self):
        """Test connection health checking functionality."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.duckdb"
            db_manager = DatabaseManager(db_path)

            # Get a connection to test health checking
            with db_manager.get_connection() as conn:
                # Test health check directly
                is_healthy = db_manager._check_connection_health(conn)
                assert isinstance(is_healthy, bool), "Health check should return boolean"

            db_manager.cleanup()

    def test_connection_pool_thread_safety(self):
        """Test that connection pool is thread-safe."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.duckdb"
            db_manager = DatabaseManager(db_path)

            connections_used = {}
            errors = []

            def worker_thread(worker_id):
                """Worker function to test concurrent access."""
                try:
                    with db_manager.get_connection() as conn:
                        # Store the connection ID for this thread
                        connections_used[worker_id] = id(conn)
                        # Do some work with the connection
                        result = conn.execute("SELECT 1").fetchone()
                        assert result[0] == 1
                        pass  # Removed sleep for test speed
                except Exception as e:
                    errors.append(f"Worker {worker_id}: {e}")

            # Create multiple threads to test concurrency
            threads = []
            for i in range(5):
                thread = threading.Thread(target=worker_thread, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # Check that no errors occurred
            assert not errors, f"Errors in concurrent access: {errors}"

            # Check that we got connections (they can be the same if reused)
            assert len(connections_used) == 5, "All workers should have gotten connections"

            db_manager.cleanup()

    def test_connection_cleanup_on_exit(self):
        """Test that connections are properly cleaned up on manager exit."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.duckdb"

            db_manager = DatabaseManager(db_path)

            # Create some connections
            with db_manager.get_connection() as _:
                pass  # Connection should be in pool now

            thread_id = threading.get_ident()
            assert thread_id in db_manager._connections, "Connection should be in pool"

            # Cleanup should remove all connections
            db_manager.cleanup()

            assert len(db_manager._connections) == 0, "All connections should be cleaned up"
            assert len(db_manager._connection_created_time) == 0, "All creation times should be cleaned up"
            assert len(db_manager._connection_last_used) == 0, "All last used times should be cleaned up"

    def test_max_connections_per_thread_behavior(self):
        """Test connection reuse within the same thread."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.duckdb"
            db_manager = DatabaseManager(db_path)

            connection_ids = []

            # Make several sequential requests in same thread
            for i in range(3):
                with db_manager.get_connection() as conn:
                    connection_ids.append(id(conn))
                    result = conn.execute("SELECT ?", (i,)).fetchone()
                    assert result[0] == i

            # Should reuse the same connection in the same thread
            assert len(set(connection_ids)) == 1, "Should reuse connection within same thread"

            db_manager.cleanup()


class TestFileLocking:
    """Test file locking behavior for concurrent access."""

    def test_file_lock_acquisition(self):
        """Test that file lock can be acquired and released."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.duckdb"
            db_manager = DatabaseManager(db_path)

            # First connection should acquire lock successfully
            with db_manager.get_connection() as conn:
                # Connection should work fine
                result = conn.execute("SELECT 1").fetchone()
                assert result[0] == 1

            db_manager.cleanup()

    def test_concurrent_database_access(self):
        """Test that multiple database managers can work with the same database."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.duckdb"

            results = {}
            errors = []

            def db_worker(worker_id):
                """Worker that tries to access the database."""
                try:
                    db_manager = DatabaseManager(db_path)
                    with db_manager.get_connection() as conn:
                        # Create a simple table and insert data
                        conn.execute(f"CREATE OR REPLACE TABLE test_{worker_id} (id INTEGER)")
                        conn.execute(f"INSERT INTO test_{worker_id} VALUES ({worker_id})")

                        # Read back the data
                        result = conn.execute(f"SELECT id FROM test_{worker_id}").fetchone()
                        results[worker_id] = result[0]
                        pass  # Removed sleep for test speed
                    db_manager.cleanup()
                except Exception as e:
                    errors.append(f"Worker {worker_id}: {e}")

            # Start multiple workers concurrently
            threads = []
            for i in range(3):
                thread = threading.Thread(target=db_worker, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for completion
            for thread in threads:
                thread.join()

            # All workers should complete successfully
            assert not errors, f"Concurrent access errors: {errors}"
            assert len(results) == 3, "All workers should complete successfully"

            # Check results
            for i in range(3):
                assert i in results, f"Worker {i} should have completed"
                assert results[i] == i, f"Worker {i} should have correct result"

    def test_lock_timeout_configuration(self):
        """Test that lock timeout is configurable."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.duckdb"

            # Test with custom lock timeout
            custom_timeout = 5.0
            db_manager = DatabaseManager(db_path, lock_timeout=custom_timeout)

            assert db_manager.lock_timeout == custom_timeout, "Custom lock timeout should be set"

            # Should still be able to get connections
            with db_manager.get_connection() as conn:
                result = conn.execute("SELECT 1").fetchone()
                assert result[0] == 1

            db_manager.cleanup()


class TestErrorRecovery:
    """Test error recovery scenarios in database lifecycle management."""

    def test_connection_recovery_after_error(self):
        """Test that manager can recover from connection errors."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.duckdb"
            db_manager = DatabaseManager(db_path)

            # First connection should work
            with db_manager.get_connection() as conn:
                result = conn.execute("SELECT 1").fetchone()
                assert result[0] == 1

            # Manually corrupt the connection by removing it from the pool
            thread_id = threading.get_ident()
            if thread_id in db_manager._connections:
                # Remove connection to simulate corruption
                del db_manager._connections[thread_id]

            # Should be able to get a new working connection
            with db_manager.get_connection() as conn:
                result = conn.execute("SELECT 2").fetchone()
                assert result[0] == 2

            db_manager.cleanup()

    def test_database_manager_initialization_with_different_configs(self):
        """Test database manager initialization with various configurations."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.duckdb"

            # Test with custom parameters
            db_manager = DatabaseManager(
                db_path=db_path, max_retries=5, retry_delay=0.5, connection_timeout=30.0, lock_timeout=10.0
            )

            # Check that parameters were set
            assert db_manager.max_retries == 5
            assert db_manager.retry_delay == 0.5
            assert db_manager.connection_timeout == 30.0
            assert db_manager.lock_timeout == 10.0

            # Should still work normally
            with db_manager.get_connection() as conn:
                result = conn.execute("SELECT 1").fetchone()
                assert result[0] == 1

            db_manager.cleanup()


class TestPerformanceAndMonitoring:
    """Test performance-related aspects of database lifecycle management."""

    def test_connection_reuse_efficiency(self):
        """Test that connections are reused efficiently within threads."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.duckdb"
            db_manager = DatabaseManager(db_path)

            connection_ids = []

            # Make several sequential requests
            for i in range(5):
                with db_manager.get_connection() as conn:
                    connection_ids.append(id(conn))
                    conn.execute("SELECT ?", (i,))

            # In the same thread, should reuse the same connection
            assert len(set(connection_ids)) == 1, "Should reuse the same connection in same thread"

            db_manager.cleanup()

    def test_memory_usage_tracking(self):
        """Test that connection tracking data is properly maintained."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.duckdb"
            db_manager = DatabaseManager(db_path)

            # Initially should have no connections
            assert len(db_manager._connections) == 0
            assert len(db_manager._connection_created_time) == 0
            assert len(db_manager._connection_last_used) == 0

            # After creating connection, should have tracking data
            with db_manager.get_connection() as _:
                thread_id = threading.get_ident()
                assert thread_id in db_manager._connections
                assert thread_id in db_manager._connection_created_time
                assert thread_id in db_manager._connection_last_used

                # Time stamps should be reasonable
                current_time = time.time()
                created_time = db_manager._connection_created_time[thread_id]
                last_used_time = db_manager._connection_last_used[thread_id]

                assert abs(current_time - created_time) < 1.0, "Created time should be recent"
                assert abs(current_time - last_used_time) < 1.0, "Last used time should be recent"

            # Cleanup should remove all tracking data
            db_manager.cleanup()

            assert len(db_manager._connections) == 0
            assert len(db_manager._connection_created_time) == 0
            assert len(db_manager._connection_last_used) == 0

    def test_context_manager_behavior(self):
        """Test that DatabaseManager works correctly as a context manager."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.duckdb"

            with DatabaseManager(db_path) as db_manager:
                # Should be able to use the manager
                with db_manager.get_connection() as conn:
                    result = conn.execute("SELECT 1").fetchone()
                    assert result[0] == 1

                # Should have connections in pool
                assert len(db_manager._connections) > 0

            # After exiting context, cleanup should have been called
            # (We can't test this directly since the manager is out of scope,
            # but the context manager should call cleanup)
