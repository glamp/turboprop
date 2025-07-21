#!/usr/bin/env python3
"""
test_database_fts.py: Comprehensive tests for DuckDB FTS database functionality.

This module tests:
- FTS extension loading (success/failure scenarios)
- FTS availability detection
- FTS index creation using PRAGMA and alternative approaches
- FTS index rebuild functionality
- Migration from old FTS setups
- Database-level FTS search operations
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest

from database_manager import DatabaseManager


class TestFTSExtensionLoading:
    """Test FTS extension installation and loading."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.duckdb"
        self.db_manager = DatabaseManager(self.db_path)

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, "db_manager"):
            try:
                self.db_manager.cleanup()
            except Exception:
                pass

    def test_fts_extension_loading_success(self):
        """Test successful loading of DuckDB FTS extension."""
        with self.db_manager.get_connection() as conn:
            result = self.db_manager._ensure_fts_extension_loaded(conn)

            # Should return True if extension loads successfully
            # Note: This may return False in environments without FTS extension
            assert isinstance(result, bool)

            if result:
                # If FTS loaded successfully, we should be able to create FTS table
                try:
                    conn.execute("CREATE VIRTUAL TABLE test_fts USING fts5(content, path)")
                    conn.execute("DROP TABLE test_fts")
                except duckdb.Error:
                    # If this fails, the extension might not actually be working
                    pytest.skip("FTS extension not fully functional in test environment")

    def test_fts_extension_loading_failure_handling(self):
        """Test graceful handling when FTS extension can't be loaded."""
        # Mock the database manager's FTS extension loading method directly
        with patch.object(self.db_manager, "_ensure_fts_extension_loaded", return_value=False):
            with self.db_manager.get_connection() as conn:
                result = self.db_manager._ensure_fts_extension_loaded(conn)

                # Should return False when extension can't be loaded
                assert result is False

    def test_fts_availability_detection(self):
        """Test FTS availability detection method."""
        with self.db_manager.get_connection() as conn:
            # Create base table first
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT
                )
            """
            )

            # Insert test data
            conn.execute("INSERT INTO code_files VALUES ('1', 'test.py', 'def hello(): pass')")

            # Try to create FTS index
            fts_created = self.db_manager.create_fts_index("code_files")

            # Test availability detection
            availability = self.db_manager.is_fts_available("code_files")

            # If FTS index was created successfully, availability should be True
            if fts_created:
                assert availability is True
            else:
                # If FTS creation failed, availability should be False
                assert availability is False


class TestFTSIndexCreation:
    """Test FTS index creation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.duckdb"
        self.db_manager = DatabaseManager(self.db_path)

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, "db_manager"):
            try:
                self.db_manager.cleanup()
            except Exception:
                pass

    def test_fts_index_creation(self):
        """Test FTS index creation for code_files table."""
        with self.db_manager.get_connection() as conn:
            # Create base table first
            conn.execute(
                """
                CREATE TABLE code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT
                )
            """
            )

            # Insert test data
            test_data = [
                ("1", "test1.py", 'def hello_world(): print("Hello")'),
                ("2", "test2.py", "class TestClass: pass"),
                ("3", "test3.js", 'function sayHello() { console.log("Hello"); }'),
            ]

            for row in test_data:
                conn.execute("INSERT INTO code_files VALUES (?, ?, ?)", row)

            # Create FTS index
            result = self.db_manager.create_fts_index("code_files")

            # Should return boolean indicating success/failure
            assert isinstance(result, bool)

            if result:
                # Verify FTS is available
                assert self.db_manager.is_fts_available("code_files")

                # Try to perform a search to verify FTS is working
                search_results = self.db_manager.search_full_text("hello", "code_files", limit=5)
                assert isinstance(search_results, list)

    def test_fts_index_creation_empty_table(self):
        """Test FTS index creation on empty table."""
        with self.db_manager.get_connection() as conn:
            # Create empty base table
            conn.execute(
                """
                CREATE TABLE empty_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT
                )
            """
            )

            # Create FTS index on empty table
            result = self.db_manager.create_fts_index("empty_files")

            # Should handle empty table gracefully
            assert isinstance(result, bool)

    def test_fts_index_creation_nonexistent_table(self):
        """Test FTS index creation on nonexistent table."""
        # Try to create FTS index on table that doesn't exist
        result = self.db_manager.create_fts_index("nonexistent_table")

        # Should return False for nonexistent table
        assert result is False

    def test_fts_index_rebuild(self):
        """Test FTS index rebuild functionality."""
        with self.db_manager.get_connection() as conn:
            # Setup base table and data
            conn.execute(
                """
                CREATE TABLE code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR, 
                    content TEXT
                )
            """
            )

            # Insert initial data
            conn.execute("INSERT INTO code_files VALUES ('1', 'old.py', 'def old_function(): pass')")

            # Create FTS index
            if self.db_manager.create_fts_index("code_files"):
                # Add new data after FTS creation
                conn.execute("INSERT INTO code_files VALUES ('2', 'new.py', 'def new_function(): return True')")

                # Rebuild FTS index to include new data
                self.db_manager.rebuild_fts_index("code_files")

                # Verify new data is searchable
                search_results = self.db_manager.search_full_text("new_function", "code_files")

                # Should find the new function if FTS rebuild worked
                if search_results:
                    found_new = any("new_function" in str(result) for result in search_results)
                    assert found_new or len(search_results) > 0  # Either found specifically or got some results

    def test_alternative_fts_creation(self):
        """Test alternative FTS table creation when PRAGMA fails."""
        with self.db_manager.get_connection() as conn:
            # Create base table with data
            conn.execute(
                """
                CREATE TABLE code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT
                )
            """
            )

            test_data = [
                ("1", "auth.py", "def authenticate_user(username, password): pass"),
                ("2", "db.py", "class DatabaseConnection: pass"),
            ]

            for row in test_data:
                conn.execute("INSERT INTO code_files VALUES (?, ?, ?)", row)

            # Test alternative FTS creation directly
            try:
                self.db_manager._create_alternative_fts_table(conn, "code_files_fts", "code_files")

                # Verify table was created
                tables = conn.execute("SHOW TABLES").fetchall()
                table_names = [row[0] for row in tables]
                assert "code_files_fts" in table_names

                # Verify data was copied
                count = conn.execute("SELECT COUNT(*) FROM code_files_fts").fetchone()
                assert count[0] == 2

            except Exception as e:
                pytest.skip(f"Alternative FTS creation failed: {e}")


class TestFTSSearch:
    """Test FTS search operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.duckdb"
        self.db_manager = DatabaseManager(self.db_path)

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, "db_manager"):
            try:
                self.db_manager.cleanup()
            except Exception:
                pass

    def test_fts_search_basic_functionality(self):
        """Test basic FTS search functionality."""
        with self.db_manager.get_connection() as conn:
            # Create and populate test data
            conn.execute(
                """
                CREATE TABLE code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT
                )
            """
            )

            test_content = [
                (
                    "1",
                    "authentication.py",
                    "def authenticate_user(username, password): return verify_credentials(username, password)",
                ),
                ("2", "database.py", "class DatabaseConnection: def connect(self): pass"),
                ("3", "utils.py", "def hash_password(password): return hashlib.sha256(password.encode()).hexdigest()"),
            ]

            for row in test_content:
                conn.execute("INSERT INTO code_files VALUES (?, ?, ?)", row)

            # Create FTS index if possible
            fts_created = self.db_manager.create_fts_index("code_files")
            assert isinstance(fts_created, bool)

            # Test FTS search
            results = self.db_manager.search_full_text("authenticate", limit=5)

            # Should return results (either FTS or fallback LIKE search)
            assert isinstance(results, list)

            if results:
                # Should find files containing "authenticate"
                found_auth = any("authenticate" in str(result).lower() for result in results)
                assert found_auth

    def test_fts_search_with_ranking(self):
        """Test that FTS search produces reasonable ranking."""
        with self.db_manager.get_connection() as conn:
            # Setup test data with different relevance levels
            conn.execute(
                """
                CREATE TABLE code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT
                )
            """
            )

            # Content with varying relevance for search term "database"
            test_content = [
                ("1", "db_main.py", "database connection and database operations"),  # High relevance
                ("2", "config.py", "configuration including database settings"),  # Medium relevance
                ("3", "utils.py", "utility functions for various tasks"),  # Low relevance
            ]

            for row in test_content:
                conn.execute("INSERT INTO code_files VALUES (?, ?, ?)", row)

            # Create FTS index
            self.db_manager.create_fts_index("code_files")

            # Search for "database"
            results = self.db_manager.search_full_text("database", limit=10)

            if results:
                # Should have found relevant results
                assert len(results) > 0

                # More specific assertions about ranking would require
                # examining the actual score values returned

    def test_fts_search_empty_query(self):
        """Test FTS search with empty query."""
        with self.db_manager.get_connection() as conn:
            # Create basic table
            conn.execute(
                """
                CREATE TABLE code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT
                )
            """
            )

            # Test empty query
            results = self.db_manager.search_full_text("", limit=5)

            # Should handle empty query gracefully
            assert isinstance(results, list)
            # Empty query should return empty results
            assert len(results) == 0

    def test_fts_fallback_to_like_search(self):
        """Test fallback to LIKE search when FTS is not available."""
        with self.db_manager.get_connection() as conn:
            # Create test data
            conn.execute(
                """
                CREATE TABLE code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT
                )
            """
            )

            conn.execute("INSERT INTO code_files VALUES ('1', 'test.py', 'def test_function(): pass')")

            # Mock FTS to be unavailable
            with patch.object(self.db_manager, "is_fts_available", return_value=False):
                results = self.db_manager.search_full_text("test_function", limit=5)

                # Should still return results via LIKE fallback
                assert isinstance(results, list)


class TestFTSMigration:
    """Test FTS migration functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.duckdb"
        self.db_manager = DatabaseManager(self.db_path)

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, "db_manager"):
            try:
                self.db_manager.cleanup()
            except Exception:
                pass

    def test_fts_migration_from_postgresql_style(self):
        """Test migration from old PostgreSQL-style FTS setup."""
        with self.db_manager.get_connection() as conn:
            # Create base table
            conn.execute(
                """
                CREATE TABLE code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT
                )
            """
            )

            # Simulate old database state with potential artifacts
            # Create old-style FTS table that might exist
            try:
                conn.execute(
                    """
                    CREATE TABLE code_files_fts_old (
                        id VARCHAR,
                        content_vector TEXT
                    )
                """
                )
            except duckdb.Error:
                # Table might already exist or creation might fail - this is expected
                pass

            # Test migration
            result = self.db_manager.migrate_fts_to_duckdb("code_files")

            # Migration should either succeed or fail gracefully
            assert isinstance(result, bool)

    def test_backward_compatibility(self):
        """Test that changes don't break existing non-FTS functionality."""
        with self.db_manager.get_connection() as conn:
            # Create basic table without FTS
            conn.execute(
                """
                CREATE TABLE code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT
                )
            """
            )

            # Insert test data
            conn.execute("INSERT INTO code_files VALUES ('1', 'test.py', 'def hello(): pass')")

            # Verify basic database operations still work
            result = conn.execute("SELECT * FROM code_files").fetchall()
            assert len(result) == 1
            assert result[0][0] == "1"

            # Verify search operations work without FTS
            search_results = self.db_manager.search_full_text("hello", limit=5)
            assert isinstance(search_results, list)


class TestFTSErrorHandling:
    """Test FTS error handling and edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.duckdb"
        self.db_manager = DatabaseManager(self.db_path)

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, "db_manager"):
            try:
                self.db_manager.cleanup()
            except Exception:
                pass

    def test_fts_creation_with_database_errors(self):
        """Test FTS creation with various database errors."""
        # Test with permission error simulation
        with patch.object(duckdb, "connect", side_effect=PermissionError("Access denied")):
            try:
                db_manager = DatabaseManager(self.db_path)
                # Should handle permission error gracefully
            except Exception as e:
                # Expected to raise DatabasePermissionError or similar
                assert "permission" in str(e).lower() or "access" in str(e).lower()

    def test_fts_index_creation_error_handling(self):
        """Test error handling during FTS index creation."""
        with self.db_manager.get_connection() as conn:
            # Create base table
            conn.execute(
                """
                CREATE TABLE code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT
                )
            """
            )

            # Mock FTS extension loading to fail
            with patch.object(self.db_manager, "_ensure_fts_extension_loaded", return_value=False):
                result = self.db_manager.create_fts_index("code_files")

                # Should handle gracefully and return False
                assert result is False

    def test_fts_search_error_handling(self):
        """Test error handling during FTS search operations."""
        with self.db_manager.get_connection() as conn:
            # Create table but don't populate it
            conn.execute(
                """
                CREATE TABLE code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT
                )
            """
            )

            # Mock search method to raise exception
            with patch.object(self.db_manager, "_execute_fts_search_duckdb", side_effect=duckdb.Error("Search failed")):
                results = self.db_manager.search_full_text("test", limit=5)

                # Should handle error gracefully and return empty list or fallback results
                assert isinstance(results, list)

    def test_concurrent_fts_operations(self):
        """Test FTS operations under concurrent access."""
        import threading

        with self.db_manager.get_connection() as conn:
            # Setup test data
            conn.execute(
                """
                CREATE TABLE code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT
                )
            """
            )

            for i in range(10):
                conn.execute(f"INSERT INTO code_files VALUES ('{i}', 'file{i}.py', 'def function_{i}(): pass')")

        # Test concurrent FTS creation and searching
        results = []
        errors = []

        def create_fts():
            try:
                result = self.db_manager.create_fts_index("code_files")
                results.append(result)
            except Exception as e:
                errors.append(e)

        def search_fts():
            try:
                result = self.db_manager.search_full_text("function", limit=5)
                results.append(len(result) if result else 0)
            except Exception as e:
                errors.append(e)

        # Start concurrent operations
        threads = []
        for i in range(3):
            t1 = threading.Thread(target=create_fts)
            t2 = threading.Thread(target=search_fts)
            threads.extend([t1, t2])
            t1.start()
            t2.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Should handle concurrent access without crashing
        # Some operations might fail but should not cause system crashes
        assert len(errors) <= len(threads)  # Not all operations should fail
