#!/usr/bin/env python3
"""
Tests for database schema migration functionality.

This test suite verifies that the schema migration function works correctly
for both fresh databases and existing databases with various schema states.
"""

import tempfile
from pathlib import Path

import pytest

from database_manager import DatabaseManager


class TestDatabaseMigration:
    """Test database schema migration functionality."""

    def test_migrate_schema_fresh_database(self):
        """Test migration on a fresh database (should add all columns)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.duckdb"
            db_manager = DatabaseManager(db_path)

            # Create a basic table with minimal columns
            with db_manager.get_connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE code_files (
                        id VARCHAR PRIMARY KEY,
                        path VARCHAR,
                        content TEXT,
                        embedding DOUBLE[384]
                    )
                """
                )

            # Run migration
            db_manager.migrate_schema("code_files")

            # Check that all expected columns exist
            with db_manager.get_connection() as conn:
                result = conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'code_files' ORDER BY ordinal_position"
                ).fetchall()

                column_names = [row[0] for row in result]
                expected_columns = [
                    "id",
                    "path",
                    "content",
                    "embedding",
                    "last_modified",
                    "file_mtime",
                    "file_type",
                    "language",
                    "size_bytes",
                    "line_count",
                    "category"
                ]

                for col in expected_columns:
                    assert col in column_names, f"Column {col} should exist after migration"

            db_manager.cleanup()

    def test_migrate_schema_partial_migration(self):
        """Test migration on a database with some but not all columns."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.duckdb"
            db_manager = DatabaseManager(db_path)

            # Create table with some of the expected columns
            with db_manager.get_connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE code_files (
                        id VARCHAR PRIMARY KEY,
                        path VARCHAR,
                        content TEXT,
                        embedding DOUBLE[384],
                        last_modified TIMESTAMP,
                        file_type VARCHAR
                    )
                """
                )

            # Run migration
            db_manager.migrate_schema("code_files")

            # Check that all expected columns exist
            with db_manager.get_connection() as conn:
                result = conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'code_files'"
                ).fetchall()

                column_names = [row[0] for row in result]
                expected_new_columns = [
                    "file_mtime",
                    "language",
                    "size_bytes",
                    "line_count"
                ]

                for col in expected_new_columns:
                    assert col in column_names, f"New column {col} should be added during migration"

            db_manager.cleanup()

    def test_migrate_schema_already_migrated(self):
        """Test migration on a database that already has all columns."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.duckdb"
            db_manager = DatabaseManager(db_path)

            # Create table with all expected columns
            with db_manager.get_connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE code_files (
                        id VARCHAR PRIMARY KEY,
                        path VARCHAR,
                        content TEXT,
                        embedding DOUBLE[384],
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

            # Run migration (should be no-op)
            db_manager.migrate_schema("code_files")

            # Verify all columns still exist and no duplicates were created
            with db_manager.get_connection() as conn:
                result = conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'code_files'"
                ).fetchall()

                column_names = [row[0] for row in result]
                expected_columns = [
                    "id",
                    "path",
                    "content",
                    "embedding",
                    "last_modified",
                    "file_mtime",
                    "file_type",
                    "language",
                    "size_bytes",
                    "line_count",
                    "category"
                ]

                assert len(column_names) == len(expected_columns), "Should not have duplicate columns"
                for col in expected_columns:
                    assert col in column_names, f"Column {col} should still exist"

            db_manager.cleanup()

    def test_migrate_schema_nonexistent_table(self):
        """Test migration on a nonexistent table (should raise error)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.duckdb"
            db_manager = DatabaseManager(db_path)

            # Try to migrate a table that doesn't exist
            with pytest.raises(Exception):
                db_manager.migrate_schema("nonexistent_table")

            db_manager.cleanup()

    def test_migrate_schema_preserves_data(self):
        """Test that migration preserves existing data in the table."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.duckdb"
            db_manager = DatabaseManager(db_path)

            # Create table and insert some test data
            with db_manager.get_connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE code_files (
                        id VARCHAR PRIMARY KEY,
                        path VARCHAR,
                        content TEXT,
                        embedding DOUBLE[384]
                    )
                """
                )
                # Insert test data
                test_embedding = [0.1] * 384
                conn.execute(
                    "INSERT INTO code_files (id, path, content, embedding) VALUES (?, ?, ?, ?)",
                    ("test_id", "/test/path.py", "print('hello')", test_embedding)
                )

            # Run migration
            db_manager.migrate_schema("code_files")

            # Verify data is preserved and new columns are null
            with db_manager.get_connection() as conn:
                result = conn.execute(
                    "SELECT id, path, content, file_type FROM code_files WHERE id = 'test_id'"
                ).fetchone()
                assert result is not None, "Data should be preserved after migration"
                assert result[0] == "test_id"
                assert result[1] == "/test/path.py"
                assert result[2] == "print('hello')"
                assert result[3] is None  # New column should be null

            db_manager.cleanup()
