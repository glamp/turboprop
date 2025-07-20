#!/usr/bin/env python3
"""
Unit tests for new features added to turboprop.

These tests focus on real functionality rather than over-mocking.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from code_index import (
    TABLE_NAME,
    DebouncedHandler,
    init_db,
    print_indexed_files_tree,
    reindex_all,
    remove_orphaned_files,
)


class TestPrintIndexedFilesTree:
    """Test the source tree printing functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)

        # Create a realistic file structure
        (self.repo_path / "src").mkdir()
        (self.repo_path / "src" / "main.py").write_text("print('hello')")
        (self.repo_path / "src" / "utils.py").write_text("def helper(): pass")

        (self.repo_path / "tests").mkdir()
        (self.repo_path / "tests" / "test_main.py").write_text("def test_main(): pass")

        (self.repo_path / "config.json").write_text('{"key": "value"}')

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_print_indexed_files_tree_with_files(self, capsys):
        """Test tree printing with actual files."""
        files = [
            self.repo_path / "src" / "main.py",
            self.repo_path / "src" / "utils.py",
            self.repo_path / "tests" / "test_main.py",
            self.repo_path / "config.json",
        ]

        print_indexed_files_tree(files, self.repo_path)

        captured = capsys.readouterr()
        output = captured.out

        # Check that the output contains expected elements
        assert "📂 Indexed files" in output
        assert "src/" in output
        assert "tests/" in output
        assert "main.py" in output
        assert "config.json" in output
        assert "Total: 4 files indexed" in output

    def test_print_indexed_files_tree_empty(self, capsys):
        """Test tree printing with no files."""
        print_indexed_files_tree([], self.repo_path)

        captured = capsys.readouterr()
        output = captured.out

        assert "📁 No files indexed" in output

    def test_print_indexed_files_tree_sorts_files(self, capsys):
        """Test that files are sorted alphabetically."""
        files = [
            self.repo_path / "zebra.py",
            self.repo_path / "alpha.py",
            self.repo_path / "beta.py",
        ]

        print_indexed_files_tree(files, self.repo_path)

        captured = capsys.readouterr()
        output = captured.out

        # Check that alpha comes before beta comes before zebra
        alpha_pos = output.find("alpha.py")
        beta_pos = output.find("beta.py")
        zebra_pos = output.find("zebra.py")

        assert alpha_pos < beta_pos < zebra_pos


class TestRemoveOrphanedFiles:
    """Test orphaned file removal functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.duckdb"

        # Initialize database
        with patch("code_index.DB_PATH", str(self.db_path)):
            self.db_manager = init_db()

        # Ensure table exists with full schema
        self.db_manager.execute_with_retry(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id VARCHAR PRIMARY KEY,
                path VARCHAR,
                content TEXT,
                embedding DOUBLE[384],
                last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_mtime TIMESTAMP,
                file_type VARCHAR,
                language VARCHAR,
                size_bytes INTEGER,
                line_count INTEGER
            )
        """
        )

        # Clear any existing data
        self.db_manager.execute_with_retry(f"DELETE FROM {TABLE_NAME}")

        # Insert some test data with full schema
        test_data = [
            ("id1", "/path/to/existing.py", "code1", [0.1] * 384, None, None, ".py", "python", 100, 10),
            ("id2", "/path/to/deleted.py", "code2", [0.2] * 384, None, None, ".py", "python", 200, 20),
            ("id3", "/path/to/another_existing.py", "code3", [0.3] * 384, None, None, ".py", "python", 300, 30),
        ]

        for data in test_data:
            self.db_manager.execute_with_retry(f"INSERT INTO {TABLE_NAME} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", data)

    def teardown_method(self):
        """Clean up test environment."""
        self.db_manager.close()
        if self.db_path.exists():
            self.db_path.unlink()
        shutil.rmtree(self.temp_dir)

    def test_remove_orphaned_files_removes_correct_files(self):
        """Test that orphaned files are removed correctly."""
        # Current files (missing /path/to/deleted.py)
        current_files = [
            Path("/path/to/existing.py"),
            Path("/path/to/another_existing.py"),
        ]

        # Remove orphaned files
        removed_count = remove_orphaned_files(self.db_manager, current_files)

        # Should have removed 1 file
        assert removed_count == 1

        # Check database state
        result = self.db_manager.execute_with_retry(f"SELECT path FROM {TABLE_NAME} ORDER BY path")
        paths = [row[0] for row in result]

        assert "/path/to/existing.py" in paths
        assert "/path/to/another_existing.py" in paths
        assert "/path/to/deleted.py" not in paths

    def test_remove_orphaned_files_no_orphans(self):
        """Test behavior when no orphaned files exist."""
        # All files still exist
        current_files = [
            Path("/path/to/existing.py"),
            Path("/path/to/deleted.py"),
            Path("/path/to/another_existing.py"),
        ]

        removed_count = remove_orphaned_files(self.db_manager, current_files)

        # Should have removed 0 files
        assert removed_count == 0

        # All files should still be in database
        result = self.db_manager.execute_with_retry(f"SELECT COUNT(*) FROM {TABLE_NAME}")
        count = result[0][0]
        assert count == 3


class TestEnhancedReindexAll:
    """Test the enhanced reindex_all function with orphan removal."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)

        # Create a git repository
        subprocess.run(["git", "init"], cwd=self.repo_path, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=self.repo_path,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=self.repo_path,
            capture_output=True,
        )

        # Create some test files
        (self.repo_path / "main.py").write_text("print('hello')")
        (self.repo_path / "utils.py").write_text("def helper(): pass")

        # Add to git
        subprocess.run(["git", "add", "."], cwd=self.repo_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=self.repo_path, capture_output=True)

        # Initialize database
        self.db_manager = init_db(self.repo_path)

        # Ensure table exists with full schema
        self.db_manager.execute_with_retry(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id VARCHAR PRIMARY KEY,
                path VARCHAR,
                content TEXT,
                embedding DOUBLE[384],
                last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_mtime TIMESTAMP,
                file_type VARCHAR,
                language VARCHAR,
                size_bytes INTEGER,
                line_count INTEGER
            )
        """
        )

        # Mock embedder
        import numpy as np

        self.mock_embedder = Mock()
        self.mock_embedder.encode.return_value = np.array([0.1] * 384)

    def teardown_method(self):
        """Clean up test environment."""
        self.db_manager.close()
        shutil.rmtree(self.temp_dir)

    def test_reindex_all_removes_orphaned_files(self):
        """Test that reindex_all removes orphaned files."""
        # First, manually add an orphaned entry
        self.db_manager.execute_with_retry(
            f"INSERT INTO {TABLE_NAME} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "orphan_id",
                str(self.repo_path / "deleted.py"),
                "old code",
                [0.5] * 384,
                None,
                None,
                ".py",
                "python",
                100,
                10,
            ),
        )

        # Run reindex_all
        total_files, processed_files, elapsed = reindex_all(
            self.repo_path, 1024 * 1024, self.db_manager, self.mock_embedder
        )

        # Should have found 2 files in repo
        assert total_files == 2

        # Check that orphaned file was removed
        result = self.db_manager.execute_with_retry(f"SELECT path FROM {TABLE_NAME}")
        paths = [row[0] for row in result]

        # The orphaned file should be removed
        assert not any("deleted.py" in path for path in paths)
        # The current files should still be there
        assert any("main.py" in path for path in paths)
        assert any("utils.py" in path for path in paths)


class TestDebouncedHandlerShouldIndexFile:
    """Test the _should_index_file method with git integration."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)

        # Create a git repository
        subprocess.run(["git", "init"], cwd=self.repo_path, capture_output=True)

        # Create .gitignore
        (self.repo_path / ".gitignore").write_text("*.log\n__pycache__/\n.env\n")

        # Create test files
        (self.repo_path / "main.py").write_text("print('hello')")
        (self.repo_path / "test.log").write_text("log content")
        (self.repo_path / "config.json").write_text('{"key": "value"}')
        (self.repo_path / ".env").write_text("SECRET=123")

        # Add to git
        subprocess.run(
            ["git", "add", "main.py", "config.json", ".gitignore"],
            cwd=self.repo_path,
            capture_output=True,
        )

        # Create handler
        self.handler = DebouncedHandler(self.repo_path, 1024 * 1024, Mock(), Mock(), 5.0)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_should_index_file_respects_gitignore(self):
        """Test that _should_index_file respects .gitignore."""
        # Should index regular Python file
        assert self.handler._should_index_file(self.repo_path / "main.py")

        # Should index JSON file
        assert self.handler._should_index_file(self.repo_path / "config.json")

        # Should NOT index .log file (in .gitignore)
        assert not self.handler._should_index_file(self.repo_path / "test.log")

        # Should NOT index .env file (in .gitignore)
        assert not self.handler._should_index_file(self.repo_path / ".env")

    def test_should_index_file_no_extension_filtering(self):
        """Test that _should_index_file indexes all Git-tracked files regardless of
        extension."""
        # Create files with different extensions
        (self.repo_path / "document.txt").write_text("text content")
        (self.repo_path / "image.png").write_text("binary content")
        (self.repo_path / "script.py").write_text("python code")

        # Should index ALL files (no extension filtering)
        assert self.handler._should_index_file(self.repo_path / "document.txt")
        assert self.handler._should_index_file(self.repo_path / "image.png")
        assert self.handler._should_index_file(self.repo_path / "script.py")

    def test_should_index_file_checks_file_size(self):
        """Test that _should_index_file respects file size limits."""
        # Create handler with small size limit
        small_handler = DebouncedHandler(self.repo_path, 100, Mock(), Mock(), 5.0)  # 100 bytes limit

        # Create small file
        small_file = self.repo_path / "small.py"
        small_file.write_text("x = 1")

        # Create large file
        large_file = self.repo_path / "large.py"
        large_file.write_text("x = '" + "a" * 1000 + "'")

        # Should index small file
        assert small_handler._should_index_file(small_file)

        # Should NOT index large file
        assert not small_handler._should_index_file(large_file)


if __name__ == "__main__":
    pytest.main([__file__])
