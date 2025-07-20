#!/usr/bin/env python3
"""
Cleaned up unit tests for the code_index module.

This test suite covers the core functionality that's actually worth testing
without excessive mocking. Tests focus on real behavior and edge cases.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

# Import the modules to test
from code_index import compute_id, init_db, scan_repo
from config import config

# For backward compatibility in tests
CODE_EXTENSIONS = config.file_processing.CODE_EXTENSIONS
DIMENSIONS = config.embedding.DIMENSIONS
TABLE_NAME = config.database.TABLE_NAME


class TestComputeId:
    """Test the compute_id function for generating unique file identifiers."""

    def test_compute_id_deterministic(self):
        """Test that compute_id produces the same hash for identical input."""
        text = "def hello():\n    return 'world'"
        id1 = compute_id(text)
        id2 = compute_id(text)
        assert id1 == id2

    def test_compute_id_different_inputs(self):
        """Test that different inputs produce different hashes."""
        text1 = "def hello():\n    return 'world'"
        text2 = "def goodbye():\n    return 'cruel world'"
        id1 = compute_id(text1)
        id2 = compute_id(text2)
        assert id1 != id2

    def test_compute_id_unicode_handling(self):
        """Test that compute_id handles unicode characters properly."""
        text_with_unicode = "# This is a comment with émojis 🚀 and Chinese 中文"
        id1 = compute_id(text_with_unicode)
        id2 = compute_id(text_with_unicode)
        assert id1 == id2
        assert isinstance(id1, str)
        assert len(id1) == 64  # SHA-256 produces 64 character hex string


class TestInitDb:
    """Test database initialization functionality."""

    def test_init_db_creates_table(self):
        """Test that init_db creates the code_files table with correct schema."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Initialize database
            db_manager = init_db(tmp_path)

            # Check that table exists with correct schema
            result = db_manager.execute_with_retry(
                "SELECT table_name FROM information_schema.tables WHERE table_name = ?",
                (TABLE_NAME,),
            )
            assert len(result) > 0

            # Check column structure
            columns = db_manager.execute_with_retry(
                f"SELECT column_name FROM information_schema.columns " f"WHERE table_name = '{TABLE_NAME}'"
            )
            column_names = [col[0] for col in columns]
            expected_columns = ["id", "path", "content", "embedding"]
            assert all(col in column_names for col in expected_columns)

            db_manager.close()


class TestScanRepo:
    """Test repository scanning and file filtering."""

    def setup_method(self):
        """Set up a temporary repository for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)

        # Initialize git repository
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

        # Create test files
        (self.repo_path / "main.py").write_text("print('hello')")
        (self.repo_path / "utils.py").write_text("def helper(): pass")
        (self.repo_path / "config.json").write_text('{"key": "value"}')
        (self.repo_path / "README.md").write_text("# Project")
        (self.repo_path / "large_file.py").write_text("x = '" + "a" * 1000000 + "'")  # >1MB

        # Create subdirectory
        subdir = self.repo_path / "subdir"
        subdir.mkdir()
        (subdir / "utils.py").write_text("def util_func(): pass")

        # Create .gitignore
        (self.repo_path / ".gitignore").write_text("*.log\n__pycache__/\n")

        # Create ignored files
        (self.repo_path / "debug.log").write_text("log content")
        pycache_dir = self.repo_path / "__pycache__"
        pycache_dir.mkdir()
        (pycache_dir / "main.cpython-39.pyc").write_text("bytecode")

        # Add files to git
        subprocess.run(
            [
                "git",
                "add",
                "main.py",
                "utils.py",
                "config.json",
                "subdir/utils.py",
                ".gitignore",
            ],
            cwd=self.repo_path,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=self.repo_path,
            capture_output=True,
        )

    def teardown_method(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_scan_repo_finds_all_tracked_files(self):
        """Test that scan_repo finds all Git-tracked files regardless of extension."""
        files = scan_repo(self.repo_path, 1024 * 1024)  # 1MB limit
        file_names = [f.name for f in files]

        # Should find all Git-tracked files
        assert "main.py" in file_names
        assert "utils.py" in file_names
        assert "config.json" in file_names

        # Should now find .md files too (no extension filtering)
        assert "README.md" in file_names

    def test_scan_repo_respects_gitignore(self):
        """Test that scan_repo respects .gitignore rules."""
        files = scan_repo(self.repo_path, 1024 * 1024)
        file_names = [f.name for f in files]

        # Should not find ignored files
        assert "debug.log" not in file_names
        assert "main.cpython-39.pyc" not in file_names

    def test_scan_repo_respects_size_limit(self):
        """Test that scan_repo respects the max_bytes parameter."""
        files = scan_repo(self.repo_path, 1000)  # 1KB limit
        file_names = [f.name for f in files]

        # Should find small files
        assert "main.py" in file_names
        assert "utils.py" in file_names

        # Should not find large files
        assert "large_file.py" not in file_names

    def test_scan_repo_handles_subdirectories(self):
        """Test that scan_repo recursively finds files in subdirectories."""
        files = scan_repo(self.repo_path, 1024 * 1024)
        file_paths = [str(f) for f in files]

        # Should find files in subdirectories
        assert any("subdir" in path and "utils.py" in path for path in file_paths)

    def test_scan_repo_handles_non_git_directory(self):
        """Test that scan_repo falls back gracefully for non-git directories."""
        # Create a non-git directory
        non_git_dir = tempfile.mkdtemp()
        non_git_path = Path(non_git_dir)
        (non_git_path / "test.py").write_text("print('test')")

        try:
            files = scan_repo(non_git_path, 1024 * 1024)
            file_names = [f.name for f in files]

            # Should still find the file using fallback method
            assert "test.py" in file_names

        finally:
            shutil.rmtree(non_git_dir)


class TestCodeExtensions:
    """Test that CODE_EXTENSIONS contains expected file types."""

    def test_code_extensions_includes_common_types(self):
        """Test that CODE_EXTENSIONS includes common programming languages."""
        expected_extensions = {
            ".py",
            ".js",
            ".ts",
            ".java",
            ".go",
            ".rs",
            ".c",
            ".cpp",
            ".h",
            ".cs",
            ".rb",
            ".php",
            ".swift",
            ".kt",
            ".json",
            ".yaml",
            ".yml",
        }

        for ext in expected_extensions:
            assert ext in CODE_EXTENSIONS, f"Missing extension: {ext}"

    def test_code_extensions_excludes_common_non_code(self):
        """Test that CODE_EXTENSIONS excludes common non-code file types."""
        non_code_extensions = {
            ".txt",
            ".md",
            ".pdf",
            ".doc",
            ".docx",
            ".png",
            ".jpg",
            ".gif",
            ".mp4",
            ".mp3",
            ".zip",
            ".tar",
            ".gz",
        }

        for ext in non_code_extensions:
            assert ext not in CODE_EXTENSIONS, f"Should not include: {ext}"


class TestDimensions:
    """Test that DIMENSIONS constant is correct."""

    def test_dimensions_is_384(self):
        """Test that DIMENSIONS matches the all-MiniLM-L6-v2 model."""
        assert DIMENSIONS == 384
        assert isinstance(DIMENSIONS, int)


if __name__ == "__main__":
    pytest.main([__file__])
