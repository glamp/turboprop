#!/usr/bin/env python3
"""
Tests for git integration and repository context extraction.

This module tests the git repository information extraction, project type detection,
and dependency parsing functionality.
"""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from git_integration import (
    GitRepository,
    ProjectDetector,
    RepositoryContext,
    RepositoryContextExtractor
)


class TestGitRepository(unittest.TestCase):
    """Test GitRepository class for git information extraction."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.git_repo = GitRepository(self.temp_dir)

    def test_is_git_repository_false_for_non_git_dir(self):
        """Test that non-git directory returns False."""
        result = self.git_repo.is_git_repository()
        self.assertFalse(result)

    @patch('subprocess.run')
    def test_get_current_branch_success(self, mock_run):
        """Test successful git branch extraction."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "main\n"
        mock_run.return_value.stderr = ""

        result = self.git_repo.get_current_branch()
        self.assertEqual(result, "main")

    @patch('subprocess.run')
    def test_get_current_branch_failure(self, mock_run):
        """Test git branch extraction failure."""
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "git", "fatal: not a git repository"
        )

        result = self.git_repo.get_current_branch()
        self.assertIsNone(result)

    @patch('subprocess.run')
    def test_get_current_commit_success(self, mock_run):
        """Test successful git commit hash extraction."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "abc123def456\n"
        mock_run.return_value.stderr = ""

        result = self.git_repo.get_current_commit()
        self.assertEqual(result, "abc123def456")

    @patch('subprocess.run')
    def test_get_remote_urls_success(self, mock_run):
        """Test successful git remote URL extraction."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = (
            "origin\tgit@github.com:user/repo.git (fetch)\n"
            "origin\tgit@github.com:user/repo.git (push)\n"
        )
        mock_run.return_value.stderr = ""

        result = self.git_repo.get_remote_urls()
        expected = {"origin": "git@github.com:user/repo.git"}
        self.assertEqual(result, expected)

    @patch('subprocess.run')
    def test_get_repository_root_success(self, mock_run):
        """Test successful git repository root detection."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "/path/to/repo\n"
        mock_run.return_value.stderr = ""

        result = self.git_repo.get_repository_root()
        self.assertEqual(result, Path("/path/to/repo"))


class TestProjectDetector(unittest.TestCase):
    """Test ProjectDetector class for project type and dependency detection."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.detector = ProjectDetector(self.temp_dir)

    def test_detect_python_project_with_pyproject_toml(self):
        """Test Python project detection with pyproject.toml."""
        # Create a pyproject.toml file
        pyproject_file = self.temp_dir / "pyproject.toml"
        pyproject_file.write_text("""
[project]
name = "test-project"
version = "0.1.0"
dependencies = ["requests>=2.25.0", "click>=8.0.0"]
""")

        result = self.detector.detect_project_type()
        self.assertEqual(result, "python")

    def test_detect_javascript_project_with_package_json(self):
        """Test JavaScript project detection with package.json."""
        # Create a package.json file
        package_file = self.temp_dir / "package.json"
        package_file.write_text("""
{
  "name": "test-project",
  "version": "1.0.0",
  "dependencies": {
    "express": "^4.17.1",
    "lodash": "^4.17.21"
  }
}
""")

        result = self.detector.detect_project_type()
        self.assertEqual(result, "javascript")

    def test_detect_rust_project_with_cargo_toml(self):
        """Test Rust project detection with Cargo.toml."""
        # Create a Cargo.toml file
        cargo_file = self.temp_dir / "Cargo.toml"
        cargo_file.write_text("""
[package]
name = "test-project"
version = "0.1.0"

[dependencies]
serde = "1.0"
tokio = "1.0"
""")

        result = self.detector.detect_project_type()
        self.assertEqual(result, "rust")

    def test_detect_mixed_project_returns_primary(self):
        """Test mixed project detection returns primary type."""
        # Create both Python and JavaScript files
        (self.temp_dir / "pyproject.toml").write_text("[project]\nname='test'")
        (self.temp_dir / "package.json").write_text('{"name": "test"}')

        result = self.detector.detect_project_type()
        # Should return the first detected type (order matters in implementation)
        self.assertIn(result, ["python", "javascript"])

    def test_extract_python_dependencies_from_pyproject_toml(self):
        """Test Python dependency extraction from pyproject.toml."""
        pyproject_file = self.temp_dir / "pyproject.toml"
        pyproject_file.write_text("""
[project]
name = "test-project"
dependencies = [
    "requests>=2.25.0",
    "click>=8.0.0",
    "numpy"
]
""")

        result = self.detector.extract_dependencies()
        expected = [
            {"name": "requests", "version": ">=2.25.0", "source": "pyproject.toml"},
            {"name": "click", "version": ">=8.0.0", "source": "pyproject.toml"},
            {"name": "numpy", "version": None, "source": "pyproject.toml"}
        ]
        self.assertEqual(result, expected)

    def test_extract_javascript_dependencies_from_package_json(self):
        """Test JavaScript dependency extraction from package.json."""
        package_file = self.temp_dir / "package.json"
        package_file.write_text("""
{
  "name": "test-project",
  "dependencies": {
    "express": "^4.17.1",
    "lodash": "^4.17.21"
  },
  "devDependencies": {
    "jest": "^27.0.0"
  }
}
""")

        result = self.detector.extract_dependencies()
        expected = [
            {
                "name": "express",
                "version": "^4.17.1",
                "source": "package.json",
                "type": "production"
            },
            {
                "name": "lodash",
                "version": "^4.17.21",
                "source": "package.json",
                "type": "production"
            },
            {"name": "jest", "version": "^27.0.0", "source": "package.json", "type": "development"}
        ]
        self.assertEqual(result, expected)

    def test_get_package_managers(self):
        """Test package manager detection."""
        # Create package manager files
        (self.temp_dir / "package.json").write_text('{"name": "test"}')
        (self.temp_dir / "package-lock.json").write_text('{}')
        (self.temp_dir / "pyproject.toml").write_text("[project]\nname='test'")

        result = self.detector.get_package_managers()
        self.assertIn("npm", result)
        self.assertIn("pip", result)


class TestRepositoryContext(unittest.TestCase):
    """Test RepositoryContext data class."""

    def test_repository_context_creation(self):
        """Test RepositoryContext creation with all fields."""
        context = RepositoryContext(
            repository_id="abc123",
            repository_path="/path/to/repo",
            git_branch="main",
            git_commit="def456",
            git_remote_url="git@github.com:user/repo.git",
            project_type="python",
            dependencies=[{"name": "requests", "version": "2.25.0"}],
            package_managers=["pip"]
        )

        self.assertEqual(context.repository_id, "abc123")
        self.assertEqual(context.project_type, "python")
        self.assertEqual(len(context.dependencies), 1)

    def test_compute_repository_id(self):
        """Test repository ID computation from path."""
        repo_path = "/path/to/repo"
        context = RepositoryContext.compute_repository_id(repo_path)

        # Should be a 64-character hex string (SHA-256)
        self.assertEqual(len(context), 64)
        self.assertTrue(all(c in '0123456789abcdef' for c in context))

    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        context = RepositoryContext(
            repository_id="abc123",
            repository_path="/path/to/repo",
            git_branch="main",
            project_type="python"
        )

        result = context.to_dict()
        self.assertIsInstance(result, dict)
        self.assertEqual(result["repository_id"], "abc123")
        self.assertEqual(result["project_type"], "python")


class TestRepositoryContextExtractor(unittest.TestCase):
    """Test RepositoryContextExtractor integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.extractor = RepositoryContextExtractor()

    @patch('git_integration.GitRepository')
    @patch('git_integration.ProjectDetector')
    def test_extract_context_success(self, mock_detector, mock_git_repo):
        """Test successful repository context extraction."""
        # Mock git repository
        mock_git_instance = Mock()
        mock_git_instance.is_git_repository.return_value = True
        mock_git_instance.get_repository_root.return_value = self.temp_dir
        mock_git_instance.get_current_branch.return_value = "main"
        mock_git_instance.get_current_commit.return_value = "abc123"
        mock_git_instance.get_remote_urls.return_value = {"origin": "git@github.com:user/repo.git"}
        mock_git_repo.return_value = mock_git_instance

        # Mock project detector
        mock_detector_instance = Mock()
        mock_detector_instance.detect_project_type.return_value = "python"
        mock_detector_instance.extract_dependencies.return_value = [
            {"name": "requests", "version": "2.25.0"}
        ]
        mock_detector_instance.get_package_managers.return_value = ["pip"]
        mock_detector.return_value = mock_detector_instance

        result = self.extractor.extract_context(self.temp_dir)

        self.assertIsNotNone(result)
        self.assertEqual(result.project_type, "python")
        self.assertEqual(result.git_branch, "main")
        self.assertEqual(result.git_commit, "abc123")

    def test_extract_context_non_git_directory(self):
        """Test context extraction for non-git directory."""
        with patch('git_integration.ProjectDetector') as mock_detector:
            mock_detector_instance = Mock()
            mock_detector_instance.detect_project_type.return_value = "python"
            mock_detector_instance.extract_dependencies.return_value = []
            mock_detector_instance.get_package_managers.return_value = ["pip"]
            mock_detector.return_value = mock_detector_instance

            result = self.extractor.extract_context(self.temp_dir)

            self.assertIsNotNone(result)
            self.assertEqual(result.project_type, "python")
            self.assertIsNone(result.git_branch)
            self.assertIsNone(result.git_commit)


if __name__ == "__main__":
    unittest.main()
