#!/usr/bin/env python3
"""
Cleaned up server tests focusing on real API behavior.

These tests avoid over-mocking and focus on actual API responses and behavior.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Import the server app
from server import app


class TestServerBasicFunctionality:
    """Test basic server functionality without excessive mocking."""

    def setup_method(self):
        """Set up test environment."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)

        # Create a simple test repository
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
        (self.repo_path / "test.py").write_text("def hello(): return 'world'")
        (self.repo_path / "utils.py").write_text("def helper(): pass")

        # Add to git
        subprocess.run(["git", "add", "."], cwd=self.repo_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "initial"], cwd=self.repo_path, capture_output=True
        )

        # Create test client
        self.client = TestClient(app)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_status_endpoint_returns_valid_response(self):
        """Test that /status endpoint returns a valid response."""
        response = self.client.get("/status")
        assert response.status_code == 200

        data = response.json()
        required_fields = [
            "files_indexed",
            "database_size_mb",
            "search_index_ready",
            "last_updated",
            "embedding_dimensions",
            "model_name",
        ]

        for field in required_fields:
            assert field in data, f"Missing field: {field}"

        # Check data types
        assert isinstance(data["files_indexed"], int)
        assert isinstance(data["database_size_mb"], float)
        assert isinstance(data["search_index_ready"], bool)
        assert isinstance(data["embedding_dimensions"], int)
        assert isinstance(data["model_name"], str)

        # Check reasonable values
        assert data["files_indexed"] >= 0
        assert data["database_size_mb"] >= 0.0
        assert data["embedding_dimensions"] == 384  # all-MiniLM-L6-v2
        assert data["model_name"] == "all-MiniLM-L6-v2"

    def test_index_endpoint_accepts_valid_request(self):
        """Test that /index endpoint accepts valid requests."""
        response = self.client.post(
            "/index", json={"repo": str(self.repo_path), "max_mb": 1.0}
        )

        # Should return 200 and valid response structure
        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "files" in data
        assert data["status"] == "indexed"
        assert isinstance(data["files"], int)
        assert data["files"] >= 0

    def test_index_endpoint_handles_invalid_path(self):
        """Test that /index endpoint handles invalid repository paths."""
        response = self.client.post(
            "/index", json={"repo": "/nonexistent/path", "max_mb": 1.0}
        )

        # Should return 500 for invalid path (server error)
        assert response.status_code == 500

    def test_search_endpoint_with_empty_index(self):
        """Test that /search endpoint handles empty index gracefully."""
        response = self.client.get("/search", params={"query": "test query", "k": 5})

        # Should return 200 with results (may have results from previous tests)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # Just check the response structure is correct
        assert len(data) >= 0

    def test_search_endpoint_parameter_validation(self):
        """Test that /search endpoint validates parameters."""
        # Test with missing query
        response = self.client.get("/search")
        assert response.status_code == 422  # Validation error

        # Test with valid parameters
        response = self.client.get("/search", params={"query": "test", "k": 3})
        assert response.status_code == 200

        # Test with default k value
        response = self.client.get("/search", params={"query": "test"})
        assert response.status_code == 200

    def test_search_response_format(self):
        """Test that search responses have correct format."""
        response = self.client.get("/search", params={"query": "test", "k": 5})

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

        # Each result should have the expected structure
        for result in data:
            assert "path" in result
            assert "snippet" in result
            assert "distance" in result
            assert isinstance(result["path"], str)
            assert isinstance(result["snippet"], str)
            assert isinstance(result["distance"], (int, float))


class TestServerConfiguration:
    """Test server configuration and constants."""

    def test_app_metadata(self):
        """Test that the app has correct metadata."""
        assert app.title == "Code Index MCP"
        assert app.description == "Semantic code search and indexing API"
        assert app.version == "0.1.9"

    def test_client_creation(self):
        """Test that we can create a test client."""
        client = TestClient(app)
        assert client is not None

    def test_app_routes_exist(self):
        """Test that expected routes are registered."""
        # Test that routes exist (don't necessarily need to work)
        routes = [route.path for route in app.routes]

        expected_routes = ["/index", "/search", "/status"]
        for route in expected_routes:
            assert route in routes, f"Missing route: {route}"


if __name__ == "__main__":
    pytest.main([__file__])
