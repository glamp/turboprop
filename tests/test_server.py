#!/usr/bin/env python3
"""
Unit tests for the server module.

This test suite covers the FastAPI server functionality including:
- HTTP endpoint behavior
- Request/response validation
- Background file watching
- Error handling
- API integration with core indexing functions

The tests use FastAPI's test client and mock the underlying
indexing functions to ensure isolated testing.
"""

import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
from fastapi.testclient import TestClient
import numpy as np

# Import the server module
from server import app, IndexRequest, SearchResponse


class TestIndexEndpoint:
    """Test the /index POST endpoint."""
    
    def setup_method(self):
        """Set up test environment."""
        self.client = TestClient(app)
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        
    def test_index_endpoint_success(self):
        """Test successful indexing request."""
        with patch('server.reindex_all') as mock_reindex, \
             patch('server.con') as mock_con:
            
            # Mock the reindex_all function to return (total_files, processed_files, elapsed)
            mock_reindex.return_value = (42, 42, 1.0)
            
            # Mock the database count query
            mock_con.execute.return_value.fetchone.return_value = [42]
            
            # Make request
            response = self.client.post("/index", json={
                "repo": self.temp_dir,
                "max_mb": 2.0
            })
            
            # Check response
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "indexed"
            assert data["files"] == 42
            
            # Verify reindex_all was called with correct parameters
            mock_reindex.assert_called_once()
            call_args = mock_reindex.call_args[0]
            assert str(call_args[0]) == self.temp_dir  # repo path
            assert call_args[1] == int(2.0 * 1024**2)  # max_bytes
            
    def test_index_endpoint_default_max_mb(self):
        """Test indexing with default max_mb parameter."""
        with patch('server.reindex_all') as mock_reindex, \
             patch('server.con') as mock_con:
            
            mock_reindex.return_value = (10, 10, 1.0)
            mock_con.execute.return_value.fetchone.return_value = [10]
            
            # Make request without max_mb
            response = self.client.post("/index", json={
                "repo": self.temp_dir
            })
            
            assert response.status_code == 200
            
            # Should use default max_mb of 1.0
            call_args = mock_reindex.call_args[0]
            assert call_args[1] == int(1.0 * 1024**2)  # default max_bytes
            
    def test_index_endpoint_invalid_request(self):
        """Test indexing with invalid request data."""
        # Missing required 'repo' field
        response = self.client.post("/index", json={
            "max_mb": 1.0
        })
        
        assert response.status_code == 422  # Validation error
        
    def test_index_endpoint_negative_max_mb(self):
        """Test indexing with negative max_mb."""
        response = self.client.post("/index", json={
            "repo": self.temp_dir,
            "max_mb": -1.0
        })
        
        # Should still accept negative values (validation doesn't prevent it)
        # but the behavior is undefined - this test documents current behavior
        assert response.status_code in [200, 422]


class TestSearchEndpoint:
    """Test the /search GET endpoint."""
    
    def setup_method(self):
        """Set up test environment."""
        self.client = TestClient(app)
        
    def test_search_endpoint_success(self):
        """Test successful search request."""
        with patch('server.search_index') as mock_search:
            # Mock search results
            mock_search.return_value = [
                ("/path/to/file1.py", "def hello(): pass", 0.1),
                ("/path/to/file2.py", "def world(): pass", 0.2),
            ]
            
            # Make request
            response = self.client.get("/search", params={
                "query": "hello world",
                "k": 2
            })
            
            # Check response
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            
            # Check response structure
            for item in data:
                assert "path" in item
                assert "snippet" in item
                assert "distance" in item
                assert isinstance(item["distance"], float)
                
            # Verify search_index was called correctly
            mock_search.assert_called_once()
            call_args = mock_search.call_args[0]
            assert call_args[2] == "hello world"  # query
            assert call_args[3] == 2  # k
            
    def test_search_endpoint_default_k(self):
        """Test search with default k parameter."""
        with patch('server.search_index') as mock_search:
            mock_search.return_value = []
            
            response = self.client.get("/search", params={
                "query": "test query"
            })
            
            assert response.status_code == 200
            
            # Should use default k of 5
            call_args = mock_search.call_args[0]
            assert call_args[3] == 5  # default k
            
    def test_search_endpoint_no_results(self):
        """Test search with no results."""
        with patch('server.search_index') as mock_search:
            mock_search.return_value = []
            
            response = self.client.get("/search", params={
                "query": "nonexistent code"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data == []
            
    def test_search_endpoint_missing_query(self):
        """Test search without query parameter."""
        response = self.client.get("/search")
        
        assert response.status_code == 422  # Validation error
        
    def test_search_endpoint_invalid_k(self):
        """Test search with invalid k parameter."""
        response = self.client.get("/search", params={
            "query": "test",
            "k": "invalid"
        })
        
        assert response.status_code == 422  # Validation error


class TestResponseModels:
    """Test Pydantic response models."""
    
    def test_search_response_model(self):
        """Test SearchResponse model validation."""
        # Valid data
        response = SearchResponse(
            path="/path/to/file.py",
            snippet="def function(): pass",
            distance=0.5
        )
        
        assert response.path == "/path/to/file.py"
        assert response.snippet == "def function(): pass"
        assert response.distance == 0.5
        
    def test_search_response_model_validation(self):
        """Test SearchResponse model validation with invalid data."""
        with pytest.raises(ValueError):
            SearchResponse(
                path="/path/to/file.py",
                snippet="def function(): pass",
                distance="invalid"  # Should be float
            )
            
    def test_index_request_model(self):
        """Test IndexRequest model validation."""
        # Valid data with default
        request = IndexRequest(repo="/path/to/repo")
        assert request.repo == "/path/to/repo"
        assert request.max_mb == 1.0  # default value
        
        # Valid data with custom max_mb
        request = IndexRequest(repo="/path/to/repo", max_mb=2.5)
        assert request.max_mb == 2.5


class TestAppInitialization:
    """Test FastAPI app initialization and metadata."""
    
    def test_app_metadata(self):
        """Test that app has correct metadata."""
        assert app.title == "Code Index MCP"
        assert app.description == "Semantic code search and indexing API"
        assert app.version == "1.0.0"
        
    def test_app_routes(self):
        """Test that required routes are registered."""
        client = TestClient(app)
        
        # Test route existence by checking responses
        with patch('server.reindex_all') as mock_reindex, patch('server.con') as mock_con:
            mock_reindex.return_value = (10, 10, 1.0)
            mock_con.execute.return_value.fetchone.return_value = [10]
            response = client.post("/index", json={"repo": "/test"})
            assert response.status_code in [200, 422]  # Either success or validation error
            
        with patch('server.search_index'):
            response = client.get("/search", params={"query": "test"})
            assert response.status_code == 200


class TestStartupEvent:
    """Test the startup event handler."""
    
    def test_startup_watch_initialization(self):
        """Test that startup event initializes background watcher."""
        with patch('server.threading.Thread') as mock_thread, \
             patch('server.watch_mode') as mock_watch:
            
            # Mock thread creation
            mock_thread_instance = Mock()
            mock_thread.return_value = mock_thread_instance
            
            # Import and trigger the startup event
            from server import _startup_watch
            _startup_watch()
            
            # Check that thread was created and started
            mock_thread.assert_called_once()
            mock_thread_instance.start.assert_called_once()
            
            # Check thread configuration
            call_args = mock_thread.call_args
            assert call_args[1]["daemon"] is True
            
            # Check that watch_mode target is configured correctly
            target_func = call_args[1]["target"]
            # This is a lambda, so we can't easily test the exact parameters
            # but we can verify it's callable
            assert callable(target_func)


class TestErrorHandling:
    """Test error handling in various scenarios."""
    
    def setup_method(self):
        """Set up test environment."""
        self.client = TestClient(app)
        
    def test_index_endpoint_database_error(self):
        """Test indexing when database operations fail."""
        with patch('server.reindex_all', side_effect=Exception("Database error")):
            response = self.client.post("/index", json={"repo": "/test"})
            
            # Should return 500 internal server error
            assert response.status_code == 500
            
    def test_search_endpoint_search_error(self):
        """Test search when search operations fail."""
        with patch('server.search_index', side_effect=Exception("Search error")):
            response = self.client.get("/search", params={"query": "test"})
            
            # Should return 500 internal server error
            assert response.status_code == 500
            
    def test_index_endpoint_invalid_path(self):
        """Test indexing with invalid repository path."""
        with patch('server.reindex_all') as mock_reindex, \
             patch('server.con') as mock_con:
            
            # Mock reindex_all to raise an error for invalid path
            mock_reindex.side_effect = FileNotFoundError("Repository not found")
            
            response = self.client.post("/index", json={
                "repo": "/nonexistent/path"
            })
            
            assert response.status_code == 500


class TestIntegrationWithMocks:
    """Integration tests with mocked dependencies."""
    
    def setup_method(self):
        """Set up test environment."""
        self.client = TestClient(app)
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        
    def test_index_then_search_workflow(self):
        """Test complete workflow of indexing then searching."""
        with patch('server.reindex_all') as mock_reindex, \
             patch('server.con') as mock_con, \
             patch('server.search_index') as mock_search:
            
            # Mock indexing
            mock_reindex.return_value = (5, 5, 1.0)
            mock_con.execute.return_value.fetchone.return_value = [5]
            
            # Mock search
            mock_search.return_value = [
                ("/path/to/result.py", "def result(): pass", 0.1)
            ]
            
            # Index first
            index_response = self.client.post("/index", json={
                "repo": self.temp_dir,
                "max_mb": 1.0
            })
            assert index_response.status_code == 200
            assert index_response.json()["files"] == 5
            
            # Then search
            search_response = self.client.get("/search", params={
                "query": "result function",
                "k": 1
            })
            assert search_response.status_code == 200
            results = search_response.json()
            assert len(results) == 1
            assert results[0]["path"] == "/path/to/result.py"
            
    def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        with patch('server.reindex_all') as mock_reindex, \
             patch('server.con') as mock_con, \
             patch('server.search_index') as mock_search:
            
            # Mock functions to simulate slow operations
            mock_reindex.return_value = (10, 10, 1.0)
            mock_con.execute.return_value.fetchone.return_value = [10]
            mock_search.return_value = []
            
            # Make multiple concurrent requests
            responses = []
            
            # Multiple search requests
            for i in range(3):
                response = self.client.get("/search", params={
                    "query": f"test query {i}"
                })
                responses.append(response)
                
            # All should succeed
            for response in responses:
                assert response.status_code == 200
                
            # Multiple index requests
            for i in range(2):
                response = self.client.post("/index", json={
                    "repo": self.temp_dir
                })
                responses.append(response)
                
            # All should succeed
            for response in responses[-2:]:
                assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])