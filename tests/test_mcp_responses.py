#!/usr/bin/env python3
"""
Tests for MCP structured response types and enhanced search functionality.

This test suite verifies the structured response data types (SearchResponse,
IndexResponse, StatusResponse) and their JSON serialization, as well as the
enhanced search operations with metadata and clustering.
"""

import json
from unittest.mock import Mock

import pytest

from turboprop.mcp_response_types import (
    IndexResponse,
    QueryAnalysis,
    ResultCluster,
    SearchResponse,
    StatusResponse,
    create_search_response_from_results,
)
from turboprop.search_operations import (
    _analyze_search_query,
    cluster_results_by_confidence,
    cluster_results_by_directory,
    cluster_results_by_language,
    generate_cross_references,
)
from turboprop.search_result_types import CodeSearchResult, CodeSnippet


class TestQueryAnalysis:
    """Test the QueryAnalysis data class and analysis functions."""

    def test_create_query_analysis(self):
        """Test creating a QueryAnalysis instance."""
        analysis = QueryAnalysis(
            original_query="test query",
            suggested_refinements=["try this", "or that"],
            query_complexity="medium",
            estimated_result_count=5,
            search_hints=["hint 1", "hint 2"],
        )

        assert analysis.original_query == "test query"
        assert len(analysis.suggested_refinements) == 2
        assert analysis.query_complexity == "medium"
        assert analysis.estimated_result_count == 5
        assert len(analysis.search_hints) == 2

    def test_query_analysis_to_dict(self):
        """Test converting QueryAnalysis to dictionary."""
        analysis = QueryAnalysis(original_query="test query", query_complexity="low")

        result_dict = analysis.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["original_query"] == "test query"
        assert result_dict["query_complexity"] == "low"
        assert "suggested_refinements" in result_dict
        assert "search_hints" in result_dict

    def test_analyze_simple_query(self):
        """Test analyzing a simple single-word query."""
        results = []  # No results
        analysis = _analyze_search_query("authentication", results)

        assert analysis.original_query == "authentication"
        assert analysis.query_complexity == "low"
        assert any("Single-word queries may be too broad" in hint for hint in analysis.search_hints)
        assert len(analysis.suggested_refinements) > 0
        assert any("No results found" in hint for hint in analysis.search_hints)

    def test_analyze_complex_query(self):
        """Test analyzing a complex multi-word query."""
        results = []
        analysis = _analyze_search_query("JWT token authentication implementation", results)

        assert analysis.query_complexity == "high"
        assert "Complex queries are good for specific searches" in analysis.search_hints


class TestResultCluster:
    """Test the ResultCluster data class and clustering functions."""

    def test_create_result_cluster(self):
        """Test creating a ResultCluster instance."""
        # Create mock search results
        snippet = CodeSnippet(text="def test():", start_line=1, end_line=1)
        result1 = CodeSearchResult("test1.py", snippet, 0.9)
        result2 = CodeSearchResult("test2.py", snippet, 0.8)

        cluster = ResultCluster(
            cluster_name="Python Files",
            cluster_type="language",
            results=[result1, result2],
            cluster_score=0.85,
            cluster_description="Python source files",
        )

        assert cluster.cluster_name == "Python Files"
        assert cluster.cluster_type == "language"
        assert len(cluster.results) == 2
        assert cluster.cluster_score == 0.85
        assert cluster.cluster_description == "Python source files"

    def test_result_cluster_to_dict(self):
        """Test converting ResultCluster to dictionary."""
        snippet = CodeSnippet(text="def test():", start_line=1, end_line=1)
        result = CodeSearchResult("test.py", snippet, 0.9)

        cluster = ResultCluster(
            cluster_name="Test Cluster", cluster_type="language", results=[result], cluster_score=0.9
        )

        result_dict = cluster.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["cluster_name"] == "Test Cluster"
        assert result_dict["cluster_type"] == "language"
        assert result_dict["result_count"] == 1
        assert "results" in result_dict
        assert isinstance(result_dict["results"], list)

    def test_cluster_results_by_language(self):
        """Test clustering search results by programming language."""
        # Create results with different languages
        snippet = CodeSnippet(text="code", start_line=1, end_line=1)

        py_result1 = CodeSearchResult("test1.py", snippet, 0.9, {"language": "Python"})
        py_result2 = CodeSearchResult("test2.py", snippet, 0.8, {"language": "Python"})
        js_result = CodeSearchResult("test.js", snippet, 0.7, {"language": "JavaScript"})

        results = [py_result1, py_result2, js_result]
        clusters = cluster_results_by_language(results)

        # Should have one cluster for Python (2 results), none for JS (only 1 result)
        assert len(clusters) == 1
        assert clusters[0].cluster_name == "Python Files"
        assert clusters[0].cluster_type == "language"
        assert len(clusters[0].results) == 2

    def test_cluster_results_by_directory(self):
        """Test clustering search results by directory."""
        snippet = CodeSnippet(text="code", start_line=1, end_line=1)

        result1 = CodeSearchResult("/project/src/file1.py", snippet, 0.9)
        result2 = CodeSearchResult("/project/src/file2.py", snippet, 0.8)
        result3 = CodeSearchResult("/project/tests/test.py", snippet, 0.7)

        results = [result1, result2, result3]
        clusters = cluster_results_by_directory(results)

        # Should have one cluster for src/ directory (2 results)
        assert len(clusters) == 1
        assert clusters[0].cluster_name == "src/ directory"
        assert clusters[0].cluster_type == "directory"
        assert len(clusters[0].results) == 2

    def test_cluster_results_by_confidence(self):
        """Test clustering search results by confidence level."""
        snippet = CodeSnippet(text="code", start_line=1, end_line=1)

        high_result = CodeSearchResult("test1.py", snippet, 0.95)  # High confidence
        medium_result = CodeSearchResult("test2.py", snippet, 0.75)  # Medium confidence
        low_result = CodeSearchResult("test3.py", snippet, 0.55)  # Low confidence

        results = [high_result, medium_result, low_result]
        clusters = cluster_results_by_confidence(results)

        # Should have 3 clusters (one for each confidence level)
        assert len(clusters) == 3

        # Check that clusters are ordered by confidence (high -> medium -> low)
        assert clusters[0].cluster_name == "High Confidence"
        assert clusters[1].cluster_name == "Medium Confidence"
        assert clusters[2].cluster_name == "Low Confidence"


class TestSearchResponse:
    """Test the SearchResponse data class and JSON serialization."""

    def test_create_search_response(self):
        """Test creating a SearchResponse instance."""
        snippet = CodeSnippet(text="def test():", start_line=1, end_line=1)
        result = CodeSearchResult("test.py", snippet, 0.9, {"language": "Python"})

        response = SearchResponse(query="test query", results=[result], execution_time=0.5)

        assert response.query == "test query"
        assert len(response.results) == 1
        assert response.total_results == 1
        assert response.execution_time == 0.5
        assert response.version == "1.0"
        assert response.timestamp is not None

        # Check computed fields
        assert "Python" in response.language_breakdown
        assert response.language_breakdown["Python"] == 1
        assert response.confidence_distribution["high"] == 1  # 0.9 is high confidence

    def test_search_response_to_json(self):
        """Test converting SearchResponse to JSON."""
        snippet = CodeSnippet(text="def test():", start_line=1, end_line=1)
        result = CodeSearchResult("test.py", snippet, 0.9)

        response = SearchResponse(query="test query", results=[result], execution_time=0.5)

        json_str = response.to_json()

        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert parsed["query"] == "test query"
        assert parsed["total_results"] == 1
        assert parsed["execution_time"] == 0.5
        assert "results" in parsed
        assert len(parsed["results"]) == 1

    def test_search_response_add_methods(self):
        """Test SearchResponse add_* methods."""
        response = SearchResponse(query="test", results=[])

        response.add_suggestion("try this")
        response.add_suggestion("or that")
        response.add_suggestion("try this")  # Duplicate should be ignored

        response.add_navigation_hint("hint 1")
        response.add_navigation_hint("hint 2")
        response.add_navigation_hint("hint 1")  # Duplicate should be ignored

        assert len(response.suggested_queries) == 2
        assert "try this" in response.suggested_queries
        assert "or that" in response.suggested_queries

        assert len(response.navigation_hints) == 2
        assert "hint 1" in response.navigation_hints
        assert "hint 2" in response.navigation_hints

    def test_create_search_response_from_results(self):
        """Test the helper function for creating SearchResponse."""
        snippet = CodeSnippet(text="def test():", start_line=1, end_line=1)
        result1 = CodeSearchResult("test1.py", snippet, 0.9, {"language": "Python"})
        result2 = CodeSearchResult("test2.py", snippet, 0.8, {"language": "Python"})

        response = create_search_response_from_results(
            query="test query", results=[result1, result2], execution_time=0.5, add_clusters=True, add_suggestions=True
        )

        assert response.query == "test query"
        assert len(response.results) == 2
        assert response.execution_time == 0.5
        assert len(response.suggested_queries) > 0  # Should have suggestions


class TestIndexResponse:
    """Test the IndexResponse data class and JSON serialization."""

    def test_create_index_response(self):
        """Test creating an IndexResponse instance."""
        response = IndexResponse(
            operation="index",
            status="success",
            message="Successfully indexed files",
            files_processed=10,
            files_skipped=2,
            total_files_scanned=12,
            execution_time=30.5,
            repository_path="/test/repo",
        )

        assert response.operation == "index"
        assert response.status == "success"
        assert response.files_processed == 10
        assert response.files_skipped == 2
        assert response.total_files_scanned == 12
        assert response.execution_time == 30.5
        assert response.repository_path == "/test/repo"
        assert response.version == "1.0"
        assert response.timestamp is not None

        # Check computed processing rate
        expected_rate = 10 / 30.5
        assert abs(response.processing_rate - expected_rate) < 0.01

    def test_index_response_add_methods(self):
        """Test IndexResponse add_* methods."""
        response = IndexResponse(operation="index", status="success", message="test")

        response.add_warning("Warning 1")
        response.add_warning("Warning 2")
        response.add_warning("Warning 1")  # Duplicate should be ignored

        response.add_error("Error 1")
        response.add_error("Error 2")
        response.add_error("Error 1")  # Duplicate should be ignored

        assert len(response.warnings) == 2
        assert "Warning 1" in response.warnings
        assert "Warning 2" in response.warnings

        assert len(response.errors) == 2
        assert "Error 1" in response.errors
        assert "Error 2" in response.errors

    def test_index_response_is_successful(self):
        """Test IndexResponse success checking."""
        # Successful response
        success_response = IndexResponse(operation="index", status="success", message="success")
        assert success_response.is_successful() is True

        # Failed response
        failed_response = IndexResponse(operation="index", status="failed", message="failed")
        assert failed_response.is_successful() is False

        # Response with errors
        error_response = IndexResponse(operation="index", status="success", message="success")
        error_response.add_error("Some error")
        assert error_response.is_successful() is False

    def test_index_response_to_json(self):
        """Test converting IndexResponse to JSON."""
        response = IndexResponse(
            operation="index", status="success", message="Successfully indexed files", files_processed=5
        )

        json_str = response.to_json()

        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert parsed["operation"] == "index"
        assert parsed["status"] == "success"
        assert parsed["files_processed"] == 5


class TestStatusResponse:
    """Test the StatusResponse data class and JSON serialization."""

    def test_create_status_response(self):
        """Test creating a StatusResponse instance."""
        response = StatusResponse(
            status="healthy",
            is_ready_for_search=True,
            total_files=100,
            files_with_embeddings=95,
            total_embeddings=95,
            database_size_mb=50.5,
            repository_path="/test/repo",
            file_types={"Python": 70, "JavaScript": 25, "Other": 5},
        )

        assert response.status == "healthy"
        assert response.is_ready_for_search is True
        assert response.total_files == 100
        assert response.files_with_embeddings == 95
        assert response.database_size_mb == 50.5
        assert response.repository_path == "/test/repo"
        assert len(response.file_types) == 3
        assert response.version == "1.0"
        assert response.timestamp is not None

    def test_status_response_add_methods(self):
        """Test StatusResponse add_* methods."""
        response = StatusResponse(status="healthy", is_ready_for_search=True)

        response.add_recommendation("Rec 1")
        response.add_recommendation("Rec 2")
        response.add_recommendation("Rec 1")  # Duplicate should be ignored

        response.add_warning("Warning 1")
        response.add_warning("Warning 2")
        response.add_warning("Warning 1")  # Duplicate should be ignored

        assert len(response.recommendations) == 2
        assert "Rec 1" in response.recommendations
        assert "Rec 2" in response.recommendations

        assert len(response.warnings) == 2
        assert "Warning 1" in response.warnings
        assert "Warning 2" in response.warnings

    def test_status_response_health_score(self):
        """Test StatusResponse health score computation."""
        # Perfect health
        perfect_response = StatusResponse(
            status="healthy", is_ready_for_search=True, total_files=100, files_with_embeddings=100, is_index_fresh=True
        )
        assert perfect_response.compute_health_score() == 100.0

        # Degraded health - missing embeddings
        degraded_response = StatusResponse(
            status="degraded",
            is_ready_for_search=True,
            total_files=100,
            files_with_embeddings=80,  # 20% missing
            is_index_fresh=True,
        )
        # Should lose 30% of 20% missing = 6 points
        assert perfect_response.compute_health_score() - degraded_response.compute_health_score() == pytest.approx(
            6, abs=1
        )

        # Not ready for search
        not_ready_response = StatusResponse(
            status="offline", is_ready_for_search=False, total_files=0, files_with_embeddings=0, is_index_fresh=True
        )
        # Should lose 30 points for not being ready
        assert not_ready_response.compute_health_score() == 70.0

    def test_status_response_to_json(self):
        """Test converting StatusResponse to JSON."""
        response = StatusResponse(status="healthy", is_ready_for_search=True, total_files=50)

        json_str = response.to_json()

        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert parsed["status"] == "healthy"
        assert parsed["is_ready_for_search"] is True
        assert parsed["total_files"] == 50
        assert "health_score" in parsed  # Should be computed automatically


class TestCrossReferences:
    """Test cross-reference generation functionality."""

    def test_generate_cross_references(self):
        """Test generating cross-references between related files."""
        snippet = CodeSnippet(text="code", start_line=1, end_line=1)

        # Files in same directory
        result1 = CodeSearchResult("/project/src/auth.py", snippet, 0.9)
        result2 = CodeSearchResult("/project/src/user.py", snippet, 0.8)
        result3 = CodeSearchResult("/project/tests/test_auth.py", snippet, 0.7)

        # Files with similar names
        result4 = CodeSearchResult("/project/models/auth.py", snippet, 0.6)
        result5 = CodeSearchResult("/project/models/auth.js", snippet, 0.5)

        results = [result1, result2, result3, result4, result5]
        cross_refs = generate_cross_references(results)

        assert len(cross_refs) > 0

        # Should find related files in src/ directory
        src_ref = next((ref for ref in cross_refs if "src/" in ref), None)
        assert src_ref is not None

        # Should find related 'auth' files
        auth_ref = next((ref for ref in cross_refs if "'auth'" in ref), None)
        assert auth_ref is not None


class TestIntegrationScenarios:
    """Integration tests for complete structured response workflows."""

    def test_empty_search_response(self):
        """Test structured response for empty search results."""
        response = SearchResponse(query="nonexistent code", results=[], total_results=0, execution_time=0.1)

        json_str = response.to_json()
        parsed = json.loads(json_str)

        assert parsed["total_results"] == 0
        assert len(parsed["results"]) == 0
        assert parsed["execution_time"] == 0.1

    def test_complex_search_response(self):
        """Test structured response with all features enabled."""
        # Create diverse search results
        snippet = CodeSnippet(text="def authenticate(user):", start_line=10, end_line=12)

        results = [
            CodeSearchResult("/app/auth/login.py", snippet, 0.95, {"language": "Python"}),
            CodeSearchResult("/app/auth/token.py", snippet, 0.90, {"language": "Python"}),
            CodeSearchResult("/app/frontend/auth.js", snippet, 0.80, {"language": "JavaScript"}),
            CodeSearchResult("/app/tests/test_auth.py", snippet, 0.70, {"language": "Python"}),
        ]

        # Create comprehensive response
        response = create_search_response_from_results(
            query="user authentication", results=results, execution_time=0.25, add_clusters=True, add_suggestions=True
        )

        # Add query analysis
        response.query_analysis = _analyze_search_query("user authentication", results)

        # Verify comprehensive structure
        assert response.query == "user authentication"
        assert len(response.results) == 4
        assert response.execution_time == 0.25

        # Check language breakdown
        assert response.language_breakdown["Python"] == 3
        assert response.language_breakdown["JavaScript"] == 1

        # Check confidence distribution
        assert response.confidence_distribution["high"] >= 2  # 0.95 and 0.90 are high
        assert response.confidence_distribution["medium"] >= 1  # 0.80 is medium

        # Check clusters were created
        assert len(response.result_clusters) > 0

        # Check suggestions were generated
        assert len(response.suggested_queries) > 0

        # Check query analysis
        assert response.query_analysis is not None
        assert response.query_analysis.original_query == "user authentication"
        assert response.query_analysis.estimated_result_count == 4

        # Verify JSON serialization works
        json_str = response.to_json()
        parsed = json.loads(json_str)

        assert parsed["query"] == "user authentication"
        assert len(parsed["results"]) == 4
        assert "query_analysis" in parsed
        assert "result_clusters" in parsed
        assert "language_breakdown" in parsed
        assert "confidence_distribution" in parsed

    def test_error_response_scenarios(self):
        """Test structured error responses."""
        # Search error response
        search_error = SearchResponse(
            query="failed query", results=[], total_results=0, performance_notes=["Database connection failed"]
        )

        json_str = search_error.to_json()
        parsed = json.loads(json_str)

        assert parsed["total_results"] == 0
        assert "Database connection failed" in parsed["performance_notes"]

        # Index error response
        index_error = IndexResponse(operation="index", status="failed", message="Repository not found")
        index_error.add_error("Path does not exist")

        json_str = index_error.to_json()
        parsed = json.loads(json_str)

        assert parsed["status"] == "failed"
        assert "Path does not exist" in parsed["errors"]
        assert index_error.is_successful() is False

        # Status error response
        status_error = StatusResponse(
            status="error", is_ready_for_search=False, total_files=0, files_with_embeddings=0, total_embeddings=0
        )
        status_error.add_warning("Index check failed")

        json_str = status_error.to_json()
        parsed = json.loads(json_str)

        assert parsed["status"] == "error"
        assert parsed["is_ready_for_search"] is False
        assert "Index check failed" in parsed["warnings"]
        assert parsed["health_score"] < 100  # Should be degraded


# Test fixtures and utilities
@pytest.fixture
def sample_search_results():
    """Fixture providing sample search results for testing."""
    snippet = CodeSnippet(text="def test_function():", start_line=1, end_line=1)

    return [
        CodeSearchResult("test1.py", snippet, 0.95, {"language": "Python"}),
        CodeSearchResult("test2.js", snippet, 0.85, {"language": "JavaScript"}),
        CodeSearchResult("test3.py", snippet, 0.75, {"language": "Python"}),
    ]


@pytest.fixture
def mock_database():
    """Fixture providing a mock database for testing."""
    mock_db = Mock()
    mock_db.execute_with_retry.return_value = [
        ("test.py", "def test(): pass", 0.1),
        ("main.py", "if __name__ == '__main__':", 0.2),
    ]
    return mock_db


@pytest.fixture
def mock_embedder():
    """Fixture providing a mock embedder for testing."""
    mock_embedder = Mock()
    mock_embedder.encode.return_value = [0.1] * 384  # Mock 384-dim embedding
    return mock_embedder


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__])
