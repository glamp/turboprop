#!/usr/bin/env python3
"""
test_construct_search.py: Tests for construct-level semantic search functionality.

This module tests all aspects of the construct search implementation including:
- Basic construct search operations
- Specialized searches (functions, classes, imports)
- Hybrid search functionality
- Result formatting and display
- Error handling and edge cases
"""

from unittest.mock import Mock, patch

import pytest
from construct_search import ConstructSearchOperations, ConstructSearchResult, format_construct_search_results
from database_manager import DatabaseManager
from embedding_helper import EmbeddingGenerator
from exceptions import SearchError
from search_operations import format_hybrid_search_results, search_hybrid
from search_result_types import CodeSearchResult, CodeSnippet


class TestConstructSearchResult:
    """Test the ConstructSearchResult data class."""

    def test_construct_search_result_creation(self):
        """Test creating a basic ConstructSearchResult."""
        result = ConstructSearchResult.create(
            construct_id="test_id",
            file_path="/test/file.py",
            construct_type="function",
            name="test_function",
            signature="def test_function(param: str) -> bool:",
            start_line=10,
            end_line=15,
            similarity_score=0.85,
            docstring="Test function docstring",
        )

        assert result.construct_id == "test_id"
        assert result.file_path == "/test/file.py"
        assert result.construct_type == "function"
        assert result.name == "test_function"
        assert result.similarity_score == 0.85
        assert result.confidence_level == "high"  # Should be auto-calculated

    def test_confidence_level_calculation(self):
        """Test automatic confidence level calculation based on similarity score."""
        # High confidence
        high_result = ConstructSearchResult.create(
            construct_id="high",
            file_path="test.py",
            construct_type="function",
            name="test",
            signature="def test():",
            start_line=1,
            end_line=1,
            similarity_score=0.9,
        )
        assert high_result.confidence_level == "high"

        # Medium confidence
        med_result = ConstructSearchResult.create(
            construct_id="med",
            file_path="test.py",
            construct_type="function",
            name="test",
            signature="def test():",
            start_line=1,
            end_line=1,
            similarity_score=0.6,
        )
        assert med_result.confidence_level == "medium"

        # Low confidence
        low_result = ConstructSearchResult.create(
            construct_id="low",
            file_path="test.py",
            construct_type="function",
            name="test",
            signature="def test():",
            start_line=1,
            end_line=1,
            similarity_score=0.3,
        )
        assert low_result.confidence_level == "low"

    def test_to_code_search_result_conversion(self):
        """Test converting ConstructSearchResult to CodeSearchResult."""
        construct_result = ConstructSearchResult.create(
            construct_id="test_id",
            file_path="/test/file.py",
            construct_type="function",
            name="test_function",
            signature="def test_function(param: str) -> bool:",
            start_line=10,
            end_line=15,
            similarity_score=0.85,
            docstring="Test function docstring",
        )

        code_result = construct_result.to_code_search_result()

        assert isinstance(code_result, CodeSearchResult)
        assert code_result.file_path == "/test/file.py"
        assert code_result.similarity_score == 0.85
        assert code_result.confidence_level == "high"
        assert "test_function" in code_result.snippet.text
        assert code_result.file_metadata["construct_type"] == "function"
        assert code_result.file_metadata["construct_name"] == "test_function"

    def test_to_dict_serialization(self):
        """Test dictionary serialization for JSON output."""
        result = ConstructSearchResult.create(
            construct_id="test_id",
            file_path="/test/file.py",
            construct_type="function",
            name="test_function",
            signature="def test_function():",
            start_line=10,
            end_line=15,
            similarity_score=0.85,
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["construct_id"] == "test_id"
        assert result_dict["construct_type"] == "function"
        assert result_dict["name"] == "test_function"
        assert result_dict["similarity_score"] == 0.85


class TestConstructSearchOperations:
    """Test the ConstructSearchOperations class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_db_manager = Mock(spec=DatabaseManager)
        self.mock_embedder = Mock(spec=EmbeddingGenerator)
        self.construct_ops = ConstructSearchOperations(self.mock_db_manager, self.mock_embedder)

    def test_initialization(self):
        """Test ConstructSearchOperations initialization."""
        assert self.construct_ops.db_manager == self.mock_db_manager
        assert self.construct_ops.embedder == self.mock_embedder

    def test_search_constructs_basic(self):
        """Test basic construct search functionality."""
        # Mock embedding generation
        self.mock_embedder.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

        # Mock database results
        mock_results = [
            (
                "construct_1",
                "/test/file.py",
                "function",
                "test_func",
                "def test_func():",
                10,
                15,
                "Test docstring",
                None,
                "file_1",
                0.85,
            ),
            (
                "construct_2",
                "/test/file2.py",
                "class",
                "TestClass",
                "class TestClass:",
                1,
                20,
                "Test class",
                None,
                "file_2",
                0.75,
            ),
        ]

        mock_connection = Mock()
        mock_connection.execute.return_value.fetchall.return_value = mock_results

        # Configure context manager behavior
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=mock_connection)
        context_manager.__exit__ = Mock(return_value=None)
        self.mock_db_manager.get_connection.return_value = context_manager

        # Execute search
        results = self.construct_ops.search_constructs("test query", k=10)

        # Verify results
        assert len(results) == 2
        assert all(isinstance(r, ConstructSearchResult) for r in results)
        assert results[0].name == "test_func"
        assert results[0].construct_type == "function"
        assert results[1].name == "TestClass"
        assert results[1].construct_type == "class"

    def test_search_constructs_with_type_filter(self):
        """Test construct search with construct type filtering."""
        self.mock_embedder.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

        mock_results = [
            (
                "construct_1",
                "/test/file.py",
                "function",
                "test_func",
                "def test_func():",
                10,
                15,
                None,
                None,
                "file_1",
                0.85,
            )
        ]

        mock_connection = Mock()
        mock_connection.execute.return_value.fetchall.return_value = mock_results

        # Configure context manager behavior
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=mock_connection)
        context_manager.__exit__ = Mock(return_value=None)
        self.mock_db_manager.get_connection.return_value = context_manager

        # Execute search with type filter
        results = self.construct_ops.search_constructs("test query", k=10, construct_types=["function"])

        # Verify the SQL query was called with type filtering
        mock_connection.execute.assert_called()
        call_args = mock_connection.execute.call_args
        sql_query = call_args[0][0]
        assert "construct_type IN" in sql_query

        # Verify results
        assert len(results) == 1
        assert results[0].construct_type == "function"

    def test_search_functions_specific(self):
        """Test specialized function search."""
        self.mock_embedder.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

        mock_results = [
            (
                "func_1",
                "/test/file.py",
                "function",
                "test_func",
                "def test_func():",
                10,
                15,
                None,
                None,
                "file_1",
                0.85,
            ),
            (
                "method_1",
                "/test/file.py",
                "method",
                "test_method",
                "def test_method(self):",
                20,
                25,
                None,
                "class_1",
                "file_1",
                0.75,
            ),
        ]

        mock_connection = Mock()
        mock_connection.execute.return_value.fetchall.return_value = mock_results

        # Configure context manager behavior
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=mock_connection)
        context_manager.__exit__ = Mock(return_value=None)
        self.mock_db_manager.get_connection.return_value = context_manager

        # Execute function search
        results = self.construct_ops.search_functions("test query", k=10)

        # Verify results contain only functions and methods
        assert len(results) == 2
        assert all(r.construct_type in ["function", "method"] for r in results)

    def test_search_classes_specific(self):
        """Test specialized class search."""
        self.mock_embedder.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

        mock_results = [
            (
                "class_1",
                "/test/file.py",
                "class",
                "TestClass",
                "class TestClass:",
                1,
                30,
                "Test class docstring",
                None,
                "file_1",
                0.85,
            )
        ]

        mock_connection = Mock()
        mock_connection.execute.return_value.fetchall.return_value = mock_results

        # Configure context manager behavior
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=mock_connection)
        context_manager.__exit__ = Mock(return_value=None)
        self.mock_db_manager.get_connection.return_value = context_manager

        # Execute class search
        results = self.construct_ops.search_classes("test query", k=10)

        # Verify results contain only classes
        assert len(results) == 1
        assert results[0].construct_type == "class"
        assert results[0].name == "TestClass"

    def test_search_imports_specific(self):
        """Test specialized import search."""
        self.mock_embedder.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

        mock_results = [
            ("import_1", "/test/file.py", "import", "requests", "import requests", 1, 1, None, None, "file_1", 0.85),
            (
                "import_2",
                "/test/file.py",
                "import",
                "json.loads",
                "from json import loads",
                2,
                2,
                None,
                None,
                "file_1",
                0.75,
            ),
        ]

        mock_connection = Mock()
        mock_connection.execute.return_value.fetchall.return_value = mock_results

        # Configure context manager behavior
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=mock_connection)
        context_manager.__exit__ = Mock(return_value=None)
        self.mock_db_manager.get_connection.return_value = context_manager

        # Execute import search
        results = self.construct_ops.search_imports("test query", k=10)

        # Verify results contain only imports
        assert len(results) == 2
        assert all(r.construct_type == "import" for r in results)

    def test_get_related_constructs(self):
        """Test finding related constructs."""
        # Mock construct info lookup
        construct_info = ("file_1", None, "class", "TestClass")

        # Mock related constructs results
        related_results = [
            (
                "method_1",
                "/test/file.py",
                "method",
                "method1",
                "def method1(self):",
                10,
                15,
                None,
                "class_1",
                "file_1",
                0.8,
            ),
            (
                "method_2",
                "/test/file.py",
                "method",
                "method2",
                "def method2(self):",
                20,
                25,
                None,
                "class_1",
                "file_1",
                0.8,
            ),
        ]

        mock_connection = Mock()
        mock_connection.execute.return_value.fetchone.return_value = construct_info
        mock_connection.execute.return_value.fetchall.return_value = related_results

        # Configure context manager behavior
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=mock_connection)
        context_manager.__exit__ = Mock(return_value=None)
        self.mock_db_manager.get_connection.return_value = context_manager

        # Execute related constructs search
        results = self.construct_ops.get_related_constructs("class_1", k=5)

        # Verify results
        assert len(results) == 2
        assert all(r.construct_type == "method" for r in results)
        assert results[0].parent_construct_id == "class_1"

    def test_get_construct_statistics(self):
        """Test getting construct statistics."""
        # Mock statistics queries
        total_count = 100
        type_counts = [("function", 50), ("class", 30), ("method", 15), ("import", 5)]
        embedded_count = 95

        mock_connection = Mock()
        mock_connection.execute.side_effect = [
            Mock(fetchone=Mock(return_value=[total_count])),  # Total count
            Mock(fetchall=Mock(return_value=type_counts)),  # Type counts
            Mock(fetchone=Mock(return_value=[embedded_count])),  # Embedded count
        ]

        # Configure context manager behavior
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=mock_connection)
        context_manager.__exit__ = Mock(return_value=None)
        self.mock_db_manager.get_connection.return_value = context_manager

        # Execute statistics gathering
        stats = self.construct_ops.get_construct_statistics()

        # Verify statistics
        assert stats["total_constructs"] == 100
        assert stats["embedded_constructs"] == 95
        assert stats["embedding_coverage"] == 0.95
        assert stats["construct_types"]["function"] == 50
        assert stats["construct_types"]["class"] == 30

    def test_error_handling(self):
        """Test error handling in construct search."""
        # Mock database error
        self.mock_embedder.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
        self.mock_db_manager.get_connection.side_effect = Exception("Database error")

        # Execute search that should raise SearchError for general exceptions
        with pytest.raises(SearchError, match="construct search failed"):
            self.construct_ops.search_constructs("test query")


class TestHybridSearch:
    """Test hybrid search functionality combining files and constructs."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_db_manager = Mock(spec=DatabaseManager)
        self.mock_embedder = Mock(spec=EmbeddingGenerator)

        # Mock embedder methods
        import numpy as np

        self.mock_embedder.encode.return_value = np.array([0.1, 0.2, 0.3] * 128)  # 384-dimensional vector
        self.mock_embedder.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

        # Mock database manager methods
        self.mock_db_manager.execute_with_retry.return_value = [
            ("/test/file.py", "test content", 0.2)  # (path, content, distance)
        ]
        self.mock_db_manager.search_full_text.return_value = [
            ("file_id", "/test/file.py", "test content", 0.8)  # (file_id, path, content, relevance)
        ]
        self.mock_db_manager.create_fts_index.return_value = True

    @patch("search_operations.ConstructSearchOperations")
    @patch("search_operations.search_index_enhanced")
    def test_search_hybrid_basic(self, mock_search_enhanced, mock_construct_ops_class):
        """Test basic hybrid search functionality."""
        # Mock construct search results
        mock_construct_results = [
            ConstructSearchResult.create(
                construct_id="func_1",
                file_path="/test/file1.py",
                construct_type="function",
                name="test_func",
                signature="def test_func():",
                start_line=10,
                end_line=15,
                similarity_score=0.85,
            )
        ]

        # Mock file search results
        mock_file_results = [
            CodeSearchResult(
                file_path="/test/file2.py",
                snippet=CodeSnippet(text="test content", start_line=1, end_line=5),
                similarity_score=0.75,
            )
        ]

        # Setup mocks
        mock_construct_ops = Mock()
        mock_construct_ops.search_constructs.return_value = mock_construct_results
        mock_construct_ops_class.return_value = mock_construct_ops
        mock_search_enhanced.return_value = mock_file_results

        # Execute hybrid search
        results = search_hybrid(self.mock_db_manager, self.mock_embedder, "test query", k=10)

        # Verify results
        assert len(results) == 2  # Should have both construct and file results
        assert any(r.file_path == "/test/file1.py" for r in results)
        assert any(r.file_path == "/test/file2.py" for r in results)

    def test_format_hybrid_search_results(self):
        """Test formatting of hybrid search results."""
        # Create test results with construct context
        results = [
            CodeSearchResult(
                file_path="/test/file.py",
                snippet=CodeSnippet(text="def test_function():", start_line=10, end_line=15),
                similarity_score=0.85,
                file_metadata={
                    "construct_context": {
                        "related_constructs": 2,
                        "construct_types": ["function", "class"],
                        "top_constructs": [
                            {"name": "test_func", "type": "function", "line": 10, "signature": "def test_func():"},
                            {"name": "TestClass", "type": "class", "line": 1, "signature": "class TestClass:"},
                        ],
                    }
                },
            )
        ]

        # Format results
        formatted = format_hybrid_search_results(results, "test query", show_construct_context=True)

        # Verify formatting
        assert "hybrid search results" in formatted
        assert "test query" in formatted
        assert "2 constructs" in formatted
        assert "test_func" in formatted
        assert "TestClass" in formatted


class TestResultFormatting:
    """Test result formatting functions."""

    def test_format_construct_search_results_basic(self):
        """Test basic construct search result formatting."""
        results = [
            ConstructSearchResult.create(
                construct_id="test_1",
                file_path="/test/file.py",
                construct_type="function",
                name="test_function",
                signature="def test_function(param: str) -> bool:",
                start_line=10,
                end_line=15,
                similarity_score=0.85,
                docstring="Test function docstring",
            ),
            ConstructSearchResult.create(
                construct_id="test_2",
                file_path="/test/file.py",
                construct_type="class",
                name="TestClass",
                signature="class TestClass(BaseClass):",
                start_line=1,
                end_line=30,
                similarity_score=0.75,
                docstring="Test class docstring",
            ),
        ]

        formatted = format_construct_search_results(results, "test query")

        # Verify formatting includes key information
        assert "2 construct matches" in formatted
        assert "test query" in formatted
        assert "test_function" in formatted
        assert "TestClass" in formatted
        assert "def test_function" in formatted
        assert "class TestClass" in formatted
        assert "/test/file.py:10-15" in formatted
        assert "/test/file.py:1-30" in formatted

    def test_format_construct_search_results_no_docstrings(self):
        """Test formatting construct results without showing docstrings."""
        results = [
            ConstructSearchResult.create(
                construct_id="test_1",
                file_path="/test/file.py",
                construct_type="function",
                name="test_function",
                signature="def test_function():",
                start_line=10,
                end_line=15,
                similarity_score=0.85,
                docstring="This should not appear",
            )
        ]

        formatted = format_construct_search_results(results, "test query", show_docstrings=False)

        # Verify docstring is not included
        assert "This should not appear" not in formatted
        assert "test_function" in formatted

    def test_format_construct_search_results_empty(self):
        """Test formatting empty construct search results."""
        results = []
        formatted = format_construct_search_results(results, "test query")

        assert "No construct matches found" in formatted
        assert "test query" in formatted


class TestErrorHandling:
    """Test error handling in construct search functionality."""

    def test_construct_search_database_error(self):
        """Test handling of database errors in construct search."""
        mock_db_manager = Mock(spec=DatabaseManager)
        mock_embedder = Mock(spec=EmbeddingGenerator)

        # Mock database error
        mock_db_manager.get_connection.side_effect = Exception("Database connection failed")
        mock_embedder.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

        construct_ops = ConstructSearchOperations(mock_db_manager, mock_embedder)

        # Should raise SearchError for general exceptions
        with pytest.raises(SearchError, match="construct search failed"):
            construct_ops.search_constructs("test query")

    def test_construct_search_embedding_error(self):
        """Test handling of embedding generation errors."""
        mock_db_manager = Mock(spec=DatabaseManager)
        mock_embedder = Mock(spec=EmbeddingGenerator)

        # Mock embedding error
        mock_embedder.generate_embeddings.side_effect = Exception("Embedding generation failed")

        construct_ops = ConstructSearchOperations(mock_db_manager, mock_embedder)

        # Should raise SearchError for general exceptions
        with pytest.raises(SearchError, match="construct search failed"):
            construct_ops.search_constructs("test query")


class TestIntegration:
    """Integration tests for construct search functionality."""

    def test_end_to_end_construct_search(self):
        """Test end-to-end construct search with mocked components."""
        # This would be a more comprehensive test in a real environment
        # For now, we verify the main components work together

        mock_db_manager = Mock(spec=DatabaseManager)
        mock_embedder = Mock(spec=EmbeddingGenerator)

        # Mock successful embedding generation
        mock_embedder.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

        # Mock successful database query
        mock_results = [
            (
                "construct_1",
                "/test/file.py",
                "function",
                "test_func",
                "def test_func():",
                10,
                15,
                "Test docstring",
                None,
                "file_1",
                0.85,
            )
        ]

        mock_connection = Mock()
        mock_connection.execute.return_value.fetchall.return_value = mock_results

        # Configure context manager behavior
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=mock_connection)
        context_manager.__exit__ = Mock(return_value=None)
        mock_db_manager.get_connection.return_value = context_manager

        # Execute full construct search workflow
        construct_ops = ConstructSearchOperations(mock_db_manager, mock_embedder)
        results = construct_ops.search_constructs("test query")
        formatted = format_construct_search_results(results, "test query")

        # Verify end-to-end functionality
        assert len(results) == 1
        assert results[0].name == "test_func"
        assert "test_func" in formatted
        assert "test query" in formatted


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
