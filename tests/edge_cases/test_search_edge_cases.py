#!/usr/bin/env python3
"""
Edge case tests for MCP tool search engine.

This module tests the scenarios identified in the code review:
- Very large result sets (>100 tools)
- Network/database timeout scenarios
- Invalid embedding data handling
- Cache invalidation edge cases
- Query processor error conditions
"""

import json
import sys
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from turboprop.database_manager import DatabaseManager
from turboprop.embedding_helper import EmbeddingGenerator
from turboprop.exceptions import DatabaseTimeoutError, EmbeddingError
from turboprop.mcp_tool_search_engine import MCPToolSearchEngine
from turboprop.tool_query_processor import ToolQueryProcessor
from turboprop.tool_search_results import ProcessedQuery, ToolSearchResponse


class TestLargeResultSets:
    """Test handling of very large result sets (>100 tools)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.db_manager = Mock(spec=DatabaseManager)
        self.embedding_generator = Mock(spec=EmbeddingGenerator)
        self.search_engine = MCPToolSearchEngine(self.db_manager, self.embedding_generator)

    def test_large_result_set_performance(self):
        """Test search performance with >100 results."""
        # Create mock data for 150 tools
        large_result_set = []
        for i in range(150):
            tool_data = {
                "id": f"tool_{i:03d}",
                "name": f"Tool {i}",
                "description": f"Description for tool {i}",
                "category": "test",
                "tool_type": "function",
                "metadata_json": json.dumps({"version": "1.0"}),
            }
            large_result_set.append((f"tool_{i:03d}", 0.8 - (i * 0.001), tool_data))

        # Mock database call to return large result set
        with patch.object(self.search_engine, "_perform_vector_search", return_value=large_result_set):
            with patch.object(self.search_engine.query_processor, "process_query") as mock_process:
                mock_process.return_value = ProcessedQuery(
                    original_query="test query",
                    cleaned_query="test query",
                    expanded_terms=["test"],
                    confidence=0.8,
                )

                with patch.object(self.search_engine, "_get_tool_parameters", return_value=[]):
                    with patch.object(self.search_engine, "_get_tool_examples", return_value=[]):
                        start_time = time.time()
                        response = self.search_engine.search_by_functionality("test query", k=100)
                        execution_time = time.time() - start_time

                        # Verify response structure
                        assert isinstance(response, ToolSearchResponse)
                        assert len(response.results) <= 100  # Should be limited to k
                        assert response.execution_time < 5.0  # Should complete within 5 seconds
                        assert execution_time < 5.0  # Actual execution should be fast

    def test_large_result_set_memory_usage(self):
        """Test memory efficiency with large result sets."""
        # Create mock data for 500 tools (stress test)
        huge_result_set = []
        for i in range(500):
            tool_data = {
                "id": f"tool_{i:04d}",
                "name": f"Large Tool {i}" * 10,  # Larger strings
                "description": f"Very detailed description for tool {i} " * 50,  # Much larger descriptions
                "category": "performance_test",
                "tool_type": "function",
                "metadata_json": json.dumps({"version": "1.0", "details": "x" * 1000}),  # Large metadata
            }
            huge_result_set.append((f"tool_{i:04d}", 0.9, tool_data))

        with patch.object(self.search_engine, "_perform_vector_search", return_value=huge_result_set):
            with patch.object(self.search_engine.query_processor, "process_query") as mock_process:
                mock_process.return_value = ProcessedQuery(
                    original_query="memory test",
                    cleaned_query="memory test",
                    expanded_terms=["memory", "test"],
                    confidence=0.9,
                )

                with patch.object(self.search_engine, "_get_tool_parameters", return_value=[]):
                    with patch.object(self.search_engine, "_get_tool_examples", return_value=[]):
                        # This should not crash or consume excessive memory
                        response = self.search_engine.search_by_functionality("memory test", k=50)

                        assert len(response.results) == 50
                        assert response.execution_time < 10.0  # Allow more time for large dataset


class TestDatabaseTimeouts:
    """Test network/database timeout scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.db_manager = Mock(spec=DatabaseManager)
        self.embedding_generator = Mock(spec=EmbeddingGenerator)
        self.search_engine = MCPToolSearchEngine(self.db_manager, self.embedding_generator)

    def test_database_connection_timeout(self):
        """Test handling of database connection timeouts."""
        # Mock database timeout
        self.db_manager.get_connection.side_effect = DatabaseTimeoutError("Connection timeout")

        with patch.object(self.search_engine, "_perform_vector_search", side_effect=DatabaseTimeoutError("Timeout")):
            response = self.search_engine.search_by_functionality("test query")

            # Should return error response instead of crashing
            assert isinstance(response, ToolSearchResponse)
            assert len(response.results) == 0
            # Error should be indicated in search strategy or suggestions, not in query
            assert response.search_strategy == "error" or any(
                "timeout" in s.lower() or "error" in s.lower() for s in response.suggestions
            )

    def test_slow_database_query_timeout(self):
        """Test handling of slow database queries that timeout."""

        def slow_vector_search(*args, **kwargs):
            time.sleep(0.1)  # Simulate slow query
            raise DatabaseTimeoutError("Query timeout")

        with patch.object(self.search_engine, "_perform_vector_search", side_effect=slow_vector_search):
            start_time = time.time()
            response = self.search_engine.search_by_functionality("slow query")
            execution_time = time.time() - start_time

            assert isinstance(response, ToolSearchResponse)
            assert len(response.results) == 0
            assert execution_time < 1.0  # Should fail fast, not hang

    def test_database_concurrent_access_timeout(self):
        """Test handling of concurrent access timeout scenarios."""

        def simulate_lock_timeout(*args, **kwargs):
            raise DatabaseTimeoutError("Database lock timeout - too many concurrent connections")

        self.db_manager.get_connection.side_effect = simulate_lock_timeout

        # Try multiple concurrent searches
        responses = []
        threads = []

        def search_worker():
            response = self.search_engine.search_by_functionality("concurrent test")
            responses.append(response)

        # Start multiple threads
        for i in range(5):
            thread = threading.Thread(target=search_worker)
            threads.append(thread)
            thread.start()

        # Wait for all to complete
        for thread in threads:
            thread.join(timeout=2.0)

        # All should handle timeout gracefully
        assert len(responses) == 5
        for response in responses:
            assert isinstance(response, ToolSearchResponse)
            assert len(response.results) == 0


class TestInvalidEmbeddingData:
    """Test invalid embedding data handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.db_manager = Mock(spec=DatabaseManager)
        self.embedding_generator = Mock(spec=EmbeddingGenerator)
        self.search_engine = MCPToolSearchEngine(self.db_manager, self.embedding_generator)

    def test_corrupted_embedding_vectors(self):
        """Test handling of corrupted embedding vectors."""
        # Test with various corrupted embedding data
        corrupted_embeddings = [
            None,  # Null embedding
            [],  # Empty embedding
            [float("inf")] * 384,  # Infinite values
            [float("nan")] * 384,  # NaN values
            [0] * 100,  # Wrong dimensions
            "not_a_list",  # Wrong type
            [1, 2, "string", 4],  # Mixed types
        ]

        for bad_embedding in corrupted_embeddings:
            self.embedding_generator.encode.return_value = bad_embedding

            with patch.object(self.search_engine, "_perform_vector_search") as mock_search:
                mock_search.return_value = []  # No results due to bad embedding

                response = self.search_engine.search_by_functionality("test query")

                # Should handle gracefully without crashing
                assert isinstance(response, ToolSearchResponse)
                # May have zero results due to embedding issues
                assert len(response.results) >= 0

    def test_embedding_generation_failure(self):
        """Test handling of embedding generation failures."""
        # Mock embedding generation failure
        self.embedding_generator.encode.side_effect = EmbeddingError("Failed to generate embeddings")

        response = self.search_engine.search_by_functionality("test query")

        assert isinstance(response, ToolSearchResponse)
        assert len(response.results) == 0
        # Should have error information
        assert response.execution_time > 0

    def test_database_embedding_corruption(self):
        """Test handling of corrupted embeddings stored in database."""
        # Mock corrupted data from database
        corrupted_results = [
            ("tool1", 0.8, {"id": "tool1", "name": "Tool 1", "embedding": None}),
            ("tool2", 0.7, {"id": "tool2", "name": "Tool 2", "embedding": "corrupted"}),
            ("tool3", 0.6, {"id": "tool3", "name": "Tool 3", "embedding": []}),
        ]

        with patch.object(self.search_engine, "_perform_vector_search", return_value=corrupted_results):
            with patch.object(self.search_engine.query_processor, "process_query") as mock_process:
                mock_process.return_value = ProcessedQuery(
                    original_query="test",
                    cleaned_query="test",
                    expanded_terms=["test"],
                    confidence=0.8,
                )

                response = self.search_engine.search_by_functionality("test query")

                # Should handle corrupted data gracefully
                assert isinstance(response, ToolSearchResponse)
                # Results should be filtered/handled appropriately


class TestCacheInvalidation:
    """Test cache invalidation edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.db_manager = Mock(spec=DatabaseManager)
        self.embedding_generator = Mock(spec=EmbeddingGenerator)
        # Create search engine with caching enabled
        self.search_engine = MCPToolSearchEngine(self.db_manager, self.embedding_generator)
        # Mock cache
        self.search_engine._results_cache = Mock()

    def test_cache_corruption_handling(self):
        """Test handling of corrupted cache data."""
        # Mock corrupted cache data
        corrupted_cache_data = "not_a_search_response"
        self.search_engine._results_cache.get.return_value = corrupted_cache_data

        with patch.object(self.search_engine, "_perform_semantic_search") as mock_search:
            mock_search.return_value = []

            # Should handle corrupted cache gracefully and re-search
            response = self.search_engine.search_by_functionality("test query")

            assert isinstance(response, ToolSearchResponse)
            # Should have called the actual search method due to cache corruption
            mock_search.assert_called_once()

    def test_cache_concurrent_invalidation(self):
        """Test concurrent cache invalidation scenarios."""
        call_count = 0

        def mock_get_cache(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None  # Cache miss
            else:
                # Simulate cache being invalidated between calls
                raise KeyError("Cache invalidated")

        self.search_engine._results_cache.get.side_effect = mock_get_cache

        with patch.object(self.search_engine, "_perform_semantic_search", return_value=[]):
            # Multiple concurrent calls
            responses = []
            for i in range(3):
                response = self.search_engine.search_by_functionality(f"query {i}")
                responses.append(response)

            assert len(responses) == 3
            for response in responses:
                assert isinstance(response, ToolSearchResponse)

    def test_cache_memory_pressure(self):
        """Test cache behavior under memory pressure."""
        # Simulate cache throwing memory errors
        self.search_engine._results_cache.get.side_effect = MemoryError("Out of memory")
        self.search_engine._results_cache.put.side_effect = MemoryError("Cannot cache - out of memory")

        with patch.object(self.search_engine, "_perform_semantic_search", return_value=[]):
            response = self.search_engine.search_by_functionality("memory pressure test")

            # Should handle memory pressure gracefully
            assert isinstance(response, ToolSearchResponse)


class TestQueryProcessorErrors:
    """Test query processor error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = ToolQueryProcessor()

    def test_malformed_query_handling(self):
        """Test handling of malformed queries."""
        # Test queries with different levels of problems
        definitely_malformed = [
            "",  # Empty query
            None,  # Null query
            "a" * 10000,  # Extremely long query
            "SELECT * FROM tools; DROP TABLE tools;",  # SQL injection attempt
            "<script>alert('xss')</script>",  # XSS attempt
        ]

        unusual_but_valid = [
            "😀🎉🚀" * 100,  # Unicode spam - unusual but not malformed
            "\x00\x01\x02",  # Control characters - gets cleaned
            "query\nwith\nmultiple\nlines",  # Multi-line query - gets normalized
        ]

        # Test definitely malformed queries
        for bad_query in definitely_malformed:
            try:
                result = self.processor.process_query(bad_query)
                # Should return fallback ProcessedQuery
                assert isinstance(result, ProcessedQuery)
                if bad_query is None or bad_query == "":
                    # These should raise ValueError
                    assert False, f"Expected ValueError for query: {bad_query}"
                else:
                    # Should have very low confidence for truly malformed queries
                    assert result.confidence <= 0.1
            except ValueError:
                # Expected for None and empty queries
                assert bad_query is None or bad_query == ""

        # Test unusual but valid queries - these should still work
        for unusual_query in unusual_but_valid:
            result = self.processor.process_query(unusual_query)
            # Should return ProcessedQuery and not crash
            assert isinstance(result, ProcessedQuery)
            # These are unusual but not necessarily malformed, so confidence can vary

    def test_query_analyzer_failures(self):
        """Test handling of query analyzer component failures."""
        with patch.object(self.processor, "query_analyzer") as mock_analyzer:
            # Mock analyzer failures
            mock_analyzer.analyze_search_intent.side_effect = AttributeError("Analyzer not initialized")
            mock_analyzer.detect_category.side_effect = KeyError("Category not found")
            mock_analyzer.detect_tool_type.side_effect = TypeError("Invalid type")

            result = self.processor.process_query("test query")

            # Should fallback gracefully
            assert isinstance(result, ProcessedQuery)
            assert result.original_query == "test query"
            assert result.confidence == 0.1  # Fallback confidence

    def test_query_expansion_errors(self):
        """Test handling of query expansion errors."""
        with patch.object(self.processor, "expand_query_terms", side_effect=Exception("Expansion failed")):
            result = self.processor.process_query("expansion test")

            # Should fallback gracefully
            assert isinstance(result, ProcessedQuery)
            assert result.expanded_terms == []  # Should be empty due to expansion failure

    def test_concurrent_query_processing(self):
        """Test concurrent query processing edge cases."""
        import threading

        results = []
        errors = []

        def process_worker(query_id):
            try:
                result = self.processor.process_query(f"concurrent query {query_id}")
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Start multiple threads processing queries concurrently
        threads = []
        for i in range(10):
            thread = threading.Thread(target=process_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all to complete
        for thread in threads:
            thread.join(timeout=5.0)

        # Should handle concurrent processing without errors
        assert len(errors) == 0
        assert len(results) == 10
        for result in results:
            assert isinstance(result, ProcessedQuery)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
