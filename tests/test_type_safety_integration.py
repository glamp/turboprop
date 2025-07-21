#!/usr/bin/env python3
"""
Integration tests for type safety improvements in Step 000039.

This module tests the integration of decimal/float type conversion fixes
with the broader search system, including formatting functions, database
integration, and hybrid search scenarios.
"""

import time
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from search_result_types import CodeSearchResult, CodeSnippet, SearchMetadata

# Test constants
BATCH_PROCESSING_TIMEOUT = 5.0  # seconds - maximum time for batch processing tests
PRECISION_TOLERANCE_HIGH = 1e-15  # high precision tolerance for exact comparisons
PRECISION_TOLERANCE_STANDARD = 1e-10  # standard precision tolerance for float comparisons


class TestSearchResultFormattingWithDecimalScores:
    """Test search result formatting functions handle Decimal scores correctly."""

    def test_search_result_formatting_with_decimal_scores(self):
        """Test that search result formatting works with Decimal scores."""
        from decimal import Decimal

        from search_operations import format_hybrid_search_results
        from search_result_types import CodeSearchResult, CodeSnippet

        # Create test results with Decimal scores
        snippet = CodeSnippet(text="test code", start_line=5, end_line=8)
        results = [
            CodeSearchResult(file_path="file1.py", snippet=snippet, similarity_score=Decimal("0.856")),
            CodeSearchResult(file_path="file2.py", snippet=snippet, similarity_score=0.742),  # float for comparison
        ]

        # This should not raise any TypeError
        formatted = format_hybrid_search_results(results, "test query")
        assert isinstance(formatted, str)
        assert "85.6%" in formatted  # Decimal converted correctly
        assert "74.2%" in formatted  # Float works as expected

    def test_format_hybrid_search_results_mixed_types(self):
        """Test formatting with mixed Decimal and float similarity scores."""
        from decimal import Decimal

        from search_operations import format_hybrid_search_results
        from search_result_types import CodeSearchResult, CodeSnippet

        snippet1 = CodeSnippet(text="def function1():", start_line=1, end_line=2)
        snippet2 = CodeSnippet(text="def function2():", start_line=5, end_line=6)
        snippet3 = CodeSnippet(text="class MyClass:", start_line=10, end_line=12)

        # Mix of Decimal and float scores
        results = [
            CodeSearchResult(file_path="decimal_result.py", snippet=snippet1, similarity_score=Decimal("0.95")),
            CodeSearchResult(file_path="float_result.py", snippet=snippet2, similarity_score=0.87),
            CodeSearchResult(file_path="another_decimal.py", snippet=snippet3, similarity_score=Decimal("0.73")),
        ]

        # Should handle mixed types without errors
        formatted = format_hybrid_search_results(results, "mixed type search")

        assert isinstance(formatted, str)
        assert "95.0%" in formatted
        assert "87.0%" in formatted
        assert "73.0%" in formatted
        assert "decimal_result.py" in formatted
        assert "float_result.py" in formatted
        assert "another_decimal.py" in formatted

    def test_format_hybrid_search_results_edge_case_scores(self):
        """Test formatting with edge case similarity scores."""
        from decimal import Decimal

        from search_operations import format_hybrid_search_results
        from search_result_types import CodeSearchResult, CodeSnippet

        snippet = CodeSnippet(text="test", start_line=1, end_line=1)

        results = [
            CodeSearchResult(file_path="perfect_match.py", snippet=snippet, similarity_score=Decimal("1.0")),  # 100%
            CodeSearchResult(file_path="no_match.py", snippet=snippet, similarity_score=0.0),  # 0%
            CodeSearchResult(file_path="tiny_match.py", snippet=snippet, similarity_score=Decimal("0.001")),  # 0.1%
        ]

        formatted = format_hybrid_search_results(results, "edge cases")

        assert isinstance(formatted, str)
        assert "100.0%" in formatted
        assert "0.0%" in formatted
        assert "0.1%" in formatted


class TestDatabaseResultTypes:
    """Test handling of different result types from database operations."""

    def test_code_search_result_from_simulated_duckdb_decimal(self):
        """Test creating CodeSearchResult from simulated DuckDB Decimal results."""
        from decimal import Decimal

        from search_result_types import CodeSearchResult, CodeSnippet

        # Simulate what DuckDB might return (Decimal for calculated similarity)
        duckdb_similarity = Decimal("0.8547362")  # High precision from database calculation

        snippet = CodeSnippet(
            text="def calculate_similarity(query, document):\n    return cosine_similarity(query, document)",
            start_line=45,
            end_line=46,
        )

        result = CodeSearchResult(
            file_path="/path/to/similarity.py",
            snippet=snippet,
            similarity_score=duckdb_similarity,
            file_metadata={"language": "python", "size": 2048},
        )

        # Verify proper type conversion
        assert isinstance(result.similarity_score, float)
        assert abs(result.similarity_score - float(duckdb_similarity)) < 1e-15
        assert isinstance(result.similarity_percentage, float)
        assert abs(result.similarity_percentage - (float(duckdb_similarity) * 100)) < 1e-10

    def test_batch_results_with_mixed_database_types(self):
        """Test processing batch results with mixed database types."""
        from decimal import Decimal

        from search_result_types import CodeSearchResult, CodeSnippet

        # Simulate a batch of results from database with different types
        batch_data = [
            ("file1.py", "def func1():", Decimal("0.92")),
            ("file2.py", "def func2():", 0.87),  # Some results might be float
            ("file3.py", "class MyClass:", Decimal("0.76")),
            ("file4.py", "import numpy", 0.83),
        ]

        results = []
        for file_path, code_text, similarity in batch_data:
            snippet = CodeSnippet(text=code_text, start_line=1, end_line=1)
            result = CodeSearchResult(file_path=file_path, snippet=snippet, similarity_score=similarity)
            results.append(result)

        # All results should have float similarity_score after initialization
        for result in results:
            assert isinstance(result.similarity_score, float)
            assert isinstance(result.similarity_percentage, float)
            assert 0.0 <= result.similarity_score <= 1.0
            assert 0.0 <= result.similarity_percentage <= 100.0


class TestHybridSearchIntegration:
    """Test hybrid search integration with type safety improvements."""

    def test_hybrid_search_result_ranking_with_decimals(self):
        """Test that ranking works correctly with Decimal similarity scores."""
        from decimal import Decimal

        from search_result_types import CodeSearchResult, CodeSnippet

        # Create results with Decimal scores that need ranking
        results = [
            CodeSearchResult(
                file_path="low_score.py",
                snippet=CodeSnippet(text="low relevance", start_line=1, end_line=1),
                similarity_score=Decimal("0.45"),
            ),
            CodeSearchResult(
                file_path="high_score.py",
                snippet=CodeSnippet(text="high relevance", start_line=1, end_line=1),
                similarity_score=Decimal("0.92"),
            ),
            CodeSearchResult(
                file_path="medium_score.py",
                snippet=CodeSnippet(text="medium relevance", start_line=1, end_line=1),
                similarity_score=Decimal("0.73"),
            ),
        ]

        # Sort by similarity score (should work with converted float values)
        sorted_results = sorted(results, key=lambda r: r.similarity_score, reverse=True)

        assert sorted_results[0].file_path == "high_score.py"
        assert sorted_results[1].file_path == "medium_score.py"
        assert sorted_results[2].file_path == "low_score.py"

        # Verify all scores are properly converted to float
        for result in sorted_results:
            assert isinstance(result.similarity_score, float)

    def test_confidence_level_assignment_with_decimal_scores(self):
        """Test that confidence level assignment works with Decimal scores."""
        from decimal import Decimal

        from search_result_types import CodeSearchResult, CodeSnippet

        snippet = CodeSnippet(text="test", start_line=1, end_line=1)

        # Test different confidence thresholds with Decimal inputs
        test_cases = [
            (Decimal("0.95"), "high"),  # Should be high confidence
            (Decimal("0.75"), "medium"),  # Should be medium confidence
            (Decimal("0.45"), "low"),  # Should be low confidence
        ]

        for decimal_score, expected_confidence in test_cases:
            result = CodeSearchResult(file_path="test.py", snippet=snippet, similarity_score=decimal_score)

            assert result.confidence_level == expected_confidence
            assert isinstance(result.similarity_score, float)


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases for type conversion integration."""

    def test_serialization_with_converted_scores(self):
        """Test that serialization works correctly after type conversion."""
        import json
        from decimal import Decimal

        from search_result_types import CodeSearchResult, CodeSnippet

        snippet = CodeSnippet(text="test serialization", start_line=10, end_line=12)
        result = CodeSearchResult(
            file_path="serialize_test.py",
            snippet=snippet,
            similarity_score=Decimal("0.847"),
            file_metadata={"language": "python"},
        )

        # Convert to dictionary for JSON serialization
        result_dict = result.to_dict()

        # Should be serializable to JSON (no Decimal objects)
        json_str = json.dumps(result_dict)
        assert isinstance(json_str, str)

        # Verify the data is preserved correctly
        parsed = json.loads(json_str)
        assert abs(parsed["similarity_score"] - 0.847) < PRECISION_TOLERANCE_STANDARD
        assert abs(parsed["similarity_percentage"] - 84.7) < PRECISION_TOLERANCE_STANDARD

    def test_legacy_tuple_conversion_with_decimal_input(self):
        """Test legacy tuple conversion handles Decimal similarity scores."""
        from decimal import Decimal

        from search_result_types import CodeSearchResult, CodeSnippet

        snippet = CodeSnippet(text="legacy test", start_line=5, end_line=7)
        result = CodeSearchResult(file_path="legacy.py", snippet=snippet, similarity_score=Decimal("0.83"))

        # Convert to legacy tuple format
        tuple_result = result.to_tuple()

        assert len(tuple_result) == 3
        assert tuple_result[0] == "legacy.py"
        assert tuple_result[1] == "legacy test"
        # Distance should be 1 - similarity = 1 - 0.83 = 0.17
        assert abs(tuple_result[2] - 0.17) < PRECISION_TOLERANCE_STANDARD

    def test_round_trip_legacy_conversion_with_decimals(self):
        """Test round-trip conversion through legacy format preserves data."""
        from decimal import Decimal

        from search_result_types import CodeSearchResult, CodeSnippet

        # Start with Decimal similarity score
        original_snippet = CodeSnippet(text="round trip test", start_line=1, end_line=3)
        original_result = CodeSearchResult(
            file_path="roundtrip.py", snippet=original_snippet, similarity_score=Decimal("0.756")
        )

        # Convert to legacy tuple and back
        tuple_format = original_result.to_tuple()
        recreated_result = CodeSearchResult.from_tuple(tuple_format)

        # Values should be preserved (within floating point precision)
        assert recreated_result.file_path == original_result.file_path
        assert recreated_result.snippet.text == original_result.snippet.text
        assert abs(recreated_result.similarity_score - original_result.similarity_score) < 1e-10


class TestConcurrencyAndBatchProcessing:
    """Test type conversion under concurrent and batch processing scenarios."""

    def test_batch_processing_performance(self):
        """Test performance of type conversion with large batches."""
        import time
        from decimal import Decimal

        from search_result_types import CodeSearchResult, CodeSnippet

        # Create large batch of results with Decimal scores
        batch_size = 500
        start_time = time.time()

        results = []
        for i in range(batch_size):
            snippet = CodeSnippet(text=f"function_{i}()", start_line=i + 1, end_line=i + 2)
            score = Decimal(str(0.5 + (i % 50) / 100.0))  # Vary scores from 0.5 to 0.99

            result = CodeSearchResult(file_path=f"file_{i}.py", snippet=snippet, similarity_score=score)
            results.append(result)

        processing_time = time.time() - start_time

        # Verify reasonable batch processing performance
        assert processing_time < BATCH_PROCESSING_TIMEOUT  # Should process 500 results quickly
        assert len(results) == batch_size

        # Verify all conversions were successful
        for result in results:
            assert isinstance(result.similarity_score, float)
            assert isinstance(result.similarity_percentage, float)

    def test_memory_usage_with_large_decimal_conversions(self):
        """Test that type conversion doesn't cause memory leaks with many conversions."""
        import gc
        from decimal import Decimal

        from search_result_types import _ensure_float

        # Test many conversions in sequence
        initial_objects = len(gc.get_objects())

        for _ in range(1000):
            decimal_val = Decimal("0.123456789")
            float_result = _ensure_float(decimal_val)
            assert isinstance(float_result, float)

        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())

        # Object count shouldn't grow significantly (allow some tolerance)
        object_growth = final_objects - initial_objects
        assert object_growth < 100  # Reasonable tolerance for test overhead


class TestRegressionSafety:
    """Regression tests to ensure type conversion doesn't break existing functionality."""

    def test_existing_float_workflows_unchanged(self):
        """Test that existing workflows with float scores continue to work."""
        from search_result_types import CodeSearchResult, CodeSnippet

        # Test traditional float-based workflow
        snippet = CodeSnippet(text="traditional workflow", start_line=1, end_line=1)
        result = CodeSearchResult(file_path="traditional.py", snippet=snippet, similarity_score=0.85)  # Regular float

        # All operations should work as before
        assert result.similarity_score == 0.85
        assert result.similarity_percentage == 85.0
        assert isinstance(result.similarity_score, float)
        assert isinstance(result.similarity_percentage, float)

        # Legacy operations should work
        tuple_format = result.to_tuple()
        assert len(tuple_format) == 3
        assert abs(tuple_format[2] - 0.15) < PRECISION_TOLERANCE_STANDARD  # distance = 1 - similarity

    def test_metadata_and_context_preservation(self):
        """Test that metadata and context are preserved during type conversion."""
        from decimal import Decimal

        from search_result_types import CodeSearchResult, CodeSnippet

        snippet = CodeSnippet(
            text="preserved context test",
            start_line=10,
            end_line=15,
            context_before="# Setup section",
            context_after="# Cleanup section",
        )

        metadata = {"language": "python", "size": 2048, "last_modified": "2024-01-15", "complexity": "medium"}

        result = CodeSearchResult(
            file_path="metadata_test.py",
            snippet=snippet,
            similarity_score=Decimal("0.789"),
            file_metadata=metadata,
            repository_context={"repo": "test_repo", "branch": "main"},
        )

        # Verify type conversion occurred
        assert isinstance(result.similarity_score, float)
        assert abs(result.similarity_score - 0.789) < PRECISION_TOLERANCE_HIGH

        # Verify all metadata is preserved
        assert result.file_metadata == metadata
        assert result.repository_context["repo"] == "test_repo"
        assert result.snippet.context_before == "# Setup section"
        assert result.snippet.context_after == "# Cleanup section"
