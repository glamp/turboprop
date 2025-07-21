#!/usr/bin/env python3
"""
Test module for search result data structures.

This module tests the enhanced data structures that replace simple tuples
in search results, providing structured data for AI reasoning.
"""

from decimal import Decimal

import pytest
from search_result_types import CodeSearchResult, CodeSnippet, SearchMetadata

# Test constants for improved maintainability
PRECISION_TOLERANCE_STANDARD = 1e-10  # standard precision tolerance for float comparisons
PERFORMANCE_TIMEOUT_FAST = 0.1  # seconds - for very fast operations
PERFORMANCE_TIMEOUT_MODERATE = 1.0  # seconds - for moderate performance requirements


class TestCodeSnippet:
    """Test the CodeSnippet dataclass."""

    def test_code_snippet_creation(self):
        """Test basic CodeSnippet creation with all fields."""
        snippet = CodeSnippet(
            text="def hello_world():\n    print('Hello, World!')",
            start_line=10,
            end_line=11,
            context_before="# This is a test function",
            context_after="# End of function",
        )

        assert snippet.text == "def hello_world():\n    print('Hello, World!')"
        assert snippet.start_line == 10
        assert snippet.end_line == 11
        assert snippet.context_before == "# This is a test function"
        assert snippet.context_after == "# End of function"

    def test_code_snippet_minimal_creation(self):
        """Test CodeSnippet creation with minimal required fields."""
        snippet = CodeSnippet(text="print('hello')", start_line=5, end_line=5)

        assert snippet.text == "print('hello')"
        assert snippet.start_line == 5
        assert snippet.end_line == 5
        assert snippet.context_before is None
        assert snippet.context_after is None

    def test_code_snippet_str_representation(self):
        """Test that CodeSnippet has meaningful string representation."""
        snippet = CodeSnippet(text="def test():\n    pass", start_line=1, end_line=2)

        str_repr = str(snippet)
        assert "def test():" in str_repr
        assert "1" in str_repr  # Should include line numbers
        assert "2" in str_repr

    def test_code_snippet_to_dict(self):
        """Test JSON serialization support."""
        snippet = CodeSnippet(text="x = 42", start_line=1, end_line=1, context_before="# Setup", context_after="# Done")

        result = snippet.to_dict()
        expected = {
            "text": "x = 42",
            "start_line": 1,
            "end_line": 1,
            "context_before": "# Setup",
            "context_after": "# Done",
        }
        assert result == expected


class TestCodeSearchResult:
    """Test the CodeSearchResult dataclass."""

    def test_code_search_result_creation(self):
        """Test basic CodeSearchResult creation."""
        snippet = CodeSnippet(
            text="def calculate_area(radius):\n    return 3.14 * radius * radius", start_line=5, end_line=6
        )

        result = CodeSearchResult(
            file_path="/path/to/file.py",
            snippet=snippet,
            similarity_score=0.85,
            file_metadata={"language": "python", "size": 1024, "type": "source", "extension": ".py"},
            confidence_level="high",
        )

        assert result.file_path == "/path/to/file.py"
        assert result.snippet == snippet
        assert result.similarity_score == 0.85
        assert result.file_metadata["language"] == "python"
        assert result.confidence_level == "high"

    def test_code_search_result_relative_path(self):
        """Test relative path calculation."""
        snippet = CodeSnippet(text="test", start_line=1, end_line=1)
        result = CodeSearchResult(file_path="/home/user/project/src/main.py", snippet=snippet, similarity_score=0.9)

        relative = result.get_relative_path("/home/user/project")
        assert relative == "src/main.py"

        # Test when file_path doesn't start with base_path
        relative_unchanged = result.get_relative_path("/different/path")
        assert relative_unchanged == "/home/user/project/src/main.py"

    def test_code_search_result_similarity_percentage(self):
        """Test similarity score conversion to percentage."""
        snippet = CodeSnippet(text="test", start_line=1, end_line=1)
        result = CodeSearchResult(file_path="test.py", snippet=snippet, similarity_score=0.75)

        assert result.similarity_percentage == 75.0

    def test_code_search_result_from_tuple(self):
        """Test backward compatibility with legacy tuple format."""
        # Legacy format: (file_path, snippet_text, distance_score)
        legacy_tuple = ("/path/to/test.py", "def test():\n    pass", 0.3)

        result = CodeSearchResult.from_tuple(legacy_tuple)

        assert result.file_path == "/path/to/test.py"
        assert result.snippet.text == "def test():\n    pass"
        assert result.similarity_score == 0.7  # 1 - 0.3 (distance to similarity)
        assert result.snippet.start_line == 1
        assert result.snippet.end_line == 1

    def test_code_search_result_to_tuple(self):
        """Test conversion back to legacy tuple format."""
        snippet = CodeSnippet(text="print('hello')", start_line=5, end_line=5)
        result = CodeSearchResult(file_path="hello.py", snippet=snippet, similarity_score=0.8)

        tuple_result = result.to_tuple()

        # Test the tuple structure and values with floating point tolerance
        assert len(tuple_result) == 3
        assert tuple_result[0] == "hello.py"
        assert tuple_result[1] == "print('hello')"
        assert abs(tuple_result[2] - 0.2) < PRECISION_TOLERANCE_STANDARD  # distance = 1 - similarity

    def test_code_search_result_str_backward_compatibility(self):
        """Test that string representation matches legacy tuple format for compatibility."""
        snippet = CodeSnippet(text="x = 1", start_line=1, end_line=1)
        result = CodeSearchResult(file_path="test.py", snippet=snippet, similarity_score=0.9)

        # Should be usable in contexts expecting tuple-like behavior
        str_repr = str(result)
        assert "test.py" in str_repr
        assert "x = 1" in str_repr
        # Should show similarity as distance for backward compatibility
        assert "0.1" in str_repr or "10%" in str_repr

    def test_code_search_result_to_dict(self):
        """Test JSON serialization."""
        snippet = CodeSnippet(text="return 42", start_line=3, end_line=3)
        result = CodeSearchResult(
            file_path="calc.py",
            snippet=snippet,
            similarity_score=0.95,
            file_metadata={"language": "python"},
            confidence_level="high",
        )

        result_dict = result.to_dict()

        assert result_dict["file_path"] == "calc.py"
        assert result_dict["similarity_score"] == 0.95
        assert result_dict["snippet"]["text"] == "return 42"
        assert result_dict["file_metadata"]["language"] == "python"
        assert result_dict["confidence_level"] == "high"

    def test_code_search_result_with_decimal_similarity_score(self):
        """Test CodeSearchResult handles Decimal similarity scores from DuckDB."""
        snippet = CodeSnippet(text="test", start_line=1, end_line=1)
        # Simulate DuckDB returning a Decimal
        decimal_score = Decimal("0.8547")

        result = CodeSearchResult(file_path="test.py", snippet=snippet, similarity_score=decimal_score)

        # After __post_init__, similarity_score should be converted to float
        assert isinstance(result.similarity_score, float)
        assert result.similarity_score == 0.8547

        # similarity_percentage should work without TypeError
        percentage = result.similarity_percentage
        assert percentage == 85.47

    def test_code_search_result_similarity_percentage_with_decimal(self):
        """Test similarity_percentage calculation with Decimal input."""
        snippet = CodeSnippet(text="test", start_line=1, end_line=1)

        # Test with various Decimal values
        test_cases = [
            (Decimal("0.75"), 75.0),
            (Decimal("0.0"), 0.0),
            (Decimal("1.0"), 100.0),
            (Decimal("0.123456"), 12.3456),
        ]

        for decimal_score, expected_percentage in test_cases:
            result = CodeSearchResult(file_path="test.py", snippet=snippet, similarity_score=decimal_score)
            assert result.similarity_percentage == expected_percentage

    def test_code_search_result_similarity_percentage_with_float(self):
        """Test similarity_percentage calculation still works with float input."""
        snippet = CodeSnippet(text="test", start_line=1, end_line=1)

        # Test with various float values (backward compatibility)
        test_cases = [
            (0.75, 75.0),
            (0.0, 0.0),
            (1.0, 100.0),
            (0.123456, 12.3456),
        ]

        for float_score, expected_percentage in test_cases:
            result = CodeSearchResult(file_path="test.py", snippet=snippet, similarity_score=float_score)
            assert result.similarity_percentage == expected_percentage


class TestTypeConversion:
    """Test type conversion utilities for Decimal/float handling."""

    @pytest.mark.parametrize(
        "input_value,expected_output",
        [
            # Float inputs (should return unchanged)
            (0.0, 0.0),
            (0.5, 0.5),
            (1.0, 1.0),
            (0.123456789, 0.123456789),
            (0.000001, 0.000001),
            (0.999999, 0.999999),
        ],
    )
    def test_ensure_float_with_float_input(self, input_value, expected_output):
        """Test _ensure_float with valid float inputs."""
        from search_result_types import _ensure_float

        result = _ensure_float(input_value)
        assert result == expected_output
        assert isinstance(result, float)

    @pytest.mark.parametrize(
        "decimal_input,expected_float",
        [
            # Decimal inputs (should convert to float)
            (Decimal("0.0"), 0.0),
            (Decimal("0.5"), 0.5),
            (Decimal("1.0"), 1.0),
            (Decimal("0.123456789"), 0.123456789),
            (Decimal("0.000001"), 0.000001),
            (Decimal("0.999999"), 0.999999),
        ],
    )
    def test_ensure_float_with_decimal_input(self, decimal_input, expected_float):
        """Test _ensure_float with valid Decimal inputs."""
        from search_result_types import _ensure_float

        result = _ensure_float(decimal_input)
        assert result == expected_float
        assert isinstance(result, float)

    def test_ensure_float_with_none_input(self):
        """Test _ensure_float with None input (should raise ValueError)."""
        from search_result_types import _ensure_float

        with pytest.raises(ValueError, match="Similarity score cannot be None"):
            _ensure_float(None)

    @pytest.mark.parametrize(
        "invalid_type",
        [
            "0.5",  # string
            [0.5],  # list
            {"value": 0.5},  # dict
            complex(0.5, 0),  # complex number
        ],
    )
    def test_ensure_float_with_invalid_types(self, invalid_type):
        """Test _ensure_float with invalid input types."""
        from search_result_types import _ensure_float

        with pytest.raises(TypeError, match="Unsupported type for similarity score"):
            _ensure_float(invalid_type)

    @pytest.mark.parametrize(
        "out_of_bounds_value",
        [
            -0.1,  # negative
            -1.0,  # negative
            1.1,  # greater than 1.0
            2.0,  # much greater than 1.0
            float("inf"),  # infinity
            Decimal("-0.1"),  # negative Decimal
            Decimal("1.1"),  # Decimal greater than 1.0
        ],
    )
    def test_ensure_float_out_of_bounds(self, out_of_bounds_value):
        """Test _ensure_float with values outside the valid range [0.0, 1.0]."""
        from search_result_types import _ensure_float

        with pytest.raises(ValueError, match="Similarity score must be between 0.0 and 1.0"):
            _ensure_float(out_of_bounds_value)


class TestSearchMetadata:
    """Test the SearchMetadata dataclass."""

    def test_search_metadata_creation(self):
        """Test basic SearchMetadata creation."""
        metadata = SearchMetadata(
            query="test function",
            total_results=5,
            execution_time=0.123,
            confidence_distribution={"high": 2, "medium": 2, "low": 1},
            search_parameters={"k": 5, "model": "all-MiniLM-L6-v2"},
        )

        assert metadata.query == "test function"
        assert metadata.total_results == 5
        assert metadata.execution_time == 0.123
        assert metadata.confidence_distribution["high"] == 2
        assert metadata.search_parameters["k"] == 5

    def test_search_metadata_minimal(self):
        """Test SearchMetadata with minimal fields."""
        metadata = SearchMetadata(query="simple search", total_results=0)

        assert metadata.query == "simple search"
        assert metadata.total_results == 0
        assert metadata.execution_time is None
        assert metadata.confidence_distribution is None
        assert metadata.search_parameters is None

    def test_search_metadata_to_dict(self):
        """Test SearchMetadata serialization."""
        metadata = SearchMetadata(query="serialize test", total_results=3, execution_time=0.05)

        result = metadata.to_dict()
        expected = {
            "query": "serialize test",
            "total_results": 3,
            "execution_time": 0.05,
            "confidence_distribution": None,
            "search_parameters": None,
        }

        assert result == expected


class TestIntegrationScenarios:
    """Test integrated usage scenarios."""

    def test_full_search_result_workflow(self):
        """Test complete workflow from creation to serialization."""
        # Create a typical search result
        snippet = CodeSnippet(
            text="def authenticate_user(username, password):\n    # Authentication logic here\n    return True",
            start_line=45,
            end_line=47,
            context_before="# User authentication functions",
            context_after="def logout_user():",
        )

        result = CodeSearchResult(
            file_path="/project/auth/login.py",
            snippet=snippet,
            similarity_score=0.92,
            file_metadata={"language": "python", "size": 2048, "type": "source", "extension": ".py"},
            confidence_level="high",
        )

        # Test all operations work together
        assert result.similarity_percentage == 92.0
        assert result.get_relative_path("/project") == "auth/login.py"

        # Test backward compatibility
        tuple_format = result.to_tuple()
        assert isinstance(tuple_format, tuple)
        assert len(tuple_format) == 3

        # Test round-trip through legacy format
        recreated = CodeSearchResult.from_tuple(tuple_format)
        assert recreated.file_path == result.file_path
        assert recreated.snippet.text == result.snippet.text

        # Test serialization
        result_dict = result.to_dict()
        assert "file_path" in result_dict
        assert "snippet" in result_dict
        assert "similarity_score" in result_dict

    def test_metadata_integration(self):
        """Test SearchMetadata integration with results."""
        results = [
            CodeSearchResult(
                file_path="file1.py",
                snippet=CodeSnippet(text="test1", start_line=1, end_line=1),
                similarity_score=0.9,
                confidence_level="high",
            ),
            CodeSearchResult(
                file_path="file2.py",
                snippet=CodeSnippet(text="test2", start_line=1, end_line=1),
                similarity_score=0.7,
                confidence_level="medium",
            ),
        ]

        # Create metadata that describes the search
        metadata = SearchMetadata(
            query="authentication function",
            total_results=len(results),
            execution_time=0.156,
            confidence_distribution={"high": 1, "medium": 1, "low": 0},
        )

        # Ensure metadata accurately describes results
        assert metadata.total_results == 2
        high_confidence_count = sum(1 for r in results if r.confidence_level == "high")
        assert metadata.confidence_distribution["high"] == high_confidence_count

    def test_multi_snippet_support(self):
        """Test CodeSearchResult with multiple snippets."""
        snippet1 = CodeSnippet(text="def function1():", start_line=1, end_line=2)
        snippet2 = CodeSnippet(text="def function2():", start_line=5, end_line=6)
        snippet3 = CodeSnippet(text="class MyClass:", start_line=10, end_line=15)

        # Create result with multiple snippets
        result = CodeSearchResult.from_multi_snippets(
            file_path="/test/file.py", snippets=[snippet1, snippet2, snippet3], similarity_score=0.85
        )

        assert result.file_path == "/test/file.py"
        assert result.snippet == snippet1  # Primary snippet
        assert len(result.additional_snippets) == 2
        assert result.additional_snippets[0] == snippet2
        assert result.additional_snippets[1] == snippet3
        assert len(result.all_snippets) == 3

    def test_add_snippet_method(self):
        """Test adding additional snippets to a search result."""
        primary_snippet = CodeSnippet(text="def main():", start_line=1, end_line=3)
        result = CodeSearchResult(file_path="/test/main.py", snippet=primary_snippet, similarity_score=0.9)

        # Initially no additional snippets
        assert len(result.additional_snippets) == 0
        assert len(result.all_snippets) == 1

        # Add an additional snippet
        additional_snippet = CodeSnippet(text="def helper():", start_line=5, end_line=7)
        result.add_snippet(additional_snippet)

        assert len(result.additional_snippets) == 1
        assert len(result.all_snippets) == 2
        assert result.additional_snippets[0] == additional_snippet

    def test_multi_snippet_to_dict(self):
        """Test dictionary serialization with multiple snippets."""
        snippet1 = CodeSnippet(text="def test1():", start_line=1, end_line=2)
        snippet2 = CodeSnippet(text="def test2():", start_line=4, end_line=5)

        result = CodeSearchResult.from_multi_snippets(
            file_path="/test/multi.py",
            snippets=[snippet1, snippet2],
            similarity_score=0.75,
            file_metadata={"language": "python"},
        )

        result_dict = result.to_dict()

        # Check basic fields
        assert result_dict["file_path"] == "/test/multi.py"
        assert result_dict["similarity_score"] == 0.75
        assert result_dict["file_metadata"] == {"language": "python"}

        # Check snippet fields
        assert "snippet" in result_dict
        assert "additional_snippets" in result_dict
        assert "total_snippets" in result_dict
        assert result_dict["total_snippets"] == 2
        assert len(result_dict["additional_snippets"]) == 1

    def test_multi_snippet_from_empty_list_raises_error(self):
        """Test that creating multi-snippet result from empty list raises error."""
        with pytest.raises(ValueError, match="At least one snippet is required"):
            CodeSearchResult.from_multi_snippets(file_path="/test/empty.py", snippets=[], similarity_score=0.5)


class TestComprehensiveTypeConversion:
    """Comprehensive tests for type conversion utilities as specified in Issue 000042."""

    def test_ensure_float_with_decimal_input_detailed(self):
        """Test _ensure_float utility with comprehensive Decimal input cases."""
        from decimal import Decimal

        from search_result_types import _ensure_float

        # Test with various Decimal values from issue specification
        test_cases = [
            (Decimal("0.75"), 0.75),
            (Decimal("0.0"), 0.0),
            (Decimal("1.0"), 1.0),
            (Decimal("0.123456789"), 0.123456789),
        ]

        for decimal_input, expected_float in test_cases:
            result = _ensure_float(decimal_input)
            assert isinstance(result, float)
            assert abs(result - expected_float) < 1e-10

    def test_ensure_float_with_float_input_detailed(self):
        """Test _ensure_float utility with comprehensive float input cases."""
        from search_result_types import _ensure_float

        test_values = [0.75, 0.0, 1.0, 0.123456789]

        for float_input in test_values:
            result = _ensure_float(float_input)
            assert isinstance(result, float)
            assert result == float_input

    def test_similarity_percentage_with_decimal_score_detailed(self):
        """Test similarity_percentage property with Decimal similarity_score."""
        from decimal import Decimal

        from search_result_types import CodeSearchResult, CodeSnippet

        snippet = CodeSnippet(text="test code", start_line=1, end_line=3)

        # Test with Decimal similarity score
        result = CodeSearchResult(file_path="test.py", snippet=snippet, similarity_score=Decimal("0.756"))

        percentage = result.similarity_percentage
        assert isinstance(percentage, float)
        assert abs(percentage - 75.6) < PRECISION_TOLERANCE_STANDARD

    def test_similarity_percentage_with_float_score_detailed(self):
        """Test similarity_percentage property with float similarity_score."""
        from search_result_types import CodeSearchResult, CodeSnippet

        snippet = CodeSnippet(text="test code", start_line=1, end_line=3)

        # Test with float similarity score
        result = CodeSearchResult(file_path="test.py", snippet=snippet, similarity_score=0.756)

        percentage = result.similarity_percentage
        assert isinstance(percentage, float)
        assert abs(percentage - 75.6) < PRECISION_TOLERANCE_STANDARD

    def test_similarity_percentage_edge_cases_detailed(self):
        """Test similarity_percentage with edge case values."""
        from decimal import Decimal

        from search_result_types import CodeSearchResult, CodeSnippet

        snippet = CodeSnippet(text="test", start_line=1, end_line=1)

        edge_cases = [
            # (input_score, expected_percentage)
            (Decimal("0"), 0.0),
            (Decimal("1"), 100.0),
            (0.0, 0.0),
            (1.0, 100.0),
            (Decimal("0.001"), 0.1),
            (0.999, 99.9),
        ]

        for score, expected in edge_cases:
            result = CodeSearchResult(file_path="test.py", snippet=snippet, similarity_score=score)

            percentage = result.similarity_percentage
            assert isinstance(percentage, float)
            assert abs(percentage - expected) < 1e-10

    def test_post_init_converts_decimal_to_float_detailed(self):
        """Test that __post_init__ converts Decimal similarity_score to float."""
        from decimal import Decimal

        from search_result_types import CodeSearchResult, CodeSnippet

        snippet = CodeSnippet(text="test", start_line=1, end_line=1)

        result = CodeSearchResult(file_path="test.py", snippet=snippet, similarity_score=Decimal("0.85"))

        # After __post_init__, similarity_score should be float
        assert isinstance(result.similarity_score, float)
        assert abs(result.similarity_score - 0.85) < PRECISION_TOLERANCE_STANDARD

    def test_type_conversion_with_invalid_inputs_comprehensive(self):
        """Test type conversion error handling with comprehensive invalid inputs."""
        from search_result_types import _ensure_float

        invalid_inputs = [None, "string", [], {}, complex(1, 2)]

        for invalid_input in invalid_inputs:
            with pytest.raises((TypeError, ValueError, AttributeError)):
                _ensure_float(invalid_input)


class TestPerformanceAndPrecision:
    """Performance and precision tests for type conversion."""

    def test_type_conversion_performance(self):
        """Test that type conversion doesn't significantly impact performance."""
        import time
        from decimal import Decimal

        from search_result_types import _ensure_float

        # Test with large number of conversions
        decimal_values = [Decimal(str(i / 1000.0)) for i in range(1000)]
        float_values = [i / 1000.0 for i in range(1000)]

        # Test Decimal conversion performance
        start = time.time()
        decimal_results = [_ensure_float(val) for val in decimal_values]
        decimal_time = time.time() - start

        # Test float passthrough performance
        start = time.time()
        float_results = [_ensure_float(val) for val in float_values]
        float_time = time.time() - start

        # Verify reasonable performance (specific thresholds depend on requirements)
        assert decimal_time < PERFORMANCE_TIMEOUT_MODERATE  # Should be reasonably fast
        assert float_time < PERFORMANCE_TIMEOUT_FAST  # Should be very fast

        # Verify correctness
        assert len(decimal_results) == 1000
        assert len(float_results) == 1000
        assert all(isinstance(r, float) for r in decimal_results)
        assert all(isinstance(r, float) for r in float_results)

    def test_precision_preservation(self):
        """Test that type conversion preserves precision appropriately."""
        from decimal import Decimal

        from search_result_types import _ensure_float

        # Test precision with various decimal values
        test_cases = [
            "0.123456789012345",  # High precision
            "0.1",  # Simple decimal
            "0.999999999999999",  # Near 1
            "0.000000000000001",  # Very small
        ]

        for decimal_str in test_cases:
            decimal_val = Decimal(decimal_str)
            float_val = _ensure_float(decimal_val)

            # Convert back to verify reasonable precision preservation
            # (within float precision limits)
            assert abs(float(decimal_val) - float_val) < 1e-15


class TestCodeSearchResultPostInitBehavior:
    """Detailed tests for CodeSearchResult __post_init__ behavior."""

    def test_post_init_decimal_conversion_various_values(self):
        """Test __post_init__ conversion with various Decimal values."""
        from decimal import Decimal

        from search_result_types import CodeSearchResult, CodeSnippet

        snippet = CodeSnippet(text="test", start_line=1, end_line=1)

        test_decimals = [Decimal("0.1"), Decimal("0.5"), Decimal("0.999"), Decimal("0.123456789")]

        for decimal_score in test_decimals:
            result = CodeSearchResult(file_path="test.py", snippet=snippet, similarity_score=decimal_score)

            # After __post_init__, similarity_score should be converted to float
            assert isinstance(result.similarity_score, float)
            assert abs(result.similarity_score - float(decimal_score)) < 1e-15

            # similarity_percentage should work correctly
            expected_percentage = float(decimal_score) * 100.0
            assert abs(result.similarity_percentage - expected_percentage) < 1e-10

    def test_post_init_float_passthrough(self):
        """Test __post_init__ with float values (should remain unchanged)."""
        from search_result_types import CodeSearchResult, CodeSnippet

        snippet = CodeSnippet(text="test", start_line=1, end_line=1)

        float_values = [0.1, 0.5, 0.999, 0.123456789]

        for float_score in float_values:
            result = CodeSearchResult(file_path="test.py", snippet=snippet, similarity_score=float_score)

            # Should remain as float with same value
            assert isinstance(result.similarity_score, float)
            assert result.similarity_score == float_score

            # similarity_percentage should work correctly
            expected_percentage = float_score * 100.0
            assert abs(result.similarity_percentage - expected_percentage) < 1e-10
