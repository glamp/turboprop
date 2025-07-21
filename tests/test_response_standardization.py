#!/usr/bin/env python3
"""
Comprehensive tests for MCP response standardization system.

Tests the standardizer, optimizer, validator, and cache components to ensure
proper integration and functionality.
"""

import json
import time
import unittest
from unittest.mock import Mock, patch

from turboprop.mcp_response_optimizer import MCPResponseOptimizer, ResponseCompressor
from turboprop.mcp_response_standardizer import (
    MCPResponseStandardizer,
    classify_error,
    generate_error_suggestions,
    generate_recovery_options,
    generate_response_id,
    get_relevant_documentation,
    standardize_mcp_tool_response,
)
from turboprop.mcp_response_validator import MCPResponseValidator
from turboprop.tool_search_response_cache import CacheStats, ToolSearchResponseCache


class TestMCPResponseStandardizer(unittest.TestCase):
    """Test the MCPResponseStandardizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.standardizer = MCPResponseStandardizer()
        self.sample_response_data = {
            "results": [
                {"id": "tool1", "name": "Test Tool", "description": "A test tool"},
                {"id": "tool2", "name": "Other Tool", "description": "Another tool"},
            ],
            "total_results": 2,
        }
        self.sample_query_context = {"args": ("test query",), "kwargs": {"max_results": 10}}

    def test_standardize_successful_response(self):
        """Test standardization of successful tool response."""
        result = self.standardizer.standardize_tool_search_response(
            response_data=self.sample_response_data,
            tool_name="search_mcp_tools",
            query_context=self.sample_query_context,
        )

        # Check required fields
        self.assertTrue(result["success"])
        self.assertEqual(result["tool"], "search_mcp_tools")
        self.assertIn("timestamp", result)
        self.assertIn("version", result)
        self.assertIn("response_id", result)

        # Check query context is included
        self.assertEqual(result["query_context"], self.sample_query_context)

        # Check performance metadata
        self.assertIn("performance", result)
        self.assertEqual(result["performance"]["result_count"], 2)
        self.assertFalse(result["performance"]["cached"])  # New response shouldn't be cached

        # Check navigation hints
        self.assertIn("navigation", result)
        self.assertIn("follow_up_suggestions", result["navigation"])
        self.assertIn("related_queries", result["navigation"])
        self.assertIn("improvement_hints", result["navigation"])

    def test_standardize_error_response(self):
        """Test standardization of error responses."""
        error_message = "Tool not found"
        context = "test_tool"
        suggestions = ["Check tool name", "Use search function"]

        result = self.standardizer.standardize_error_response(
            error_message=error_message, tool_name="get_tool_details", context=context, suggestions=suggestions
        )

        # Check error response structure
        self.assertFalse(result["success"])
        self.assertEqual(result["tool"], "get_tool_details")
        self.assertIn("error", result)

        # Check error details
        error = result["error"]
        self.assertEqual(error["message"], error_message)
        self.assertEqual(error["context"], context)
        self.assertEqual(error["suggestions"], suggestions)
        self.assertIn("error_type", error)
        self.assertIn("recovery_options", error)
        self.assertIn("documentation_links", error)

        # Check debug info
        self.assertIn("debug_info", result)
        self.assertIn("tool_version", result["debug_info"])
        self.assertIn("system_state", result["debug_info"])

    def test_generate_follow_up_suggestions(self):
        """Test generation of follow-up suggestions."""
        # Test search_mcp_tools suggestions
        response_data_few = {"total_results": 1}
        suggestions = self.standardizer._generate_follow_up_suggestions(response_data_few, "search_mcp_tools")

        self.assertIn("Try broader search terms or different synonyms", suggestions)
        self.assertIn("Use get_tool_details() for comprehensive information about specific tools", suggestions)

        # Test with many results
        response_data_many = {"total_results": 15}
        suggestions = self.standardizer._generate_follow_up_suggestions(response_data_many, "search_mcp_tools")

        self.assertIn("Use more specific terms to narrow results", suggestions)

    def test_count_results(self):
        """Test result counting logic."""
        # Test with results
        response_with_results = {"results": [1, 2, 3]}
        count = self.standardizer._count_results(response_with_results)
        self.assertEqual(count, 3)

        # Test with recommendations
        response_with_recommendations = {"recommendations": [1, 2]}
        count = self.standardizer._count_results(response_with_recommendations)
        self.assertEqual(count, 2)

        # Test with alternatives
        response_with_alternatives = {"alternatives": [1]}
        count = self.standardizer._count_results(response_with_alternatives)
        self.assertEqual(count, 1)

        # Test with no results
        response_empty = {}
        count = self.standardizer._count_results(response_empty)
        self.assertEqual(count, 0)


class TestStandardizationDecorator(unittest.TestCase):
    """Test the standardize_mcp_tool_response decorator."""

    def test_decorator_successful_response(self):
        """Test decorator on successful function."""

        @standardize_mcp_tool_response
        def test_function(query: str):
            return {
                "query": query,
                "results": [
                    {"id": "tool1", "name": "Tool 1", "description": "First tool", "category": "testing"},
                    {"id": "tool2", "name": "Tool 2", "description": "Second tool", "category": "testing"},
                ],
                "total_results": 2,
            }

        result = test_function("test query")

        # Should be standardized JSON string, not original response
        self.assertIsInstance(result, str)
        result_dict = json.loads(result)

        self.assertIn("success", result_dict)
        self.assertIn("tool", result_dict)
        self.assertIn("timestamp", result_dict)
        self.assertEqual(result_dict["tool"], "test_function")
        self.assertTrue(result_dict["success"])
        self.assertIn("execution_time", result_dict)

    def test_decorator_error_handling(self):
        """Test decorator error handling."""

        @standardize_mcp_tool_response
        def failing_function(query: str):
            raise ValueError("Something went wrong")

        result = failing_function("test query")

        # Should return standardized error response as JSON string
        self.assertIsInstance(result, str)
        result_dict = json.loads(result)

        self.assertFalse(result_dict["success"])
        self.assertEqual(result_dict["tool"], "failing_function")
        self.assertIn("error", result_dict)
        self.assertEqual(result_dict["error"]["message"], "Something went wrong")


class TestMCPResponseOptimizer(unittest.TestCase):
    """Test the MCPResponseOptimizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = MCPResponseOptimizer()
        self.sample_response = {
            "success": True,
            "tool": "search_mcp_tools",
            "results": [
                {"id": f"tool{i}", "name": f"Tool {i}", "description": f"Description {i}"}
                for i in range(15)  # Large result set
            ],
        }

    def test_optimize_response_structure(self):
        """Test response structure optimization."""
        optimized = self.optimizer._optimize_result_structure(self.sample_response)

        # Should have result summary for large result sets
        self.assertIn("result_summary", optimized)
        self.assertEqual(optimized["result_summary"]["total_count"], 15)
        self.assertIn("top_categories", optimized["result_summary"])
        self.assertIn("confidence_summary", optimized["result_summary"])

    def test_progressive_disclosure(self):
        """Test progressive disclosure for complex responses."""
        complex_response = {
            "success": True,
            "tool": "compare_mcp_tools",
            "results": [{"id": f"tool{i}"} for i in range(25)],  # Many results
        }

        optimized = self.optimizer._add_progressive_disclosure(complex_response)

        # Should have summary view for complex responses
        self.assertIn("summary_view", optimized)
        self.assertIn("key_points", optimized["summary_view"])
        self.assertIn("quick_recommendations", optimized["summary_view"])
        self.assertIn("next_steps", optimized["summary_view"])

        # Should mark detailed sections
        self.assertIn("detailed_sections", optimized)
        self.assertTrue(optimized["detailed_sections"]["available"])

    def test_metadata_optimization(self):
        """Test metadata optimization."""
        response_with_issues = {"performance": {"execution_time": float("nan"), "result_count": None}}

        optimized = self.optimizer._optimize_metadata(response_with_issues)

        # Should fix problematic values
        self.assertEqual(optimized["performance"]["execution_time"], 0)
        self.assertEqual(optimized["performance"]["result_count"], 0)

        # Should add processing hints
        self.assertIn("processing_hints", optimized)
        self.assertEqual(optimized["processing_hints"]["format_version"], "1.0")

    def test_extract_key_points(self):
        """Test key points extraction."""
        response_with_results = {"results": [{"id": "tool1"}] * 5, "performance": {"cached": True}}

        key_points = self.optimizer._extract_key_points(response_with_results)

        self.assertIn("Found 5 matching tools", key_points)
        self.assertIn("Results retrieved from cache for faster response", key_points)


class TestResponseCompressor(unittest.TestCase):
    """Test the ResponseCompressor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.compressor = ResponseCompressor()

    def test_large_response_detection(self):
        """Test detection of large responses."""
        small_response = {"data": "small"}
        large_response = {"data": "x" * 20000}  # Large response

        self.assertFalse(self.compressor._is_large_response(small_response))
        self.assertTrue(self.compressor._is_large_response(large_response))

    def test_redundant_metadata_removal(self):
        """Test removal of redundant metadata from large responses."""
        large_response = {
            "debug_info": {"detailed_trace": "x" * 60000, "important_field": "keep_this"}  # Very large debug info
        }

        compressed = self.compressor._remove_redundant_metadata(large_response)

        # Should summarize debug info for very large responses
        if "debug_info" in compressed and len(str(large_response)) > 50000:
            self.assertEqual(compressed["debug_info"], {"summarized": True})


class TestMCPResponseValidator(unittest.TestCase):
    """Test the MCPResponseValidator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = MCPResponseValidator()

    def test_validate_successful_response(self):
        """Test validation of successful response."""
        valid_response = {
            "success": True,
            "tool": "search_mcp_tools",
            "timestamp": "2024-07-21 12:00:00 UTC",
            "version": "1.0",
            "response_id": "test123",
            "query": "test query",
            "results": [{"id": "tool1", "name": "Tool 1", "description": "A test tool", "category": "testing"}],
            "total_results": 1,
            "navigation": {
                "follow_up_suggestions": ["Test suggestion"],
                "related_queries": ["Test query"],
                "improvement_hints": ["Test hint"],
            },
            "performance": {"cached": False, "execution_time": 0.5, "result_count": 1},
        }

        validated = self.validator.validate_response(valid_response, "search_mcp_tools")

        # Should pass validation
        self.assertTrue(validated["validation"]["valid"])
        self.assertEqual(validated["validation"]["error_count"], 0)
        self.assertIn("validated_at", validated["validation"])

    def test_validate_missing_required_fields(self):
        """Test validation with missing required fields."""
        invalid_response = {
            "success": True,
            "tool": "search_mcp_tools"
            # Missing timestamp, version, response_id, query, results, total_results
        }

        validated = self.validator.validate_response(invalid_response, "search_mcp_tools")

        # Should fail validation
        self.assertFalse(validated["validation"]["valid"])
        self.assertGreater(validated["validation"]["error_count"], 0)

        errors = validated["validation"]["errors"]
        self.assertTrue(any("timestamp" in error for error in errors))
        self.assertTrue(any("version" in error for error in errors))

    def test_validate_field_types(self):
        """Test validation of field types."""
        response_with_wrong_types = {
            "success": "yes",  # Should be boolean
            "tool": "search_mcp_tools",
            "timestamp": "2024-07-21 12:00:00 UTC",
            "version": "1.0",
            "response_id": "test123",
            "query": "test",
            "results": "not a list",  # Should be list
            "total_results": "5",  # Should be int
        }

        errors = self.validator._validate_field_types(response_with_wrong_types, "search_mcp_tools")

        self.assertTrue(any("success" in error and "bool" in error for error in errors))
        self.assertTrue(any("results" in error and "list" in error for error in errors))
        self.assertTrue(any("total_results" in error and "int" in error for error in errors))

    def test_validate_timestamp_format(self):
        """Test timestamp format validation."""
        # Valid timestamp
        self.assertTrue(self.validator._is_valid_timestamp("2024-07-21 12:00:00 UTC"))

        # Invalid timestamps
        self.assertFalse(self.validator._is_valid_timestamp("2024-07-21"))
        self.assertFalse(self.validator._is_valid_timestamp("invalid"))
        self.assertFalse(self.validator._is_valid_timestamp("2024-07-21T12:00:00Z"))

    def test_validate_version_format(self):
        """Test version format validation."""
        # Valid versions
        self.assertTrue(self.validator._is_valid_version("1.0"))
        self.assertTrue(self.validator._is_valid_version("2.1.3"))

        # Invalid versions
        self.assertFalse(self.validator._is_valid_version("v1.0"))
        self.assertFalse(self.validator._is_valid_version("1.0-beta"))
        self.assertFalse(self.validator._is_valid_version(""))

    def test_json_serialization_validation(self):
        """Test JSON serialization validation."""
        # Valid response
        valid_response = {"data": "test", "number": 42}
        errors = self.validator._validate_json_serialization(valid_response)
        self.assertEqual(len(errors), 0)

        # Response with non-serializable data
        invalid_response = {"data": set([1, 2, 3])}  # Sets are not JSON serializable
        errors = self.validator._validate_json_serialization(invalid_response)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("JSON serialization error" in error for error in errors))

    def test_performance_warnings(self):
        """Test performance-related warnings."""
        large_response = {
            "data": "x" * 60000,  # Large response
            "performance": {"execution_time": 6.0},  # Slow execution
        }

        warnings = self.validator._validate_performance_metrics(large_response)

        self.assertTrue(any("Response size" in warning for warning in warnings))
        self.assertTrue(any("Execution time" in warning and "exceeds" in warning for warning in warnings))


class TestToolSearchResponseCache(unittest.TestCase):
    """Test the ToolSearchResponseCache class."""

    def setUp(self):
        """Set up test fixtures."""
        self.cache = ToolSearchResponseCache(cache_size=5, cache_ttl=60)
        self.sample_response = {"tool": "search_mcp_tools", "results": [{"id": "tool1"}], "timestamp": time.time()}

    def test_cache_set_and_get(self):
        """Test basic cache set and get operations."""
        cache_key = "test_key"

        # Cache miss initially
        result = self.cache.get(cache_key)
        self.assertIsNone(result)

        # Set cache entry
        self.cache.set(cache_key, self.sample_response)

        # Cache hit
        result = self.cache.get(cache_key)
        self.assertIsNotNone(result)
        self.assertEqual(result["tool"], "search_mcp_tools")

    def test_cache_expiration(self):
        """Test cache expiration."""
        cache = ToolSearchResponseCache(cache_size=5, cache_ttl=0.1)  # Very short TTL
        cache_key = "test_key"

        cache.set(cache_key, self.sample_response)

        # Should be available immediately
        result = cache.get(cache_key)
        self.assertIsNotNone(result)

        # Wait for expiration
        time.sleep(0.2)

        # Should be expired
        result = cache.get(cache_key)
        self.assertIsNone(result)

    def test_cache_size_limit(self):
        """Test cache size limits and LRU eviction."""
        cache = ToolSearchResponseCache(cache_size=2)  # Very small cache

        # Fill cache
        cache.set("key1", {"data": "data1", "tool": "tool1"})
        cache.set("key2", {"data": "data2", "tool": "tool2"})

        # Both should be available
        self.assertIsNotNone(cache.get("key1"))
        self.assertIsNotNone(cache.get("key2"))

        # Add third item - should evict oldest
        cache.set("key3", {"data": "data3", "tool": "tool3"})

        # key1 should be evicted
        self.assertIsNone(cache.get("key1"))
        self.assertIsNotNone(cache.get("key2"))
        self.assertIsNotNone(cache.get("key3"))

    def test_cache_stats(self):
        """Test cache statistics."""
        cache_key = "test_key"

        # Initial stats
        stats = self.cache.get_cache_stats()
        self.assertEqual(stats["hit_rate"], 0.0)
        self.assertEqual(stats["total_requests"], 0)

        # Cache miss
        self.cache.get(cache_key)

        # Set and hit
        self.cache.set(cache_key, self.sample_response)
        self.cache.get(cache_key)

        stats = self.cache.get_cache_stats()
        self.assertEqual(stats["hit_rate"], 0.5)  # 1 hit out of 2 requests
        self.assertEqual(stats["total_requests"], 2)

    def test_invalidate_tool_cache(self):
        """Test selective cache invalidation by tool."""
        # Set cache entries for different tools
        self.cache.set("key1", {"tool": "search_mcp_tools", "data": "data1"})
        self.cache.set("key2", {"tool": "search_mcp_tools", "data": "data2"})
        self.cache.set("key3", {"tool": "get_tool_details", "data": "data3"})

        # Invalidate search_mcp_tools cache
        invalidated = self.cache.invalidate_tool_cache("search_mcp_tools")
        self.assertEqual(invalidated, 2)

        # search_mcp_tools entries should be gone
        self.assertIsNone(self.cache.get("key1"))
        self.assertIsNone(self.cache.get("key2"))

        # get_tool_details entry should remain
        self.assertIsNotNone(self.cache.get("key3"))

    def test_cache_health_metrics(self):
        """Test cache health monitoring."""
        # Fill cache with some entries
        self.cache.set("key1", {"tool": "tool1", "data": "data1"})

        health = self.cache.get_cache_health()

        self.assertIn("status", health)
        self.assertIn("total_entries", health)
        self.assertIn("capacity_usage", health)
        self.assertIn("recommendations", health)

        # Should be healthy with low usage
        self.assertEqual(health["status"], "healthy")
        self.assertLess(health["capacity_usage"], 0.5)

    def test_cleanup_expired_entries(self):
        """Test cleanup of expired entries."""
        cache = ToolSearchResponseCache(cache_size=10, cache_ttl=0.1)  # Short TTL

        # Add entries
        cache.set("key1", {"tool": "tool1"})
        cache.set("key2", {"tool": "tool2"})

        # Wait for expiration
        time.sleep(0.2)

        # Cleanup expired entries
        cleaned = cache.cleanup_expired_entries()
        self.assertEqual(cleaned, 2)

        # Cache should be empty
        stats = cache.get_cache_stats()
        self.assertEqual(stats["cache_size"], 0)


class TestCacheStats(unittest.TestCase):
    """Test the CacheStats class."""

    def setUp(self):
        """Set up test fixtures."""
        self.stats = CacheStats()

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        # No requests yet
        self.assertEqual(self.stats.hit_rate(), 0.0)

        # Record some hits and misses
        self.stats.record_hit(0.5)
        self.stats.record_miss()
        self.stats.record_hit(1.0)

        # Should be 2 hits out of 3 total
        self.assertAlmostEqual(self.stats.hit_rate(), 2 / 3, places=2)

    def test_total_requests(self):
        """Test total request counting."""
        self.assertEqual(self.stats.total_requests(), 0)

        self.stats.record_hit()
        self.stats.record_miss()

        self.assertEqual(self.stats.total_requests(), 2)

    def test_average_time_saved(self):
        """Test average time saved calculation."""
        self.assertEqual(self.stats.avg_time_saved(), 0.0)

        self.stats.record_hit(1.0)
        self.stats.record_hit(2.0)

        self.assertEqual(self.stats.avg_time_saved(), 1.5)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def test_generate_response_id(self):
        """Test response ID generation."""
        id1 = generate_response_id()
        id2 = generate_response_id()

        # Should be strings of length 8
        self.assertIsInstance(id1, str)
        self.assertEqual(len(id1), 8)

        # Should be unique (with high probability)
        self.assertNotEqual(id1, id2)

    def test_classify_error(self):
        """Test error classification."""
        self.assertEqual(classify_error("Tool not found"), "NOT_FOUND")
        self.assertEqual(classify_error("Invalid input parameter"), "INVALID_INPUT")
        self.assertEqual(classify_error("Permission denied"), "ACCESS_DENIED")
        self.assertEqual(classify_error("Connection timeout"), "NETWORK_ERROR")
        self.assertEqual(classify_error("Something weird happened"), "UNKNOWN_ERROR")

    def test_generate_error_suggestions(self):
        """Test error suggestion generation."""
        suggestions = generate_error_suggestions("Tool not found", "get_tool_details")

        self.assertIn("Check that the requested resource exists", suggestions)
        self.assertIn("Verify the tool ID exists using search_mcp_tools first", suggestions)

    def test_generate_recovery_options(self):
        """Test recovery option generation."""
        options = generate_recovery_options("Invalid input", "search_mcp_tools")

        self.assertIsInstance(options, list)
        self.assertGreater(len(options), 0)
        self.assertIn("Retry the operation with corrected parameters", options)

    def test_get_relevant_documentation(self):
        """Test documentation link generation."""
        docs = get_relevant_documentation("search_mcp_tools", "Tool not found")

        self.assertIsInstance(docs, list)
        self.assertGreater(len(docs), 0)
        self.assertIn("/docs/mcp-tools/search_mcp_tools", docs)


class TestIntegrationScenarios(unittest.TestCase):
    """Test end-to-end integration scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.standardizer = MCPResponseStandardizer()

    def test_complete_standardization_flow(self):
        """Test complete flow from raw response to standardized output."""
        # Simulate a tool function response
        raw_response = {
            "query": "test search",
            "results": [
                {"id": "bash", "name": "Bash", "description": "Execute shell commands"},
                {"id": "read", "name": "Read", "description": "Read file contents"},
            ],
            "total_results": 2,
            "execution_time": 0.45,
        }

        # Standardize the response
        standardized = self.standardizer.standardize_tool_search_response(
            response_data=raw_response,
            tool_name="search_mcp_tools",
            query_context={"args": ("test search",), "kwargs": {}},
        )

        # Verify complete standardization
        self.assertTrue(standardized["success"])
        self.assertEqual(standardized["tool"], "search_mcp_tools")
        self.assertIn("timestamp", standardized)
        self.assertIn("version", standardized)
        self.assertIn("response_id", standardized)

        # Original data should be preserved
        self.assertEqual(standardized["query"], "test search")
        self.assertEqual(len(standardized["results"]), 2)
        self.assertEqual(standardized["total_results"], 2)

        # Additional metadata should be added
        self.assertIn("performance", standardized)
        self.assertIn("navigation", standardized)
        self.assertIn("validation", standardized)

        # Should be JSON serializable
        json_str = json.dumps(standardized)
        self.assertIsInstance(json_str, str)

    def test_error_handling_integration(self):
        """Test error handling throughout the system."""
        error_response = self.standardizer.standardize_error_response(
            error_message="Database connection failed", tool_name="search_mcp_tools", context="user query: test"
        )

        # Should be properly formatted error response
        self.assertFalse(error_response["success"])
        self.assertEqual(error_response["tool"], "search_mcp_tools")
        self.assertIn("error", error_response)
        self.assertIn("debug_info", error_response)

        # Error should be classified
        self.assertEqual(error_response["error"]["error_type"], "NETWORK_ERROR")

        # Should have helpful suggestions
        self.assertGreater(len(error_response["error"]["suggestions"]), 0)
        self.assertGreater(len(error_response["error"]["recovery_options"]), 0)

        # Should be JSON serializable
        json_str = json.dumps(error_response)
        self.assertIsInstance(json_str, str)

    @patch("turboprop.mcp_response_validator.MCPResponseValidator")
    @patch("turboprop.mcp_response_optimizer.MCPResponseOptimizer")
    @patch("turboprop.tool_search_response_cache.ToolSearchResponseCache")
    def test_component_integration(self, mock_cache, mock_optimizer, mock_validator):
        """Test integration between standardizer components."""
        # Setup mocks
        mock_validator_instance = Mock()
        mock_optimizer_instance = Mock()
        mock_cache_instance = Mock()

        mock_validator.return_value = mock_validator_instance
        mock_optimizer.return_value = mock_optimizer_instance
        mock_cache.return_value = mock_cache_instance

        # Configure mock returns
        mock_validator_instance.validate_response.return_value = {
            "test": "validated",
            "validation": {"valid": True, "errors": []},
        }
        mock_optimizer_instance.optimize_response.return_value = {
            "test": "optimized",
            "validation": {"valid": True, "errors": []},
        }

        # Create standardizer (should initialize components)
        standardizer = MCPResponseStandardizer()

        # Process a response
        result = standardizer.standardize_tool_search_response(response_data={"test": "data"}, tool_name="test_tool")

        # Verify component interactions
        mock_validator_instance.validate_response.assert_called_once()
        mock_optimizer_instance.optimize_response.assert_called_once()

        # Result should be optimized version
        self.assertEqual(result["test"], "optimized")


if __name__ == "__main__":
    unittest.main()
