#!/usr/bin/env python3
"""
Integration Tests for MCP Tool Search with Real Database Operations

This module provides integration tests that verify the MCP tool search functionality
works correctly with actual database operations, testing the complete pipeline
from search to response generation.
"""

import json
import tempfile
from pathlib import Path

from database_manager import DatabaseManager
from embedding_helper import EmbeddingGenerator
from tool_search_mcp_tools import (
    get_representative_tools,
    get_tool_details,
    initialize_search_engines,
    list_tool_categories,
    load_tool_categories,
    search_mcp_tools,
    search_tools_by_capability,
)


class TestMCPToolSearchIntegration:
    """Integration tests for MCP tool search with real database operations."""

    @classmethod
    def setup_class(cls):
        """Set up test database and sample data for all tests."""
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.db_path = cls.temp_dir / "integration_test.duckdb"

        # Initialize database manager
        cls.db_manager = DatabaseManager(cls.db_path)
        cls.db_manager.create_mcp_tool_tables()

        # Initialize embedding generator
        cls.embedding_generator = EmbeddingGenerator()

        # Initialize search engines
        initialize_search_engines(cls.db_manager, cls.embedding_generator)

        # Insert test data
        cls._insert_test_data()

    @classmethod
    def _insert_test_data(cls):
        """Insert test data for integration tests."""
        test_tools = [
            {
                "tool_id": "bash",
                "name": "Bash Command Execution",
                "description": "Execute shell commands with error handling",
                "category": "execution",
                "tool_type": "system",
                "complexity_score": 0.6,
            },
            {
                "tool_id": "read",
                "name": "File Reading",
                "description": "Read file contents with encoding support",
                "category": "file_ops",
                "tool_type": "system",
                "complexity_score": 0.3,
            },
            {
                "tool_id": "search_code",
                "name": "Code Search",
                "description": "Search through code repositories",
                "category": "search",
                "tool_type": "custom",
                "complexity_score": 0.8,
            },
        ]

        test_parameters = [
            {"tool_id": "bash", "parameter_name": "command", "parameter_type": "string", "is_required": True},
            {"tool_id": "bash", "parameter_name": "timeout", "parameter_type": "number", "is_required": False},
            {"tool_id": "read", "parameter_name": "file_path", "parameter_type": "string", "is_required": True},
            {"tool_id": "search_code", "parameter_name": "query", "parameter_type": "string", "is_required": True},
            {"tool_id": "search_code", "parameter_name": "language", "parameter_type": "string", "is_required": False},
        ]

        test_examples = [
            {
                "tool_id": "bash",
                "use_case": "List directory contents",
                "example_call": 'bash(command="ls -la")',
                "expected_output": "Directory listing with file details",
                "effectiveness_score": 0.9,
            },
            {
                "tool_id": "read",
                "use_case": "Read configuration file",
                "example_call": 'read(file_path="config.json")',
                "expected_output": "JSON configuration content",
                "effectiveness_score": 0.8,
            },
        ]

        # Insert test data into database
        with cls.db_manager.get_connection() as conn:
            # Insert tools
            for tool in test_tools:
                conn.execute(
                    """
                    INSERT INTO mcp_tools (tool_id, name, description, category, tool_type, complexity_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        tool["tool_id"],
                        tool["name"],
                        tool["description"],
                        tool["category"],
                        tool["tool_type"],
                        tool["complexity_score"],
                    ),
                )

            # Insert parameters
            for param in test_parameters:
                conn.execute(
                    """
                    INSERT INTO tool_parameters (tool_id, parameter_name, parameter_type, is_required)
                    VALUES (?, ?, ?, ?)
                """,
                    (param["tool_id"], param["parameter_name"], param["parameter_type"], param["is_required"]),
                )

            # Insert examples
            for example in test_examples:
                conn.execute(
                    """
                    INSERT INTO tool_examples (tool_id, use_case, example_call, expected_output, effectiveness_score)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        example["tool_id"],
                        example["use_case"],
                        example["example_call"],
                        example["expected_output"],
                        example["effectiveness_score"],
                    ),
                )

    def test_search_mcp_tools_integration(self):
        """Test search_mcp_tools with real database operations."""
        # Test basic search
        result_json = search_mcp_tools("file operations")
        result = json.loads(result_json)

        assert result["success"] is True
        assert "results" in result
        assert len(result["results"]) > 0
        assert "query" in result
        assert result["query"] == "file operations"

        # Check that file-related tools are found
        tool_names = [tool["name"] for tool in result["results"]]
        assert any("file" in name.lower() or "read" in name.lower() for name in tool_names)

    def test_get_tool_details_integration(self):
        """Test get_tool_details with real database operations."""
        result_json = get_tool_details("bash", include_schema=True, include_examples=True)
        result = json.loads(result_json)

        assert result["success"] is True
        assert result["tool_id"] == "bash"
        assert "tool_details" in result

        # Check tool details structure
        tool_details = result["tool_details"]
        assert tool_details["name"] == "Bash Command Execution"
        assert tool_details["category"] == "execution"
        assert "parameters" in tool_details
        assert len(tool_details["parameters"]) > 0

        # Check parameters include expected ones
        param_names = [p["name"] for p in tool_details["parameters"]]
        assert "command" in param_names
        assert "timeout" in param_names

    def test_list_tool_categories_integration(self):
        """Test list_tool_categories with real database operations."""
        result_json = list_tool_categories()
        result = json.loads(result_json)

        assert result["success"] is True
        assert "categories" in result
        assert len(result["categories"]) > 0

        # Check that our test categories are present
        category_names = [cat["name"] for cat in result["categories"]]
        assert "execution" in category_names
        assert "file_ops" in category_names
        assert "search" in category_names

        # Check category structure
        for category in result["categories"]:
            assert "name" in category
            assert "tool_count" in category
            assert category["tool_count"] > 0

    def test_search_tools_by_capability_integration(self):
        """Test search_tools_by_capability with real database operations."""
        result_json = search_tools_by_capability(
            "command execution", required_parameters=["command"], preferred_complexity="any"
        )
        result = json.loads(result_json)

        assert result["success"] is True
        assert "results" in result
        assert len(result["results"]) > 0

        # Check that bash tool is found (it matches command execution)
        tool_ids = [match["tool"]["tool_id"] for match in result["results"]]
        assert "bash" in tool_ids

    def test_load_tool_categories_integration(self):
        """Test load_tool_categories helper function with real database."""
        categories = load_tool_categories()

        assert len(categories) > 0
        assert all(hasattr(cat, "name") for cat in categories)
        assert all(hasattr(cat, "tool_count") for cat in categories)
        assert all(cat.tool_count > 0 for cat in categories)

    def test_get_representative_tools_integration(self):
        """Test get_representative_tools with real database operations."""
        tools = get_representative_tools("execution", limit=2)

        assert isinstance(tools, list)
        assert len(tools) <= 2
        # bash tool should be in execution category
        if tools:  # Only check if tools exist
            assert "bash" in tools

    def test_database_error_handling_integration(self):
        """Test that database errors are handled gracefully."""
        # Test with non-existent tool
        result_json = get_tool_details("nonexistent_tool")
        result = json.loads(result_json)

        assert result["success"] is False
        assert "error" in result
        assert "not found" in result["error"]["message"].lower()

    def test_parameter_validation_integration(self):
        """Test parameter validation with real scenarios."""
        # Test with invalid parameters
        result_json = search_tools_by_capability("", max_results=1000)  # Empty capability description  # Exceeds limit
        result = json.loads(result_json)

        assert result["success"] is False
        assert "error" in result

    @classmethod
    def teardown_class(cls):
        """Clean up test database."""
        if cls.db_path.exists():
            cls.db_path.unlink()
        if cls.temp_dir.exists():
            cls.temp_dir.rmdir()


if __name__ == "__main__":
    # Run the integration tests
    test_suite = TestMCPToolSearchIntegration()
    test_suite.setup_class()

    try:
        print("🧪 Running MCP Tool Search Integration Tests...")

        test_methods = [
            test_suite.test_search_mcp_tools_integration,
            test_suite.test_get_tool_details_integration,
            test_suite.test_list_tool_categories_integration,
            test_suite.test_search_tools_by_capability_integration,
            test_suite.test_load_tool_categories_integration,
            test_suite.test_get_representative_tools_integration,
            test_suite.test_database_error_handling_integration,
            test_suite.test_parameter_validation_integration,
        ]

        passed = 0
        failed = 0

        for test_method in test_methods:
            try:
                print(f"  Running {test_method.__name__}...")
                test_method()
                print(f"  ✅ {test_method.__name__} passed")
                passed += 1
            except Exception as e:
                print(f"  ❌ {test_method.__name__} failed: {e}")
                failed += 1

        print(f"\n📊 Results: {passed} passed, {failed} failed")

        if failed == 0:
            print("🎉 All integration tests passed!")
        else:
            print(f"⚠️  {failed} integration tests failed")

    finally:
        test_suite.teardown_class()
