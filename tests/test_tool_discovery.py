#!/usr/bin/env python3
"""
Test suite for MCP Tool Discovery Framework

This module provides comprehensive testing for the tool discovery system,
covering system tool discovery, metadata extraction, and tool registry management.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from turboprop.database_manager import DatabaseManager
from turboprop.embedding_helper import EmbeddingGenerator


class TestMCPToolDiscovery(unittest.TestCase):
    """Test the MCPToolDiscovery class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_tools.duckdb"
        self.db_manager = DatabaseManager(self.db_path)

        # Create MCP tool tables
        self.db_manager.create_mcp_tool_tables()

        # Mock embedding generator
        import numpy as np

        self.embedding_generator = Mock(spec=EmbeddingGenerator)
        self.embedding_generator.encode.return_value = np.array([0.1] * 384, dtype=np.float32)  # Mock 384-dim embedding

    def tearDown(self):
        """Clean up test fixtures."""
        self.db_manager.cleanup()

    def test_discovery_engine_initialization(self):
        """Test that MCPToolDiscovery initializes correctly."""
        from turboprop.mcp_tool_discovery import MCPToolDiscovery

        discovery = MCPToolDiscovery(self.db_manager, self.embedding_generator)

        self.assertIsNotNone(discovery.db_manager)
        self.assertIsNotNone(discovery.embedding_generator)
        self.assertIsNotNone(discovery.tool_registry)

    def test_discover_system_tools_returns_expected_tools(self):
        """Test that system tool discovery returns expected Claude Code tools."""
        from turboprop.mcp_tool_discovery import MCPToolDiscovery

        discovery = MCPToolDiscovery(self.db_manager, self.embedding_generator)
        tools = discovery.discover_system_tools()

        # Should find at least the major system tools
        tool_names = [tool.name.lower() for tool in tools]
        expected_tools = ["bash", "read", "write", "edit", "multiedit", "grep", "glob"]

        for expected_tool in expected_tools:
            self.assertIn(expected_tool, tool_names, f"Expected tool '{expected_tool}' not found in discovered tools")

        # All tools should be system type
        for tool in tools:
            self.assertEqual(tool.tool_type, "system")
            self.assertEqual(tool.provider, "claude-code")

    def test_catalog_tools_stores_in_database(self):
        """Test that cataloged tools are properly stored in database."""
        from turboprop.mcp_tool_discovery import MCPToolDiscovery

        discovery = MCPToolDiscovery(self.db_manager, self.embedding_generator)
        tools = discovery.discover_system_tools()

        # Catalog the discovered tools
        result = discovery.catalog_tools(tools)

        self.assertTrue(result.success)
        self.assertGreater(result.tools_stored, 0)
        self.assertEqual(result.tools_failed, 0)

        # Verify tools are in database
        with self.db_manager.get_connection() as conn:
            count_result = conn.execute("SELECT COUNT(*) FROM mcp_tools").fetchone()
            self.assertGreater(count_result[0], 0)

    def test_tool_metadata_extraction_completeness(self):
        """Test that tool metadata is extracted completely."""
        from turboprop.mcp_tool_discovery import MCPToolDiscovery

        discovery = MCPToolDiscovery(self.db_manager, self.embedding_generator)
        tools = discovery.discover_system_tools()

        # Check that tools have complete metadata
        for tool in tools:
            self.assertIsNotNone(tool.id)
            self.assertIsNotNone(tool.name)
            self.assertIsNotNone(tool.description)
            self.assertIn(
                tool.category, ["file_ops", "execution", "search", "development", "web", "notebook", "workflow"]
            )
            self.assertTrue(len(tool.parameters) >= 0)  # May have no parameters

    def test_tool_fingerprinting_detects_changes(self):
        """Test that tool fingerprinting can detect when tools change."""
        from turboprop.mcp_tool_discovery import MCPToolDiscovery

        discovery = MCPToolDiscovery(self.db_manager, self.embedding_generator)

        # Create a mock tool definition
        mock_tool_def = {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {"param1": {"type": "string"}},
        }

        fingerprint1 = discovery._generate_tool_fingerprint(mock_tool_def)

        # Change the tool definition
        mock_tool_def["description"] = "A modified test tool"
        fingerprint2 = discovery._generate_tool_fingerprint(mock_tool_def)

        self.assertNotEqual(fingerprint1, fingerprint2)

    def test_discovery_performance_under_10_seconds(self):
        """Test that discovery completes within performance requirement."""
        import time

        from turboprop.mcp_tool_discovery import MCPToolDiscovery

        discovery = MCPToolDiscovery(self.db_manager, self.embedding_generator)

        start_time = time.time()
        tools = discovery.discover_system_tools()
        discovery.catalog_tools(tools)
        end_time = time.time()

        execution_time = end_time - start_time
        self.assertLess(execution_time, 10.0, f"Discovery took {execution_time:.2f}s, expected < 10s")


class TestToolRegistry(unittest.TestCase):
    """Test the ToolRegistry class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def test_tool_registry_initialization(self):
        """Test that ToolRegistry initializes correctly."""
        from turboprop.tool_registry import ToolRegistry

        registry = ToolRegistry()
        self.assertIsNotNone(registry)
        self.assertEqual(len(registry.get_all_tools()), 0)

    def test_tool_registration_and_retrieval(self):
        """Test tool registration and retrieval."""
        from turboprop.mcp_tool_discovery import MCPTool
        from turboprop.tool_registry import ToolRegistry

        registry = ToolRegistry()

        # Create a test tool
        test_tool = MCPTool(
            id="test_tool_1",
            name="Test Tool",
            description="A test tool for unit testing",
            tool_type="system",
            provider="test",
            category="testing",
            parameters=[],
            examples=[],
            metadata={},
        )

        # Register the tool
        registry.register_tool(test_tool)

        # Retrieve the tool
        retrieved_tool = registry.get_tool("test_tool_1")
        self.assertIsNotNone(retrieved_tool)
        self.assertEqual(retrieved_tool.name, "Test Tool")

    def test_tool_health_checking(self):
        """Test tool health checking functionality."""
        from turboprop.mcp_tool_discovery import MCPTool
        from turboprop.tool_registry import ToolRegistry

        registry = ToolRegistry()

        test_tool = MCPTool(
            id="test_tool_1",
            name="Test Tool",
            description="A test tool",
            tool_type="system",
            provider="test",
            category="testing",
            parameters=[],
            examples=[],
            metadata={},
        )

        registry.register_tool(test_tool)

        # Check tool health
        health_status = registry.check_tool_health("test_tool_1")
        self.assertIn("status", health_status)
        self.assertIn("is_healthy", health_status)


class TestToolMetadataExtractor(unittest.TestCase):
    """Test the ToolMetadataExtractor class."""

    def test_metadata_extraction_from_tool_definition(self):
        """Test metadata extraction from tool definitions."""
        from turboprop.tool_metadata_extractor import ToolMetadataExtractor

        extractor = ToolMetadataExtractor()

        # Mock tool definition
        mock_tool_def = {
            "name": "bash",
            "description": "Executes bash commands with timeout and error handling",
            "parameters": {
                "command": {"type": "string", "description": "The command to execute", "required": True},
                "timeout": {"type": "number", "description": "Optional timeout in milliseconds", "required": False},
            },
        }

        metadata = extractor.extract_tool_metadata(mock_tool_def)

        self.assertEqual(metadata.name, "bash")
        self.assertIn("command", metadata.description.lower())
        self.assertEqual(len(metadata.parameters), 2)

        # Check parameter extraction
        command_param = next((p for p in metadata.parameters if p.name == "command"), None)
        self.assertIsNotNone(command_param)
        self.assertTrue(command_param.required)

    def test_category_identification(self):
        """Test automatic category identification."""
        from turboprop.tool_metadata_extractor import ToolMetadataExtractor

        extractor = ToolMetadataExtractor()

        # Test file operations tool
        self.assertEqual(extractor._identify_category("read", "Read file contents"), "file_ops")
        self.assertEqual(extractor._identify_category("write", "Write to file"), "file_ops")

        # Test web tools
        self.assertEqual(extractor._identify_category("webfetch", "Fetch web content"), "web")

        # Test execution tools
        self.assertEqual(extractor._identify_category("bash", "Execute shell command"), "execution")


if __name__ == "__main__":
    unittest.main()
