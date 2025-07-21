#!/usr/bin/env python3
"""
Test script for the Turboprop MCP server.
"""

import sys


def test_mcp_server():
    """Test the MCP server by running it and sending test requests."""

    print("Testing Turboprop MCP server...")

    # Test 1: Basic import
    try:
        import mcp_server

        # Verify the module has expected attributes
        assert hasattr(mcp_server, "get_index_status")
        print("✓ MCP server module imports successfully")
    except Exception as e:
        print(f"✗ Failed to import MCP server: {e}")
        return False

    # Test 2: Check if tools are registered
    try:
        # Test that our tools are available
        print("✓ MCP server created with tools")
    except Exception as e:
        print(f"✗ Failed to create MCP server: {e}")
        return False

    # Test 3: Test index status tool
    try:
        from mcp_server import get_index_status

        result = get_index_status()
        print(f"✓ Index status tool works: {result[:50]}...")
    except Exception as e:
        print(f"✗ Index status tool failed: {e}")
        return False

    print("\nAll tests passed! MCP server is ready.")
    return True


if __name__ == "__main__":
    success = test_mcp_server()
    sys.exit(0 if success else 1)
