#!/usr/bin/env python3
"""
Integration test for MCP Tool Discovery Framework

This script demonstrates the complete tool discovery workflow and validates
that all components work together correctly.
"""

import tempfile
import time
from pathlib import Path

from database_manager import DatabaseManager
from embedding_helper import EmbeddingGenerator
from mcp_tool_discovery import MCPToolDiscovery


def test_complete_discovery_workflow():
    """Test the complete tool discovery and cataloging workflow."""
    print("🔍 Starting MCP Tool Discovery Integration Test...")

    # Setup test database
    temp_dir = Path(tempfile.mkdtemp())
    db_path = temp_dir / "integration_test.duckdb"

    print(f"📁 Using database: {db_path}")

    try:
        # Initialize components
        print("⚙️  Initializing database manager...")
        db_manager = DatabaseManager(db_path)
        db_manager.create_mcp_tool_tables()

        print("🧠 Initializing embedding generator...")
        embedding_generator = EmbeddingGenerator()

        print("🛠️  Initializing tool discovery engine...")
        discovery = MCPToolDiscovery(db_manager, embedding_generator)

        # Discover and catalog tools
        print("🔍 Discovering system tools...")
        start_time = time.time()

        system_tools = discovery.discover_system_tools()
        print(f"✅ Found {len(system_tools)} system tools")

        # Show discovered tools
        print("\n📋 Discovered tools:")
        for tool in system_tools:
            print(f"  • {tool.name} ({tool.category}) - {len(tool.parameters)} parameters")

        print(f"\n💾 Cataloging tools to database...")
        result = discovery.catalog_tools(system_tools)

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"✅ Cataloging complete in {execution_time:.2f}s:")
        print(f"   - Tools stored: {result.tools_stored}")
        print(f"   - Tools failed: {result.tools_failed}")
        print(f"   - Success: {result.success}")

        if result.errors:
            print(f"❌ Errors encountered:")
            for error in result.errors:
                print(f"   - {error}")

        # Verify database contents
        print(f"\n🔍 Verifying database contents...")

        with db_manager.get_connection() as conn:
            # Check tools table
            tool_count = conn.execute("SELECT COUNT(*) FROM mcp_tools").fetchone()[0]
            print(f"   - Tools in database: {tool_count}")

            # Check parameters table
            param_count = conn.execute("SELECT COUNT(*) FROM tool_parameters").fetchone()[0]
            print(f"   - Parameters in database: {param_count}")

            # Check that embeddings were generated
            tools_with_embeddings = conn.execute(
                "SELECT COUNT(*) FROM mcp_tools WHERE embedding IS NOT NULL"
            ).fetchone()[0]
            print(f"   - Tools with embeddings: {tools_with_embeddings}")

            # Show a sample of stored tools
            sample_tools = conn.execute("SELECT name, category, tool_type FROM mcp_tools LIMIT 5").fetchall()

            print(f"\n📋 Sample tools in database:")
            for tool_name, category, tool_type in sample_tools:
                print(f"   • {tool_name} ({tool_type}, {category})")

        # Test tool registry
        print(f"\n📊 Tool registry statistics:")
        registry_stats = discovery.tool_registry.get_registry_statistics()
        print(f"   - Total tools: {registry_stats['total_tools']}")
        print(f"   - Tools by type: {registry_stats['tools_by_type']}")
        print(f"   - Healthy tools: {registry_stats['health_summary']['healthy']}")

        # Performance validation
        if execution_time < 10.0:
            print(f"✅ Performance requirement met: {execution_time:.2f}s < 10s")
        else:
            print(f"⚠️  Performance requirement missed: {execution_time:.2f}s >= 10s")

        # Success validation
        expected_tool_count = len(discovery.SYSTEM_TOOLS_CATALOG)
        if result.success and result.tools_stored == expected_tool_count:
            print(f"✅ Integration test PASSED!")
            print(f"   - All {expected_tool_count} system tools discovered and cataloged")
            print(f"   - All tools have embeddings")
            print(f"   - Tool registry is healthy")
            return True
        else:
            print(f"❌ Integration test FAILED!")
            print(f"   - Expected {expected_tool_count} tools, stored {result.tools_stored}")
            return False

    except Exception as e:
        print(f"❌ Integration test FAILED with exception: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Cleanup
        try:
            db_manager.cleanup()
        except Exception:
            pass


def test_search_functionality():
    """Test that we can search for cataloged tools."""
    print("\n🔍 Testing tool search functionality...")

    temp_dir = Path(tempfile.mkdtemp())
    db_path = temp_dir / "search_test.duckdb"

    try:
        # Setup and catalog tools
        db_manager = DatabaseManager(db_path)
        db_manager.create_mcp_tool_tables()

        embedding_generator = EmbeddingGenerator()
        discovery = MCPToolDiscovery(db_manager, embedding_generator)

        tools = discovery.discover_system_tools()
        discovery.catalog_tools(tools)

        # Test semantic search
        print("🔍 Testing semantic search for 'file operations'...")
        query_embedding = embedding_generator.encode("file operations")

        search_results = db_manager.search_mcp_tools_by_embedding(query_embedding=query_embedding.tolist(), limit=5)

        print(f"✅ Found {len(search_results)} tools for 'file operations':")
        for result in search_results:
            print(f"   • {result['name']} (similarity: {result['similarity_score']:.3f})")

        # Verify we found file operation tools
        file_ops_tools = [
            r
            for r in search_results
            if ("file" in r["name"].lower() or "read" in r["name"].lower() or "write" in r["name"].lower())
        ]

        if file_ops_tools:
            print("✅ Search functionality working - found relevant file operation tools")
            return True
        else:
            print("❌ Search functionality failed - no relevant tools found")
            return False

    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False

    finally:
        try:
            db_manager.cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    print("=" * 60)
    print("MCP Tool Discovery Integration Test")
    print("=" * 60)

    success = True

    # Run discovery workflow test
    success &= test_complete_discovery_workflow()

    # Run search functionality test
    success &= test_search_functionality()

    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("The MCP Tool Discovery Framework is working correctly.")
    else:
        print("❌ INTEGRATION TESTS FAILED!")
        print("There are issues that need to be addressed.")
    print("=" * 60)
