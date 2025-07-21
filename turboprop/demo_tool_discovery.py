#!/usr/bin/env python3
"""
Demonstration of MCP Tool Discovery Framework

This script shows how to use the MCP Tool Discovery Framework to discover
and catalog Claude Code's built-in system tools.
"""

import json
from pathlib import Path

from .database_manager import DatabaseManager
from .embedding_helper import EmbeddingGenerator
from .mcp_tool_discovery import MCPToolDiscovery


def main():
    """Demonstrate the MCP Tool Discovery Framework."""
    print("🛠️  MCP Tool Discovery Framework Demo")
    print("=" * 50)

    # Use the project's database
    db_path = Path(".turboprop/tool_discovery_demo.duckdb")

    print(f"📁 Database: {db_path}")

    # Initialize components
    print("\n🔧 Initializing components...")
    db_manager = DatabaseManager(db_path)

    # Create MCP tool tables
    db_manager.create_mcp_tool_tables()

    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator()

    # Initialize discovery engine
    discovery = MCPToolDiscovery(db_manager, embedding_generator)

    # Discover and catalog all tools
    print("\n🔍 Running complete discovery and cataloging...")
    results = discovery.discover_and_catalog_all()

    print("\n📊 Discovery Results:")
    print(f"   • System tools found: {results['system_tools_found']}")
    print(f"   • Custom tools found: {results['custom_tools_found']}")
    print(f"   • Total tools discovered: {results['total_tools_discovered']}")
    print(f"   • Execution time: {results['total_execution_time']:.2f}s")

    # Show catalog results
    catalog_result = results["catalog_result"]
    print("\n💾 Cataloging Results:")
    print(f"   • Tools stored: {catalog_result['tools_stored']}")
    print(f"   • Tools failed: {catalog_result['tools_failed']}")
    print(f"   • Success: {catalog_result['success']}")

    # Demonstrate search functionality
    print("\n🔍 Demonstrating semantic search...")

    search_queries = ["file operations", "web scraping", "bash commands", "notebook editing"]

    for query in search_queries:
        print(f"\n🔍 Searching for: '{query}'")
        query_embedding = embedding_generator.encode(query)

        search_results = db_manager.search_mcp_tools_by_embedding(query_embedding=query_embedding.tolist(), limit=3)

        for i, result in enumerate(search_results, 1):
            print(f"   {i}. {result['name']} (similarity: {result['similarity_score']:.3f})")
            print(f"      Category: {result['category']}")
            print(f"      Description: {result['description'][:80]}...")

    # Show tool registry statistics
    print("\n📊 Tool Registry Statistics:")
    stats = discovery.tool_registry.get_registry_statistics()

    print(f"   • Total tools: {stats['total_tools']}")
    print(f"   • Tools by type: {json.dumps(stats['tools_by_type'], indent=6)}")
    print(f"   • Tools by category: {json.dumps(stats['tools_by_category'], indent=6)}")
    print(f"   • Health summary: {json.dumps(stats['health_summary'], indent=6)}")

    # Show database statistics
    print("\n📊 Database Statistics:")
    db_stats = db_manager.get_mcp_tool_statistics()
    print(f"   • Total tools: {db_stats.get('total_tools', 0)}")
    print(f"   • Active tools: {db_stats.get('active_tools', 0)}")
    print(f"   • Total parameters: {db_stats.get('total_parameters', 0)}")
    print(f"   • Tools with embeddings: {db_stats.get('tools_with_embeddings', 0)}")

    # Clean up
    db_manager.cleanup()

    print("\n✅ Demo completed successfully!")
    print("📋 The MCP Tool Discovery Framework has successfully:")
    print(f"   • Discovered all {results['system_tools_found']} Claude Code system tools")
    print("   • Generated semantic embeddings for each tool and parameter")
    print("   • Stored comprehensive metadata in the database")
    print("   • Enabled semantic search functionality")
    print(f"   • Completed in under {results['total_execution_time']:.1f} seconds")


if __name__ == "__main__":
    main()
