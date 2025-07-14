#!/usr/bin/env python3
"""
MCP Server for Turboprop - Semantic Code Search and Indexing

This module implements a Model Context Protocol (MCP) server that exposes
turboprop's semantic code search and indexing capabilities as MCP tools.
This allows Claude and other MCP clients to search and index code repositories
using natural language queries.

Tools provided:
- index_repository: Build a searchable index from a code repository
- search_code: Search for code using natural language queries
- get_index_status: Check the current state of the code index
- watch_repository: Start monitoring a repository for changes

The server uses stdio transport for communication with MCP clients.
"""

import os
import asyncio
import threading
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

from mcp.server.fastmcp import FastMCP
from mcp.types import Tool, TextContent
from sentence_transformers import SentenceTransformer

# Import our existing code indexing functionality
from code_index import (
    init_db, scan_repo, embed_and_store, build_full_index, 
    search_index, watch_mode, reindex_all, TABLE_NAME, DB_PATH
)

# Initialize the MCP server
mcp = FastMCP("Turboprop Code Search")

# Global variables for shared resources and configuration
_db_connection = None
_embedder = None
_watcher_thread = None
_config = {
    'repository_path': None,
    'max_file_size_mb': 1.0,
    'debounce_seconds': 5.0,
    'auto_index': False,
    'auto_watch': False
}


def get_db_connection():
    """Get or create the database connection."""
    global _db_connection
    if _db_connection is None:
        _db_connection = init_db()
    return _db_connection


def get_embedder():
    """Get or create the sentence transformer model."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


@mcp.tool()
def index_repository(
    repository_path: str = None,
    max_file_size_mb: float = None
) -> str:
    """
    Index a code repository for semantic search.
    
    This tool scans a Git repository, generates semantic embeddings for all code files,
    and builds a searchable index. The index is persistent and stored in a local database.
    
    Args:
        repository_path: Path to the Git repository to index (uses configured path if not provided)
        max_file_size_mb: Maximum file size in MB to include (uses configured value if not provided)
    
    Returns:
        Status message with the number of files indexed
    """
    try:
        # Use provided path or fall back to configured path
        if repository_path is None:
            repository_path = _config['repository_path']
        
        if repository_path is None:
            return "Error: No repository path specified. Either provide a path or configure one at startup."
        
        repo_path = Path(repository_path).resolve()
        
        if not repo_path.exists():
            return f"Error: Repository path '{repository_path}' does not exist"
        
        if not repo_path.is_dir():
            return f"Error: '{repository_path}' is not a directory"
        
        # Use provided max file size or fall back to configured value
        if max_file_size_mb is None:
            max_file_size_mb = _config['max_file_size_mb']
        
        max_bytes = int(max_file_size_mb * 1024 * 1024)
        con = get_db_connection()
        embedder = get_embedder()
        
        # Scan repository for code files
        files = scan_repo(repo_path, max_bytes)
        
        if not files:
            return f"No code files found in repository '{repository_path}'. Make sure it's a Git repository with code files."
        
        # Generate embeddings and store in database
        embed_and_store(con, embedder, files)
        
        # Build search index
        embedding_count = build_full_index(con)
        
        return f"Successfully indexed {len(files)} files from '{repository_path}'. Index contains {embedding_count} embeddings and is ready for search."
        
    except Exception as e:
        return f"Error indexing repository: {str(e)}"


@mcp.tool()
def search_code(
    query: str,
    max_results: int = 5
) -> str:
    """
    Search for code using natural language queries.
    
    This tool performs semantic search over the indexed code files, finding
    code that matches the meaning of your query rather than just keywords.
    
    Args:
        query: Natural language description of what you're looking for
               (e.g., "JWT authentication", "function to parse JSON", "error handling")
        max_results: Maximum number of results to return (default: 5, max: 20)
    
    Returns:
        Formatted search results with file paths, similarity scores, and code snippets
    """
    try:
        if max_results > 20:
            max_results = 20
        
        con = get_db_connection()
        embedder = get_embedder()
        
        # Check if index exists
        file_count = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
        if file_count == 0:
            return "No index found. Please index a repository first using the index_repository tool."
        
        # Perform semantic search
        results = search_index(con, embedder, query, max_results)
        
        if not results:
            return f"No results found for query: '{query}'. Try different search terms or make sure the repository is indexed."
        
        # Format results
        formatted_results = []
        formatted_results.append(f"Found {len(results)} results for: '{query}'\n")
        
        for i, (path, snippet, distance) in enumerate(results, 1):
            similarity_pct = (1 - distance) * 100
            formatted_results.append(f"{i}. {path}")
            formatted_results.append(f"   Similarity: {similarity_pct:.1f}%")
            formatted_results.append(f"   Preview: {snippet.strip()[:200]}...")
            formatted_results.append("")
        
        return "\n".join(formatted_results)
        
    except Exception as e:
        return f"Error searching code: {str(e)}"


@mcp.tool()
def get_index_status() -> str:
    """
    Get the current status of the code index.
    
    Returns information about the indexed files, database size, and readiness for search.
    
    Returns:
        Status information including file count, database size, and index readiness
    """
    try:
        con = get_db_connection()
        
        # Get file count
        file_count = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
        
        # Get database size
        db_size_mb = 0
        if Path(DB_PATH).exists():
            db_size_mb = Path(DB_PATH).stat().st_size / (1024 * 1024)
        
        # Check if index is ready
        index_ready = file_count > 0
        
        # Check watcher status
        watcher_status = "Running" if _watcher_thread and _watcher_thread.is_alive() else "Not running"
        
        status_info = [
            f"Index Status:",
            f"  Files indexed: {file_count}",
            f"  Database size: {db_size_mb:.2f} MB",
            f"  Search ready: {'Yes' if index_ready else 'No'}",
            f"  Database path: {DB_PATH}",
            f"  Embedding model: all-MiniLM-L6-v2 (384 dimensions)",
            f"  Configured repository: {_config['repository_path'] or 'Not configured'}",
            f"  File watcher: {watcher_status}"
        ]
        
        if file_count == 0:
            status_info.append("\nTo get started, use the index_repository tool to index a code repository.")
        
        return "\n".join(status_info)
        
    except Exception as e:
        return f"Error getting index status: {str(e)}"


@mcp.tool()
def watch_repository(
    repository_path: str,
    max_file_size_mb: float = 1.0,
    debounce_seconds: float = 5.0
) -> str:
    """
    Start monitoring a repository for changes and update the index automatically.
    
    This tool starts a background watcher that monitors the specified repository
    for file changes and incrementally updates the search index. The watcher
    uses debouncing to avoid excessive processing during rapid file changes.
    
    Args:
        repository_path: Path to the Git repository to watch
        max_file_size_mb: Maximum file size in MB to process (default: 1.0)
        debounce_seconds: Seconds to wait before processing changes (default: 5.0)
    
    Returns:
        Status message indicating whether the watcher was started successfully
    """
    try:
        global _watcher_thread
        
        repo_path = Path(repository_path).resolve()
        
        if not repo_path.exists():
            return f"Error: Repository path '{repository_path}' does not exist"
        
        if not repo_path.is_dir():
            return f"Error: '{repository_path}' is not a directory"
        
        # Stop existing watcher if running
        if _watcher_thread and _watcher_thread.is_alive():
            return f"Watcher is already running for a repository. Only one watcher can run at a time."
        
        # Start new watcher in background thread
        def start_watcher():
            try:
                watch_mode(str(repo_path), max_file_size_mb, debounce_seconds)
            except KeyboardInterrupt:
                pass  # Normal shutdown
            except Exception as e:
                print(f"Watcher error: {e}")
        
        _watcher_thread = threading.Thread(target=start_watcher, daemon=True)
        _watcher_thread.start()
        
        return f"Started watching repository '{repository_path}' for changes. Files up to {max_file_size_mb} MB will be processed with {debounce_seconds}s debounce delay."
        
    except Exception as e:
        return f"Error starting repository watcher: {str(e)}"


@mcp.tool()
def list_indexed_files(limit: int = 20) -> str:
    """
    List the files currently in the search index.
    
    Shows the files that have been indexed and are available for search.
    
    Args:
        limit: Maximum number of files to show (default: 20)
    
    Returns:
        List of indexed file paths
    """
    try:
        con = get_db_connection()
        
        # Get file paths from database
        results = con.execute(f"""
            SELECT path, LENGTH(content) as size_bytes 
            FROM {TABLE_NAME} 
            ORDER BY path 
            LIMIT {limit}
        """).fetchall()
        
        if not results:
            return "No files are currently indexed. Use the index_repository tool to index a repository."
        
        formatted_results = [f"Indexed files (showing up to {limit}):"]
        
        for path, size_bytes in results:
            size_kb = size_bytes / 1024
            formatted_results.append(f"  {path} ({size_kb:.1f} KB)")
        
        total_count = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
        if total_count > limit:
            formatted_results.append(f"\n... and {total_count - limit} more files")
        
        return "\n".join(formatted_results)
        
    except Exception as e:
        return f"Error listing indexed files: {str(e)}"


def start_file_watcher():
    """Start the file watcher if configured to do so."""
    global _watcher_thread
    
    if not _config['repository_path'] or not _config['auto_watch']:
        return
    
    repo_path = Path(_config['repository_path']).resolve()
    
    if not repo_path.exists() or not repo_path.is_dir():
        print(f"Warning: Cannot watch repository '{_config['repository_path']}' - path does not exist or is not a directory")
        return
    
    # Stop existing watcher if running
    if _watcher_thread and _watcher_thread.is_alive():
        print("File watcher already running")
        return
    
    # Start new watcher in background thread
    def start_watcher():
        try:
            print(f"Starting file watcher for repository: {repo_path}")
            watch_mode(str(repo_path), _config['max_file_size_mb'], _config['debounce_seconds'])
        except KeyboardInterrupt:
            pass  # Normal shutdown
        except Exception as e:
            print(f"Watcher error: {e}")
    
    _watcher_thread = threading.Thread(target=start_watcher, daemon=True)
    _watcher_thread.start()
    print(f"File watcher started for '{repo_path}' (max: {_config['max_file_size_mb']}MB, debounce: {_config['debounce_seconds']}s)")


def parse_args():
    """Parse command-line arguments for MCP server configuration."""
    parser = argparse.ArgumentParser(
        prog="turboprop-mcp",
        description="Turboprop MCP Server - Semantic code search and indexing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  turboprop-mcp /path/to/repo                    # Index and watch repository
  turboprop-mcp /path/to/repo --max-mb 2.0       # Allow larger files
  turboprop-mcp /path/to/repo --no-auto-index    # Don't auto-index on startup
  turboprop-mcp /path/to/repo --no-auto-watch    # Don't auto-watch for changes
        """
    )
    
    parser.add_argument(
        "repository",
        nargs="?",
        help="Path to the repository to index and watch"
    )
    
    parser.add_argument(
        "--max-mb",
        type=float,
        default=1.0,
        help="Maximum file size in MB to process (default: 1.0)"
    )
    
    parser.add_argument(
        "--debounce-sec",
        type=float,
        default=5.0,
        help="Seconds to wait before processing file changes (default: 5.0)"
    )
    
    parser.add_argument(
        "--auto-index",
        action="store_true",
        default=True,
        help="Automatically index the repository on startup (default: True)"
    )
    
    parser.add_argument(
        "--no-auto-index",
        action="store_false",
        dest="auto_index",
        help="Don't automatically index the repository on startup"
    )
    
    parser.add_argument(
        "--auto-watch",
        action="store_true",
        default=True,
        help="Automatically watch for file changes (default: True)"
    )
    
    parser.add_argument(
        "--no-auto-watch",
        action="store_false",
        dest="auto_watch",
        help="Don't automatically watch for file changes"
    )
    
    return parser.parse_args()


def main():
    """Entry point for the MCP server."""
    global _config
    
    # Parse command-line arguments
    args = parse_args()
    
    # Update configuration with command-line arguments
    if args.repository:
        _config['repository_path'] = str(Path(args.repository).resolve())
    
    _config['max_file_size_mb'] = args.max_mb
    _config['debounce_seconds'] = args.debounce_sec
    _config['auto_index'] = args.auto_index
    _config['auto_watch'] = args.auto_watch
    
    # Print configuration
    print("🚀 Turboprop MCP Server Starting")
    print("=" * 40)
    if _config['repository_path']:
        print(f"📁 Repository: {_config['repository_path']}")
        print(f"📊 Max file size: {_config['max_file_size_mb']} MB")
        print(f"⏱️  Debounce delay: {_config['debounce_seconds']}s")
        print(f"🔍 Auto-index: {'Yes' if _config['auto_index'] else 'No'}")
        print(f"👀 Auto-watch: {'Yes' if _config['auto_watch'] else 'No'}")
        print()
    else:
        print("📁 No repository configured - use tools to specify paths")
        print()
    
    # Auto-index if configured
    if _config['repository_path'] and _config['auto_index']:
        print("🔍 Auto-indexing repository...")
        result = index_repository()
        print(f"✅ {result}")
        print()
    
    # Start file watcher if configured
    if _config['repository_path'] and _config['auto_watch']:
        start_file_watcher()
        print()
    
    print("🎯 MCP Server ready - listening for tool calls...")
    print("=" * 40)
    
    # Run the MCP server
    mcp.run()


if __name__ == "__main__":
    # Run the MCP server
    main()