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
import sys
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
    search_index, watch_mode, reindex_all, TABLE_NAME, EMBED_MODEL, DIMENSIONS
)

# Initialize the MCP server
mcp = FastMCP("🚀 Turboprop: Semantic Code Search & AI-Powered Discovery")

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
        if _config['repository_path']:
            repo_path = Path(_config['repository_path'])
            _db_connection = init_db(repo_path)
        else:
            # Use current directory as fallback
            _db_connection = init_db(Path.cwd())
    return _db_connection


def get_embedder():
    """Get or create the sentence transformer model."""
    global _embedder
    if _embedder is None:
        # Initialize ML model with MPS compatibility handling
        try:
            _embedder = SentenceTransformer(EMBED_MODEL)
        except Exception as e:
            # Handle MPS/Metal Performance Shaders compatibility issues on Apple Silicon
            if "meta tensor" in str(e) or "to_empty" in str(e):
                print("🔧 Detected MPS compatibility issue, falling back to CPU...", file=sys.stderr)
                import torch
                _embedder = SentenceTransformer(EMBED_MODEL, device='cpu')
            else:
                raise e
    return _embedder


@mcp.tool()
def index_repository(
    repository_path: str = None,
    max_file_size_mb: float = None
) -> str:
    """
    🚀 TURBOPROP: Index a code repository for semantic search
    
    BUILD YOUR SEARCHABLE CODE INDEX! This tool scans any Git repository, generates 
    semantic embeddings for all code files (.py, .js, .ts, .java, .go, .rs, etc.), 
    and builds a lightning-fast searchable index using DuckDB + ML embeddings.
    
    💡 EXAMPLES:
    • index_repository("/path/to/my/project") - Index specific repo
    • index_repository() - Index current configured repo
    • index_repository(max_file_size_mb=5.0) - Allow larger files
    
    🔍 WHAT IT INDEXES:
    • Python (.py), JavaScript (.js/.ts/.tsx), Java (.java)
    • Go (.go), Rust (.rs), C/C++ (.c/.cpp/.h)
    • Config files (.json/.yaml/.yml), HTML/CSS
    • And 20+ more code file types!
    
    Args:
        repository_path: Path to Git repo (optional - uses configured path)
        max_file_size_mb: Max file size in MB (optional - uses configured limit)
    
    Returns:
        Success message with file count and index status
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
        print(f"📂 Scanning for code files (max size: {max_file_size_mb} MB)...", file=sys.stderr)
        files = scan_repo(repo_path, max_bytes)
        print(f"📄 Found {len(files)} code files to process", file=sys.stderr)
        
        if not files:
            return f"No code files found in repository '{repository_path}'. Make sure it's a Git repository with code files."
        
        # Generate embeddings and store in database
        print(f"🔍 Generating embeddings for {len(files)} files...", file=sys.stderr)
        embed_and_store(con, embedder, files)
        
        # Build search index
        print(f"📊 Building search index...", file=sys.stderr)
        embedding_count = build_full_index(con)
        
        print(f"✅ Indexing complete! Processed {len(files)} files with {embedding_count} embeddings.", file=sys.stderr)
        print(f"🎯 Repository '{repository_path}' is ready for semantic search!", file=sys.stderr)
        
        return f"Successfully indexed {len(files)} files from '{repository_path}'. Index contains {embedding_count} embeddings and is ready for search."
        
    except Exception as e:
        return f"Error indexing repository: {str(e)}"


@mcp.tool()
def search_code(
    query: str,
    max_results: int = 5
) -> str:
    """
    🔍 TURBOPROP: Search code using natural language (SEMANTIC SEARCH!)
    
    FIND CODE BY MEANING, NOT JUST KEYWORDS! This performs semantic search over 
    your indexed code files, finding code that matches the INTENT of your query.
    
    🎯 SEARCH EXAMPLES:
    • "JWT authentication" - Find auth-related code
    • "database connection setup" - Find DB initialization
    • "error handling for HTTP requests" - Find error handling patterns
    • "password hashing function" - Find crypto/security code
    • "React component for user profile" - Find UI components
    • "API endpoint for user registration" - Find backend routes
    
    🚀 WHY IT'S AMAZING:
    • Understands CODE MEANING, not just text matching
    • Finds similar patterns across different languages
    • Discovers code you forgot you wrote
    • Perfect for exploring unfamiliar codebases
    
    Args:
        query: Natural language description of what you're looking for
        max_results: Number of results (default: 5, max: 20)
    
    Returns:
        Ranked results with file paths, similarity scores, and code previews
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
    📊 TURBOPROP: Check your code index status and health
    
    GET THE FULL PICTURE! See exactly what's indexed, how much space it's using,
    and whether your search index is ready to rock.
    
    📈 WHAT YOU'LL SEE:
    • Number of files indexed
    • Database size and location
    • Embedding model being used
    • File watcher status
    • Search readiness
    
    💡 USE CASES:
    • Check if indexing completed successfully
    • Monitor database growth over time
    • Verify search is ready before querying
    • Debug indexing issues
    
    Returns:
        Complete status report with all index metrics
    """
    try:
        con = get_db_connection()
        
        # Get file count
        file_count = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
        
        # Get database size
        db_size_mb = 0
        if _config['repository_path']:
            db_path = Path(_config['repository_path']) / "code_index.duckdb"
        else:
            db_path = Path.cwd() / "code_index.duckdb"
        
        if db_path.exists():
            db_size_mb = db_path.stat().st_size / (1024 * 1024)
        
        # Check if index is ready
        index_ready = file_count > 0
        
        # Check watcher status
        watcher_status = "Running" if _watcher_thread and _watcher_thread.is_alive() else "Not running"
        
        status_info = [
            f"Index Status:",
            f"  Files indexed: {file_count}",
            f"  Database size: {db_size_mb:.2f} MB",
            f"  Search ready: {'Yes' if index_ready else 'No'}",
            f"  Database path: {db_path}",
            f"  Embedding model: {EMBED_MODEL} ({DIMENSIONS} dimensions)",
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
    👀 TURBOPROP: Watch repository for changes (LIVE INDEX UPDATES!)
    
    KEEP YOUR INDEX FRESH! This starts a background watcher that monitors your 
    repository for file changes and automatically updates the search index.
    
    ⚡ FEATURES:
    • Real-time file change detection
    • Smart debouncing (waits for editing to finish)
    • Incremental updates (only processes changed files)
    • Background processing (won't block your work)
    
    🎯 PERFECT FOR:
    • Active development (index stays current)
    • Team environments (catches all changes)
    • Long-running projects (set and forget)
    
    ⚙️ SMART DEFAULTS:
    • 5-second debounce (adjustable)
    • 1MB file size limit (adjustable)
    • Handles rapid file changes gracefully
    
    Args:
        repository_path: Path to Git repository to watch
        max_file_size_mb: Max file size to process (default: 1.0)
        debounce_seconds: Wait time before processing (default: 5.0)
    
    Returns:
        Confirmation message with watcher configuration
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
                print(f"Watcher error: {e}", file=sys.stderr)
        
        _watcher_thread = threading.Thread(target=start_watcher, daemon=True)
        _watcher_thread.start()
        
        return f"Started watching repository '{repository_path}' for changes. Files up to {max_file_size_mb} MB will be processed with {debounce_seconds}s debounce delay."
        
    except Exception as e:
        return f"Error starting repository watcher: {str(e)}"


@mcp.tool()
def list_indexed_files(limit: int = 20) -> str:
    """
    📋 TURBOPROP: List all files in your search index
    
    SEE WHAT'S INDEXED! Browse all the files that have been processed and are 
    available for semantic search, with file sizes and paths.
    
    🎯 USEFUL FOR:
    • Verifying specific files were indexed
    • Checking index coverage of your project
    • Finding the largest files in your index
    • Debugging missing files
    
    📊 WHAT YOU'LL GET:
    • File paths (sorted alphabetically)
    • File sizes in KB
    • Total file count
    • Pagination if there are many files
    
    💡 PRO TIP: Use a higher limit for comprehensive project audits!
    
    Args:
        limit: Maximum number of files to show (default: 20)
    
    Returns:
        Formatted list of indexed files with sizes and total count
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
        print(f"Warning: Cannot watch repository '{_config['repository_path']}' - path does not exist or is not a directory", file=sys.stderr)
        return
    
    # Stop existing watcher if running
    if _watcher_thread and _watcher_thread.is_alive():
        print("File watcher already running", file=sys.stderr)
        return
    
    # Start new watcher in background thread
    def start_watcher():
        try:
            print(f"Starting file watcher for repository: {repo_path}", file=sys.stderr)
            watch_mode(str(repo_path), _config['max_file_size_mb'], _config['debounce_seconds'])
        except KeyboardInterrupt:
            pass  # Normal shutdown
        except Exception as e:
            print(f"Watcher error: {e}", file=sys.stderr)
    
    _watcher_thread = threading.Thread(target=start_watcher, daemon=True)
    _watcher_thread.start()
    print(f"File watcher started for '{repo_path}' (max: {_config['max_file_size_mb']}MB, debounce: {_config['debounce_seconds']}s)", file=sys.stderr)


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
    print("🚀 Turboprop MCP Server Starting", file=sys.stderr)
    print("=" * 40, file=sys.stderr)
    print(f"🤖 Model: {EMBED_MODEL} ({DIMENSIONS}D)", file=sys.stderr)
    if _config['repository_path']:
        print(f"📁 Repository: {_config['repository_path']}", file=sys.stderr)
        print(f"📊 Max file size: {_config['max_file_size_mb']} MB", file=sys.stderr)
        print(f"⏱️  Debounce delay: {_config['debounce_seconds']}s", file=sys.stderr)
        print(f"🔍 Auto-index: {'Yes' if _config['auto_index'] else 'No'}", file=sys.stderr)
        print(f"👀 Auto-watch: {'Yes' if _config['auto_watch'] else 'No'}", file=sys.stderr)
        print(file=sys.stderr)
    else:
        print("📁 No repository configured - use tools to specify paths", file=sys.stderr)
        print(file=sys.stderr)
    
    # Auto-index if configured (run in background to avoid MCP timeout)
    if _config['repository_path'] and _config['auto_index']:
        def start_auto_index():
            import time
            print("🔍 Auto-indexing repository...", file=sys.stderr)
            print(f"📁 Scanning repository: {_config['repository_path']}", file=sys.stderr)
            
            start_time = time.time()
            result = index_repository()
            end_time = time.time()
            
            total_time = end_time - start_time
            
            # Extract file count from result message
            import re
            file_count_match = re.search(r'Successfully indexed (\d+) files', result)
            if file_count_match:
                file_count = int(file_count_match.group(1))
                time_per_file = total_time / file_count if file_count > 0 else 0
                print(f"🚀 Hey! We indexed {file_count} files and it took {total_time:.2f}s at {time_per_file:.3f}s per file", file=sys.stderr)
            else:
                print(f"🚀 Hey! Indexing completed in {total_time:.2f}s", file=sys.stderr)
            
            print(f"✅ {result}", file=sys.stderr)
            print(file=sys.stderr)
        
        auto_index_thread = threading.Thread(target=start_auto_index, daemon=True)
        auto_index_thread.start()
        print("🔍 Auto-indexing started in background...", file=sys.stderr)
        print(file=sys.stderr)
    
    # Start file watcher if configured
    if _config['repository_path'] and _config['auto_watch']:
        start_file_watcher()
        print(file=sys.stderr)
    
    print("🎯 MCP Server ready - listening for tool calls...", file=sys.stderr)
    print("=" * 40, file=sys.stderr)
    print(file=sys.stderr)
    print("🔥 AVAILABLE TOOLS (use 'tp:' prefix):", file=sys.stderr)
    print("  • tp:search_code - Find code by meaning (semantic search)", file=sys.stderr)
    print("  • tp:index_repository - Build searchable code index", file=sys.stderr)
    print("  • tp:get_index_status - Check index health & stats", file=sys.stderr)
    print("  • tp:watch_repository - Live index updates", file=sys.stderr)
    print("  • tp:list_indexed_files - Browse indexed files", file=sys.stderr)
    print(file=sys.stderr)
    print("💡 START HERE: 'tp:search_code \"your query\"' or 'tp:get_index_status'", file=sys.stderr)
    print("=" * 40, file=sys.stderr)
    
    # Run the MCP server
    mcp.run()


if __name__ == "__main__":
    # Run the MCP server
    main()