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

import argparse
import sys
import threading
from pathlib import Path
from typing import Optional, Dict, Any

from mcp.server.fastmcp import FastMCP

# Import our existing code indexing functionality
from code_index import (
    DIMENSIONS,
    EMBED_MODEL,
    TABLE_NAME,
    build_full_index,
    check_index_freshness,
    get_version,
    init_db,
    reindex_all,
    scan_repo,
    search_index,
    watch_mode,
)
from config import config
from database_manager import DatabaseManager
from embedding_helper import EmbeddingGenerator

# Import enhanced search functionality
from search_operations import (
    search_index_enhanced,
    format_enhanced_search_results,
    search_with_comprehensive_response,
    search_hybrid,
    search_with_construct_focus,
    format_hybrid_search_results
)
from construct_search import ConstructSearchOperations, format_construct_search_results
from mcp_response_types import (
    SearchResponse, IndexResponse, StatusResponse,
    create_search_response_from_results
)
from search_result_types import CodeSearchResult
from format_utils import convert_results_to_legacy_format, convert_legacy_to_enhanced_format

# Global lock for database connection management
_db_connection_lock: threading.Lock = threading.Lock()

# Initialize the MCP server
mcp = FastMCP("🚀 Turboprop: Semantic Code Search & AI-Powered Discovery")

# Global variables for shared resources and configuration
_db_connection: Optional[DatabaseManager] = None
_embedder: Optional[EmbeddingGenerator] = None
_watcher_thread: Optional[threading.Thread] = None
_config: Dict[str, Any] = {
    "repository_path": None,
    "max_file_size_mb": config.mcp.DEFAULT_MAX_FILE_SIZE_MB,
    "debounce_seconds": config.mcp.DEFAULT_DEBOUNCE_SECONDS,
    "auto_index": False,
    "auto_watch": False,
    "force_reindex": False,
}


def get_db_connection():
    """Get or create the database connection."""
    global _db_connection
    with _db_connection_lock:
        if _db_connection is None:
            if _config["repository_path"]:
                repo_path = Path(_config["repository_path"])
                _db_connection = init_db(repo_path)
            else:
                # Use current directory as fallback
                _db_connection = init_db(Path.cwd())
        return _db_connection


def get_embedder():
    """Get or create the embedding generator with proper MPS handling."""
    global _embedder
    if _embedder is None:
        # Initialize ML model using our reliable EmbeddingGenerator class
        try:
            _embedder = EmbeddingGenerator(EMBED_MODEL)
            print("✅ Embedding generator initialized successfully", file=sys.stderr)
        except Exception as e:
            print(f"❌ Failed to initialize embedding generator: {e}", file=sys.stderr)
            raise
    return _embedder


@mcp.tool()
def index_repository(
    repository_path: str = None,
    max_file_size_mb: float = None,
    force_all: bool = False,
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
    • ALL Git-tracked files (no extension filtering!)
    • Source code in any language
    • Configuration files (.json/.yaml/.toml/.ini)
    • Documentation (.md/.rst/.txt)
    • Build files, scripts, and any other tracked files

    Args:
        repository_path: Path to Git repo (optional - uses configured path)
        max_file_size_mb: Max file size in MB (optional - uses configured limit)

    Returns:
        Success message with file count and index status
    """
    try:
        # Use provided path or fall back to configured path
        if repository_path is None:
            repository_path = _config["repository_path"]

        if repository_path is None:
            return "Error: No repository path specified. Either provide a path or " "configure one at startup."

        repo_path = Path(repository_path).resolve()

        if not repo_path.exists():
            return f"Error: Repository path '{repository_path}' does not exist"

        if not repo_path.is_dir():
            return f"Error: '{repository_path}' is not a directory"

        # Use provided max file size or fall back to configured value
        if max_file_size_mb is None:
            max_file_size_mb = _config["max_file_size_mb"]

        max_bytes = int(max_file_size_mb * 1024 * 1024)
        con = get_db_connection()
        embedder = get_embedder()

        # Scan repository for code files
        print(
            f"📂 Scanning for code files (max size: {max_file_size_mb} MB)...",
            file=sys.stderr,
        )
        files = scan_repo(repo_path, max_bytes)
        print(f"📄 Found {len(files)} code files to process", file=sys.stderr)

        if not files:
            return (
                f"No code files found in repository '{repository_path}'. "
                "Make sure it's a Git repository with code files."
            )

        # Use the improved reindexing function that handles orphaned files
        print(
            f"🔍 Processing {len(files)} files with smart incremental updates...",
            file=sys.stderr,
        )
        total_files, processed_files, elapsed = reindex_all(
            repo_path, max_bytes, con, embedder, max_workers=None, force_all=force_all
        )

        # Get final embedding count
        embedding_count = build_full_index(con)

        print(
            f"✅ Indexing complete! Processed {len(files)} files with " f"{embedding_count} embeddings.",
            file=sys.stderr,
        )
        print(
            f"🎯 Repository '{repository_path}' is ready for semantic search!",
            file=sys.stderr,
        )

        return (
            f"Successfully indexed {len(files)} files from '{repository_path}'. "
            f"Index contains {embedding_count} embeddings and is ready for search."
        )

    except Exception as e:
        return f"Error indexing repository: {str(e)}"


@mcp.tool()
def search_code(query: str, max_results: int = None) -> str:
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
        if max_results is None:
            max_results = config.search.DEFAULT_MAX_RESULTS
        if max_results > config.search.MAX_RESULTS_LIMIT:
            max_results = config.search.MAX_RESULTS_LIMIT

        con = get_db_connection()
        embedder = get_embedder()

        # Check if index exists
        file_count = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
        if file_count == 0:
            return "No index found. Please index a repository first using the " "index_repository tool."

        # Perform semantic search
        results = search_index(con, embedder, query, max_results)

        if not results:
            return (
                f"No results found for query: '{query}'. "
                "Try different search terms or make sure the repository is indexed."
            )

        # Format results
        formatted_results = []
        formatted_results.append(f"Found {len(results)} results for: '{query}'\n")

        for i, (path, snippet, distance) in enumerate(results, 1):
            similarity_pct = (1 - distance) * 100
            formatted_results.append(f"{i}. {path}")
            formatted_results.append(f"   Similarity: {similarity_pct:.1f}%")
            formatted_results.append(f"   Preview: {snippet.strip()[:config.file_processing.PREVIEW_LENGTH]}" "...")
            formatted_results.append("")

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Error searching code: {str(e)}"


@mcp.tool()
def search_code_structured(query: str, max_results: int = None) -> str:
    """
    🔍 TURBOPROP: Semantic search with comprehensive JSON metadata (STRUCTURED)

    NEXT-GENERATION STRUCTURED SEARCH! Returns rich JSON data that Claude can
    process programmatically, including result clustering, query analysis,
    confidence scoring, and intelligent suggestions.

    🎯 WHAT YOU GET (JSON FORMAT):
    • Complete search results with metadata
    • Result clustering by language and directory
    • Query analysis with complexity assessment
    • Suggested query refinements
    • Cross-references between related files
    • Performance metrics and execution timing
    • Confidence distribution across results
    • Navigation hints for IDE integration

    🚀 PERFECT FOR:
    • AI agents that need structured data
    • Advanced IDE integrations
    • Automated code analysis workflows
    • Building custom search interfaces

    Args:
        query: Natural language description of what you're looking for
        max_results: Number of results (default: 10, max: 20)

    Returns:
        JSON string with comprehensive SearchResponse data
    """
    try:
        if max_results is None:
            max_results = config.search.DEFAULT_MAX_RESULTS
        if max_results > config.search.MAX_RESULTS_LIMIT:
            max_results = config.search.MAX_RESULTS_LIMIT

        con = get_db_connection()
        embedder = get_embedder()

        # Check if index exists
        file_count = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
        if file_count == 0:
            # Return structured error response
            error_response = SearchResponse(
                query=query,
                results=[],
                total_results=0,
                performance_notes=["No index found. Please index a repository first using the index_repository tool."]
            )
            return error_response.to_json()

        # Perform comprehensive structured search
        response = search_with_comprehensive_response(
            db_manager=con,
            embedder=embedder,
            query=query,
            k=max_results,
            include_clusters=True,
            include_suggestions=True,
            include_query_analysis=True
        )

        # Add repository context if available
        repo_path = _config.get("repository_path")
        if repo_path:
            response.navigation_hints.insert(0, f"Repository: {repo_path}")

        return response.to_json()

    except Exception as e:
        # Return structured error response
        error_response = SearchResponse(
            query=query,
            results=[],
            total_results=0,
            performance_notes=[f"Error in structured search: {str(e)}"]
        )
        return error_response.to_json()


@mcp.tool()
def index_repository_structured(
    repository_path: str = None,
    max_file_size_mb: float = None,
    force_all: bool = False,
) -> str:
    """
    🚀 TURBOPROP: Index repository with comprehensive JSON response (STRUCTURED)

    ADVANCED INDEXING WITH DETAILED REPORTING! Returns structured JSON data
    about the indexing operation including file statistics, performance metrics,
    warnings, and recommendations.

    🎯 STRUCTURED DATA INCLUDES:
    • Detailed file processing statistics
    • Performance metrics and timing
    • Database size and embedding counts
    • Warnings and error details
    • Configuration used for indexing
    • Success/failure status with reasons

    Args:
        repository_path: Path to Git repo (optional - uses configured path)
        max_file_size_mb: Max file size in MB (optional - uses configured limit)
        force_all: Force complete reindexing (default: False)

    Returns:
        JSON string with comprehensive IndexResponse data
    """
    import time
    start_time = time.time()

    try:
        # Use provided path or fall back to configured path
        if repository_path is None:
            repository_path = _config["repository_path"]

        if repository_path is None:
            error_response = IndexResponse(
                operation="index",
                status="failed",
                message="No repository path specified",
                execution_time=time.time() - start_time
            )
            error_response.add_error("Either provide a path or configure one at startup")
            return error_response.to_json()

        repo_path = Path(repository_path).resolve()

        if not repo_path.exists():
            error_response = IndexResponse(
                operation="index",
                status="failed",
                message=f"Repository path does not exist",
                repository_path=repository_path,
                execution_time=time.time() - start_time
            )
            error_response.add_error(f"Path '{repository_path}' does not exist")
            return error_response.to_json()

        if not repo_path.is_dir():
            error_response = IndexResponse(
                operation="index",
                status="failed",
                message=f"Path is not a directory",
                repository_path=repository_path,
                execution_time=time.time() - start_time
            )
            error_response.add_error(f"'{repository_path}' is not a directory")
            return error_response.to_json()

        # Use provided max file size or fall back to configured value
        if max_file_size_mb is None:
            max_file_size_mb = _config["max_file_size_mb"]

        max_bytes = int(max_file_size_mb * 1024 * 1024)
        con = get_db_connection()
        embedder = get_embedder()

        # Scan repository for code files
        files = scan_repo(repo_path, max_bytes)

        if not files:
            response = IndexResponse(
                operation="index",
                status="failed",
                message="No code files found in repository",
                repository_path=str(repository_path),
                max_file_size_mb=max_file_size_mb,
                total_files_scanned=0,
                execution_time=time.time() - start_time
            )
            response.add_error("Make sure it's a Git repository with code files")
            return response.to_json()

        # Perform indexing
        total_files, processed_files, elapsed = reindex_all(
            repo_path, max_bytes, con, embedder, max_workers=None, force_all=force_all
        )

        # Get final embedding count
        embedding_count = build_full_index(con)

        # Calculate database size
        db_path = repo_path / ".turboprop" / "code_index.duckdb"
        db_size_mb = 0
        if db_path.exists():
            db_size_mb = db_path.stat().st_size / (1024 * 1024)

        # Create successful response
        execution_time = time.time() - start_time
        response = IndexResponse(
            operation="reindex" if force_all else "index",
            status="success",
            message=f"Successfully indexed {len(files)} files with {embedding_count} embeddings",
            files_processed=processed_files,
            files_skipped=total_files - processed_files if total_files > processed_files else 0,
            total_files_scanned=len(files),
            total_embeddings=embedding_count,
            database_size_mb=db_size_mb,
            execution_time=execution_time,
            repository_path=str(repository_path),
            max_file_size_mb=max_file_size_mb
        )

        # Add performance notes
        if execution_time > 30:
            response.add_warning("Indexing took longer than expected - consider optimizing repository size")
        elif execution_time < 5:
            response.performance_notes = [f"Fast indexing completed in {execution_time:.2f}s"]

        return response.to_json()

    except Exception as e:
        error_response = IndexResponse(
            operation="index",
            status="failed",
            message="Indexing failed with exception",
            repository_path=repository_path,
            max_file_size_mb=max_file_size_mb,
            execution_time=time.time() - start_time
        )
        error_response.add_error(f"Exception: {str(e)}")
        return error_response.to_json()


@mcp.tool()
def get_index_status_structured() -> str:
    """
    📊 TURBOPROP: Comprehensive index status with JSON metadata (STRUCTURED)

    DETAILED HEALTH REPORT! Returns structured JSON data about your code index
    including health metrics, recommendations, file statistics, and freshness analysis.

    🎯 COMPREHENSIVE DATA INCLUDES:
    • Index health score and readiness status
    • Detailed file and embedding statistics
    • Database size and location information
    • File type breakdown and language distribution
    • Freshness analysis and update recommendations
    • Watcher status and configuration details
    • Health recommendations and warnings

    Returns:
        JSON string with comprehensive StatusResponse data
    """
    try:
        con = get_db_connection()

        # Get basic statistics
        file_count = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
        embedding_count = con.execute(
            f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE embedding IS NOT NULL"
        ).fetchone()[0]

        # Get database information
        db_path = None
        db_size_mb = 0
        if _config["repository_path"]:
            db_path = Path(_config["repository_path"]) / ".turboprop" / "code_index.duckdb"
        else:
            db_path = Path.cwd() / ".turboprop" / "code_index.duckdb"

        if db_path.exists():
            db_size_mb = db_path.stat().st_size / (1024 * 1024)

        # Determine status
        is_ready = file_count > 0 and embedding_count > 0
        status = "healthy" if is_ready else ("building" if file_count > 0 else "offline")

        # Get file type statistics
        file_types = {}
        try:
            type_results = con.execute(f"""
                SELECT
                    CASE
                        WHEN path LIKE '%.py' THEN 'Python'
                        WHEN path LIKE '%.js' THEN 'JavaScript'
                        WHEN path LIKE '%.ts' THEN 'TypeScript'
                        WHEN path LIKE '%.java' THEN 'Java'
                        WHEN path LIKE '%.cpp' OR path LIKE '%.c' THEN 'C/C++'
                        WHEN path LIKE '%.go' THEN 'Go'
                        WHEN path LIKE '%.rs' THEN 'Rust'
                        WHEN path LIKE '%.md' THEN 'Markdown'
                        WHEN path LIKE '%.json' THEN 'JSON'
                        WHEN path LIKE '%.yml' OR path LIKE '%.yaml' THEN 'YAML'
                        ELSE 'Other'
                    END as file_type,
                    COUNT(*) as count
                FROM {TABLE_NAME}
                GROUP BY file_type
                ORDER BY count DESC
            """).fetchall()
            file_types = {row[0]: row[1] for row in type_results}
        except Exception:
            file_types = {}

        # Check watcher status
        watcher_active = _watcher_thread and _watcher_thread.is_alive()
        watcher_status = "active" if watcher_active else "inactive"

        # Check freshness if repository is configured
        files_needing_update = 0
        is_fresh = True
        freshness_reason = "Index status unknown"
        last_index_time = None

        if _config["repository_path"]:
            try:
                repo_path = Path(_config["repository_path"])
                max_bytes = int(_config["max_file_size_mb"] * 1024 * 1024)
                freshness = check_index_freshness(repo_path, max_bytes, con)
                is_fresh = freshness["is_fresh"]
                freshness_reason = freshness["reason"]
                files_needing_update = freshness["changed_files"]
                if freshness["last_index_time"]:
                    last_index_time = str(freshness["last_index_time"])
            except Exception as e:
                freshness_reason = f"Freshness check failed: {str(e)}"

        # Create status response
        response = StatusResponse(
            status=status,
            is_ready_for_search=is_ready,
            total_files=file_count,
            files_with_embeddings=embedding_count,
            total_embeddings=embedding_count,
            database_path=str(db_path) if db_path else None,
            database_size_mb=db_size_mb,
            repository_path=_config["repository_path"],
            embedding_model=EMBED_MODEL,
            embedding_dimensions=DIMENSIONS,
            watcher_active=watcher_active,
            watcher_status=watcher_status,
            last_index_time=last_index_time,
            files_needing_update=files_needing_update,
            is_index_fresh=is_fresh,
            freshness_reason=freshness_reason,
            file_types=file_types
        )

        # Add recommendations
        if not is_ready:
            response.add_recommendation("Run index_repository to build the initial index")
        elif not is_fresh:
            response.add_recommendation(f"Run index_repository to update {files_needing_update} changed files")
        elif not watcher_active and _config["repository_path"]:
            response.add_recommendation("Consider starting watch_repository for real-time updates")

        # Add warnings
        if embedding_count < file_count:
            missing_embeddings = file_count - embedding_count
            response.add_warning(f"{missing_embeddings} files lack embeddings - reindexing recommended")

        if db_size_mb > 100:
            response.add_warning(f"Large database size ({db_size_mb:.1f} MB) - consider cleanup")

        return response.to_json()

    except Exception as e:
        error_response = StatusResponse(
            status="error",
            is_ready_for_search=False,
            total_files=0,
            files_with_embeddings=0,
            total_embeddings=0
        )
        error_response.add_warning(f"Status check failed: {str(e)}")
        return error_response.to_json()


@mcp.tool()
def check_index_freshness_tool(repository_path: str = None, max_file_size_mb: float = None) -> str:
    """
    🔍 TURBOPROP: Check if your index is fresh and up-to-date

    SMART INDEX ANALYSIS! This tool checks if your code index is current with the
    actual files in your repository. It analyzes file modification times and counts
    to determine if reindexing is needed.

    🎯 WHAT IT CHECKS:
    • File modification times vs last index update
    • New files that aren't indexed yet
    • Deleted files that are still in the index
    • Total file count changes

    💡 PERFECT FOR:
    • Deciding if you need to reindex
    • Understanding why searches might be outdated
    • Monitoring index health
    • Optimizing development workflow

    Args:
        repository_path: Path to repository (optional - uses configured path)
        max_file_size_mb: Max file size in MB (optional - uses configured limit)

    Returns:
        Detailed freshness report with recommendations
    """
    try:
        # Use provided path or fall back to configured path
        if repository_path is None:
            repository_path = _config["repository_path"]

        if repository_path is None:
            return "Error: No repository path specified. Either provide a path or " "configure one at startup."

        repo_path = Path(repository_path).resolve()

        if not repo_path.exists():
            return f"Error: Repository path '{repository_path}' does not exist"

        # Use provided max file size or fall back to configured value
        if max_file_size_mb is None:
            max_file_size_mb = _config["max_file_size_mb"]

        max_bytes = int(max_file_size_mb * 1024 * 1024)
        con = get_db_connection()

        # Get freshness information
        freshness = check_index_freshness(repo_path, max_bytes, con)

        # Format the report
        report = []
        report.append(f"📊 Index Freshness Report for: {repository_path}")
        report.append("=" * config.search.SEPARATOR_LENGTH)

        if freshness["is_fresh"]:
            report.append("✅ Index Status: UP-TO-DATE")
        else:
            report.append("⚠️  Index Status: NEEDS UPDATE")

        report.append(f"📝 Reason: {freshness['reason']}")
        report.append(f"📁 Total files in repository: {freshness['total_files']}")
        report.append(f"🔄 Files that need updating: {freshness['changed_files']}")

        if freshness["last_index_time"]:
            report.append(f"📅 Last indexed: {freshness['last_index_time']}")
        else:
            report.append("📅 Last indexed: Never")

        report.append("")

        if freshness["is_fresh"]:
            report.append("🎉 Your index is current! No reindexing needed.")
        else:
            if freshness["changed_files"] > 0:
                report.append(
                    f"💡 Recommendation: Run index_repository() to update " f"{freshness['changed_files']} changed files"
                )
            else:
                report.append("💡 Recommendation: Run index_repository() to build the " "initial index")

        return "\n".join(report)

    except Exception as e:
        return f"Error checking index freshness: {str(e)}"


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
        if _config["repository_path"]:
            db_path = Path(_config["repository_path"]) / ".turboprop" / "code_index.duckdb"
        else:
            db_path = Path.cwd() / ".turboprop" / "code_index.duckdb"

        if db_path.exists():
            db_size_mb = db_path.stat().st_size / (1024 * 1024)

        # Check if index is ready
        index_ready = file_count > 0

        # Check watcher status
        watcher_status = "Running" if _watcher_thread and _watcher_thread.is_alive() else "Not running"

        status_info = [
            "Index Status:",
            f"  Files indexed: {file_count}",
            f"  Database size: {db_size_mb:.2f} MB",
            f"  Search ready: {'Yes' if index_ready else 'No'}",
            f"  Database path: {db_path}",
            f"  Embedding model: {EMBED_MODEL} ({DIMENSIONS} dimensions)",
            f"  Configured repository: " f"{_config['repository_path'] or 'Not configured'}",
            f"  File watcher: {watcher_status}",
        ]

        if file_count == 0:
            status_info.append("\nTo get started, use the index_repository tool to index a code " "repository.")

        return "\n".join(status_info)

    except Exception as e:
        return f"Error getting index status: {str(e)}"


@mcp.tool()
def watch_repository(repository_path: str, max_file_size_mb: float = 1.0, debounce_seconds: float = 5.0) -> str:
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
            return "Watcher is already running for a repository. Only one watcher " "can run at a time."

        # Start new watcher in background thread
        def start_watcher():
            """
            File watcher thread function with comprehensive error handling.

            Error scenarios handled:
            - KeyboardInterrupt: Normal shutdown, no action needed
            - FileNotFoundError: Repository path doesn't exist
            - PermissionError: Insufficient permissions to watch directory
            - OSError: Various filesystem errors (disk full, network issues)
            - Exception: Catch-all for unexpected errors

            Security note: Error messages avoid exposing sensitive path details
            to prevent information leakage in logs or user-facing messages.
            """
            try:
                watch_mode(str(repo_path), max_file_size_mb, debounce_seconds)
            except KeyboardInterrupt:
                # Normal shutdown via Ctrl+C or SIGINT - expected behavior
                pass
            except FileNotFoundError:
                # Repository path no longer exists (deleted, unmounted, etc.)
                print("❌ Watcher stopped: Repository path not found", file=sys.stderr)
            except PermissionError:
                # Insufficient permissions to watch directory or files
                print("❌ Watcher stopped: Permission denied for directory access", file=sys.stderr)
            except OSError as e:
                # Filesystem-level errors (disk full, network mount issues, etc.)
                print(f"❌ Watcher stopped: Filesystem error (code {e.errno})", file=sys.stderr)
            except Exception as e:
                # Unexpected errors - log type but not full details for security
                error_type = type(e).__name__
                print(f"❌ Watcher stopped: Unexpected {error_type} error", file=sys.stderr)

        _watcher_thread = threading.Thread(target=start_watcher, daemon=True)
        _watcher_thread.start()

        return (
            f"Started watching repository '{repository_path}' for changes. "
            f"Files up to {max_file_size_mb} MB will be processed with "
            f"{debounce_seconds}s debounce delay."
        )

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
        results = con.execute(
            f"""
            SELECT path, LENGTH(content) as size_bytes
            FROM {TABLE_NAME}
            ORDER BY path
            LIMIT {limit}
        """
        ).fetchall()

        if not results:
            return "No files are currently indexed. Use the index_repository tool " "to index a repository."

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


# Specialized Construct Search Tools

@mcp.tool()
def search_functions(query: str, max_results: int = None) -> str:
    """
    🔧 TURBOPROP: Search functions and methods semantically (CONSTRUCT-LEVEL SEARCH!)
    
    FIND FUNCTIONS BY MEANING! This performs construct-level semantic search specifically
    for functions and methods, providing much more precise results than file-level search.
    
    🎯 SPECIALIZED SEARCH FOR:
    • Function definitions (def, function, async def)
    • Class methods and static methods  
    • Arrow functions and lambda expressions
    • Function signatures and parameters
    • Function docstrings and documentation
    
    💡 EXAMPLES:
    • "password validation function" - Find functions that validate passwords
    • "async database query method" - Find async methods that query databases
    • "error handling function" - Find functions that handle errors
    • "HTTP request handler" - Find functions that handle HTTP requests
    • "data transformation function" - Find functions that transform data
    
    🏆 ADVANTAGES:
    • More precise than file-level search
    • Shows function signatures and docstrings
    • Includes parent class context for methods
    • Filters out non-function code constructs
    • Better relevance ranking for function-specific queries
    
    Args:
        query: Natural language description of the function you're looking for
        max_results: Maximum number of results to return (optional - uses configured default)
        
    Returns:
        Formatted list of matching functions with signatures, locations, and context
    """
    try:
        # Use default max results if not provided
        if max_results is None:
            max_results = config.search.DEFAULT_SEARCH_RESULTS
        
        # Validate max_results
        max_results = max(1, min(max_results, config.search.MAX_SEARCH_RESULTS))
        
        db_manager = get_db_connection()
        embedder = get_embedder()
        
        # Initialize construct search operations  
        construct_ops = ConstructSearchOperations(db_manager, embedder)
        
        # Search for functions and methods specifically
        construct_results = construct_ops.search_functions(query=query, k=max_results)
        
        if not construct_results:
            return f"No function matches found for query: '{query}'"
        
        # Format the results specifically for functions
        formatted_result = format_construct_search_results(
            results=construct_results,
            query=query,
            show_signatures=True,
            show_docstrings=True
        )
        
        return formatted_result
        
    except Exception as e:
        logger.error(f"Error in search_functions for query '{query}': {e}")
        return f"❌ Function search failed for query '{query}': {e}"


@mcp.tool()
def search_classes(query: str, max_results: int = None, include_methods: bool = True) -> str:
    """
    🏗️ TURBOPROP: Search classes semantically (CONSTRUCT-LEVEL SEARCH!)
    
    FIND CLASSES BY MEANING! This performs construct-level semantic search specifically
    for class definitions, providing detailed information about classes and their methods.
    
    🎯 SPECIALIZED SEARCH FOR:
    • Class definitions and declarations
    • Class inheritance and base classes
    • Class docstrings and documentation
    • Class methods and member functions
    • Abstract classes and interfaces
    
    💡 EXAMPLES:
    • "user authentication class" - Find classes that handle user auth
    • "database connection manager" - Find classes that manage DB connections  
    • "HTTP client class" - Find classes for making HTTP requests
    • "data model class" - Find classes that represent data models
    • "exception handling class" - Find custom exception classes
    
    🏆 ADVANTAGES:
    • Shows class signatures with inheritance
    • Includes class docstrings and documentation
    • Lists class methods when requested
    • Better relevance ranking for class-specific queries
    • Provides object-oriented code structure insights
    
    Args:
        query: Natural language description of the class you're looking for
        max_results: Maximum number of results to return (optional - uses configured default)
        include_methods: Whether to show methods of found classes (default: True)
        
    Returns:
        Formatted list of matching classes with signatures, methods, and context
    """
    try:
        # Use default max results if not provided
        if max_results is None:
            max_results = config.search.DEFAULT_SEARCH_RESULTS
        
        # Validate max_results
        max_results = max(1, min(max_results, config.search.MAX_SEARCH_RESULTS))
        
        db_manager = get_db_connection()
        embedder = get_embedder()
        
        # Initialize construct search operations  
        construct_ops = ConstructSearchOperations(db_manager, embedder)
        
        # Search for classes specifically
        class_results = construct_ops.search_classes(query=query, k=max_results)
        
        if not class_results:
            return f"No class matches found for query: '{query}'"
        
        # Format the results specifically for classes
        formatted_result = format_construct_search_results(
            results=class_results,
            query=query,
            show_signatures=True,
            show_docstrings=True
        )
        
        # Add methods for each class if requested
        if include_methods and class_results:
            enhanced_lines = formatted_result.split('\n')
            
            for class_result in class_results:
                try:
                    # Find related methods for this class
                    related_constructs = construct_ops.get_related_constructs(
                        construct_id=class_result.construct_id,
                        k=5  # Limit to top 5 methods per class
                    )
                    
                    methods = [c for c in related_constructs if c.construct_type == 'method']
                    
                    if methods:
                        # Find the position to insert method information
                        class_header = f"   📁 {class_result.file_path}:{class_result.start_line}"
                        try:
                            insert_idx = enhanced_lines.index(class_header) + 1
                            enhanced_lines.insert(insert_idx, f"   🔧 Methods ({len(methods)}):")
                            
                            for method in methods[:3]:  # Show top 3 methods
                                method_line = f"      • {method.name}() - line {method.start_line}"
                                enhanced_lines.insert(insert_idx + 1, method_line)
                                insert_idx += 1
                                
                        except ValueError:
                            # Header not found, skip method insertion for this class
                            continue
                            
                except Exception as e:
                    logger.warning(f"Error adding methods for class {class_result.name}: {e}")
                    continue
            
            formatted_result = '\n'.join(enhanced_lines)
        
        return formatted_result
        
    except Exception as e:
        logger.error(f"Error in search_classes for query '{query}': {e}")
        return f"❌ Class search failed for query '{query}': {e}"


@mcp.tool()
def search_imports(query: str, max_results: int = None) -> str:
    """
    📦 TURBOPROP: Search import statements semantically (CONSTRUCT-LEVEL SEARCH!)
    
    FIND IMPORTS BY MEANING! This performs construct-level semantic search specifically
    for import statements, helping you understand dependencies and module usage.
    
    🎯 SPECIALIZED SEARCH FOR:
    • Import statements (import, from...import)
    • Module and package imports
    • Third-party library imports
    • Relative and absolute imports
    • Import aliases and renaming
    
    💡 EXAMPLES:
    • "database connection import" - Find imports for DB libraries
    • "HTTP request library" - Find imports for HTTP client libraries
    • "JSON parsing imports" - Find imports for JSON handling
    • "testing framework import" - Find imports for test frameworks
    • "async library imports" - Find imports for async/await libraries
    
    🏆 ADVANTAGES:
    • Understand project dependencies
    • Find usage patterns of specific libraries
    • Identify import organization and structure
    • Better relevance ranking for import-specific queries
    • Track how modules are used across the codebase
    
    Args:
        query: Natural language description of the imports you're looking for
        max_results: Maximum number of results to return (optional - uses configured default)
        
    Returns:
        Formatted list of matching import statements with locations and context
    """
    try:
        # Use default max results if not provided
        if max_results is None:
            max_results = config.search.DEFAULT_SEARCH_RESULTS
        
        # Validate max_results
        max_results = max(1, min(max_results, config.search.MAX_SEARCH_RESULTS))
        
        db_manager = get_db_connection()
        embedder = get_embedder()
        
        # Initialize construct search operations  
        construct_ops = ConstructSearchOperations(db_manager, embedder)
        
        # Search for imports specifically
        import_results = construct_ops.search_imports(query=query, k=max_results)
        
        if not import_results:
            return f"No import matches found for query: '{query}'"
        
        # Format the results specifically for imports
        formatted_result = format_construct_search_results(
            results=import_results,
            query=query,
            show_signatures=True,
            show_docstrings=False  # Imports typically don't have docstrings
        )
        
        return formatted_result
        
    except Exception as e:
        logger.error(f"Error in search_imports for query '{query}': {e}")
        return f"❌ Import search failed for query '{query}': {e}"


@mcp.tool()
def search_hybrid_constructs(
    query: str, 
    max_results: int = None,
    construct_weight: float = 0.7,
    file_weight: float = 0.3,
    construct_types: str = None
) -> str:
    """
    🔀 TURBOPROP: Hybrid search combining files and constructs (BEST OF BOTH WORLDS!)
    
    INTELLIGENT HYBRID SEARCH! This combines file-level and construct-level search
    results, intelligently merging and ranking them for comprehensive code discovery.
    
    🎯 HYBRID SEARCH PROVIDES:
    • Best of file-level and construct-level search
    • Intelligent result merging and deduplication
    • Configurable weighting between search types
    • Rich construct context for file results
    • Enhanced relevance ranking
    
    💡 EXAMPLES:
    • "authentication implementation" - Find both auth files and specific functions
    • "database query handling" - Find DB files and specific query functions
    • "error logging system" - Find logging files and specific error functions
    • "API endpoint handlers" - Find API files and specific handler functions
    
    🏆 ADVANTAGES:
    • More comprehensive than single search type
    • Construct matches typically ranked higher for precision
    • File context provided for construct matches
    • Configurable search weighting
    • Better coverage for complex queries
    
    Args:
        query: Natural language search query
        max_results: Maximum number of results to return (optional - uses configured default)
        construct_weight: Weight for construct matches (0.0-1.0, default: 0.7)
        file_weight: Weight for file matches (0.0-1.0, default: 0.3)  
        construct_types: Comma-separated construct types to filter (e.g., "function,class")
        
    Returns:
        Formatted hybrid search results with rich construct context
    """
    try:
        # Use default max results if not provided
        if max_results is None:
            max_results = config.search.DEFAULT_SEARCH_RESULTS
        
        # Validate max_results
        max_results = max(1, min(max_results, config.search.MAX_SEARCH_RESULTS))
        
        # Validate weights
        construct_weight = max(0.0, min(1.0, construct_weight))
        file_weight = max(0.0, min(1.0, file_weight))
        
        # Parse construct types filter
        construct_types_list = None
        if construct_types:
            construct_types_list = [t.strip() for t in construct_types.split(',') if t.strip()]
        
        db_manager = get_db_connection()
        embedder = get_embedder()
        
        # Perform hybrid search
        hybrid_results = search_hybrid(
            db_manager=db_manager,
            embedder=embedder,
            query=query,
            k=max_results,
            construct_weight=construct_weight,
            file_weight=file_weight,
            construct_types=construct_types_list
        )
        
        if not hybrid_results:
            return f"No hybrid search results found for query: '{query}'"
        
        # Format hybrid results with construct context
        formatted_result = format_hybrid_search_results(
            results=hybrid_results,
            query=query,
            show_construct_context=True
        )
        
        return formatted_result
        
    except Exception as e:
        logger.error(f"Error in search_hybrid_constructs for query '{query}': {e}")
        return f"❌ Hybrid search failed for query '{query}': {e}"


@mcp.prompt()
def search(query: str = "") -> str:
    """
    🔍 Quick semantic search for code

    Usage: /mcp__turboprop__search your search query

    Args:
        query: What to search for in the code
    """
    if not query:
        return "Please provide a search query. Example: /mcp__turboprop__search " "JWT authentication"

    result = search_code(query, 3)
    return f"Quick search results for '{query}':\n\n{result}"


@mcp.prompt()
def index_current() -> str:
    """
    📊 Index the current configured repository

    Usage: /mcp__turboprop__index_current
    """
    if not _config["repository_path"]:
        return "No repository configured. Please specify a repository path when " "starting the MCP server."

    result = index_repository()
    return f"Indexing results:\n\n{result}"


@mcp.prompt()
def status() -> str:
    """
    📊 Show current index status

    Usage: /mcp__turboprop__status
    """
    result = get_index_status()
    return f"Current index status:\n\n{result}"


@mcp.prompt()
def files(limit: str = "10") -> str:
    """
    📋 List indexed files

    Usage: /mcp__turboprop__files [limit]

    Args:
        limit: Maximum number of files to show (default: 10)
    """
    try:
        limit_int = int(limit)
    except ValueError:
        limit_int = 10

    result = list_indexed_files(limit_int)
    return f"Indexed files:\n\n{result}"


@mcp.prompt()
def search_by_type(file_type: str, query: str = "") -> str:
    """
    🔍 Search for specific file types

    Usage: /mcp__turboprop__search_by_type python authentication

    Args:
        file_type: File type to search (python, javascript, java, etc.)
        query: Search query
    """
    if not query:
        return (
            f"Please provide both file type and search query. "
            f"Example: /mcp__turboprop__search_by_type python {file_type}"
        )

    # Combine file type and query for more targeted search
    combined_query = f"{file_type} {query}"
    result = search_code(combined_query, 5)
    return f"Search results for '{query}' in {file_type} files:\n\n{result}"


@mcp.prompt()
def help_commands() -> str:
    """
    ❓ Show available Turboprop slash commands

    Usage: /mcp__turboprop__help_commands
    """
    return """🚀 Turboprop Slash Commands:

**Quick Actions:**
• /mcp__turboprop__search <query> - Fast semantic search (3 results)
• /mcp__turboprop__status - Show index status
• /mcp__turboprop__files [limit] - List indexed files

**Advanced Search:**
• /mcp__turboprop__search_by_type <type> <query> - Search specific file types
  Example: /mcp__turboprop__search_by_type python authentication

**Management:**
• /mcp__turboprop__index_current - Reindex current repository
• /mcp__turboprop__help_commands - Show this help

**Examples:**
• /mcp__turboprop__search JWT authentication
• /mcp__turboprop__search_by_type javascript error handling
• /mcp__turboprop__files 20

💡 For more advanced operations, use the full tools:
• tp:search_code - Full semantic search with more options
• tp:index_repository - Index specific repositories
• tp:watch_repository - Start file watching"""


def start_file_watcher():
    """Start the file watcher if configured to do so."""
    global _watcher_thread

    if not _config["repository_path"] or not _config["auto_watch"]:
        return

    repo_path = Path(_config["repository_path"]).resolve()

    if not repo_path.exists() or not repo_path.is_dir():
        print(
            f"Warning: Cannot watch repository '{_config['repository_path']}' - "
            "path does not exist or is not a directory",
            file=sys.stderr,
        )
        return

    # Stop existing watcher if running
    if _watcher_thread and _watcher_thread.is_alive():
        print("File watcher already running", file=sys.stderr)
        return

    # Start new watcher in background thread
    def start_watcher():
        try:
            print(f"Starting file watcher for repository: {repo_path}", file=sys.stderr)
            watch_mode(str(repo_path), _config["max_file_size_mb"], _config["debounce_seconds"])
        except KeyboardInterrupt:
            pass  # Normal shutdown
        except Exception as e:
            print(f"Watcher error: {e}", file=sys.stderr)

    _watcher_thread = threading.Thread(target=start_watcher, daemon=True)
    _watcher_thread.start()
    print(
        f"File watcher started for '{repo_path}' "
        f"(max: {_config['max_file_size_mb']}MB, "
        f"debounce: {_config['debounce_seconds']}s)",
        file=sys.stderr,
    )


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
        """,
    )

    # Add version argument
    parser.add_argument("--version", "-v", action="version", version=f"turboprop-mcp {get_version()}")

    parser.add_argument("repository", nargs="?", help="Path to the repository to index and watch")

    parser.add_argument(
        "--repository",
        dest="repository_flag",
        help=("Path to the repository to index and watch (alternative to " "positional argument)"),
    )

    parser.add_argument(
        "--max-mb",
        type=float,
        default=1.0,
        help="Maximum file size in MB to process (default: 1.0)",
    )

    parser.add_argument(
        "--debounce-sec",
        type=float,
        default=5.0,
        help="Seconds to wait before processing file changes (default: 5.0)",
    )

    parser.add_argument(
        "--auto-index",
        action="store_true",
        default=True,
        help="Automatically index the repository on startup (default: True)",
    )

    parser.add_argument(
        "--no-auto-index",
        action="store_false",
        dest="auto_index",
        help="Don't automatically index the repository on startup",
    )

    parser.add_argument(
        "--auto-watch",
        action="store_true",
        default=True,
        help="Automatically watch for file changes (default: True)",
    )

    parser.add_argument(
        "--no-auto-watch",
        action="store_false",
        dest="auto_watch",
        help="Don't automatically watch for file changes",
    )

    parser.add_argument(
        "--force-reindex",
        action="store_true",
        default=False,
        help="Force complete reindexing, ignoring freshness checks",
    )

    return parser.parse_args()


def main():
    """Entry point for the MCP server."""
    global _config

    # Parse command-line arguments
    args = parse_args()

    # Update configuration with command-line arguments
    # Handle both positional and named repository arguments
    repository_path = None
    if args.repository_flag:
        # --repository flag takes precedence
        repository_path = args.repository_flag
    elif args.repository:
        # Use positional argument
        repository_path = args.repository

    if repository_path:
        _config["repository_path"] = str(Path(repository_path).resolve())

    _config["max_file_size_mb"] = args.max_mb
    _config["debounce_seconds"] = args.debounce_sec
    _config["auto_index"] = args.auto_index
    _config["auto_watch"] = args.auto_watch
    _config["force_reindex"] = args.force_reindex

    # Print configuration
    print("🚀 Turboprop MCP Server Starting", file=sys.stderr)
    print("=" * 40, file=sys.stderr)
    print(f"🤖 Model: {EMBED_MODEL} ({DIMENSIONS}D)", file=sys.stderr)
    if _config["repository_path"]:
        print(f"📁 Repository: {_config['repository_path']}", file=sys.stderr)
        print(f"📊 Max file size: {_config['max_file_size_mb']} MB", file=sys.stderr)
        print(f"⏱️  Debounce delay: {_config['debounce_seconds']}s", file=sys.stderr)
        print(f"🔍 Auto-index: {'Yes' if _config['auto_index'] else 'No'}", file=sys.stderr)
        print(f"👀 Auto-watch: {'Yes' if _config['auto_watch'] else 'No'}", file=sys.stderr)
        print(file=sys.stderr)
    else:
        print("📁 No repository configured - use tools to specify paths", file=sys.stderr)
        print(file=sys.stderr)

    # Smart auto-indexing with freshness checks
    if _config["repository_path"] and _config["auto_index"]:

        def start_smart_auto_index():
            import time

            print("🔍 Checking if repository needs indexing...", file=sys.stderr)

            repo_path = Path(_config["repository_path"])
            max_bytes = int(_config["max_file_size_mb"] * 1024 * 1024)

            # Check index freshness (unless force reindex is requested)
            try:
                con = get_db_connection()
                freshness = check_index_freshness(repo_path, max_bytes, con)

                print(f"📊 Index status: {freshness['reason']}", file=sys.stderr)
                print(f"📊 Total files: {freshness['total_files']}", file=sys.stderr)

                if _config["force_reindex"]:
                    print(
                        "🔄 Force reindexing requested, rebuilding entire index...",
                        file=sys.stderr,
                    )
                elif freshness["is_fresh"]:
                    print("✨ Index is up-to-date, skipping reindexing", file=sys.stderr)
                    print(
                        f"📅 Last indexed: {freshness['last_index_time']}",
                        file=sys.stderr,
                    )
                    return

                # Index needs updating
                if freshness["changed_files"] > 0:
                    print(
                        f"🔄 Found {freshness['changed_files']} changed files, " f"updating index...",
                        file=sys.stderr,
                    )
                else:
                    print(f"🔍 {freshness['reason']}, building index...", file=sys.stderr)

                start_time = time.time()
                result = index_repository(force_all=_config["force_reindex"])
                end_time = time.time()

                total_time = end_time - start_time

                # Extract file count from result message
                import re

                file_count_match = re.search(r"Successfully indexed (\d+) files", result)
                if file_count_match:
                    file_count = int(file_count_match.group(1))
                    if freshness["changed_files"] < file_count:
                        print(
                            f"⚡ Smart indexing: processed {freshness['changed_files']} "
                            f"changed files out of {file_count} total",
                            file=sys.stderr,
                        )
                    time_per_file = total_time / max(1, freshness["changed_files"])
                    print(
                        f"🚀 Indexing completed in {total_time:.2f}s at " f"{time_per_file:.3f}s per file",
                        file=sys.stderr,
                    )
                else:
                    print(f"🚀 Indexing completed in {total_time:.2f}s", file=sys.stderr)

                print(f"✅ {result}", file=sys.stderr)

            except Exception as e:
                print(f"❌ Error during smart indexing: {e}", file=sys.stderr)
                print("🔄 Falling back to standard indexing...", file=sys.stderr)
                result = index_repository()
                print(f"✅ {result}", file=sys.stderr)

            print(file=sys.stderr)

        auto_index_thread = threading.Thread(target=start_smart_auto_index, daemon=True)
        auto_index_thread.start()
        print("🧠 Smart auto-indexing started in background...", file=sys.stderr)
        print(file=sys.stderr)

    # Start file watcher if configured
    if _config["repository_path"] and _config["auto_watch"]:
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
    print("⚡ SLASH COMMANDS (type '/' to see all):", file=sys.stderr)
    print("  • /mcp__turboprop__search <query> - Fast semantic search", file=sys.stderr)
    print("  • /mcp__turboprop__status - Show index status", file=sys.stderr)
    print("  • /mcp__turboprop__files [limit] - List indexed files", file=sys.stderr)
    print("  • /mcp__turboprop__help_commands - Show all slash commands", file=sys.stderr)
    print(file=sys.stderr)
    print(
        "💡 START HERE: '/mcp__turboprop__search \"your query\"' or " "'/mcp__turboprop__status'",
        file=sys.stderr,
    )
    print("=" * 40, file=sys.stderr)

    # Run the MCP server
    mcp.run()


if __name__ == "__main__":
    # Run the MCP server
    main()
