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
import json
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

# Code indexing functionality
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

# Core dependencies - imported first to avoid circular dependencies
from config import config

# Search functionality
from construct_search import ConstructSearchOperations, format_construct_search_results
from database_manager import DatabaseManager
from embedding_helper import EmbeddingGenerator

# Response types
from mcp_response_types import IndexResponse, SearchResponse, StatusResponse
from search_operations import search_hybrid  # Legacy construct hybrid search
from search_operations import search_with_hybrid_fusion  # New semantic+text hybrid search
from search_operations import (
    format_hybrid_search_results,
    search_with_comprehensive_response,
    search_with_intelligent_routing,
)

# Tool engine imports for type hints
from context_analyzer import ContextAnalyzer
from parameter_search_engine import ParameterSearchEngine
from tool_comparison_engine import ToolComparisonEngine
from tool_recommendation_engine import ToolRecommendationEngine

# MCP tool dependencies - imported last and with lazy loading where possible


class MCPToolRegistry:
    """
    Registry for MCP tool functions with lazy loading to prevent circular imports.

    This class acts as a dependency injection container for MCP tool functions,
    loading them only when needed to break potential circular import chains.
    """

    def __init__(self):
        self._tool_recommendation_functions = {}
        self._tool_search_functions = {}
        self._tool_comparison_functions = {}
        self._tool_category_functions = {}
        self._components = {}

    def get_recommendation_function(self, function_name: str):
        """Get a tool recommendation function with lazy loading."""
        if function_name not in self._tool_recommendation_functions:
            from tool_recommendation_mcp_tools import (
                analyze_task_requirements,
                get_tool_availability_status,
                initialize_recommendation_tools,
                recommend_tool_sequence,
                recommend_tools_for_task,
                suggest_tool_alternatives,
            )

            self._tool_recommendation_functions.update(
                {
                    "analyze_task_requirements": analyze_task_requirements,
                    "get_tool_availability_status": get_tool_availability_status,
                    "initialize_recommendation_tools": initialize_recommendation_tools,
                    "recommend_tool_sequence": recommend_tool_sequence,
                    "recommend_tools_for_task": recommend_tools_for_task,
                    "suggest_tool_alternatives": suggest_tool_alternatives,
                }
            )

        return self._tool_recommendation_functions.get(function_name)

    def get_search_function(self, function_name: str):
        """Get a tool search function with lazy loading."""
        if function_name not in self._tool_search_functions:
            from tool_search_mcp_tools import (
                get_tool_details,
                initialize_search_engines,
                list_tool_categories,
                search_mcp_tools,
                search_tools_by_capability,
            )

            self._tool_search_functions.update(
                {
                    "get_tool_details": get_tool_details,
                    "initialize_search_engines": initialize_search_engines,
                    "list_tool_categories": list_tool_categories,
                    "search_mcp_tools": search_mcp_tools,
                    "search_tools_by_capability": search_tools_by_capability,
                }
            )

        return self._tool_search_functions.get(function_name)

    def get_comparison_function(self, function_name: str):
        """Get a tool comparison function with lazy loading."""
        if function_name not in self._tool_comparison_functions:
            from tool_comparison_mcp_tools import (
                analyze_tool_relationships,
                compare_mcp_tools,
                find_tool_alternatives,
                get_tool_recommendations_comparison,
                initialize_comparison_tools,
            )

            self._tool_comparison_functions.update(
                {
                    "analyze_tool_relationships": analyze_tool_relationships,
                    "compare_mcp_tools": compare_mcp_tools,
                    "find_tool_alternatives": find_tool_alternatives,
                    "get_tool_recommendations_comparison": get_tool_recommendations_comparison,
                    "initialize_comparison_tools": initialize_comparison_tools,
                }
            )

        return self._tool_comparison_functions.get(function_name)

    def get_category_function(self, function_name: str):
        """Get a tool category function with lazy loading."""
        if function_name not in self._tool_category_functions:
            from tool_category_mcp_tools import (
                browse_tools_by_category,
                explore_tool_ecosystem,
                find_tools_by_complexity,
                get_category_overview,
                get_tool_selection_guidance,
                initialize_category_tools,
            )

            self._tool_category_functions.update(
                {
                    "browse_tools_by_category": browse_tools_by_category,
                    "explore_tool_ecosystem": explore_tool_ecosystem,
                    "find_tools_by_complexity": find_tools_by_complexity,
                    "get_category_overview": get_category_overview,
                    "get_tool_selection_guidance": get_tool_selection_guidance,
                    "initialize_category_tools": initialize_category_tools,
                }
            )

        return self._tool_category_functions.get(function_name)

    def get_component(self, component_name: str):
        """Get a component with lazy loading."""
        if component_name not in self._components:
            if component_name == "ContextAnalyzer":
                from context_analyzer import ContextAnalyzer

                self._components["ContextAnalyzer"] = ContextAnalyzer
            elif component_name == "ParameterSearchEngine":
                from parameter_search_engine import ParameterSearchEngine

                self._components["ParameterSearchEngine"] = ParameterSearchEngine
            elif component_name == "ToolRecommendationEngine":
                from tool_recommendation_engine import ToolRecommendationEngine

                self._components["ToolRecommendationEngine"] = ToolRecommendationEngine
            elif component_name == "ToolComparisonEngine":
                from tool_comparison_engine import ToolComparisonEngine

                self._components["ToolComparisonEngine"] = ToolComparisonEngine
            elif component_name == "AlternativeDetector":
                from alternative_detector import AlternativeDetector

                self._components["AlternativeDetector"] = AlternativeDetector
            elif component_name == "DecisionSupport":
                from decision_support import DecisionSupport

                self._components["DecisionSupport"] = DecisionSupport
            elif component_name == "TaskAnalyzer":
                from task_analyzer import TaskAnalyzer

                self._components["TaskAnalyzer"] = TaskAnalyzer

        return self._components.get(component_name)


# Create global tool registry instance
_tool_registry = MCPToolRegistry()

# Global lock for database connection management
_db_connection_lock: threading.Lock = threading.Lock()

# Initialize logger for MCP server
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

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

# Global variables for tool recommendation system
_recommendation_engine: Optional[ToolRecommendationEngine] = None
_context_analyzer: Optional[ContextAnalyzer] = None
_parameter_search_engine: Optional[ParameterSearchEngine] = None
_recommendation_tools_initialized: bool = False

# Global variables for tool comparison and category systems
_comparison_engine: Optional[ToolComparisonEngine] = None
_alternative_detector = None
_relationship_analyzer = None
_decision_support = None
_tool_catalog = None
_task_analyzer = None
_comparison_tools_initialized: bool = False
_category_tools_initialized: bool = False


class MCPServerInitializer:
    """Handles complex MCP server initialization with proper error handling."""

    def __init__(self):
        self.config = _config.copy()
        self.initialization_errors = []

    def initialize_database_and_embedder(self) -> tuple[bool, Optional[str]]:
        """Initialize database and embedder components."""
        try:
            get_db_connection()
            get_embedder()
            return True, None
        except Exception as e:
            error_msg = f"Failed to initialize database/embedder: {e}"
            self.initialization_errors.append(error_msg)
            return False, error_msg

    def run_smart_auto_index(self) -> bool:
        """Run smart auto-indexing if configured."""
        if not (self.config["repository_path"] and self.config["auto_index"]):
            return True

        try:
            import time

            print("🔍 Checking if repository needs indexing...", file=sys.stderr)

            repo_path = Path(self.config["repository_path"])
            max_bytes = int(self.config["max_file_size_mb"] * 1024 * 1024)

            # Check index freshness
            con = get_db_connection()
            freshness = check_index_freshness(repo_path, max_bytes, con)

            print(f"📊 Index status: {freshness['reason']}", file=sys.stderr)
            print(f"📊 Total files: {freshness['total_files']}", file=sys.stderr)

            if self.config["force_reindex"]:
                print("🔄 Force reindexing requested, rebuilding entire index...", file=sys.stderr)
            elif freshness["is_fresh"]:
                print("✨ Index is up-to-date, skipping reindexing", file=sys.stderr)
                print(f"📅 Last indexed: {freshness['last_index_time']}", file=sys.stderr)
                return True

            # Index needs updating
            if freshness["changed_files"] > 0:
                print(f"🔄 Found {freshness['changed_files']} changed files, updating index...", file=sys.stderr)
            else:
                print(f"🔍 {freshness['reason']}, building index...", file=sys.stderr)

            start_time = time.time()
            result = index_repository(force_all=self.config["force_reindex"])
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
                print(f"🚀 Indexing completed in {total_time:.2f}s at {time_per_file:.3f}s per file", file=sys.stderr)
            else:
                print(f"🚀 Indexing completed in {total_time:.2f}s", file=sys.stderr)

            print(f"✅ {result}", file=sys.stderr)
            return True

        except Exception as e:
            print(f"❌ Error during smart indexing: {e}", file=sys.stderr)
            print("🔄 Falling back to standard indexing...", file=sys.stderr)
            try:
                result = index_repository()
                print(f"✅ {result}", file=sys.stderr)
                return True
            except Exception as fallback_e:
                error_msg = f"Auto-indexing failed: {fallback_e}"
                self.initialization_errors.append(error_msg)
                return False

    def initialize_mcp_tool_search(self) -> bool:
        """Initialize MCP tool search engines."""
        try:
            db_connection = get_db_connection()
            embedder = get_embedder()
            initialize_search_engines = _tool_registry.get_search_function("initialize_search_engines")
            initialize_search_engines(db_connection, embedder)
            print("🔧 MCP tool search engines initialized", file=sys.stderr)
            return True
        except Exception as e:
            print(f"⚠️  Warning: MCP tool search engines not initialized: {e}", file=sys.stderr)
            error_msg = f"MCP tool search initialization failed: {e}"
            self.initialization_errors.append(error_msg)
            return False

    def initialize_recommendation_engines(self) -> bool:
        """Initialize tool recommendation engines with proper global variable management."""
        try:
            global _recommendation_engine, _context_analyzer, _parameter_search_engine, _recommendation_tools_initialized

            # Get component classes from registry
            ContextAnalyzer = _tool_registry.get_component("ContextAnalyzer")
            ParameterSearchEngine = _tool_registry.get_component("ParameterSearchEngine")
            ToolRecommendationEngine = _tool_registry.get_component("ToolRecommendationEngine")

            # Create recommendation engine components
            from mcp_tool_search_engine import MCPToolSearchEngine

            db_connection = get_db_connection()
            embedder = get_embedder()

            # Initialize core components
            mcp_tool_search = MCPToolSearchEngine(db_connection, embedder)
            _parameter_search_engine = ParameterSearchEngine(db_connection, embedder)

            from task_analyzer import TaskAnalyzer

            task_analyzer = TaskAnalyzer()
            _context_analyzer = ContextAnalyzer()

            # Create recommendation engine
            _recommendation_engine = ToolRecommendationEngine(
                tool_search_engine=mcp_tool_search,
                parameter_search_engine=_parameter_search_engine,
                task_analyzer=task_analyzer,
                context_analyzer=_context_analyzer,
            )

            # Initialize recommendation tools
            initialize_recommendation_tools = _tool_registry.get_recommendation_function(
                "initialize_recommendation_tools"
            )
            initialize_recommendation_tools(_recommendation_engine, task_analyzer)
            _recommendation_tools_initialized = True

            print("🧠 Tool recommendation engines initialized", file=sys.stderr)
            return True

        except Exception as e:
            print(f"⚠️  Warning: Tool recommendation engines not initialized: {e}", file=sys.stderr)
            _recommendation_tools_initialized = False
            error_msg = f"Recommendation engines initialization failed: {e}"
            self.initialization_errors.append(error_msg)
            return False

    def initialize_comparison_engines(self) -> bool:
        """Initialize tool comparison engines with proper global variable management."""
        try:
            global _comparison_engine, _alternative_detector, _relationship_analyzer, _decision_support, _comparison_tools_initialized

            # Get component classes from registry
            ToolComparisonEngine = _tool_registry.get_component("ToolComparisonEngine")
            AlternativeDetector = _tool_registry.get_component("AlternativeDetector")
            DecisionSupport = _tool_registry.get_component("DecisionSupport")

            db_connection = get_db_connection()
            embedder = get_embedder()

            # Initialize comparison components
            from relationship_analyzer import RelationshipAnalyzer

            from comparison_metrics import ComparisonMetrics

            # Create comparison engine components
            comparison_metrics = ComparisonMetrics()
            _alternative_detector = AlternativeDetector(db_connection, embedder)
            _relationship_analyzer = RelationshipAnalyzer(db_connection, embedder)
            _decision_support = DecisionSupport()

            # Create comparison engine
            _comparison_engine = ToolComparisonEngine(
                alternative_detector=_alternative_detector,
                comparison_metrics=comparison_metrics,
                decision_support=_decision_support,
            )

            # Initialize comparison tools
            initialize_comparison_tools = _tool_registry.get_comparison_function("initialize_comparison_tools")
            initialize_comparison_tools(
                _comparison_engine, _decision_support, _alternative_detector, _relationship_analyzer
            )
            _comparison_tools_initialized = True

            print("🔄 Tool comparison engines initialized", file=sys.stderr)
            return True

        except Exception as e:
            print(f"⚠️  Warning: Tool comparison engines not initialized: {e}", file=sys.stderr)
            _comparison_tools_initialized = False
            self.initialization_errors.append(f"Comparison engines initialization failed: {e}")
            return False

    def initialize_category_engines(self) -> bool:
        """Initialize tool category browsing engines."""
        try:
            global _tool_catalog, _task_analyzer, _category_tools_initialized

            # Get component classes from registry
            TaskAnalyzer = _tool_registry.get_component("TaskAnalyzer")

            db_connection = get_db_connection()
            embedder = get_embedder()

            # Initialize category components
            from tool_catalog import ToolCatalog

            _tool_catalog = ToolCatalog(db_connection, embedder)
            _task_analyzer = TaskAnalyzer()

            # Initialize category tools
            initialize_category_tools = _tool_registry.get_category_function("initialize_category_tools")
            initialize_category_tools(
                _tool_catalog,
                _context_analyzer,  # Reuse from recommendation system
                _task_analyzer,
                _decision_support,  # Reuse from comparison system
            )
            _category_tools_initialized = True

            print("📂 Tool category engines initialized", file=sys.stderr)
            return True

        except Exception as e:
            print(f"⚠️  Warning: Tool category engines not initialized: {e}", file=sys.stderr)
            _category_tools_initialized = False
            self.initialization_errors.append(f"Category engines initialization failed: {e}")
            return False

    def start_file_watcher(self) -> bool:
        """Start file watcher if configured."""
        if not (self.config["repository_path"] and self.config["auto_watch"]):
            return True

        try:
            start_file_watcher()
            return True
        except Exception as e:
            error_msg = f"File watcher startup failed: {e}"
            self.initialization_errors.append(error_msg)
            return False

    def run_initialization(self) -> bool:
        """Run complete initialization sequence."""
        success = True

        # Initialize core components
        db_success, db_error = self.initialize_database_and_embedder()
        if not db_success:
            print(f"❌ Core initialization failed: {db_error}", file=sys.stderr)
            return False

        # Run auto-indexing in background
        if self.config["repository_path"] and self.config["auto_index"]:
            auto_index_thread = threading.Thread(target=self.run_smart_auto_index, daemon=True)
            auto_index_thread.start()
            print("🧠 Smart auto-indexing started in background...", file=sys.stderr)

        # Start file watcher
        if not self.start_file_watcher():
            success = False

        # Initialize MCP tool search
        if not self.initialize_mcp_tool_search():
            success = False

        # Initialize recommendation engines
        if not self.initialize_recommendation_engines():
            success = False

        # Initialize comparison engines
        if not self.initialize_comparison_engines():
            success = False

        # Initialize category engines
        if not self.initialize_category_engines():
            success = False

        return success

    def get_initialization_summary(self) -> str:
        """Get a summary of initialization results."""
        if not self.initialization_errors:
            return "✅ All components initialized successfully"

        error_summary = "\n".join([f"  • {error}" for error in self.initialization_errors])
        return f"⚠️  Initialization completed with warnings:\n{error_summary}"


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

        # Format results with IDE navigation
        formatted_results = []
        formatted_results.append(f"Found {len(results)} results for: '{query}'\n")

        for i, (path, snippet, distance) in enumerate(results, 1):
            similarity_pct = (1 - distance) * 100
            formatted_results.append(f"{i}. {path}")
            formatted_results.append(f"   Similarity: {similarity_pct:.1f}%")
            formatted_results.append(f"   Preview: {snippet.strip()[:config.file_processing.PREVIEW_LENGTH]}" "...")

            # Add IDE navigation URLs for enhanced user experience
            from ide_integration import get_ide_navigation_urls

            nav_urls = get_ide_navigation_urls(path, 1)  # Use line 1 as default
            if nav_urls:
                # Show the first available IDE navigation URL
                primary_url = next((url for url in nav_urls if url.is_available), nav_urls[0])
                formatted_results.append(f"   🔗 Open in {primary_url.display_name}: {primary_url.url}")

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
                performance_notes=["No index found. Please index a repository first using the index_repository tool."],
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
            include_query_analysis=True,
        )

        # Add IDE navigation data to results
        for result in response.results:
            # Generate IDE navigation URLs for each result
            result.generate_ide_navigation()
            # Generate syntax highlighting hints if we can read the file
            result.generate_syntax_hints()

        # Add repository context if available
        repo_path = _config.get("repository_path")
        if repo_path:
            response.navigation_hints.insert(0, f"Repository: {repo_path}")

        return response.to_json()

    except Exception as e:
        # Return structured error response
        error_response = SearchResponse(
            query=query, results=[], total_results=0, performance_notes=[f"Error in structured search: {str(e)}"]
        )
        return error_response.to_json()


@mcp.tool()
def search_code_hybrid(
    query: str,
    search_mode: str = "auto",
    max_results: int = None,
    semantic_weight: float = 0.6,
    text_weight: float = 0.4,
    enable_advanced_features: bool = True,
) -> str:
    """
    🔀 TURBOPROP: Advanced hybrid search combining semantic + exact text matching!

    ULTIMATE SEARCH EXPERIENCE! This combines the best of semantic understanding
    with exact text matching using advanced fusion algorithms. Gets better results
    by understanding both MEANING and EXACT CONTENT.

    🎯 SEARCH MODES:
    • "auto" - Smart routing based on query type (RECOMMENDED)
    • "hybrid" - Full semantic+text fusion with custom weights
    • "semantic" - Pure semantic search using AI embeddings
    • "text" - Exact text matching with Boolean operators

    🚀 QUERY TYPES HANDLED:
    • "function to parse JSON" → semantic search
    • "def parse_json" → exact text matching
    • "JWT AND authentication" → Boolean text search
    • "class UserAuth" → intelligent routing
    • filetype:py authentication → file type filtering
    • "exact phrase" → quoted phrase matching

    ⚖️ FUSION ALGORITHM:
    • Reciprocal Rank Fusion (RRF) combines rankings
    • Weighted scoring balances semantic vs text matches
    • Exact match boosting for precise queries
    • Smart query expansion for semantic searches

    🔧 ADVANCED FEATURES:
    • Regex pattern matching
    • File type filtering (filetype:py, ext:js)
    • Date range filtering for recent changes
    • Wildcard matching with *
    • Boolean operators (AND, OR, NOT)

    Args:
        query: Natural language or exact search query
        search_mode: "auto", "hybrid", "semantic", or "text"
        max_results: Number of results (default: 5, max: 20)
        semantic_weight: Weight for semantic matches (0.0-1.0)
        text_weight: Weight for text matches (0.0-1.0)
        enable_advanced_features: Enable regex, wildcards, file filters

    Returns:
        Enhanced search results with fusion scoring and match explanations
    """
    try:
        if max_results is None:
            max_results = config.search.DEFAULT_MAX_RESULTS
        if max_results > config.search.MAX_RESULTS_LIMIT:
            max_results = config.search.MAX_RESULTS_LIMIT

        # Validate search mode
        valid_modes = ["auto", "hybrid", "semantic", "text"]
        if search_mode not in valid_modes:
            search_mode = "auto"

        # Normalize weights to sum to 1.0
        total_weight = semantic_weight + text_weight
        if total_weight > 0:
            semantic_weight = semantic_weight / total_weight
            text_weight = text_weight / total_weight
        else:
            semantic_weight, text_weight = 0.6, 0.4

        db_manager = get_db_connection()
        embedder = get_embedder()

        # Check if index exists
        file_count = db_manager.execute_with_retry(f"SELECT COUNT(*) FROM {TABLE_NAME}")[0][0]
        if file_count == 0:
            return "No index found. Please index a repository first using the index_repository tool."

        # Execute hybrid search
        if search_mode == "auto":
            results = search_with_intelligent_routing(
                db_manager, embedder, query, max_results, enable_advanced_features
            )
        else:
            fusion_weights = {
                "semantic_weight": semantic_weight,
                "text_weight": text_weight,
                "rrf_k": config.search.RRF_K,
                "boost_exact_matches": True,
                "exact_match_boost": 1.5,
            }
            results = search_with_hybrid_fusion(
                db_manager, embedder, query, max_results, search_mode, fusion_weights, enable_query_expansion=True
            )

        if not results:
            return (
                f"No results found for hybrid search: '{query}' (mode: {search_mode}). "
                "Try different search terms, change the search mode, or check that the repository is indexed."
            )

        # Format results with hybrid search formatting
        repo_path = _config.get("repository_path")
        formatted_results = format_hybrid_search_results(
            results, query, show_fusion_details=(search_mode == "hybrid"), repo_path=repo_path
        )

        # Add search mode and configuration info
        header_lines = [
            "🔀 HYBRID SEARCH RESULTS",
            f"Query: '{query}' | Mode: {search_mode} | Results: {len(results)}",
            f"Weights: semantic={semantic_weight:.2f}, text={text_weight:.2f}",
            "=" * 60,
            "",
        ]

        return "\n".join(header_lines) + formatted_results

    except Exception as e:
        return f"Error in hybrid search: {str(e)}"


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
                execution_time=time.time() - start_time,
            )
            error_response.add_error("Either provide a path or configure one at startup")
            return error_response.to_json()

        repo_path = Path(repository_path).resolve()

        if not repo_path.exists():
            error_response = IndexResponse(
                operation="index",
                status="failed",
                message="Repository path does not exist",
                repository_path=repository_path,
                execution_time=time.time() - start_time,
            )
            error_response.add_error(f"Path '{repository_path}' does not exist")
            return error_response.to_json()

        if not repo_path.is_dir():
            error_response = IndexResponse(
                operation="index",
                status="failed",
                message="Path is not a directory",
                repository_path=repository_path,
                execution_time=time.time() - start_time,
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
                execution_time=time.time() - start_time,
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
            max_file_size_mb=max_file_size_mb,
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
            execution_time=time.time() - start_time,
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
        embedding_count = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE embedding IS NOT NULL").fetchone()[0]

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
            type_results = con.execute(
                f"""
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
            """
            ).fetchall()
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
            file_types=file_types,
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
            status="error", is_ready_for_search=False, total_files=0, files_with_embeddings=0, total_embeddings=0
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
            results=construct_results, query=query, show_signatures=True, show_docstrings=True
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
            results=class_results, query=query, show_signatures=True, show_docstrings=True
        )

        # Add methods for each class if requested
        if include_methods and class_results:
            enhanced_lines = formatted_result.split("\n")

            for class_result in class_results:
                try:
                    # Find related methods for this class
                    related_constructs = construct_ops.get_related_constructs(
                        construct_id=class_result.construct_id, k=5  # Limit to top 5 methods per class
                    )

                    methods = [c for c in related_constructs if c.construct_type == "method"]

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

            formatted_result = "\n".join(enhanced_lines)

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
            show_docstrings=False,  # Imports typically don't have docstrings
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
    construct_types: str = None,
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
            construct_types_list = [t.strip() for t in construct_types.split(",") if t.strip()]

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
            construct_types=construct_types_list,
        )

        if not hybrid_results:
            return f"No hybrid search results found for query: '{query}'"

        # Format hybrid results with construct context
        formatted_result = format_hybrid_search_results(
            results=hybrid_results, query=query, show_construct_context=True
        )

        return formatted_result

    except Exception as e:
        logger.error(f"Error in search_hybrid_constructs for query '{query}': {e}")
        return f"❌ Hybrid search failed for query '{query}': {e}"


# MCP Tool Search Tools


@mcp.tool()
def search_mcp_tools(
    query: str,
    category: str = None,
    tool_type: str = None,
    max_results: int = 10,
    include_examples: bool = True,
    search_mode: str = "hybrid",
) -> str:
    """
    🔍 TURBOPROP: Search MCP tools by functionality or description

    FIND MCP TOOLS BY MEANING! This performs semantic search over available MCP tools,
    finding tools that match the functional requirements described in natural language.

    🎯 SEARCH EXAMPLES:
    • "file operations with error handling" - Find file manipulation tools
    • "execute shell commands" - Find command execution tools
    • "web scraping tools" - Find web and HTTP tools
    • "data analysis functions" - Find analysis and processing tools

    🏆 ADVANTAGES:
    • Understands TOOL FUNCTIONALITY, not just names
    • Finds tools across different categories
    • Provides usage examples and guidance
    • Perfect for discovering tools you didn't know existed

    Args:
        query: Natural language description of desired functionality
        category: Optional filter by tool category (file_ops, web, analysis, etc.)
        tool_type: Optional filter by tool type (system, custom, third_party)
        max_results: Maximum number of tools to return (1-50)
        include_examples: Whether to include usage examples in results
        search_mode: Search strategy ('semantic', 'hybrid', 'keyword')

    Returns:
        JSON with tool search results, metadata, and suggestions
    """
    try:
        search_mcp_tools = _tool_registry.get_search_function("search_mcp_tools")
        result = search_mcp_tools(query, category, tool_type, max_results, include_examples, search_mode)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error searching MCP tools: {str(e)}"


@mcp.tool()
def get_tool_details(
    tool_id: str,
    include_schema: bool = True,
    include_examples: bool = True,
    include_relationships: bool = True,
    include_usage_guidance: bool = True,
) -> str:
    """
    📋 TURBOPROP: Get comprehensive information about a specific MCP tool

    DEEP DIVE INTO ANY TOOL! This provides detailed information about a specific MCP tool
    including parameters, usage examples, relationships with other tools, and best practices.

    🎯 WHAT YOU GET:
    • Complete parameter schema with types and constraints
    • Real usage examples with expected outputs
    • Alternative and complementary tools
    • Best practices and common pitfalls
    • Implementation guidance and tips

    💡 PERFECT FOR:
    • Understanding tool capabilities before use
    • Learning proper usage patterns
    • Finding alternative tools for comparison
    • Debugging tool usage issues

    Args:
        tool_id: Identifier of the tool to inspect (e.g., 'bash', 'read', 'search_code')
        include_schema: Include full parameter schema and type information
        include_examples: Include usage examples and code snippets
        include_relationships: Include alternative and complementary tools
        include_usage_guidance: Include best practices and common pitfalls

    Returns:
        JSON with comprehensive tool documentation and metadata
    """
    try:
        get_tool_details = _tool_registry.get_search_function("get_tool_details")
        result = get_tool_details(
            tool_id, include_schema, include_examples, include_relationships, include_usage_guidance
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error getting tool details: {str(e)}"


@mcp.tool()
def list_tool_categories() -> str:
    """
    📊 TURBOPROP: Get overview of available tool categories and their contents

    BROWSE TOOLS BY CATEGORY! This provides a structured overview of all available tool
    categories, helping you understand the organization of tools and browse by functional area.

    🎯 CATEGORY OVERVIEW INCLUDES:
    • Category names and descriptions
    • Tool counts per category
    • Representative tools in each category
    • Usage patterns and common workflows

    💡 USEFUL FOR:
    • Understanding tool organization
    • Discovering tools in specific domains
    • Planning multi-tool workflows
    • Getting oriented with available functionality

    Returns:
        JSON with categories, tool counts, descriptions, and representative tools
    """
    try:
        list_tool_categories = _tool_registry.get_search_function("list_tool_categories")
        result = list_tool_categories()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error listing tool categories: {str(e)}"


@mcp.tool()
def search_tools_by_capability(
    capability_description: str,
    required_parameters: List[str] = None,
    preferred_complexity: str = "any",
    max_results: int = 10,
) -> str:
    """
    🎯 TURBOPROP: Search tools by specific capability requirements

    PRECISION TOOL MATCHING! This finds tools that have specific capabilities or parameter
    requirements, enabling precise tool matching for technical requirements.

    🎯 CAPABILITY SEARCH FOR:
    • Specific parameter requirements
    • Complexity preferences (simple, moderate, complex)
    • Technical capabilities and features
    • Performance characteristics

    💡 EXAMPLES:
    • "timeout support" - Find tools with timeout capabilities
    • "file path handling" with required_parameters=["file_path"]
    • "error handling" with preferred_complexity="simple"
    • "batch processing" - Find tools that can process multiple items

    🏆 ADVANTAGES:
    • Matches exact technical requirements
    • Filters by complexity to match user skill level
    • Explains why each tool matches requirements
    • Provides implementation guidance

    Args:
        capability_description: Description of required capability
        required_parameters: List of parameter names that must be supported
        preferred_complexity: Complexity preference ('simple', 'moderate', 'complex', 'any')
        max_results: Maximum number of tools to return

    Returns:
        JSON with tools matching capability requirements and match explanations
    """
    try:
        search_tools_by_capability = _tool_registry.get_search_function("search_tools_by_capability")
        result = search_tools_by_capability(
            capability_description, required_parameters, preferred_complexity, max_results
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error searching tools by capability: {str(e)}"


# Tool Recommendation MCP Tools


@mcp.tool()
def recommend_tools_for_task(
    task_description: str,
    context: str = None,
    max_recommendations: int = 5,
    include_alternatives: bool = True,
    complexity_preference: str = "balanced",
    explain_reasoning: bool = True,
) -> str:
    """
    🧠 TURBOPROP: Get intelligent tool recommendations for development tasks

    SMART TOOL SELECTION! This analyzes your task description and recommends the most
    appropriate MCP tools based on functionality, complexity, and context. Get detailed
    explanations and alternatives to choose the optimal approach.

    🎯 PERFECT FOR:
    • Understanding what tools to use for a specific task
    • Getting explanations for why tools are recommended
    • Exploring alternative approaches and tools
    • Learning about tool capabilities and usage patterns

    💡 EXAMPLES:
    • "read configuration file and parse JSON data"
    • "search codebase for specific function implementations"
    • "execute shell commands with timeout handling"
    • "process CSV files and generate reports"

    Args:
        task_description: Natural language description of what you want to accomplish
        context: Additional context about environment, constraints, or preferences
        max_recommendations: Maximum number of tool recommendations (1-10)
        include_alternatives: Whether to include alternative tool options
        complexity_preference: Tool complexity preference ('simple', 'balanced', 'powerful')
        explain_reasoning: Whether to include detailed explanations

    Returns:
        JSON with ranked tool recommendations, explanations, and alternatives
    """
    try:
        recommend_tools_for_task = _tool_registry.get_recommendation_function("recommend_tools_for_task")
        result = recommend_tools_for_task(
            task_description,
            context,
            max_recommendations,
            include_alternatives,
            complexity_preference,
            explain_reasoning,
        )
        return json.dumps(result, indent=2) if isinstance(result, dict) else result
    except Exception as e:
        return f"Error getting tool recommendations: {str(e)}"


@mcp.tool()
def analyze_task_requirements(
    task_description: str, detail_level: str = "standard", include_suggestions: bool = True
) -> str:
    """
    🔬 TURBOPROP: Analyze task requirements and complexity

    DEEP TASK ANALYSIS! This provides detailed analysis of what your task requires,
    helping you understand complexity, required capabilities, and potential approaches
    before selecting tools.

    🎯 ANALYSIS INCLUDES:
    • Task complexity assessment (simple, moderate, complex)
    • Required capabilities and functional requirements
    • Input/output specifications and data flow
    • Potential challenges and implementation considerations
    • Skill level requirements and learning resources

    💡 PERFECT FOR:
    • Understanding task complexity before starting
    • Identifying requirements you might have missed
    • Getting suggestions to improve your task description
    • Planning multi-step workflows and processes

    Args:
        task_description: Description of the task to analyze
        detail_level: Level of analysis ('basic', 'standard', 'comprehensive')
        include_suggestions: Whether to include task improvement suggestions

    Returns:
        JSON with comprehensive task analysis and insights
    """
    try:
        analyze_task_requirements = _tool_registry.get_recommendation_function("analyze_task_requirements")
        result = analyze_task_requirements(task_description, detail_level, include_suggestions)
        return json.dumps(result, indent=2) if isinstance(result, dict) else result
    except Exception as e:
        return f"Error analyzing task requirements: {str(e)}"


@mcp.tool()
def suggest_tool_alternatives(
    primary_tool: str, task_context: str = None, max_alternatives: int = 5, include_comparisons: bool = True
) -> str:
    """
    🔄 TURBOPROP: Find alternative tools for your primary choice

    EXPLORE YOUR OPTIONS! This finds alternative tools that could accomplish similar
    tasks, providing detailed comparisons and guidance on when each alternative
    might be preferred over your primary choice.

    🎯 COMPARISON FEATURES:
    • Similarity analysis and functional overlap
    • Complexity and learning curve comparisons
    • Performance and reliability trade-offs
    • Use case recommendations and preferences
    • Migration effort and implementation guidance

    💡 PERFECT FOR:
    • Exploring different approaches to the same problem
    • Finding simpler or more powerful alternatives
    • Understanding tool trade-offs and limitations
    • Making informed decisions between similar tools

    Args:
        primary_tool: The primary tool you're considering (e.g., "bash", "read")
        task_context: Context about your specific task or use case
        max_alternatives: Maximum number of alternatives to suggest
        include_comparisons: Whether to include detailed comparisons

    Returns:
        JSON with alternative suggestions, comparisons, and usage guidance
    """
    try:
        suggest_tool_alternatives = _tool_registry.get_recommendation_function("suggest_tool_alternatives")
        result = suggest_tool_alternatives(primary_tool, task_context, max_alternatives, include_comparisons)
        return json.dumps(result, indent=2) if isinstance(result, dict) else result
    except Exception as e:
        return f"Error suggesting tool alternatives: {str(e)}"


@mcp.tool()
def recommend_tool_sequence(
    workflow_description: str,
    optimization_goal: str = "balanced",
    max_sequence_length: int = 10,
    allow_parallel_tools: bool = False,
) -> str:
    """
    ⚡ TURBOPROP: Recommend tool sequences for complex workflows

    WORKFLOW OPTIMIZATION! This analyzes complex multi-step workflows and recommends
    optimal sequences of tools to accomplish your tasks efficiently and reliably.
    Perfect for complex processes that require multiple tools working together.

    🎯 SEQUENCE FEATURES:
    • Step-by-step tool recommendations with parameters
    • Data flow analysis between tools
    • Parallel execution opportunities identification
    • Error handling and reliability considerations
    • Performance optimization and bottleneck analysis

    💡 OPTIMIZATION GOALS:
    • "speed" - Optimize for fast execution and minimal delays
    • "reliability" - Prioritize error handling and robust execution
    • "simplicity" - Prefer simple tools and straightforward approaches
    • "balanced" - Balance all factors for general-purpose workflows

    🔧 PERFECT FOR:
    • Multi-step data processing workflows
    • Complex deployment and build processes
    • Automated testing and validation sequences
    • Data transformation and analysis pipelines

    Args:
        workflow_description: Description of the complete workflow or process
        optimization_goal: What to optimize for ('speed', 'reliability', 'simplicity', 'balanced')
        max_sequence_length: Maximum number of tools in sequence (2-20)
        allow_parallel_tools: Whether to suggest parallel tool execution

    Returns:
        JSON with recommended tool sequences, optimization analysis, and guidance
    """
    try:
        recommend_tool_sequence = _tool_registry.get_recommendation_function("recommend_tool_sequence")
        result = recommend_tool_sequence(
            workflow_description, optimization_goal, max_sequence_length, allow_parallel_tools
        )
        return json.dumps(result, indent=2) if isinstance(result, dict) else result
    except Exception as e:
        return f"Error recommending tool sequence: {str(e)}"


# Tool Comparison and Category MCP Tools


@mcp.tool()
def compare_mcp_tools(
    tool_ids: List[str],
    comparison_criteria: List[str] = None,
    include_decision_guidance: bool = True,
    comparison_context: str = None,
    detail_level: str = "standard",
) -> str:
    """
    🔄 TURBOPROP: Compare multiple MCP tools across various dimensions

    COMPREHENSIVE TOOL COMPARISON! This provides side-by-side comparison of MCP tools,
    helping to understand differences, trade-offs, and optimal use cases for each tool.

    🎯 COMPARISON FEATURES:
    • Multi-dimensional analysis across functionality, usability, performance, complexity
    • Decision guidance with recommendations for optimal tool selection
    • Context-aware comparisons based on specific use cases
    • Trade-off analysis highlighting strengths and limitations

    💡 PERFECT FOR:
    • Choosing between similar tools for specific tasks
    • Understanding tool capabilities and limitations
    • Making informed decisions with comprehensive analysis
    • Learning about tool relationships and alternatives

    Args:
        tool_ids: List of tool IDs to compare (2-10 tools)
        comparison_criteria: Specific aspects to focus on ['functionality', 'usability', 'performance', 'complexity']
        include_decision_guidance: Whether to include selection recommendations
        comparison_context: Context for the comparison (e.g., "for file processing tasks")
        detail_level: Level of comparison detail ('basic', 'standard', 'comprehensive')

    Returns:
        JSON with comprehensive tool comparison, rankings, and decision guidance
    """
    try:
        compare_mcp_tools_func = _tool_registry.get_comparison_function("compare_mcp_tools")
        result = compare_mcp_tools_func(
            tool_ids, comparison_criteria, include_decision_guidance, comparison_context, detail_level
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error comparing MCP tools: {str(e)}"


@mcp.tool()
def find_tool_alternatives(
    reference_tool: str,
    similarity_threshold: float = 0.7,
    max_alternatives: int = 8,
    include_comparison: bool = True,
    context_filter: str = None,
) -> str:
    """
    🔍 TURBOPROP: Find alternative tools similar to a reference tool

    DISCOVER TOOL ALTERNATIVES! This finds tools with similar functionality,
    helping to explore different approaches and find optimal tools for specific use cases.

    🎯 ALTERNATIVE ANALYSIS:
    • Similarity scoring based on functionality overlap
    • Complexity and learning curve comparisons
    • Unique capabilities and performance differences
    • Context-based filtering for specific requirements

    💡 PERFECT FOR:
    • Exploring different approaches to the same problem
    • Finding simpler or more advanced alternatives
    • Understanding the tool ecosystem landscape
    • Making informed tool selection decisions

    Args:
        reference_tool: Tool ID to find alternatives for
        similarity_threshold: Minimum similarity score (0.0-1.0)
        max_alternatives: Maximum number of alternatives to return
        include_comparison: Whether to include comparison with reference tool
        context_filter: Optional context to filter alternatives (e.g., "simple tools only")

    Returns:
        JSON with alternative tools, similarity scores, and comparisons
    """
    try:
        find_tool_alternatives_func = _tool_registry.get_comparison_function("find_tool_alternatives")
        result = find_tool_alternatives_func(
            reference_tool, similarity_threshold, max_alternatives, include_comparison, context_filter
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error finding tool alternatives: {str(e)}"


@mcp.tool()
def analyze_tool_relationships(
    tool_id: str, relationship_types: List[str] = None, max_relationships: int = 20, include_explanations: bool = True
) -> str:
    """
    🕸️ TURBOPROP: Analyze relationships between a tool and other tools

    UNDERSTAND TOOL RELATIONSHIPS! This explores how a tool relates to others,
    including alternatives, complements, prerequisites, and workflow connections.

    🎯 RELATIONSHIP TYPES:
    • Alternatives - Tools that provide similar functionality
    • Complements - Tools that work well together in workflows
    • Prerequisites - Tools often needed before using this tool
    • Dependents - Tools that often use this tool as part of their functionality

    💡 PERFECT FOR:
    • Understanding tool ecosystem connections
    • Planning multi-tool workflows and processes
    • Finding complementary tools for complete solutions
    • Learning about tool dependencies and relationships

    Args:
        tool_id: Tool ID to analyze relationships for
        relationship_types: Types of relationships ['alternatives', 'complements', 'prerequisites', 'dependents']
        max_relationships: Maximum relationships to return per type
        include_explanations: Whether to explain why relationships exist

    Returns:
        JSON with comprehensive relationship analysis and explanations
    """
    try:
        analyze_tool_relationships_func = _tool_registry.get_comparison_function("analyze_tool_relationships")
        result = analyze_tool_relationships_func(tool_id, relationship_types, max_relationships, include_explanations)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error analyzing tool relationships: {str(e)}"


@mcp.tool()
def browse_tools_by_category(
    category: str,
    sort_by: str = "popularity",
    max_tools: int = 20,
    include_descriptions: bool = True,
    complexity_filter: str = None,
) -> str:
    """
    📂 TURBOPROP: Browse tools within a specific category

    SYSTEMATIC TOOL EXPLORATION! This enables organized browsing of tools by category,
    helping to discover tools with similar functionality and purposes.

    🎯 CATEGORY FEATURES:
    • Organized tool browsing by functional categories
    • Flexible sorting options (popularity, complexity, name, functionality)
    • Complexity filtering to match skill levels
    • Rich tool descriptions and metadata

    💡 AVAILABLE CATEGORIES:
    • file_ops - File operations (read, write, edit, etc.)
    • web - Web operations (WebFetch, WebSearch, etc.)
    • execution - Command execution (bash, task, etc.)
    • search - Search operations (grep, glob, etc.)
    • analysis - Analysis and processing tools

    Args:
        category: Category to browse (file_ops, web, analysis, etc.)
        sort_by: Sorting method ('popularity', 'complexity', 'name', 'functionality')
        max_tools: Maximum number of tools to return
        include_descriptions: Whether to include tool descriptions
        complexity_filter: Filter by complexity ('simple', 'moderate', 'complex')

    Returns:
        JSON with categorized tools, metadata, and organization
    """
    try:
        browse_tools_by_category_func = _tool_registry.get_category_function("browse_tools_by_category")
        result = browse_tools_by_category_func(category, sort_by, max_tools, include_descriptions, complexity_filter)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error browsing tools by category: {str(e)}"


@mcp.tool()
def get_category_overview() -> str:
    """
    🗺️ TURBOPROP: Get overview of all tool categories and characteristics

    ECOSYSTEM OVERVIEW! This provides a high-level view of the tool ecosystem,
    helping to understand the organization and scope of available tools.

    🎯 OVERVIEW FEATURES:
    • Comprehensive category statistics and breakdowns
    • Tool counts and distribution across categories
    • Ecosystem maturity and completeness metrics
    • Representative tools for each category

    💡 PERFECT FOR:
    • Getting oriented with the tool ecosystem
    • Understanding tool organization and structure
    • Planning comprehensive tool usage strategies
    • Discovering new functional areas and categories

    Returns:
        JSON with comprehensive category overview and ecosystem statistics
    """
    try:
        get_category_overview_func = _tool_registry.get_category_function("get_category_overview")
        result = get_category_overview_func()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error getting category overview: {str(e)}"


@mcp.tool()
def get_tool_selection_guidance(
    task_description: str,
    available_tools: List[str] = None,
    constraints: List[str] = None,
    optimization_goal: str = "balanced",
) -> str:
    """
    🧭 TURBOPROP: Get guidance for selecting optimal tools for specific tasks

    INTELLIGENT TOOL SELECTION! This provides decision support for tool selection,
    considering task requirements, available options, constraints, and optimization goals.

    🎯 SELECTION GUIDANCE:
    • Task analysis and requirement identification
    • Constraint-aware tool recommendations
    • Optimization for speed, reliability, simplicity, or balanced approach
    • Decision reasoning and alternative suggestions

    💡 OPTIMIZATION GOALS:
    • 'speed' - Optimize for fast execution and minimal delays
    • 'reliability' - Prioritize error handling and robust execution
    • 'simplicity' - Prefer simple tools and straightforward approaches
    • 'balanced' - Balance all factors for general-purpose tasks

    Args:
        task_description: Description of the task requiring tool selection
        available_tools: List of tools to choose from (if limited)
        constraints: Constraints to consider (e.g., ["no complex tools", "performance critical"])
        optimization_goal: What to optimize for ('speed', 'reliability', 'simplicity', 'balanced')

    Returns:
        JSON with tool selection guidance, reasoning, and alternatives
    """
    try:
        get_tool_selection_guidance_func = _tool_registry.get_category_function("get_tool_selection_guidance")
        result = get_tool_selection_guidance_func(task_description, available_tools, constraints, optimization_goal)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error getting tool selection guidance: {str(e)}"


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

    # Initialize all server components using the new initializer class
    initializer = MCPServerInitializer()
    initializer.run_initialization()

    # Print initialization summary
    print(file=sys.stderr)
    print(initializer.get_initialization_summary(), file=sys.stderr)
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
    print("🧰 MCP TOOL SEARCH:", file=sys.stderr)
    print("  • tp:search_mcp_tools - Find MCP tools by functionality", file=sys.stderr)
    print("  • tp:get_tool_details - Get comprehensive tool information", file=sys.stderr)
    print("  • tp:list_tool_categories - Browse tools by category", file=sys.stderr)
    print("  • tp:search_tools_by_capability - Find tools by specific capabilities", file=sys.stderr)
    print(file=sys.stderr)
    print("🧠 INTELLIGENT TOOL RECOMMENDATIONS (NEW!):", file=sys.stderr)
    print("  • tp:recommend_tools_for_task - Get smart tool recommendations for tasks", file=sys.stderr)
    print("  • tp:analyze_task_requirements - Analyze task complexity and requirements", file=sys.stderr)
    print("  • tp:suggest_tool_alternatives - Find alternative tools with comparisons", file=sys.stderr)
    print("  • tp:recommend_tool_sequence - Get tool sequences for complex workflows", file=sys.stderr)
    print(file=sys.stderr)
    print("🔄 TOOL COMPARISON & ANALYSIS (NEW!):", file=sys.stderr)
    print("  • tp:compare_mcp_tools - Compare multiple tools across dimensions", file=sys.stderr)
    print("  • tp:find_tool_alternatives - Find alternative tools with similarity analysis", file=sys.stderr)
    print("  • tp:analyze_tool_relationships - Understand tool connections and dependencies", file=sys.stderr)
    print(file=sys.stderr)
    print("📂 TOOL CATEGORY BROWSING (NEW!):", file=sys.stderr)
    print("  • tp:browse_tools_by_category - Browse tools by functional categories", file=sys.stderr)
    print("  • tp:get_category_overview - Get ecosystem overview and statistics", file=sys.stderr)
    print("  • tp:get_tool_selection_guidance - Get intelligent tool selection guidance", file=sys.stderr)
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
