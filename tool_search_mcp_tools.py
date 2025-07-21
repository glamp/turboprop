#!/usr/bin/env python3
"""
Tool Search MCP Tools

This module implements the core MCP tools for tool search functionality,
providing natural language tool discovery, detailed tool inspection,
category browsing, and capability matching.
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from database_manager import DatabaseManager
from embedding_helper import EmbeddingGenerator
from logging_config import get_logger
from mcp_metadata_types import ParameterAnalysis, ToolExample, ToolId
from mcp_response_standardizer import standardize_mcp_tool_response
from mcp_tool_search_engine import MCPToolSearchEngine
from mcp_tool_search_responses import (
    CapabilityMatch,
    CapabilitySearchResponse,
    ToolCategoriesResponse,
    ToolCategory,
    ToolDetailsResponse,
    ToolSearchMCPResponse,
    create_error_response,
)
from mcp_tool_validator import MCPToolValidator, ValidatedCapabilityParams, ValidatedSearchParams, tool_exists
from parameter_search_engine import ParameterSearchEngine
from tool_search_results import ToolSearchResult

logger = get_logger(__name__)


@dataclass
class OperationContext:
    """Context information for structured logging and operation tracking."""

    operation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    operation_name: str = ""
    user_id: Optional[str] = None
    query: Optional[str] = None
    tool_id: Optional[str] = None
    start_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def start(self) -> None:
        """Mark the start of the operation."""
        self.start_time = time.time()

    def elapsed(self) -> float:
        """Get elapsed time since operation start."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def to_log_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for structured logging."""
        log_data = {
            "operation_id": self.operation_id,
            "operation_name": self.operation_name,
            "elapsed_seconds": self.elapsed(),
        }

        if self.user_id:
            log_data["user_id"] = self.user_id
        if self.query:
            log_data["query"] = self.query[:100] + "..." if len(self.query) > 100 else self.query
        if self.tool_id:
            log_data["tool_id"] = self.tool_id
        if self.metadata:
            log_data.update(self.metadata)

        return log_data


def log_operation(operation_name: str, context: OperationContext = None):
    """
    Decorator for adding structured logging to operations.

    Args:
        operation_name: Name of the operation being logged
        context: Optional operation context for additional data
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create or use existing context
            op_context = context or OperationContext(operation_name=operation_name)
            if not op_context.operation_name:
                op_context.operation_name = operation_name

            # Extract context from common parameters
            if args:
                if hasattr(args[0], "__name__") or isinstance(args[0], str):
                    if "query" in str(args[0]) or "search" in operation_name.lower():
                        op_context.query = str(args[0])
                    elif "tool" in operation_name.lower():
                        op_context.tool_id = str(args[0])

            op_context.start()

            # Log operation start
            logger.info("Starting %s", operation_name, extra={"structured_data": op_context.to_log_dict()})

            try:
                result = func(*args, **kwargs)

                # Log successful completion
                completion_data = op_context.to_log_dict()
                if hasattr(result, "__len__") and not isinstance(result, str):
                    completion_data["result_count"] = len(result)

                logger.info(
                    "Completed %s successfully in %.3fs",
                    operation_name,
                    op_context.elapsed(),
                    extra={"structured_data": completion_data},
                )

                return result

            except Exception as e:
                # Log operation failure
                error_data = op_context.to_log_dict()
                error_data.update({"error_type": type(e).__name__, "error_message": str(e), "success": False})

                logger.error(
                    "Failed %s after %.3fs: %s",
                    operation_name,
                    op_context.elapsed(),
                    str(e),
                    extra={"structured_data": error_data},
                )
                raise

        return wrapper

    return decorator


# Search configuration constants
SEARCH_CONFIG = {
    "semantic_weight": 0.7,
    "keyword_weight": 0.3,
    "default_max_results": 10,
    "max_results_limit": 50,
    "representative_tools_limit": 3,
    "suggestion_thresholds": {
        "few_results": 3,
        "many_results": 15,
    },
    "scoring": {
        "perfect_match": 1.0,
        "default_effectiveness": 0.8,
        "high_param_match_threshold": 0.8,
        "medium_param_match_threshold": 0.5,
        "complexity_preference_center": 0.5,
    },
}

# Database configuration constants
DB_CONFIG = {
    "retry_attempts": 3,
    "retry_delay": 1.0,
    "connection_timeout": 30.0,
}

# Cache configuration constants
CACHE_CONFIG = {
    "category_cache_ttl": 300,  # 5 minutes
    "representative_tools_cache_ttl": 600,  # 10 minutes
    "max_cache_size": 100,
}


@dataclass
class CacheEntry:
    """Simple cache entry with TTL support."""

    value: Any
    timestamp: float
    ttl: float

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return time.time() - self.timestamp > self.ttl

    def is_valid(self) -> bool:
        """Check if the cache entry is still valid."""
        return not self.is_expired()


class SimpleCache:
    """
    Simple in-memory cache with TTL support for performance optimization.

    This cache is designed for caching expensive operations like database
    queries for category information that don't change frequently.
    """

    def __init__(self, max_size: int = None):
        self.max_size = max_size or CACHE_CONFIG["max_cache_size"]
        self._cache: Dict[str, CacheEntry] = {}

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache if it exists and is not expired.

        Args:
            key: Cache key

        Returns:
            Cached value if valid, None otherwise
        """
        if key in self._cache:
            entry = self._cache[key]
            if entry.is_valid():
                return entry.value
            else:
                # Remove expired entry
                del self._cache[key]
        return None

    def set(self, key: str, value: Any, ttl: float) -> None:
        """
        Set a value in cache with specified TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        # Implement simple LRU by removing oldest entries if at capacity
        if len(self._cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
            del self._cache[oldest_key]

        self._cache[key] = CacheEntry(value=value, timestamp=time.time(), ttl=ttl)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    def cleanup_expired(self) -> int:
        """Remove expired entries and return count of removed entries."""
        expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
        for key in expired_keys:
            del self._cache[key]
        return len(expired_keys)


# Global cache instance
_cache = SimpleCache()


def _with_db_retry(max_attempts: int = None, delay: float = None):
    """
    Decorator that adds retry logic for database operations.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """
    if max_attempts is None:
        max_attempts = DB_CONFIG["retry_attempts"]
    if delay is None:
        delay = DB_CONFIG["retry_delay"]

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    # Check if this is a database connection error that should be retried
                    if _is_retryable_db_error(e):
                        if attempt < max_attempts - 1:  # Don't sleep on final attempt
                            logger.warning(
                                "Database operation failed (attempt %d/%d): %s. Retrying in %.1fs...",
                                attempt + 1,
                                max_attempts,
                                str(e),
                                delay,
                            )
                            time.sleep(delay)
                            continue

                    # Non-retryable error or final attempt
                    break

            # All attempts failed
            logger.error("Database operation failed after %d attempts: %s", max_attempts, str(last_exception))
            raise last_exception

        return wrapper

    return decorator


def _is_retryable_db_error(error: Exception) -> bool:
    """
    Check if a database error should be retried.

    Args:
        error: Exception that occurred during database operation

    Returns:
        True if the error is retryable (connection issues, timeouts, etc.)
    """
    error_str = str(error).lower()
    retryable_patterns = [
        "connection",
        "timeout",
        "busy",
        "locked",
        "network",
        "unavailable",
        "refused",
        "broken pipe",
        "reset",
    ]
    return any(pattern in error_str for pattern in retryable_patterns)


def _handle_mcp_error(function_name: str, error: Exception, context: Optional[str] = None) -> str:
    """
    Standardized error handling for MCP tool functions.

    Args:
        function_name: Name of the function that encountered the error
        error: The exception that occurred
        context: Optional context information (query, tool_id, etc.)

    Returns:
        JSON string containing standardized error response
    """
    error_msg = str(error)
    logger.error("Error in %s: %s (context: %s)", function_name, error_msg, context or "none")
    return json.dumps(create_error_response(function_name, error_msg, context))


@dataclass
class MCPToolSearchContext:
    """
    Dependency injection container for MCP tool search operations.

    This class manages all dependencies required for MCP tool search functionality,
    providing a cleaner alternative to global state management. It supports
    dependency injection for better testability and maintainability.
    """

    db_manager: DatabaseManager
    embedding_generator: EmbeddingGenerator
    tool_search_engine: Optional[MCPToolSearchEngine] = None
    parameter_search_engine: Optional[ParameterSearchEngine] = None
    validator: Optional[MCPToolValidator] = None

    def __post_init__(self):
        """Initialize dependent components after construction."""
        if self.validator is None:
            self.validator = MCPToolValidator()

        if self.tool_search_engine is None:
            self.tool_search_engine = MCPToolSearchEngine(self.db_manager, self.embedding_generator)

        if self.parameter_search_engine is None:
            self.parameter_search_engine = ParameterSearchEngine(self.db_manager)

    def get_search_engine(self) -> MCPToolSearchEngine:
        """Get the tool search engine instance."""
        if self.tool_search_engine is None:
            raise RuntimeError("Tool search engine not initialized")
        return self.tool_search_engine

    def get_parameter_search_engine(self) -> ParameterSearchEngine:
        """Get the parameter search engine instance."""
        if self.parameter_search_engine is None:
            raise RuntimeError("Parameter search engine not initialized")
        return self.parameter_search_engine

    def get_validator(self) -> MCPToolValidator:
        """Get the validator instance."""
        if self.validator is None:
            raise RuntimeError("Validator not initialized")
        return self.validator

    def get_db_manager(self) -> DatabaseManager:
        """Get the database manager instance."""
        return self.db_manager


# Global context instance (will be replaced with proper DI in production)
_search_context: Optional[MCPToolSearchContext] = None


def _execute_mcp_search(validated_params: ValidatedSearchParams, search_mode: str):
    """Execute the actual tool search based on mode and parameters."""
    search_engine = _get_search_engine()

    if search_mode == "hybrid":
        return search_engine.search_hybrid(
            query=validated_params.query,
            k=validated_params.max_results,
            semantic_weight=SEARCH_CONFIG["semantic_weight"],
            keyword_weight=SEARCH_CONFIG["keyword_weight"],
        )
    else:
        return search_engine.search_by_functionality(
            query=validated_params.query,
            k=validated_params.max_results,
            category_filter=validated_params.category,
            tool_type_filter=validated_params.tool_type,
        )


def _create_search_response(
    query: str, search_results, search_mode: str, execution_time: float
) -> ToolSearchMCPResponse:
    """Create structured response from search results."""
    response = ToolSearchMCPResponse(
        query=query,
        results=search_results.results,
        search_mode=search_mode,
        execution_time=execution_time,
        search_strategy=search_results.search_strategy,
    )
    return response


def _enhance_search_response(response: ToolSearchMCPResponse, query: str) -> None:
    """Add suggestions and navigation hints to search response."""
    response.add_suggestions(generate_query_suggestions(query, response.results))
    response.add_navigation_hints(generate_navigation_hints(response.results))


def _validate_capability_business_logic(validated_params) -> None:
    """
    Perform additional business logic validation for capability search parameters.

    Args:
        validated_params: Already validated parameters from MCPToolValidator

    Raises:
        ValueError: If business logic constraints are violated
    """
    errors = []

    # Check for duplicate parameters
    if validated_params.required_parameters:
        unique_params = set(validated_params.required_parameters)
        if len(unique_params) != len(validated_params.required_parameters):
            errors.append("Duplicate parameters are not allowed in required_parameters")

        # Check for overly specific requirements (too many required parameters)
        if len(validated_params.required_parameters) > 10:
            errors.append(
                "Too many required parameters (max 10). " "Consider using multiple searches with fewer constraints"
            )

        # Check for suspicious parameter patterns
        suspicious_params = ["password", "secret", "key", "token", "auth"]
        found_suspicious = [
            p for p in validated_params.required_parameters if any(s in p.lower() for s in suspicious_params)
        ]
        if found_suspicious:
            logger.warning("Search includes security-sensitive parameters: %s", found_suspicious)

    # Validate capability description content
    capability_lower = validated_params.capability_description.lower()

    # Check for overly broad descriptions that might return too many results
    broad_terms = ["tool", "function", "command", "any", "all", "everything"]
    if any(term in capability_lower for term in broad_terms) and len(capability_lower.split()) <= 2:
        errors.append(
            "Capability description is too broad. " "Please be more specific about the functionality you need"
        )

    # Check for conflicting complexity and parameter requirements
    if (
        validated_params.preferred_complexity == "simple"
        and validated_params.required_parameters
        and len(validated_params.required_parameters) > 5
    ):
        errors.append(
            "Requesting 'simple' complexity with many required parameters is contradictory. "
            "Consider reducing parameters or changing complexity preference"
        )

    if errors:
        raise ValueError(f"Business logic validation errors: {'; '.join(errors)}")


def initialize_search_engines(db_manager: DatabaseManager, embedding_generator: EmbeddingGenerator) -> None:
    """
    Initialize the search engines with database and embedding components.

    This function should be called once during server startup to initialize
    the dependency injection context with the provided components.

    Args:
        db_manager: Database manager for data access
        embedding_generator: Embedding generator for semantic operations
    """
    global _search_context

    logger.info("Initializing MCP tool search context...")

    # Create dependency injection context
    _search_context = MCPToolSearchContext(db_manager=db_manager, embedding_generator=embedding_generator)

    logger.info("✅ MCPToolValidator initialized")
    logger.info("✅ MCPToolSearchEngine initialized")
    logger.info("✅ ParameterSearchEngine initialized")
    logger.info("🚀 MCP tool search context initialized successfully")


def get_search_context() -> MCPToolSearchContext:
    """
    Get the current search context.

    Returns:
        The initialized MCPToolSearchContext instance

    Raises:
        RuntimeError: If the context has not been initialized
    """
    if _search_context is None:
        raise RuntimeError("MCP tool search context not initialized. " "Call initialize_search_engines() first.")
    return _search_context


def _get_search_engine() -> MCPToolSearchEngine:
    """Get the initialized tool search engine."""
    return get_search_context().get_search_engine()


def _get_parameter_search_engine() -> ParameterSearchEngine:
    """Get the initialized parameter search engine."""
    return get_search_context().get_parameter_search_engine()


def _get_validator() -> MCPToolValidator:
    """Get the initialized validator."""
    return get_search_context().get_validator()


def _get_db_manager() -> DatabaseManager:
    """Get the initialized database manager."""
    return get_search_context().get_db_manager()


@log_operation("mcp_tool_search")
@standardize_mcp_tool_response
def search_mcp_tools(
    query: str,
    category: Optional[str] = None,
    tool_type: Optional[str] = None,
    max_results: int = 10,
    include_examples: bool = True,
    search_mode: str = "hybrid",
) -> dict:
    """
    Search for MCP tools by functionality or description.

    This tool enables Claude to discover available MCP tools using natural language
    descriptions. It combines semantic search with keyword matching to find tools
    that match functional requirements.

    Args:
        query: Natural language description of desired functionality
               Examples: "file operations with error handling", "execute shell commands"
        category: Optional filter by tool category (file_ops, web, analysis, etc.)
        tool_type: Optional filter by tool type (system, custom, third_party)
        max_results: Maximum number of tools to return (1-50)
        include_examples: Whether to include usage examples in results
        search_mode: Search strategy ('semantic', 'hybrid', 'keyword')

    Returns:
        Structured JSON with tool search results, metadata, and suggestions

    Examples:
        search_mcp_tools("find tools for reading files")
        search_mcp_tools("web scraping tools", category="web")
        search_mcp_tools("command execution", tool_type="system", max_results=5)
    """
    try:
        logger.info("Starting MCP tool search for query: '%s'", query)
        start_time = time.time()

        # Validate and process query parameters
        validator = _get_validator()
        validated_params = validator.validate_search_parameters(
            query, category, tool_type, max_results, include_examples, search_mode
        )

        # Execute search using appropriate strategy
        search_results = _execute_mcp_search(validated_params, search_mode)

        # Create and enhance response
        response = _create_search_response(query, search_results, search_mode, time.time() - start_time)
        _enhance_search_response(response, query)

        logger.info("MCP tool search completed: %d results in %.3fs", len(response.results), response.execution_time)

        return response.to_dict()

    except Exception as e:
        error_response = create_error_response("search_mcp_tools", str(e), query)
        logger.error("Error in search_mcp_tools: %s (context: %s)", str(e), query or "none")
        return error_response


@log_operation("get_tool_details")
@standardize_mcp_tool_response
def get_tool_details(
    tool_id: str,
    include_schema: bool = True,
    include_examples: bool = True,
    include_relationships: bool = True,
    include_usage_guidance: bool = True,
) -> dict:
    """
    Get comprehensive information about a specific MCP tool.

    This tool provides detailed information about a specific MCP tool including
    its parameters, usage examples, relationships with other tools, and best practices.

    Args:
        tool_id: Identifier of the tool to inspect (e.g., 'bash', 'read', 'search_code')
        include_schema: Include full parameter schema and type information
        include_examples: Include usage examples and code snippets
        include_relationships: Include alternative and complementary tools
        include_usage_guidance: Include best practices and common pitfalls

    Returns:
        Comprehensive tool documentation and metadata

    Examples:
        get_tool_details("bash")
        get_tool_details("read", include_schema=False)
        get_tool_details("search_code", include_relationships=False)
    """
    try:
        logger.info("Getting tool details for: %s", tool_id)
        start_time = time.time()

        # Validate tool existence
        if not tool_exists(tool_id):
            error_response = create_error_response("get_tool_details", f"Tool '{tool_id}' not found", tool_id)
            return error_response

        # Validate parameters
        validator = _get_validator()
        validated_params = validator.validate_tool_details_parameters(
            tool_id, include_schema, include_examples, include_relationships, include_usage_guidance
        )

        # Load comprehensive tool information
        tool_details = load_tool_details(
            validated_params.tool_id,
            validated_params.include_schema,
            validated_params.include_examples,
            validated_params.include_relationships,
            validated_params.include_usage_guidance,
        )

        # Create detailed response
        response = ToolDetailsResponse(
            tool_id=tool_id,
            tool_details=tool_details,
            included_sections={
                "schema": include_schema,
                "examples": include_examples,
                "relationships": include_relationships,
                "usage_guidance": include_usage_guidance,
            },
        )

        # Add optional sections based on request
        if include_schema:
            response.parameter_schema = _generate_parameter_schema(tool_details)

        if include_examples:
            response.usage_examples = _format_usage_examples(tool_details.examples)

        if include_relationships:
            response.relationships = _get_tool_relationships(ToolId(tool_id))

        if include_usage_guidance:
            response.usage_guidance = _generate_usage_guidance(tool_details)

        logger.info("Tool details retrieved for %s in %.3fs", tool_id, time.time() - start_time)

        return response.to_dict()

    except Exception as e:
        error_response = create_error_response("get_tool_details", str(e), tool_id)
        logger.error("Error in get_tool_details: %s (context: %s)", str(e), tool_id or "none")
        return error_response


@standardize_mcp_tool_response
def list_tool_categories() -> dict:
    """
    Get overview of available tool categories and their contents.

    This tool provides a structured overview of all available tool categories,
    helping Claude understand the organization of available tools and browse
    by functional area.

    Returns:
        Dictionary of categories with tool counts, descriptions, and representative tools

    Examples:
        list_tool_categories()
    """
    try:
        logger.info("Loading tool categories overview")
        start_time = time.time()

        # Load category information from catalog
        categories = load_tool_categories()

        # Create category overview response
        response = ToolCategoriesResponse(categories=categories)

        # Add category descriptions and representative tools
        for category in response.categories:
            category.add_description(get_category_description(category.name))
            category.add_representative_tools(
                get_representative_tools(category.name, limit=SEARCH_CONFIG["representative_tools_limit"])
            )

        logger.info("Tool categories loaded: %d categories in %.3fs", len(categories), time.time() - start_time)

        return response.to_dict()

    except Exception as e:
        error_response = create_error_response("list_tool_categories", str(e), None)
        logger.error("Error in list_tool_categories: %s", str(e))
        return error_response


@log_operation("capability_search")
@standardize_mcp_tool_response
def search_tools_by_capability(
    capability_description: str,
    required_parameters: Optional[List[str]] = None,
    preferred_complexity: str = "any",
    max_results: int = 10,
) -> dict:
    """
    Search for tools by specific capability requirements.

    This tool finds tools that have specific capabilities or parameter requirements,
    enabling precise tool matching for technical requirements.

    Args:
        capability_description: Description of required capability
        required_parameters: List of parameter names that must be supported
        preferred_complexity: Complexity preference ('simple', 'moderate', 'complex', 'any')
        max_results: Maximum number of tools to return

    Returns:
        Tools matching capability requirements with explanations

    Examples:
        search_tools_by_capability("timeout support")
        search_tools_by_capability("file path handling", required_parameters=["file_path"])
        search_tools_by_capability("error handling", preferred_complexity="simple")
    """
    try:
        logger.info("Starting capability search: %s", capability_description)
        start_time = time.time()

        # Validate parameters
        validator = _get_validator()
        validated_params = validator.validate_capability_parameters(
            capability_description, required_parameters, preferred_complexity, max_results
        )

        # Additional runtime validations
        _validate_capability_business_logic(validated_params)

        # Search by capability using parameter-aware search
        parameter_search_engine = _get_parameter_search_engine()

        # Convert complexity preference
        from parameter_ranking import ComplexityPreference

        complexity_map = {
            "simple": ComplexityPreference.SIMPLE,
            "moderate": ComplexityPreference.MODERATE,
            "complex": ComplexityPreference.COMPLEX,
            "any": ComplexityPreference.ANY,
        }
        complexity_pref = complexity_map.get(validated_params.preferred_complexity, ComplexityPreference.ANY)

        capability_results = parameter_search_engine.search_by_parameters(
            required_parameters=validated_params.required_parameters,
            complexity_preference=complexity_pref,
            k=validated_params.max_results,
        )

        # Convert results to capability matches
        capability_matches = []
        for tool_result in capability_results:
            capability_match = CapabilityMatch(
                tool=tool_result,
                capability_score=tool_result.relevance_score,
                parameter_match_score=_calculate_parameter_match_score(
                    tool_result, validated_params.required_parameters
                ),
                complexity_match_score=_calculate_complexity_match_score(tool_result, complexity_pref),
                overall_match_score=tool_result.relevance_score,
                match_explanation=_generate_capability_match_explanation(tool_result, validated_params),
            )
            capability_matches.append(capability_match)

        # Create capability search response
        response = CapabilitySearchResponse(
            capability_description=capability_description,
            results=capability_matches,
            execution_time=time.time() - start_time,
            search_criteria={
                "required_parameters": validated_params.required_parameters,
                "preferred_complexity": validated_params.preferred_complexity,
                "max_results": validated_params.max_results,
            },
        )

        logger.info(
            "Capability search completed: %d results in %.3fs", len(capability_matches), response.execution_time
        )

        return response.to_dict()

    except Exception as e:
        error_response = create_error_response("search_tools_by_capability", str(e), capability_description)
        logger.error("Error in search_tools_by_capability: %s (context: %s)", str(e), capability_description or "none")
        return error_response


# Helper functions


@_with_db_retry()
def _load_basic_tool_data(db_manager: DatabaseManager, tool_id: str) -> ToolSearchResult:
    """Load basic tool information from database."""
    tool_data = db_manager.get_mcp_tool(tool_id)
    if not tool_data:
        raise ValueError(f"Tool {tool_id} not found in database")

    return ToolSearchResult(
        tool_id=ToolId(tool_id),
        name=tool_data.get("name", tool_id),
        description=tool_data.get("description", ""),
        category=tool_data.get("category", "unknown"),
        tool_type=tool_data.get("tool_type", "unknown"),
        similarity_score=SEARCH_CONFIG["scoring"]["perfect_match"],  # Perfect match for direct lookup
        relevance_score=SEARCH_CONFIG["scoring"]["perfect_match"],
        confidence_level="high",
    )


@_with_db_retry()
def _load_tool_parameters(db_manager: DatabaseManager, tool_result: ToolSearchResult, tool_id: str) -> None:
    """Load tool parameters and add to tool result."""
    parameters = db_manager.get_tool_parameters(ToolId(tool_id))
    tool_result.parameters = [
        ParameterAnalysis(
            name=p["parameter_name"],
            type=p["parameter_type"] or "string",
            required=p["is_required"],
            description=p["description"] or "",
        )
        for p in parameters
    ]


@_with_db_retry()
def _load_tool_examples(db_manager: DatabaseManager, tool_result: ToolSearchResult, tool_id: str) -> None:
    """Load tool examples and add to tool result."""
    examples = db_manager.get_tool_examples(ToolId(tool_id))
    tool_result.examples = [
        ToolExample(
            use_case=ex["use_case"] or "",
            example_call=ex["example_call"] or "",
            expected_output=ex["expected_output"] or "",
            context=ex["context"] or "",
            effectiveness_score=ex["effectiveness_score"] or SEARCH_CONFIG["scoring"]["default_effectiveness"],
        )
        for ex in examples
    ]


def load_tool_details(
    tool_id: str,
    include_schema: bool,
    include_examples: bool,
    include_relationships: bool,
    include_usage_guidance: bool,
) -> ToolSearchResult:
    """Load comprehensive tool details from the database."""
    db_manager = _get_db_manager()

    # Load basic tool information
    tool_result = _load_basic_tool_data(db_manager, tool_id)

    # Load additional data based on flags
    if include_schema:
        _load_tool_parameters(db_manager, tool_result, tool_id)

    if include_examples:
        _load_tool_examples(db_manager, tool_result, tool_id)

    # Note: include_relationships and include_usage_guidance are not yet implemented
    # These would be added as additional helper functions when the functionality is built

    return tool_result


@_with_db_retry()
def load_tool_categories() -> List[ToolCategory]:
    """Load available tool categories from the database with caching."""
    # Try to get from cache first
    cache_key = "tool_categories"
    cached_result = _cache.get(cache_key)
    if cached_result is not None:
        logger.debug("Returning cached tool categories (%d categories)", len(cached_result))
        return cached_result

    try:
        db_manager = _get_db_manager()

        with db_manager.get_connection() as conn:
            # Get category statistics from database
            category_stats = conn.execute(
                """
                SELECT category, COUNT(*) as tool_count
                FROM mcp_tools
                GROUP BY category
                ORDER BY tool_count DESC
            """
            ).fetchall()

            categories = []
            for category_name, tool_count in category_stats:
                category = ToolCategory(
                    name=category_name or "unknown",
                    description="",  # Will be populated by get_category_description
                    tool_count=tool_count,
                )
                categories.append(category)

            # Cache the result before returning
            _cache.set(cache_key, categories, CACHE_CONFIG["category_cache_ttl"])
            logger.debug(
                "Cached tool categories (%d categories) for %.1f seconds",
                len(categories),
                CACHE_CONFIG["category_cache_ttl"],
            )

            return categories

    except Exception as e:
        logger.error("Error loading tool categories: %s", e)
        raise


def get_category_description(category_name: str) -> str:
    """Get description for a tool category."""
    descriptions = {
        "file_ops": "Tools for reading, writing, and manipulating files and directories",
        "execution": "Tools for executing commands, scripts, and system operations",
        "search": "Tools for searching and discovering content across various sources",
        "web": "Tools for web scraping, HTTP requests, and internet operations",
        "analysis": "Tools for analyzing data, code, and system information",
        "development": "Tools for software development and debugging",
        "notebook": "Tools for working with Jupyter notebooks and interactive documents",
        "workflow": "Tools for automating and managing complex workflows",
        "data": "Tools for data processing, transformation, and manipulation",
        "system": "Tools for system administration and monitoring",
        "unknown": "Uncategorized tools with various functionalities",
    }
    return descriptions.get(category_name, f"Tools in the {category_name} category")


@_with_db_retry()
def get_representative_tools(category_name: str, limit: int = None) -> List[str]:
    """Get representative tools for a category with caching."""
    if limit is None:
        limit = SEARCH_CONFIG["representative_tools_limit"]

    # Try to get from cache first
    cache_key = f"representative_tools_{category_name}_{limit}"
    cached_result = _cache.get(cache_key)
    if cached_result is not None:
        logger.debug("Returning cached representative tools for %s (%d tools)", category_name, len(cached_result))
        return cached_result

    try:
        db_manager = _get_db_manager()

        with db_manager.get_connection() as conn:
            # Get most common tools in this category
            tools = conn.execute(
                """
                SELECT name
                FROM mcp_tools
                WHERE category = ?
                ORDER BY name
                LIMIT ?
            """,
                (category_name, limit),
            ).fetchall()

            tools_list = [tool[0] for tool in tools]

            # Cache the result before returning
            _cache.set(cache_key, tools_list, CACHE_CONFIG["representative_tools_cache_ttl"])
            logger.debug(
                "Cached representative tools for %s (%d tools) for %.1f seconds",
                category_name,
                len(tools_list),
                CACHE_CONFIG["representative_tools_cache_ttl"],
            )

            return tools_list

    except Exception as e:
        logger.error("Error getting representative tools for %s: %s", category_name, e)
        raise


def generate_query_suggestions(query: str, results: List[ToolSearchResult]) -> List[str]:
    """
    Generate intelligent query refinement suggestions based on search results.

    This function analyzes the number and quality of search results to provide
    contextual suggestions for improving the search query. The suggestions are
    tailored to help users refine their queries for better tool discovery.

    Algorithm:
    - For empty results: Suggests broadening the search scope
    - For too few results: Suggests adding more specific context
    - For too many results: Suggests narrowing the search scope
    - For normal results: Provides advanced refinement tips

    Args:
        query: The original search query string that was executed
        results: List of ToolSearchResult objects returned from the search
                Empty list indicates no tools matched the query

    Returns:
        List of human-readable suggestion strings that help users refine their
        search queries. Suggestions are ordered by usefulness, with the most
        actionable suggestions first.

    Examples:
        >>> suggestions = generate_query_suggestions("file", [])
        >>> print(suggestions[0])
        "Try broader terms related to 'file'"

        >>> suggestions = generate_query_suggestions("very specific query", many_results)
        >>> print(suggestions[0])
        "Add more specific context to 'very specific query' to narrow results"
    """
    suggestions = []

    if not results:
        suggestions.extend(
            [
                f"Try broader terms related to '{query}'",
                "Check if the tool catalog is up to date",
                "Consider using synonyms or related technical terms",
            ]
        )
    elif len(results) < SEARCH_CONFIG["suggestion_thresholds"]["few_results"]:
        suggestions.extend(
            [
                f"Try more specific terms to refine '{query}'",
                "Add programming language or framework context",
                "Include related function or parameter names",
            ]
        )
    elif len(results) > SEARCH_CONFIG["suggestion_thresholds"]["many_results"]:
        suggestions.extend(
            [
                f"Add more specific context to '{query}' to narrow results",
                "Include tool type or category constraints",
                "Focus on particular implementation patterns",
            ]
        )

    return suggestions


def generate_navigation_hints(results: List[ToolSearchResult]) -> List[str]:
    """
    Generate contextual navigation hints to help users understand search results.

    This function analyzes the characteristics of search results to provide
    actionable insights about the tools that were found. It helps users understand
    the complexity distribution, availability of examples, and other metadata
    that can guide their tool selection decisions.

    Algorithm:
    - Counts total results and provides overview
    - Analyzes tool complexity distribution (simple vs complex tools)
    - Identifies tools with comprehensive usage examples
    - Provides guidance on parameter requirements and usage patterns

    Args:
        results: List of ToolSearchResult objects to analyze for hints
                Can be empty list if no results were found

    Returns:
        List of human-readable hint strings that help users navigate and
        understand the search results. Hints are ordered by importance,
        with overview information first followed by specific characteristics.

    Examples:
        >>> hints = generate_navigation_hints([simple_tool, complex_tool])
        >>> print(hints[0])
        "Found 2 matching tools"

        >>> hints = generate_navigation_hints([tool_with_examples])
        >>> print(hints[-1])
        "1 tools have detailed usage examples"

    Note:
        This function assumes that ToolSearchResult objects implement
        the methods is_simple_tool(), is_complex_tool(), and has_good_examples().
        If these methods are not available, some hints may be skipped.
    """
    hints = []

    if results:
        hints.append(f"Found {len(results)} matching tools")

        # Check for different complexity levels
        simple_tools = sum(1 for r in results if r.is_simple_tool())
        complex_tools = sum(1 for r in results if r.is_complex_tool())

        if simple_tools > 0:
            hints.append(f"{simple_tools} tools are simple to use with few parameters")
        if complex_tools > 0:
            hints.append(f"{complex_tools} tools offer advanced functionality with more parameters")

        # Check for tools with good examples
        tools_with_examples = sum(1 for r in results if r.has_good_examples())
        if tools_with_examples > 0:
            hints.append(f"{tools_with_examples} tools have detailed usage examples")

    return hints


def _generate_parameter_schema(tool_details: ToolSearchResult) -> Dict[str, Any]:
    """Generate JSON schema for tool parameters."""
    schema = {"type": "object", "properties": {}, "required": []}

    for param in tool_details.parameters:
        param_schema = {"type": param.type, "description": param.description}

        if param.default_value is not None:
            param_schema["default"] = param.default_value

        if param.examples:
            param_schema["examples"] = param.examples

        if param.constraints:
            param_schema.update(param.constraints)

        schema["properties"][param.name] = param_schema

        if param.required:
            schema["required"].append(param.name)

    return schema


def _format_usage_examples(examples: List[ToolExample]) -> List[Dict[str, Any]]:
    """Format tool examples for response."""
    return [example.to_dict() for example in examples]


def _get_tool_relationships(tool_id: ToolId) -> Dict[str, List[str]]:
    """Get relationships for a tool."""
    try:
        db_manager = _get_db_manager()

        with db_manager.get_connection() as conn:
            # Get relationships from database
            relationships = conn.execute(
                """
                SELECT relationship_type, related_tool_id
                FROM tool_relationships
                WHERE tool_id = ?
            """,
                (str(tool_id),),
            ).fetchall()

            result = {}
            for rel_type, related_id in relationships:
                if rel_type not in result:
                    result[rel_type] = []
                result[rel_type].append(related_id)

            return result

    except Exception as e:
        logger.error("Error getting relationships for %s: %s", tool_id, e)
        raise


def _generate_usage_guidance(tool_details: ToolSearchResult) -> Dict[str, List[str]]:
    """Generate usage guidance for a tool."""
    guidance = {"best_practices": [], "common_pitfalls": [], "tips": []}

    # Add guidance based on tool characteristics
    if tool_details.is_simple_tool():
        guidance["tips"].append("This is a simple tool with minimal parameters")
        guidance["best_practices"].append("Ideal for quick operations and scripting")
    elif tool_details.is_complex_tool():
        guidance["tips"].append("This is a complex tool with many configuration options")
        guidance["best_practices"].append("Review all parameters before use")
        guidance["common_pitfalls"].append("Don't forget to set required parameters")

    if tool_details.has_good_examples():
        guidance["tips"].append("Check the examples for common usage patterns")

    return guidance


def _calculate_parameter_match_score(tool_result: ToolSearchResult, required_params: List[str]) -> float:
    """
    Calculate parameter match score for capability-based tool search.

    This function computes a normalized score (0.0 to 1.0) indicating how well
    a tool's available parameters match the user's required parameters. The
    algorithm performs case-insensitive matching and returns a perfect score
    if no specific parameters are required.

    Args:
        tool_result: ToolSearchResult containing the tool's parameter information
        required_params: List of parameter names that the user requires
                        Empty list means no specific parameters required

    Returns:
        Float between 0.0 and 1.0 where:
        - 1.0: Perfect match (all required parameters available)
        - 0.5-0.9: Partial match (some required parameters available)
        - 0.0: No match (none of the required parameters available)

    Algorithm:
        - If no parameters required, returns 1.0 (perfect match)
        - Performs case-insensitive matching of parameter names
        - Calculates ratio of matched parameters to total required parameters

    Examples:
        >>> score = _calculate_parameter_match_score(tool, ["file_path", "timeout"])
        >>> # If tool has both parameters: score = 1.0
        >>> # If tool has one parameter: score = 0.5
        >>> # If tool has neither: score = 0.0
    """
    if not required_params:
        return 1.0

    tool_param_names = {p.name.lower() for p in tool_result.parameters}
    matches = sum(1 for param in required_params if param.lower() in tool_param_names)

    return matches / len(required_params) if required_params else 1.0


def _calculate_complexity_match_score(tool_result: ToolSearchResult, complexity_pref) -> float:
    """Calculate how well tool complexity matches preference."""
    tool_complexity = tool_result.complexity_score

    if complexity_pref.value == "any":
        return 1.0
    elif complexity_pref.value == "simple":
        return 1.0 - tool_complexity  # Lower complexity is better
    elif complexity_pref.value == "moderate":
        return 1.0 - abs(tool_complexity - 0.5) * 2  # Closer to 0.5 is better
    elif complexity_pref.value == "complex":
        return tool_complexity  # Higher complexity is better
    else:
        return 0.5


def _generate_capability_match_explanation(tool_result: ToolSearchResult, params: ValidatedCapabilityParams) -> str:
    """Generate explanation for why tool matches capability requirements."""
    explanations = []

    # Parameter match explanation
    if params.required_parameters:
        param_score = _calculate_parameter_match_score(tool_result, params.required_parameters)
        if param_score > 0.8:
            explanations.append("Strong parameter match")
        elif param_score > 0.5:
            explanations.append("Partial parameter match")
        else:
            explanations.append("Limited parameter match")

    # Complexity explanation
    if params.preferred_complexity != "any":
        if tool_result.is_simple_tool() and params.preferred_complexity == "simple":
            explanations.append("Matches simple complexity preference")
        elif tool_result.is_complex_tool() and params.preferred_complexity == "complex":
            explanations.append("Matches complex tool preference")
        elif (
            params.preferred_complexity == "moderate"
            and not tool_result.is_simple_tool()
            and not tool_result.is_complex_tool()
        ):
            explanations.append("Matches moderate complexity preference")

    return "; ".join(explanations) if explanations else "General capability match"
