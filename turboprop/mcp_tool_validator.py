#!/usr/bin/env python3
"""
MCP Tool Validator

This module provides comprehensive parameter validation and error handling
for MCP tool operations, ensuring robust input validation and helpful error messages.
"""

from dataclasses import dataclass
from typing import List, Optional

from .mcp_tool_search_responses import create_error_response

# Valid categories for MCP tools
VALID_CATEGORIES = [
    "file_ops",
    "execution",
    "search",
    "web",
    "analysis",
    "development",
    "notebook",
    "workflow",
    "data",
    "system",
]

# Valid tool types
VALID_TOOL_TYPES = ["system", "custom", "third_party"]

# Valid search modes
VALID_SEARCH_MODES = ["semantic", "hybrid", "keyword"]

# Valid complexity preferences
VALID_COMPLEXITY_PREFERENCES = ["simple", "moderate", "complex", "any"]

# Default parameters for MCP tools
DEFAULT_MAX_RESULTS = 10
DEFAULT_SEARCH_MODE = "hybrid"
DEFAULT_INCLUDE_EXAMPLES = True

# Parameter limits
MIN_MAX_RESULTS = 1
MAX_MAX_RESULTS = 50
MAX_QUERY_LENGTH = 1000
MAX_TOOL_ID_LENGTH = 100


@dataclass
class ValidatedSearchParams:
    """Validated parameters for search operations."""

    query: str
    category: Optional[str]
    tool_type: Optional[str]
    max_results: int
    include_examples: bool
    search_mode: str


@dataclass
class ValidatedToolDetailsParams:
    """Validated parameters for tool details operations."""

    tool_id: str
    include_schema: bool
    include_examples: bool
    include_relationships: bool
    include_usage_guidance: bool


@dataclass
class ValidatedCapabilityParams:
    """Validated parameters for capability search operations."""

    capability_description: str
    required_parameters: List[str]
    preferred_complexity: str
    max_results: int


class MCPToolValidator:
    """Validation for MCP tool parameters and responses."""

    def validate_search_parameters(
        self,
        query: str,
        category: Optional[str],
        tool_type: Optional[str],
        max_results: int,
        include_examples: bool,
        search_mode: str,
    ) -> ValidatedSearchParams:
        """
        Validate search_mcp_tools parameters.

        Args:
            query: Search query string
            category: Optional tool category filter
            tool_type: Optional tool type filter
            max_results: Maximum number of results
            include_examples: Whether to include examples
            search_mode: Search strategy mode

        Returns:
            ValidatedSearchParams object

        Raises:
            ValueError: If any parameter is invalid
        """
        errors = []

        # Validate query
        if not query or not query.strip():
            errors.append("Query cannot be empty")
        elif len(query) > MAX_QUERY_LENGTH:
            errors.append(f"Query too long (max {MAX_QUERY_LENGTH} characters)")

        # Validate category
        if category and category not in VALID_CATEGORIES:
            errors.append(f"Invalid category '{category}'. Valid options: {', '.join(VALID_CATEGORIES)}")

        # Validate tool_type
        if tool_type and tool_type not in VALID_TOOL_TYPES:
            errors.append(f"Invalid tool_type '{tool_type}'. Valid options: {', '.join(VALID_TOOL_TYPES)}")

        # Validate max_results
        if max_results < MIN_MAX_RESULTS or max_results > MAX_MAX_RESULTS:
            errors.append(f"max_results must be between {MIN_MAX_RESULTS} and {MAX_MAX_RESULTS}")

        # Validate search_mode
        if search_mode not in VALID_SEARCH_MODES:
            errors.append(f"Invalid search_mode '{search_mode}'. Valid options: {', '.join(VALID_SEARCH_MODES)}")

        if errors:
            raise ValueError(f"Parameter validation errors: {'; '.join(errors)}")

        return ValidatedSearchParams(
            query=query.strip(),
            category=category,
            tool_type=tool_type,
            max_results=max_results,
            include_examples=include_examples,
            search_mode=search_mode,
        )

    def validate_tool_details_parameters(
        self,
        tool_id: str,
        include_schema: bool,
        include_examples: bool,
        include_relationships: bool,
        include_usage_guidance: bool,
    ) -> ValidatedToolDetailsParams:
        """
        Validate get_tool_details parameters.

        Args:
            tool_id: Tool identifier
            include_schema: Whether to include parameter schema
            include_examples: Whether to include usage examples
            include_relationships: Whether to include tool relationships
            include_usage_guidance: Whether to include usage guidance

        Returns:
            ValidatedToolDetailsParams object

        Raises:
            ValueError: If any parameter is invalid
        """
        errors = []

        # Validate tool_id
        if not tool_id or not tool_id.strip():
            errors.append("Tool ID cannot be empty")
        elif len(tool_id) > MAX_TOOL_ID_LENGTH:
            errors.append(f"Tool ID too long (max {MAX_TOOL_ID_LENGTH} characters)")

        # Validate boolean parameters (just ensure they are actually booleans)
        for param_name, param_value in [
            ("include_schema", include_schema),
            ("include_examples", include_examples),
            ("include_relationships", include_relationships),
            ("include_usage_guidance", include_usage_guidance),
        ]:
            if not isinstance(param_value, bool):
                errors.append(f"{param_name} must be a boolean value")

        if errors:
            raise ValueError(f"Parameter validation errors: {'; '.join(errors)}")

        return ValidatedToolDetailsParams(
            tool_id=tool_id.strip(),
            include_schema=include_schema,
            include_examples=include_examples,
            include_relationships=include_relationships,
            include_usage_guidance=include_usage_guidance,
        )

    def validate_capability_parameters(
        self,
        capability_description: str,
        required_parameters: Optional[List[str]],
        preferred_complexity: str,
        max_results: int,
    ) -> ValidatedCapabilityParams:
        """
        Validate search_tools_by_capability parameters.

        Args:
            capability_description: Description of required capability
            required_parameters: List of required parameter names
            preferred_complexity: Complexity preference
            max_results: Maximum number of results

        Returns:
            ValidatedCapabilityParams object

        Raises:
            ValueError: If any parameter is invalid
        """
        errors = []

        # Validate capability_description
        if not capability_description or not capability_description.strip():
            errors.append("Capability description cannot be empty")
        elif len(capability_description) > MAX_QUERY_LENGTH:
            errors.append(f"Capability description too long (max {MAX_QUERY_LENGTH} characters)")

        # Validate required_parameters
        validated_params = []
        if required_parameters:
            for param in required_parameters:
                if not isinstance(param, str):
                    errors.append("All required parameters must be strings")
                elif not param.strip():
                    errors.append("Parameter names cannot be empty")
                else:
                    validated_params.append(param.strip())

        # Validate preferred_complexity
        if preferred_complexity not in VALID_COMPLEXITY_PREFERENCES:
            errors.append(
                f"Invalid preferred_complexity '{preferred_complexity}'. "
                f"Valid options: {', '.join(VALID_COMPLEXITY_PREFERENCES)}"
            )

        # Validate max_results
        if max_results < MIN_MAX_RESULTS or max_results > MAX_MAX_RESULTS:
            errors.append(f"max_results must be between {MIN_MAX_RESULTS} and {MAX_MAX_RESULTS}")

        if errors:
            raise ValueError(f"Parameter validation errors: {'; '.join(errors)}")

        return ValidatedCapabilityParams(
            capability_description=capability_description.strip(),
            required_parameters=validated_params,
            preferred_complexity=preferred_complexity,
            max_results=max_results,
        )

    def validate_tool_exists(self, tool_id: str, available_tools: List[str]) -> bool:
        """
        Check if a tool exists in the available tools list.

        Args:
            tool_id: Tool identifier to check
            available_tools: List of available tool IDs

        Returns:
            True if tool exists, False otherwise
        """
        return tool_id in available_tools

    def sanitize_query(self, query: str) -> str:
        """
        Sanitize search query by removing potentially problematic characters.

        Args:
            query: Raw query string

        Returns:
            Sanitized query string
        """
        if not query:
            return ""

        # Remove control characters and normalize whitespace
        sanitized = "".join(char for char in query if ord(char) >= 32)
        sanitized = " ".join(sanitized.split())

        return sanitized[:MAX_QUERY_LENGTH]

    def generate_validation_error_response(
        self, tool_name: str, validation_error: ValueError, context: Optional[str] = None
    ) -> dict:
        """
        Generate a formatted error response for validation failures.

        Args:
            tool_name: Name of the MCP tool that failed validation
            validation_error: The validation error that occurred
            context: Optional context information

        Returns:
            Formatted error response dictionary
        """
        return create_error_response(tool_name, str(validation_error), context)


def validate_search_parameters(
    query: str,
    category: Optional[str] = None,
    tool_type: Optional[str] = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    include_examples: bool = DEFAULT_INCLUDE_EXAMPLES,
    search_mode: str = DEFAULT_SEARCH_MODE,
) -> ValidatedSearchParams:
    """
    Convenience function for validating search parameters.

    Args:
        query: Search query string
        category: Optional tool category filter
        tool_type: Optional tool type filter
        max_results: Maximum number of results
        include_examples: Whether to include examples
        search_mode: Search strategy mode

    Returns:
        ValidatedSearchParams object

    Raises:
        ValueError: If any parameter is invalid
    """
    validator = MCPToolValidator()
    return validator.validate_search_parameters(query, category, tool_type, max_results, include_examples, search_mode)


def tool_exists(tool_id: str) -> bool:
    """
    Check if a tool exists in the system.

    This is a placeholder function that would be replaced with actual
    tool existence checking logic.

    Args:
        tool_id: Tool identifier to check

    Returns:
        True if tool exists, False otherwise
    """
    # Placeholder implementation - in practice this would check the database
    # or tool registry
    return bool(tool_id and tool_id.strip())


def generate_error_suggestions(tool_name: str, error_message: str) -> List[str]:
    """
    Generate helpful suggestions based on the error message.

    Args:
        tool_name: Name of the tool that generated the error
        error_message: The error message

    Returns:
        List of helpful suggestion strings
    """
    suggestions = []

    error_lower = error_message.lower()

    if "query" in error_lower and "empty" in error_lower:
        suggestions.extend(
            [
                "Provide a descriptive search query like 'file operations' or 'web scraping'",
                "Use natural language to describe the functionality you need",
                "Try being more specific about what the tool should do",
            ]
        )
    elif "category" in error_lower:
        suggestions.extend(
            [
                f"Valid categories are: {', '.join(VALID_CATEGORIES)}",
                "Leave category empty to search all categories",
                "Use common category names like 'file_ops' or 'web'",
            ]
        )
    elif "tool_type" in error_lower:
        suggestions.extend(
            [
                f"Valid tool types are: {', '.join(VALID_TOOL_TYPES)}",
                "Leave tool_type empty to search all types",
                "Use 'system' for built-in tools or 'custom' for user-defined tools",
            ]
        )
    elif "max_results" in error_lower:
        suggestions.extend(
            [
                f"Use a number between {MIN_MAX_RESULTS} and {MAX_MAX_RESULTS}",
                "Start with a smaller number like 10 for better performance",
                "Increase the limit if you need more comprehensive results",
            ]
        )
    elif "search_mode" in error_lower:
        suggestions.extend(
            [
                f"Valid search modes are: {', '.join(VALID_SEARCH_MODES)}",
                "Use 'hybrid' for best results (combines semantic and keyword search)",
                "Use 'semantic' for meaning-based search or 'keyword' for exact matches",
            ]
        )
    elif "tool" in error_lower and "not found" in error_lower:
        suggestions.extend(
            [
                "Check that the tool name or ID is spelled correctly",
                "Use search_mcp_tools to find available tools",
                "Some tools may not be available in your current environment",
            ]
        )
    else:
        suggestions.append("Check the parameter values and try again")

    return suggestions
