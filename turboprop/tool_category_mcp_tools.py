#!/usr/bin/env python3
"""
tool_category_mcp_tools.py: MCP tools for tool category browsing and organization.

This module implements MCP tools that enable browsing tools by category,
understanding tool organization, and making informed tool selection decisions.
"""

from typing import Any, Dict, List, Optional

from .comparison_response_types import (
    CategoryBrowseResponse,
    CategoryOverviewResponse,
    SelectionContext,
    SelectionGuidanceResponse,
    create_error_response,
)
from .context_analyzer import ContextAnalyzer
from .decision_support import DecisionSupport
from .logging_config import get_logger
from .task_analyzer import TaskAnalyzer

logger = get_logger(__name__)

# Valid categories - should be updated based on actual tool ecosystem
VALID_CATEGORIES = [
    "file_ops",  # File operations (read, write, edit, etc.)
    "web",  # Web operations (WebFetch, WebSearch, etc.)
    "execution",  # Command execution (bash, task, etc.)
    "search",  # Search operations (grep, glob, etc.)
    "analysis",  # Analysis tools (code analysis, etc.)
    "development",  # Development tools
    "data",  # Data processing tools
    "system",  # System tools
    "utility",  # Utility tools
    "integration",  # Integration tools
]

# Global components - will be initialized by the initialization function
_tool_catalog = None
_context_analyzer: Optional[ContextAnalyzer] = None
_task_analyzer: Optional[TaskAnalyzer] = None
_decision_support: Optional[DecisionSupport] = None


def initialize_category_tools(
    tool_catalog,
    context_analyzer: ContextAnalyzer,
    task_analyzer: TaskAnalyzer,
    decision_support: DecisionSupport,
) -> None:
    """Initialize the category tools with required components."""
    global _tool_catalog, _context_analyzer, _task_analyzer, _decision_support

    _tool_catalog = tool_catalog
    _context_analyzer = context_analyzer
    _task_analyzer = task_analyzer
    _decision_support = decision_support

    logger.info("Tool category MCP tools initialized")


def browse_tools_by_category(
    category: str,
    sort_by: str = "popularity",
    max_tools: int = 20,
    include_descriptions: bool = True,
    complexity_filter: Optional[str] = None,
) -> dict:
    """
    Browse tools within a specific category.

    This tool enables systematic exploration of tools by category,
    helping to discover tools with similar functionality and purposes.

    Args:
        category: Category to browse (file_ops, web, analysis, etc.)
        sort_by: Sorting method ('popularity', 'complexity', 'name', 'functionality')
        max_tools: Maximum number of tools to return
        include_descriptions: Whether to include tool descriptions
        complexity_filter: Filter by complexity ('simple', 'moderate', 'complex')

    Returns:
        List of tools in category with metadata and organization

    Examples:
        browse_tools_by_category("file_ops")
        browse_tools_by_category("web", sort_by="complexity", complexity_filter="simple")
    """
    try:
        # Validate category
        if category not in VALID_CATEGORIES:
            return create_error_response(
                "browse_tools_by_category",
                f"Invalid category '{category}'. Valid options: {', '.join(VALID_CATEGORIES)}",
            )

        # Check if tool catalog is initialized
        if not _tool_catalog:
            return create_error_response("browse_tools_by_category", "Tool catalog not initialized")

        # Get tools in category
        category_tools = _tool_catalog.get_tools_by_category(
            category=category, sort_by=sort_by, max_tools=max_tools, complexity_filter=complexity_filter
        )

        # Create response
        response = CategoryBrowseResponse(
            category=category, tools=category_tools, sort_by=sort_by, complexity_filter=complexity_filter
        )

        # Add category overview
        overview = get_category_overview_data(category)
        response.add_category_overview(overview)

        return response.to_dict()

    except Exception as e:
        logger.error(f"Error in browse_tools_by_category: {e}")
        return create_error_response("browse_tools_by_category", str(e), category)


def get_category_overview() -> dict:
    """
    Get overview of all tool categories and their characteristics.

    This tool provides a high-level view of the tool ecosystem,
    helping to understand the organization and scope of available tools.

    Returns:
        Comprehensive overview of all tool categories with statistics

    Examples:
        get_category_overview()
    """
    try:
        # Check if tool catalog is initialized
        if not _tool_catalog:
            return create_error_response("get_category_overview", "Tool catalog not initialized")

        # Get category statistics and overviews
        categories = _tool_catalog.get_all_categories()

        # Create comprehensive overview
        response = CategoryOverviewResponse(categories=categories)

        # Add ecosystem statistics
        stats = calculate_ecosystem_statistics(categories)
        response.add_ecosystem_stats(stats)

        return response.to_dict()

    except Exception as e:
        logger.error(f"Error in get_category_overview: {e}")
        return create_error_response("get_category_overview", str(e))


def get_tool_selection_guidance(
    task_description: str,
    available_tools: Optional[List[str]] = None,
    constraints: Optional[List[str]] = None,
    optimization_goal: str = "balanced",
) -> dict:
    """
    Get guidance for selecting the optimal tool for a specific task.

    This tool provides decision support for tool selection, considering
    task requirements, available options, constraints, and optimization goals.

    Args:
        task_description: Description of the task requiring tool selection
        available_tools: List of tools to choose from (if limited)
        constraints: Constraints to consider (e.g., "no complex tools", "performance critical")
        optimization_goal: What to optimize for ('speed', 'reliability', 'simplicity', 'balanced')

    Returns:
        Tool selection guidance with reasoning and alternatives

    Examples:
        get_tool_selection_guidance("read configuration file safely")
        get_tool_selection_guidance("process large files", constraints=["performance critical"])
    """
    try:
        # Check if required components are initialized
        if not _task_analyzer:
            return create_error_response("get_tool_selection_guidance", "Task analyzer not initialized")
        if not _decision_support:
            return create_error_response("get_tool_selection_guidance", "Decision support not initialized")

        # Analyze task and create context
        task_analysis = _task_analyzer.analyze_task(task_description)
        selection_context = create_selection_context(
            constraints=constraints, optimization_goal=optimization_goal, available_tools=available_tools
        )

        # Get selection guidance
        guidance = _decision_support.get_tool_selection_guidance(task_analysis=task_analysis, context=selection_context)

        # Create response
        response = SelectionGuidanceResponse(
            task_description=task_description,
            guidance=guidance,
            task_analysis=task_analysis,
            selection_context=selection_context,
        )

        return response.to_dict()

    except Exception as e:
        logger.error(f"Error in get_tool_selection_guidance: {e}")
        return create_error_response("get_tool_selection_guidance", str(e), task_description)


def find_tools_by_complexity(
    complexity_level: str, category_filter: Optional[str] = None, max_tools: int = 15, sort_by: str = "popularity"
) -> dict:
    """
    Find tools by complexity level.

    This tool helps discover tools based on their complexity level,
    useful for matching tools to user skill levels.

    Args:
        complexity_level: Complexity to filter by ('simple', 'moderate', 'complex')
        category_filter: Optional category filter
        max_tools: Maximum number of tools to return
        sort_by: Sorting method ('popularity', 'name', 'functionality')

    Returns:
        Tools filtered by complexity level
    """
    try:
        # Validate complexity level
        valid_complexity = ["simple", "moderate", "complex"]
        if complexity_level not in valid_complexity:
            return create_error_response(
                "find_tools_by_complexity",
                f"Invalid complexity '{complexity_level}'. Valid options: {', '.join(valid_complexity)}",
            )

        # Check if tool catalog is initialized
        if not _tool_catalog:
            return create_error_response("find_tools_by_complexity", "Tool catalog not initialized")

        # Find tools by complexity
        tools = _tool_catalog.get_tools_by_complexity(
            complexity_level=complexity_level, category_filter=category_filter, max_tools=max_tools, sort_by=sort_by
        )

        return {
            "complexity_level": complexity_level,
            "category_filter": category_filter,
            "tools": [tool.to_dict() for tool in tools],
            "total_tools": len(tools),
            "sort_by": sort_by,
            "success": True,
        }

    except Exception as e:
        logger.error(f"Error in find_tools_by_complexity: {e}")
        return create_error_response("find_tools_by_complexity", str(e), complexity_level)


def explore_tool_ecosystem(
    focus_area: Optional[str] = None,
    max_categories: int = 10,
    tools_per_category: int = 5,
    include_relationships: bool = True,
) -> dict:
    """
    Explore the tool ecosystem with comprehensive overview.

    This tool provides a comprehensive view of the tool ecosystem,
    showing categories, representative tools, and relationships.

    Args:
        focus_area: Optional area to focus on (e.g., "file processing", "web operations")
        max_categories: Maximum categories to include
        tools_per_category: Tools to show per category
        include_relationships: Whether to include tool relationships

    Returns:
        Comprehensive tool ecosystem exploration
    """
    try:
        # Check if tool catalog is initialized
        if not _tool_catalog:
            return create_error_response("explore_tool_ecosystem", "Tool catalog not initialized")

        # Get ecosystem exploration data
        ecosystem_data = _tool_catalog.explore_ecosystem(
            focus_area=focus_area,
            max_categories=max_categories,
            tools_per_category=tools_per_category,
            include_relationships=include_relationships,
        )

        return {
            "focus_area": focus_area,
            "ecosystem_data": ecosystem_data,
            "max_categories": max_categories,
            "tools_per_category": tools_per_category,
            "include_relationships": include_relationships,
            "success": True,
        }

    except Exception as e:
        logger.error(f"Error in explore_tool_ecosystem: {e}")
        return create_error_response("explore_tool_ecosystem", str(e), str(focus_area))


# Utility functions
def get_category_overview_data(category: str) -> Dict[str, Any]:
    """Get overview data for a specific category"""
    try:
        if not _tool_catalog:
            return {}

        return _tool_catalog.get_category_overview(category)
    except Exception as e:
        logger.warning(f"Failed to get category overview for {category}: {e}")
        return {}


def create_selection_context(
    constraints: Optional[List[str]] = None,
    optimization_goal: str = "balanced",
    available_tools: Optional[List[str]] = None,
) -> SelectionContext:
    """Create selection context from parameters"""
    return SelectionContext(
        constraints=constraints, optimization_goal=optimization_goal, available_tools=available_tools
    )


def calculate_ecosystem_statistics(categories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate ecosystem-wide statistics"""
    if not categories:
        return {}

    total_tools = sum(cat.get("tool_count", 0) for cat in categories)
    avg_tools_per_category = total_tools / len(categories) if categories else 0

    most_populated = max(categories, key=lambda c: c.get("tool_count", 0)) if categories else None

    return {
        "total_categories": len(categories),
        "total_tools": total_tools,
        "average_tools_per_category": round(avg_tools_per_category, 1),
        "most_populated_category": most_populated.get("name") if most_populated else None,
        "ecosystem_maturity": "mature" if total_tools > 50 else "developing",
        "categories_with_tools": len([cat for cat in categories if cat.get("tool_count", 0) > 0]),
    }
