#!/usr/bin/env python3
"""
tool_comparison_mcp_tools.py: MCP tools for tool comparison and relationship analysis.

This module implements MCP tools that enable comprehensive comparison of tools,
finding alternatives, and analyzing tool relationships.
"""

from typing import Any, Dict, List, Optional

from .comparison_response_types import (
    AlternativeAnalysis,
    AlternativesFoundResponse,
    ToolComparisonMCPResponse,
    ToolRelationshipsResponse,
    create_error_response,
)
from .context_analyzer import TaskContext
from .decision_support import DecisionSupport
from .logging_config import get_logger
from .mcp_response_standardizer import standardize_mcp_tool_response
from .mcp_tool_validator import tool_exists
from .tool_comparison_engine import ToolComparisonEngine

logger = get_logger(__name__)

# Global components - will be initialized by the initialization function
_comparison_engine: Optional[ToolComparisonEngine] = None
_decision_support: Optional[DecisionSupport] = None
_alternative_detector = None
_relationship_analyzer = None


def initialize_comparison_tools(
    comparison_engine: ToolComparisonEngine,
    decision_support: DecisionSupport,
    alternative_detector,
    relationship_analyzer,
) -> None:
    """Initialize the comparison tools with required components."""
    global _comparison_engine, _decision_support, _alternative_detector, _relationship_analyzer

    _comparison_engine = comparison_engine
    _decision_support = decision_support
    _alternative_detector = alternative_detector
    _relationship_analyzer = relationship_analyzer

    logger.info("Tool comparison MCP tools initialized")


def create_task_context(description: str) -> Optional[TaskContext]:
    """Create a task context from description if provided."""
    if not description:
        return None

    try:
        from context_analyzer import ContextAnalyzer

        analyzer = ContextAnalyzer()
        return analyzer.analyze_task_context(description)
    except Exception as e:
        logger.warning(f"Failed to create task context: {e}")
        return None


@standardize_mcp_tool_response
def compare_mcp_tools(
    tool_ids: List[str],
    comparison_criteria: Optional[List[str]] = None,
    include_decision_guidance: bool = True,
    comparison_context: Optional[str] = None,
    detail_level: str = "standard",
) -> dict:
    """
    Compare multiple MCP tools across various dimensions.

    This tool provides comprehensive side-by-side comparison of MCP tools,
    helping to understand differences, trade-offs, and optimal use cases for each tool.

    Args:
        tool_ids: List of tool IDs to compare (2-10 tools)
        comparison_criteria: Specific aspects to focus on
                           Options: ['functionality', 'usability', 'performance', 'complexity']
        include_decision_guidance: Whether to include selection recommendations
        comparison_context: Context for the comparison (e.g., "for file processing tasks")
        detail_level: Level of comparison detail ('basic', 'standard', 'comprehensive')

    Returns:
        Comprehensive tool comparison with rankings and decision guidance

    Examples:
        compare_mcp_tools(["read", "write", "edit"])
        compare_mcp_tools(["bash", "task"], comparison_criteria=["usability", "complexity"])
        compare_mcp_tools(["grep", "glob"], comparison_context="for code search")
    """
    try:
        # Validate tool IDs
        if len(tool_ids) < 2:
            return create_error_response("compare_mcp_tools", "At least 2 tools required for comparison")
        elif len(tool_ids) > 10:
            return create_error_response("compare_mcp_tools", "Maximum 10 tools can be compared at once")

        # Validate all tools exist
        missing_tools = [tool_id for tool_id in tool_ids if not tool_exists(tool_id)]
        if missing_tools:
            return create_error_response("compare_mcp_tools", f"Tools not found: {', '.join(missing_tools)}")

        # Check if comparison engine is initialized
        if not _comparison_engine:
            return create_error_response("compare_mcp_tools", "Comparison engine not initialized")

        # Create task context if comparison context provided
        task_context = create_task_context(comparison_context) if comparison_context else None

        # Perform comparison
        comparison_result = _comparison_engine.compare_tools(
            tool_ids=tool_ids,
            comparison_criteria=comparison_criteria or ["functionality", "usability", "complexity"],
            context=task_context,
        )

        # Create response
        response = ToolComparisonMCPResponse(
            tool_ids=tool_ids,
            comparison_result=comparison_result,
            comparison_criteria=comparison_criteria,
            detail_level=detail_level,
        )

        # Add decision guidance if requested
        if include_decision_guidance and _decision_support:
            try:
                guidance = _decision_support.generate_selection_guidance(comparison_result, task_context)
                response.add_decision_guidance(guidance)
            except Exception as e:
                logger.warning(f"Failed to generate decision guidance: {e}")

        return response.to_dict()

    except Exception as e:
        logger.error(f"Error in compare_mcp_tools: {e}")
        return create_error_response("compare_mcp_tools", str(e), str(tool_ids))


@standardize_mcp_tool_response
def find_tool_alternatives(
    reference_tool: str,
    similarity_threshold: float = 0.7,
    max_alternatives: int = 8,
    include_comparison: bool = True,
    context_filter: Optional[str] = None,
) -> dict:
    """
    Find alternative tools similar to a reference tool.

    This tool discovers tools with similar functionality, helping to explore
    different approaches and find optimal tools for specific use cases.

    Args:
        reference_tool: Tool ID to find alternatives for
        similarity_threshold: Minimum similarity score (0.0-1.0)
        max_alternatives: Maximum number of alternatives to return
        include_comparison: Whether to include comparison with reference tool
        context_filter: Optional context to filter alternatives (e.g., "simple tools only")

    Returns:
        Alternative tools with similarity scores and comparisons

    Examples:
        find_tool_alternatives("bash")
        find_tool_alternatives("read", similarity_threshold=0.5, max_alternatives=5)
        find_tool_alternatives("search_code", context_filter="beginner-friendly")
    """
    try:
        # Validate reference tool
        if not tool_exists(reference_tool):
            return create_error_response("find_tool_alternatives", f"Reference tool '{reference_tool}' not found")

        # Check if alternative detector is initialized
        if not _alternative_detector:
            return create_error_response("find_tool_alternatives", "Alternative detector not initialized")

        # Find alternatives
        alternatives = _alternative_detector.find_alternatives(
            reference_tool=reference_tool, similarity_threshold=similarity_threshold, max_alternatives=max_alternatives
        )

        # Apply context filter if specified
        if context_filter:
            alternatives = apply_context_filter(alternatives, context_filter)

        # Create response
        response = AlternativesFoundResponse(
            reference_tool=reference_tool,
            alternatives=alternatives,
            similarity_threshold=similarity_threshold,
            context_filter=context_filter,
        )

        # Add comparisons if requested
        if include_comparison and alternatives:
            try:
                comparisons = generate_alternative_comparisons(reference_tool, alternatives[:3])
                response.add_comparisons(comparisons)
            except Exception as e:
                logger.warning(f"Failed to generate comparisons: {e}")

        return response.to_dict()

    except Exception as e:
        logger.error(f"Error in find_tool_alternatives: {e}")
        return create_error_response("find_tool_alternatives", str(e), reference_tool)


@standardize_mcp_tool_response
def analyze_tool_relationships(
    tool_id: str,
    relationship_types: Optional[List[str]] = None,
    max_relationships: int = 20,
    include_explanations: bool = True,
) -> dict:
    """
    Analyze relationships between a tool and other tools in the ecosystem.

    This tool explores how a tool relates to others, including alternatives,
    complements, prerequisites, and tools it can work with in workflows.

    Args:
        tool_id: Tool ID to analyze relationships for
        relationship_types: Types of relationships to include
                          Options: ['alternatives', 'complements', 'prerequisites', 'dependents']
        max_relationships: Maximum relationships to return per type
        include_explanations: Whether to explain why relationships exist

    Returns:
        Comprehensive relationship analysis with explanations

    Examples:
        analyze_tool_relationships("bash")
        analyze_tool_relationships("read", relationship_types=["alternatives", "complements"])
    """
    try:
        # Validate tool
        if not tool_exists(tool_id):
            return create_error_response("analyze_tool_relationships", f"Tool '{tool_id}' not found")

        # Check if relationship analyzer is initialized
        if not _relationship_analyzer:
            return create_error_response("analyze_tool_relationships", "Relationship analyzer not initialized")

        # Get relationship types
        rel_types = relationship_types or ["alternatives", "complements", "prerequisites"]

        # Analyze relationships
        relationships = _relationship_analyzer.analyze_tool_relationships(
            tool_id=tool_id, relationship_types=rel_types, max_per_type=max_relationships
        )

        # Create response
        response = ToolRelationshipsResponse(tool_id=tool_id, relationships=relationships, relationship_types=rel_types)

        # Add explanations if requested
        if include_explanations:
            explanations = generate_relationship_explanations(tool_id, relationships)
            response.add_explanations(explanations)

        return response.to_dict()

    except Exception as e:
        logger.error(f"Error in analyze_tool_relationships: {e}")
        return create_error_response("analyze_tool_relationships", str(e), tool_id)


@standardize_mcp_tool_response
def get_tool_recommendations_comparison(
    task_description: str,
    candidate_tools: Optional[List[str]] = None,
    max_recommendations: int = 5,
    include_detailed_analysis: bool = True,
) -> dict:
    """
    Compare different tool recommendation options for a task.

    This tool analyzes multiple potential tools for a task and provides
    detailed comparison to help choose the best option.

    Args:
        task_description: Description of the task requiring tools
        candidate_tools: Optional list of specific tools to compare
        max_recommendations: Maximum number of tools to compare
        include_detailed_analysis: Whether to include detailed comparison analysis

    Returns:
        Comparison of tool recommendations with decision guidance
    """
    try:
        # Check if comparison engine is initialized
        if not _comparison_engine:
            return create_error_response("get_tool_recommendations_comparison", "Comparison engine not initialized")

        # Use comparison engine's task-specific comparison
        comparison_result = _comparison_engine.compare_for_task(
            task_description=task_description, candidate_tools=candidate_tools, max_comparisons=max_recommendations
        )

        return comparison_result.to_dict()

    except Exception as e:
        logger.error(f"Error in get_tool_recommendations_comparison: {e}")
        return create_error_response("get_tool_recommendations_comparison", str(e), task_description)


# Utility functions
def apply_context_filter(alternatives: List[AlternativeAnalysis], context_filter: str) -> List[AlternativeAnalysis]:
    """Apply context-based filtering to alternatives"""
    if "simple" in context_filter.lower():
        return [alt for alt in alternatives if alt.complexity_comparison in ["simpler", "similar"]]
    elif "beginner" in context_filter.lower():
        return [alt for alt in alternatives if alt.learning_curve == "easy"]
    elif "advanced" in context_filter.lower():
        return [alt for alt in alternatives if "advanced" in " ".join(alt.unique_capabilities)]

    return alternatives


def generate_alternative_comparisons(reference_tool: str, alternatives: List[AlternativeAnalysis]) -> Dict[str, Any]:
    """Generate comparison data for alternatives against reference tool"""
    if not _comparison_engine:
        return {}

    try:
        # Compare top alternatives with reference tool
        alt_tool_ids = [alt.tool_id for alt in alternatives]
        comparison_tools = [reference_tool] + alt_tool_ids

        comparison_result = _comparison_engine.compare_tools(comparison_tools)

        return {
            "reference_tool": reference_tool,
            "comparison_matrix": comparison_result.comparison_matrix,
            "rankings": comparison_result.overall_ranking,
            "key_differentiators": comparison_result.key_differentiators,
        }
    except Exception as e:
        logger.warning(f"Failed to generate alternative comparisons: {e}")
        return {}


def generate_relationship_explanations(tool_id: str, relationships: Dict[str, List[str]]) -> List[str]:
    """Generate explanations for why relationships exist"""
    explanations = []

    for rel_type, related_tools in relationships.items():
        if not related_tools:
            continue

        if rel_type == "alternatives":
            explanations.append(f"{tool_id} alternatives provide similar functionality with different approaches")
        elif rel_type == "complements":
            explanations.append(f"Tools that work well with {tool_id} to create complete workflows")
        elif rel_type == "prerequisites":
            explanations.append(f"Tools often needed before using {tool_id} effectively")
        elif rel_type == "dependents":
            explanations.append(f"Tools that often use {tool_id} as part of their functionality")

    return explanations
