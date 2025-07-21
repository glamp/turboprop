#!/usr/bin/env python3
"""
comparison_response_types.py: Response types for MCP comparison and category tools.

This module defines structured response types used by the MCP comparison
and category tools for consistent and comprehensive data exchange.
"""

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .decision_support import SelectionGuidance
from .tool_comparison_engine import ToolComparisonResult


@dataclass
class ToolComparisonMCPResponse:
    """Response for compare_mcp_tools MCP tool"""

    tool_ids: List[str]
    comparison_result: ToolComparisonResult
    comparison_criteria: Optional[List[str]]
    detail_level: str

    # Decision support
    decision_guidance: Optional[SelectionGuidance] = None
    trade_off_analysis: List[str] = field(default_factory=list)

    # Metadata
    timestamp: Optional[str] = None
    version: str = "1.0"

    def add_decision_guidance(self, guidance: SelectionGuidance) -> None:
        """Add decision guidance to the response"""
        self.decision_guidance = guidance

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "tool_ids": self.tool_ids,
            "comparison_result": self.comparison_result.to_dict(),
            "comparison_criteria": self.comparison_criteria,
            "detail_level": self.detail_level,
            "trade_off_analysis": self.trade_off_analysis,
            "timestamp": self.timestamp or time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "version": self.version,
            "success": True,
        }

        if self.decision_guidance:
            result["decision_guidance"] = asdict(self.decision_guidance)

        return result


@dataclass
class AlternativeAnalysis:
    """Analysis of a tool alternative"""

    tool_id: str
    similarity_score: float
    complexity_comparison: str  # 'simpler', 'similar', 'more_complex'
    learning_curve: str  # 'easy', 'moderate', 'steep'
    unique_capabilities: List[str]
    performance_comparison: str  # 'faster', 'similar', 'slower'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class AlternativesFoundResponse:
    """Response for find_tool_alternatives MCP tool"""

    reference_tool: str
    alternatives: List[AlternativeAnalysis]
    similarity_threshold: float
    context_filter: Optional[str]

    # Comparison data
    comparisons: Optional[Dict[str, Any]] = None

    # Metadata
    timestamp: Optional[str] = None
    version: str = "1.0"

    def add_comparisons(self, comparisons: Dict[str, Any]) -> None:
        """Add comparison data for alternatives"""
        self.comparisons = comparisons

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "reference_tool": self.reference_tool,
            "alternatives": [alt.to_dict() for alt in self.alternatives],
            "similarity_threshold": self.similarity_threshold,
            "context_filter": self.context_filter,
            "total_alternatives": len(self.alternatives),
            "timestamp": self.timestamp or time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "version": self.version,
            "success": True,
        }

        if self.comparisons:
            result["comparisons"] = self.comparisons

        return result


@dataclass
class ToolRelationshipsResponse:
    """Response for analyze_tool_relationships MCP tool"""

    tool_id: str
    relationships: Dict[str, List[str]]  # relationship_type -> tool_ids
    relationship_types: List[str]

    # Explanations
    explanations: List[str] = field(default_factory=list)

    # Metadata
    timestamp: Optional[str] = None
    version: str = "1.0"

    def add_explanations(self, explanations: List[str]) -> None:
        """Add explanations for relationships"""
        self.explanations = explanations

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "tool_id": self.tool_id,
            "relationships": self.relationships,
            "relationship_types": self.relationship_types,
            "explanations": self.explanations,
            "total_relationships": sum(len(tools) for tools in self.relationships.values()),
            "timestamp": self.timestamp or time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "version": self.version,
            "success": True,
        }


@dataclass
class ToolSearchResult:
    """Basic tool search result"""

    tool_id: str
    name: str
    description: str
    category: str
    complexity: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class CategoryBrowseResponse:
    """Response for browse_tools_by_category MCP tool"""

    category: str
    tools: List[ToolSearchResult]
    sort_by: str
    complexity_filter: Optional[str]

    # Category information
    category_overview: Optional[Dict[str, Any]] = None
    category_statistics: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    timestamp: Optional[str] = None
    version: str = "1.0"

    def add_category_overview(self, overview: Dict[str, Any]) -> None:
        """Add category overview information"""
        self.category_overview = overview

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "category": self.category,
            "tools": [tool.to_dict() for tool in self.tools],
            "sort_by": self.sort_by,
            "complexity_filter": self.complexity_filter,
            "category_overview": self.category_overview,
            "category_statistics": self.category_statistics,
            "total_tools": len(self.tools),
            "timestamp": self.timestamp or time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "version": self.version,
            "success": True,
        }


@dataclass
class CategoryOverviewResponse:
    """Response for get_category_overview MCP tool"""

    categories: List[Dict[str, Any]]

    # Ecosystem statistics
    ecosystem_stats: Optional[Dict[str, Any]] = None

    # Metadata
    timestamp: Optional[str] = None
    version: str = "1.0"

    def add_ecosystem_stats(self, stats: Dict[str, Any]) -> None:
        """Add ecosystem statistics"""
        self.ecosystem_stats = stats

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "categories": self.categories,
            "ecosystem_stats": self.ecosystem_stats,
            "total_categories": len(self.categories),
            "timestamp": self.timestamp or time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "version": self.version,
            "success": True,
        }


@dataclass
class TaskAnalysis:
    """Analysis of a task for tool selection"""

    task_description: str
    complexity: str  # 'simple', 'moderate', 'complex'
    required_capabilities: List[str]
    input_requirements: List[str]
    output_requirements: List[str]
    constraints: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class SelectionContext:
    """Context for tool selection"""

    constraints: Optional[List[str]]
    optimization_goal: str
    available_tools: Optional[List[str]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class SelectionGuidanceResponse:
    """Response for get_tool_selection_guidance MCP tool"""

    task_description: str
    guidance: SelectionGuidance
    task_analysis: TaskAnalysis
    selection_context: SelectionContext

    # Metadata
    timestamp: Optional[str] = None
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "task_description": self.task_description,
            "guidance": asdict(self.guidance),
            "task_analysis": self.task_analysis.to_dict(),
            "selection_context": self.selection_context.to_dict(),
            "timestamp": self.timestamp or time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "version": self.version,
            "success": True,
        }


# Error response utility
def create_error_response(tool_name: str, error_message: str, context: str = None) -> Dict[str, Any]:
    """Create standardized error response for MCP tools"""
    return {
        "tool_name": tool_name,
        "error": error_message,
        "context": context,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "success": False,
    }
