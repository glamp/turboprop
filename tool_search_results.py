#!/usr/bin/env python3
"""
Tool Search Results

This module contains data structures for MCP tool search results, providing
comprehensive result objects with metadata, ranking information, and explanations.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from mcp_metadata_types import ParameterAnalysis, ToolExample


@dataclass
class ToolSearchResult:
    """Comprehensive tool search result with detailed matching information."""

    # Core tool information
    tool_id: str
    name: str
    description: str
    category: str
    tool_type: str

    # Matching information
    similarity_score: float
    relevance_score: float  # Combined semantic + contextual relevance
    confidence_level: str  # 'high', 'medium', 'low'
    match_reasons: List[str] = field(default_factory=list)  # Explanations for why this tool matched

    # Tool metadata
    parameters: List[ParameterAnalysis] = field(default_factory=list)
    parameter_count: int = 0
    required_parameter_count: int = 0
    complexity_score: float = 0.0

    # Usage information
    examples: List[ToolExample] = field(default_factory=list)
    best_practices: List[str] = field(default_factory=list)
    common_pitfalls: List[str] = field(default_factory=list)

    # Relationships
    alternatives: List[str] = field(default_factory=list)  # IDs of alternative tools
    complements: List[str] = field(default_factory=list)  # IDs of complementary tools
    prerequisites: List[str] = field(default_factory=list)  # IDs of prerequisite tools

    def __post_init__(self):
        """Validate and compute derived fields after initialization."""
        # Validate confidence level
        if self.confidence_level not in ["high", "medium", "low"]:
            raise ValueError(f"Invalid confidence_level: {self.confidence_level}")

        # Validate scores
        if not 0.0 <= self.similarity_score <= 1.0:
            raise ValueError(f"similarity_score must be between 0.0 and 1.0, got {self.similarity_score}")
        if not 0.0 <= self.relevance_score <= 1.0:
            raise ValueError(f"relevance_score must be between 0.0 and 1.0, got {self.relevance_score}")
        if not 0.0 <= self.complexity_score <= 1.0:
            raise ValueError(f"complexity_score must be between 0.0 and 1.0, got {self.complexity_score}")

        # Compute parameter counts if not provided
        if self.parameter_count == 0 and self.parameters:
            self.parameter_count = len(self.parameters)

        if self.required_parameter_count == 0 and self.parameters:
            self.required_parameter_count = sum(1 for p in self.parameters if p.required)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert ParameterAnalysis and ToolExample objects to dictionaries
        result["parameters"] = [p.to_dict() for p in self.parameters]
        result["examples"] = [ex.to_dict() for ex in self.examples]
        return result

    def get_confidence_score(self) -> float:
        """Get numeric confidence score based on confidence level."""
        confidence_mapping = {"high": 0.8, "medium": 0.5, "low": 0.2}
        return confidence_mapping.get(self.confidence_level, 0.0)

    def is_simple_tool(self) -> bool:
        """Check if this is a simple tool based on complexity and parameter count."""
        return self.complexity_score < 0.3 and self.required_parameter_count <= 2

    def is_complex_tool(self) -> bool:
        """Check if this is a complex tool based on complexity and parameter count."""
        return self.complexity_score > 0.7 or self.required_parameter_count > 5

    def has_good_examples(self) -> bool:
        """Check if this tool has good usage examples."""
        return len(self.examples) > 0 and any(ex.effectiveness_score > 0.7 for ex in self.examples)


@dataclass
class SearchIntent:
    """Analyzed intent from a search query."""

    query_type: str  # 'specific', 'general', 'comparison', 'alternative'
    target_category: Optional[str] = None
    target_functionality: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class ProcessedQuery:
    """Processed search query with metadata."""

    original_query: str
    cleaned_query: str
    expanded_terms: List[str] = field(default_factory=list)
    detected_category: Optional[str] = None
    detected_tool_type: Optional[str] = None
    search_intent: Optional[SearchIntent] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        if self.search_intent:
            result["search_intent"] = asdict(self.search_intent)
        return result


@dataclass
class ToolResultCluster:
    """Cluster of related search results."""

    cluster_name: str
    cluster_description: str
    tools: List[ToolSearchResult] = field(default_factory=list)
    average_relevance: float = 0.0
    cluster_size: int = 0

    def __post_init__(self):
        """Compute cluster metrics after initialization."""
        if self.tools:
            self.cluster_size = len(self.tools)
            self.average_relevance = sum(tool.relevance_score for tool in self.tools) / len(self.tools)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "cluster_name": self.cluster_name,
            "cluster_description": self.cluster_description,
            "tools": [tool.to_dict() for tool in self.tools],
            "average_relevance": self.average_relevance,
            "cluster_size": self.cluster_size,
        }


@dataclass
class ToolSearchResponse:
    """Complete response for tool search operations."""

    query: str
    results: List[ToolSearchResult] = field(default_factory=list)
    total_results: int = 0
    execution_time: float = 0.0

    # Query analysis
    processed_query: Optional[ProcessedQuery] = None
    suggested_refinements: List[str] = field(default_factory=list)

    # Result organization
    result_clusters: List[ToolResultCluster] = field(default_factory=list)
    category_breakdown: Dict[str, int] = field(default_factory=dict)

    # Search metadata
    search_strategy: str = "semantic"  # 'semantic', 'hybrid', 'keyword'
    confidence_distribution: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        """Compute derived fields after initialization."""
        if not self.total_results and self.results:
            self.total_results = len(self.results)

        # Compute category breakdown
        if self.results and not self.category_breakdown:
            category_counts = {}
            for result in self.results:
                category = result.category or "unknown"
                category_counts[category] = category_counts.get(category, 0) + 1
            self.category_breakdown = category_counts

        # Compute confidence distribution
        if self.results and not self.confidence_distribution:
            confidence_counts = {}
            for result in self.results:
                level = result.confidence_level
                confidence_counts[level] = confidence_counts.get(level, 0) + 1
            self.confidence_distribution = confidence_counts

    def to_json(self) -> str:
        """Convert to JSON for MCP tool responses."""
        return json.dumps(self.to_dict(), indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "total_results": self.total_results,
            "execution_time": self.execution_time,
            "suggested_refinements": self.suggested_refinements,
            "result_clusters": [cluster.to_dict() for cluster in self.result_clusters],
            "category_breakdown": self.category_breakdown,
            "search_strategy": self.search_strategy,
            "confidence_distribution": self.confidence_distribution,
        }

        if self.processed_query:
            result["processed_query"] = self.processed_query.to_dict()

        return result

    def get_high_confidence_results(self) -> List[ToolSearchResult]:
        """Get only high confidence results."""
        return [result for result in self.results if result.confidence_level == "high"]

    def get_results_by_category(self, category: str) -> List[ToolSearchResult]:
        """Get results filtered by category."""
        return [result for result in self.results if result.category == category]

    def get_simple_tools(self) -> List[ToolSearchResult]:
        """Get results for simple tools only."""
        return [result for result in self.results if result.is_simple_tool()]

    def get_complex_tools(self) -> List[ToolSearchResult]:
        """Get results for complex tools only."""
        return [result for result in self.results if result.is_complex_tool()]

    def has_good_results(self) -> bool:
        """Check if the search returned good quality results."""
        if not self.results:
            return False

        # Consider results good if we have at least one high confidence result
        # or multiple medium confidence results
        high_confidence_count = len(self.get_high_confidence_results())
        medium_confidence_count = len([r for r in self.results if r.confidence_level == "medium"])

        return high_confidence_count > 0 or medium_confidence_count >= 2

    def get_best_result(self) -> Optional[ToolSearchResult]:
        """Get the single best result based on relevance and confidence."""
        if not self.results:
            return None

        # Sort by confidence level first, then by relevance score
        confidence_order = {"high": 3, "medium": 2, "low": 1}

        return max(
            self.results,
            key=lambda r: (confidence_order.get(r.confidence_level, 0), r.relevance_score),
        )
