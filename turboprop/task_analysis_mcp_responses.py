#!/usr/bin/env python3
"""
task_analysis_mcp_responses.py: Response types for task analysis MCP tools.

This module defines comprehensive dataclasses for MCP tool recommendation responses
that provide structured JSON data for task analysis, tool recommendations, and
alternative suggestions.

Classes:
- TaskRecommendationResponse: Main response for recommend_tools_for_task
- TaskAnalysisResponse: Response for analyze_task_requirements
- AlternativesResponse: Response for suggest_tool_alternatives
- ToolSequenceResponse: Response for recommend_tool_sequence
"""

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from . import response_config


@dataclass
class ToolRecommendationMeta:
    """
    Metadata for a single tool recommendation.

    Provides detailed information about why a tool was recommended
    and how it fits the task requirements.
    """

    tool_id: str
    tool_name: str
    confidence_score: float
    relevance_score: float
    task_alignment: float

    # Explanation details
    recommendation_reasons: List[str] = field(default_factory=list)
    usage_guidance: List[str] = field(default_factory=list)
    parameter_suggestions: Dict[str, Any] = field(default_factory=dict)

    # Context information
    complexity_fit: str = "unknown"  # simple, moderate, complex
    skill_level_match: str = "unknown"  # beginner, intermediate, advanced

    # Alternative information
    alternatives_available: bool = False
    alternative_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class TaskRecommendationResponse:
    """
    Response for recommend_tools_for_task MCP tool.

    Provides comprehensive tool recommendations with explanations,
    alternatives, and context-aware suggestions.
    """

    task_description: str
    recommendations: List[ToolRecommendationMeta] = field(default_factory=list)
    task_analysis: Optional[Dict[str, Any]] = None
    context_factors: Optional[Dict[str, Any]] = None

    # Additional metadata
    recommendation_strategy: str = "intelligent"
    confidence_distribution: Dict[str, int] = field(default_factory=dict)
    explanations: List[str] = field(default_factory=list)

    # Response metadata
    timestamp: Optional[str] = None
    version: str = response_config.RESPONSE_VERSION
    total_recommendations: int = 0
    processing_time: Optional[float] = None

    # Enhancement features
    suggested_refinements: List[str] = field(default_factory=list)
    complexity_assessment: Optional[str] = None
    skill_level_guidance: Optional[str] = None

    def __post_init__(self):
        """Initialize computed fields after construction."""
        if self.timestamp is None:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

        if self.total_recommendations == 0:
            self.total_recommendations = len(self.recommendations)

        # Auto-compute confidence distribution
        if not self.confidence_distribution and self.recommendations:
            self._compute_confidence_distribution()

    def _compute_confidence_distribution(self):
        """Compute confidence level distribution from recommendations."""
        self.confidence_distribution = {"high": 0, "medium": 0, "low": 0}
        for rec in self.recommendations:
            if rec.confidence_score >= 0.8:
                self.confidence_distribution["high"] += 1
            elif rec.confidence_score >= 0.6:
                self.confidence_distribution["medium"] += 1
            else:
                self.confidence_distribution["low"] += 1

    def add_explanations(self, explanations: List[str]) -> None:
        """Add recommendation explanations."""
        self.explanations.extend(explanations)

    def add_refinement_suggestion(self, suggestion: str) -> None:
        """Add a task description refinement suggestion."""
        if suggestion not in self.suggested_refinements:
            self.suggested_refinements.append(suggestion)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_description": self.task_description,
            "recommendations": [rec.to_dict() for rec in self.recommendations],
            "task_analysis": self.task_analysis,
            "context_factors": self.context_factors,
            "recommendation_strategy": self.recommendation_strategy,
            "confidence_distribution": self.confidence_distribution,
            "explanations": self.explanations,
            "total_recommendations": self.total_recommendations,
            "timestamp": self.timestamp,
            "version": self.version,
            "processing_time": self.processing_time,
            "suggested_refinements": self.suggested_refinements,
            "complexity_assessment": self.complexity_assessment,
            "skill_level_guidance": self.skill_level_guidance,
            "success": True,
        }

    def to_json(self) -> str:
        """Convert to JSON string for MCP tool responses."""
        return json.dumps(self.to_dict(), indent=response_config.JSON_INDENT)


@dataclass
class TaskAnalysisResponse:
    """
    Response for analyze_task_requirements MCP tool.

    Provides detailed analysis of task requirements, complexity,
    and suggestions for task improvement.
    """

    task_description: str
    analysis: Optional[Dict[str, Any]] = None
    detail_level: str = "standard"

    # Analysis insights
    complexity_assessment: Optional[str] = None
    required_capabilities: List[str] = field(default_factory=list)
    potential_challenges: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    # Requirements breakdown
    functional_requirements: List[str] = field(default_factory=list)
    non_functional_requirements: List[str] = field(default_factory=list)
    input_specifications: List[str] = field(default_factory=list)
    output_specifications: List[str] = field(default_factory=list)

    # Context analysis
    estimated_complexity: str = "moderate"
    estimated_steps: int = 3
    skill_level_required: str = "intermediate"
    confidence_score: float = 0.7

    # Enhancement suggestions
    task_improvement_suggestions: List[str] = field(default_factory=list)
    alternative_approaches: List[str] = field(default_factory=list)

    # Metadata
    timestamp: Optional[str] = None
    version: str = response_config.RESPONSE_VERSION
    processing_time: Optional[float] = None

    def __post_init__(self):
        """Initialize computed fields after construction."""
        if self.timestamp is None:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    def add_suggestions(self, suggestions: List[str]) -> None:
        """Add task improvement suggestions."""
        self.suggestions.extend(suggestions)

    def add_challenge(self, challenge: str) -> None:
        """Add a potential challenge."""
        if challenge not in self.potential_challenges:
            self.potential_challenges.append(challenge)

    def add_capability(self, capability: str) -> None:
        """Add a required capability."""
        if capability not in self.required_capabilities:
            self.required_capabilities.append(capability)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_description": self.task_description,
            "analysis": self.analysis,
            "detail_level": self.detail_level,
            "complexity_assessment": self.complexity_assessment,
            "required_capabilities": self.required_capabilities,
            "potential_challenges": self.potential_challenges,
            "suggestions": self.suggestions,
            "functional_requirements": self.functional_requirements,
            "non_functional_requirements": self.non_functional_requirements,
            "input_specifications": self.input_specifications,
            "output_specifications": self.output_specifications,
            "estimated_complexity": self.estimated_complexity,
            "estimated_steps": self.estimated_steps,
            "skill_level_required": self.skill_level_required,
            "confidence_score": self.confidence_score,
            "task_improvement_suggestions": self.task_improvement_suggestions,
            "alternative_approaches": self.alternative_approaches,
            "timestamp": self.timestamp,
            "version": self.version,
            "processing_time": self.processing_time,
            "success": True,
        }

    def to_json(self) -> str:
        """Convert to JSON string for MCP tool responses."""
        return json.dumps(self.to_dict(), indent=response_config.JSON_INDENT)


@dataclass
class AlternativeToolMeta:
    """
    Metadata for an alternative tool recommendation.

    Provides comparison information and guidance on when to use
    the alternative vs the primary tool.
    """

    tool_id: str
    tool_name: str
    similarity_score: float
    advantages: List[str] = field(default_factory=list)
    disadvantages: List[str] = field(default_factory=list)
    use_cases: List[str] = field(default_factory=list)

    # Comparison with primary tool
    complexity_comparison: str = "similar"  # simpler, similar, more_complex
    feature_comparison: str = "similar"  # fewer, similar, more
    performance_comparison: str = "similar"  # slower, similar, faster

    # Usage guidance
    when_to_prefer: List[str] = field(default_factory=list)
    migration_effort: str = "low"  # low, medium, high

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class AlternativesResponse:
    """
    Response for suggest_tool_alternatives MCP tool.

    Provides alternative tool suggestions with detailed comparisons
    and guidance on when to use each alternative.
    """

    primary_tool: str
    alternatives: List[AlternativeToolMeta] = field(default_factory=list)
    task_context: Optional[str] = None

    # Comparison analysis
    primary_tool_advantages: List[str] = field(default_factory=list)
    alternative_categories: Dict[str, List[str]] = field(default_factory=dict)
    trade_off_analysis: List[str] = field(default_factory=list)

    # Selection guidance
    selection_criteria: List[str] = field(default_factory=list)
    decision_framework: List[str] = field(default_factory=list)

    # Metadata
    timestamp: Optional[str] = None
    version: str = response_config.RESPONSE_VERSION
    total_alternatives: int = 0
    processing_time: Optional[float] = None

    def __post_init__(self):
        """Initialize computed fields after construction."""
        if self.timestamp is None:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

        if self.total_alternatives == 0:
            self.total_alternatives = len(self.alternatives)

    def add_comparisons(self, comparisons: List[str]) -> None:
        """Add comparison analysis points."""
        self.trade_off_analysis.extend(comparisons)

    def add_selection_criterion(self, criterion: str) -> None:
        """Add a selection criterion."""
        if criterion not in self.selection_criteria:
            self.selection_criteria.append(criterion)

    def categorize_alternative(self, category: str, tool_id: str) -> None:
        """Categorize an alternative tool."""
        if category not in self.alternative_categories:
            self.alternative_categories[category] = []
        if tool_id not in self.alternative_categories[category]:
            self.alternative_categories[category].append(tool_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "primary_tool": self.primary_tool,
            "alternatives": [alt.to_dict() for alt in self.alternatives],
            "task_context": self.task_context,
            "primary_tool_advantages": self.primary_tool_advantages,
            "alternative_categories": self.alternative_categories,
            "trade_off_analysis": self.trade_off_analysis,
            "selection_criteria": self.selection_criteria,
            "decision_framework": self.decision_framework,
            "timestamp": self.timestamp,
            "version": self.version,
            "total_alternatives": self.total_alternatives,
            "processing_time": self.processing_time,
            "success": True,
        }

    def to_json(self) -> str:
        """Convert to JSON string for MCP tool responses."""
        return json.dumps(self.to_dict(), indent=response_config.JSON_INDENT)


@dataclass
class ToolSequenceStep:
    """
    A single step in a tool sequence recommendation.

    Represents one tool usage in a multi-step workflow.
    """

    step_number: int
    tool_id: str
    tool_name: str
    purpose: str

    # Parameters and configuration
    suggested_parameters: Dict[str, Any] = field(default_factory=dict)
    input_from_previous: List[str] = field(default_factory=list)
    output_to_next: List[str] = field(default_factory=list)

    # Step metadata
    estimated_time: Optional[str] = None
    complexity: str = "moderate"
    optional: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ToolSequenceMeta:
    """
    Metadata for a complete tool sequence recommendation.

    Represents a complete workflow with multiple tools.
    """

    sequence_id: str
    sequence_name: str
    steps: List[ToolSequenceStep] = field(default_factory=list)

    # Sequence characteristics
    total_steps: int = 0
    estimated_duration: Optional[str] = None
    complexity_level: str = "moderate"
    parallel_execution_possible: bool = False

    # Quality metrics
    reliability_score: float = 0.8
    efficiency_score: float = 0.8
    maintainability_score: float = 0.8

    # Usage guidance
    prerequisites: List[str] = field(default_factory=list)
    common_pitfalls: List[str] = field(default_factory=list)
    optimization_tips: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize computed fields after construction."""
        if self.total_steps == 0:
            self.total_steps = len(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sequence_id": self.sequence_id,
            "sequence_name": self.sequence_name,
            "steps": [step.to_dict() for step in self.steps],
            "total_steps": self.total_steps,
            "estimated_duration": self.estimated_duration,
            "complexity_level": self.complexity_level,
            "parallel_execution_possible": self.parallel_execution_possible,
            "reliability_score": self.reliability_score,
            "efficiency_score": self.efficiency_score,
            "maintainability_score": self.maintainability_score,
            "prerequisites": self.prerequisites,
            "common_pitfalls": self.common_pitfalls,
            "optimization_tips": self.optimization_tips,
        }


@dataclass
class ToolSequenceResponse:
    """
    Response for recommend_tool_sequence MCP tool.

    Provides tool sequence recommendations for complex workflows
    with detailed step-by-step guidance.
    """

    workflow_description: str
    sequences: List[ToolSequenceMeta] = field(default_factory=list)
    workflow_analysis: Optional[Dict[str, Any]] = None
    optimization_goal: str = "balanced"

    # Workflow insights
    complexity_assessment: str = "moderate"
    parallel_opportunities: List[str] = field(default_factory=list)
    bottleneck_analysis: List[str] = field(default_factory=list)

    # Recommendations
    recommended_sequence_id: Optional[str] = None
    alternative_approaches: List[str] = field(default_factory=list)
    customization_suggestions: List[str] = field(default_factory=list)

    # Metadata
    timestamp: Optional[str] = None
    version: str = response_config.RESPONSE_VERSION
    total_sequences: int = 0
    processing_time: Optional[float] = None

    def __post_init__(self):
        """Initialize computed fields after construction."""
        if self.timestamp is None:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

        if self.total_sequences == 0:
            self.total_sequences = len(self.sequences)

        # Set recommended sequence to first one if not specified
        if self.recommended_sequence_id is None and self.sequences:
            self.recommended_sequence_id = self.sequences[0].sequence_id

    def add_parallel_opportunity(self, opportunity: str) -> None:
        """Add a parallel execution opportunity."""
        if opportunity not in self.parallel_opportunities:
            self.parallel_opportunities.append(opportunity)

    def add_bottleneck(self, bottleneck: str) -> None:
        """Add a workflow bottleneck analysis."""
        if bottleneck not in self.bottleneck_analysis:
            self.bottleneck_analysis.append(bottleneck)

    def add_customization_suggestion(self, suggestion: str) -> None:
        """Add a customization suggestion."""
        if suggestion not in self.customization_suggestions:
            self.customization_suggestions.append(suggestion)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "workflow_description": self.workflow_description,
            "sequences": [seq.to_dict() for seq in self.sequences],
            "workflow_analysis": self.workflow_analysis,
            "optimization_goal": self.optimization_goal,
            "complexity_assessment": self.complexity_assessment,
            "parallel_opportunities": self.parallel_opportunities,
            "bottleneck_analysis": self.bottleneck_analysis,
            "recommended_sequence_id": self.recommended_sequence_id,
            "alternative_approaches": self.alternative_approaches,
            "customization_suggestions": self.customization_suggestions,
            "timestamp": self.timestamp,
            "version": self.version,
            "total_sequences": self.total_sequences,
            "processing_time": self.processing_time,
            "success": True,
        }

    def to_json(self) -> str:
        """Convert to JSON string for MCP tool responses."""
        return json.dumps(self.to_dict(), indent=response_config.JSON_INDENT)


# Utility functions for creating responses


def create_error_response(tool_name: str, error_message: str, context: str = "") -> Dict[str, Any]:
    """
    Create a standardized error response for MCP tools.

    Args:
        tool_name: Name of the tool that encountered the error
        error_message: Description of the error
        context: Additional context about what was being processed

    Returns:
        Dictionary with error response structure
    """
    return {
        "success": False,
        "error": {
            "tool": tool_name,
            "message": error_message,
            "context": context,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "version": response_config.RESPONSE_VERSION,
        },
    }


def validate_response_size(response_dict: Dict[str, Any], max_size_mb: float = 10.0) -> bool:
    """
    Validate that response size is reasonable for MCP transmission.

    Args:
        response_dict: Response dictionary to validate
        max_size_mb: Maximum size in megabytes

    Returns:
        True if size is acceptable, False otherwise
    """
    response_json = json.dumps(response_dict)
    size_mb = len(response_json.encode("utf-8")) / (1024 * 1024)
    return size_mb <= max_size_mb
