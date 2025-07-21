#!/usr/bin/env python3
"""
task_analysis_response_types.py: Refactored response types for task analysis MCP tools.

This module defines focused, single-responsibility dataclasses for MCP tool recommendation
responses. The classes are split to reduce complexity and interdependencies.

Core Classes:
- ResponseMetadata: Common metadata for all responses
- AnalysisMetrics: Analysis scores and assessments
- RecommendationCore: Essential recommendation data
- TaskAnalysisCore: Core task analysis data

Response Classes:
- TaskRecommendationResponse: Composed response for recommend_tools_for_task
- TaskAnalysisResponse: Composed response for analyze_task_requirements
- AlternativesResponse: Composed response for suggest_tool_alternatives
- ToolSequenceResponse: Composed response for recommend_tool_sequence
"""

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from . import response_config


@dataclass
class ResponseMetadata:
    """Common metadata for all MCP tool responses."""

    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
    version: str = response_config.RESPONSE_VERSION
    processing_time: Optional[float] = None
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisMetrics:
    """Analysis scores and complexity assessments."""

    complexity_assessment: str = "moderate"
    confidence_score: float = 0.7
    estimated_steps: int = 3
    skill_level_required: str = "intermediate"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecommendationCore:
    """Core data for a single tool recommendation."""

    tool_id: str
    tool_name: str
    confidence_score: float
    relevance_score: float
    task_alignment: float
    complexity_fit: str = "moderate"
    skill_level_match: str = "intermediate"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecommendationEnhancements:
    """Enhancement data for tool recommendations."""

    recommendation_reasons: List[str] = field(default_factory=list)
    usage_guidance: List[str] = field(default_factory=list)
    parameter_suggestions: Dict[str, Any] = field(default_factory=dict)
    alternatives_available: bool = False
    alternative_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskAnalysisCore:
    """Core data for task analysis."""

    task_description: str
    detail_level: str = "standard"
    required_capabilities: List[str] = field(default_factory=list)
    potential_challenges: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RequirementsBreakdown:
    """Detailed requirements breakdown."""

    functional_requirements: List[str] = field(default_factory=list)
    non_functional_requirements: List[str] = field(default_factory=list)
    input_specifications: List[str] = field(default_factory=list)
    output_specifications: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnhancementSuggestions:
    """Enhancement and improvement suggestions."""

    suggestions: List[str] = field(default_factory=list)
    task_improvement_suggestions: List[str] = field(default_factory=list)
    alternative_approaches: List[str] = field(default_factory=list)
    suggested_refinements: List[str] = field(default_factory=list)

    def add_suggestion(self, suggestion: str) -> None:
        """Add a suggestion if not already present."""
        if suggestion not in self.suggestions:
            self.suggestions.append(suggestion)

    def add_refinement(self, refinement: str) -> None:
        """Add a refinement suggestion if not already present."""
        if refinement not in self.suggested_refinements:
            self.suggested_refinements.append(refinement)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AlternativeToolCore:
    """Core data for an alternative tool recommendation."""

    tool_id: str
    tool_name: str
    similarity_score: float
    complexity_comparison: str = "similar"
    feature_comparison: str = "similar"
    performance_comparison: str = "similar"
    migration_effort: str = "low"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AlternativeToolDetails:
    """Detailed information for alternative tools."""

    advantages: List[str] = field(default_factory=list)
    disadvantages: List[str] = field(default_factory=list)
    use_cases: List[str] = field(default_factory=list)
    when_to_prefer: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolSequenceStep:
    """A single step in a tool sequence."""

    step_number: int
    tool_id: str
    tool_name: str
    purpose: str
    complexity: str = "moderate"
    optional: bool = False
    estimated_time: Optional[str] = None

    # Data flow information
    suggested_parameters: Dict[str, Any] = field(default_factory=dict)
    input_from_previous: List[str] = field(default_factory=list)
    output_to_next: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SequenceMetrics:
    """Quality metrics for tool sequences."""

    reliability_score: float = 0.8
    efficiency_score: float = 0.8
    maintainability_score: float = 0.8
    estimated_duration: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SequenceGuidance:
    """Usage guidance for tool sequences."""

    prerequisites: List[str] = field(default_factory=list)
    common_pitfalls: List[str] = field(default_factory=list)
    optimization_tips: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Composed Response Classes


@dataclass
class ToolRecommendation:
    """Complete tool recommendation combining core data and enhancements."""

    core: RecommendationCore
    enhancements: RecommendationEnhancements = field(default_factory=RecommendationEnhancements)

    def to_dict(self) -> Dict[str, Any]:
        result = self.core.to_dict()
        result.update(self.enhancements.to_dict())
        return result


@dataclass
class AlternativeTool:
    """Complete alternative tool recommendation."""

    core: AlternativeToolCore
    details: AlternativeToolDetails = field(default_factory=AlternativeToolDetails)

    def to_dict(self) -> Dict[str, Any]:
        result = self.core.to_dict()
        result.update(self.details.to_dict())
        return result


@dataclass
class ToolSequence:
    """Complete tool sequence with steps, metrics and guidance."""

    sequence_id: str
    sequence_name: str
    steps: List[ToolSequenceStep] = field(default_factory=list)
    complexity_level: str = "moderate"
    parallel_execution_possible: bool = False
    total_steps: int = 0

    metrics: SequenceMetrics = field(default_factory=SequenceMetrics)
    guidance: SequenceGuidance = field(default_factory=SequenceGuidance)

    def __post_init__(self):
        if self.total_steps == 0:
            self.total_steps = len(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "sequence_name": self.sequence_name,
            "steps": [step.to_dict() for step in self.steps],
            "total_steps": self.total_steps,
            "complexity_level": self.complexity_level,
            "parallel_execution_possible": self.parallel_execution_possible,
            **self.metrics.to_dict(),
            **self.guidance.to_dict(),
        }


# Main Response Classes


@dataclass
class TaskRecommendationResponse:
    """Composed response for recommend_tools_for_task MCP tool."""

    task_description: str
    recommendations: List[ToolRecommendation] = field(default_factory=list)

    # Composed components
    metadata: ResponseMetadata = field(default_factory=ResponseMetadata)
    analysis_metrics: Optional[AnalysisMetrics] = None
    enhancements: EnhancementSuggestions = field(default_factory=EnhancementSuggestions)

    # Core response data
    context_factors: Optional[Dict[str, Any]] = None
    recommendation_strategy: str = "intelligent"
    total_recommendations: int = 0
    explanations: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.total_recommendations == 0:
            self.total_recommendations = len(self.recommendations)

    def add_explanations(self, explanations: List[str]) -> None:
        """Add recommendation explanations."""
        self.explanations.extend(explanations)

    def add_refinement_suggestion(self, suggestion: str) -> None:
        """Add a task description refinement suggestion."""
        self.enhancements.add_refinement(suggestion)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_description": self.task_description,
            "recommendations": [rec.to_dict() for rec in self.recommendations],
            "task_analysis": self.analysis_metrics.to_dict() if self.analysis_metrics else None,
            "context_factors": self.context_factors,
            "recommendation_strategy": self.recommendation_strategy,
            "total_recommendations": self.total_recommendations,
            "explanations": self.explanations,
            **self.metadata.to_dict(),
            **(self.analysis_metrics.to_dict() if self.analysis_metrics else {}),
            **self.enhancements.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=response_config.JSON_INDENT)


@dataclass
class TaskAnalysisResponse:
    """Composed response for analyze_task_requirements MCP tool."""

    core: TaskAnalysisCore

    # Composed components
    metadata: ResponseMetadata = field(default_factory=ResponseMetadata)
    metrics: AnalysisMetrics = field(default_factory=AnalysisMetrics)
    requirements: RequirementsBreakdown = field(default_factory=RequirementsBreakdown)
    enhancements: EnhancementSuggestions = field(default_factory=EnhancementSuggestions)

    # Additional analysis data
    analysis: Optional[Dict[str, Any]] = None

    def add_challenge(self, challenge: str) -> None:
        """Add a potential challenge to the core analysis."""
        if challenge not in self.core.potential_challenges:
            self.core.potential_challenges.append(challenge)

    def add_capability(self, capability: str) -> None:
        """Add a required capability to the core analysis."""
        if capability not in self.core.required_capabilities:
            self.core.required_capabilities.append(capability)

    def add_suggestions(self, suggestions: List[str]) -> None:
        """Add suggestions to the enhancements."""
        for suggestion in suggestions:
            self.enhancements.add_suggestion(suggestion)

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.core.to_dict(),
            "analysis": self.analysis,
            **self.metadata.to_dict(),
            **self.metrics.to_dict(),
            **self.requirements.to_dict(),
            **self.enhancements.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=response_config.JSON_INDENT)


@dataclass
class AlternativesResponse:
    """Composed response for suggest_tool_alternatives MCP tool."""

    primary_tool: str
    alternatives: List[AlternativeTool] = field(default_factory=list)
    task_context: Optional[str] = None

    # Composed components
    metadata: ResponseMetadata = field(default_factory=ResponseMetadata)
    enhancements: EnhancementSuggestions = field(default_factory=EnhancementSuggestions)

    # Analysis data
    primary_tool_advantages: List[str] = field(default_factory=list)
    alternative_categories: Dict[str, List[str]] = field(default_factory=dict)
    selection_criteria: List[str] = field(default_factory=list)
    total_alternatives: int = 0

    def __post_init__(self):
        if self.total_alternatives == 0:
            self.total_alternatives = len(self.alternatives)

    def add_selection_criterion(self, criterion: str) -> None:
        """Add a selection criterion for choosing alternatives."""
        if criterion not in self.selection_criteria:
            self.selection_criteria.append(criterion)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_tool": self.primary_tool,
            "alternatives": [alt.to_dict() for alt in self.alternatives],
            "task_context": self.task_context,
            "primary_tool_advantages": self.primary_tool_advantages,
            "alternative_categories": self.alternative_categories,
            "selection_criteria": self.selection_criteria,
            "total_alternatives": self.total_alternatives,
            **self.metadata.to_dict(),
            **self.enhancements.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=response_config.JSON_INDENT)


@dataclass
class ToolSequenceResponse:
    """Composed response for recommend_tool_sequence MCP tool."""

    workflow_description: str
    sequences: List[ToolSequence] = field(default_factory=list)
    optimization_goal: str = "balanced"

    # Composed components
    metadata: ResponseMetadata = field(default_factory=ResponseMetadata)
    metrics: AnalysisMetrics = field(default_factory=AnalysisMetrics)
    enhancements: EnhancementSuggestions = field(default_factory=EnhancementSuggestions)

    # Workflow analysis
    workflow_analysis: Optional[Dict[str, Any]] = None
    parallel_opportunities: List[str] = field(default_factory=list)
    bottleneck_analysis: List[str] = field(default_factory=list)
    recommended_sequence_id: Optional[str] = None
    total_sequences: int = 0

    def __post_init__(self):
        if self.total_sequences == 0:
            self.total_sequences = len(self.sequences)
        if self.recommended_sequence_id is None and self.sequences:
            self.recommended_sequence_id = self.sequences[0].sequence_id

    def add_parallel_opportunity(self, opportunity: str) -> None:
        """Add a parallel execution opportunity."""
        if opportunity not in self.parallel_opportunities:
            self.parallel_opportunities.append(opportunity)

    def add_bottleneck(self, bottleneck: str) -> None:
        """Add a bottleneck analysis point."""
        if bottleneck not in self.bottleneck_analysis:
            self.bottleneck_analysis.append(bottleneck)

    def add_customization_suggestion(self, suggestion: str) -> None:
        """Add a customization suggestion."""
        self.enhancements.add_suggestion(suggestion)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_description": self.workflow_description,
            "sequences": [seq.to_dict() for seq in self.sequences],
            "optimization_goal": self.optimization_goal,
            "workflow_analysis": self.workflow_analysis,
            "parallel_opportunities": self.parallel_opportunities,
            "bottleneck_analysis": self.bottleneck_analysis,
            "recommended_sequence_id": self.recommended_sequence_id,
            "total_sequences": self.total_sequences,
            **self.metadata.to_dict(),
            **self.metrics.to_dict(),
            **self.enhancements.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=response_config.JSON_INDENT)


# Utility functions


def create_error_response(tool_name: str, error_message: str, context: str = "") -> Dict[str, Any]:
    """Create a standardized error response for MCP tools."""
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
    """Validate that response size is reasonable for MCP transmission."""
    response_json = json.dumps(response_dict)
    size_mb = len(response_json.encode("utf-8")) / (1024 * 1024)
    return size_mb <= max_size_mb
