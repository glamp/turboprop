#!/usr/bin/env python3
"""
recommendation_algorithms.py: Advanced Algorithms for Tool Recommendation

This module implements sophisticated algorithms for ranking and scoring tools
based on task requirements, user context, and tool capabilities.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union

from .logging_config import get_logger
from .search_result_types import CodeSearchResult
from .task_analyzer import TaskAnalysis

logger = get_logger(__name__)

# Type aliases for better type safety
ToolSearchResult = Union[CodeSearchResult, Any]  # Allow for mock results in testing

# Configuration constants for recommendation scoring
DEFAULT_CAPABILITY_WEIGHT = 0.4
DEFAULT_COMPLEXITY_WEIGHT = 0.3
DEFAULT_PARAMETER_WEIGHT = 0.2
DEFAULT_EFFECTIVENESS_WEIGHT = 0.1

# Complexity alignment scoring parameters
COMPLEXITY_PERFECT_MATCH_SCORE = 1.0
COMPLEXITY_MISMATCH_PENALTY = 0.4
COMPLEXITY_SKILL_BONUS = 0.1

# Performance optimization constants
MAX_CANDIDATES_TO_PROCESS = 50  # Limit candidate processing for performance

# Confidence scoring thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.8
MEDIUM_CONFIDENCE_THRESHOLD = 0.6
LOW_CONFIDENCE_THRESHOLD = 0.4


@dataclass
class ToolRecommendation:
    """Complete tool recommendation with metadata."""

    tool: ToolSearchResult
    recommendation_score: float
    confidence_level: str  # 'high', 'medium', 'low'

    # Fit analysis
    task_alignment: float
    capability_match: float
    complexity_alignment: float
    parameter_compatibility: float

    # Explanations
    recommendation_reasons: List[str]
    potential_issues: List[str]
    usage_guidance: List[str]

    # Alternatives and context
    alternative_tools: List[str]
    when_to_use: str
    when_not_to_use: str

    # Metadata
    recommendation_strategy: str
    context_factors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ToolSequenceRecommendation:
    """Recommendation for a sequence of tools in a workflow."""

    tool_sequence: List[str]
    workflow_score: float
    efficiency_score: float
    reliability_score: float
    data_flow_compatibility: float

    # Explanations
    sequence_rationale: List[str]
    optimization_applied: List[str]
    potential_bottlenecks: List[str]
    error_recovery_strategy: str

    # Metadata
    estimated_execution_time: float
    resource_requirements: Dict[str, Any]
    success_probability: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class AlternativeRecommendation:
    """Alternative tool recommendation with comparison."""

    tool: ToolSearchResult
    comparison_score: float
    advantages: List[str]
    disadvantages: List[str]
    trade_offs: List[str]
    use_case_differences: List[str]
    migration_effort: str  # 'low', 'medium', 'high'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class WorkflowRequirements:
    """Requirements for optimizing tool workflows."""

    steps: List[str]
    data_flow_requirements: Dict[str, str]
    error_handling_strategy: str
    performance_requirements: Dict[str, Any]

    # Optional constraints
    time_constraints: Optional[Dict[str, Any]] = None
    resource_constraints: Optional[Dict[str, Any]] = None
    quality_constraints: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class RecommendationAlgorithms:
    """Advanced algorithms for tool recommendation."""

    def __init__(self):
        """Initialize the recommendation algorithms."""
        self.scoring_weights = self._load_scoring_weights()
        self.effectiveness_data = self._load_effectiveness_data()

        # Initialize scoring components
        self.capability_weight = DEFAULT_CAPABILITY_WEIGHT
        self.complexity_weight = DEFAULT_COMPLEXITY_WEIGHT
        self.parameter_weight = DEFAULT_PARAMETER_WEIGHT
        self.effectiveness_weight = DEFAULT_EFFECTIVENESS_WEIGHT

        logger.info("Recommendation algorithms initialized")

    def calculate_task_tool_fit(
        self,
        task_analysis: TaskAnalysis,
        tool: ToolSearchResult,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calculate how well a tool fits a specific task."""
        logger.debug(f"Calculating task-tool fit for {tool.name}")

        # Extract tool metadata
        tool_metadata = getattr(tool, "metadata", {})
        tool_capabilities = tool_metadata.get("capabilities", [])
        tool_complexity = tool_metadata.get("complexity", "moderate")
        tool_category = tool_metadata.get("category", "general")

        # Calculate component scores
        capability_score = self._calculate_capability_match_score(
            task_analysis.required_capabilities, tool_capabilities
        )

        complexity_score = self._calculate_complexity_alignment_score(
            task_analysis.complexity_level, tool_complexity, task_analysis.skill_level_required
        )

        category_score = self._calculate_category_alignment_score(task_analysis.task_category, tool_category)

        parameter_score = self._calculate_parameter_compatibility_score(task_analysis, tool_metadata)

        # Apply context adjustments if provided
        context_bonus = 0.0
        if context:
            context_bonus = self._calculate_context_bonus(task_analysis, tool, context)

        # Combine scores using weighted average
        combined_score = (
            capability_score * self.capability_weight
            + complexity_score * self.complexity_weight
            + category_score * 0.2
            + parameter_score * self.parameter_weight  # Category alignment weight
        )

        # Apply context bonus
        final_score = min(combined_score + context_bonus, 1.0)

        logger.debug(f"Task-tool fit calculated: {final_score:.3f}")
        return final_score

    def apply_ensemble_ranking(
        self, candidates: List[Any], task_analysis: TaskAnalysis, context: Optional[Any] = None
    ) -> List[ToolRecommendation]:
        """Apply ensemble ranking combining multiple algorithms."""
        logger.info(f"Applying ensemble ranking to {len(candidates)} candidates")

        # Performance optimization: limit candidate processing for large sets
        if len(candidates) > MAX_CANDIDATES_TO_PROCESS:
            logger.info(
                f"Large candidate set ({len(candidates)}), limiting to {MAX_CANDIDATES_TO_PROCESS} for performance"
            )
            candidates = candidates[:MAX_CANDIDATES_TO_PROCESS]

        recommendations = []

        for tool in candidates:
            # Calculate core fit score
            fit_score = self.calculate_task_tool_fit(task_analysis, tool, context)

            # Calculate detailed component scores for explanation
            tool_metadata = getattr(tool, "metadata", {})
            capability_match = self._calculate_capability_match_score(
                task_analysis.required_capabilities, tool_metadata.get("capabilities", [])
            )

            complexity_alignment = self._calculate_complexity_alignment_score(
                task_analysis.complexity_level,
                tool_metadata.get("complexity", "moderate"),
                task_analysis.skill_level_required,
            )

            parameter_compatibility = self._calculate_parameter_compatibility_score(task_analysis, tool_metadata)

            # Generate recommendation reasons
            reasons = self._generate_recommendation_reasons(
                task_analysis, tool, fit_score, capability_match, complexity_alignment
            )

            # Create recommendation
            recommendation = ToolRecommendation(
                tool=tool,
                recommendation_score=fit_score,
                confidence_level=self._determine_confidence_level(fit_score),
                task_alignment=fit_score,
                capability_match=capability_match,
                complexity_alignment=complexity_alignment,
                parameter_compatibility=parameter_compatibility,
                recommendation_reasons=reasons,
                potential_issues=[],  # Will be populated by explainer
                usage_guidance=[],  # Will be populated by explainer
                alternative_tools=[],  # Will be populated by explainer
                when_to_use="",  # Will be populated by explainer
                when_not_to_use="",  # Will be populated by explainer
                recommendation_strategy="ensemble_scoring",
                context_factors=self._extract_context_factors(context),
            )

            recommendations.append(recommendation)

        # Sort by recommendation score (descending)
        recommendations.sort(key=lambda x: x.recommendation_score, reverse=True)

        top_tool = recommendations[0].tool.name if recommendations else "none"
        logger.info(f"Ensemble ranking complete, top recommendation: {top_tool}")
        return recommendations

    def calculate_recommendation_confidence(
        self, recommendation: ToolRecommendation, task_analysis: TaskAnalysis
    ) -> float:
        """Calculate confidence in a tool recommendation."""
        # Base confidence from alignment scores
        alignment_scores = [
            recommendation.task_alignment,
            recommendation.capability_match,
            recommendation.complexity_alignment,
            recommendation.parameter_compatibility,
        ]

        base_confidence = sum(alignment_scores) / len(alignment_scores)

        # Apply bonuses and penalties
        confidence_adjustments = 0.0

        # Bonus for high task analysis confidence
        if task_analysis.confidence > 0.8:
            confidence_adjustments += 0.1

        # Penalty for low capability match
        if recommendation.capability_match < 0.5:
            confidence_adjustments -= 0.2

        # Bonus for perfect complexity alignment
        if recommendation.complexity_alignment > 0.9:
            confidence_adjustments += 0.05

        final_confidence = max(0.0, min(1.0, base_confidence + confidence_adjustments))

        return final_confidence

    def optimize_tool_sequence(
        self, tool_chain: List[str], workflow_requirements: WorkflowRequirements
    ) -> List[ToolSequenceRecommendation]:
        """Optimize tool sequences for workflow efficiency."""
        logger.info(f"Optimizing tool sequence of {len(tool_chain)} tools")

        # For now, return a basic optimized sequence
        # In a full implementation, this would analyze data flow, compatibility, etc.
        optimized_sequence = ToolSequenceRecommendation(
            tool_sequence=tool_chain,
            workflow_score=0.8,
            efficiency_score=0.75,
            reliability_score=0.85,
            data_flow_compatibility=0.9,
            sequence_rationale=["Tools ordered by data flow requirements"],
            optimization_applied=["Removed redundant processing steps"],
            potential_bottlenecks=["Data serialization between tools"],
            error_recovery_strategy=workflow_requirements.error_handling_strategy,
            estimated_execution_time=len(tool_chain) * 2.0,  # Rough estimate
            resource_requirements={"memory": "moderate", "cpu": "low"},
            success_probability=0.85,
        )

        return [optimized_sequence]

    def _calculate_capability_match_score(
        self, required_capabilities: List[str], tool_capabilities: List[str]
    ) -> float:
        """Calculate capability match score between required and available."""
        if not required_capabilities:
            return 1.0  # No requirements means any tool matches

        if not tool_capabilities:
            return 0.0  # Tool has no capabilities

        # Convert to sets for easier comparison
        required_set = set(required_capabilities)
        tool_set = set(tool_capabilities)

        # Calculate intersection ratio
        intersection = required_set.intersection(tool_set)
        match_ratio = len(intersection) / len(required_set)

        return float(match_ratio)

    def _calculate_complexity_alignment_score(
        self, task_complexity: str, tool_complexity: str, user_skill: str
    ) -> float:
        """Calculate how well tool complexity aligns with task and user."""
        # Define complexity levels
        complexity_levels = {"simple": 1, "moderate": 2, "complex": 3}
        skill_levels = {"beginner": 1, "intermediate": 2, "advanced": 3}

        task_level = complexity_levels.get(task_complexity, 2)
        tool_level = complexity_levels.get(tool_complexity, 2)
        user_level = skill_levels.get(user_skill, 2)

        # Perfect match bonus
        if task_level == tool_level:
            base_score = COMPLEXITY_PERFECT_MATCH_SCORE
        else:
            # Penalty for mismatch
            mismatch_penalty = abs(task_level - tool_level) * COMPLEXITY_MISMATCH_PENALTY
            base_score = max(0.0, 1.0 - mismatch_penalty)

        # User skill bonus - advanced users can handle complex tools better
        if user_level >= tool_level:
            skill_bonus = (user_level - tool_level) * COMPLEXITY_SKILL_BONUS
            base_score = min(1.0, base_score + skill_bonus)

        return base_score

    def _calculate_category_alignment_score(self, task_category: str, tool_category: str) -> float:
        """Calculate category alignment score."""
        if task_category == tool_category:
            return 1.0

        # Define related categories that have partial alignment
        related_categories = {
            "file_operation": ["data_processing", "data_extraction"],
            "data_processing": ["file_operation", "machine_learning"],
            "web_scraping": ["data_extraction", "data_processing"],
            "machine_learning": ["data_processing", "analysis"],
        }

        if tool_category in related_categories.get(task_category, []):
            return 0.7  # Partial alignment

        return 0.3  # No alignment, but still possible to use

    def _calculate_parameter_compatibility_score(self, task_analysis: TaskAnalysis, tool_metadata: Dict) -> float:
        """Calculate parameter compatibility score."""
        # For now, return a baseline score
        # In full implementation, this would analyze required vs available parameters
        return 0.8

    def _calculate_context_bonus(
        self, task_analysis: TaskAnalysis, tool: ToolSearchResult, context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate context-based bonus/penalty."""
        bonus = 0.0

        # Check if context has user skill level
        if hasattr(context, "user_skill_level"):
            tool_metadata = getattr(tool, "metadata", {})
            tool_skill = tool_metadata.get("skill_level", "intermediate")

            # Bonus for skill alignment
            if context.user_skill_level == tool_skill:
                bonus += 0.1
            elif context.user_skill_level == "advanced" and tool_skill == "complex":
                bonus += 0.05

        return min(bonus, 0.2)  # Cap bonus at 0.2

    def _generate_recommendation_reasons(
        self,
        task_analysis: TaskAnalysis,
        tool: ToolSearchResult,
        fit_score: float,
        capability_match: float,
        complexity_alignment: float,
    ) -> List[str]:
        """Generate human-readable recommendation reasons."""
        reasons = []

        if capability_match >= 0.8:
            reasons.append("Excellent capability match for task requirements")
        elif capability_match >= 0.6:
            reasons.append("Good capability match for task requirements")

        if complexity_alignment >= 0.9:
            reasons.append("Perfect complexity alignment with task and user skill")
        elif complexity_alignment >= 0.7:
            reasons.append("Good complexity alignment")

        if fit_score >= 0.8:
            reasons.append("High overall suitability for this task")

        if not reasons:
            reasons.append("Reasonable fit based on available capabilities")

        return reasons

    def _determine_confidence_level(self, score: float) -> str:
        """Determine confidence level from score."""
        if score >= HIGH_CONFIDENCE_THRESHOLD:
            return "high"
        elif score >= MEDIUM_CONFIDENCE_THRESHOLD:
            return "medium"
        else:
            return "low"

    def _extract_context_factors(self, context: Any) -> List[str]:
        """Extract relevant context factors."""
        factors = []

        if context:
            if hasattr(context, "user_skill_level"):
                factors.append(f"user_skill:{context.user_skill_level}")
            if hasattr(context, "project_type"):
                factors.append(f"project:{context.project_type}")

        return factors

    def _load_scoring_weights(self) -> Dict[str, float]:
        """Load scoring weights from configuration."""
        # In a full implementation, this would load from a config file
        return {
            "capability": DEFAULT_CAPABILITY_WEIGHT,
            "complexity": DEFAULT_COMPLEXITY_WEIGHT,
            "parameter": DEFAULT_PARAMETER_WEIGHT,
            "effectiveness": DEFAULT_EFFECTIVENESS_WEIGHT,
        }

    def _load_effectiveness_data(self) -> Dict[str, Any]:
        """Load historical effectiveness data."""
        # In a full implementation, this would load from a database
        return {}
