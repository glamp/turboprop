#!/usr/bin/env python3
"""
Parameter Ranking

This module provides sophisticated ranking algorithms based on parameter analysis,
enabling improved tool search results through parameter match quality assessment.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .logging_config import get_logger
from .mcp_metadata_types import ParameterAnalysis
from .parameter_analyzer import ParameterAnalyzer, ParameterRequirements
from .type_compatibility_analyzer import TypeCompatibilityAnalyzer

logger = get_logger(__name__)


class ComplexityPreference(Enum):
    """Tool complexity preferences for ranking."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ANY = "any"


@dataclass
class ParameterMatchScores:
    """Detailed parameter matching scores."""

    type_compatibility_score: float = 0.0
    name_match_score: float = 0.0
    required_parameter_score: float = 0.0
    optional_parameter_score: float = 0.0
    constraint_match_score: float = 0.0
    complexity_alignment_score: float = 0.0
    overall_parameter_score: float = 0.0
    penalty_deductions: float = 0.0
    boost_additions: float = 0.0
    confidence_level: float = 0.0


@dataclass
class RankingExplanation:
    """Explanation of ranking decisions."""

    score_breakdown: Dict[str, float] = field(default_factory=dict)
    boosts_applied: List[str] = field(default_factory=list)
    penalties_applied: List[str] = field(default_factory=list)
    ranking_factors: List[str] = field(default_factory=list)
    confidence_factors: List[str] = field(default_factory=list)
    suggested_improvements: List[str] = field(default_factory=list)


@dataclass
class RankingContext:
    """Context for parameter-based ranking."""

    parameter_requirements: Optional[ParameterRequirements] = None
    complexity_preference: ComplexityPreference = ComplexityPreference.ANY
    boost_weight: float = 0.3
    penalty_weight: float = 0.2
    name_matching_importance: float = 0.4
    type_matching_importance: float = 0.6
    prefer_fewer_parameters: bool = False
    prefer_more_parameters: bool = False
    penalize_missing_required: bool = True
    reward_optional_matches: bool = True


class ParameterRanking:
    """Ranking algorithms based on parameter analysis."""

    def __init__(
        self,
        parameter_analyzer: Optional[ParameterAnalyzer] = None,
        type_analyzer: Optional[TypeCompatibilityAnalyzer] = None,
    ):
        """
        Initialize parameter ranking system.

        Args:
            parameter_analyzer: Parameter analyzer for detailed analysis
            type_analyzer: Type compatibility analyzer for type matching
        """
        self.parameter_analyzer = parameter_analyzer or ParameterAnalyzer()
        self.type_analyzer = type_analyzer or TypeCompatibilityAnalyzer()

        # Scoring weights
        self.score_weights = {
            "type_compatibility": 0.25,
            "name_matching": 0.20,
            "required_parameters": 0.25,
            "optional_parameters": 0.15,
            "complexity_alignment": 0.10,
            "constraint_matching": 0.05,
        }

        # Complexity thresholds
        self.complexity_thresholds = {
            ComplexityPreference.SIMPLE: (0.0, 0.3),
            ComplexityPreference.MODERATE: (0.3, 0.7),
            ComplexityPreference.COMPLEX: (0.7, 1.0),
            ComplexityPreference.ANY: (0.0, 1.0),
        }

        logger.info("Initialized ParameterRanking with score weights: %s", self.score_weights)

    def calculate_parameter_match_score(
        self, tool: Any, requirements: ParameterRequirements, context: Optional[RankingContext] = None
    ) -> ParameterMatchScores:
        """
        Calculate parameter-specific match score for a tool.

        Args:
            tool: Tool to evaluate
            requirements: Parameter requirements to match against
            context: Ranking context with preferences

        Returns:
            ParameterMatchScores with detailed scoring breakdown
        """
        if not context:
            context = RankingContext(parameter_requirements=requirements)

        try:
            # Get tool parameters
            tool_parameters = self._extract_tool_parameters(tool)

            # Calculate individual score components
            type_score = self._calculate_type_compatibility_score(tool_parameters, requirements)
            name_score = self._calculate_name_match_score(tool_parameters, requirements, context)
            required_score = self._calculate_required_parameter_score(tool_parameters, requirements)
            optional_score = self._calculate_optional_parameter_score(tool_parameters, requirements)
            constraint_score = self._calculate_constraint_match_score(tool_parameters, requirements)
            complexity_score = self._calculate_complexity_alignment_score(tool, requirements, context)

            # Calculate penalties and boosts
            penalties = self._calculate_penalties(tool_parameters, requirements, context)
            boosts = self._calculate_boosts(tool_parameters, requirements, context)

            # Combine scores with weights
            weighted_score = (
                type_score * self.score_weights["type_compatibility"]
                + name_score * self.score_weights["name_matching"]
                + required_score * self.score_weights["required_parameters"]
                + optional_score * self.score_weights["optional_parameters"]
                + complexity_score * self.score_weights["complexity_alignment"]
                + constraint_score * self.score_weights["constraint_matching"]
            )

            # Apply penalties and boosts
            final_score = weighted_score + boosts - penalties
            final_score = max(0.0, min(1.0, final_score))

            # Calculate confidence level
            confidence = self._calculate_scoring_confidence(tool_parameters, requirements, final_score)

            return ParameterMatchScores(
                type_compatibility_score=type_score,
                name_match_score=name_score,
                required_parameter_score=required_score,
                optional_parameter_score=optional_score,
                constraint_match_score=constraint_score,
                complexity_alignment_score=complexity_score,
                overall_parameter_score=final_score,
                penalty_deductions=penalties,
                boost_additions=boosts,
                confidence_level=confidence,
            )

        except Exception as e:
            logger.error("Error calculating parameter match score: %s", e)
            return ParameterMatchScores(overall_parameter_score=0.5)  # Neutral score on error

    def apply_parameter_ranking_boost(
        self,
        results: List[Any],
        parameter_context: ParameterRequirements,
        boost_weight: float = 0.3,
        ranking_context: Optional[RankingContext] = None,
    ) -> List[Any]:
        """
        Apply parameter-based ranking boosts to search results.

        Args:
            results: List of tool search results
            parameter_context: Parameter requirements context
            boost_weight: Weight for parameter boost (0.0-1.0)
            ranking_context: Additional ranking context

        Returns:
            Re-ranked list of tools with parameter boosts applied
        """
        if not results:
            return results

        if not ranking_context:
            ranking_context = RankingContext(parameter_requirements=parameter_context, boost_weight=boost_weight)

        try:
            # Calculate parameter scores for all tools
            enhanced_results = []

            for tool in results:
                param_scores = self.calculate_parameter_match_score(tool, parameter_context, ranking_context)

                # Get original relevance score
                original_score = getattr(tool, "relevance_score", 0.5)

                # Apply weighted boost
                boosted_score = (
                    original_score * (1.0 - boost_weight) + param_scores.overall_parameter_score * boost_weight
                )

                # Update tool with boosted score
                if hasattr(tool, "relevance_score"):
                    tool.relevance_score = boosted_score

                # Add parameter ranking metadata
                if hasattr(tool, "match_reasons"):
                    ranking_explanation = self._generate_ranking_explanation(param_scores, ranking_context)
                    tool.match_reasons.extend(ranking_explanation.ranking_factors)

                enhanced_results.append(tool)

            # Sort by boosted relevance score
            enhanced_results.sort(key=lambda t: getattr(t, "relevance_score", 0.0), reverse=True)

            logger.debug("Applied parameter ranking boost to %d tools with weight %.2f", len(results), boost_weight)

            return enhanced_results

        except Exception as e:
            logger.error("Error applying parameter ranking boost: %s", e)
            return results  # Return original on error

    def rank_by_complexity_preference(
        self, results: List[Any], complexity_preference: ComplexityPreference, preserve_top_results: int = 3
    ) -> List[Any]:
        """
        Rank tools by complexity preference while preserving top results.

        Args:
            results: List of tool results
            complexity_preference: Desired complexity level
            preserve_top_results: Number of top results to preserve regardless of complexity

        Returns:
            Re-ranked list based on complexity preference
        """
        if not results or complexity_preference == ComplexityPreference.ANY:
            return results

        try:
            min_complexity, max_complexity = self.complexity_thresholds[complexity_preference]

            # Separate top results to preserve
            preserved_results = results[:preserve_top_results]
            remaining_results = results[preserve_top_results:]

            # Score remaining results based on complexity preference
            complexity_scored = []

            for tool in remaining_results:
                complexity = self._get_tool_complexity(tool)

                # Calculate complexity alignment score
                if min_complexity <= complexity <= max_complexity:
                    # Perfect alignment
                    alignment_score = 1.0
                else:
                    # Distance-based penalty
                    if complexity < min_complexity:
                        distance = min_complexity - complexity
                    else:
                        distance = complexity - max_complexity

                    alignment_score = max(0.0, 1.0 - (distance * 2))

                # Combine with original relevance
                original_score = getattr(tool, "relevance_score", 0.5)
                combined_score = original_score * 0.7 + alignment_score * 0.3

                complexity_scored.append((tool, combined_score))

            # Sort by combined score
            complexity_scored.sort(key=lambda x: x[1], reverse=True)
            ranked_remaining = [tool for tool, score in complexity_scored]

            final_results = preserved_results + ranked_remaining

            logger.debug("Ranked %d tools by complexity preference: %s", len(results), complexity_preference.value)

            return final_results

        except Exception as e:
            logger.error("Error ranking by complexity preference: %s", e)
            return results

    def generate_parameter_ranking_explanation(
        self, tool: Any, requirements: ParameterRequirements, context: Optional[RankingContext] = None
    ) -> RankingExplanation:
        """
        Generate detailed explanation of parameter ranking decisions.

        Args:
            tool: Tool that was ranked
            requirements: Parameter requirements used for ranking
            context: Ranking context

        Returns:
            RankingExplanation with detailed breakdown
        """
        if not context:
            context = RankingContext(parameter_requirements=requirements)

        try:
            param_scores = self.calculate_parameter_match_score(tool, requirements, context)
            return self._generate_ranking_explanation(param_scores, context)

        except Exception as e:
            logger.error("Error generating ranking explanation: %s", e)
            return RankingExplanation(
                score_breakdown={"error": 0.0}, ranking_factors=[f"Error generating explanation: {str(e)}"]
            )

    # Helper methods

    def _extract_tool_parameters(self, tool: Any) -> List[ParameterAnalysis]:
        """Extract parameter analysis list from tool."""
        if hasattr(tool, "parameters") and tool.parameters:
            return tool.parameters
        return []

    def _calculate_type_compatibility_score(
        self, tool_parameters: List[ParameterAnalysis], requirements: ParameterRequirements
    ) -> float:
        """Calculate type compatibility score."""
        if not requirements.input_types and not requirements.output_types:
            return 0.8  # Neutral score when no type requirements

        if not tool_parameters:
            return 0.5  # Neutral for tools with no parameters

        compatibility_scores = []

        # Check input type compatibility
        for req_type in requirements.input_types:
            for param in tool_parameters:
                result = self.type_analyzer.analyze_type_compatibility(param.type, req_type)
                compatibility_scores.append(result.compatibility_score)

        # Calculate average compatibility
        if compatibility_scores:
            return sum(compatibility_scores) / len(compatibility_scores)

        return 0.5

    def _calculate_name_match_score(
        self, tool_parameters: List[ParameterAnalysis], requirements: ParameterRequirements, context: RankingContext
    ) -> float:
        """Calculate parameter name matching score."""
        all_required_names = requirements.required_parameters + requirements.optional_parameters
        if not all_required_names:
            return 0.8  # Neutral score when no name requirements

        if not tool_parameters:
            return 0.0  # No parameters to match

        tool_names = {p.name.lower() for p in tool_parameters}
        matches = 0
        fuzzy_matches = 0

        for req_name in all_required_names:
            req_name_lower = req_name.lower()

            # Exact match
            if req_name_lower in tool_names:
                matches += 1
            else:
                # Fuzzy match
                for tool_name in tool_names:
                    if self._is_fuzzy_name_match(req_name_lower, tool_name):
                        fuzzy_matches += 1
                        break

        # Calculate score with full credit for exact matches, partial for fuzzy
        exact_score = matches / len(all_required_names)
        fuzzy_score = fuzzy_matches / len(all_required_names) * 0.5

        return min(1.0, exact_score + fuzzy_score)

    def _calculate_required_parameter_score(
        self, tool_parameters: List[ParameterAnalysis], requirements: ParameterRequirements
    ) -> float:
        """Calculate required parameter matching score."""
        if not requirements.required_parameters:
            return 0.9  # High score when no required parameters needed

        if not tool_parameters:
            return 0.0  # Tool has no parameters but requirements exist

        tool_param_names = {p.name.lower() for p in tool_parameters}

        matches = sum(1 for req_param in requirements.required_parameters if req_param.lower() in tool_param_names)

        return matches / len(requirements.required_parameters)

    def _calculate_optional_parameter_score(
        self, tool_parameters: List[ParameterAnalysis], requirements: ParameterRequirements
    ) -> float:
        """Calculate optional parameter matching score."""
        if not requirements.optional_parameters:
            return 0.7  # Neutral score when no optional requirements

        if not tool_parameters:
            return 0.5  # Neutral for tools with no parameters

        tool_param_names = {p.name.lower() for p in tool_parameters}

        matches = sum(1 for opt_param in requirements.optional_parameters if opt_param.lower() in tool_param_names)

        return matches / len(requirements.optional_parameters)

    def _calculate_constraint_match_score(
        self, tool_parameters: List[ParameterAnalysis], requirements: ParameterRequirements
    ) -> float:
        """Calculate parameter constraint matching score."""
        if not requirements.parameter_constraints:
            return 0.8  # Neutral score when no constraints

        # Simplified constraint matching - could be more sophisticated
        constraint_matches = 0
        total_constraints = len(requirements.parameter_constraints)

        for constraint_key, constraint_value in requirements.parameter_constraints.items():
            # Look for matching parameter constraints in tool
            for param in tool_parameters:
                if hasattr(param, "constraints") and param.constraints:
                    if constraint_key in param.constraints:
                        # Simple value comparison - could be more sophisticated
                        if param.constraints[constraint_key] == constraint_value:
                            constraint_matches += 1

        if total_constraints > 0:
            return constraint_matches / total_constraints
        return 0.8

    def _calculate_complexity_alignment_score(
        self, tool: Any, requirements: ParameterRequirements, context: RankingContext
    ) -> float:
        """Calculate complexity alignment score."""
        tool_complexity = self._get_tool_complexity(tool)
        preference = context.complexity_preference

        if preference == ComplexityPreference.ANY:
            return 0.8  # Neutral score for any complexity

        min_complexity, max_complexity = self.complexity_thresholds[preference]

        if min_complexity <= tool_complexity <= max_complexity:
            return 1.0  # Perfect alignment

        # Calculate distance penalty
        if tool_complexity < min_complexity:
            distance = min_complexity - tool_complexity
        else:
            distance = tool_complexity - max_complexity

        return max(0.0, 1.0 - (distance * 1.5))

    def _calculate_penalties(
        self, tool_parameters: List[ParameterAnalysis], requirements: ParameterRequirements, context: RankingContext
    ) -> float:
        """Calculate penalties for parameter mismatches."""
        penalties = 0.0

        # Penalty for missing required parameters
        if context.penalize_missing_required and requirements.required_parameters:
            tool_names = {p.name.lower() for p in tool_parameters}
            missing_required = [req for req in requirements.required_parameters if req.lower() not in tool_names]
            penalties += len(missing_required) * 0.1

        # Penalty for too many parameters if simplicity is preferred
        if context.prefer_fewer_parameters:
            param_count = len(tool_parameters)
            if param_count > 5:  # Arbitrary threshold
                penalties += (param_count - 5) * 0.02

        return min(0.5, penalties)  # Cap penalties

    def _calculate_boosts(
        self, tool_parameters: List[ParameterAnalysis], requirements: ParameterRequirements, context: RankingContext
    ) -> float:
        """Calculate boosts for parameter advantages."""
        boosts = 0.0

        # Boost for optional parameter matches
        if context.reward_optional_matches and requirements.optional_parameters:
            tool_names = {p.name.lower() for p in tool_parameters}
            optional_matches = sum(
                1 for opt_param in requirements.optional_parameters if opt_param.lower() in tool_names
            )
            boosts += optional_matches * 0.05

        # Boost for having more parameters if complexity is preferred
        if context.prefer_more_parameters:
            param_count = len(tool_parameters)
            if param_count > 3:
                boosts += (param_count - 3) * 0.02

        return min(0.3, boosts)  # Cap boosts

    def _calculate_scoring_confidence(
        self, tool_parameters: List[ParameterAnalysis], requirements: ParameterRequirements, final_score: float
    ) -> float:
        """Calculate confidence level for the scoring."""
        confidence_factors = []

        # Parameter count confidence
        if tool_parameters:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.4)

        # Requirements specificity confidence
        total_requirements = (
            len(requirements.required_parameters)
            + len(requirements.optional_parameters)
            + len(requirements.input_types)
            + len(requirements.output_types)
        )

        if total_requirements > 0:
            confidence_factors.append(min(1.0, total_requirements * 0.2))
        else:
            confidence_factors.append(0.3)

        # Score consistency confidence
        if 0.3 <= final_score <= 0.8:
            confidence_factors.append(0.9)  # Moderate scores are most reliable
        else:
            confidence_factors.append(0.6)  # Extreme scores less reliable

        return sum(confidence_factors) / len(confidence_factors)

    def _get_tool_complexity(self, tool: Any) -> float:
        """Get complexity score for tool."""
        if hasattr(tool, "complexity_score"):
            return float(tool.complexity_score)

        # Fallback: estimate from parameter count
        param_count = len(self._extract_tool_parameters(tool))
        return min(1.0, param_count * 0.15)

    def _is_fuzzy_name_match(self, name1: str, name2: str, threshold: float = 0.7) -> bool:
        """Check if two parameter names are fuzzy matches."""
        # Simple fuzzy matching - could use more sophisticated algorithms
        if name1 == name2:
            return True

        # Check for common variations
        variations = [
            ("file_path", "filepath", "path", "file"),
            ("timeout", "time_out", "max_time"),
            ("max_size", "maxsize", "size_limit"),
            ("output_format", "format", "output_type"),
        ]

        for variation in variations:
            if name1 in variation and name2 in variation:
                return True

        # Simple character similarity
        if len(name1) > 0 and len(name2) > 0:
            common_chars = sum(1 for c1, c2 in zip(name1, name2) if c1 == c2)
            similarity = common_chars / max(len(name1), len(name2))
            return similarity >= threshold

        return False

    def _generate_ranking_explanation(
        self, scores: ParameterMatchScores, context: RankingContext
    ) -> RankingExplanation:
        """Generate detailed ranking explanation."""
        explanation = RankingExplanation()

        # Score breakdown
        explanation.score_breakdown = {
            "type_compatibility": scores.type_compatibility_score,
            "name_matching": scores.name_match_score,
            "required_parameters": scores.required_parameter_score,
            "optional_parameters": scores.optional_parameter_score,
            "complexity_alignment": scores.complexity_alignment_score,
            "constraint_matching": scores.constraint_match_score,
            "overall": scores.overall_parameter_score,
        }

        # Generate ranking factors
        if scores.type_compatibility_score > 0.8:
            explanation.ranking_factors.append("Excellent type compatibility")
        elif scores.type_compatibility_score > 0.6:
            explanation.ranking_factors.append("Good type compatibility")
        elif scores.type_compatibility_score < 0.4:
            explanation.ranking_factors.append("Limited type compatibility")

        if scores.required_parameter_score == 1.0:
            explanation.ranking_factors.append("All required parameters matched")
        elif scores.required_parameter_score > 0.7:
            explanation.ranking_factors.append("Most required parameters matched")
        elif scores.required_parameter_score < 0.3:
            explanation.ranking_factors.append("Few required parameters matched")

        # Confidence factors
        if scores.confidence_level > 0.8:
            explanation.confidence_factors.append("High confidence in parameter analysis")
        elif scores.confidence_level > 0.6:
            explanation.confidence_factors.append("Moderate confidence in parameter analysis")
        else:
            explanation.confidence_factors.append("Limited confidence in parameter analysis")

        return explanation
