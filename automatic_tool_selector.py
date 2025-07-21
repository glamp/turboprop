#!/usr/bin/env python3
"""
Automatic Tool Selector - Core Automatic Selection Engine

This module implements the main orchestrator for the intelligent automatic tool
selection system. It coordinates usage pattern analysis, proactive suggestions,
learning systems, and effectiveness tracking to provide seamless tool selection.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from automatic_selection_config import CONFIG
from error_handling import ErrorSeverity, create_error_handler
from logging_config import get_logger
from proactive_suggestion_engine import ProactiveSuggestion

logger = get_logger(__name__)


@dataclass
class ToolRanking:
    """Tool ranking with scoring details."""

    tool_id: str
    ranking_score: float
    context: Dict[str, Any]
    factors: Dict[str, float]

    # Ranking metadata
    reasoning: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class AutomaticSelectionResult:
    """Result of automatic tool selection analysis."""

    context: Dict[str, Any]
    suggested_tools: List[ProactiveSuggestion]
    learned_preferences: Dict[str, Any]
    confidence_scores: Dict[str, float]
    reasoning: List[str]

    # Selection metadata
    selection_timestamp: float = field(default_factory=time.time)
    selection_strategy: str = "automatic"
    context_confidence: float = 0.0

    def get_top_suggestion(self) -> Optional[ProactiveSuggestion]:
        """Get highest confidence suggestion."""
        if not self.suggested_tools:
            return None

        return max(self.suggested_tools, key=lambda s: self.confidence_scores.get(s.tool_id, 0))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "context": self.context,
            "suggested_tools": [s.to_dict() for s in self.suggested_tools],
            "learned_preferences": self.learned_preferences,
            "confidence_scores": self.confidence_scores,
            "reasoning": self.reasoning,
            "selection_timestamp": self.selection_timestamp,
            "selection_strategy": self.selection_strategy,
            "context_confidence": self.context_confidence,
        }


class SelectionContextManager:
    """Manages context analysis for tool selection."""

    def analyze_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze context for tool suitability."""
        analysis = {
            "task_type": context.get("task", "unknown"),
            "complexity_score": self._calculate_complexity_score(context),
            "urgency": context.get("urgency", "normal"),
            "user_experience": context.get("user_level", "intermediate"),
        }

        logger.debug("Context analysis: %s", analysis)
        return analysis

    def _calculate_complexity_score(self, context: Dict[str, Any]) -> float:
        """Calculate complexity score for the context."""
        complexity_factors = 0.0
        config = CONFIG["complexity_scoring"]

        # Task complexity indicators
        if "complex" in str(context.get("task", "")).lower():
            complexity_factors += config["complex_keyword_weight"]
        if "advanced" in str(context.get("user_input", "")).lower():
            complexity_factors += config["advanced_keyword_weight"]
        if len(str(context.get("user_input", ""))) > config["long_input_threshold"]:
            complexity_factors += config["long_input_weight"]

        return min(config["max_complexity"], complexity_factors + config["base_complexity"])


class AutomaticToolSelector:
    """Intelligent automatic tool selection system."""

    def __init__(self, usage_analyzer, suggestion_engine, learning_system, effectiveness_tracker):
        """Initialize with required components."""
        self.usage_analyzer = usage_analyzer
        self.suggestion_engine = suggestion_engine
        self.learning_system = learning_system
        self.effectiveness_tracker = effectiveness_tracker
        self.context_manager = SelectionContextManager()
        self.error_handler = create_error_handler("AutomaticToolSelector")

        logger.info("Initialized Automatic Tool Selector")

    def analyze_and_suggest(
        self,
        current_context: Dict[str, Any],
        active_task: Optional[str] = None,
        user_history: Optional[List[Dict]] = None,
    ) -> AutomaticSelectionResult:
        """Analyze context and provide automatic tool suggestions."""

        try:
            logger.debug("Starting analysis for context: %s", current_context)

            # Analyze current usage patterns
            usage_patterns = self.usage_analyzer.analyze_current_session(
                context=current_context, task=active_task, history=user_history
            )

            # Get proactive suggestions based on patterns
            suggestions = self.suggestion_engine.generate_proactive_suggestions(
                patterns=usage_patterns, context=current_context
            )

            # Apply learning-based improvements
            learned_preferences = self.learning_system.get_learned_preferences(
                context=current_context, user_patterns=usage_patterns
            )

            # Create comprehensive selection result
            result = AutomaticSelectionResult(
                context=current_context,
                suggested_tools=suggestions,
                learned_preferences=learned_preferences,
                confidence_scores=self._calculate_confidence_scores(suggestions),
                reasoning=self._generate_selection_reasoning(suggestions, usage_patterns),
            )

            # Track selection for learning
            self.effectiveness_tracker.track_selection_event(
                context=current_context, suggestions=suggestions, selection_result=result
            )

            logger.info(
                "Generated %d suggestions with avg confidence %.2f",
                len(suggestions),
                sum(result.confidence_scores.values()) / max(1, len(result.confidence_scores)),
            )

            return result

        except Exception as e:
            # Use standardized error handling with fallback
            return self.error_handler.handle_with_default(
                operation="analyze_and_suggest",
                exception=e,
                default_value=AutomaticSelectionResult(
                    context=current_context,
                    suggested_tools=[],
                    learned_preferences={},
                    confidence_scores={},
                    reasoning=[f"Analysis failed: {str(e)}"],
                ),
                severity=ErrorSeverity.HIGH,
                context={"user_context": current_context},
            )

    def pre_rank_tools_for_context(self, available_tools: List[str], context: Dict[str, Any]) -> List[ToolRanking]:
        """Pre-rank tools based on context for faster selection."""

        try:
            # Analyze context for tool suitability
            context_analysis = self.context_manager.analyze_context(context)

            # Apply learned preferences
            user_preferences = self.learning_system.get_context_preferences(context)

            # Calculate ranking for each tool
            rankings = []
            for tool_id in available_tools:
                ranking = self._calculate_tool_ranking(
                    tool_id=tool_id, context_analysis=context_analysis, user_preferences=user_preferences
                )
                rankings.append(ranking)

            # Sort by ranking score
            ranked_tools = sorted(rankings, key=lambda r: r.ranking_score, reverse=True)

            logger.debug(
                "Ranked %d tools, top tool: %s (score: %.2f)",
                len(ranked_tools),
                ranked_tools[0].tool_id if ranked_tools else "none",
                ranked_tools[0].ranking_score if ranked_tools else 0.0,
            )

            return ranked_tools

        except Exception as e:
            # Use standardized error handling with fallback
            return self.error_handler.handle_with_default(
                operation="pre_rank_tools_for_context",
                exception=e,
                default_value=[
                    ToolRanking(
                        tool_id=tool_id,
                        ranking_score=CONFIG["default_values"]["default_ranking_score"],
                        context=context,
                        factors={"default": CONFIG["default_values"]["default_score"]},
                    )
                    for tool_id in available_tools
                ],
                severity=ErrorSeverity.MEDIUM,
                context={"available_tools": available_tools, "context": context},
            )

    def _calculate_confidence_scores(self, suggestions: List[ProactiveSuggestion]) -> Dict[str, float]:
        """Calculate confidence scores for automatic suggestions."""
        scores = {}
        config = CONFIG["confidence_weights"]
        defaults = CONFIG["default_values"]

        try:
            for suggestion in suggestions:
                # Base confidence on pattern strength
                pattern_confidence = getattr(suggestion, "pattern_strength", defaults["default_confidence"])

                # Adjust based on historical effectiveness
                historical_effectiveness = self.effectiveness_tracker.get_tool_effectiveness(
                    suggestion.tool_id, getattr(suggestion, "context", {})
                )

                # Combine factors using configured weights
                confidence = (
                    pattern_confidence * config["pattern_weight"]
                    + historical_effectiveness * config["historical_weight"]
                )
                scores[suggestion.tool_id] = min(config["max_confidence"], confidence)

        except Exception as e:
            logger.error("Error calculating confidence scores: %s", e)
            # Provide default scores
            for suggestion in suggestions:
                scores[suggestion.tool_id] = defaults["default_confidence"]

        return scores

    def _generate_selection_reasoning(self, suggestions: List[ProactiveSuggestion], usage_patterns) -> List[str]:
        """Generate reasoning for tool selections."""
        reasoning = []

        try:
            if not suggestions:
                reasoning.append("No proactive suggestions available for current context")
                return reasoning

            # Add pattern-based reasoning
            if hasattr(usage_patterns, "inefficiency_patterns") and usage_patterns.inefficiency_patterns:
                reasoning.append(f"Detected {len(usage_patterns.inefficiency_patterns)} inefficiency patterns")

            # Add suggestion-specific reasoning
            for suggestion in suggestions[:3]:  # Top 3 suggestions
                if hasattr(suggestion, "reasoning") and suggestion.reasoning:
                    reasoning.append(f"{suggestion.tool_id}: {suggestion.reasoning}")
                else:
                    reasoning.append(f"{suggestion.tool_id}: Recommended based on usage patterns")

        except Exception as e:
            logger.error("Error generating selection reasoning: %s", e)
            reasoning.append("Reasoning generation failed")

        return reasoning

    def _calculate_tool_ranking(
        self, tool_id: str, context_analysis: Dict[str, Any], user_preferences: Dict[str, float]
    ) -> ToolRanking:
        """Calculate ranking for a specific tool."""
        weights = CONFIG["ranking_weights"]
        defaults = CONFIG["default_values"]
        config = CONFIG["confidence_weights"]

        factors = {}

        # User preference factor
        user_score = user_preferences.get(tool_id, defaults["default_score"])
        factors["user_preference"] = user_score

        # Context suitability factor
        context_score = self._calculate_context_suitability(tool_id, context_analysis)
        factors["context_suitability"] = context_score

        # Historical effectiveness factor
        effectiveness_score = self.effectiveness_tracker.get_tool_effectiveness(tool_id, context_analysis)
        factors["historical_effectiveness"] = effectiveness_score

        # Calculate weighted final score using configured weights
        final_score = (
            user_score * weights["user_preference_weight"]
            + context_score * weights["context_suitability_weight"]
            + effectiveness_score * weights["historical_effectiveness_weight"]
        )

        return ToolRanking(
            tool_id=tool_id,
            ranking_score=final_score,
            context=context_analysis,
            factors=factors,
            confidence=min(final_score + config["confidence_boost"], config["max_confidence"]),
        )

    def _calculate_context_suitability(self, tool_id: str, context_analysis: Dict[str, Any]) -> float:
        """Calculate how suitable a tool is for the given context."""
        defaults = CONFIG["default_values"]
        suitability_rules = CONFIG["tool_suitability_rules"]
        thresholds = CONFIG["complexity_thresholds"]
        limits = CONFIG["score_limits"]

        # Simple rule-based context matching
        task_type = context_analysis.get("task_type", "").lower()
        complexity = context_analysis.get("complexity_score", defaults["default_score"])

        # Default suitability
        base_score = defaults["default_score"]

        # Apply task-based rules
        for task_keyword, tool_scores in suitability_rules.items():
            if task_keyword in task_type:
                for tool_keyword, score in tool_scores.items():
                    if tool_keyword in tool_id.lower():
                        base_score = max(base_score, score)

        # Adjust for complexity using configured thresholds
        if complexity > thresholds["high_complexity_threshold"] and "advanced" in tool_id.lower():
            base_score += thresholds["complexity_bonus"]
        elif complexity < thresholds["low_complexity_threshold"] and "simple" in tool_id.lower():
            base_score += thresholds["complexity_bonus"]

        return min(limits["max_score"], base_score)
