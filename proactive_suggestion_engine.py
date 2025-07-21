#!/usr/bin/env python3
"""
Proactive Suggestion Engine - Context-aware Tool Suggestions

This module generates proactive tool suggestions based on usage patterns,
context analysis, and learned preferences. It provides intelligent
recommendations to improve tool selection efficiency.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from logging_config import get_logger
from usage_pattern_analyzer import InefficiencyPattern, UsagePatternAnalysis

logger = get_logger(__name__)


@dataclass
class ProactiveSuggestion:
    """Proactive tool suggestion."""

    suggestion_type: str  # 'tool_replacement', 'workflow_improvement', 'parameter_optimization'
    tool_id: str
    current_tool: Optional[str] = None
    reasoning: str = ""
    confidence: float = 0.0
    expected_improvement: str = ""
    pattern_strength: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)

    # Suggestion metadata
    suggestion_timestamp: float = field(default_factory=time.time)
    priority: str = "medium"  # 'low', 'medium', 'high'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class ContextAnalyzer:
    """Analyzes context for better suggestions."""

    def __init__(self):
        self.context_patterns = self._load_context_patterns()
        logger.debug("Initialized Context Analyzer")

    def _load_context_patterns(self) -> Dict[str, Any]:
        """Load context analysis patterns."""
        # Default context patterns - could be loaded from config
        return {
            "task_types": {
                "search": {
                    "keywords": ["find", "search", "locate", "look"],
                    "optimal_tools": ["search_files", "grep", "semantic_search"],
                    "complexity_indicators": ["regex", "pattern", "filter"],
                },
                "file_ops": {
                    "keywords": ["read", "write", "edit", "create", "file"],
                    "optimal_tools": ["read_file", "write_file", "edit_file"],
                    "complexity_indicators": ["batch", "multiple", "recursive"],
                },
                "analysis": {
                    "keywords": ["analyze", "examine", "investigate", "understand"],
                    "optimal_tools": ["analyze_code", "extract_metadata", "profile"],
                    "complexity_indicators": ["deep", "comprehensive", "detailed"],
                },
            },
            "context_hints": {
                "urgency": ["quick", "fast", "urgent", "immediate"],
                "thoroughness": ["complete", "comprehensive", "detailed", "full"],
                "simplicity": ["simple", "basic", "easy", "straightforward"],
            },
        }

    def analyze_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze context to extract insights for suggestions."""
        analysis = {
            "task_type": self._identify_task_type(context),
            "urgency_level": self._assess_urgency(context),
            "complexity_level": self._assess_complexity(context),
            "user_expertise": self._estimate_user_expertise(context),
            "optimal_tools": [],
            "context_confidence": 0.5,
        }

        # Get optimal tools for identified task type
        task_type = analysis["task_type"]
        if task_type in self.context_patterns["task_types"]:
            analysis["optimal_tools"] = self.context_patterns["task_types"][task_type]["optimal_tools"]

        # Calculate overall context confidence
        analysis["context_confidence"] = self._calculate_context_confidence(context, analysis)

        logger.debug(
            "Context analysis: task=%s, urgency=%s, complexity=%s",
            task_type,
            analysis["urgency_level"],
            analysis["complexity_level"],
        )

        return analysis

    def _identify_task_type(self, context: Dict[str, Any]) -> str:
        """Identify the primary task type from context."""
        user_input = str(context.get("user_input", "")).lower()
        task = str(context.get("task", "")).lower()
        combined_text = f"{user_input} {task}"

        task_scores = {}

        for task_type, patterns in self.context_patterns["task_types"].items():
            score = 0
            for keyword in patterns["keywords"]:
                if keyword in combined_text:
                    score += 1
            task_scores[task_type] = score

        # Return task type with highest score, or 'general' if no clear match
        if task_scores:
            best_task = max(task_scores, key=task_scores.get)
            if task_scores[best_task] > 0:
                return best_task

        return "general"

    def _assess_urgency(self, context: Dict[str, Any]) -> str:
        """Assess urgency level from context."""
        user_input = str(context.get("user_input", "")).lower()

        urgency_keywords = self.context_patterns["context_hints"]["urgency"]
        urgency_count = sum(1 for keyword in urgency_keywords if keyword in user_input)

        if urgency_count >= 2:
            return "high"
        elif urgency_count >= 1:
            return "medium"
        else:
            return "low"

    def _assess_complexity(self, context: Dict[str, Any]) -> str:
        """Assess complexity level from context."""
        user_input = str(context.get("user_input", "")).lower()
        task_type = self._identify_task_type(context)

        complexity_score = 0

        # Check for complexity indicators in task type
        if task_type in self.context_patterns["task_types"]:
            complexity_indicators = self.context_patterns["task_types"][task_type]["complexity_indicators"]
            complexity_score += sum(1 for indicator in complexity_indicators if indicator in user_input)

        # Check for general complexity hints
        thoroughness_keywords = self.context_patterns["context_hints"]["thoroughness"]
        complexity_score += sum(1 for keyword in thoroughness_keywords if keyword in user_input)

        # Check for simplicity indicators (reduces complexity)
        simplicity_keywords = self.context_patterns["context_hints"]["simplicity"]
        complexity_score -= sum(1 for keyword in simplicity_keywords if keyword in user_input)

        if complexity_score >= 2:
            return "high"
        elif complexity_score >= 1:
            return "medium"
        else:
            return "low"

    def _estimate_user_expertise(self, context: Dict[str, Any]) -> str:
        """Estimate user expertise level."""
        # Simple heuristic based on context complexity and language used
        user_input = str(context.get("user_input", "")).lower()

        expert_indicators = ["regex", "advanced", "optimize", "performance", "efficient"]
        beginner_indicators = ["help", "how to", "simple", "easy", "basic"]

        expert_score = sum(1 for indicator in expert_indicators if indicator in user_input)
        beginner_score = sum(1 for indicator in beginner_indicators if indicator in user_input)

        if expert_score > beginner_score and expert_score >= 2:
            return "expert"
        elif beginner_score > expert_score and beginner_score >= 2:
            return "beginner"
        else:
            return "intermediate"

    def _calculate_context_confidence(self, context: Dict[str, Any], analysis: Dict[str, Any]) -> float:
        """Calculate confidence in context analysis."""
        confidence_factors = []

        # Confidence based on input length
        user_input = str(context.get("user_input", ""))
        if len(user_input) > 50:
            confidence_factors.append(0.8)
        elif len(user_input) > 20:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.4)

        # Confidence based on task identification clarity
        if analysis["task_type"] != "general":
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.3)

        # Confidence based on available context fields
        context_richness = len([v for v in context.values() if v])
        if context_richness >= 4:
            confidence_factors.append(0.8)
        elif context_richness >= 2:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.4)

        return sum(confidence_factors) / len(confidence_factors)


class SuggestionRuleEngine:
    """Rule-based suggestion generation system."""

    def __init__(self):
        self.suggestion_rules = self._load_suggestion_rules()
        logger.debug("Initialized Suggestion Rule Engine with %d rules", len(self.suggestion_rules))

    def _load_suggestion_rules(self) -> Dict[str, Any]:
        """Load suggestion rules configuration."""
        return {
            "inefficiency_rules": {
                "suboptimal_choice": {
                    "template": "tool_replacement",
                    "confidence_multiplier": 0.8,
                    "priority": "high",
                    "reasoning_template": "More efficient tool for this task: {description}",
                },
                "excessive_trial_error": {
                    "template": "workflow_improvement",
                    "suggested_tool": "analyze_requirements",
                    "confidence": 0.9,
                    "priority": "high",
                    "reasoning": "Analyze requirements first to avoid trial-and-error",
                },
                "repetitive_usage": {
                    "template": "workflow_improvement",
                    "suggested_tool": "batch_operations",
                    "confidence": 0.7,
                    "priority": "medium",
                    "reasoning": "Consider batch operations for repetitive tasks",
                },
            },
            "context_rules": {
                "search_tasks": {
                    "high_urgency": {
                        "suggested_tool": "quick_search",
                        "confidence": 0.8,
                        "reasoning": "Fast search for urgent requests",
                    },
                    "high_complexity": {
                        "suggested_tool": "semantic_search",
                        "confidence": 0.9,
                        "reasoning": "Semantic search for complex queries",
                    },
                },
                "file_ops": {
                    "batch_indicators": {
                        "suggested_tool": "bulk_file_ops",
                        "confidence": 0.8,
                        "reasoning": "Bulk operations for multiple files",
                    }
                },
            },
            "expertise_rules": {
                "beginner": {"prefer_simple_tools": True, "confidence_penalty": -0.1, "add_explanations": True},
                "expert": {"prefer_advanced_tools": True, "confidence_bonus": 0.1, "suggest_optimizations": True},
            },
        }

    def generate_rule_suggestions(
        self, patterns: UsagePatternAnalysis, context_analysis: Dict[str, Any]
    ) -> List[ProactiveSuggestion]:
        """Generate suggestions based on rules."""
        suggestions = []

        try:
            # Generate inefficiency-based suggestions
            for inefficiency in patterns.inefficiency_patterns:
                suggestion = self._create_inefficiency_suggestion(inefficiency, context_analysis)
                if suggestion:
                    suggestions.append(suggestion)

            # Generate context-based suggestions
            context_suggestions = self._create_context_suggestions(patterns.context, context_analysis)
            suggestions.extend(context_suggestions)

            # Apply expertise-based adjustments
            self._apply_expertise_adjustments(suggestions, context_analysis)

        except Exception as e:
            logger.error("Error in generate_rule_suggestions: %s", e)

        return suggestions

    def _create_inefficiency_suggestion(
        self, inefficiency: InefficiencyPattern, context_analysis: Dict[str, Any]
    ) -> Optional[ProactiveSuggestion]:
        """Create suggestion to address inefficiency."""
        try:
            rules = self.suggestion_rules["inefficiency_rules"]

            if inefficiency.type in rules:
                rule = rules[inefficiency.type]

                if inefficiency.type == "suboptimal_choice":
                    return ProactiveSuggestion(
                        suggestion_type=rule["template"],
                        tool_id=inefficiency.suggestion,
                        current_tool=self._extract_current_tool_from_context(context_analysis),
                        reasoning=rule["reasoning_template"].format(description=inefficiency.description),
                        confidence=inefficiency.confidence * rule["confidence_multiplier"],
                        expected_improvement="Reduce task completion time by 30-50%",
                        pattern_strength=inefficiency.confidence,
                        priority=rule["priority"],
                    )
                else:
                    return ProactiveSuggestion(
                        suggestion_type=rule["template"],
                        tool_id=rule.get("suggested_tool", "analyze_requirements"),
                        reasoning=rule["reasoning"],
                        confidence=rule["confidence"],
                        expected_improvement="Reduce errors and improve first-attempt success rate",
                        pattern_strength=inefficiency.confidence,
                        priority=rule["priority"],
                    )

        except Exception as e:
            logger.warning("Error creating inefficiency suggestion: %s", e)

        return None

    def _create_context_suggestions(
        self, context: Dict[str, Any], context_analysis: Dict[str, Any]
    ) -> List[ProactiveSuggestion]:
        """Create suggestions based on context analysis."""
        suggestions = []

        try:
            task_type = context_analysis.get("task_type", "general")
            urgency = context_analysis.get("urgency_level", "medium")
            complexity = context_analysis.get("complexity_level", "medium")

            # Apply context-based rules
            context_rules = self.suggestion_rules["context_rules"]

            if task_type in context_rules:
                task_rules = context_rules[task_type]

                # Check urgency-based rules
                if urgency == "high" and "high_urgency" in task_rules:
                    rule = task_rules["high_urgency"]
                    suggestions.append(
                        ProactiveSuggestion(
                            suggestion_type="context_optimization",
                            tool_id=rule["suggested_tool"],
                            reasoning=rule["reasoning"],
                            confidence=rule["confidence"],
                            expected_improvement="Faster completion for urgent tasks",
                            pattern_strength=context_analysis.get("context_confidence", 0.5),
                            context=context,
                        )
                    )

                # Check complexity-based rules
                if complexity == "high" and "high_complexity" in task_rules:
                    rule = task_rules["high_complexity"]
                    suggestions.append(
                        ProactiveSuggestion(
                            suggestion_type="context_optimization",
                            tool_id=rule["suggested_tool"],
                            reasoning=rule["reasoning"],
                            confidence=rule["confidence"],
                            expected_improvement="Better handling of complex requirements",
                            pattern_strength=context_analysis.get("context_confidence", 0.5),
                            context=context,
                        )
                    )

        except Exception as e:
            logger.warning("Error creating context suggestions: %s", e)

        return suggestions

    def _apply_expertise_adjustments(self, suggestions: List[ProactiveSuggestion], context_analysis: Dict[str, Any]):
        """Apply expertise-based adjustments to suggestions."""
        try:
            expertise = context_analysis.get("user_expertise", "intermediate")
            expertise_rules = self.suggestion_rules["expertise_rules"]

            if expertise in expertise_rules:
                rules = expertise_rules[expertise]

                for suggestion in suggestions:
                    # Apply confidence adjustments
                    if "confidence_penalty" in rules:
                        suggestion.confidence = max(0.0, suggestion.confidence + rules["confidence_penalty"])
                    if "confidence_bonus" in rules:
                        suggestion.confidence = min(1.0, suggestion.confidence + rules["confidence_bonus"])

                    # Adjust reasoning for beginners
                    if rules.get("add_explanations", False):
                        suggestion.reasoning += " (Recommended for ease of use)"

                    # Suggest advanced alternatives for experts
                    if rules.get("suggest_optimizations", False):
                        suggestion.expected_improvement += " Consider advanced configuration options."

        except Exception as e:
            logger.warning("Error applying expertise adjustments: %s", e)

    def _extract_current_tool_from_context(self, context_analysis: Dict[str, Any]) -> Optional[str]:
        """Extract current tool from context if available."""
        # This would need to be implemented based on available context information
        return None


class ProactiveSuggestionEngine:
    """Generate proactive tool suggestions based on patterns."""

    def __init__(self):
        """Initialize suggestion engine with analyzers and rule engine."""
        self.context_analyzer = ContextAnalyzer()
        self.rule_engine = SuggestionRuleEngine()
        self.suggestion_history = []

        logger.info("Initialized Proactive Suggestion Engine")

    def generate_proactive_suggestions(
        self, patterns: UsagePatternAnalysis, context: Dict[str, Any], max_suggestions: int = 3
    ) -> List[ProactiveSuggestion]:
        """Generate proactive suggestions based on usage patterns."""

        try:
            logger.debug("Generating suggestions for context: %s", context)

            # Analyze context for better suggestions
            context_analysis = self.context_analyzer.analyze_context(context)

            # Generate rule-based suggestions
            suggestions = self.rule_engine.generate_rule_suggestions(patterns, context_analysis)

            # Generate workflow improvement suggestions
            workflow_suggestions = self._generate_workflow_suggestions(patterns, context_analysis)
            suggestions.extend(workflow_suggestions)

            # Rank and limit suggestions
            ranked_suggestions = self._rank_suggestions(suggestions, context_analysis)
            final_suggestions = ranked_suggestions[:max_suggestions]

            # Record suggestions for learning
            self._record_suggestions(final_suggestions, context)

            logger.info("Generated %d suggestions (from %d candidates)", len(final_suggestions), len(suggestions))

            return final_suggestions

        except Exception as e:
            logger.error("Error in generate_proactive_suggestions: %s", e)
            return []

    def _generate_workflow_suggestions(
        self, patterns: UsagePatternAnalysis, context_analysis: Dict[str, Any]
    ) -> List[ProactiveSuggestion]:
        """Generate workflow improvement suggestions."""
        suggestions = []

        try:
            # Analyze recent patterns for workflow improvements
            for pattern in patterns.recent_patterns:
                if pattern.pattern_type == "sequence" and pattern.frequency >= 2:
                    suggestions.append(
                        ProactiveSuggestion(
                            suggestion_type="workflow_improvement",
                            tool_id="create_workflow",
                            reasoning=f"Create workflow for common sequence: {pattern.description}",
                            confidence=0.7,
                            expected_improvement="Automate repetitive task sequences",
                            pattern_strength=pattern.confidence,
                        )
                    )

                elif pattern.pattern_type == "repetitive" and pattern.confidence > 0.7:
                    suggestions.append(
                        ProactiveSuggestion(
                            suggestion_type="workflow_improvement",
                            tool_id="batch_processor",
                            reasoning="Use batch processing for repetitive operations",
                            confidence=0.8,
                            expected_improvement="Process multiple items efficiently",
                            pattern_strength=pattern.confidence,
                        )
                    )

        except Exception as e:
            logger.warning("Error generating workflow suggestions: %s", e)

        return suggestions

    def _rank_suggestions(
        self, suggestions: List[ProactiveSuggestion], context_analysis: Dict[str, Any]
    ) -> List[ProactiveSuggestion]:
        """Rank suggestions by relevance and confidence."""

        try:

            def calculate_rank_score(suggestion: ProactiveSuggestion) -> float:
                base_score = suggestion.confidence

                # Boost high-priority suggestions
                priority_boost = {"high": 0.2, "medium": 0.1, "low": 0.0}.get(suggestion.priority, 0.0)

                # Boost suggestions with strong pattern evidence
                pattern_boost = suggestion.pattern_strength * 0.3

                # Recent suggestions get slight penalty to encourage diversity
                recency_penalty = self._calculate_recency_penalty(suggestion)

                return base_score + priority_boost + pattern_boost - recency_penalty

            # Sort by rank score
            return sorted(suggestions, key=calculate_rank_score, reverse=True)

        except Exception as e:
            logger.warning("Error ranking suggestions: %s", e)
            return suggestions

    def _calculate_recency_penalty(self, suggestion: ProactiveSuggestion) -> float:
        """Calculate penalty for recently suggested tools."""
        penalty = 0.0

        try:
            # Check if this tool was suggested recently
            recent_suggestions = [
                s for s in self.suggestion_history[-10:] if time.time() - s.get("timestamp", 0) < 300
            ]  # Last 5 minutes

            recent_tools = [s.get("tool_id") for s in recent_suggestions]

            if suggestion.tool_id in recent_tools:
                penalty = 0.1  # Small penalty for recent suggestions

        except Exception as e:
            logger.debug("Error calculating recency penalty: %s", e)

        return penalty

    def _record_suggestions(self, suggestions: List[ProactiveSuggestion], context: Dict[str, Any]):
        """Record suggestions for learning and analysis."""
        try:
            for suggestion in suggestions:
                self.suggestion_history.append(
                    {
                        "timestamp": time.time(),
                        "tool_id": suggestion.tool_id,
                        "suggestion_type": suggestion.suggestion_type,
                        "confidence": suggestion.confidence,
                        "context": context.copy(),
                    }
                )

            # Keep history manageable (last 1000 suggestions)
            if len(self.suggestion_history) > 1000:
                self.suggestion_history = self.suggestion_history[-1000:]

        except Exception as e:
            logger.warning("Error recording suggestions: %s", e)

    def get_suggestion_statistics(self) -> Dict[str, Any]:
        """Get statistics about suggestion generation."""
        try:
            if not self.suggestion_history:
                return {"total_suggestions": 0}

            recent_suggestions = [
                s for s in self.suggestion_history if time.time() - s["timestamp"] < 3600
            ]  # Last hour

            tool_counts = {}
            for suggestion in recent_suggestions:
                tool_id = suggestion["tool_id"]
                tool_counts[tool_id] = tool_counts.get(tool_id, 0) + 1

            return {
                "total_suggestions": len(self.suggestion_history),
                "recent_suggestions": len(recent_suggestions),
                "most_suggested_tools": sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:5],
                "average_confidence": sum(s["confidence"] for s in recent_suggestions)
                / max(1, len(recent_suggestions)),
            }

        except Exception as e:
            logger.error("Error getting suggestion statistics: %s", e)
            return {"error": str(e)}
