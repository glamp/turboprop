#!/usr/bin/env python3
"""
recommendation_explainer_mcp.py: MCP-specific explanation formatting.

This module provides specialized explanation generation and formatting
for MCP tool recommendation responses. It builds on the core recommendation
explainer to create user-friendly explanations optimized for MCP tools.
"""

import time
from typing import Any, Dict, List, Optional

from logging_config import get_logger
from task_analysis_response_types import ToolRecommendation
from task_analyzer import TaskAnalysis

logger = get_logger(__name__)


class MCPExplanationFormatter:
    """
    Formats explanations specifically for MCP tool responses.

    Provides clear, concise explanations that work well in
    conversational interfaces and API responses.
    """

    def __init__(self):
        """Initialize the MCP explanation formatter."""
        # Templates for different types of explanations
        self.explanation_templates = {
            "capability_match": "This tool is recommended because it {capability_description}",
            "complexity_fit": "The tool's {complexity_level} complexity {complexity_fit_reason}",
            "parameter_compatibility": "Tool parameters are {compatibility_level} with your requirements",
            "confidence_high": "High confidence recommendation based on {reasons}",
            "confidence_medium": "Good match with some considerations: {considerations}",
            "confidence_low": "Potential option but may require {additional_requirements}",
        }

        # Reason categorization
        self.reason_categories = {
            "functionality": [
                "provides required functionality",
                "supports needed operations",
                "handles specified tasks",
            ],
            "performance": ["optimized for performance", "efficient processing", "fast execution"],
            "simplicity": ["easy to use", "straightforward interface", "minimal setup required"],
            "flexibility": ["highly configurable", "supports multiple formats", "adaptable to various use cases"],
            "reliability": ["robust error handling", "stable and tested", "reliable operation"],
        }

    def generate_recommendation_explanations(
        self, recommendations: List[ToolRecommendation], task_analysis: Optional[TaskAnalysis]
    ) -> List[str]:
        """
        Generate explanations for why tools were recommended.

        Args:
            recommendations: List of tool recommendations
            task_analysis: Analysis of the original task

        Returns:
            List of human-readable explanations
        """
        explanations = []

        for i, rec in enumerate(recommendations):
            try:
                explanation = self._format_single_recommendation_explanation(rec, i + 1, task_analysis)
                explanations.append(explanation)
            except Exception as e:
                logger.warning(f"Failed to generate explanation for {rec.core.tool_name}: {e}")
                # Fallback explanation
                confidence = f"{rec.core.confidence_score:.1%}"
                explanations.append(
                    f"#{i+1} {rec.core.tool_name}: Recommended for your task with {confidence} confidence."
                )

        return explanations

    def _format_single_recommendation_explanation(
        self, rec: ToolRecommendation, rank: int, task_analysis: Optional[TaskAnalysis]
    ) -> str:
        """
        Format explanation for a single recommendation.

        Args:
            rec: Tool recommendation metadata
            rank: Ranking position (1-based)
            task_analysis: Task analysis for context

        Returns:
            Formatted explanation string
        """
        parts = [f"#{rank} {rec.core.tool_name}:"]

        # Add primary reasons
        if rec.enhancements.recommendation_reasons:
            reasons_text = self._format_reasons(rec.enhancements.recommendation_reasons[:2])
            parts.append(f"Recommended because {reasons_text}.")

        # Add task alignment info
        alignment_text = self._format_task_alignment(rec.core.task_alignment)
        if alignment_text:
            parts.append(alignment_text)

        # Add complexity fit
        if rec.core.complexity_fit != "unknown":
            complexity_text = self._format_complexity_fit(rec.core.complexity_fit, task_analysis)
            if complexity_text:
                parts.append(complexity_text)

        # Add usage guidance
        if rec.enhancements.usage_guidance:
            guidance_text = f"Best used when {rec.enhancements.usage_guidance[0]}"
            parts.append(guidance_text)

        # Add confidence indicator
        confidence_text = self._format_confidence_indicator(rec.core.confidence_score)
        if confidence_text:
            parts.append(confidence_text)

        return " ".join(parts)

    def _format_reasons(self, reasons: List[str]) -> str:
        """Format a list of reasons into natural language."""
        if not reasons:
            return "it matches your requirements"

        if len(reasons) == 1:
            return reasons[0]
        elif len(reasons) == 2:
            return f"{reasons[0]} and {reasons[1]}"
        else:
            return f"{', '.join(reasons[:-1])}, and {reasons[-1]}"

    def _format_task_alignment(self, alignment_score: float) -> str:
        """Format task alignment score into descriptive text."""
        if alignment_score >= 0.9:
            return "Excellent match for your task requirements."
        elif alignment_score >= 0.8:
            return "Very good fit for your task needs."
        elif alignment_score >= 0.7:
            return "Good alignment with your requirements."
        elif alignment_score >= 0.6:
            return "Reasonable fit with some adaptation needed."
        else:
            return "May require significant customization."

    def _format_complexity_fit(self, complexity_fit: str, task_analysis: Optional[TaskAnalysis]) -> str:
        """Format complexity fit explanation."""
        complexity_map = {
            "simple": "straightforward and easy to use",
            "moderate": "balanced complexity appropriate for most users",
            "complex": "advanced but provides comprehensive functionality",
        }

        description = complexity_map.get(complexity_fit, "")
        if not description:
            return ""

        if task_analysis and hasattr(task_analysis, "complexity_level"):
            task_complexity = task_analysis.complexity_level
            if task_complexity == complexity_fit:
                return f"Complexity level ({description}) matches your task perfectly."
            elif task_complexity == "simple" and complexity_fit in ["moderate", "complex"]:
                return f"More sophisticated than needed, but {description}."
            elif task_complexity == "complex" and complexity_fit == "simple":
                return f"Simpler than your task might require, but {description}."

        return f"Tool is {description}."

    def _format_confidence_indicator(self, confidence_score: float) -> str:
        """Format confidence score into descriptive text."""
        if confidence_score >= 0.9:
            return "Very high confidence recommendation."
        elif confidence_score >= 0.8:
            return "High confidence in this recommendation."
        elif confidence_score >= 0.7:
            return "Good confidence level."
        elif confidence_score >= 0.6:
            return "Moderate confidence - worth considering."
        else:
            return "Lower confidence - consider alternatives."


class TaskDescriptionSuggestionGenerator:
    """
    Generates suggestions for improving task descriptions.

    Helps users provide better task descriptions that lead to
    more accurate tool recommendations.
    """

    def __init__(self):
        """Initialize the suggestion generator."""
        self.suggestion_patterns = {
            "too_vague": [
                "Consider adding more specific details about what you want to accomplish",
                "Include the type of data or files you're working with",
                "Specify the expected output or result format",
            ],
            "missing_context": [
                "Add context about your environment or constraints",
                "Mention any specific tools or frameworks you're already using",
                "Include performance or quality requirements",
            ],
            "unclear_scope": [
                "Clarify whether this is a one-time task or recurring workflow",
                "Specify if you need a simple solution or comprehensive functionality",
                "Indicate the scale of data or complexity you're dealing with",
            ],
            "technical_level": [
                "Indicate your technical skill level for appropriate recommendations",
                "Mention if you prefer simple tools or don't mind complexity",
                "Specify if you need beginner-friendly or advanced solutions",
            ],
        }

    def generate_task_description_suggestions(
        self, task_description: str, analysis: Optional[TaskAnalysis]
    ) -> List[str]:
        """
        Generate suggestions for improving task descriptions.

        Args:
            task_description: Original task description
            analysis: Task analysis results

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        # Check for clarity issues
        if analysis and hasattr(analysis, "confidence") and analysis.confidence < 0.7:
            suggestions.extend(self.suggestion_patterns["too_vague"][:1])

        # Check for missing input/output specifications
        if analysis:
            if not getattr(analysis, "input_specifications", []):
                suggestions.append("Specify what type of input data or files you're working with")

            if not getattr(analysis, "output_specifications", []):
                suggestions.append("Describe what output or result you expect")

        # Check for complexity clarity
        if analysis and hasattr(analysis, "complexity_level"):
            if analysis.complexity_level == "complex" and len(task_description.split()) < 10:
                suggestions.extend(self.suggestion_patterns["unclear_scope"][:1])

        # Check for missing context
        if "performance" not in task_description.lower() and "fast" not in task_description.lower():
            if len(task_description.split()) < 8:
                suggestions.extend(self.suggestion_patterns["missing_context"][:1])

        # Check for technical level indication
        technical_indicators = ["beginner", "simple", "advanced", "complex", "easy", "sophisticated"]
        if not any(indicator in task_description.lower() for indicator in technical_indicators):
            suggestions.extend(self.suggestion_patterns["technical_level"][:1])

        return suggestions[:3]  # Limit to top 3 suggestions


class AlternativeComparisonFormatter:
    """
    Formats alternative tool comparisons for MCP responses.

    Provides clear comparisons between primary tools and alternatives
    to help users make informed decisions.
    """

    def __init__(self):
        """Initialize the comparison formatter."""
        pass

    def generate_alternative_comparisons(self, primary_tool: str, alternatives: List[Any]) -> List[str]:
        """
        Generate comparison explanations between primary tool and alternatives.

        Args:
            primary_tool: Name of the primary tool
            alternatives: List of alternative tool recommendations

        Returns:
            List of comparison explanations
        """
        comparisons = []

        if not alternatives:
            comparisons.append(f"{primary_tool} is the best match - no significant alternatives found")
            return comparisons

        # General comparison framework
        comparisons.append(f"{primary_tool} vs alternatives:")

        for alt in alternatives[:3]:  # Limit to top 3 alternatives
            try:
                alt_name = getattr(alt, "tool_name", getattr(alt, "name", str(alt)))
                comparison = self._format_single_alternative_comparison(primary_tool, alt_name, alt)
                comparisons.append(f"• {comparison}")
            except Exception as e:
                logger.warning(f"Failed to format alternative comparison: {e}")
                alt_name = getattr(alt, "tool_name", "Alternative tool")
                comparisons.append(f"• {alt_name}: Alternative option worth considering")

        return comparisons

    def _format_single_alternative_comparison(self, primary_tool: str, alt_name: str, alternative: Any) -> str:
        """
        Format comparison for a single alternative.

        Args:
            primary_tool: Name of primary tool
            alt_name: Name of alternative tool
            alternative: Alternative tool metadata

        Returns:
            Formatted comparison string
        """
        # Default comparison if no specific metadata available
        if not hasattr(alternative, "advantages") and not hasattr(alternative, "similarity_score"):
            return f"{alt_name}: Similar functionality with different approach"

        parts = [alt_name]

        # Add similarity context
        if hasattr(alternative, "similarity_score"):
            similarity = float(alternative.similarity_score)
            if similarity >= 0.8:
                parts.append("very similar functionality")
            elif similarity >= 0.6:
                parts.append("similar core features")
            else:
                parts.append("different approach")

        # Add key advantages
        if hasattr(alternative, "advantages") and alternative.advantages:
            advantage = alternative.advantages[0]  # Take first advantage
            parts.append(f"advantage: {advantage}")

        # Add usage context
        if hasattr(alternative, "when_to_prefer") and alternative.when_to_prefer:
            context = alternative.when_to_prefer[0]  # Take first context
            parts.append(f"prefer when {context}")

        return " - ".join(parts)


class WorkflowAnalysisFormatter:
    """
    Formats workflow analysis for tool sequence recommendations.

    Provides insights into workflow complexity and optimization
    opportunities for multi-step tool sequences.
    """

    def __init__(self):
        """Initialize the workflow analysis formatter."""
        pass

    def analyze_workflow(self, workflow_description: str) -> Dict[str, Any]:
        """
        Analyze workflow description for sequence recommendations.

        Args:
            workflow_description: Description of the workflow

        Returns:
            Dictionary with workflow analysis
        """
        # Simple workflow analysis
        words = workflow_description.split()
        steps = len([word for word in words if word.lower() in ["then", "next", "after", "and"]])

        complexity = "simple" if len(words) < 15 else "moderate" if len(words) < 30 else "complex"

        # Look for coordination requirements
        coordination_keywords = ["together", "parallel", "simultaneously", "combine", "merge"]
        requires_coordination = any(keyword in workflow_description.lower() for keyword in coordination_keywords)

        # Look for data flow
        data_flow_keywords = ["input", "output", "result", "data", "file", "transform"]
        has_data_flow = any(keyword in workflow_description.lower() for keyword in data_flow_keywords)

        return {
            "complexity": complexity,
            "estimated_steps": max(steps + 1, 2),  # At least 2 steps for a workflow
            "requires_coordination": requires_coordination,
            "has_data_flow": has_data_flow,
            "workflow_length": len(words),
            "analysis_timestamp": time.time(),
        }

    def create_workflow_context(
        self, optimization_goal: str = "balanced", max_length: int = 10, allow_parallel: bool = False
    ) -> Dict[str, Any]:
        """
        Create workflow context for sequence optimization.

        Args:
            optimization_goal: What to optimize for
            max_length: Maximum sequence length
            allow_parallel: Whether parallel execution is allowed

        Returns:
            Workflow context dictionary
        """
        return {
            "optimization_goal": optimization_goal,
            "constraints": {"max_sequence_length": max_length, "allow_parallel_execution": allow_parallel},
            "preferences": {
                "prefer_simple_tools": optimization_goal == "simplicity",
                "prefer_fast_execution": optimization_goal == "speed",
                "prefer_reliable_tools": optimization_goal == "reliability",
            },
            "created_at": time.time(),
        }


# Utility functions for MCP explanation formatting


def format_explanation_list(explanations: List[str], max_length: int = 500) -> str:
    """
    Format a list of explanations into a single string with appropriate length.

    Args:
        explanations: List of explanation strings
        max_length: Maximum total length

    Returns:
        Formatted explanation string
    """
    if not explanations:
        return "No explanations available."

    formatted = "\n".join(f"• {exp}" for exp in explanations)

    if len(formatted) <= max_length:
        return formatted

    # Truncate if too long
    truncated = formatted[: max_length - 20]
    last_bullet = truncated.rfind("•")
    if last_bullet > 0:
        truncated = truncated[:last_bullet]

    return truncated + "\n• [Additional explanations available...]"


def create_confidence_summary(recommendations: List[ToolRecommendation]) -> str:
    """
    Create a summary of confidence levels across recommendations.

    Args:
        recommendations: List of recommendations

    Returns:
        Confidence summary string
    """
    if not recommendations:
        return "No recommendations to analyze."

    high_confidence = sum(1 for r in recommendations if r.confidence_score >= 0.8)
    medium_confidence = sum(1 for r in recommendations if 0.6 <= r.confidence_score < 0.8)
    low_confidence = sum(1 for r in recommendations if r.confidence_score < 0.6)

    total = len(recommendations)

    summary_parts = []
    if high_confidence > 0:
        summary_parts.append(f"{high_confidence} high-confidence")
    if medium_confidence > 0:
        summary_parts.append(f"{medium_confidence} medium-confidence")
    if low_confidence > 0:
        summary_parts.append(f"{low_confidence} lower-confidence")

    if not summary_parts:
        return f"All {total} recommendations have standard confidence levels."

    return f"Confidence distribution: {', '.join(summary_parts)} recommendation(s)."
