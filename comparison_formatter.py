#!/usr/bin/env python3
"""
comparison_formatter.py: Results formatting for tool comparison system.

This module provides comprehensive formatting capabilities for tool comparison
results, decision guidance, and trade-off analysis with support for multiple
output formats (text, JSON, structured tables).
"""

import json
import textwrap
from typing import Any, Dict, List, Optional

from comparison_types import ToolComparisonResult
from decision_support import SelectionGuidance, TradeOffAnalysis
from logging_config import get_logger

logger = get_logger(__name__)

# Formatting configuration
FORMATTING_CONFIG = {
    "table_width": 80,
    "column_width": 20,
    "indent_size": 2,
    "max_text_width": 60,
    "score_precision": 2,
    "confidence_precision": 1,
    "section_separator": "=" * 60,
    "subsection_separator": "-" * 40,
}

# Color codes for terminal output (optional)
COLORS = {
    "header": "\033[1;36m",  # Cyan bold
    "subheader": "\033[1;33m",  # Yellow bold
    "success": "\033[92m",  # Green
    "warning": "\033[93m",  # Yellow
    "error": "\033[91m",  # Red
    "info": "\033[94m",  # Blue
    "reset": "\033[0m",  # Reset
    "bold": "\033[1m",  # Bold
    "underline": "\033[4m",  # Underline
}


class ComparisonFormatter:
    """Formats comparison results for different presentation needs."""

    def __init__(self, use_colors: bool = False):
        """
        Initialize the comparison formatter.

        Args:
            use_colors: Whether to use terminal colors in output
        """
        self.use_colors = use_colors
        self.config = FORMATTING_CONFIG.copy()
        logger.info(f"Comparison formatter initialized (colors: {use_colors})")

    def format_comparison_table(self, comparison_result: ToolComparisonResult, format_type: str = "detailed") -> str:
        """
        Format comparison results as a readable table.

        Args:
            comparison_result: Results from tool comparison
            format_type: 'detailed', 'summary', or 'compact'

        Returns:
            Formatted comparison table as string
        """
        try:
            logger.debug(f"Formatting comparison table ({format_type} format)")

            if format_type == "detailed":
                return self._format_detailed_comparison_table(comparison_result)
            elif format_type == "summary":
                return self._format_summary_comparison_table(comparison_result)
            elif format_type == "compact":
                return self._format_compact_comparison_table(comparison_result)
            else:
                logger.warning(f"Unknown format type: {format_type}, using detailed")
                return self._format_detailed_comparison_table(comparison_result)

        except Exception as e:
            logger.error(f"Error formatting comparison table: {e}")
            return f"Error formatting comparison results: {str(e)}"

    def format_decision_summary(self, selection_guidance: SelectionGuidance) -> str:
        """
        Format decision summary with clear recommendations.

        Args:
            selection_guidance: Selection guidance to format

        Returns:
            Formatted decision summary as string
        """
        try:
            logger.debug("Formatting decision summary")

            sections = []

            # Header
            header = self._format_header("TOOL SELECTION RECOMMENDATION")
            sections.append(header)

            # Main recommendation
            confidence_text = f"{selection_guidance.confidence:.1%}"
            recommendation_section = f"""
{self._format_subheader("Recommended Tool")}
{self._colorize(selection_guidance.recommended_tool.upper(), 'success')}
Confidence: {self._colorize(confidence_text, 'info')}
"""
            sections.append(recommendation_section.strip())

            # Key factors
            if selection_guidance.key_factors:
                factors_section = f"""
{self._format_subheader("Key Decision Factors")}
{self._format_bullet_list(selection_guidance.key_factors)}
"""
                sections.append(factors_section.strip())

            # Why recommended
            if selection_guidance.why_recommended:
                why_section = f"""
{self._format_subheader("Why This Tool is Recommended")}
{self._format_bullet_list(selection_guidance.why_recommended)}
"""
                sections.append(why_section.strip())

            # Context-specific guidance
            context_section = f"""
{self._format_subheader("Context-Specific Guidance")}

{self._colorize("For Beginners:", 'info')}
{self._wrap_text(selection_guidance.beginner_guidance)}

{self._colorize("For Advanced Users:", 'info')}
{self._wrap_text(selection_guidance.advanced_user_guidance)}

{self._colorize("Performance-Critical Scenarios:", 'info')}
{self._wrap_text(selection_guidance.performance_critical_guidance)}
"""
            sections.append(context_section.strip())

            # Alternatives
            if selection_guidance.close_alternatives:
                alternatives_section = f"""
{self._format_subheader("Close Alternatives")}
{self._format_tool_list(selection_guidance.close_alternatives)}
"""
                sections.append(alternatives_section.strip())

            # When to reconsider
            if selection_guidance.when_to_reconsider:
                reconsider_section = f"""
{self._format_subheader("When to Reconsider This Choice")}
{self._format_bullet_list(selection_guidance.when_to_reconsider)}
"""
                sections.append(reconsider_section.strip())

            # Fallback options
            if selection_guidance.fallback_options:
                fallback_section = f"""
{self._format_subheader("Fallback Options")}
{self._format_tool_list(selection_guidance.fallback_options)}
"""
                sections.append(fallback_section.strip())

            return "\n\n".join(sections)

        except Exception as e:
            logger.error(f"Error formatting decision summary: {e}")
            return f"Error formatting decision summary: {str(e)}"

    def format_trade_off_analysis(self, trade_offs: List[TradeOffAnalysis]) -> str:
        """
        Format trade-off analysis for decision support.

        Args:
            trade_offs: List of trade-off analyses to format

        Returns:
            Formatted trade-off analysis as string
        """
        try:
            logger.debug(f"Formatting {len(trade_offs)} trade-off analyses")

            if not trade_offs:
                return self._format_header("No significant trade-offs identified")

            sections = []

            # Header
            header = self._format_header("TRADE-OFF ANALYSIS")
            sections.append(header)

            for i, trade_off in enumerate(trade_offs, 1):
                trade_off_section = self._format_single_trade_off(trade_off, i)
                sections.append(trade_off_section)

            return "\n\n".join(sections)

        except Exception as e:
            logger.error(f"Error formatting trade-off analysis: {e}")
            return f"Error formatting trade-off analysis: {str(e)}"

    def format_alternative_analysis(self, alternatives: List[Any]) -> str:
        """
        Format alternative analysis results.

        Args:
            alternatives: List of AlternativeAnalysis objects

        Returns:
            Formatted alternative analysis as string
        """
        try:
            logger.debug(f"Formatting {len(alternatives)} alternative analyses")

            if not alternatives:
                return self._format_header("No alternatives found")

            sections = []

            # Header
            header = self._format_header("ALTERNATIVE TOOLS ANALYSIS")
            sections.append(header)

            for i, alt in enumerate(alternatives, 1):
                alt_section = self._format_single_alternative(alt, i)
                sections.append(alt_section)

            return "\n\n".join(sections)

        except Exception as e:
            logger.error(f"Error formatting alternative analysis: {e}")
            return f"Error formatting alternative analysis: {str(e)}"

    def format_json_output(self, comparison_result: ToolComparisonResult) -> str:
        """
        Format comparison results as structured JSON.

        Args:
            comparison_result: Results to format as JSON

        Returns:
            JSON-formatted string
        """
        try:
            logger.debug("Formatting comparison results as JSON")

            # Convert to dictionary
            result_dict = comparison_result.to_dict()

            # Pretty print JSON
            return json.dumps(result_dict, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error formatting JSON output: {e}")
            return json.dumps({"error": f"Failed to format JSON: {str(e)}"}, indent=2)

    # Helper methods for specific formatting tasks

    def _build_formatted_sections(
        self,
        title: str,
        comparison_result: ToolComparisonResult,
        include_tools: bool = True,
        include_full_matrix: bool = True,
        include_ranking: bool = True,
        include_differentiators: bool = True,
        include_guidance: bool = True,
        key_metrics_only: Optional[List[str]] = None,
    ) -> str:
        """Build formatted sections with configurable content."""
        sections = []

        # Header
        header = self._format_header(title)
        sections.append(header)

        # Tools being compared
        if include_tools:
            tools_section = f"""
{self._format_subheader("Tools Compared")}
{self._format_tool_list(comparison_result.compared_tools)}
"""
            sections.append(tools_section.strip())

        # Comparison matrix
        if comparison_result.comparison_matrix:
            if key_metrics_only:
                # Filter matrix to key metrics only
                filtered_matrix = {}
                for tool_id, metrics in comparison_result.comparison_matrix.items():
                    filtered_matrix[tool_id] = {
                        metric: score for metric, score in metrics.items() if metric in key_metrics_only
                    }
                matrix_data = filtered_matrix
                matrix_title = "Key Metrics Comparison"
            else:
                matrix_data = comparison_result.comparison_matrix
                matrix_title = "Comparison Matrix"

            if matrix_data and include_full_matrix:
                matrix_section = f"""
{self._format_subheader(matrix_title)}
{self._format_metrics_table(matrix_data)}
"""
                sections.append(matrix_section.strip())

        # Overall ranking
        if comparison_result.overall_ranking and include_ranking:
            ranking_section = f"""
{self._format_subheader("Overall Ranking")}
{self._format_ranking_list(comparison_result.overall_ranking, comparison_result.confidence_scores)}
"""
            sections.append(ranking_section.strip())

        # Key differentiators
        if comparison_result.key_differentiators and include_differentiators:
            diff_section = f"""
{self._format_subheader("Key Differentiators")}
{self._format_key_differentiators(comparison_result.key_differentiators)}
"""
            sections.append(diff_section.strip())

        # Decision guidance
        if comparison_result.decision_guidance and include_guidance:
            guidance_section = f"""
{self._format_subheader("Decision Guidance")}
{self._format_decision_guidance(comparison_result.decision_guidance)}
"""
            sections.append(guidance_section.strip())

        return "\n\n".join(sections)

    def _format_detailed_comparison_table(self, comparison_result: ToolComparisonResult) -> str:
        """Format detailed comparison table with all metrics."""
        return self._build_formatted_sections("DETAILED TOOL COMPARISON", comparison_result)

    def _format_summary_comparison_table(self, comparison_result: ToolComparisonResult) -> str:
        """Format summary comparison table with key metrics only."""
        key_metrics = ["functionality", "usability", "reliability"]
        return self._build_formatted_sections(
            "TOOL COMPARISON SUMMARY",
            comparison_result,
            include_tools=False,
            include_differentiators=False,
            key_metrics_only=key_metrics,
        )

    def _format_compact_comparison_table(self, comparison_result: ToolComparisonResult) -> str:
        """Format compact comparison table for quick reference."""
        if not comparison_result.overall_ranking:
            return "No comparison data available"

        lines = []
        lines.append(self._colorize("TOOL COMPARISON (Compact)", "header"))
        lines.append("")

        # Show ranking with overall scores
        for i, tool in enumerate(comparison_result.overall_ranking, 1):
            confidence = comparison_result.confidence_scores.get(tool, 0.0)
            confidence_text = f"{confidence:.0%}"

            rank_symbol = "★" if i == 1 else f"{i}."
            status_color = "success" if i == 1 else "info"

            line = f"{rank_symbol} {self._colorize(tool, status_color)} ({confidence_text})"
            lines.append(line)

        return "\n".join(lines)

    def _format_metrics_table(self, comparison_matrix: Dict[str, Dict[str, float]]) -> str:
        """Format metrics comparison as a table."""
        if not comparison_matrix:
            return "No metrics data available"

        # Get all tools and metrics
        tools = list(comparison_matrix.keys())
        all_metrics = set()
        for tool_metrics in comparison_matrix.values():
            all_metrics.update(tool_metrics.keys())
        metrics = sorted(list(all_metrics))

        # Calculate column widths
        tool_width = max(len(tool) for tool in tools) + 2
        metric_width = max(
            8, max(len(metric.title()) for metric in metrics) if metrics else 8
        )  # Dynamic width based on longest metric name

        lines = []

        # Header row
        header_parts = ["Tool".ljust(tool_width)]
        for metric in metrics:
            header_parts.append(metric.title().ljust(metric_width))
        lines.append(" | ".join(header_parts))

        # Separator
        sep_parts = ["-" * tool_width]
        for _ in metrics:
            sep_parts.append("-" * metric_width)
        lines.append("-+-".join(sep_parts))

        # Data rows
        for tool in tools:
            row_parts = [tool.ljust(tool_width)]
            tool_metrics = comparison_matrix.get(tool, {})

            for metric in metrics:
                score = tool_metrics.get(metric, 0.0)
                score_text = f"{score:.2f}".ljust(metric_width)

                # Color code scores
                if score >= 0.8:
                    score_text = self._colorize(score_text, "success")
                elif score >= 0.6:
                    score_text = self._colorize(score_text, "info")
                elif score < 0.4:
                    score_text = self._colorize(score_text, "warning")

                row_parts.append(score_text)

            lines.append(" | ".join(row_parts))

        return "\n".join(lines)

    def _format_single_trade_off(self, trade_off: TradeOffAnalysis, index: int) -> str:
        """Format a single trade-off analysis."""
        sections = []

        # Trade-off header
        importance_color = {"critical": "error", "important": "warning", "minor": "info"}.get(
            trade_off.decision_importance, "info"
        )

        trade_off_name = trade_off.trade_off_name.replace("_", " ").title()
        header = f"""
{self._format_subheader(f"Trade-off #{index}: {trade_off_name}")}
Magnitude: {trade_off.magnitude:.1%} | Importance: {self._colorize(trade_off.decision_importance.title(), importance_color)}
Tools Involved: {', '.join(trade_off.tools_involved)}
"""
        sections.append(header.strip())

        # Competing factors
        factors_section = f"""
{self._colorize("Competing Factors:", 'info')}
• {' vs '.join(trade_off.competing_factors)}
"""
        sections.append(factors_section.strip())

        # When each factor matters
        if trade_off.when_factor_a_matters and trade_off.when_factor_b_matters:
            when_section = f"""
{self._colorize(f"When {trade_off.competing_factors[0]} matters more:", 'info')}
{self._format_bullet_list(trade_off.when_factor_a_matters)}

{self._colorize(f"When {trade_off.competing_factors[1]} matters more:", 'info')}
{self._format_bullet_list(trade_off.when_factor_b_matters)}
"""
            sections.append(when_section.strip())

        # Recommendation
        if trade_off.recommendation:
            rec_section = f"""
{self._colorize("Recommendation:", 'info')}
{self._wrap_text(trade_off.recommendation)}
"""
            sections.append(rec_section.strip())

        return "\n\n".join(sections)

    def _format_single_alternative(self, alternative: Any, index: int) -> str:
        """Format a single alternative analysis."""
        sections = []

        # Alternative header
        similarity_text = f"{alternative.similarity_score:.1%}"
        confidence_text = f"{alternative.confidence:.1%}"

        header = f"""
{self._format_subheader(f"Alternative #{index}: {alternative.tool_name}")}
Similarity: {similarity_text} | Confidence: {confidence_text}
Complexity: {alternative.complexity_comparison} | Learning Curve: {alternative.learning_curve}
"""
        sections.append(header.strip())

        # Capabilities analysis
        if alternative.shared_capabilities or alternative.unique_capabilities:
            capabilities_section = f"""
{self._colorize("Capability Analysis:", 'info')}
"""
            if alternative.shared_capabilities:
                shared_list = ", ".join(alternative.shared_capabilities[:3])
                ellipsis = "..." if len(alternative.shared_capabilities) > 3 else ""
                capabilities_section += f"Shared: {shared_list}{ellipsis}\n"

            if alternative.unique_capabilities:
                unique_list = ", ".join(alternative.unique_capabilities[:3])
                ellipsis = "..." if len(alternative.unique_capabilities) > 3 else ""
                capabilities_section += f"Unique: {unique_list}{ellipsis}\n"

            sections.append(capabilities_section.strip())

        # Advantages
        if alternative.advantages:
            adv_section = f"""
{self._colorize("Advantages:", 'success')}
{self._format_bullet_list(alternative.advantages[:3])}
"""
            sections.append(adv_section.strip())

        # When to prefer
        if alternative.when_to_prefer:
            prefer_section = f"""
{self._colorize("When to Prefer This Tool:", 'info')}
{self._format_bullet_list(alternative.when_to_prefer[:3])}
"""
            sections.append(prefer_section.strip())

        return "\n\n".join(sections)

    def _format_header(self, text: str) -> str:
        """Format a main header."""
        decorated_text = f" {text} "
        separator = "=" * max(len(decorated_text), 40)

        return f"""
{self._colorize(separator, 'header')}
{self._colorize(decorated_text.center(len(separator)), 'header')}
{self._colorize(separator, 'header')}
""".strip()

    def _format_subheader(self, text: str) -> str:
        """Format a section subheader."""
        return self._colorize(f"{text}", "subheader")

    def _format_bullet_list(self, items: List[str]) -> str:
        """Format a list of items as bullets."""
        if not items:
            return "None"

        return "\n".join(f"• {self._wrap_text(item, indent=2)}" for item in items)

    def _format_tool_list(self, tools: List[str]) -> str:
        """Format a list of tools."""
        if not tools:
            return "None"

        return ", ".join(self._colorize(tool, "info") for tool in tools)

    def _format_key_differentiators(self, differentiators: List[str]) -> str:
        """Format key differentiators as a bullet list."""
        if not differentiators:
            return "None"

        return "\n".join(f"• {self._wrap_text(diff.replace('_', ' ').title(), indent=2)}" for diff in differentiators)

    def _format_decision_guidance(self, guidance: Any) -> str:
        """Format decision guidance information."""
        if not guidance:
            return "No guidance available"

        if hasattr(guidance, "recommendation"):
            return self._wrap_text(guidance.recommendation)
        elif isinstance(guidance, str):
            return self._wrap_text(guidance)
        else:
            return str(guidance)

    def _format_ranking_list(self, ranking: List[str], confidence_scores: Dict[str, float]) -> str:
        """Format a ranking list with confidence scores."""
        lines = []

        for i, tool in enumerate(ranking, 1):
            confidence = confidence_scores.get(tool, 0.0)
            confidence_text = f"{confidence:.1%}"

            if i == 1:
                line = f"{i}. {self._colorize(tool, 'success')} ({confidence_text}) ← Recommended"
            else:
                line = f"{i}. {self._colorize(tool, 'info')} ({confidence_text})"

            lines.append(line)

        return "\n".join(lines)

    def _wrap_text(self, text: str, width: Optional[int] = None, indent: int = 0) -> str:
        """Wrap text to specified width with optional indentation."""
        if width is None:
            width = self.config["max_text_width"]

        wrapped = textwrap.fill(text, width=width, initial_indent=" " * indent, subsequent_indent=" " * indent)
        return wrapped

    def _colorize(self, text: str, color: str) -> str:
        """Apply color formatting if colors are enabled."""
        if not self.use_colors or color not in COLORS:
            return text

        return f"{COLORS[color]}{text}{COLORS['reset']}"

    def set_formatting_config(self, **kwargs) -> None:
        """Update formatting configuration."""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                logger.debug(f"Updated formatting config: {key} = {value}")
            else:
                logger.warning(f"Unknown formatting config key: {key}")

    def enable_colors(self, enable: bool = True) -> None:
        """Enable or disable color output."""
        self.use_colors = enable
        logger.info(f"Color output {'enabled' if enable else 'disabled'}")
