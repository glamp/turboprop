#!/usr/bin/env python3
"""
Search Result Formatter

This module provides formatting utilities for presenting MCP tool search results
in various user-friendly formats including console output, JSON, and structured summaries.
"""

import json
from typing import Any, Dict, List

from .logging_config import get_logger
from .tool_search_results import ToolSearchResponse, ToolSearchResult

logger = get_logger(__name__)


class SearchResultFormatter:
    """Formats search results for different output formats and use cases."""

    def __init__(self, max_description_length: int = 100, max_examples_shown: int = 2):
        """
        Initialize the search result formatter.

        Args:
            max_description_length: Maximum length for tool descriptions in summaries
            max_examples_shown: Maximum number of examples to show per tool
        """
        self.max_description_length = max_description_length
        self.max_examples_shown = max_examples_shown

    def format_console_output(
        self, response: ToolSearchResponse, show_details: bool = True, max_results: int = 10
    ) -> str:
        """
        Format search results for console/terminal display.

        Args:
            response: Search response to format
            show_details: Whether to show detailed information
            max_results: Maximum number of results to display

        Returns:
            Formatted string suitable for console output
        """
        if not response.results:
            return self._format_no_results(response)

        output_lines = []

        # Header
        output_lines.append(f"🔍 Search Results for: '{response.query}'")
        output_lines.append(f"Found {response.total_results} tools in {response.execution_time:.2f}s")
        output_lines.append(f"Strategy: {response.search_strategy}")
        output_lines.append("")

        # Results
        displayed_count = min(len(response.results), max_results)
        for i, result in enumerate(response.results[:displayed_count], 1):
            output_lines.extend(self._format_single_result_console(result, i, show_details))
            output_lines.append("")

        # Summary footer
        if len(response.results) > max_results:
            output_lines.append(f"... and {len(response.results) - max_results} more results")
            output_lines.append("")

        # Category breakdown
        if response.category_breakdown:
            output_lines.append("📊 Results by Category:")
            for category, count in sorted(response.category_breakdown.items()):
                output_lines.append(f"  • {category}: {count}")
            output_lines.append("")

        # Suggestions
        if response.suggested_refinements:
            output_lines.append("💡 Suggestions to refine your search:")
            for suggestion in response.suggested_refinements:
                output_lines.append(f"  • {suggestion}")

        return "\n".join(output_lines)

    def format_json_output(self, response: ToolSearchResponse, pretty_print: bool = True) -> str:
        """
        Format search results as JSON.

        Args:
            response: Search response to format
            pretty_print: Whether to format JSON with indentation

        Returns:
            JSON string representation of the search results
        """
        try:
            response_dict = response.to_dict()
            if pretty_print:
                return json.dumps(response_dict, indent=2, ensure_ascii=False)
            else:
                return json.dumps(response_dict, ensure_ascii=False)
        except Exception as e:
            logger.error("Error formatting JSON output: %s", e)
            return json.dumps({"error": f"Failed to format results: {e}"}, indent=2)

    def format_summary_output(self, response: ToolSearchResponse) -> str:
        """
        Format a concise summary of search results.

        Args:
            response: Search response to format

        Returns:
            Concise summary string
        """
        if not response.results:
            return f"No tools found for query: '{response.query}'"

        summary_lines = []

        # Quick stats
        high_conf_count = len(response.get_high_confidence_results())
        summary_lines.append(
            f"Found {response.total_results} tools ({high_conf_count} high confidence) "
            f"in {response.execution_time:.2f}s"
        )

        # Top result
        best_result = response.get_best_result()
        if best_result:
            summary_lines.append(
                f"Best match: {best_result.name} "
                f"({best_result.confidence_level} confidence, {best_result.relevance_score:.1%} relevance)"
            )

        # Category distribution
        if len(response.category_breakdown) > 1:
            categories = ", ".join(response.category_breakdown.keys())
            summary_lines.append(f"Categories: {categories}")

        return " | ".join(summary_lines)

    def format_tool_list_output(self, response: ToolSearchResponse, format_style: str = "simple") -> str:
        """
        Format results as a simple tool list.

        Args:
            response: Search response to format
            format_style: Style of formatting ('simple', 'detailed', 'compact')

        Returns:
            Formatted tool list string
        """
        if not response.results:
            return "No tools found."

        if format_style == "simple":
            return self._format_simple_list(response.results)
        elif format_style == "detailed":
            return self._format_detailed_list(response.results)
        elif format_style == "compact":
            return self._format_compact_list(response.results)
        else:
            return self._format_simple_list(response.results)

    def format_clustered_output(self, response: ToolSearchResponse) -> str:
        """
        Format results organized by clusters.

        Args:
            response: Search response to format

        Returns:
            Formatted string with clustered results
        """
        if not response.result_clusters:
            return self.format_console_output(response, show_details=False)

        output_lines = []
        output_lines.append(f"🔍 Clustered Search Results for: '{response.query}'")
        output_lines.append("")

        for cluster in response.result_clusters:
            output_lines.append(f"📁 {cluster.cluster_name}")
            output_lines.append(f"   {cluster.cluster_description}")
            output_lines.append(f"   {cluster.cluster_size} tools, avg relevance: {cluster.average_relevance:.1%}")
            output_lines.append("")

            for i, tool in enumerate(cluster.tools, 1):
                output_lines.append(
                    f"   {i}. {tool.name} ({tool.confidence_level}) - {self._truncate_text(tool.description, 60)}"
                )

            output_lines.append("")

        return "\n".join(output_lines)

    def format_comparison_output(self, results: List[ToolSearchResult]) -> str:
        """
        Format results as a comparison table.

        Args:
            results: List of search results to compare

        Returns:
            Formatted comparison string
        """
        if not results:
            return "No tools to compare."

        output_lines = []
        output_lines.append("🔄 Tool Comparison")
        output_lines.append("")

        # Table header
        output_lines.append(f"{'Tool':<20} {'Category':<15} {'Confidence':<12} {'Complexity':<10} {'Parameters':<10}")
        output_lines.append("-" * 75)

        # Table rows
        for result in results[:10]:  # Limit to 10 for readability
            name = self._truncate_text(result.name, 18)
            category = self._truncate_text(result.category or "N/A", 13)
            confidence = result.confidence_level.capitalize()
            complexity = f"{result.complexity_score:.1%}" if result.complexity_score > 0 else "N/A"
            param_count = f"{result.required_parameter_count}/{result.parameter_count}"

            output_lines.append(f"{name:<20} {category:<15} {confidence:<12} {complexity:<10} {param_count:<10}")

        return "\n".join(output_lines)

    def _format_no_results(self, response: ToolSearchResponse) -> str:
        """Format output when no results are found."""
        lines = []
        lines.append(f"🔍 No tools found for: '{response.query}'")
        lines.append(f"Search completed in {response.execution_time:.2f}s")
        lines.append("")

        if response.suggested_refinements:
            lines.append("💡 Try these suggestions:")
            for suggestion in response.suggested_refinements:
                lines.append(f"  • {suggestion}")
        else:
            lines.append("💡 Try:")
            lines.append("  • Using more general terms")
            lines.append("  • Checking spelling")
            lines.append("  • Searching for tool categories like 'file operations' or 'web tools'")

        return "\n".join(lines)

    def _format_single_result_console(self, result: ToolSearchResult, index: int, show_details: bool) -> List[str]:
        """Format a single result for console output."""
        lines = []

        # Header line
        confidence_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(result.confidence_level, "⚪")
        lines.append(f"{index}. {confidence_emoji} {result.name} ({result.category or 'Unknown'})")

        # Description
        description = self._truncate_text(result.description, self.max_description_length)
        lines.append(f"   📄 {description}")

        # Relevance and similarity scores
        lines.append(
            f"   📊 Relevance: {result.relevance_score:.1%}, "
            f"Similarity: {result.similarity_score:.1%}, "
            f"Confidence: {result.confidence_level}"
        )

        if show_details:
            # Match reasons
            if result.match_reasons:
                lines.append("   🎯 Why it matches:")
                for reason in result.match_reasons[:3]:  # Show top 3 reasons
                    lines.append(f"      • {reason}")

            # Parameters summary
            if result.parameters:
                req_count = result.required_parameter_count
                total_count = result.parameter_count
                lines.append(f"   ⚙️  Parameters: {req_count} required, {total_count} total")

                # Show key parameters
                key_params = [p.name for p in result.parameters if p.required][:3]
                if key_params:
                    lines.append(f"      Key: {', '.join(key_params)}")

            # Examples
            if result.examples:
                lines.append(f"   📝 {len(result.examples)} usage examples available")
                for example in result.examples[: self.max_examples_shown]:
                    use_case = self._truncate_text(example.use_case, 50)
                    lines.append(f"      • {use_case}")

        return lines

    def _format_simple_list(self, results: List[ToolSearchResult]) -> str:
        """Format results as a simple list."""
        lines = []
        for i, result in enumerate(results, 1):
            confidence_indicator = {"high": "●", "medium": "◐", "low": "○"}.get(result.confidence_level, "?")
            lines.append(f"{i:2d}. {confidence_indicator} {result.name} - {result.category or 'Unknown'}")
        return "\n".join(lines)

    def _format_detailed_list(self, results: List[ToolSearchResult]) -> str:
        """Format results as a detailed list."""
        lines = []
        for i, result in enumerate(results, 1):
            lines.append(f"{i:2d}. {result.name} ({result.confidence_level} confidence)")
            lines.append(f"    Category: {result.category or 'Unknown'}")
            lines.append(f"    Description: {self._truncate_text(result.description, 80)}")
            lines.append(f"    Relevance: {result.relevance_score:.1%}")
            if result.match_reasons:
                lines.append(f"    Match: {result.match_reasons[0]}")
            lines.append("")
        return "\n".join(lines)

    def _format_compact_list(self, results: List[ToolSearchResult]) -> str:
        """Format results as a compact list."""
        lines = []
        for result in results:
            conf_short = result.confidence_level[0].upper()  # H, M, L
            lines.append(
                f"{result.name} [{conf_short}] "
                f"({result.relevance_score:.0%}) - "
                f"{self._truncate_text(result.description, 40)}"
            )
        return "\n".join(lines)

    def _truncate_text(self, text: str, max_length: int, suffix: str = "...") -> str:
        """
        Truncate text to maximum length with suffix.

        Args:
            text: Text to truncate
            max_length: Maximum length including suffix
            suffix: Suffix to add when truncating

        Returns:
            Truncated text
        """
        if not text:
            return ""

        if len(text) <= max_length:
            return text

        return text[: max_length - len(suffix)] + suffix

    def create_export_data(self, response: ToolSearchResponse) -> Dict[str, Any]:
        """
        Create data suitable for export to external systems.

        Args:
            response: Search response to export

        Returns:
            Dictionary with exportable data
        """
        export_data: Dict[str, Any] = {
            "metadata": {
                "query": response.query,
                "timestamp": response.execution_time,
                "total_results": response.total_results,
                "search_strategy": response.search_strategy,
            },
            "results": [],
            "summary": {
                "categories": response.category_breakdown,
                "confidence_distribution": response.confidence_distribution,
                "has_good_results": response.has_good_results(),
            },
        }

        for result in response.results:
            export_result = {
                "tool_id": result.tool_id,
                "name": result.name,
                "description": result.description,
                "category": result.category,
                "scores": {
                    "relevance": result.relevance_score,
                    "similarity": result.similarity_score,
                    "confidence": result.confidence_level,
                },
                "parameters": len(result.parameters),
                "required_parameters": result.required_parameter_count,
                "examples": len(result.examples),
                "match_reasons": result.match_reasons,
            }
            export_data["results"].append(export_result)

        return export_data
