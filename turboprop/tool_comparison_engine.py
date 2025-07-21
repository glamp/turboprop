#!/usr/bin/env python3
"""
tool_comparison_engine.py: Core tool comparison and analysis system.

This module provides the central orchestrator for comprehensive tool comparison
capabilities, integrating multi-dimensional analysis, alternative detection,
and decision support for intelligent tool selection.
"""

import statistics
import time
from typing import Any, Dict, List, Optional

from .alternative_detector import AlternativeDetector
from .comparison_formatter import ComparisonFormatter
from .comparison_metrics import ComparisonMetrics
from .comparison_types import DetailedComparison, TaskComparisonResult, ToolComparisonResult
from .context_analyzer import TaskContext
from .decision_support import DecisionSupport
from .logging_config import get_logger
from .tool_search_results import ToolSearchResult

logger = get_logger(__name__)


class ToolComparisonEngine:
    """Comprehensive tool comparison and analysis system."""

    def __init__(
        self,
        alternative_detector: AlternativeDetector,
        comparison_metrics: ComparisonMetrics,
        decision_support: DecisionSupport,
    ):
        """
        Initialize the tool comparison engine.

        Args:
            alternative_detector: System for detecting alternative tools
            comparison_metrics: System for calculating comparison metrics
            decision_support: System for providing decision guidance
        """
        self.alternative_detector = alternative_detector
        self.comparison_metrics = comparison_metrics
        self.decision_support = decision_support
        self.formatter = ComparisonFormatter()

        # Performance tracking
        self.comparison_cache = {}
        self.performance_metrics = {"total_comparisons": 0, "cache_hits": 0, "avg_comparison_time": 0.0}

        logger.info("Tool comparison engine initialized")

    def compare_tools(
        self,
        tool_ids: List[str],
        comparison_criteria: Optional[List[str]] = None,
        context: Optional[TaskContext] = None,
    ) -> ToolComparisonResult:
        """
        Perform comprehensive comparison of multiple tools.

        Args:
            tool_ids: List of tool IDs to compare
            comparison_criteria: Optional specific criteria to focus on
            context: Optional task context for contextualized comparison

        Returns:
            ToolComparisonResult with comprehensive analysis
        """
        start_time = time.time()

        # Validation - let these exceptions bubble up to caller
        if len(tool_ids) < 2:
            raise ValueError("At least two tools are required for comparison")

        try:
            logger.info(f"Starting comparison of {len(tool_ids)} tools: {tool_ids}")

            # Check cache first
            cached_result = self._check_comparison_cache(tool_ids, comparison_criteria, context)
            if cached_result:
                return cached_result

            # Perform the comparison analysis
            comparison_data = self._perform_comparison_analysis(tool_ids, comparison_criteria, context)

            # Create and cache the final result
            result = self._create_final_comparison_result(
                comparison_data, tool_ids, comparison_criteria, context, start_time
            )

            # Update performance tracking
            self._update_comparison_performance_metrics(start_time)

            logger.info(f"Tool comparison completed in {result.execution_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Error during tool comparison: {e}")
            execution_time = time.time() - start_time
            return self._create_error_result(tool_ids, str(e), execution_time)

    def compare_for_task(
        self, task_description: str, candidate_tools: Optional[List[str]] = None, max_comparisons: int = 5
    ) -> TaskComparisonResult:
        """
        Compare tools specifically for a given task.

        Args:
            task_description: Description of the task to optimize for
            candidate_tools: Optional list of candidate tools (if None, will search)
            max_comparisons: Maximum number of tools to compare

        Returns:
            TaskComparisonResult with task-specific analysis
        """
        start_time = time.time()

        try:
            logger.info(f"Starting task-specific comparison for: {task_description[:50]}...")

            # Step 1: Find or validate candidate tools
            if not candidate_tools:
                candidate_tools = self._find_task_relevant_tools(task_description, max_comparisons)

            if len(candidate_tools) < 2:
                logger.warning(f"Insufficient candidate tools found: {candidate_tools}")
                candidate_tools.extend(["read", "write"])  # Add defaults for comparison

            # Limit to max_comparisons
            candidate_tools = candidate_tools[:max_comparisons]

            # Step 2: Load tool data
            tools_data = self._load_tools_data(candidate_tools)

            # Step 3: Calculate task-aware metrics
            logger.debug("Calculating task-specific metrics")
            task_context = TaskContext()  # Would be enhanced with actual task analysis
            comparison_matrix = self.comparison_metrics.calculate_all_metrics(tools_data, task_context)

            # Step 4: Calculate task fit scores
            task_fit_scores = self._calculate_task_fit_scores(tools_data, task_description, comparison_matrix)

            # Step 5: Generate task-specific ranking
            task_ranking = self._calculate_task_specific_ranking(comparison_matrix, task_fit_scores)

            # Step 6: Calculate recommendation confidence
            recommendation_confidence = self._calculate_recommendation_confidence(
                task_ranking, task_fit_scores, comparison_matrix
            )

            result = TaskComparisonResult(
                task_description=task_description,
                candidate_tools=candidate_tools,
                comparison_matrix=comparison_matrix,
                task_specific_ranking=task_ranking,
                task_fit_scores=task_fit_scores,
                recommendation_confidence=recommendation_confidence,
            )

            execution_time = time.time() - start_time
            logger.info(f"Task-specific comparison completed in {execution_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Error in task-specific comparison: {e}")
            execution_time = time.time() - start_time
            return self._create_task_error_result(task_description, candidate_tools or [], str(e), execution_time)

    def get_detailed_comparison(
        self, tool_a: str, tool_b: str, focus_areas: Optional[List[str]] = None
    ) -> DetailedComparison:
        """
        Get detailed head-to-head comparison of two tools.

        Args:
            tool_a: First tool for comparison
            tool_b: Second tool for comparison
            focus_areas: Optional specific areas to focus analysis on

        Returns:
            DetailedComparison with comprehensive head-to-head analysis
        """
        try:
            logger.info(f"Performing detailed comparison: {tool_a} vs {tool_b}")

            # Perform detailed analysis
            return self._perform_detailed_analysis(tool_a, tool_b, focus_areas)

        except Exception as e:
            logger.error(f"Error in detailed comparison: {e}")
            return self._create_detailed_comparison_error_result(tool_a, tool_b, str(e))

    # Helper methods

    def _check_comparison_cache(
        self, tool_ids: List[str], comparison_criteria: Optional[List[str]], context: Optional[TaskContext]
    ) -> Optional[ToolComparisonResult]:
        """Check if comparison result is cached."""
        cache_key = self._generate_comparison_cache_key(tool_ids, comparison_criteria, context)
        if cache_key in self.comparison_cache:
            logger.debug("Returning cached comparison result")
            self.performance_metrics["cache_hits"] += 1
            return self.comparison_cache[cache_key]
        return None

    def _perform_comparison_analysis(
        self, tool_ids: List[str], comparison_criteria: Optional[List[str]], context: Optional[TaskContext]
    ) -> Dict[str, Any]:
        """Perform the core comparison analysis."""
        # Step 1: Load tool metadata
        tools_data = self._load_tools_data(tool_ids)
        if not tools_data:
            raise ValueError("Could not load data for any of the specified tools")

        # Step 2: Calculate comparison metrics
        logger.debug("Calculating comparison metrics")
        comparison_matrix = self.comparison_metrics.calculate_all_metrics(tools_data, context)

        # Filter by comparison criteria if specified
        if comparison_criteria:
            comparison_matrix = self._filter_by_criteria(comparison_matrix, comparison_criteria)

        # Step 3: Generate rankings
        logger.debug("Generating tool rankings")
        overall_ranking = self._calculate_overall_ranking(comparison_matrix)
        category_rankings = self._calculate_category_rankings(tools_data, comparison_matrix)

        # Step 4: Identify key differentiators
        key_differentiators = self._identify_key_differentiators(comparison_matrix)

        # Step 5: Analyze trade-offs
        logger.debug("Analyzing trade-offs")
        trade_offs = self.decision_support.analyze_trade_offs(tools_data, comparison_matrix)

        # Step 6: Generate decision guidance
        logger.debug("Generating decision guidance")
        confidence_scores = self._calculate_confidence_scores(comparison_matrix, tools_data)

        return {
            "tools_data": tools_data,
            "comparison_matrix": comparison_matrix,
            "overall_ranking": overall_ranking,
            "category_rankings": category_rankings,
            "key_differentiators": key_differentiators,
            "trade_offs": trade_offs,
            "confidence_scores": confidence_scores,
        }

    def _create_final_comparison_result(
        self,
        comparison_data: Dict[str, Any],
        tool_ids: List[str],
        comparison_criteria: Optional[List[str]],
        context: Optional[TaskContext],
        start_time: float,
    ) -> ToolComparisonResult:
        """Create and cache the final comparison result."""
        # Create preliminary result for decision guidance
        preliminary_result = ToolComparisonResult(
            compared_tools=tool_ids,
            comparison_matrix=comparison_data["comparison_matrix"],
            overall_ranking=comparison_data["overall_ranking"],
            category_rankings=comparison_data["category_rankings"],
            key_differentiators=comparison_data["key_differentiators"],
            trade_off_analysis=comparison_data["trade_offs"],
            decision_guidance=None,  # Will be filled next
            comparison_criteria=comparison_criteria
            or list(comparison_data["comparison_matrix"].get(tool_ids[0], {}).keys()),
            context_factors=self._extract_context_factors(context),
            confidence_scores=comparison_data["confidence_scores"],
        )

        decision_guidance = self.decision_support.generate_selection_guidance(preliminary_result, context)

        # Create final result
        result = ToolComparisonResult(
            compared_tools=tool_ids,
            comparison_matrix=comparison_data["comparison_matrix"],
            overall_ranking=comparison_data["overall_ranking"],
            category_rankings=comparison_data["category_rankings"],
            key_differentiators=comparison_data["key_differentiators"],
            trade_off_analysis=comparison_data["trade_offs"],
            decision_guidance=decision_guidance,
            comparison_criteria=comparison_criteria
            or list(comparison_data["comparison_matrix"].get(tool_ids[0], {}).keys()),
            context_factors=self._extract_context_factors(context),
            confidence_scores=comparison_data["confidence_scores"],
            execution_time=time.time() - start_time,
            tools_analyzed=len(comparison_data["tools_data"]),
        )

        # Cache result
        cache_key = self._generate_comparison_cache_key(tool_ids, comparison_criteria, context)
        self.comparison_cache[cache_key] = result

        return result

    def _update_comparison_performance_metrics(self, start_time: float) -> None:
        """Update performance metrics for comparison operation."""
        self.performance_metrics["total_comparisons"] += 1
        self._update_average_time(time.time() - start_time)

    def _load_tools_data(self, tool_ids: List[str]) -> List[ToolSearchResult]:
        """Load ToolSearchResult objects for the given tool IDs."""
        tools_data = []
        failed_tools = []

        for tool_id in tool_ids:
            try:
                # Mock tool data creation - would integrate with actual data source
                tool_data = self._create_mock_tool_data(tool_id)
                tools_data.append(tool_data)

            except Exception as e:
                logger.warning(f"Could not load data for tool {tool_id}: {e}")
                failed_tools.append(tool_id)
                continue

        if failed_tools:
            logger.info(f"Successfully loaded {len(tools_data)} tools, failed to load {len(failed_tools)} tools")

        if not tools_data:
            raise ValueError(f"Failed to load data for any of the specified tools: {tool_ids}")

        return tools_data

    def _create_mock_tool_data(self, tool_id: str) -> ToolSearchResult:
        """Create mock ToolSearchResult for testing."""
        from .mcp_metadata_types import ParameterAnalysis, ToolId

        # Mock parameter data based on tool type
        param_data = {
            "read": [
                ParameterAnalysis(name="file_path", type="string", required=True, description="Path to file"),
                ParameterAnalysis(name="limit", type="number", required=False, description="Line limit"),
            ],
            "write": [
                ParameterAnalysis(name="file_path", type="string", required=True, description="Path to file"),
                ParameterAnalysis(name="content", type="string", required=True, description="Content to write"),
            ],
            "edit": [
                ParameterAnalysis(name="file_path", type="string", required=True, description="Path to file"),
                ParameterAnalysis(name="old_string", type="string", required=True, description="Text to replace"),
                ParameterAnalysis(name="new_string", type="string", required=True, description="Replacement text"),
            ],
            "grep": [
                ParameterAnalysis(name="pattern", type="string", required=True, description="Search pattern"),
                ParameterAnalysis(name="path", type="string", required=False, description="Search path"),
            ],
            "bash": [ParameterAnalysis(name="command", type="string", required=True, description="Command to execute")],
        }

        descriptions = {
            "read": "Read file contents from the local filesystem",
            "write": "Write content to a file on the local filesystem",
            "edit": "Edit files by replacing text content",
            "grep": "Search for patterns in file contents",
            "bash": "Execute bash commands in a shell environment",
        }

        parameters = param_data.get(tool_id, [])
        description = descriptions.get(tool_id, f"Mock description for {tool_id}")

        return ToolSearchResult(
            tool_id=ToolId(tool_id),
            name=tool_id,
            description=description,
            category="mock_category",
            tool_type="function",
            similarity_score=0.8,
            relevance_score=0.8,
            confidence_level="high",
            parameters=parameters,
            parameter_count=len(parameters),
            required_parameter_count=sum(1 for p in parameters if p.required),
        )

    def _filter_by_criteria(
        self, comparison_matrix: Dict[str, Dict[str, float]], criteria: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Filter comparison matrix to only include specified criteria."""
        filtered_matrix = {}

        for tool_id, metrics in comparison_matrix.items():
            filtered_metrics = {criterion: score for criterion, score in metrics.items() if criterion in criteria}
            filtered_matrix[tool_id] = filtered_metrics

        return filtered_matrix

    def _calculate_overall_ranking(self, comparison_matrix: Dict[str, Dict[str, float]]) -> List[str]:
        """Calculate overall ranking based on weighted average scores."""
        tool_scores = {}

        for tool_id, metrics in comparison_matrix.items():
            if metrics:
                # Simple average for now - could be weighted
                overall_score = statistics.mean(metrics.values())
                tool_scores[tool_id] = overall_score
            else:
                tool_scores[tool_id] = 0.0

        # Sort by score (descending)
        ranked_tools = sorted(tool_scores.keys(), key=lambda t: tool_scores[t], reverse=True)

        return ranked_tools

    def _calculate_category_rankings(
        self, tools_data: List[ToolSearchResult], comparison_matrix: Dict[str, Dict[str, float]]
    ) -> Dict[str, List[str]]:
        """Calculate rankings within categories."""
        # Group tools by category
        category_groups = {}
        for tool in tools_data:
            category = tool.category or "unknown"
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(str(tool.tool_id))

        # Rank within each category
        category_rankings = {}
        for category, tool_ids in category_groups.items():
            if len(tool_ids) > 1:
                # Calculate scores for tools in this category
                category_scores = {}
                for tool_id in tool_ids:
                    if tool_id in comparison_matrix and comparison_matrix[tool_id]:
                        category_scores[tool_id] = statistics.mean(comparison_matrix[tool_id].values())
                    else:
                        category_scores[tool_id] = 0.0

                # Sort by score
                category_rankings[category] = sorted(
                    category_scores.keys(), key=lambda t: category_scores[t], reverse=True
                )
            else:
                category_rankings[category] = tool_ids

        return category_rankings

    def _identify_key_differentiators(self, comparison_matrix: Dict[str, Dict[str, float]]) -> List[str]:
        """Identify metrics that show the most variation between tools."""
        if not comparison_matrix:
            return []

        # Get all metrics
        all_metrics = set()
        for tool_metrics in comparison_matrix.values():
            all_metrics.update(tool_metrics.keys())

        # Calculate variance for each metric
        metric_variances = {}
        for metric in all_metrics:
            scores = []
            for tool_metrics in comparison_matrix.values():
                if metric in tool_metrics:
                    scores.append(tool_metrics[metric])

            if len(scores) > 1:
                variance = statistics.variance(scores)
                metric_variances[metric] = variance

        # Return metrics with highest variance (most differentiating)
        sorted_metrics = sorted(metric_variances.keys(), key=lambda m: metric_variances[m], reverse=True)

        return sorted_metrics[:3]  # Top 3 differentiators

    def _calculate_confidence_scores(
        self, comparison_matrix: Dict[str, Dict[str, float]], tools_data: List[ToolSearchResult]
    ) -> Dict[str, float]:
        """Calculate confidence scores for each tool's assessment."""
        confidence_scores = {}

        for tool in tools_data:
            tool_id = str(tool.tool_id)

            # Base confidence on data completeness and score consistency
            if tool_id in comparison_matrix:
                metrics = comparison_matrix[tool_id]

                # Data completeness factor
                completeness = len(metrics) / 6.0  # Assuming 6 total metrics

                # Score consistency factor (lower variance = higher confidence)
                if len(metrics) > 1:
                    variance = statistics.variance(metrics.values())
                    consistency = max(0.0, 1.0 - variance)
                else:
                    consistency = 0.5

                # Overall score factor (moderate scores are more confident)
                avg_score = statistics.mean(metrics.values()) if metrics else 0.5
                score_confidence = 1.0 - abs(avg_score - 0.5)  # Peak confidence at 0.5

                # Combine factors
                confidence = completeness * 0.4 + consistency * 0.4 + score_confidence * 0.2
                confidence_scores[tool_id] = min(confidence, 1.0)
            else:
                confidence_scores[tool_id] = 0.1  # Very low confidence if no data

        return confidence_scores

    def _extract_context_factors(self, context: Optional[TaskContext]) -> List[str]:
        """Extract relevant context factors for analysis."""
        factors = []

        if context:
            if context.user_context:
                factors.append("user_context_provided")
            if context.project_context:
                factors.append("project_context_provided")
            if context.environmental_constraints:
                factors.append("environmental_constraints")
            if context.time_constraints:
                factors.append("time_constraints")
            if context.quality_requirements:
                factors.append("quality_requirements")

        return factors

    def _find_task_relevant_tools(self, task_description: str, max_tools: int) -> List[str]:
        """Find tools relevant to a specific task (mock implementation)."""
        # Mock implementation - would use actual search
        task_lower = task_description.lower()

        candidates = []

        if any(word in task_lower for word in ["read", "view", "display", "show"]):
            candidates.append("read")
        if any(word in task_lower for word in ["write", "create", "save"]):
            candidates.append("write")
        if any(word in task_lower for word in ["edit", "modify", "change", "update"]):
            candidates.append("edit")
        if any(word in task_lower for word in ["search", "find", "grep"]):
            candidates.append("grep")
        if any(word in task_lower for word in ["execute", "run", "command"]):
            candidates.append("bash")

        # Add defaults if not enough candidates
        if len(candidates) < 2:
            candidates.extend(["read", "write"])

        return candidates[:max_tools]

    def _calculate_task_fit_scores(
        self, tools_data: List[ToolSearchResult], task_description: str, comparison_matrix: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate how well each tool fits the specific task."""
        task_fit_scores = {}
        task_lower = task_description.lower()

        for tool in tools_data:
            tool_id = str(tool.tool_id)

            # Base fit score on tool description matching task
            description_lower = tool.description.lower()

            # Simple keyword matching
            fit_score = 0.5  # Base score

            # Boost for direct keyword matches
            tool_keywords = set(description_lower.split())
            task_keywords = set(task_lower.split())
            common_keywords = tool_keywords & task_keywords

            keyword_boost = min(len(common_keywords) * 0.1, 0.3)
            fit_score += keyword_boost

            # Adjust based on functionality score if available
            if tool_id in comparison_matrix:
                functionality_score = comparison_matrix[tool_id].get("functionality", 0.5)
                fit_score = (fit_score * 0.7) + (functionality_score * 0.3)

            task_fit_scores[tool_id] = min(fit_score, 1.0)

        return task_fit_scores

    def _calculate_task_specific_ranking(
        self, comparison_matrix: Dict[str, Dict[str, float]], task_fit_scores: Dict[str, float]
    ) -> List[str]:
        """Calculate ranking optimized for the specific task."""
        combined_scores = {}

        for tool_id in task_fit_scores.keys():
            task_fit = task_fit_scores[tool_id]

            # Get overall quality score
            if tool_id in comparison_matrix and comparison_matrix[tool_id]:
                quality_score = statistics.mean(comparison_matrix[tool_id].values())
            else:
                quality_score = 0.5

            # Combine task fit and quality (favor task fit)
            combined_score = (task_fit * 0.6) + (quality_score * 0.4)
            combined_scores[tool_id] = combined_score

        # Sort by combined score
        return sorted(combined_scores.keys(), key=lambda t: combined_scores[t], reverse=True)

    def _calculate_recommendation_confidence(
        self, task_ranking: List[str], task_fit_scores: Dict[str, float], comparison_matrix: Dict[str, Dict[str, float]]
    ) -> float:
        """Calculate confidence in the task-specific recommendation."""
        if not task_ranking:
            return 0.0

        top_tool = task_ranking[0]
        top_score = task_fit_scores.get(top_tool, 0.0)

        # Higher confidence if there's a clear winner
        if len(task_ranking) > 1:
            second_score = task_fit_scores.get(task_ranking[1], 0.0)
            score_gap = top_score - second_score
            confidence = min(top_score + (score_gap * 0.5), 1.0)
        else:
            confidence = top_score

        return confidence

    def _perform_detailed_analysis(
        self, tool_a: str, tool_b: str, focus_areas: Optional[List[str]]
    ) -> DetailedComparison:
        """Perform detailed head-to-head analysis of two tools."""
        # Mock detailed analysis - would be more comprehensive in real implementation
        similarities = [
            f"Both {tool_a} and {tool_b} handle file operations",
            "Similar parameter structure and usage patterns",
            "Compatible with existing workflows",
        ]

        differences = [
            f"{tool_a} focuses on reading while {tool_b} supports modification",
            "Different complexity levels and feature sets",
            "Varying performance characteristics",
        ]

        tool_a_advantages = [f"{tool_a} offers simpler interface", f"{tool_a} has better performance for its use case"]

        tool_b_advantages = [
            f"{tool_b} provides more comprehensive functionality",
            f"{tool_b} supports advanced features",
        ]

        use_case_scenarios = {
            tool_a: [
                f"Use {tool_a} for simple, straightforward operations",
                f"{tool_a} is ideal for read-only scenarios",
            ],
            tool_b: [f"Use {tool_b} for complex, multi-step operations", f"{tool_b} excels in modification scenarios"],
        }

        switching_guidance = [
            f"Switch from {tool_a} to {tool_b} when you need more functionality",
            f"Switch from {tool_b} to {tool_a} when simplicity is preferred",
        ]

        return DetailedComparison(
            tool_a=tool_a,
            tool_b=tool_b,
            similarities=similarities,
            differences=differences,
            tool_a_advantages=tool_a_advantages,
            tool_b_advantages=tool_b_advantages,
            use_case_scenarios=use_case_scenarios,
            switching_guidance=switching_guidance,
            confidence=0.8,
            analysis_depth="standard",
        )

    def _generate_comparison_cache_key(
        self, tool_ids: List[str], criteria: Optional[List[str]], context: Optional[TaskContext]
    ) -> str:
        """Generate cache key for comparison results."""
        import hashlib

        # Create deterministic string representation
        key_parts = [
            "_".join(sorted(tool_ids)),
            "_".join(sorted(criteria or [])),
            str(hash(str(context))) if context else "no_context",
        ]

        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _update_average_time(self, execution_time: float) -> None:
        """Update running average of execution time."""
        total = self.performance_metrics["total_comparisons"]
        current_avg = self.performance_metrics["avg_comparison_time"]

        # Update running average
        new_avg = ((current_avg * (total - 1)) + execution_time) / total
        self.performance_metrics["avg_comparison_time"] = new_avg

    def _create_error_result(
        self, tool_ids: List[str], error_message: str, execution_time: float
    ) -> ToolComparisonResult:
        """Create error result when comparison fails."""
        logger.warning(f"Creating error result for comparison: {error_message}")

        return ToolComparisonResult(
            compared_tools=tool_ids,
            comparison_matrix={},
            overall_ranking=tool_ids,  # Return in original order
            category_rankings={},
            key_differentiators=[],
            trade_off_analysis=[],
            decision_guidance=None,
            comparison_criteria=[],
            context_factors=[f"error: {error_message}"],
            confidence_scores={tool_id: 0.1 for tool_id in tool_ids},
            execution_time=execution_time,
            tools_analyzed=0,
        )

    def _create_task_error_result(
        self, task_description: str, candidate_tools: List[str], error_message: str, execution_time: float
    ) -> TaskComparisonResult:
        """Create error result when task-specific comparison fails."""
        logger.warning(f"Creating task error result for comparison: {error_message}")

        return TaskComparisonResult(
            task_description=task_description,
            candidate_tools=candidate_tools,
            comparison_matrix={},
            task_specific_ranking=[],
            task_fit_scores={tool_id: 0.1 for tool_id in candidate_tools},
            recommendation_confidence=0.0,
        )

    def _create_detailed_comparison_error_result(
        self, tool_a: str, tool_b: str, error_message: str
    ) -> DetailedComparison:
        """Create error result when detailed comparison fails."""
        logger.warning(f"Creating detailed comparison error result: {error_message}")

        return DetailedComparison(
            tool_a=tool_a,
            tool_b=tool_b,
            similarities=[f"Error occurred during comparison: {error_message}"],
            differences=[],
            tool_a_advantages=[],
            tool_b_advantages=[],
            use_case_scenarios={tool_a: [], tool_b: []},
            switching_guidance=[],
            confidence=0.1,
            analysis_depth="error",
        )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the comparison engine."""
        cache_hit_rate = self.performance_metrics["cache_hits"] / max(self.performance_metrics["total_comparisons"], 1)

        return {**self.performance_metrics, "cache_hit_rate": cache_hit_rate, "cache_size": len(self.comparison_cache)}

    def clear_cache(self) -> None:
        """Clear the comparison cache."""
        cache_size = len(self.comparison_cache)
        self.comparison_cache.clear()
        logger.info(f"Cleared comparison cache ({cache_size} entries removed)")
