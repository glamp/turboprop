#!/usr/bin/env python3
"""
decision_support.py: Decision support system for tool selection.

This module provides intelligent guidance for tool selection decisions,
including trade-off analysis, selection guidance, and decision trees.
"""

import statistics
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from comparison_constants import CONFIDENCE_FACTORS, DECISION_THRESHOLDS, SCORE_LIMITS, TASK_SCORING_WEIGHTS
from context_analyzer import TaskContext
from logging_config import get_logger
from tool_search_results import ToolSearchResult

if TYPE_CHECKING:
    from comparison_types import ToolComparisonResult

logger = get_logger(__name__)


@dataclass
class TradeOffAnalysis:
    """Analysis of trade-offs between tool options."""

    trade_off_name: str
    tools_involved: List[str]

    # Trade-off details
    competing_factors: List[str]
    magnitude: float  # How significant is this trade-off (0-1)
    decision_importance: str  # 'critical', 'important', 'minor'

    # Guidance
    when_factor_a_matters: List[str]
    when_factor_b_matters: List[str]
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class SelectionGuidance:
    """Guidance for tool selection decision."""

    recommended_tool: str
    confidence: float  # 0-1 confidence in recommendation

    # Decision rationale
    key_factors: List[str]
    why_recommended: List[str]
    when_to_reconsider: List[str]

    # Alternatives
    close_alternatives: List[str]
    fallback_options: List[str]

    # Context-specific advice
    beginner_guidance: str
    advanced_user_guidance: str
    performance_critical_guidance: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class DecisionNode:
    """Node in a decision tree."""

    criterion: str
    threshold: float
    true_branch: Optional["DecisionNode"]
    false_branch: Optional["DecisionNode"]
    leaf_recommendation: Optional[str]
    confidence: float
    explanation: str


@dataclass
class DecisionTree:
    """Decision tree for tool selection."""

    root_node: DecisionNode
    tools_considered: List[str]
    decision_factors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tools_considered": self.tools_considered,
            "decision_factors": self.decision_factors,
            "tree_structure": self._node_to_dict(self.root_node),
        }

    def _node_to_dict(self, node: DecisionNode) -> Dict[str, Any]:
        """Convert decision node to dictionary."""
        result = {
            "criterion": node.criterion,
            "threshold": node.threshold,
            "confidence": node.confidence,
            "explanation": node.explanation,
        }

        if node.leaf_recommendation:
            result["recommendation"] = node.leaf_recommendation
        else:
            if node.true_branch:
                result["true_branch"] = self._node_to_dict(node.true_branch)
            if node.false_branch:
                result["false_branch"] = self._node_to_dict(node.false_branch)

        return result


# Decision rule templates
DECISION_RULES = {
    "usability_first": {
        "priority_metrics": ["usability", "documentation"],
        "threshold": DECISION_THRESHOLDS["excellence_threshold"],
        "description": "Prioritize ease of use and documentation quality",
    },
    "functionality_first": {
        "priority_metrics": ["functionality", "reliability"],
        "threshold": DECISION_THRESHOLDS["competence_threshold"],
        "description": "Prioritize feature richness and reliability",
    },
    "performance_first": {
        "priority_metrics": ["performance", "compatibility"],
        "threshold": DECISION_THRESHOLDS["quality_threshold"],
        "description": "Prioritize speed and compatibility",
    },
    "balanced": {
        "priority_metrics": ["usability", "functionality", "reliability"],
        "threshold": DECISION_THRESHOLDS["usability_threshold"],
        "description": "Balance multiple factors",
    },
}

# Trade-off patterns
TRADE_OFF_PATTERNS = [
    {
        "name": "functionality_vs_usability",
        "metrics": ["functionality", "usability"],
        "threshold": DECISION_THRESHOLDS["trade_off_threshold"],
        "description": "Feature richness versus ease of use",
    },
    {
        "name": "performance_vs_functionality",
        "metrics": ["performance", "functionality"],
        "threshold": DECISION_THRESHOLDS["performance_difference_threshold"],
        "description": "Speed versus feature completeness",
    },
    {
        "name": "usability_vs_performance",
        "metrics": ["usability", "performance"],
        "threshold": DECISION_THRESHOLDS["performance_difference_threshold"],
        "description": "Ease of use versus execution speed",
    },
    {
        "name": "reliability_vs_functionality",
        "metrics": ["reliability", "functionality"],
        "threshold": DECISION_THRESHOLDS["complexity_difference_threshold"],
        "description": "Stability versus feature richness",
    },
]


class DecisionSupport:
    """Intelligent decision support system for tool selection."""

    def __init__(self):
        """Initialize the decision support system."""
        self.decision_rules = DECISION_RULES.copy()
        self.trade_off_patterns = TRADE_OFF_PATTERNS.copy()
        self.scenario_templates = self._load_scenario_templates()

        logger.info("Decision support system initialized")

    def generate_selection_guidance(
        self, comparison_result: "ToolComparisonResult", task_context: Optional[TaskContext] = None
    ) -> SelectionGuidance:
        """
        Generate guidance for tool selection based on comparison results.

        Args:
            comparison_result: Results from tool comparison
            task_context: Optional task context for personalized guidance

        Returns:
            SelectionGuidance with recommendation and rationale
        """
        try:
            logger.info(f"Generating selection guidance for {len(comparison_result.compared_tools)} tools")

            # Determine recommendation strategy based on context
            strategy = self._determine_recommendation_strategy(task_context)

            # Apply strategy to find best tool
            recommended_tool, confidence = self._apply_recommendation_strategy(comparison_result, strategy)

            # Generate rationale
            key_factors = self._identify_key_decision_factors(comparison_result, recommended_tool)
            why_recommended = self._generate_recommendation_reasons(comparison_result, recommended_tool, strategy)
            when_to_reconsider = self._generate_reconsideration_scenarios(comparison_result, recommended_tool)

            # Find alternatives
            close_alternatives = self._find_close_alternatives(comparison_result, recommended_tool)
            fallback_options = self._find_fallback_options(comparison_result, recommended_tool)

            # Generate context-specific guidance
            context_guidance = self._generate_context_specific_guidance(
                comparison_result, recommended_tool, task_context
            )

            guidance = SelectionGuidance(
                recommended_tool=recommended_tool,
                confidence=confidence,
                key_factors=key_factors,
                why_recommended=why_recommended,
                when_to_reconsider=when_to_reconsider,
                close_alternatives=close_alternatives,
                fallback_options=fallback_options,
                beginner_guidance=context_guidance["beginner"],
                advanced_user_guidance=context_guidance["advanced"],
                performance_critical_guidance=context_guidance["performance"],
            )

            logger.info(f"Generated guidance recommending '{recommended_tool}' with {confidence:.2f} confidence")
            return guidance

        except Exception as e:
            logger.error(f"Error generating selection guidance: {e}")
            # Return fallback guidance
            return self._create_fallback_guidance(comparison_result)

    def analyze_trade_offs(
        self, tools: List[ToolSearchResult], metrics: Dict[str, Dict[str, float]]
    ) -> List[TradeOffAnalysis]:
        """
        Analyze trade-offs between tool choices.

        Args:
            tools: List of tools being compared
            metrics: Metric scores for each tool

        Returns:
            List of TradeOffAnalysis objects
        """
        try:
            logger.info(f"Analyzing trade-offs for {len(tools)} tools")
            trade_offs = []

            for pattern in self.trade_off_patterns:
                trade_off = self._analyze_trade_off_pattern(pattern, tools, metrics)
                if trade_off:
                    trade_offs.append(trade_off)

            # Sort by magnitude (most significant first)
            trade_offs.sort(key=lambda t: t.magnitude, reverse=True)

            logger.info(f"Identified {len(trade_offs)} significant trade-offs")
            return trade_offs

        except Exception as e:
            logger.error(f"Error analyzing trade-offs: {e}")
            return []

    def create_decision_tree(self, tools: List[str], context: TaskContext) -> DecisionTree:
        """
        Create decision tree for tool selection.

        Args:
            tools: List of tool names
            context: Task context for decision criteria

        Returns:
            DecisionTree for systematic tool selection
        """
        try:
            logger.info(f"Creating decision tree for {len(tools)} tools")

            # Determine decision factors based on context
            decision_factors = self._determine_decision_factors(context)

            # Build decision tree
            root_node = self._build_decision_node(tools, decision_factors, context)

            tree = DecisionTree(root_node=root_node, tools_considered=tools, decision_factors=decision_factors)

            logger.info("Decision tree created successfully")
            return tree

        except Exception as e:
            logger.error(f"Error creating decision tree: {e}")
            # Return simple fallback tree
            return self._create_fallback_decision_tree(tools)

    # Helper methods

    def _determine_recommendation_strategy(self, context: Optional[TaskContext]) -> str:
        """Determine which recommendation strategy to use based on context."""
        if not context:
            return "balanced"

        # Analyze context to determine strategy
        if context.user_context and hasattr(context.user_context, "skill_level"):
            if context.user_context.skill_level == "beginner":
                return "usability_first"
            elif context.user_context.skill_level == "expert":
                return "functionality_first"

        if context.quality_requirements and "performance" in str(context.quality_requirements).lower():
            return "performance_first"

        return "balanced"

    def _apply_recommendation_strategy(
        self, comparison_result: "ToolComparisonResult", strategy: str
    ) -> Tuple[str, float]:
        """Apply recommendation strategy to find best tool."""
        rule = self.decision_rules.get(strategy, self.decision_rules["balanced"])
        priority_metrics = rule["priority_metrics"]
        threshold = rule["threshold"]

        # Calculate weighted scores for each tool
        tool_scores = {}

        for tool_id in comparison_result.compared_tools:
            if tool_id in comparison_result.comparison_matrix:
                metrics = comparison_result.comparison_matrix[tool_id]

                # Calculate priority score
                priority_score = statistics.mean(
                    metrics.get(metric, SCORE_LIMITS["default_score"]) for metric in priority_metrics
                )

                # Overall score (priority metrics weighted more heavily)
                all_metrics_score = statistics.mean(metrics.values()) if metrics else SCORE_LIMITS["default_score"]
                weighted_score = (
                    priority_score * TASK_SCORING_WEIGHTS["priority_metrics_weight"]
                    + all_metrics_score * TASK_SCORING_WEIGHTS["all_metrics_weight"]
                )

                tool_scores[tool_id] = weighted_score

        # Filter tools that meet threshold requirement
        qualifying_tools = {tool_id: score for tool_id, score in tool_scores.items() if score >= threshold}

        # Find best tool
        if not qualifying_tools:
            # No tools meet threshold, fall back to best available tool if any
            if tool_scores:
                recommended_tool = max(tool_scores.keys(), key=lambda t: tool_scores[t])
            else:
                return (
                    comparison_result.compared_tools[0] if comparison_result.compared_tools else "unknown",
                    SCORE_LIMITS["default_score"],
                )
        else:
            recommended_tool = max(qualifying_tools.keys(), key=lambda t: qualifying_tools[t])
        confidence = min(
            tool_scores[recommended_tool] * CONFIDENCE_FACTORS["confidence_boost"],
            SCORE_LIMITS["max_score"],
        )

        return recommended_tool, confidence

    def _identify_key_decision_factors(
        self, comparison_result: "ToolComparisonResult", recommended_tool: str
    ) -> List[str]:
        """Identify the key factors that led to the recommendation."""
        if recommended_tool not in comparison_result.comparison_matrix:
            return ["insufficient_data"]

        tool_metrics = comparison_result.comparison_matrix[recommended_tool]

        # Find metrics where this tool excels
        key_factors = []
        for metric, score in tool_metrics.items():
            if score >= DECISION_THRESHOLDS["excellence_threshold"]:  # Tool excels in this metric
                key_factors.append(f"high_{metric}")
            elif score >= SCORE_LIMITS["default_score"]:
                key_factors.append(f"good_{metric}")

        return key_factors[:4]  # Limit to top 4 factors

    def _generate_recommendation_reasons(
        self, comparison_result: "ToolComparisonResult", recommended_tool: str, strategy: str
    ) -> List[str]:
        """Generate human-readable reasons for the recommendation."""
        reasons = []

        if recommended_tool not in comparison_result.comparison_matrix:
            return ["Tool selected based on available information"]

        tool_metrics = comparison_result.comparison_matrix[recommended_tool]
        rule = self.decision_rules.get(strategy, self.decision_rules["balanced"])

        # Generate reasons based on strategy priorities
        for metric in rule["priority_metrics"]:
            score = tool_metrics.get(metric, SCORE_LIMITS["default_score"])
            if score >= DECISION_THRESHOLDS["excellence_threshold"]:
                reason = self._metric_to_reason(metric, score, "high")
                reasons.append(reason)
            elif score >= 0.6:
                reason = self._metric_to_reason(metric, score, "good")
                reasons.append(reason)

        # Add comparative advantages
        comparative_advantage = self._find_comparative_advantages(comparison_result, recommended_tool)
        if comparative_advantage:
            reasons.append(comparative_advantage)

        return reasons[:3]  # Limit to top 3 reasons

    def _generate_reconsideration_scenarios(
        self, comparison_result: "ToolComparisonResult", recommended_tool: str
    ) -> List[str]:
        """Generate scenarios when to reconsider the recommendation."""
        scenarios = []

        if recommended_tool not in comparison_result.comparison_matrix:
            return ["When more information becomes available"]

        tool_metrics = comparison_result.comparison_matrix[recommended_tool]

        # Find weaknesses
        for metric, score in tool_metrics.items():
            if score < 0.4:
                scenario = self._metric_to_reconsideration_scenario(metric)
                scenarios.append(scenario)

        # Add general scenarios
        if not scenarios:
            scenarios.append("When requirements change significantly")

        return scenarios[:3]

    def _find_close_alternatives(self, comparison_result: "ToolComparisonResult", recommended_tool: str) -> List[str]:
        """Find close alternative tools."""
        if recommended_tool not in comparison_result.comparison_matrix:
            return comparison_result.compared_tools[1:3] if len(comparison_result.compared_tools) > 1 else []

        recommended_metrics = comparison_result.comparison_matrix[recommended_tool]
        recommended_score = (
            statistics.mean(recommended_metrics.values()) if recommended_metrics else SCORE_LIMITS["default_score"]
        )

        # Find tools with similar overall scores
        alternatives = []
        for tool_id in comparison_result.compared_tools:
            if tool_id != recommended_tool and tool_id in comparison_result.comparison_matrix:
                tool_metrics = comparison_result.comparison_matrix[tool_id]
                tool_score = statistics.mean(tool_metrics.values()) if tool_metrics else SCORE_LIMITS["default_score"]

                # Consider close if within 0.15 points
                if abs(tool_score - recommended_score) <= 0.15:
                    alternatives.append(tool_id)

        return alternatives[:2]  # Limit to top 2 alternatives

    def _find_fallback_options(self, comparison_result: "ToolComparisonResult", recommended_tool: str) -> List[str]:
        """Find fallback options if recommended tool doesn't work."""
        # Return tools not in close alternatives
        close_alternatives = self._find_close_alternatives(comparison_result, recommended_tool)

        fallbacks = [
            tool
            for tool in comparison_result.compared_tools
            if tool != recommended_tool and tool not in close_alternatives
        ]

        return fallbacks[:2]  # Limit to top 2 fallbacks

    def _generate_context_specific_guidance(
        self, comparison_result: "ToolComparisonResult", recommended_tool: str, context: Optional[TaskContext]
    ) -> Dict[str, str]:
        """Generate guidance for different user contexts."""
        guidance = {
            "beginner": f"Start with '{recommended_tool}' - it offers the best balance of capability and ease of use for beginners",
            "advanced": f"'{recommended_tool}' is recommended, but consider the trade-offs based on your specific requirements",
            "performance": f"For performance-critical scenarios, '{recommended_tool}' provides good efficiency while maintaining functionality",
        }

        # Customize based on actual metrics
        if recommended_tool in comparison_result.comparison_matrix:
            metrics = comparison_result.comparison_matrix[recommended_tool]

            if metrics.get("usability", SCORE_LIMITS["default_score"]) >= 0.8:
                guidance["beginner"] = f"'{recommended_tool}' is excellent for beginners with its high usability score"
            elif metrics.get("usability", SCORE_LIMITS["default_score"]) < SCORE_LIMITS["default_score"]:
                guidance[
                    "beginner"
                ] = f"'{recommended_tool}' may be challenging for beginners - consider getting help initially"

            if metrics.get("functionality", SCORE_LIMITS["default_score"]) >= 0.8:
                guidance["advanced"] = f"'{recommended_tool}' offers rich functionality perfect for advanced use cases"

            if metrics.get("performance", SCORE_LIMITS["default_score"]) >= 0.8:
                guidance["performance"] = f"'{recommended_tool}' excels in performance for time-critical operations"
            elif metrics.get("performance", SCORE_LIMITS["default_score"]) < SCORE_LIMITS["default_score"]:
                guidance["performance"] = f"'{recommended_tool}' may not be ideal for performance-critical scenarios"

        return guidance

    def _analyze_trade_off_pattern(
        self, pattern: Dict[str, Any], tools: List[ToolSearchResult], metrics: Dict[str, Dict[str, float]]
    ) -> Optional[TradeOffAnalysis]:
        """Analyze a specific trade-off pattern."""
        metric_a, metric_b = pattern["metrics"]
        threshold = pattern["threshold"]

        # Find tools that exemplify this trade-off
        trade_off_tools = []
        max_difference = 0.0

        for tool in tools:
            tool_id = str(tool.tool_id)
            if tool_id in metrics:
                tool_metrics = metrics[tool_id]
                score_a = tool_metrics.get(metric_a, SCORE_LIMITS["default_score"])
                score_b = tool_metrics.get(metric_b, SCORE_LIMITS["default_score"])
                difference = abs(score_a - score_b)

                if difference >= threshold:
                    trade_off_tools.append(tool_id)
                    max_difference = max(max_difference, difference)

        if len(trade_off_tools) < 2:
            return None  # Not enough tools show this trade-off

        # Generate trade-off analysis
        return TradeOffAnalysis(
            trade_off_name=pattern["name"],
            tools_involved=trade_off_tools,
            competing_factors=[metric_a, metric_b],
            magnitude=min(max_difference, 1.0),
            decision_importance=self._assess_trade_off_importance(max_difference),
            when_factor_a_matters=self._get_factor_scenarios(metric_a),
            when_factor_b_matters=self._get_factor_scenarios(metric_b),
            recommendation=f"Choose based on whether {metric_a} or {metric_b} is more important for your use case",
        )

    def _assess_trade_off_importance(self, magnitude: float) -> str:
        """Assess the importance of a trade-off based on magnitude."""
        if magnitude >= SCORE_LIMITS["default_score"]:
            return "critical"
        elif magnitude >= 0.3:
            return "important"
        else:
            return "minor"

    def _get_factor_scenarios(self, metric: str) -> List[str]:
        """Get scenarios when a specific metric matters most."""
        scenarios = {
            "functionality": [
                "Complex tasks requiring many features",
                "Advanced workflows with multiple steps",
                "When you need comprehensive capabilities",
            ],
            "usability": ["New users learning the tool", "Time-pressured scenarios", "When ease of use is paramount"],
            "performance": ["Large-scale operations", "Time-critical tasks", "Resource-constrained environments"],
            "reliability": ["Mission-critical operations", "Production environments", "When stability is essential"],
            "compatibility": ["Integration with existing tools", "Multi-tool workflows", "Cross-platform requirements"],
            "documentation": ["Learning new tools", "Training team members", "Complex implementation scenarios"],
        }

        return scenarios.get(metric, [f"When {metric} is a priority"])

    def _determine_decision_factors(self, context: TaskContext) -> List[str]:
        """Determine relevant decision factors based on context."""
        factors = ["usability", "functionality", "reliability"]  # Base factors

        # Add context-specific factors
        if context.quality_requirements:
            if "performance" in str(context.quality_requirements).lower():
                factors.append("performance")
            if "compatibility" in str(context.quality_requirements).lower():
                factors.append("compatibility")

        return factors

    def _build_decision_node(
        self, tools: List[str], factors: List[str], context: TaskContext, depth: int = 0
    ) -> DecisionNode:
        """Build a decision tree node."""
        # Simple decision tree - real implementation would be more sophisticated
        if depth >= 2 or len(tools) <= 1:
            # Leaf node
            recommended = tools[0] if tools else "no_tool"
            return DecisionNode(
                criterion="final_decision",
                threshold=0.0,
                true_branch=None,
                false_branch=None,
                leaf_recommendation=recommended,
                confidence=0.8,
                explanation=f"Recommend {recommended} based on analysis",
            )

        # Internal node
        primary_factor = factors[depth % len(factors)]

        # Split tools based on primary factor (simplified)
        mid = len(tools) // 2
        high_tools = tools[:mid]
        low_tools = tools[mid:]

        return DecisionNode(
            criterion=primary_factor,
            threshold=0.6,
            true_branch=self._build_decision_node(high_tools, factors, context, depth + 1),
            false_branch=self._build_decision_node(low_tools, factors, context, depth + 1),
            leaf_recommendation=None,
            confidence=0.7,
            explanation=f"Split based on {primary_factor} requirements",
        )

    def _metric_to_reason(self, metric: str, score: float, level: str) -> str:
        """Convert metric score to human-readable reason."""
        reasons = {
            "functionality": {
                "high": f"Offers comprehensive features and capabilities ({score:.1%})",
                "good": f"Provides good feature coverage ({score:.1%})",
            },
            "usability": {
                "high": f"Exceptionally easy to use and learn ({score:.1%})",
                "good": f"User-friendly with good documentation ({score:.1%})",
            },
            "reliability": {
                "high": f"Highly reliable and stable ({score:.1%})",
                "good": f"Good reliability for most use cases ({score:.1%})",
            },
            "performance": {
                "high": f"Excellent performance characteristics ({score:.1%})",
                "good": f"Good performance for typical workloads ({score:.1%})",
            },
            "compatibility": {
                "high": f"Excellent integration with other tools ({score:.1%})",
                "good": f"Good compatibility with existing workflows ({score:.1%})",
            },
            "documentation": {
                "high": f"Outstanding documentation and examples ({score:.1%})",
                "good": f"Well-documented with helpful guidance ({score:.1%})",
            },
        }

        return reasons.get(metric, {}).get(level, f"{level.title()} {metric} score ({score:.1%})")

    def _metric_to_reconsideration_scenario(self, metric: str) -> str:
        """Convert low metric score to reconsideration scenario."""
        scenarios = {
            "functionality": "When you need more advanced features",
            "usability": "If ease of use becomes a priority",
            "reliability": "For mission-critical applications",
            "performance": "When speed becomes essential",
            "compatibility": "If integration issues arise",
            "documentation": "When comprehensive guidance is needed",
        }

        return scenarios.get(metric, f"When {metric} becomes more important")

    def _find_comparative_advantages(
        self, comparison_result: "ToolComparisonResult", recommended_tool: str
    ) -> Optional[str]:
        """Find comparative advantages of recommended tool."""
        if recommended_tool not in comparison_result.comparison_matrix:
            return None

        recommended_metrics = comparison_result.comparison_matrix[recommended_tool]

        # Find metric where this tool significantly outperforms others
        for metric, score in recommended_metrics.items():
            if score >= DECISION_THRESHOLDS["excellence_threshold"]:
                # Check if this tool is significantly better than others in this metric
                other_scores = []
                for tool_id, tool_metrics in comparison_result.comparison_matrix.items():
                    if tool_id != recommended_tool:
                        other_scores.append(tool_metrics.get(metric, SCORE_LIMITS["default_score"]))

                if other_scores and score > max(other_scores) + 0.15:
                    return f"Significantly better {metric} than alternatives"

        return None

    def _create_fallback_guidance(self, comparison_result: "ToolComparisonResult") -> SelectionGuidance:
        """Create fallback guidance when analysis fails."""
        recommended_tool = comparison_result.compared_tools[0] if comparison_result.compared_tools else "unknown"

        return SelectionGuidance(
            recommended_tool=recommended_tool,
            confidence=SCORE_LIMITS["default_score"],
            key_factors=["available_information"],
            why_recommended=["Selected based on available data"],
            when_to_reconsider=["When more information becomes available"],
            close_alternatives=comparison_result.compared_tools[1:2]
            if len(comparison_result.compared_tools) > 1
            else [],
            fallback_options=comparison_result.compared_tools[2:3] if len(comparison_result.compared_tools) > 2 else [],
            beginner_guidance=f"Start with {recommended_tool} and learn its capabilities",
            advanced_user_guidance=f"Evaluate {recommended_tool} against your specific requirements",
            performance_critical_guidance=f"Test {recommended_tool} performance in your environment",
        )

    def _create_fallback_decision_tree(self, tools: List[str]) -> DecisionTree:
        """Create fallback decision tree when construction fails."""
        recommended = tools[0] if tools else "no_tool"

        root_node = DecisionNode(
            criterion="default",
            threshold=0.0,
            true_branch=None,
            false_branch=None,
            leaf_recommendation=recommended,
            confidence=SCORE_LIMITS["default_score"],
            explanation="Simple recommendation based on available tools",
        )

        return DecisionTree(root_node=root_node, tools_considered=tools, decision_factors=["availability"])

    def _load_scenario_templates(self) -> Dict[str, Any]:
        """Load scenario-based decision templates."""
        return {
            "beginner_friendly": {
                "priority_metrics": ["usability", "documentation"],
                "weight": 0.8,
                "description": "Optimized for new users",
            },
            "expert_focused": {
                "priority_metrics": ["functionality", "performance"],
                "weight": 0.9,
                "description": "Maximizes capabilities for experienced users",
            },
            "production_ready": {
                "priority_metrics": ["reliability", "performance", "compatibility"],
                "weight": 0.85,
                "description": "Suitable for production environments",
            },
        }
