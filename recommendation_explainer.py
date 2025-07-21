#!/usr/bin/env python3
"""
recommendation_explainer.py: Recommendation Explanation System

This module generates clear explanations for tool recommendations, including
comparisons between alternatives and detailed usage guidance.
"""

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from logging_config import get_logger
from task_analyzer import TaskAnalysis

if TYPE_CHECKING:
    from recommendation_algorithms import ToolRecommendation

logger = get_logger(__name__)


@dataclass
class RecommendationExplanation:
    """Comprehensive explanation for a tool recommendation."""

    primary_reasons: List[str]
    capability_match_explanation: str
    complexity_fit_explanation: str
    parameter_compatibility_explanation: str

    # Guidance
    setup_requirements: List[str]
    usage_best_practices: List[str]
    common_pitfalls: List[str]
    troubleshooting_tips: List[str]

    # Context
    when_this_is_optimal: List[str]
    when_to_consider_alternatives: List[str]
    skill_level_guidance: str

    # Confidence and limitations
    confidence_explanation: str
    known_limitations: List[str]
    uncertainty_areas: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class AlternativeComparison:
    """Comparison between recommended alternatives."""

    primary_tool: "ToolRecommendation"
    alternatives: List["ToolRecommendation"]

    # Analysis
    key_differences: List[str]
    decision_factors: List[Dict[str, Any]]
    recommendation_summary: str

    # Guidance
    when_to_choose_primary: List[str]
    when_to_choose_alternatives: Dict[str, List[str]]
    migration_considerations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class UsageGuidance:
    """Task-specific usage guidance for a tool."""

    tool_name: str

    # Parameter guidance
    parameter_recommendations: List[str]
    configuration_suggestions: List[str]

    # Skill-level specific guidance
    complexity_guidance: str
    step_by_step_instructions: List[str]
    advanced_features: List[str]

    # Best practices
    common_pitfalls: List[str]
    optimization_tips: List[str]
    error_handling_advice: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class ExplanationGenerator:
    """Generate explanations for recommendation components."""

    def __init__(self):
        """Initialize the explanation generator."""
        self.capability_thresholds = {"excellent": 0.85, "good": 0.65, "adequate": 0.45, "limited": 0.25}

        self.complexity_descriptions = {
            "simple": "straightforward and easy to use",
            "moderate": "balanced complexity with good functionality",
            "complex": "advanced tool with extensive capabilities",
        }

    def generate_capability_explanation(
        self, required_capabilities: List[str], tool_capabilities: List[str], match_score: float
    ) -> str:
        """Generate explanation for capability matching."""
        # Find common and missing capabilities
        required_set = set(required_capabilities)
        tool_set = set(tool_capabilities)
        common = required_set.intersection(tool_set)
        missing = required_set - tool_set

        if match_score >= self.capability_thresholds["excellent"]:
            common_caps = ", ".join(list(common)[:3])
            return (
                f"Excellent capability match - tool provides {len(common)}/{len(required_set)} "
                f"required capabilities including {common_caps}."
            )
        elif match_score >= self.capability_thresholds["good"]:
            match_ratio = f"{len(common)}/{len(required_set)}"
            return (
                f"Good capability match - tool covers most required functionality "
                f"with {match_ratio} matching capabilities."
            )
        elif match_score >= self.capability_thresholds["adequate"]:
            missing_str = f", missing {', '.join(list(missing)[:2])}" if missing else ""
            match_ratio = f"{len(common)}/{len(required_set)}"
            return f"Adequate capability match - tool provides {match_ratio} required capabilities{missing_str}."
        else:
            match_ratio = f"{len(common)}/{len(required_set)}"
            missing_features = ", ".join(list(missing)[:3])
            return (
                f"Limited capability match - tool provides only {match_ratio} required "
                f"capabilities. Missing key features: {missing_features}."
            )

    def generate_complexity_explanation(
        self, task_complexity: str, tool_complexity: str, alignment_score: float
    ) -> str:
        """Generate explanation for complexity alignment."""
        task_desc = self.complexity_descriptions.get(task_complexity, task_complexity)
        tool_desc = self.complexity_descriptions.get(tool_complexity, tool_complexity)

        if alignment_score >= 0.9:
            return (
                f"Perfect complexity alignment - both task and tool are {task_complexity}, "
                "ensuring optimal usability."
            )
        elif alignment_score >= 0.7:
            return (
                f"Good complexity match - task is {task_desc} and tool is {tool_desc}, "
                "providing appropriate functionality."
            )
        elif task_complexity == "simple" and tool_complexity in ["moderate", "complex"]:
            return (
                f"Tool complexity exceeds task requirements - tool is {tool_desc} "
                f"for a {task_desc} task, but offers room to grow."
            )
        elif task_complexity in ["moderate", "complex"] and tool_complexity == "simple":
            return f"Tool may be too simple - {tool_desc} tool for a {task_desc} task " "may lack advanced features."
        else:
            return f"Complexity mismatch - task requires {task_complexity} approach " f"but tool is {tool_complexity}."

    def generate_parameter_explanation(self, tool_parameters: Dict[str, Any], task_requirements: Dict[str, Any]) -> str:
        """Generate explanation for parameter compatibility."""
        if not tool_parameters:
            return "Tool has minimal configuration requirements, making it easy to use."

        param_count = len(tool_parameters)
        required_params = sum(1 for p in tool_parameters.values() if p.get("required", False))

        if required_params == 0:
            return (
                f"Excellent parameter compatibility - tool has {param_count} optional "
                f"parameters with sensible defaults."
            )
        elif required_params <= 2:
            return (
                f"Good parameter compatibility - tool requires only {required_params} "
                f"parameters with {param_count - required_params} optional configurations."
            )
        else:
            return (
                f"Moderate parameter complexity - tool requires {required_params} "
                f"parameters and offers {param_count - required_params} optional settings."
            )

    def generate_confidence_explanation(
        self, recommendation_score: float, task_confidence: float, alignment_factors: Dict[str, float]
    ) -> str:
        """Generate explanation for recommendation confidence."""
        if recommendation_score >= 0.8 and task_confidence >= 0.8:
            return "High confidence - strong task-tool alignment with clear requirements match."
        elif recommendation_score >= 0.6:
            return "Medium confidence - good overall fit with some areas for improvement."
        else:
            weak_areas = [k for k, v in alignment_factors.items() if v < 0.5]
            weak_str = f" Areas of concern: {', '.join(weak_areas)}" if weak_areas else ""
            return f"Lower confidence - recommendation has limitations.{weak_str}"

    def _get_match_quality(self, score: float) -> str:
        """Get match quality description from score."""
        if score >= self.capability_thresholds["excellent"]:
            return "excellent"
        elif score >= self.capability_thresholds["good"]:
            return "good"
        elif score >= self.capability_thresholds["adequate"]:
            return "adequate"
        else:
            return "limited"


class ComparisonAnalyzer:
    """Analyze differences between tool alternatives."""

    def __init__(self):
        """Initialize the comparison analyzer."""
        self.comparison_factors = ["capability", "complexity", "usability", "performance"]

    def analyze_capability_differences(
        self, primary_capabilities: List[str], alternative_capabilities: List[str]
    ) -> Dict[str, List[str]]:
        """Analyze capability differences between tools."""
        primary_set = set(primary_capabilities)
        alt_set = set(alternative_capabilities)

        return {
            "common": list(primary_set.intersection(alt_set)),
            "primary_only": list(primary_set - alt_set),
            "alternative_only": list(alt_set - primary_set),
        }

    def generate_trade_off_analysis(
        self, primary_scores: Dict[str, float], alternative_scores: Dict[str, float]
    ) -> List[str]:
        """Generate trade-off analysis between tools."""
        trade_offs = []

        for factor in self.comparison_factors:
            if factor in primary_scores and factor in alternative_scores:
                primary_score = primary_scores[factor]
                alt_score = alternative_scores[factor]

                if primary_score > alt_score + 0.2:
                    trade_offs.append(f"Primary tool excels in {factor} ({primary_score:.1f} vs {alt_score:.1f})")
                elif alt_score > primary_score + 0.2:
                    trade_offs.append(f"Alternative excels in {factor} ({alt_score:.1f} vs {primary_score:.1f})")

        return trade_offs

    def identify_decision_factors(
        self, primary_rec: "ToolRecommendation", alternative_rec: "ToolRecommendation"
    ) -> List[Dict[str, Any]]:
        """Identify key factors for decision making between tools."""
        factors = []

        # Capability factor
        if primary_rec.capability_match > alternative_rec.capability_match + 0.1:
            factors.append(
                {
                    "factor": "capability_coverage",
                    "advantage": "primary",
                    "description": "Primary tool covers more required capabilities",
                    "trade_off": "May be more complex than needed",
                }
            )

        # Complexity factor
        if alternative_rec.complexity_alignment > primary_rec.complexity_alignment + 0.1:
            factors.append(
                {
                    "factor": "complexity_alignment",
                    "advantage": "alternative",
                    "description": "Alternative tool has better complexity match",
                    "trade_off": "May lack advanced features",
                }
            )

        # Score differential
        score_diff = abs(primary_rec.recommendation_score - alternative_rec.recommendation_score)
        if score_diff < 0.1:
            factors.append(
                {
                    "factor": "overall_suitability",
                    "advantage": "similar",
                    "description": "Both tools are similarly suitable",
                    "trade_off": "Choice depends on specific preferences",
                }
            )

        return factors


class GuidanceGenerator:
    """Generate usage guidance and recommendations."""

    def __init__(self):
        """Initialize the guidance generator."""
        self.skill_levels = ["beginner", "intermediate", "advanced"]

    def generate_parameter_recommendations(self, tool_parameters: Dict[str, Any]) -> List[str]:
        """Generate parameter configuration recommendations."""
        recommendations = []

        for param_name, param_info in tool_parameters.items():
            if param_info.get("required", False):
                recommendations.append(
                    f"Required parameter '{param_name}': {param_info.get('description', 'No description')}"
                )
            elif "default" in param_info:
                recommendations.append(
                    f"Parameter '{param_name}': Keep default '{param_info['default']}' for most use cases"
                )

        if not recommendations:
            recommendations.append("Tool uses sensible defaults - no configuration needed for basic usage")

        return recommendations

    def generate_step_by_step_instructions(self, tool_name: str, user_skill: str) -> List[str]:
        """Generate step-by-step usage instructions."""
        if user_skill == "beginner":
            return [
                f"1. Initialize {tool_name} with default settings",
                "2. Prepare your input data in the required format",
                f"3. Run {tool_name} with minimal parameters first",
                "4. Verify the output meets your requirements",
                "5. Gradually add additional parameters as needed",
            ]
        elif user_skill == "intermediate":
            return [
                f"1. Configure {tool_name} with task-specific parameters",
                "2. Set up error handling and validation",
                "3. Run tool and monitor performance",
                "4. Optimize parameters based on results",
            ]
        else:  # advanced
            return [
                f"1. Configure {tool_name} for optimal performance",
                "2. Implement custom error handling and logging",
                "3. Integrate with existing workflow automation",
            ]

    def generate_optimization_tips(self, tool_metadata: Dict[str, Any], user_skill: str) -> List[str]:
        """Generate optimization tips for tool usage."""
        tips = []

        if user_skill == "advanced":
            if "performance_features" in tool_metadata:
                tips.extend(
                    [
                        f"Enable {feature} for better performance"
                        for feature in tool_metadata["performance_features"][:2]
                    ]
                )

            if "advanced_parameters" in tool_metadata:
                tips.append("Tune advanced parameters for your specific use case")

            tips.append("Monitor resource usage and adjust batch sizes accordingly")
        else:
            tips.extend(
                [
                    "Start with default settings and adjust based on results",
                    "Use built-in validation features to ensure data quality",
                ]
            )

        if not tips:
            tips.append("Use recommended default settings for optimal performance")

        return tips

    def generate_common_pitfalls(self, tool_complexity: str, user_skill: str) -> List[str]:
        """Generate common pitfalls to avoid."""
        pitfalls = []

        if tool_complexity == "complex" and user_skill == "beginner":
            pitfalls.extend(
                [
                    "Don't modify advanced parameters without understanding their impact",
                    "Start with simple use cases before attempting complex workflows",
                    "Read documentation thoroughly before using advanced features",
                ]
            )
        elif tool_complexity == "simple" and user_skill == "advanced":
            pitfalls.append("Tool may not support all advanced features you might expect")

        pitfalls.extend(
            ["Always validate input data format before processing", "Keep backups when performing data transformations"]
        )

        return pitfalls


class RecommendationExplainer:
    """Generate explanations for tool recommendations."""

    def __init__(self):
        """Initialize the recommendation explainer."""
        self.explanation_generator = ExplanationGenerator()
        self.comparison_analyzer = ComparisonAnalyzer()
        self.guidance_generator = GuidanceGenerator()

        logger.info("Recommendation explainer initialized")

    def explain_recommendation(
        self, recommendation: "ToolRecommendation", task_analysis: TaskAnalysis
    ) -> RecommendationExplanation:
        """Generate comprehensive recommendation explanation."""
        logger.debug(f"Generating explanation for {recommendation.tool.name}")

        # Extract tool metadata
        tool_metadata = getattr(recommendation.tool, "metadata", {})
        tool_capabilities = tool_metadata.get("capabilities", [])
        tool_complexity = tool_metadata.get("complexity", "moderate")
        tool_parameters = tool_metadata.get("parameters", {})

        # Generate component explanations
        capability_explanation = self.explanation_generator.generate_capability_explanation(
            task_analysis.required_capabilities, tool_capabilities, recommendation.capability_match
        )

        complexity_explanation = self.explanation_generator.generate_complexity_explanation(
            task_analysis.complexity_level, tool_complexity, recommendation.complexity_alignment
        )

        parameter_explanation = self.explanation_generator.generate_parameter_explanation(
            tool_parameters, {}  # Task parameter requirements (would be extracted from task analysis)
        )

        confidence_explanation = self.explanation_generator.generate_confidence_explanation(
            recommendation.recommendation_score,
            task_analysis.confidence,
            {
                "capability": recommendation.capability_match,
                "complexity": recommendation.complexity_alignment,
                "parameters": recommendation.parameter_compatibility,
            },
        )

        # Generate guidance
        setup_requirements = self._generate_setup_requirements(tool_metadata)
        best_practices = self._generate_best_practices(tool_complexity, task_analysis.skill_level_required)
        pitfalls = self.guidance_generator.generate_common_pitfalls(tool_complexity, task_analysis.skill_level_required)

        # Create explanation
        explanation = RecommendationExplanation(
            primary_reasons=recommendation.recommendation_reasons,
            capability_match_explanation=capability_explanation,
            complexity_fit_explanation=complexity_explanation,
            parameter_compatibility_explanation=parameter_explanation,
            setup_requirements=setup_requirements,
            usage_best_practices=best_practices,
            common_pitfalls=pitfalls,
            troubleshooting_tips=self._generate_troubleshooting_tips(tool_metadata),
            when_this_is_optimal=self._generate_optimal_scenarios(recommendation, task_analysis),
            when_to_consider_alternatives=self._generate_alternative_scenarios(recommendation),
            skill_level_guidance=self._generate_skill_guidance(tool_complexity, task_analysis.skill_level_required),
            confidence_explanation=confidence_explanation,
            known_limitations=self._extract_tool_limitations(tool_metadata),
            uncertainty_areas=self._identify_uncertainty_areas(recommendation, task_analysis),
        )

        logger.debug("Recommendation explanation generated successfully")
        return explanation

    def compare_alternatives(
        self, primary: "ToolRecommendation", alternatives: List["ToolRecommendation"]
    ) -> AlternativeComparison:
        """Generate comparison between recommended alternatives."""
        logger.debug(f"Comparing {primary.tool.name} with {len(alternatives)} alternatives")

        # Analyze differences with each alternative
        key_differences = []
        decision_factors = []

        for alt in alternatives:
            # Capability differences
            primary_caps = getattr(primary.tool, "metadata", {}).get("capabilities", [])
            alt_caps = getattr(alt.tool, "metadata", {}).get("capabilities", [])
            cap_diff = self.comparison_analyzer.analyze_capability_differences(primary_caps, alt_caps)

            if cap_diff["primary_only"]:
                key_differences.append(
                    f"{primary.tool.name} offers additional: {', '.join(cap_diff['primary_only'][:2])}"
                )

            # Decision factors
            factors = self.comparison_analyzer.identify_decision_factors(primary, alt)
            decision_factors.extend(factors)

        # Generate when-to-choose guidance
        when_primary = [
            "When you need the highest capability coverage",
            "For complex tasks requiring advanced features",
        ]

        when_alternatives = {}
        for alt in alternatives:
            when_alternatives[alt.tool.name] = [
                "When simplicity is preferred over features",
                "For basic tasks not requiring advanced capabilities",
            ]

        comparison = AlternativeComparison(
            primary_tool=primary,
            alternatives=alternatives,
            key_differences=key_differences,
            decision_factors=decision_factors,
            recommendation_summary=self._generate_comparison_summary(primary, alternatives),
            when_to_choose_primary=when_primary,
            when_to_choose_alternatives=when_alternatives,
            migration_considerations=["Consider data format compatibility", "Evaluate learning curve"],
        )

        logger.debug("Alternative comparison generated successfully")
        return comparison

    def generate_usage_guidance(self, tool: "ToolRecommendation", context: Optional[Dict[str, Any]]) -> UsageGuidance:
        """Generate task-specific usage guidance."""
        logger.debug(f"Generating usage guidance for {tool.name}")

        tool_metadata = getattr(tool, "metadata", {})
        tool_parameters = tool_metadata.get("parameters", {})
        user_skill = getattr(context, "user_skill_level", "intermediate")

        guidance = UsageGuidance(
            tool_name=tool.name,
            parameter_recommendations=self.guidance_generator.generate_parameter_recommendations(tool_parameters),
            configuration_suggestions=self._generate_configuration_suggestions(tool_metadata, user_skill),
            complexity_guidance=self._generate_complexity_guidance(tool_metadata, user_skill),
            step_by_step_instructions=self.guidance_generator.generate_step_by_step_instructions(tool.name, user_skill),
            advanced_features=self._extract_advanced_features(tool_metadata, user_skill),
            common_pitfalls=self.guidance_generator.generate_common_pitfalls(
                tool_metadata.get("complexity", "moderate"), user_skill
            ),
            optimization_tips=self.guidance_generator.generate_optimization_tips(tool_metadata, user_skill),
            error_handling_advice=self._generate_error_handling_advice(tool_metadata),
        )

        logger.debug("Usage guidance generated successfully")
        return guidance

    # Helper methods
    def _generate_setup_requirements(self, tool_metadata: Dict) -> List[str]:
        """Generate setup requirements for the tool."""
        requirements = ["Install tool dependencies"]

        if "input_types" in tool_metadata:
            requirements.append(f"Prepare data in supported format: {', '.join(tool_metadata['input_types'][:2])}")

        return requirements

    def _generate_best_practices(self, tool_complexity: str, user_skill: str) -> List[str]:
        """Generate usage best practices."""
        practices = ["Test with sample data first", "Validate results before proceeding"]

        if tool_complexity == "complex" and user_skill != "advanced":
            practices.append("Start with default parameters and adjust gradually")

        return practices

    def _generate_troubleshooting_tips(self, tool_metadata: Dict) -> List[str]:
        """Generate troubleshooting tips."""
        return [
            "Check input data format if errors occur",
            "Verify all required parameters are provided",
            "Review tool documentation for parameter details",
        ]

    def _generate_optimal_scenarios(
        self, recommendation: "ToolRecommendation", task_analysis: TaskAnalysis
    ) -> List[str]:
        """Generate scenarios when this tool is optimal."""
        return [
            f"When task requires {', '.join(task_analysis.required_capabilities[:2])}",
            f"For {task_analysis.complexity_level} complexity tasks",
        ]

    def _generate_alternative_scenarios(self, recommendation: "ToolRecommendation") -> List[str]:
        """Generate scenarios when to consider alternatives."""
        return ["When simpler approach is sufficient", "If tool complexity exceeds requirements"]

    def _generate_skill_guidance(self, tool_complexity: str, user_skill: str) -> str:
        """Generate skill-level specific guidance."""
        if user_skill == "beginner" and tool_complexity == "complex":
            return "This tool is advanced - consider starting with simpler alternatives or investing time in learning."
        elif user_skill == "advanced" and tool_complexity == "simple":
            return "This tool is straightforward - you may want more advanced features for complex scenarios."
        else:
            return "Tool complexity aligns well with your skill level."

    def _extract_tool_limitations(self, tool_metadata: Dict) -> List[str]:
        """Extract known tool limitations."""
        return ["Performance may vary with large datasets"]

    def _identify_uncertainty_areas(
        self, recommendation: "ToolRecommendation", task_analysis: TaskAnalysis
    ) -> List[str]:
        """Identify areas of uncertainty in the recommendation."""
        uncertainties = []

        if recommendation.parameter_compatibility < 0.7:
            uncertainties.append("Parameter compatibility needs verification")

        if task_analysis.confidence < 0.8:
            uncertainties.append("Task requirements have some ambiguity")

        return uncertainties

    def _generate_comparison_summary(
        self, primary: "ToolRecommendation", alternatives: List["ToolRecommendation"]
    ) -> str:
        """Generate summary for alternative comparison."""
        return (
            f"{primary.tool.name} recommended as primary choice, with {len(alternatives)} "
            f"viable alternatives for different scenarios."
        )

    def _generate_configuration_suggestions(self, tool_metadata: Dict, user_skill: str) -> List[str]:
        """Generate configuration suggestions."""
        return ["Use recommended default settings initially"]

    def _generate_complexity_guidance(self, tool_metadata: Dict, user_skill: str) -> str:
        """Generate complexity-specific guidance."""
        if user_skill == "beginner":
            return "Start with basic features and gradually explore advanced options."
        else:
            return "Tool complexity is appropriate for your skill level."

    def _extract_advanced_features(self, tool_metadata: Dict, user_skill: str) -> List[str]:
        """Extract advanced features for experienced users."""
        if user_skill == "advanced":
            return tool_metadata.get("advanced_parameters", ["Custom parameter tuning available"])
        return []

    def _generate_error_handling_advice(self, tool_metadata: Dict) -> List[str]:
        """Generate error handling advice."""
        return ["Enable verbose logging for debugging", "Implement proper exception handling around tool calls"]
