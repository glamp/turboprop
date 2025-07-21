#!/usr/bin/env python3
"""
comparison_metrics.py: Multi-dimensional metrics for tool comparison.

This module defines and calculates comprehensive metrics for comparing tools
across multiple dimensions including functionality, usability, reliability,
performance, compatibility, and documentation quality.
"""

import re
from typing import Dict, List, Optional

from comparison_constants import (
    DOCUMENTATION_SCORING,
    FUNCTIONALITY_CONFIG,
    METRIC_WEIGHTS,
    PARAMETER_COMPLEXITY_PENALTIES,
    PERFORMANCE_CONFIG,
    RELIABILITY_CONFIG,
    SCORE_LIMITS,
    USABILITY_CONFIG,
)
from context_analyzer import TaskContext
from logging_config import get_logger
from tool_search_results import ToolSearchResult

logger = get_logger(__name__)

# Metric definitions with weights and calculation parameters
COMPARISON_METRICS = {
    "functionality": {
        "weight": METRIC_WEIGHTS["functionality"],
        "description": "Feature richness and capability breadth",
        "factors": ["parameter_count", "capability_scope", "feature_completeness"],
        "normalization_max": FUNCTIONALITY_CONFIG["max_parameter_count_normalization"],
    },
    "usability": {
        "weight": METRIC_WEIGHTS["usability"],
        "description": "Ease of use and learning curve",
        "factors": ["parameter_simplicity", "documentation_quality", "required_param_ratio"],
        "normalization_max": 1.0,
    },
    "reliability": {
        "weight": METRIC_WEIGHTS["usability"],
        "description": "Stability and error handling",
        "factors": ["error_handling_indicators", "validation_features", "maturity_indicators"],
        "normalization_max": 1.0,
    },
    "performance": {
        "weight": METRIC_WEIGHTS["performance"],
        "description": "Speed and resource efficiency",
        "factors": ["estimated_complexity", "resource_efficiency", "scalability_indicators"],
        "normalization_max": 1.0,
    },
    "compatibility": {
        "weight": METRIC_WEIGHTS["compatibility"],
        "description": "Integration and workflow compatibility",
        "factors": ["input_output_compatibility", "tool_chain_support", "ecosystem_fit"],
        "normalization_max": 1.0,
    },
    "documentation": {
        "weight": METRIC_WEIGHTS["documentation"],
        "description": "Documentation quality and examples",
        "factors": ["description_quality", "parameter_documentation", "example_availability"],
        "normalization_max": 1.0,
    },
}

# Scoring configuration
SCORING_CONFIG = {
    "min_score": SCORE_LIMITS["min_score"],
    "max_score": SCORE_LIMITS["max_score"],
    "default_score": SCORE_LIMITS["default_score"],
    "confidence_threshold": SCORE_LIMITS["confidence_threshold"],
    "parameter_complexity_penalties": PARAMETER_COMPLEXITY_PENALTIES,
    "description_quality_weights": {
        "length_bonus_threshold": 50,
        "technical_term_bonus": DOCUMENTATION_SCORING["technical_term_bonus"],
        "example_mention_bonus": DOCUMENTATION_SCORING["example_mention_bonus"],
        "empty_penalty": DOCUMENTATION_SCORING["empty_penalty"],
    },
}


class ComparisonMetrics:
    """Multi-dimensional metrics calculation system for tool comparison."""

    def __init__(self):
        """Initialize the comparison metrics system."""
        self.metric_definitions = COMPARISON_METRICS.copy()
        self.scoring_config = SCORING_CONFIG.copy()
        self._parameter_type_complexity_cache = {}
        self._description_quality_cache = {}

        logger.info("Comparison metrics system initialized")

    def calculate_all_metrics(
        self, tools: List[ToolSearchResult], context: Optional[TaskContext] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate all comparison metrics for a set of tools.

        Args:
            tools: List of ToolSearchResult objects to analyze
            context: Optional task context for context-aware scoring

        Returns:
            Dictionary mapping tool_id -> metric_name -> score
        """
        if not self._validate_tools_input(tools):
            return {}

        logger.info(f"Calculating metrics for {len(tools)} tools")

        results = {}
        for tool in tools:
            tool_metrics = self._calculate_tool_metrics_safely(tool, context)
            results[str(tool.tool_id)] = tool_metrics

        logger.info(f"Metrics calculation completed for {len(results)} tools")
        return results

    def _validate_tools_input(self, tools: List[ToolSearchResult]) -> bool:
        """Validate input tools list."""
        if not tools:
            logger.warning("No tools provided for metrics calculation")
            return False
        return True

    def _calculate_tool_metrics_safely(
        self, tool: ToolSearchResult, context: Optional[TaskContext]
    ) -> Dict[str, float]:
        """Calculate metrics for a single tool with error handling."""
        try:
            tool_metrics = self._calculate_individual_tool_metrics(tool, context)
            tool_metrics = self._normalize_all_scores(tool_metrics)

            logger.debug(f"Calculated metrics for {tool.name}: {tool_metrics}")
            return tool_metrics

        except Exception as e:
            logger.error(f"Error calculating metrics for tool {tool.tool_id}: {e}")
            return self._get_default_metrics()

    def _calculate_individual_tool_metrics(
        self, tool: ToolSearchResult, context: Optional[TaskContext]
    ) -> Dict[str, float]:
        """Calculate individual metrics for a tool."""
        return {
            "functionality": self.calculate_functionality_score(tool),
            "usability": self.calculate_usability_score(tool),
            "reliability": self.calculate_reliability_score(tool),
            "performance": self.calculate_performance_score(tool),
            "compatibility": self.calculate_compatibility_score(tool, context),
            "documentation": self.calculate_documentation_score(tool),
        }

    def _normalize_all_scores(self, tool_metrics: Dict[str, float]) -> Dict[str, float]:
        """Normalize all scores in the metrics dictionary."""
        return {k: self._normalize_score(v) for k, v in tool_metrics.items()}

    def _get_default_metrics(self) -> Dict[str, float]:
        """Get default metrics when calculation fails."""
        return {metric: self.scoring_config["default_score"] for metric in self.metric_definitions.keys()}

    def calculate_functionality_score(self, tool: ToolSearchResult) -> float:
        """
        Calculate functionality richness score based on parameters and capabilities.

        Args:
            tool: ToolSearchResult to analyze

        Returns:
            Functionality score (0.0-1.0)
        """
        try:
            # Parameter count factor (normalized)
            param_count_score = (
                min(tool.parameter_count / FUNCTIONALITY_CONFIG["max_parameter_count_normalization"], 1.0)
                if tool.parameter_count
                else FUNCTIONALITY_CONFIG["default_param_score"]
            )

            # Parameter type complexity
            complexity_score = self._calculate_parameter_complexity_score(tool)

            # Description richness (indicates capability scope)
            description_richness = self._calculate_description_richness(tool.description)

            # Category influence (some categories inherently more functional)
            category_bonus = self._get_category_functionality_bonus(tool.category)

            # Combine factors
            raw_score = (
                param_count_score * FUNCTIONALITY_CONFIG["param_count_weight"]
                + complexity_score * FUNCTIONALITY_CONFIG["complexity_weight"]
                + description_richness * FUNCTIONALITY_CONFIG["description_weight"]
                + category_bonus * FUNCTIONALITY_CONFIG["category_weight"]
            )

            return self._normalize_score(raw_score)

        except Exception as e:
            logger.warning(f"Error calculating functionality score for {tool.tool_id}: {e}")
            return self.scoring_config["default_score"]

    def calculate_usability_score(self, tool: ToolSearchResult) -> float:
        """
        Calculate usability and ease-of-use score.

        Args:
            tool: ToolSearchResult to analyze

        Returns:
            Usability score (0.0-1.0)
        """
        try:
            # Required parameter ratio (fewer required = more usable)
            req_ratio = (tool.required_parameter_count / max(tool.parameter_count, 1)) if tool.parameter_count else 0
            required_param_score = 1.0 - (req_ratio * USABILITY_CONFIG["required_param_penalty_factor"])

            # Parameter complexity penalty
            complexity_penalty = self._calculate_parameter_complexity_penalty(tool)

            # Description clarity (clear descriptions = more usable)
            description_clarity = self._calculate_description_clarity(tool.description)

            # Parameter documentation quality
            param_doc_quality = self._calculate_parameter_documentation_quality(tool)

            # Category usability bias
            category_usability = self._get_category_usability_bias(tool.category)

            # Combine factors
            raw_score = (
                required_param_score * 0.3
                + (1.0 - complexity_penalty) * 0.25
                + description_clarity * 0.25
                + param_doc_quality * 0.15
                + category_usability * 0.05
            )

            return self._normalize_score(raw_score)

        except Exception as e:
            logger.warning(f"Error calculating usability score for {tool.tool_id}: {e}")
            return self.scoring_config["default_score"]

    def calculate_reliability_score(self, tool: ToolSearchResult) -> float:
        """
        Calculate reliability and stability score.

        Args:
            tool: ToolSearchResult to analyze

        Returns:
            Reliability score (0.0-1.0)
        """
        try:
            # Error handling indicators in description
            error_handling_score = self._detect_error_handling_features(tool.description)

            # Validation parameter presence
            validation_score = self._detect_validation_features(tool)

            # Maturity indicators (established tool names, common patterns)
            maturity_score = self._assess_tool_maturity(tool)

            # Parameter safety (required params for critical operations)
            safety_score = self._assess_parameter_safety(tool)

            # Category reliability baseline
            category_reliability = self._get_category_reliability_baseline(tool.category)

            # Combine factors
            raw_score = (
                error_handling_score * 0.25
                + validation_score * 0.25
                + maturity_score * 0.25
                + safety_score * 0.15
                + category_reliability * 0.10
            )

            return self._normalize_score(raw_score)

        except Exception as e:
            logger.warning(f"Error calculating reliability score for {tool.tool_id}: {e}")
            return self.scoring_config["default_score"]

    def calculate_performance_score(self, tool: ToolSearchResult) -> float:
        """
        Calculate estimated performance score.

        Args:
            tool: ToolSearchResult to analyze

        Returns:
            Performance score (0.0-1.0)
        """
        try:
            # Complexity estimation (simpler operations typically faster)
            complexity_factor = 1.0 - min(tool.parameter_count / 15.0, 0.8) if tool.parameter_count else 0.8

            # Operation type performance characteristics
            operation_performance = self._estimate_operation_performance(tool)

            # Category performance baseline
            category_performance = self._get_category_performance_baseline(tool.category)

            # Resource intensity indicators
            resource_efficiency = self._assess_resource_efficiency(tool)

            # Combine factors
            raw_score = (
                complexity_factor * 0.3
                + operation_performance * 0.3
                + category_performance * 0.2
                + resource_efficiency * 0.2
            )

            return self._normalize_score(raw_score)

        except Exception as e:
            logger.warning(f"Error calculating performance score for {tool.tool_id}: {e}")
            return self.scoring_config["default_score"]

    def calculate_compatibility_score(self, tool: ToolSearchResult, context: Optional[TaskContext] = None) -> float:
        """
        Calculate integration and workflow compatibility score.

        Args:
            tool: ToolSearchResult to analyze
            context: Optional task context for compatibility assessment

        Returns:
            Compatibility score (0.0-1.0)
        """
        try:
            # Standard I/O compatibility
            io_compatibility = self._assess_io_compatibility(tool)

            # Parameter type compatibility
            param_compatibility = self._assess_parameter_type_compatibility(tool)

            # Ecosystem integration
            ecosystem_fit = self._assess_ecosystem_integration(tool, context)

            # Error handling compatibility
            error_compat = self._assess_error_handling_compatibility(tool)

            # Combine factors
            raw_score = io_compatibility * 0.3 + param_compatibility * 0.3 + ecosystem_fit * 0.25 + error_compat * 0.15

            return self._normalize_score(raw_score)

        except Exception as e:
            logger.warning(f"Error calculating compatibility score for {tool.tool_id}: {e}")
            return self.scoring_config["default_score"]

    def calculate_documentation_score(self, tool: ToolSearchResult) -> float:
        """
        Calculate documentation quality score.

        Args:
            tool: ToolSearchResult to analyze

        Returns:
            Documentation score (0.0-1.0)
        """
        try:
            # Description quality
            desc_quality = self._calculate_description_quality(tool.description)

            # Parameter documentation completeness
            param_doc_quality = self._calculate_parameter_documentation_quality(tool)

            # Example availability (inferred from description)
            example_indicators = self._detect_example_indicators(tool.description)

            # Combine factors
            raw_score = desc_quality * 0.5 + param_doc_quality * 0.3 + example_indicators * 0.2

            return self._normalize_score(raw_score)

        except Exception as e:
            logger.warning(f"Error calculating documentation score for {tool.tool_id}: {e}")
            return self.scoring_config["default_score"]

    # Helper methods for metric calculations

    def _calculate_parameter_complexity_score(self, tool: ToolSearchResult) -> float:
        """Calculate complexity score based on parameter types and structure."""
        if not tool.parameters:
            return 0.2  # Low complexity for no parameters

        complexity_sum = 0.0
        for param in tool.parameters:
            param_type = param.type.lower() if param.type else "string"

            # Base complexity by type
            if param_type in ["string", "number", "boolean"]:
                complexity = 0.3
            elif param_type in ["object", "array"]:
                complexity = PERFORMANCE_CONFIG["high_complexity_threshold"]
            elif param_type == "any":
                complexity = 0.8
            else:
                complexity = 0.5

            # Required parameters are often more complex to use
            if param.required:
                complexity *= 1.2

            complexity_sum += complexity

        # Normalize by parameter count
        avg_complexity = complexity_sum / len(tool.parameters)
        return min(avg_complexity, 1.0)

    def _calculate_parameter_complexity_penalty(self, tool: ToolSearchResult) -> float:
        """Calculate usability penalty based on parameter complexity."""
        if not tool.parameters:
            return 0.0

        penalty = 0.0

        for param in tool.parameters:
            param_type = param.type.lower() if param.type else "string"

            # Apply penalties from config
            for complex_type, type_penalty in self.scoring_config["parameter_complexity_penalties"].items():
                if complex_type in param_type:
                    penalty += type_penalty

            # Empty descriptions are harder to use
            if not param.description or len(param.description.strip()) < 10:
                penalty += 0.1

        # Normalize penalty
        return min(penalty / max(len(tool.parameters), 1), 1.0)

    def _calculate_description_richness(self, description: str) -> float:
        """Calculate richness of tool description for functionality assessment."""
        if not description:
            return 0.1

        # Length factor
        length_score = min(len(description) / 200.0, 1.0)

        # Technical term density
        technical_terms = len(
            re.findall(
                r"\b(file|data|process|execute|create|update|delete|search|query|analyze)\b", description.lower()
            )
        )
        technical_score = min(technical_terms / 5.0, 1.0)

        # Action word presence
        action_words = len(re.findall(r"\b(can|will|allows|enables|provides|supports|performs)\b", description.lower()))
        action_score = min(action_words / 3.0, 1.0)

        return length_score * 0.4 + technical_score * 0.4 + action_score * 0.2

    def _calculate_description_clarity(self, description: str) -> float:
        """Calculate clarity of description for usability assessment."""
        if not description:
            return 0.1

        # Sentence structure (shorter sentences generally clearer)
        sentences = description.split(".")
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        clarity_score = max(0.2, 1.0 - (avg_sentence_length / 30.0))

        # Jargon density (less jargon = more clear)
        jargon_indicators = len(re.findall(r"\b[A-Z]{2,}\b", description))  # Acronyms
        jargon_penalty = min(
            jargon_indicators / USABILITY_CONFIG["jargon_normalization_factor"],
            USABILITY_CONFIG["jargon_penalty_max"],
        )

        # Explanation words (indicate clarity)
        explanation_words = len(re.findall(r"\b(that|which|where|when|how|why|example|such as)\b", description.lower()))
        explanation_bonus = min(explanation_words / 5.0, 0.3)

        return max(0.1, clarity_score - jargon_penalty + explanation_bonus)

    def _calculate_parameter_documentation_quality(self, tool: ToolSearchResult) -> float:
        """Calculate quality of parameter documentation."""
        if not tool.parameters:
            return 1.0  # Perfect score for no parameters

        documented_params = sum(1 for p in tool.parameters if p.description and len(p.description.strip()) > 5)
        documentation_ratio = documented_params / len(tool.parameters)

        # Quality of existing documentation
        quality_sum = 0.0
        for param in tool.parameters:
            if param.description:
                # Length factor
                length_factor = min(len(param.description) / 50.0, 1.0)

                # Contains examples or specifics
                specificity = (
                    1.0
                    if any(word in param.description.lower() for word in ["example", "e.g.", "such as", "format"])
                    else 0.5
                )

                quality_sum += length_factor * 0.6 + specificity * 0.4

        avg_quality = quality_sum / len(tool.parameters) if tool.parameters else 0.0

        return (
            documentation_ratio * USABILITY_CONFIG["documentation_quality_weight"]
            + avg_quality * USABILITY_CONFIG["avg_quality_weight"]
        )

    def _detect_error_handling_features(self, description: str) -> float:
        """Detect error handling mentions in description."""
        if not description:
            return 0.3  # Default moderate score

        error_keywords = ["error", "exception", "fail", "invalid", "validation", "check", "verify", "handle"]
        mentions = sum(1 for keyword in error_keywords if keyword in description.lower())

        return min(mentions / 3.0, 1.0)

    def _detect_validation_features(self, tool: ToolSearchResult) -> float:
        """Detect validation features in parameters."""
        if not tool.parameters:
            return 0.5

        validation_indicators = 0
        for param in tool.parameters:
            if param.description and any(
                word in param.description.lower() for word in ["valid", "format", "pattern", "required", "optional"]
            ):
                validation_indicators += 1

        return min(validation_indicators / max(len(tool.parameters), 1), 1.0)

    def _assess_tool_maturity(self, tool: ToolSearchResult) -> float:
        """Assess tool maturity based on naming and description patterns."""
        # Well-known tool patterns suggest maturity
        mature_patterns = ["read", "write", "edit", "search", "find", "copy", "move", "delete"]
        tool_name_lower = tool.name.lower()

        maturity_score = 0.5  # Base score

        # Check for established patterns
        if any(pattern in tool_name_lower for pattern in mature_patterns):
            maturity_score += 0.3

        # Description mentions stability/reliability
        if tool.description and any(
            word in tool.description.lower() for word in ["stable", "reliable", "tested", "proven"]
        ):
            maturity_score += 0.2

        return min(maturity_score, 1.0)

    def _assess_parameter_safety(self, tool: ToolSearchResult) -> float:
        """Assess parameter safety design."""
        if not tool.parameters:
            return 0.8  # Safe by default

        safety_score = RELIABILITY_CONFIG["base_safety_score"]

        # Tools with destructive operations should require confirmation
        if any(word in tool.description.lower() for word in ["delete", "remove", "overwrite", "replace"]):
            # Check if there are safety parameters
            has_safety_params = any(
                word in param.name.lower()
                for param in tool.parameters
                for word in ["confirm", "force", "backup", "dry_run"]
            )
            if has_safety_params:
                safety_score += 0.2
            else:
                safety_score -= 0.3

        return max(0.2, min(safety_score, 1.0))

    def _get_category_functionality_bonus(self, category: str) -> float:
        """Get functionality bonus based on tool category."""
        category_bonuses = {"advanced": 0.8, "comprehensive": 0.9, "specialized": 0.7, "basic": 0.3, "simple": 0.2}
        return category_bonuses.get(category.lower(), 0.5)

    def _get_category_usability_bias(self, category: str) -> float:
        """Get usability bias based on tool category."""
        category_usability = {
            "user_friendly": 0.9,
            "simple": 0.8,
            "basic": 0.7,
            "advanced": 0.3,
            "expert": 0.2,
            "specialized": 0.4,
        }
        return category_usability.get(category.lower(), 0.5)

    def _get_category_reliability_baseline(self, category: str) -> float:
        """Get reliability baseline based on tool category."""
        category_reliability = {
            "core": 0.9,
            "stable": 0.8,
            "established": 0.8,
            "experimental": 0.3,
            "beta": 0.4,
            "deprecated": 0.2,
        }
        return category_reliability.get(category.lower(), 0.6)

    def _get_category_performance_baseline(self, category: str) -> float:
        """Get performance baseline based on tool category."""
        category_performance = {
            "high_performance": 0.9,
            "optimized": 0.8,
            "fast": 0.8,
            "lightweight": 0.7,
            "heavy": 0.3,
            "resource_intensive": 0.2,
        }
        return category_performance.get(category.lower(), 0.6)

    def _estimate_operation_performance(self, tool: ToolSearchResult) -> float:
        """Estimate performance based on operation type."""
        if not tool.description:
            return 0.5

        desc_lower = tool.description.lower()

        # Fast operations
        if any(word in desc_lower for word in ["read", "get", "retrieve", "view", "display"]):
            return 0.8

        # Medium operations
        elif any(word in desc_lower for word in ["search", "find", "filter", "parse"]):
            return 0.6

        # Slower operations
        elif any(word in desc_lower for word in ["process", "analyze", "transform", "generate"]):
            return 0.4

        # Potentially slow operations
        elif any(word in desc_lower for word in ["index", "build", "compile", "aggregate"]):
            return 0.3

        return 0.5

    def _assess_resource_efficiency(self, tool: ToolSearchResult) -> float:
        """Assess estimated resource efficiency."""
        # Simple heuristic based on parameter complexity and operation type
        param_efficiency = 1.0 - (tool.parameter_count / 20.0) if tool.parameter_count else 0.8

        # Operation efficiency from description
        if tool.description:
            desc_lower = tool.description.lower()
            if any(word in desc_lower for word in ["efficient", "fast", "lightweight", "optimized"]):
                return min(param_efficiency + 0.2, 1.0)
            elif any(word in desc_lower for word in ["heavy", "intensive", "complex", "comprehensive"]):
                return max(param_efficiency - 0.3, 0.2)

        return param_efficiency

    def _assess_io_compatibility(self, tool: ToolSearchResult) -> float:
        """Assess input/output compatibility."""
        # Tools with standard I/O patterns are more compatible
        if tool.parameters:
            has_input_param = any(
                "input" in param.name.lower() or "file" in param.name.lower() for param in tool.parameters
            )
            has_output_param = any(
                "output" in param.name.lower() or "result" in param.name.lower() for param in tool.parameters
            )

            if has_input_param and has_output_param:
                return 0.9
            elif has_input_param or has_output_param:
                return 0.7

        return 0.6

    def _assess_parameter_type_compatibility(self, tool: ToolSearchResult) -> float:
        """Assess parameter type compatibility."""
        if not tool.parameters:
            return 0.8  # Compatible by default

        # Standard types are more compatible
        standard_types = {"string", "number", "boolean", "array"}
        compatible_params = sum(1 for p in tool.parameters if p.type and p.type.lower() in standard_types)

        return compatible_params / len(tool.parameters)

    def _assess_ecosystem_integration(self, tool: ToolSearchResult, context: Optional[TaskContext]) -> float:
        """Assess ecosystem integration compatibility."""
        # Base compatibility
        base_score = 0.6

        # Context-aware adjustments
        if context and context.project_context:
            # This would be enhanced with actual project context analysis
            pass

        return base_score

    def _assess_error_handling_compatibility(self, tool: ToolSearchResult) -> float:
        """Assess error handling compatibility."""
        if tool.description:
            desc_lower = tool.description.lower()
            if any(word in desc_lower for word in ["error", "exception", "fail", "status"]):
                return 0.8

        return 0.5

    def _calculate_description_quality(self, description: str) -> float:
        """Calculate overall description quality."""
        if not description or not description.strip():
            return 0.1

        # Use cached result if available
        cache_key = hash(description)
        if cache_key in self._description_quality_cache:
            return self._description_quality_cache[cache_key]

        config = self.scoring_config["description_quality_weights"]

        # Length factor
        length_score = 0.5
        if len(description) > config["length_bonus_threshold"]:
            length_score = min(len(description) / 200.0, 1.0)
        elif len(description) < 10:
            length_score = 0.2

        # Technical content
        technical_terms = len(
            re.findall(r"\b(parameter|function|method|class|object|data|file|process)\b", description.lower())
        )
        technical_score = min(technical_terms / 3.0, 1.0)

        # Example mentions
        has_examples = any(word in description.lower() for word in ["example", "e.g.", "such as", "for instance"])
        example_score = config["example_mention_bonus"] if has_examples else 0.0

        quality_score = length_score * 0.5 + technical_score * 0.3 + example_score

        # Cache and return
        self._description_quality_cache[cache_key] = quality_score
        return quality_score

    def _detect_example_indicators(self, description: str) -> float:
        """Detect example indicators in description."""
        if not description:
            return 0.0

        example_patterns = ["example", "e.g.", "such as", "for instance", "like", "usage:", "example:"]
        mentions = sum(1 for pattern in example_patterns if pattern in description.lower())

        return min(mentions / 2.0, 1.0)

    def _normalize_score(self, score: float) -> float:
        """Normalize score to valid range."""
        return max(self.scoring_config["min_score"], min(score, self.scoring_config["max_score"]))
