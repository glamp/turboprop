#!/usr/bin/env python3
"""
Parameter Analyzer

This module provides comprehensive parameter schema analysis and matching
for parameter-aware tool search capabilities.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from logging_config import get_logger
from mcp_metadata_types import ParameterAnalysis, ToolId

logger = get_logger(__name__)


@dataclass
class ParameterRequirements:
    """Structured parameter requirements for search."""

    input_types: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)
    required_parameters: List[str] = field(default_factory=list)
    optional_parameters: List[str] = field(default_factory=list)
    parameter_constraints: Dict[str, Any] = field(default_factory=dict)
    min_parameters: Optional[int] = None
    max_parameters: Optional[int] = None
    complexity_preference: str = "any"  # 'simple', 'moderate', 'complex', 'any'


@dataclass
class ParameterMatchResult:
    """Result of parameter matching analysis."""

    overall_match_score: float
    type_compatibility: Dict[str, bool] = field(default_factory=dict)
    required_parameter_matches: List[str] = field(default_factory=list)
    optional_parameter_matches: List[str] = field(default_factory=list)
    missing_requirements: List[str] = field(default_factory=list)
    compatibility_explanation: str = ""
    suggested_modifications: List[str] = field(default_factory=list)


@dataclass
class ParameterAnalysisResult:
    """Result of parameter schema analysis."""

    total_count: int = 0
    required_count: int = 0
    optional_count: int = 0
    required_parameters: List[str] = field(default_factory=list)
    optional_parameters: List[str] = field(default_factory=list)
    parameter_types: Dict[str, str] = field(default_factory=dict)
    complexity_score: float = 0.0
    has_nested_objects: bool = False
    has_arrays: bool = False
    validation_errors: List[str] = field(default_factory=list)


class ParameterAnalyzer:
    """Analyze and match tool parameters."""

    def __init__(self):
        """Initialize parameter analyzer with type mappings."""
        self.type_mappings = self._build_type_mappings()
        self.complexity_weights = {
            "basic_type": 0.1,
            "object_type": 0.3,
            "array_type": 0.2,
            "constraint_complexity": 0.15,
            "nested_depth": 0.25,
        }

    def analyze_parameter_schema(self, schema: Dict[str, Any]) -> ParameterAnalysisResult:
        """
        Deep analysis of parameter schema.

        Args:
            schema: JSON schema for parameters

        Returns:
            ParameterAnalysisResult with detailed analysis
        """
        try:
            result = ParameterAnalysisResult()

            # Extract properties and required fields
            properties = schema.get("properties", {})
            required_fields = set(schema.get("required", []))

            result.total_count = len(properties)
            result.required_count = len(required_fields)
            result.optional_count = result.total_count - result.required_count

            # Analyze each parameter
            for param_name, param_schema in properties.items():
                param_type = param_schema.get("type", "string")
                result.parameter_types[param_name] = param_type

                if param_name in required_fields:
                    result.required_parameters.append(param_name)
                else:
                    result.optional_parameters.append(param_name)

                # Check for complex types
                if param_type == "object":
                    result.has_nested_objects = True
                elif param_type == "array":
                    result.has_arrays = True

            # Calculate complexity score
            result.complexity_score = self._calculate_schema_complexity(schema)

            logger.debug(
                "Analyzed schema: %d total, %d required, complexity=%.2f",
                result.total_count,
                result.required_count,
                result.complexity_score,
            )

            return result

        except Exception as e:
            logger.error("Error analyzing parameter schema: %s", e)
            result = ParameterAnalysisResult()
            result.validation_errors.append(f"Schema analysis failed: {str(e)}")
            return result

    def analyze_parameter_complexity(self, parameters: List[ParameterAnalysis]) -> Dict[str, Any]:
        """
        Analyze parameter complexity and characteristics.

        Args:
            parameters: List of ParameterAnalysis objects

        Returns:
            Dictionary with complexity metrics
        """
        if not parameters:
            return {
                "total_count": 0,
                "required_count": 0,
                "optional_count": 0,
                "complexity_score": 0.0,
                "has_nested": False,
                "has_arrays": False,
                "type_distribution": {},
                "constraint_complexity": 0.0,
            }

        total_count = len(parameters)
        required_count = sum(1 for p in parameters if p.required)
        optional_count = total_count - required_count

        # Analyze parameter types and complexity
        type_distribution = {}
        has_nested = False
        has_arrays = False
        total_constraint_complexity = 0.0

        for param in parameters:
            # Count type distribution
            param_type = param.type.lower()
            type_distribution[param_type] = type_distribution.get(param_type, 0) + 1

            # Check for complex types
            if param_type in ["object", "dict"]:
                has_nested = True
            elif param_type in ["array", "list"]:
                has_arrays = True

            # Analyze constraint complexity
            constraint_score = self._analyze_constraint_complexity(param.constraints)
            total_constraint_complexity += constraint_score

        # Calculate overall complexity score
        complexity_score = self._calculate_parameter_set_complexity(
            parameters, has_nested, has_arrays, total_constraint_complexity
        )

        return {
            "total_count": total_count,
            "required_count": required_count,
            "optional_count": optional_count,
            "complexity_score": complexity_score,
            "has_nested": has_nested,
            "has_arrays": has_arrays,
            "type_distribution": type_distribution,
            "constraint_complexity": total_constraint_complexity / max(total_count, 1),
        }

    def match_parameter_requirements(
        self, requirements: ParameterRequirements, tool_parameters: List[ParameterAnalysis]
    ) -> ParameterMatchResult:
        """
        Match parameter requirements against tool schema.

        Args:
            requirements: Parameter requirements to match
            tool_parameters: Tool's parameter analysis list

        Returns:
            ParameterMatchResult with detailed matching analysis
        """
        # Build parameter name sets for quick lookup
        tool_param_names = {p.name.lower() for p in tool_parameters}
        tool_param_types = {p.name.lower(): p.type.lower() for p in tool_parameters}
        required_tool_params = {p.name.lower() for p in tool_parameters if p.required}

        # Track matches
        required_matches = []
        optional_matches = []
        missing_requirements = []
        type_compatibility = {}

        # Check required parameter matches
        for req_param in requirements.required_parameters:
            req_param_lower = req_param.lower()
            if req_param_lower in tool_param_names:
                required_matches.append(req_param)
            else:
                missing_requirements.append(req_param)

        # Check optional parameter matches
        for opt_param in requirements.optional_parameters:
            opt_param_lower = opt_param.lower()
            if opt_param_lower in tool_param_names:
                optional_matches.append(opt_param)

        # Check type compatibility
        for param_name, param_type in tool_param_types.items():
            for req_type in requirements.input_types:
                is_compatible = self._check_type_compatibility(param_type, req_type.lower())
                type_compatibility[f"{param_name}:{req_type}"] = is_compatible

        # Calculate overall match score
        overall_score = self._calculate_overall_match_score(
            requirements, required_matches, optional_matches, missing_requirements, type_compatibility
        )

        # Generate explanation
        explanation = self._generate_match_explanation(
            requirements, required_matches, optional_matches, missing_requirements
        )

        return ParameterMatchResult(
            overall_match_score=overall_score,
            type_compatibility=type_compatibility,
            required_parameter_matches=required_matches,
            optional_parameter_matches=optional_matches,
            missing_requirements=missing_requirements,
            compatibility_explanation=explanation,
            suggested_modifications=self._suggest_modifications(requirements, tool_parameters),
        )

    def calculate_parameter_similarity(
        self, params_a: List[ParameterAnalysis], params_b: List[ParameterAnalysis]
    ) -> float:
        """
        Calculate similarity between parameter sets.

        Args:
            params_a: First parameter set
            params_b: Second parameter set

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not params_a and not params_b:
            return 1.0
        if not params_a or not params_b:
            return 0.0

        # Build parameter name and type sets
        names_a = {p.name.lower() for p in params_a}
        names_b = {p.name.lower() for p in params_b}
        types_a = {p.type.lower() for p in params_a}
        types_b = {p.type.lower() for p in params_b}

        # Calculate name similarity (Jaccard index)
        name_intersection = len(names_a & names_b)
        name_union = len(names_a | names_b)
        name_similarity = name_intersection / name_union if name_union > 0 else 0.0

        # Calculate type similarity
        type_intersection = len(types_a & types_b)
        type_union = len(types_a | types_b)
        type_similarity = type_intersection / type_union if type_union > 0 else 0.0

        # Calculate semantic similarity using descriptions
        semantic_similarity = self._calculate_semantic_similarity(params_a, params_b)

        # Calculate requirement pattern similarity
        required_a = {p.name.lower() for p in params_a if p.required}
        required_b = {p.name.lower() for p in params_b if p.required}
        req_intersection = len(required_a & required_b)
        req_union = len(required_a | required_b)
        requirement_similarity = req_intersection / req_union if req_union > 0 else 0.0

        # Weighted combination
        similarity = (
            name_similarity * 0.4 + type_similarity * 0.25 + semantic_similarity * 0.2 + requirement_similarity * 0.15
        )

        logger.debug(
            "Parameter similarity: name=%.2f, type=%.2f, semantic=%.2f, req=%.2f -> %.2f",
            name_similarity,
            type_similarity,
            semantic_similarity,
            requirement_similarity,
            similarity,
        )

        return min(1.0, max(0.0, similarity))

    def _build_type_mappings(self) -> Dict[str, Set[str]]:
        """Build type compatibility mappings."""
        return {
            "string": {"text", "path", "url", "pattern", "format", "name"},
            "number": {"int", "integer", "float", "double", "timeout", "limit", "size"},
            "boolean": {"bool", "flag", "enable", "disable", "switch"},
            "object": {"dict", "config", "options", "settings", "params"},
            "array": {"list", "collection", "sequence", "items", "elements"},
        }

    def _calculate_schema_complexity(self, schema: Dict[str, Any]) -> float:
        """Calculate complexity score for a JSON schema."""
        complexity = 0.0

        properties = schema.get("properties", {})
        required_fields = schema.get("required", [])

        # Base complexity from parameter count
        param_count = len(properties)
        complexity += min(0.4, param_count * 0.05)

        # Complexity from required parameters
        complexity += min(0.2, len(required_fields) * 0.03)

        # Analyze each property for type complexity
        for prop_name, prop_schema in properties.items():
            prop_type = prop_schema.get("type", "string")

            if prop_type == "object":
                complexity += 0.15
                # Nested complexity
                if "properties" in prop_schema:
                    complexity += min(0.1, len(prop_schema["properties"]) * 0.02)
            elif prop_type == "array":
                complexity += 0.1
                # Array item complexity
                if "items" in prop_schema:
                    item_type = prop_schema["items"].get("type", "string")
                    if item_type == "object":
                        complexity += 0.05

            # Constraint complexity
            constraints = len(prop_schema.keys()) - 2  # Exclude type and description
            complexity += min(0.05, constraints * 0.01)

        return min(1.0, complexity)

    def _calculate_parameter_set_complexity(
        self, parameters: List[ParameterAnalysis], has_nested: bool, has_arrays: bool, constraint_complexity: float
    ) -> float:
        """Calculate complexity for a set of parameters."""
        if not parameters:
            return 0.0

        base_complexity = min(0.4, len(parameters) * 0.08)
        type_complexity = 0.0

        if has_nested:
            type_complexity += 0.25
        if has_arrays:
            type_complexity += 0.15

        # Add individual parameter complexities
        param_complexity = sum(getattr(p, "complexity_score", 0.1) for p in parameters) / len(parameters)

        total_complexity = base_complexity + type_complexity + param_complexity * 0.2 + constraint_complexity * 0.1

        return min(1.0, total_complexity)

    def _analyze_constraint_complexity(self, constraints: Dict[str, Any]) -> float:
        """Analyze complexity of parameter constraints."""
        if not constraints:
            return 0.0

        complexity = 0.0

        # Basic constraint types
        simple_constraints = {"minimum", "maximum", "minLength", "maxLength", "pattern"}
        complex_constraints = {"enum", "properties", "items", "anyOf", "oneOf", "allOf"}

        for constraint_name in constraints:
            if constraint_name in simple_constraints:
                complexity += 0.1
            elif constraint_name in complex_constraints:
                complexity += 0.2
            else:
                complexity += 0.05  # Unknown constraint

        # Nested complexity for object properties
        if "properties" in constraints:
            nested_props = constraints["properties"]
            complexity += len(nested_props) * 0.05

        return min(1.0, complexity)

    def _check_type_compatibility(self, tool_type: str, required_type: str) -> bool:
        """Check if tool parameter type is compatible with required type."""
        tool_type = tool_type.lower()
        required_type = required_type.lower()

        # Direct match
        if tool_type == required_type:
            return True

        # Check type mappings
        for base_type, compatible_types in self.type_mappings.items():
            if tool_type == base_type and required_type in compatible_types:
                return True
            if required_type == base_type and tool_type in compatible_types:
                return True
            if tool_type in compatible_types and required_type in compatible_types:
                return True

        return False

    def _calculate_overall_match_score(
        self,
        requirements: ParameterRequirements,
        required_matches: List[str],
        optional_matches: List[str],
        missing_requirements: List[str],
        type_compatibility: Dict[str, bool],
    ) -> float:
        """Calculate overall match score for parameter requirements."""
        score = 0.5  # Base score

        # Required parameter matching (heavily weighted)
        if requirements.required_parameters:
            required_ratio = len(required_matches) / len(requirements.required_parameters)
            score += required_ratio * 0.4

            # Penalty for missing required parameters
            if missing_requirements:
                penalty = len(missing_requirements) / len(requirements.required_parameters)
                score -= penalty * 0.3
        else:
            score += 0.2  # Bonus for no specific requirements

        # Optional parameter matching (moderately weighted)
        if requirements.optional_parameters:
            optional_ratio = len(optional_matches) / len(requirements.optional_parameters)
            score += optional_ratio * 0.2

        # Type compatibility bonus
        if type_compatibility:
            compatible_count = sum(1 for is_compatible in type_compatibility.values() if is_compatible)
            total_checks = len(type_compatibility)
            if total_checks > 0:
                type_score = compatible_count / total_checks
                score += type_score * 0.1

        return min(1.0, max(0.0, score))

    def _generate_match_explanation(
        self,
        requirements: ParameterRequirements,
        required_matches: List[str],
        optional_matches: List[str],
        missing_requirements: List[str],
    ) -> str:
        """Generate human-readable explanation of parameter matching."""
        explanations = []

        if required_matches:
            explanations.append(f"Matches {len(required_matches)} required parameters: {', '.join(required_matches)}")

        if optional_matches:
            explanations.append(f"Matches {len(optional_matches)} optional parameters: {', '.join(optional_matches)}")

        if missing_requirements:
            explanations.append(
                f"Missing {len(missing_requirements)} required parameters: {', '.join(missing_requirements)}"
            )

        if not explanations:
            return "No specific parameter requirements to match"

        return ". ".join(explanations)

    def _suggest_modifications(
        self, requirements: ParameterRequirements, tool_parameters: List[ParameterAnalysis]
    ) -> List[str]:
        """Suggest modifications to improve parameter compatibility."""
        suggestions = []

        tool_param_names = {p.name.lower() for p in tool_parameters}

        # Suggest similar parameter names for missing requirements
        for missing_param in requirements.required_parameters:
            if missing_param.lower() not in tool_param_names:
                similar_params = [p.name for p in tool_parameters if self._is_similar_name(missing_param, p.name)]
                if similar_params:
                    suggestions.append(f"Consider using '{similar_params[0]}' instead of '{missing_param}'")

        # Suggest relaxing requirements if too restrictive
        if len(requirements.required_parameters) > len(tool_parameters):
            suggestions.append("Consider making some required parameters optional")

        return suggestions

    def _calculate_semantic_similarity(
        self, params_a: List[ParameterAnalysis], params_b: List[ParameterAnalysis]
    ) -> float:
        """Calculate semantic similarity between parameter descriptions."""
        if not params_a or not params_b:
            return 0.0

        # Simple keyword-based semantic similarity
        # In practice, this could use embeddings for better accuracy
        descriptions_a = [p.description.lower() for p in params_a if p.description]
        descriptions_b = [p.description.lower() for p in params_b if p.description]

        if not descriptions_a or not descriptions_b:
            return 0.5  # Neutral score if no descriptions

        # Extract keywords from descriptions
        keywords_a = set()
        keywords_b = set()

        for desc in descriptions_a:
            keywords_a.update(desc.split())
        for desc in descriptions_b:
            keywords_b.update(desc.split())

        # Calculate keyword overlap
        intersection = len(keywords_a & keywords_b)
        union = len(keywords_a | keywords_b)

        return intersection / union if union > 0 else 0.0

    def _is_similar_name(self, name1: str, name2: str) -> bool:
        """Check if two parameter names are similar."""
        name1_lower = name1.lower()
        name2_lower = name2.lower()

        # Check for common variations
        variations = [
            ("file_path", "filepath", "path", "file"),
            ("timeout", "time_out", "max_time", "timeout_sec"),
            ("max_size", "maxsize", "size_limit", "limit"),
            ("output_format", "format", "output_type", "type"),
        ]

        for variation in variations:
            if name1_lower in variation and name2_lower in variation:
                return True

        # Simple edit distance check (could be improved)
        if abs(len(name1) - len(name2)) <= 2:
            common_chars = sum(1 for c1, c2 in zip(name1_lower, name2_lower) if c1 == c2)
            similarity = common_chars / max(len(name1), len(name2))
            return similarity > 0.7

        return False
