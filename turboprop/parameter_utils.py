#!/usr/bin/env python3
"""
Parameter Processing Utilities

This module provides shared utility functions for processing tool parameters
to reduce code duplication across the codebase.
"""

from typing import Any, Dict, List, Optional, Tuple

from .mcp_metadata_types import ParameterAnalysis


def calculate_parameter_counts(parameters: List[ParameterAnalysis]) -> Tuple[int, int]:
    """
    Calculate total and required parameter counts.

    Args:
        parameters: List of ParameterAnalysis objects

    Returns:
        Tuple of (total_count, required_count)
    """
    if not parameters:
        return 0, 0

    total_count = len(parameters)
    required_count = sum(1 for p in parameters if p.required)

    return total_count, required_count


def validate_parameter_counts(
    parameter_count: int, required_parameter_count: int, parameters: Optional[List[ParameterAnalysis]] = None
) -> Tuple[int, int]:
    """
    Validate and correct parameter counts, computing from parameters if needed.

    Args:
        parameter_count: Current parameter count
        required_parameter_count: Current required parameter count
        parameters: Optional list of parameters to compute from

    Returns:
        Tuple of (validated_parameter_count, validated_required_parameter_count)
    """
    # If counts are zero but parameters exist, calculate from parameters
    if parameter_count == 0 and required_parameter_count == 0 and parameters:
        return calculate_parameter_counts(parameters)

    # If only one count is zero, calculate it from parameters if available
    if parameters:
        actual_total, actual_required = calculate_parameter_counts(parameters)

        if parameter_count == 0 and actual_total > 0:
            parameter_count = actual_total

        if required_parameter_count == 0 and actual_required > 0:
            required_parameter_count = actual_required

    return parameter_count, required_parameter_count


def analyze_parameter_complexity(parameters: List[ParameterAnalysis]) -> Dict[str, Any]:
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
        }

    total_count, required_count = calculate_parameter_counts(parameters)
    optional_count = total_count - required_count

    # Analyze parameter types for complexity
    has_nested = any(p.type and "object" in str(p.type).lower() for p in parameters)
    has_arrays = any(p.type and "array" in str(p.type).lower() for p in parameters)

    # Simple complexity score based on count and types
    complexity_score = min(1.0, (total_count * 0.1) + (0.2 if has_nested else 0) + (0.1 if has_arrays else 0))

    return {
        "total_count": total_count,
        "required_count": required_count,
        "optional_count": optional_count,
        "complexity_score": complexity_score,
        "has_nested": has_nested,
        "has_arrays": has_arrays,
    }


def extract_parameter_names(parameters: List[ParameterAnalysis]) -> Dict[str, List[str]]:
    """
    Extract parameter names categorized by requirement type.

    Args:
        parameters: List of ParameterAnalysis objects

    Returns:
        Dictionary with 'required' and 'optional' parameter name lists
    """
    required_names = []
    optional_names = []

    for param in parameters:
        if param.required:
            required_names.append(param.name)
        else:
            optional_names.append(param.name)

    return {
        "required": required_names,
        "optional": optional_names,
    }


def calculate_parameter_compatibility_score(
    tool_parameters: List[ParameterAnalysis], constraints: Dict[str, Any]
) -> float:
    """
    Calculate parameter compatibility score based on constraints.

    Args:
        tool_parameters: List of tool parameters
        constraints: Dictionary of parameter constraints

    Returns:
        Compatibility score between 0.0 and 1.0
    """
    if not tool_parameters:
        return 0.5  # Neutral score for tools with no parameters

    total_count, required_count = calculate_parameter_counts(tool_parameters)
    compatibility_score = 0.5  # Base score

    # Check parameter count constraints
    if "max_params" in constraints:
        max_params = int(constraints["max_params"])
        if total_count <= max_params:
            compatibility_score += 0.2
        else:
            compatibility_score -= 0.3

    if "min_params" in constraints:
        min_params = int(constraints["min_params"])
        if total_count >= min_params:
            compatibility_score += 0.1
        else:
            compatibility_score -= 0.2

    # Check required parameter constraints
    if "has_required_params" in constraints:
        has_required = constraints["has_required_params"].lower() == "true"
        if (has_required and required_count > 0) or (not has_required and required_count == 0):
            compatibility_score += 0.2

    return max(0.0, min(1.0, compatibility_score))
