#!/usr/bin/env python3
"""
Advanced Filters

This module provides sophisticated multi-dimensional filtering capabilities
for parameter-aware tool search, including parameter count, type, and
complexity-based filtering.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .logging_config import get_logger

logger = get_logger(__name__)


class FilterOperator(Enum):
    """Filter operators for parameter matching."""

    EQUALS = "equals"
    CONTAINS = "contains"
    MATCHES = "matches"  # regex
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    GREATER_EQUAL = "gte"
    LESS_EQUAL = "lte"
    IN = "in"
    NOT_IN = "not_in"


class IntegrationLevel(Enum):
    """Tool integration complexity levels."""

    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


@dataclass
class ParameterFilterSet:
    """Set of parameter-based filters."""

    min_parameters: Optional[int] = None
    max_parameters: Optional[int] = None
    required_parameter_types: List[str] = field(default_factory=list)
    forbidden_parameter_types: List[str] = field(default_factory=list)
    min_required_parameters: Optional[int] = None
    max_required_parameters: Optional[int] = None
    min_optional_parameters: Optional[int] = None
    max_optional_parameters: Optional[int] = None
    max_complexity: Optional[float] = None
    min_complexity: Optional[float] = None
    parameter_name_patterns: List[str] = field(default_factory=list)
    required_parameter_names: List[str] = field(default_factory=list)
    forbidden_parameter_names: List[str] = field(default_factory=list)
    allow_nested_objects: Optional[bool] = None
    allow_arrays: Optional[bool] = None


@dataclass
class CompatibilityRequirements:
    """Tool compatibility requirements."""

    input_compatibility: List[str] = field(default_factory=list)
    output_compatibility: List[str] = field(default_factory=list)
    chaining_compatibility: bool = False
    integration_level: IntegrationLevel = IntegrationLevel.BASIC
    exclude_incompatible: bool = True
    min_compatibility_score: float = 0.5
    required_capabilities: List[str] = field(default_factory=list)
    forbidden_capabilities: List[str] = field(default_factory=list)


@dataclass
class FilterResult:
    """Result of filtering operation."""

    passed_tools: List[Any] = field(default_factory=list)
    filtered_out: List[Any] = field(default_factory=list)
    filter_statistics: Dict[str, int] = field(default_factory=dict)
    applied_filters: List[str] = field(default_factory=list)
    execution_time: float = 0.0


class FilterCondition:
    """A single filter condition with operator and value."""

    def __init__(self, field_path: str, operator: FilterOperator, value: Any):
        """
        Initialize filter condition.

        Args:
            field_path: Dot-separated path to field (e.g., "parameters.name")
            operator: Filter operator
            value: Value to compare against
        """
        self.field_path = field_path
        self.operator = operator
        self.value = value

    def evaluate(self, tool_data: Any) -> bool:
        """Evaluate this condition against tool data."""
        try:
            actual_value = self._get_field_value(tool_data, self.field_path)
            return self._apply_operator(actual_value, self.operator, self.value)
        except Exception as e:
            logger.debug("Filter condition evaluation failed: %s", e)
            return False

    def _get_field_value(self, data: Any, field_path: str) -> Any:
        """Get value from nested field path."""
        parts = field_path.split(".")
        current = data

        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, list) and part.isdigit():
                current = current[int(part)]
            else:
                raise ValueError(f"Field path {field_path} not found")

        return current

    def _apply_operator(self, actual: Any, operator: FilterOperator, expected: Any) -> bool:
        """Apply operator to compare values."""
        if operator == FilterOperator.EQUALS:
            return actual == expected
        elif operator == FilterOperator.CONTAINS:
            return expected in str(actual).lower()
        elif operator == FilterOperator.MATCHES:
            return re.search(str(expected), str(actual), re.IGNORECASE) is not None
        elif operator == FilterOperator.GREATER_THAN:
            return float(actual) > float(expected)
        elif operator == FilterOperator.LESS_THAN:
            return float(actual) < float(expected)
        elif operator == FilterOperator.GREATER_EQUAL:
            return float(actual) >= float(expected)
        elif operator == FilterOperator.LESS_EQUAL:
            return float(actual) <= float(expected)
        elif operator == FilterOperator.IN:
            return actual in expected
        elif operator == FilterOperator.NOT_IN:
            return actual not in expected
        else:
            return False


class AdvancedFilters:
    """Multi-dimensional filtering for tool search."""

    def __init__(self):
        """Initialize advanced filters with performance tracking."""
        self.filter_cache = {}
        self.performance_stats = {"total_filters_applied": 0, "cache_hits": 0, "average_execution_time": 0.0}
        logger.info("Initialized AdvancedFilters")

    def apply_parameter_filters(self, tools: List[Any], filters: ParameterFilterSet) -> List[Any]:
        """
        Apply parameter-based filters to tool results.

        Args:
            tools: List of tool results to filter
            filters: Parameter filter set to apply

        Returns:
            Filtered list of tools
        """
        if not tools:
            return []

        filtered_tools = tools.copy()
        applied_filters = []

        try:
            # Parameter count filters
            if filters.min_parameters is not None:
                filtered_tools = [
                    tool for tool in filtered_tools if self._get_parameter_count(tool) >= filters.min_parameters
                ]
                applied_filters.append(f"min_parameters>={filters.min_parameters}")

            if filters.max_parameters is not None:
                filtered_tools = [
                    tool for tool in filtered_tools if self._get_parameter_count(tool) <= filters.max_parameters
                ]
                applied_filters.append(f"max_parameters<={filters.max_parameters}")

            # Required parameter count filters
            if filters.min_required_parameters is not None:
                filtered_tools = [
                    tool
                    for tool in filtered_tools
                    if self._get_required_parameter_count(tool) >= filters.min_required_parameters
                ]
                applied_filters.append(f"min_required>={filters.min_required_parameters}")

            if filters.max_required_parameters is not None:
                filtered_tools = [
                    tool
                    for tool in filtered_tools
                    if self._get_required_parameter_count(tool) <= filters.max_required_parameters
                ]
                applied_filters.append(f"max_required<={filters.max_required_parameters}")

            # Optional parameter count filters
            if filters.min_optional_parameters is not None:
                filtered_tools = [
                    tool
                    for tool in filtered_tools
                    if self._get_optional_parameter_count(tool) >= filters.min_optional_parameters
                ]
                applied_filters.append(f"min_optional>={filters.min_optional_parameters}")

            if filters.max_optional_parameters is not None:
                filtered_tools = [
                    tool
                    for tool in filtered_tools
                    if self._get_optional_parameter_count(tool) <= filters.max_optional_parameters
                ]
                applied_filters.append(f"max_optional<={filters.max_optional_parameters}")

            # Parameter type filters
            if filters.required_parameter_types:
                filtered_tools = [
                    tool
                    for tool in filtered_tools
                    if self._has_required_parameter_types(tool, filters.required_parameter_types)
                ]
                applied_filters.append(f"required_types={filters.required_parameter_types}")

            if filters.forbidden_parameter_types:
                filtered_tools = [
                    tool
                    for tool in filtered_tools
                    if not self._has_forbidden_parameter_types(tool, filters.forbidden_parameter_types)
                ]
                applied_filters.append(f"forbidden_types={filters.forbidden_parameter_types}")

            # Complexity filters
            if filters.max_complexity is not None:
                filtered_tools = [
                    tool for tool in filtered_tools if self._get_complexity_score(tool) <= filters.max_complexity
                ]
                applied_filters.append(f"max_complexity<={filters.max_complexity}")

            if filters.min_complexity is not None:
                filtered_tools = [
                    tool for tool in filtered_tools if self._get_complexity_score(tool) >= filters.min_complexity
                ]
                applied_filters.append(f"min_complexity>={filters.min_complexity}")

            # Parameter name filters
            if filters.parameter_name_patterns:
                filtered_tools = [
                    tool
                    for tool in filtered_tools
                    if self._matches_parameter_name_patterns(tool, filters.parameter_name_patterns)
                ]
                applied_filters.append(f"name_patterns={filters.parameter_name_patterns}")

            if filters.required_parameter_names:
                filtered_tools = [
                    tool
                    for tool in filtered_tools
                    if self._has_required_parameter_names(tool, filters.required_parameter_names)
                ]
                applied_filters.append(f"required_names={filters.required_parameter_names}")

            if filters.forbidden_parameter_names:
                filtered_tools = [
                    tool
                    for tool in filtered_tools
                    if not self._has_forbidden_parameter_names(tool, filters.forbidden_parameter_names)
                ]
                applied_filters.append(f"forbidden_names={filters.forbidden_parameter_names}")

            # Structural filters
            if filters.allow_nested_objects is not None:
                if not filters.allow_nested_objects:
                    filtered_tools = [tool for tool in filtered_tools if not self._has_nested_objects(tool)]
                    applied_filters.append("no_nested_objects")

            if filters.allow_arrays is not None:
                if not filters.allow_arrays:
                    filtered_tools = [tool for tool in filtered_tools if not self._has_arrays(tool)]
                    applied_filters.append("no_arrays")

            logger.debug(
                "Parameter filters applied: %s. Results: %d -> %d tools",
                ", ".join(applied_filters),
                len(tools),
                len(filtered_tools),
            )

            return filtered_tools

        except Exception as e:
            logger.error("Error applying parameter filters: %s", e)
            return tools  # Return original list on error

    def apply_compatibility_filters(
        self, tools: List[Any], compatibility_requirements: CompatibilityRequirements
    ) -> List[Any]:
        """
        Filter tools by compatibility requirements.

        Args:
            tools: List of tool results to filter
            compatibility_requirements: Compatibility requirements

        Returns:
            Filtered list of tools
        """
        if not tools:
            return []

        filtered_tools = tools.copy()
        applied_filters = []

        try:
            # Input compatibility filtering
            if compatibility_requirements.input_compatibility:
                filtered_tools = [
                    tool
                    for tool in filtered_tools
                    if self._check_input_compatibility(tool, compatibility_requirements.input_compatibility)
                ]
                applied_filters.append(f"input_types={compatibility_requirements.input_compatibility}")

            # Output compatibility filtering
            if compatibility_requirements.output_compatibility:
                filtered_tools = [
                    tool
                    for tool in filtered_tools
                    if self._check_output_compatibility(tool, compatibility_requirements.output_compatibility)
                ]
                applied_filters.append(f"output_types={compatibility_requirements.output_compatibility}")

            # Chaining compatibility
            if compatibility_requirements.chaining_compatibility:
                filtered_tools = [tool for tool in filtered_tools if self._supports_chaining(tool)]
                applied_filters.append("chaining_compatible")

            # Integration level filtering
            if compatibility_requirements.integration_level != IntegrationLevel.BASIC:
                filtered_tools = [
                    tool
                    for tool in filtered_tools
                    if self._meets_integration_level(tool, compatibility_requirements.integration_level)
                ]
                applied_filters.append(f"integration_level={compatibility_requirements.integration_level.value}")

            # Minimum compatibility score
            if compatibility_requirements.min_compatibility_score > 0.0:
                filtered_tools = [
                    tool
                    for tool in filtered_tools
                    if self._get_compatibility_score(tool) >= compatibility_requirements.min_compatibility_score
                ]
                applied_filters.append(f"min_compat_score>={compatibility_requirements.min_compatibility_score}")

            # Required capabilities
            if compatibility_requirements.required_capabilities:
                filtered_tools = [
                    tool
                    for tool in filtered_tools
                    if self._has_required_capabilities(tool, compatibility_requirements.required_capabilities)
                ]
                applied_filters.append(f"required_caps={compatibility_requirements.required_capabilities}")

            # Forbidden capabilities
            if compatibility_requirements.forbidden_capabilities:
                filtered_tools = [
                    tool
                    for tool in filtered_tools
                    if not self._has_forbidden_capabilities(tool, compatibility_requirements.forbidden_capabilities)
                ]
                applied_filters.append(f"forbidden_caps={compatibility_requirements.forbidden_capabilities}")

            logger.debug(
                "Compatibility filters applied: %s. Results: %d -> %d tools",
                ", ".join(applied_filters),
                len(tools),
                len(filtered_tools),
            )

            return filtered_tools

        except Exception as e:
            logger.error("Error applying compatibility filters: %s", e)
            return tools  # Return original list on error

    def apply_custom_filters(self, tools: List[Any], filter_conditions: List[FilterCondition]) -> FilterResult:
        """
        Apply custom filter conditions.

        Args:
            tools: List of tools to filter
            filter_conditions: List of custom filter conditions

        Returns:
            FilterResult with detailed filtering information
        """
        import time

        start_time = time.time()

        if not tools or not filter_conditions:
            return FilterResult(
                passed_tools=tools,
                filtered_out=[],
                filter_statistics={},
                applied_filters=[],
                execution_time=time.time() - start_time,
            )

        passed_tools = []
        filtered_out = []
        filter_statistics: Dict[str, int] = {}
        applied_filters = []

        for tool in tools:
            passed = True
            failed_conditions = []

            for condition in filter_conditions:
                if not condition.evaluate(tool):
                    passed = False
                    failed_conditions.append(f"{condition.field_path} {condition.operator.value} {condition.value}")

            if passed:
                passed_tools.append(tool)
            else:
                filtered_out.append(tool)
                # Track which conditions caused filtering
                for failed in failed_conditions:
                    filter_statistics[failed] = filter_statistics.get(failed, 0) + 1

        # Build applied filters list
        applied_filters = [f"{cond.field_path} {cond.operator.value} {cond.value}" for cond in filter_conditions]

        execution_time = time.time() - start_time

        logger.debug(
            "Custom filters applied in %.3fs. %d conditions, %d tools passed, %d filtered out",
            execution_time,
            len(filter_conditions),
            len(passed_tools),
            len(filtered_out),
        )

        return FilterResult(
            passed_tools=passed_tools,
            filtered_out=filtered_out,
            filter_statistics=filter_statistics,
            applied_filters=applied_filters,
            execution_time=execution_time,
        )

    # Helper methods for extracting tool information

    def _get_parameter_count(self, tool: Any) -> int:
        """Get total parameter count from tool."""
        if hasattr(tool, "parameter_count"):
            return tool.parameter_count
        elif hasattr(tool, "parameters"):
            return len(tool.parameters) if tool.parameters else 0
        return 0

    def _get_required_parameter_count(self, tool: Any) -> int:
        """Get required parameter count from tool."""
        if hasattr(tool, "required_parameter_count"):
            return tool.required_parameter_count
        elif hasattr(tool, "parameters"):
            return sum(1 for p in (tool.parameters or []) if getattr(p, "required", False))
        return 0

    def _get_optional_parameter_count(self, tool: Any) -> int:
        """Get optional parameter count from tool."""
        total = self._get_parameter_count(tool)
        required = self._get_required_parameter_count(tool)
        return total - required

    def _get_complexity_score(self, tool: Any) -> float:
        """Get complexity score from tool."""
        if hasattr(tool, "complexity_score"):
            return float(tool.complexity_score)
        return 0.5  # Default neutral complexity

    def _has_required_parameter_types(self, tool: Any, required_types: List[str]) -> bool:
        """Check if tool has all required parameter types."""
        if not hasattr(tool, "parameters") or not tool.parameters:
            return False

        tool_types = {getattr(p, "type", "").lower() for p in tool.parameters}
        required_types_set = {t.lower() for t in required_types}

        return required_types_set.issubset(tool_types)

    def _has_forbidden_parameter_types(self, tool: Any, forbidden_types: List[str]) -> bool:
        """Check if tool has any forbidden parameter types."""
        if not hasattr(tool, "parameters") or not tool.parameters:
            return False

        tool_types = {getattr(p, "type", "").lower() for p in tool.parameters}
        forbidden_types_set = {t.lower() for t in forbidden_types}

        return bool(tool_types & forbidden_types_set)

    def _matches_parameter_name_patterns(self, tool: Any, patterns: List[str]) -> bool:
        """Check if tool parameters match any of the name patterns."""
        if not hasattr(tool, "parameters") or not tool.parameters:
            return False

        tool_names = [getattr(p, "name", "").lower() for p in tool.parameters]

        for pattern in patterns:
            for name in tool_names:
                if re.search(pattern.lower(), name):
                    return True

        return False

    def _has_required_parameter_names(self, tool: Any, required_names: List[str]) -> bool:
        """Check if tool has all required parameter names."""
        if not hasattr(tool, "parameters") or not tool.parameters:
            return False

        tool_names = {getattr(p, "name", "").lower() for p in tool.parameters}
        required_names_set = {name.lower() for name in required_names}

        return required_names_set.issubset(tool_names)

    def _has_forbidden_parameter_names(self, tool: Any, forbidden_names: List[str]) -> bool:
        """Check if tool has any forbidden parameter names."""
        if not hasattr(tool, "parameters") or not tool.parameters:
            return False

        tool_names = {getattr(p, "name", "").lower() for p in tool.parameters}
        forbidden_names_set = {name.lower() for name in forbidden_names}

        return bool(tool_names & forbidden_names_set)

    def _has_nested_objects(self, tool: Any) -> bool:
        """Check if tool has nested object parameters."""
        if not hasattr(tool, "parameters") or not tool.parameters:
            return False

        return any(getattr(p, "type", "").lower() in ["object", "dict"] for p in tool.parameters)

    def _has_arrays(self, tool: Any) -> bool:
        """Check if tool has array parameters."""
        if not hasattr(tool, "parameters") or not tool.parameters:
            return False

        return any(getattr(p, "type", "").lower() in ["array", "list"] for p in tool.parameters)

    def _check_input_compatibility(self, tool: Any, input_types: List[str]) -> bool:
        """Check if tool is compatible with input types."""
        # Simplified compatibility check - in practice would use TypeCompatibilityAnalyzer
        if not hasattr(tool, "parameters") or not tool.parameters:
            return True  # No parameters means any input is compatible

        tool_input_types = {getattr(p, "type", "").lower() for p in tool.parameters if getattr(p, "required", False)}

        if not tool_input_types:
            return True  # No required parameters

        # Check if any required input type is compatible
        input_types_set = {t.lower() for t in input_types}
        return bool(tool_input_types & input_types_set)

    def _check_output_compatibility(self, tool: Any, output_types: List[str]) -> bool:
        """Check if tool produces compatible output types."""
        # In practice, this would analyze return types from tool metadata
        # For now, assume tools with certain names/categories produce certain outputs
        tool_name = getattr(tool, "name", "").lower()
        tool_category = getattr(tool, "category", "").lower()

        output_types_lower = [t.lower() for t in output_types]

        # Simple heuristics - in practice would be more sophisticated
        if "object" in output_types_lower or "dict" in output_types_lower:
            if any(word in tool_name for word in ["search", "get", "find", "list"]):
                return True
            if tool_category in ["search", "analysis", "data"]:
                return True

        if "string" in output_types_lower:
            if any(word in tool_name for word in ["read", "cat", "echo", "print"]):
                return True

        return True  # Default to compatible

    def _supports_chaining(self, tool: Any) -> bool:
        """Check if tool supports chaining with other tools."""
        # Heuristic: tools that take standard input/output types support chaining
        if not hasattr(tool, "parameters"):
            return True

        # Check for standard chainable parameter patterns
        param_names = [getattr(p, "name", "").lower() for p in (tool.parameters or [])]
        chainable_patterns = ["input", "data", "content", "text", "file"]

        return any(pattern in name for pattern in chainable_patterns for name in param_names)

    def _meets_integration_level(self, tool: Any, level: IntegrationLevel) -> bool:
        """Check if tool meets integration level requirements."""
        complexity = self._get_complexity_score(tool)
        param_count = self._get_parameter_count(tool)

        if level == IntegrationLevel.BASIC:
            return complexity <= 0.5 and param_count <= 3
        elif level == IntegrationLevel.INTERMEDIATE:
            return complexity <= 0.8 and param_count <= 6
        else:  # ADVANCED
            return True  # No restrictions for advanced level

    def _get_compatibility_score(self, tool: Any) -> float:
        """Get compatibility score for tool."""
        # In practice, this would be calculated by TypeCompatibilityAnalyzer
        if hasattr(tool, "compatibility_score"):
            return tool.compatibility_score
        elif hasattr(tool, "relevance_score"):
            return tool.relevance_score
        return 0.7  # Default neutral score

    def _has_required_capabilities(self, tool: Any, capabilities: List[str]) -> bool:
        """Check if tool has required capabilities."""
        tool_description = getattr(tool, "description", "").lower()
        tool_category = getattr(tool, "category", "").lower()
        tool_name = getattr(tool, "name", "").lower()

        tool_text = f"{tool_name} {tool_description} {tool_category}"

        return all(cap.lower() in tool_text for cap in capabilities)

    def _has_forbidden_capabilities(self, tool: Any, capabilities: List[str]) -> bool:
        """Check if tool has forbidden capabilities."""
        tool_description = getattr(tool, "description", "").lower()
        tool_category = getattr(tool, "category", "").lower()
        tool_name = getattr(tool, "name", "").lower()

        tool_text = f"{tool_name} {tool_description} {tool_category}"

        return any(cap.lower() in tool_text for cap in capabilities)
