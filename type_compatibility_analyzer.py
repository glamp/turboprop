#!/usr/bin/env python3
"""
Type Compatibility Analyzer

This module provides sophisticated type compatibility analysis for parameter-aware
tool search, including type hierarchies, conversion chains, and compatibility scoring.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from logging_config import get_logger

logger = get_logger(__name__)


class CompatibilityLevel(Enum):
    """Levels of type compatibility."""

    EXACT = "exact"
    COMPATIBLE = "compatible"
    CONVERTIBLE = "convertible"
    INCOMPATIBLE = "incompatible"


@dataclass
class ConversionStep:
    """A single step in a type conversion chain."""

    from_type: str
    to_type: str
    method: str  # 'cast', 'parse', 'transform', 'extract'
    confidence: float  # 0.0 - 1.0
    description: str
    complexity: int = 1  # Number of operations required


@dataclass
class ConversionChain:
    """A complete type conversion chain."""

    source_type: str
    target_type: str
    steps: List[ConversionStep] = field(default_factory=list)
    overall_confidence: float = 0.0
    total_complexity: int = 0
    reliability_score: float = 0.0

    def __post_init__(self):
        """Calculate derived metrics."""
        if self.steps:
            self.overall_confidence = min(step.confidence for step in self.steps)
            self.total_complexity = sum(step.complexity for step in self.steps)
            self.reliability_score = self.overall_confidence / max(1, self.total_complexity * 0.1)


@dataclass
class TypeCompatibilityResult:
    """Result of type compatibility analysis."""

    is_compatible: bool
    compatibility_score: float  # 0.0 - 1.0
    compatibility_level: CompatibilityLevel
    direct_match: bool
    conversion_required: bool
    conversion_steps: List[str] = field(default_factory=list)
    compatibility_explanation: str = ""
    confidence: float = 0.0
    suggested_alternatives: List[str] = field(default_factory=list)


class TypeCompatibilityAnalyzer:
    """Analyze type compatibility between tools."""

    def __init__(self):
        """Initialize the type compatibility analyzer."""
        self.type_hierarchy = self._build_type_hierarchy()
        self.conversion_rules = self._load_conversion_rules()
        self.compatibility_cache = {}  # Cache for performance

        logger.info("Initialized TypeCompatibilityAnalyzer with %d base types", len(self.type_hierarchy))

    def analyze_type_compatibility(self, source_type: str, target_type: str) -> TypeCompatibilityResult:
        """
        Analyze compatibility between two types.

        Args:
            source_type: Source data type
            target_type: Target data type

        Returns:
            TypeCompatibilityResult with detailed compatibility analysis
        """
        # Normalize type names
        source_normalized = self._normalize_type_name(source_type)
        target_normalized = self._normalize_type_name(target_type)

        # Check cache first
        cache_key = f"{source_normalized}:{target_normalized}"
        if cache_key in self.compatibility_cache:
            return self.compatibility_cache[cache_key]

        try:
            result = self._perform_compatibility_analysis(source_normalized, target_normalized)

            # Cache result
            self.compatibility_cache[cache_key] = result

            logger.debug(
                "Type compatibility %s -> %s: %s (score=%.2f)",
                source_type,
                target_type,
                result.compatibility_level.value,
                result.compatibility_score,
            )

            return result

        except Exception as e:
            logger.error("Error analyzing type compatibility %s -> %s: %s", source_type, target_type, e)
            return TypeCompatibilityResult(
                is_compatible=False,
                compatibility_score=0.0,
                compatibility_level=CompatibilityLevel.INCOMPATIBLE,
                direct_match=False,
                conversion_required=False,
                compatibility_explanation=f"Analysis failed: {str(e)}",
                confidence=0.0,
            )

    def find_type_conversion_chain(self, source_types: List[str], target_types: List[str]) -> List[ConversionChain]:
        """
        Find possible type conversion chains.

        Args:
            source_types: List of source types
            target_types: List of target types

        Returns:
            List of viable conversion chains, sorted by reliability
        """
        conversion_chains = []

        for source_type in source_types:
            for target_type in target_types:
                chain = self._find_single_conversion_chain(source_type, target_type)
                if chain and chain.reliability_score > 0.1:  # Minimum viability threshold
                    conversion_chains.append(chain)

        # Sort by reliability score (descending)
        conversion_chains.sort(key=lambda x: x.reliability_score, reverse=True)

        logger.debug(
            "Found %d conversion chains from %d source types to %d target types",
            len(conversion_chains),
            len(source_types),
            len(target_types),
        )

        return conversion_chains[:10]  # Return top 10 chains

    def get_type_compatibility_matrix(self, types: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Generate compatibility matrix for a set of types.

        Args:
            types: List of types to analyze

        Returns:
            Matrix of compatibility scores
        """
        matrix: Dict[str, Dict[str, float]] = {}

        for source_type in types:
            matrix[source_type] = {}
            for target_type in types:
                if source_type == target_type:
                    matrix[source_type][target_type] = 1.0
                else:
                    result = self.analyze_type_compatibility(source_type, target_type)
                    matrix[source_type][target_type] = result.compatibility_score

        return matrix

    def suggest_compatible_types(self, target_type: str, limit: int = 5) -> List[Tuple[str, float]]:
        """
        Suggest types that are compatible with the target type.

        Args:
            target_type: Target type to find compatible types for
            limit: Maximum number of suggestions

        Returns:
            List of (type, compatibility_score) tuples
        """
        target_normalized = self._normalize_type_name(target_type)
        suggestions = []

        # Get all known types
        all_types = set()
        all_types.update(self.type_hierarchy.keys())
        for subtypes in self.type_hierarchy.values():
            all_types.update(subtypes)

        # Analyze compatibility with each type
        for source_type in all_types:
            if source_type != target_normalized:
                result = self.analyze_type_compatibility(source_type, target_normalized)
                if result.compatibility_score > 0.3:  # Minimum threshold
                    suggestions.append((source_type, result.compatibility_score))

        # Sort by compatibility score and return top suggestions
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:limit]

    def _build_type_hierarchy(self) -> Dict[str, List[str]]:
        """Build type inheritance and compatibility hierarchy."""
        return {
            # Core primitive types
            "string": [
                "text",
                "str",
                "path",
                "filepath",
                "url",
                "uri",
                "pattern",
                "regex",
                "format",
                "name",
                "id",
                "identifier",
                "email",
                "phone",
                "address",
                "description",
                "title",
            ],
            "number": [
                "int",
                "integer",
                "float",
                "double",
                "decimal",
                "timeout",
                "duration",
                "limit",
                "count",
                "size",
                "length",
                "width",
                "height",
                "weight",
                "distance",
                "percentage",
                "ratio",
                "score",
                "rating",
            ],
            "boolean": [
                "bool",
                "flag",
                "enable",
                "disable",
                "switch",
                "toggle",
                "active",
                "visible",
                "enabled",
                "required",
            ],
            # Complex types
            "object": [
                "dict",
                "dictionary",
                "map",
                "config",
                "configuration",
                "options",
                "settings",
                "params",
                "parameters",
                "metadata",
                "properties",
                "attributes",
                "struct",
            ],
            "array": [
                "list",
                "collection",
                "sequence",
                "items",
                "elements",
                "set",
                "tuple",
                "vector",
                "series",
                "batch",
            ],
            # File and I/O types
            "file": ["path", "filepath", "filename", "directory", "folder", "stream", "buffer", "content", "data"],
            "binary": ["bytes", "blob", "buffer", "stream", "file_content", "image", "video", "audio", "document"],
            # Time and date types
            "datetime": ["date", "time", "timestamp", "duration", "interval", "period", "schedule"],
            # Network and communication types
            "network": ["url", "uri", "endpoint", "host", "port", "protocol", "scheme", "domain"],
        }

    def _load_conversion_rules(self) -> Dict[str, Dict[str, ConversionStep]]:
        """Load type conversion rules."""
        return {
            "string": {
                "number": ConversionStep("string", "number", "parse", 0.8, "Parse string as number"),
                "boolean": ConversionStep("string", "boolean", "parse", 0.9, "Parse string as boolean"),
                "array": ConversionStep("string", "array", "split", 0.7, "Split string into array"),
                "object": ConversionStep("string", "object", "parse_json", 0.6, "Parse JSON string"),
                "path": ConversionStep("string", "path", "cast", 0.95, "Use string as path"),
                "url": ConversionStep("string", "url", "cast", 0.95, "Use string as URL"),
            },
            "number": {
                "string": ConversionStep("number", "string", "cast", 0.95, "Convert number to string"),
                "boolean": ConversionStep("number", "boolean", "cast", 0.9, "Convert number to boolean"),
                "int": ConversionStep("number", "int", "cast", 0.9, "Convert to integer"),
                "float": ConversionStep("number", "float", "cast", 0.95, "Convert to float"),
            },
            "boolean": {
                "string": ConversionStep("boolean", "string", "cast", 0.95, "Convert boolean to string"),
                "number": ConversionStep("boolean", "number", "cast", 0.9, "Convert boolean to number"),
                "int": ConversionStep("boolean", "int", "cast", 0.9, "Convert boolean to 0/1"),
            },
            "array": {
                "string": ConversionStep("array", "string", "join", 0.8, "Join array elements"),
                "object": ConversionStep("array", "object", "to_dict", 0.6, "Convert array to object"),
                "set": ConversionStep("array", "set", "cast", 0.95, "Convert to set"),
            },
            "object": {
                "string": ConversionStep("object", "string", "serialize", 0.8, "Serialize object to string"),
                "array": ConversionStep("object", "array", "values", 0.7, "Extract object values"),
                "dict": ConversionStep("object", "dict", "cast", 0.95, "Convert to dictionary"),
            },
            "path": {
                "string": ConversionStep("path", "string", "cast", 0.95, "Convert path to string"),
                "file": ConversionStep("path", "file", "cast", 0.9, "Use path as file reference"),
            },
        }

    def _normalize_type_name(self, type_name: str) -> str:
        """Normalize type name for consistent processing."""
        if not isinstance(type_name, str):
            return "unknown"

        normalized = type_name.lower().strip()

        # Handle common aliases
        aliases = {
            "str": "string",
            "int": "number",
            "float": "number",
            "double": "number",
            "bool": "boolean",
            "dict": "object",
            "list": "array",
        }

        return aliases.get(normalized, normalized)

    def _perform_compatibility_analysis(self, source_type: str, target_type: str) -> TypeCompatibilityResult:
        """Perform detailed compatibility analysis between two types."""
        # Direct match
        if source_type == target_type:
            return TypeCompatibilityResult(
                is_compatible=True,
                compatibility_score=1.0,
                compatibility_level=CompatibilityLevel.EXACT,
                direct_match=True,
                conversion_required=False,
                compatibility_explanation=f"Exact type match: {source_type}",
                confidence=1.0,
            )

        # Check hierarchy compatibility
        hierarchy_result = self._check_hierarchy_compatibility(source_type, target_type)
        if hierarchy_result:
            return hierarchy_result

        # Check conversion possibilities
        conversion_result = self._check_conversion_compatibility(source_type, target_type)
        if conversion_result:
            return conversion_result

        # No compatibility found
        return TypeCompatibilityResult(
            is_compatible=False,
            compatibility_score=0.0,
            compatibility_level=CompatibilityLevel.INCOMPATIBLE,
            direct_match=False,
            conversion_required=False,
            compatibility_explanation=f"No compatibility found between {source_type} and {target_type}",
            confidence=1.0,
            suggested_alternatives=self._suggest_alternative_types(source_type, target_type),
        )

    def _check_hierarchy_compatibility(self, source_type: str, target_type: str) -> Optional[TypeCompatibilityResult]:
        """Check compatibility based on type hierarchy."""
        # Check if source is a subtype of target's hierarchy
        for base_type, subtypes in self.type_hierarchy.items():
            if target_type == base_type and source_type in subtypes:
                return TypeCompatibilityResult(
                    is_compatible=True,
                    compatibility_score=0.9,
                    compatibility_level=CompatibilityLevel.COMPATIBLE,
                    direct_match=False,
                    conversion_required=False,
                    compatibility_explanation=f"{source_type} is compatible with {target_type} (subtype relationship)",
                    confidence=0.9,
                )
            elif source_type == base_type and target_type in subtypes:
                return TypeCompatibilityResult(
                    is_compatible=True,
                    compatibility_score=0.85,
                    compatibility_level=CompatibilityLevel.COMPATIBLE,
                    direct_match=False,
                    conversion_required=False,
                    compatibility_explanation=f"{source_type} is compatible with {target_type} (supertype relationship)",
                    confidence=0.85,
                )
            elif source_type in subtypes and target_type in subtypes:
                return TypeCompatibilityResult(
                    is_compatible=True,
                    compatibility_score=0.8,
                    compatibility_level=CompatibilityLevel.COMPATIBLE,
                    direct_match=False,
                    conversion_required=False,
                    compatibility_explanation=f"{source_type} and {target_type} are both subtypes of {base_type}",
                    confidence=0.8,
                )

        return None

    def _check_conversion_compatibility(self, source_type: str, target_type: str) -> Optional[TypeCompatibilityResult]:
        """Check compatibility based on conversion rules."""
        if source_type in self.conversion_rules:
            if target_type in self.conversion_rules[source_type]:
                conversion_step = self.conversion_rules[source_type][target_type]
                return TypeCompatibilityResult(
                    is_compatible=True,
                    compatibility_score=conversion_step.confidence * 0.7,  # Reduce score for conversion
                    compatibility_level=CompatibilityLevel.CONVERTIBLE,
                    direct_match=False,
                    conversion_required=True,
                    conversion_steps=[conversion_step.description],
                    compatibility_explanation=f"{source_type} can be converted to {target_type}: {conversion_step.description}",
                    confidence=conversion_step.confidence,
                )

        return None

    def _find_single_conversion_chain(self, source_type: str, target_type: str) -> Optional[ConversionChain]:
        """Find conversion chain between two specific types."""
        # Direct conversion
        if source_type in self.conversion_rules and target_type in self.conversion_rules[source_type]:
            step = self.conversion_rules[source_type][target_type]
            return ConversionChain(source_type=source_type, target_type=target_type, steps=[step])

        # Multi-step conversion (limited to 2 steps for performance)
        for intermediate_type in self.conversion_rules.get(source_type, {}):
            if target_type in self.conversion_rules.get(intermediate_type, {}):
                step1 = self.conversion_rules[source_type][intermediate_type]
                step2 = self.conversion_rules[intermediate_type][target_type]

                return ConversionChain(source_type=source_type, target_type=target_type, steps=[step1, step2])

        return None

    def _suggest_alternative_types(self, source_type: str, target_type: str) -> List[str]:
        """Suggest alternative types that might work better."""
        suggestions = []

        # Find types that source can convert to
        if source_type in self.conversion_rules:
            convertible_types = list(self.conversion_rules[source_type].keys())
            suggestions.extend(convertible_types[:3])  # Top 3

        # Find types in the same hierarchy as target
        for base_type, subtypes in self.type_hierarchy.items():
            if target_type == base_type or target_type in subtypes:
                # Suggest other types in the same hierarchy
                hierarchy_types = [base_type] + subtypes
                suggestions.extend([t for t in hierarchy_types if t != target_type][:2])

        return list(set(suggestions))[:5]  # Remove duplicates and limit to 5
