#!/usr/bin/env python3
"""
MCP Metadata Types

This module contains all the data type definitions for MCP tool metadata
to avoid circular imports between the various analyzer modules.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ParameterAnalysis:
    """Detailed analysis of a tool parameter."""

    name: str
    type: str
    required: bool
    description: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    default_value: Optional[Any] = None
    examples: List[Any] = field(default_factory=list)
    complexity_score: float = 0.0  # 0.0 = simple, 1.0 = complex

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "type": self.type,
            "required": self.required,
            "description": self.description,
            "constraints": self.constraints,
            "default_value": self.default_value,
            "examples": self.examples,
            "complexity_score": self.complexity_score,
        }


@dataclass
class UsagePattern:
    """Identified usage pattern for a tool."""

    pattern_name: str
    description: str
    parameter_combination: List[str]
    use_case: str
    complexity_level: str  # 'basic', 'intermediate', 'advanced'
    example_code: str
    success_probability: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pattern_name": self.pattern_name,
            "description": self.description,
            "parameter_combination": self.parameter_combination,
            "use_case": self.use_case,
            "complexity_level": self.complexity_level,
            "example_code": self.example_code,
            "success_probability": self.success_probability,
        }


@dataclass
class ComplexityAnalysis:
    """Analysis of tool complexity based on parameters and structure."""

    total_parameters: int
    required_parameters: int
    optional_parameters: int
    complex_parameters: int
    overall_complexity: float  # 0.0 = simple, 1.0 = very complex
    complexity_factors: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_parameters": self.total_parameters,
            "required_parameters": self.required_parameters,
            "optional_parameters": self.optional_parameters,
            "complex_parameters": self.complex_parameters,
            "overall_complexity": self.overall_complexity,
            "complexity_factors": self.complexity_factors,
        }


@dataclass
class ToolExample:
    """Represents a tool usage example."""

    use_case: str
    example_call: str
    expected_output: str = ""
    context: str = ""
    language: str = "python"
    effectiveness_score: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "use_case": self.use_case,
            "example_call": self.example_call,
            "expected_output": self.expected_output,
            "context": self.context,
            "language": self.language,
            "effectiveness_score": self.effectiveness_score,
        }


@dataclass
class CodeExample:
    """Represents a code example extracted from documentation."""

    language: str
    code: str
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "language": self.language,
            "code": self.code,
            "description": self.description,
        }


@dataclass
class DocumentationAnalysis:
    """Analysis of tool documentation and docstrings."""

    description: str
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    examples: List[ToolExample] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    best_practices: List[str] = field(default_factory=list)
    return_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "description": self.description,
            "parameters": self.parameters,
            "examples": [ex.to_dict() for ex in self.examples],
            "notes": self.notes,
            "warnings": self.warnings,
            "best_practices": self.best_practices,
            "return_info": self.return_info,
        }


@dataclass
class MCPToolMetadata:
    """Comprehensive metadata extracted from a tool definition."""

    name: str
    description: str
    category: str
    parameters: List[ParameterAnalysis] = field(default_factory=list)
    examples: List[ToolExample] = field(default_factory=list)
    usage_patterns: List[UsagePattern] = field(default_factory=list)
    complexity_analysis: Optional[ComplexityAnalysis] = None
    documentation_analysis: Optional[DocumentationAnalysis] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "parameters": [p.to_dict() for p in self.parameters],
            "examples": [ex.to_dict() for ex in self.examples],
            "usage_patterns": [up.to_dict() for up in self.usage_patterns],
            "complexity_analysis": (self.complexity_analysis.to_dict() if self.complexity_analysis else None),
            "documentation_analysis": (self.documentation_analysis.to_dict() if self.documentation_analysis else None),
        }


@dataclass
class ExampleValidationResult:
    """Result of validating a generated example."""

    is_valid: bool
    error_message: str = ""
    suggested_fix: str = ""
