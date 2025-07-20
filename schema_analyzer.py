#!/usr/bin/env python3
"""
Schema Analyzer

This module provides deep JSON schema parsing and analysis capabilities for
MCP tool parameters, extracting types, constraints, validation rules, and
parameter relationships.
"""

import re
from typing import Any, Dict, List, Optional

from logging_config import get_logger
from mcp_metadata_types import ParameterAnalysis

logger = get_logger(__name__)


class SchemaAnalyzer:
    """Analyze JSON schemas to extract parameter information and constraints."""

    # Complexity factor weights for scoring
    COMPLEXITY_FACTORS = {
        "parameter_count": {"weight": 0.3, "threshold_simple": 3, "threshold_complex": 8},
        "required_parameters": {"weight": 0.2, "threshold_simple": 1, "threshold_complex": 5},
        "schema_depth": {"weight": 0.2, "threshold_simple": 2, "threshold_complex": 4},
        "type_complexity": {"weight": 0.2, "multiplier_object": 2.0, "multiplier_array": 1.5},
        "constraints_count": {"weight": 0.1, "threshold_simple": 2, "threshold_complex": 5},
    }

    def __init__(self):
        """Initialize the schema analyzer."""
        logger.info("Initialized Schema Analyzer")

    def analyze_schema(self, schema: Dict[str, Any]) -> List[ParameterAnalysis]:
        """
        Analyze a JSON schema and extract parameter information.

        Args:
            schema: JSON schema dictionary

        Returns:
            List of ParameterAnalysis objects with detailed parameter metadata
        """
        if not schema:
            return []

        if isinstance(schema, dict):
            logger.debug("Analyzing JSON schema with keys: %s", list(schema.keys()))
        else:
            logger.debug("Analyzing schema with %d items", len(schema) if hasattr(schema, "__len__") else 0)

        parameters = []

        # Handle direct parameter list format (from existing system tools)
        if isinstance(schema, list):
            for param_def in schema:
                if isinstance(param_def, dict) and "name" in param_def:
                    analysis = self._analyze_parameter_from_definition(param_def)
                    parameters.append(analysis)

        # Handle object schema with properties
        elif isinstance(schema, dict) and schema.get("type") == "object" and "properties" in schema:
            required_fields = set(schema.get("required", []))

            for param_name, param_schema in schema["properties"].items():
                analysis = self._analyze_single_parameter(param_name, param_schema, param_name in required_fields)
                parameters.append(analysis)

        # Handle dictionary format
        elif isinstance(schema, dict) and not schema.get("type"):
            # Assume it's a flat parameter definition dictionary
            for param_name, param_info in schema.items():
                if isinstance(param_info, dict):
                    analysis = self._analyze_single_parameter(param_name, param_info, False)
                else:
                    # Simple type definition
                    analysis = ParameterAnalysis(
                        name=param_name,
                        type="string",
                        required=False,
                        description=f"Parameter: {param_name}",
                        constraints={},
                        default_value=None,
                        examples=[],
                        complexity_score=0.1,
                    )
                parameters.append(analysis)

        # Calculate complexity scores for all parameters
        for param in parameters:
            param.complexity_score = self.calculate_complexity_score(param)

        logger.debug("Analyzed %d parameters from schema", len(parameters))
        return parameters

    def _analyze_single_parameter(self, name: str, param_schema: Dict[str, Any], required: bool) -> ParameterAnalysis:
        """
        Analyze a single parameter schema.

        Args:
            name: Parameter name
            param_schema: Parameter schema dictionary
            required: Whether the parameter is required

        Returns:
            ParameterAnalysis object with parameter metadata
        """
        param_type = param_schema.get("type", "string")
        description = param_schema.get("description", "")
        default_value = param_schema.get("default")

        # Extract constraints from schema
        constraints = self.extract_constraints(param_schema)

        # Generate examples based on type and constraints
        examples = self._generate_parameter_examples(name, param_type, param_schema)

        return ParameterAnalysis(
            name=name,
            type=param_type,
            required=required,
            description=description,
            constraints=constraints,
            default_value=default_value,
            examples=examples,
            complexity_score=0.0,  # Will be calculated later
        )

    def _analyze_parameter_from_definition(self, param_def: Dict[str, Any]) -> ParameterAnalysis:
        """
        Analyze parameter from the existing system tools definition format.

        Args:
            param_def: Parameter definition dictionary

        Returns:
            ParameterAnalysis object with parameter metadata
        """
        name = param_def.get("name", "unknown")
        param_type = param_def.get("type", "string")
        required = param_def.get("required", False)
        description = param_def.get("description", "")
        default_value = param_def.get("default_value")

        # Convert to constraints format
        constraints = {}
        if "schema" in param_def:
            constraints.update(param_def["schema"])

        # Generate examples
        examples = self._generate_parameter_examples(name, param_type, param_def)

        return ParameterAnalysis(
            name=name,
            type=param_type,
            required=required,
            description=description,
            constraints=constraints,
            default_value=default_value,
            examples=examples,
            complexity_score=0.0,  # Will be calculated later
        )

    def extract_constraints(self, param_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract constraints from a parameter schema.

        Args:
            param_schema: Parameter schema dictionary

        Returns:
            Dictionary of constraints
        """
        constraints = {}

        # String constraints
        for key in ["minLength", "maxLength", "pattern", "format", "enum"]:
            if key in param_schema:
                constraints[key] = param_schema[key]

        # Number constraints
        for key in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"]:
            if key in param_schema:
                constraints[key] = param_schema[key]

        # Array constraints
        for key in ["minItems", "maxItems", "uniqueItems", "items"]:
            if key in param_schema:
                constraints[key] = param_schema[key]

        # Object constraints
        for key in ["properties", "required", "additionalProperties", "patternProperties"]:
            if key in param_schema:
                constraints[key] = param_schema[key]

        return constraints

    def calculate_complexity_score(self, parameter: ParameterAnalysis) -> float:
        """
        Calculate complexity score for a parameter.

        Args:
            parameter: ParameterAnalysis object

        Returns:
            Complexity score from 0.0 (simple) to 1.0 (very complex)
        """
        score = 0.0

        # Base type complexity
        type_scores = {
            "string": 0.1,
            "number": 0.2,
            "integer": 0.2,
            "boolean": 0.1,
            "array": 0.6,
            "object": 0.8,
        }
        score += type_scores.get(parameter.type, 0.3)

        # Required parameters are slightly more complex
        if parameter.required:
            score += 0.1

        # Constraint complexity
        constraint_count = len(parameter.constraints)
        if constraint_count > 0:
            score += min(constraint_count * 0.1, 0.3)

        # Nested structure complexity
        if "items" in parameter.constraints:
            items_schema = parameter.constraints["items"]
            if isinstance(items_schema, dict) and items_schema.get("type") == "object":
                score += 0.2

        if "properties" in parameter.constraints:
            nested_props = parameter.constraints["properties"]
            if isinstance(nested_props, dict):
                score += min(len(nested_props) * 0.05, 0.2)

        # Pattern constraints add complexity
        if "pattern" in parameter.constraints:
            score += 0.15

        # Enum constraints can be complex
        if "enum" in parameter.constraints:
            enum_values = parameter.constraints["enum"]
            if isinstance(enum_values, list) and len(enum_values) > 5:
                score += 0.1

        # Cap at 1.0
        return min(score, 1.0)

    def _generate_parameter_examples(self, name: str, param_type: str, param_schema: Dict[str, Any]) -> List[str]:
        """
        Generate example values for a parameter based on its schema.

        Args:
            name: Parameter name
            param_type: Parameter type
            param_schema: Parameter schema dictionary

        Returns:
            List of example values as strings
        """
        examples = []
        name_lower = name.lower()
        description = param_schema.get("description", "").lower()

        # Check for enum values first
        if "enum" in param_schema:
            enum_values = param_schema["enum"]
            examples.extend([str(v) for v in enum_values[:3]])
            return examples

        if param_type == "string":
            if "path" in name_lower or "file" in name_lower:
                examples.extend(["/Users/user/file.txt", "/home/user/document.py", "./src/main.js"])
            elif "url" in name_lower:
                examples.extend(["https://example.com", "https://api.github.com/users"])
            elif "command" in name_lower:
                examples.extend(["ls -la", "python script.py", "npm install"])
            elif "pattern" in name_lower:
                examples.extend([r"function\s+\w+", r"class.*:", r"\berror\b"])
            elif "query" in name_lower:
                examples.extend(["search term", "function definition", "class implementation"])
            else:
                examples.extend(["example_value", "sample_text", "user_input"])

        elif param_type in ["number", "integer"]:
            # Check constraints for realistic ranges
            minimum = param_schema.get("minimum", 0)
            maximum = param_schema.get("maximum", 1000)

            if "timeout" in description:
                examples.extend(["5000", "10000", "30000"])
            elif "limit" in description or "count" in description:
                examples.extend(["10", "50", "100"])
            elif "port" in name_lower:
                examples.extend(["8080", "3000", "5432"])
            else:
                # Generate examples within constraints
                if maximum <= 10:
                    examples.extend(["1", "5", str(maximum)])
                else:
                    examples.extend([str(minimum), str((minimum + maximum) // 2), str(maximum)])

        elif param_type == "boolean":
            examples.extend(["true", "false"])

        elif param_type == "array":
            if "pattern" in description or "glob" in description:
                examples.extend(['["*.py", "*.js"]', '["src/**/*"]', '["test_*.py"]'])
            elif "file" in description:
                examples.extend(['["file1.txt", "file2.txt"]', '["*.log"]'])
            else:
                examples.extend(['["item1", "item2"]', '["value"]', "[]"])

        elif param_type == "object":
            if "edit" in name_lower:
                examples.extend(['{"old_string": "old", "new_string": "new"}'])
            else:
                examples.extend(['{"key": "value"}', "{}"])

        return examples[:3]  # Limit to 3 examples
