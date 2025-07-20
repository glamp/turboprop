#!/usr/bin/env python3
"""
Schema Analyzer

This module provides deep JSON schema parsing and analysis capabilities for
MCP tool parameters, extracting types, constraints, validation rules, and
parameter relationships.
"""

from typing import Any, Dict, List

from logging_config import get_logger
from mcp_metadata_types import ParameterAnalysis

logger = get_logger(__name__)


class SchemaAnalyzer:
    """Analyze JSON schemas to extract parameter information and constraints."""

    # Configuration constants
    DEFAULT_COMPLEXITY_SCORE = 0.5
    SIMPLE_COMPLEXITY_SCORE = 0.1
    NO_COMPLEXITY_SCORE = 0.0
    MAX_COMPLEXITY_SCORE = 1.0
    SIMPLE_SCHEMA_DEPTH_THRESHOLD = 2
    COMPLEX_SCHEMA_DEPTH_THRESHOLD = 4

    # Default complexity scores by type
    TYPE_BASE_COMPLEXITY = {
        "string": 0.1,
        "number": 0.2,
        "integer": 0.2,
        "boolean": 0.1,
        "array": 0.6,
        "object": 0.8,
    }

    # Complexity modifiers
    REQUIRED_PARAMETER_COMPLEXITY_BONUS = 0.1
    CONSTRAINT_COMPLEXITY_MULTIPLIER = 0.1
    MAX_CONSTRAINT_COMPLEXITY_BONUS = 0.3
    NESTED_OBJECT_COMPLEXITY_BONUS = 0.2
    PROPERTY_COMPLEXITY_MULTIPLIER = 0.05
    MAX_PROPERTY_COMPLEXITY_BONUS = 0.2
    PATTERN_CONSTRAINT_COMPLEXITY_BONUS = 0.15
    LARGE_ENUM_COMPLEXITY_BONUS = 0.1
    LARGE_ENUM_THRESHOLD = 5

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

        try:
            if isinstance(schema, dict):
                logger.debug("Analyzing JSON schema with keys: %s", list(schema.keys()))
            else:
                logger.debug("Analyzing schema with %d items", len(schema) if hasattr(schema, "__len__") else 0)

            parameters = []

            # Handle direct parameter list format (from existing system tools)
            if isinstance(schema, list):
                for i, param_def in enumerate(schema):
                    try:
                        if isinstance(param_def, dict) and "name" in param_def:
                            analysis = self._analyze_parameter_from_definition(param_def)
                            parameters.append(analysis)
                        else:
                            logger.warning("Skipping invalid parameter definition at index %d: %s", i, param_def)
                    except Exception as e:
                        logger.error("Error processing parameter at index %d: %s", i, e)
                        continue

            # Handle object schema with properties
            elif isinstance(schema, dict) and schema.get("type") == "object" and "properties" in schema:
                try:
                    required_fields = set(schema.get("required", []))
                    properties = schema.get("properties", {})

                    if not isinstance(properties, dict):
                        logger.error("Schema properties is not a dictionary: %s", properties)
                        return []

                    for param_name, param_schema in properties.items():
                        try:
                            analysis = self._analyze_single_parameter(
                                param_name, param_schema, param_name in required_fields
                            )
                            parameters.append(analysis)
                        except Exception as e:
                            logger.error("Error processing parameter '%s': %s", param_name, e)
                            continue
                except Exception as e:
                    logger.error("Error processing object schema: %s", e)
                    return []

            # Handle dictionary format
            elif isinstance(schema, dict) and not schema.get("type"):
                # Assume it's a flat parameter definition dictionary
                for param_name, param_info in schema.items():
                    try:
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
                                complexity_score=self.SIMPLE_COMPLEXITY_SCORE,
                            )
                        parameters.append(analysis)
                    except Exception as e:
                        logger.error("Error processing parameter '%s': %s", param_name, e)
                        continue

            # Calculate complexity scores for all parameters
            for param in parameters:
                try:
                    param.complexity_score = self.calculate_complexity_score(param)
                except Exception as e:
                    logger.error("Error calculating complexity for parameter '%s': %s", param.name, e)
                    param.complexity_score = self.DEFAULT_COMPLEXITY_SCORE  # Fallback to medium complexity

            logger.debug("Analyzed %d parameters from schema", len(parameters))
            return parameters

        except Exception as e:
            logger.error("Critical error in schema analysis: %s", e)
            return []

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
        try:
            if not isinstance(param_schema, dict):
                logger.warning("Parameter schema for '%s' is not a dictionary: %s", name, param_schema)
                param_schema = {}

            param_type = param_schema.get("type", "string")
            description = param_schema.get("description", "")
            default_value = param_schema.get("default")

            # Extract constraints from schema
            try:
                constraints = self.extract_constraints(param_schema)
            except Exception as e:
                logger.error("Error extracting constraints for parameter '%s': %s", name, e)
                constraints = {}

            # Generate examples based on type and constraints
            try:
                examples = self._generate_parameter_examples(name, param_type, param_schema)
            except Exception as e:
                logger.error("Error generating examples for parameter '%s': %s", name, e)
                examples = []

            return ParameterAnalysis(
                name=name,
                type=param_type,
                required=required,
                description=description,
                constraints=constraints,
                default_value=default_value,
                examples=examples,
                complexity_score=self.NO_COMPLEXITY_SCORE,  # Will be calculated later
            )
        except Exception as e:
            logger.error("Critical error analyzing parameter '%s': %s", name, e)
            # Return a minimal valid parameter analysis
            return ParameterAnalysis(
                name=name,
                type="string",
                required=required,
                description=f"Error analyzing parameter: {name}",
                constraints={},
                default_value=None,
                examples=[],
                complexity_score=self.DEFAULT_COMPLEXITY_SCORE,
            )

    def _analyze_parameter_from_definition(self, param_def: Dict[str, Any]) -> ParameterAnalysis:
        """
        Analyze parameter from the existing system tools definition format.

        Args:
            param_def: Parameter definition dictionary

        Returns:
            ParameterAnalysis object with parameter metadata
        """
        try:
            if not isinstance(param_def, dict):
                logger.error("Parameter definition is not a dictionary: %s", param_def)
                raise ValueError("Invalid parameter definition format")

            name = param_def.get("name", "unknown")
            param_type = param_def.get("type", "string")
            required = param_def.get("required", False)
            description = param_def.get("description", "")
            default_value = param_def.get("default_value")

            # Convert to constraints format
            constraints = {}
            if "schema" in param_def:
                try:
                    schema_data = param_def["schema"]
                    if isinstance(schema_data, dict):
                        constraints.update(schema_data)
                    else:
                        logger.warning("Schema for parameter '%s' is not a dictionary: %s", name, schema_data)
                except Exception as e:
                    logger.error("Error processing schema for parameter '%s': %s", name, e)

            # Generate examples
            try:
                examples = self._generate_parameter_examples(name, param_type, param_def)
            except Exception as e:
                logger.error("Error generating examples for parameter '%s': %s", name, e)
                examples = []

            return ParameterAnalysis(
                name=name,
                type=param_type,
                required=required,
                description=description,
                constraints=constraints,
                default_value=default_value,
                examples=examples,
                complexity_score=self.NO_COMPLEXITY_SCORE,  # Will be calculated later
            )
        except Exception as e:
            logger.error("Critical error analyzing parameter from definition: %s", e)
            # Return a minimal valid parameter analysis
            name = param_def.get("name", "unknown") if isinstance(param_def, dict) else "unknown"
            return ParameterAnalysis(
                name=name,
                type="string",
                required=False,
                description=f"Error analyzing parameter: {name}",
                constraints={},
                default_value=None,
                examples=[],
                complexity_score=self.DEFAULT_COMPLEXITY_SCORE,
            )

    def extract_constraints(self, param_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract constraints from a parameter schema.

        Args:
            param_schema: Parameter schema dictionary

        Returns:
            Dictionary of constraints
        """
        try:
            if not isinstance(param_schema, dict):
                logger.warning("Parameter schema is not a dictionary, cannot extract constraints: %s", param_schema)
                return {}

            constraints = {}

            # String constraints
            for key in ["minLength", "maxLength", "pattern", "format", "enum"]:
                try:
                    if key in param_schema:
                        value = param_schema[key]
                        # Basic validation of constraint values
                        if key in ["minLength", "maxLength"] and not isinstance(value, int):
                            logger.warning("Invalid %s constraint value: %s", key, value)
                            continue
                        if key == "pattern" and not isinstance(value, str):
                            logger.warning("Invalid pattern constraint value: %s", value)
                            continue
                        if key == "enum" and not isinstance(value, list):
                            logger.warning("Invalid enum constraint value: %s", value)
                            continue
                        constraints[key] = value
                except Exception as e:
                    logger.error("Error extracting string constraint '%s': %s", key, e)
                    continue

            # Number constraints
            for key in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"]:
                try:
                    if key in param_schema:
                        value = param_schema[key]
                        # Basic validation of numeric constraints
                        if not isinstance(value, (int, float)):
                            logger.warning("Invalid %s constraint value: %s", key, value)
                            continue
                        constraints[key] = value
                except Exception as e:
                    logger.error("Error extracting number constraint '%s': %s", key, e)
                    continue

            # Array constraints
            for key in ["minItems", "maxItems", "uniqueItems", "items"]:
                try:
                    if key in param_schema:
                        value = param_schema[key]
                        # Basic validation of array constraints
                        if key in ["minItems", "maxItems"] and not isinstance(value, int):
                            logger.warning("Invalid %s constraint value: %s", key, value)
                            continue
                        if key == "uniqueItems" and not isinstance(value, bool):
                            logger.warning("Invalid uniqueItems constraint value: %s", value)
                            continue
                        constraints[key] = value
                except Exception as e:
                    logger.error("Error extracting array constraint '%s': %s", key, e)
                    continue

            # Object constraints
            for key in ["properties", "required", "additionalProperties", "patternProperties"]:
                try:
                    if key in param_schema:
                        value = param_schema[key]
                        # Basic validation of object constraints
                        if key in ["properties", "patternProperties"] and not isinstance(value, dict):
                            logger.warning("Invalid %s constraint value: %s", key, value)
                            continue
                        if key == "required" and not isinstance(value, list):
                            logger.warning("Invalid required constraint value: %s", value)
                            continue
                        constraints[key] = value
                except Exception as e:
                    logger.error("Error extracting object constraint '%s': %s", key, e)
                    continue

            return constraints
        except Exception as e:
            logger.error("Critical error extracting constraints: %s", e)
            return {}

    def calculate_complexity_score(self, parameter: ParameterAnalysis) -> float:
        """
        Calculate complexity score for a parameter.

        Args:
            parameter: ParameterAnalysis object

        Returns:
            Complexity score from 0.0 (simple) to 1.0 (very complex)
        """
        try:
            if not hasattr(parameter, "type") or not hasattr(parameter, "constraints"):
                logger.error("Invalid parameter object for complexity calculation: %s", parameter)
                return self.DEFAULT_COMPLEXITY_SCORE

            score = self.NO_COMPLEXITY_SCORE

            # Base type complexity using constants
            try:
                param_type = getattr(parameter, "type", "string")
                score += self.TYPE_BASE_COMPLEXITY.get(param_type, self.DEFAULT_COMPLEXITY_SCORE * 0.6)
            except Exception as e:
                logger.error(
                    "Error calculating type complexity for parameter '%s': %s", getattr(parameter, "name", "unknown"), e
                )
                score += self.DEFAULT_COMPLEXITY_SCORE * 0.6

            # Required parameters are slightly more complex
            try:
                if getattr(parameter, "required", False):
                    score += self.REQUIRED_PARAMETER_COMPLEXITY_BONUS
            except Exception as e:
                logger.error(
                    "Error checking required status for parameter '%s': %s", getattr(parameter, "name", "unknown"), e
                )

            # Constraint complexity
            try:
                constraints = getattr(parameter, "constraints", {})
                if isinstance(constraints, dict):
                    constraint_count = len(constraints)
                    if constraint_count > 0:
                        score += min(
                            constraint_count * self.CONSTRAINT_COMPLEXITY_MULTIPLIER,
                            self.MAX_CONSTRAINT_COMPLEXITY_BONUS,
                        )

                    # Nested structure complexity
                    if "items" in constraints:
                        try:
                            items_schema = constraints["items"]
                            if isinstance(items_schema, dict) and items_schema.get("type") == "object":
                                score += self.NESTED_OBJECT_COMPLEXITY_BONUS
                        except Exception as e:
                            logger.warning("Error processing items constraint: %s", e)

                    if "properties" in constraints:
                        try:
                            nested_props = constraints["properties"]
                            if isinstance(nested_props, dict):
                                score += min(
                                    len(nested_props) * self.PROPERTY_COMPLEXITY_MULTIPLIER,
                                    self.MAX_PROPERTY_COMPLEXITY_BONUS,
                                )
                        except Exception as e:
                            logger.warning("Error processing properties constraint: %s", e)

                    # Pattern constraints add complexity
                    if "pattern" in constraints:
                        score += self.PATTERN_CONSTRAINT_COMPLEXITY_BONUS

                    # Enum constraints can be complex
                    if "enum" in constraints:
                        try:
                            enum_values = constraints["enum"]
                            if isinstance(enum_values, list) and len(enum_values) > self.LARGE_ENUM_THRESHOLD:
                                score += self.LARGE_ENUM_COMPLEXITY_BONUS
                        except Exception as e:
                            logger.warning("Error processing enum constraint: %s", e)
                else:
                    logger.warning(
                        "Constraints is not a dictionary for parameter '%s': %s",
                        getattr(parameter, "name", "unknown"),
                        constraints,
                    )
            except Exception as e:
                logger.error(
                    "Error calculating constraint complexity for parameter '%s': %s",
                    getattr(parameter, "name", "unknown"),
                    e,
                )

            # Cap at maximum complexity score
            return min(score, self.MAX_COMPLEXITY_SCORE)

        except Exception as e:
            logger.error("Critical error calculating complexity score: %s", e)
            return self.DEFAULT_COMPLEXITY_SCORE  # Default medium complexity on error

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
        try:
            if not isinstance(name, str):
                logger.warning("Invalid parameter name type: %s", type(name))
                name = str(name) if name is not None else "unknown"

            if not isinstance(param_type, str):
                logger.warning("Invalid parameter type: %s", param_type)
                param_type = "string"

            if not isinstance(param_schema, dict):
                logger.warning("Invalid parameter schema type: %s", type(param_schema))
                param_schema = {}

            examples = []
            name_lower = name.lower()
            description = param_schema.get("description", "")

            try:
                description = description.lower() if isinstance(description, str) else ""
            except Exception as e:
                logger.warning("Error processing description: %s", e)
                description = ""

            # Check for enum values first
            try:
                if "enum" in param_schema:
                    enum_values = param_schema["enum"]
                    if isinstance(enum_values, list):
                        examples.extend([str(v) for v in enum_values[:3] if v is not None])
                        return examples
                    else:
                        logger.warning("Enum values is not a list: %s", enum_values)
            except Exception as e:
                logger.error("Error processing enum values: %s", e)

            try:
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
                    try:
                        # Check constraints for realistic ranges
                        minimum = param_schema.get("minimum", 0)
                        maximum = param_schema.get("maximum", 1000)

                        # Validate constraint values
                        if not isinstance(minimum, (int, float)):
                            minimum = 0
                        if not isinstance(maximum, (int, float)):
                            maximum = 1000

                        if "timeout" in description:
                            examples.extend(["5000", "10000", "30000"])
                        elif "limit" in description or "count" in description:
                            examples.extend(["10", "50", "100"])
                        elif "port" in name_lower:
                            examples.extend(["8080", "3000", "5432"])
                        else:
                            # Generate examples within constraints
                            if maximum <= 10:
                                examples.extend(["1", "5", str(int(maximum))])
                            else:
                                mid_value = (minimum + maximum) // 2
                                examples.extend([str(int(minimum)), str(int(mid_value)), str(int(maximum))])
                    except Exception as e:
                        logger.error("Error generating number examples: %s", e)
                        examples.extend(["1", "10", "100"])

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
                else:
                    # Unknown type
                    logger.warning("Unknown parameter type '%s', using default examples", param_type)
                    examples.extend(["example_value", "default", "value"])

            except Exception as e:
                logger.error("Error generating type-specific examples for '%s': %s", name, e)
                examples.extend(["example_value", "default"])

            return examples[:3]  # Limit to 3 examples

        except Exception as e:
            logger.error("Critical error generating examples for parameter '%s': %s", name, e)
            return ["example_value"]  # Minimal fallback
