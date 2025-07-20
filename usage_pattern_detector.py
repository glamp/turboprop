#!/usr/bin/env python3
"""
Usage Pattern Detector

This module provides capabilities to detect and classify tool usage patterns,
analyze parameter complexity, and generate confidence scores for different
usage scenarios based on tool structure and metadata.
"""

from typing import Any, Dict, List

from logging_config import get_logger
from mcp_metadata_types import ComplexityAnalysis, MCPToolMetadata, ParameterAnalysis, UsagePattern

logger = get_logger(__name__)


class UsagePatternDetector:
    """Detect and classify tool usage patterns."""

    # Complexity factor weights for overall tool scoring
    COMPLEXITY_FACTORS = {
        "parameter_count": {"weight": 0.3, "threshold_simple": 3, "threshold_complex": 8},
        "required_parameters": {"weight": 0.2, "threshold_simple": 1, "threshold_complex": 5},
        "schema_depth": {"weight": 0.2, "threshold_simple": 2, "threshold_complex": 4},
        "type_complexity": {"weight": 0.2, "multiplier_object": 2.0, "multiplier_array": 1.5},
        "documentation_quality": {"weight": 0.1, "has_examples": 0.8, "detailed_descriptions": 1.0},
    }

    # Common usage patterns based on parameter combinations
    USAGE_PATTERNS = {
        "simple_file_operation": {
            "parameters": ["file_path", "content?"],
            "description": "Basic file operation with single file path",
            "complexity": "basic",
            "success_probability": 0.9,
        },
        "search_operation": {
            "parameters": ["pattern", "path?", "options?"],
            "description": "Search or pattern matching operation",
            "complexity": "intermediate",
            "success_probability": 0.8,
        },
        "execution_task": {
            "parameters": ["command", "timeout?", "description?"],
            "description": "Command or script execution",
            "complexity": "intermediate",
            "success_probability": 0.7,
        },
        "data_transformation": {
            "parameters": ["input", "transformation_params", "output_format?"],
            "description": "Data processing and transformation",
            "complexity": "advanced",
            "success_probability": 0.6,
        },
        "bulk_operations": {
            "parameters": ["items[]", "operation_type", "options?"],
            "description": "Operations on multiple items or bulk processing",
            "complexity": "advanced",
            "success_probability": 0.5,
        },
    }

    def __init__(self):
        """Initialize the usage pattern detector."""
        logger.info("Initialized Usage Pattern Detector")

    def analyze_parameter_complexity(self, parameters: List[ParameterAnalysis]) -> ComplexityAnalysis:
        """
        Analyze tool complexity based on parameters.

        Args:
            parameters: List of parameter analyses

        Returns:
            ComplexityAnalysis object with complexity metrics
        """
        if not parameters:
            return ComplexityAnalysis(
                total_parameters=0,
                required_parameters=0,
                optional_parameters=0,
                complex_parameters=0,
                overall_complexity=0.0,
            )

        total_params = len(parameters)
        required_params = sum(1 for p in parameters if p.required)
        optional_params = total_params - required_params
        complex_params = sum(1 for p in parameters if p.complexity_score > 0.5)

        # Calculate overall complexity
        complexity_factors = {}

        # Parameter count factor
        param_count_score = self._calculate_factor_score(
            total_params,
            self.COMPLEXITY_FACTORS["parameter_count"]["threshold_simple"],
            self.COMPLEXITY_FACTORS["parameter_count"]["threshold_complex"],
        )
        complexity_factors["parameter_count"] = param_count_score

        # Required parameters factor
        required_score = self._calculate_factor_score(
            required_params,
            self.COMPLEXITY_FACTORS["required_parameters"]["threshold_simple"],
            self.COMPLEXITY_FACTORS["required_parameters"]["threshold_complex"],
        )
        complexity_factors["required_parameters"] = required_score

        # Type complexity factor
        avg_param_complexity = sum(p.complexity_score for p in parameters) / total_params
        complexity_factors["type_complexity"] = avg_param_complexity

        # Schema depth factor (based on nested structures)
        max_depth = max((self._calculate_parameter_depth(p) for p in parameters), default=1)
        depth_score = self._calculate_factor_score(max_depth, 2, 4)
        complexity_factors["schema_depth"] = depth_score

        # Calculate weighted overall complexity
        overall_complexity = 0.0
        for factor, score in complexity_factors.items():
            weight = self.COMPLEXITY_FACTORS.get(factor, {}).get("weight", 0.1)
            overall_complexity += score * weight

        # Cap at 1.0
        overall_complexity = min(overall_complexity, 1.0)

        logger.debug(
            "Analyzed parameter complexity: %d total, %d required, %.2f overall",
            total_params,
            required_params,
            overall_complexity,
        )

        return ComplexityAnalysis(
            total_parameters=total_params,
            required_parameters=required_params,
            optional_parameters=optional_params,
            complex_parameters=complex_params,
            overall_complexity=overall_complexity,
            complexity_factors=complexity_factors,
        )

    def identify_common_patterns(self, tool_metadata: MCPToolMetadata) -> List[UsagePattern]:
        """
        Identify common usage patterns for the tool.

        Args:
            tool_metadata: Tool metadata to analyze

        Returns:
            List of identified usage patterns
        """
        patterns = []

        param_names = [p.name.lower() for p in tool_metadata.parameters]
        required_params = [p.name.lower() for p in tool_metadata.parameters if p.required]

        logger.debug("Identifying patterns for tool %s with params: %s", tool_metadata.name, param_names)

        # Check against known pattern templates
        for pattern_name, pattern_info in self.USAGE_PATTERNS.items():
            if self._matches_pattern(param_names, required_params, pattern_info):
                # Generate example code for this pattern
                example_code = self._generate_pattern_example(
                    tool_metadata.name, pattern_info, tool_metadata.parameters
                )

                pattern = UsagePattern(
                    pattern_name=pattern_name,
                    description=pattern_info["description"],
                    parameter_combination=pattern_info["parameters"],
                    use_case=self._generate_use_case(tool_metadata.name, pattern_name),
                    complexity_level=pattern_info["complexity"],
                    example_code=example_code,
                    success_probability=pattern_info["success_probability"],
                )
                patterns.append(pattern)

        # Generate custom patterns based on tool structure
        custom_patterns = self._generate_custom_patterns(tool_metadata)
        patterns.extend(custom_patterns)

        logger.debug("Identified %d usage patterns for tool %s", len(patterns), tool_metadata.name)

        return patterns

    def generate_complexity_score(self, tool: MCPToolMetadata) -> float:
        """
        Generate overall tool complexity score.

        Args:
            tool: Tool metadata to analyze

        Returns:
            Complexity score from 0.0 (simple) to 1.0 (very complex)
        """
        if tool.complexity_analysis:
            base_score = tool.complexity_analysis.overall_complexity
        else:
            # Calculate basic complexity if not available
            complexity_analysis = self.analyze_parameter_complexity(tool.parameters)
            base_score = complexity_analysis.overall_complexity

        # Adjust based on additional factors
        adjustments = 0.0

        # Documentation quality adjustment
        if tool.documentation_analysis:
            if len(tool.documentation_analysis.examples) > 2:
                adjustments -= 0.1  # Good examples reduce perceived complexity
            if len(tool.documentation_analysis.warnings) > 0:
                adjustments += 0.1  # Warnings increase complexity

        # Category-based adjustments
        category_complexity = {
            "file_ops": 0.0,
            "search": 0.1,
            "execution": 0.2,
            "development": 0.3,
            "web": 0.2,
            "notebook": 0.3,
            "workflow": 0.4,
        }
        category_adj = category_complexity.get(tool.category, 0.1)
        adjustments += category_adj * 0.2  # Scale down category impact

        # Example availability adjustment
        if len(tool.examples) == 0:
            adjustments += 0.2  # No examples make tools seem more complex
        elif len(tool.examples) > 3:
            adjustments -= 0.1  # Many examples reduce complexity

        final_score = base_score + adjustments
        return max(0.0, min(1.0, final_score))  # Clamp to [0, 1]

    def _calculate_factor_score(self, value: int, simple_threshold: int, complex_threshold: int) -> float:
        """Calculate a normalized score based on thresholds."""
        if value <= simple_threshold:
            return 0.0
        elif value >= complex_threshold:
            return 1.0
        else:
            # Linear interpolation between thresholds
            range_size = complex_threshold - simple_threshold
            position = value - simple_threshold
            return position / range_size

    def _calculate_parameter_depth(self, parameter: ParameterAnalysis) -> int:
        """Calculate the nesting depth of a parameter schema."""
        depth = 1

        if parameter.type == "array" and "items" in parameter.constraints:
            items_schema = parameter.constraints["items"]
            if isinstance(items_schema, dict) and items_schema.get("type") == "object":
                depth += 1
                if "properties" in items_schema:
                    depth += 1

        elif parameter.type == "object" and "properties" in parameter.constraints:
            depth += 1
            properties = parameter.constraints["properties"]
            if isinstance(properties, dict):
                # Check for nested objects
                for prop_schema in properties.values():
                    if isinstance(prop_schema, dict) and prop_schema.get("type") in ["object", "array"]:
                        depth = max(depth, depth + 1)

        return depth

    def _matches_pattern(
        self, param_names: List[str], required_params: List[str], pattern_info: Dict[str, Any]
    ) -> bool:
        """Check if tool parameters match a usage pattern."""
        pattern_params = pattern_info["parameters"]

        # Count required matches
        required_matches = 0
        optional_matches = 0

        for pattern_param in pattern_params:
            is_optional = pattern_param.endswith("?")
            clean_param = pattern_param.rstrip("?").rstrip("[]")

            # Check for fuzzy matches (contains keywords)
            found = any(clean_param in param_name or param_name in clean_param for param_name in param_names)

            if found:
                if is_optional:
                    optional_matches += 1
                else:
                    required_matches += 1

        # Require at least one required parameter match
        required_pattern_params = [p for p in pattern_params if not p.endswith("?")]
        if required_pattern_params and required_matches == 0:
            return False

        # Calculate match ratio
        total_matches = required_matches + optional_matches
        match_ratio = total_matches / len(pattern_params) if pattern_params else 0

        return match_ratio >= 0.4  # At least 40% parameter match

    def _generate_pattern_example(
        self, tool_name: str, pattern_info: Dict[str, Any], parameters: List[ParameterAnalysis]
    ) -> str:
        """Generate example code for a usage pattern."""
        param_examples = {}

        # Collect examples for each parameter
        for param in parameters:
            if param.examples:
                param_examples[param.name] = param.examples[0]
            else:
                param_examples[param.name] = self._generate_default_example(param)

        # Create example based on tool and pattern
        tool_name_lower = tool_name.lower()

        if "file" in pattern_info.get("description", "").lower():
            return f'{tool_name_lower}(file_path="/path/to/file.txt")'
        elif "search" in pattern_info.get("description", "").lower():
            pattern_param = next((p for p in param_examples if "pattern" in p), "pattern")
            return f'{tool_name_lower}({pattern_param}="search_term")'
        elif "command" in pattern_info.get("description", "").lower():
            return f'{tool_name_lower}(command="ls -la")'
        else:
            # Generic example
            required_params = [p for p in parameters if p.required][:2]  # Max 2 for readability
            param_strs = []
            for param in required_params:
                example_val = param_examples.get(param.name, "example_value")
                param_strs.append(f'{param.name}="{example_val}"')

            return f'{tool_name_lower}({", ".join(param_strs)})'

    def _generate_use_case(self, tool_name: str, pattern_name: str) -> str:
        """Generate a use case description for a pattern."""
        use_case_templates = {
            "simple_file_operation": f"Use {tool_name} for basic file manipulation tasks",
            "search_operation": f"Use {tool_name} to find and filter content",
            "execution_task": f"Use {tool_name} to run commands and scripts",
            "data_transformation": f"Use {tool_name} to process and transform data",
            "bulk_operations": f"Use {tool_name} for batch processing multiple items",
        }

        return use_case_templates.get(pattern_name, f"Common usage pattern for {tool_name}")

    def _generate_custom_patterns(self, tool_metadata: MCPToolMetadata) -> List[UsagePattern]:
        """Generate custom patterns based on specific tool characteristics."""
        patterns = []

        # Pattern for tools with many optional parameters
        if len([p for p in tool_metadata.parameters if not p.required]) > 3:
            patterns.append(
                UsagePattern(
                    pattern_name="flexible_configuration",
                    description="Tool with many optional configuration parameters",
                    parameter_combination=[p.name for p in tool_metadata.parameters[:3]],
                    use_case=f"Configure {tool_metadata.name} with various optional settings",
                    complexity_level="intermediate",
                    example_code=f'{tool_metadata.name.lower()}({tool_metadata.parameters[0].name}="value")',
                    success_probability=0.7,
                )
            )

        # Pattern for tools with array parameters
        array_params = [p for p in tool_metadata.parameters if p.type == "array"]
        if array_params:
            patterns.append(
                UsagePattern(
                    pattern_name="batch_processing",
                    description="Tool designed for processing multiple items",
                    parameter_combination=[array_params[0].name],
                    use_case=f"Process multiple items with {tool_metadata.name}",
                    complexity_level="advanced",
                    example_code=f'{tool_metadata.name.lower()}({array_params[0].name}=["item1", "item2"])',
                    success_probability=0.6,
                )
            )

        return patterns

    def _generate_default_example(self, parameter: ParameterAnalysis) -> str:
        """Generate a default example value for a parameter."""
        if parameter.type == "string":
            if "path" in parameter.name.lower():
                return "/path/to/file"
            elif "url" in parameter.name.lower():
                return "https://example.com"
            elif "command" in parameter.name.lower():
                return "ls -la"
            else:
                return "example_value"
        elif parameter.type in ["number", "integer"]:
            return "100"
        elif parameter.type == "boolean":
            return "true"
        elif parameter.type == "array":
            return '["item1", "item2"]'
        else:
            return "value"
