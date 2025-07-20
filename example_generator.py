#!/usr/bin/env python3
"""
Example Generator

This module provides capabilities to extract usage examples from documentation
and generate synthetic examples for common use cases, creating parameter
combination examples and building example libraries for each tool.
"""

import re
from typing import List, Optional

from logging_config import get_logger
from mcp_metadata_types import ExampleValidationResult, MCPToolMetadata, ParameterAnalysis, ToolExample

logger = get_logger(__name__)


class ExampleGenerator:
    """Extract and generate usage examples for MCP tools."""

    # Common example templates by tool category
    CATEGORY_TEMPLATES = {
        "file_ops": {
            "read_file": "Read configuration from {file_path}",
            "write_file": "Save data to {file_path}",
            "edit_file": "Update {file_path} by replacing {old_value} with {new_value}",
        },
        "search": {
            "find_pattern": "Search for {pattern} in {location}",
            "filter_results": "Filter results matching {criteria}",
            "list_matches": "List all files matching {pattern}",
        },
        "execution": {
            "run_command": "Execute {command} with {timeout} timeout",
            "script_execution": "Run {script} in {environment}",
            "system_task": "Perform {task} on system",
        },
        "web": {
            "fetch_url": "Download content from {url}",
            "api_request": "Make API request to {endpoint}",
            "web_search": "Search the web for {query}",
        },
        "development": {
            "create_task": "Create task for {objective}",
            "manage_todos": "Update task list with {items}",
            "agent_action": "Have agent perform {action}",
        },
    }

    def __init__(self):
        """Initialize the example generator."""
        logger.info("Initialized Example Generator")

    def extract_examples_from_documentation(self, documentation: str) -> List[ToolExample]:
        """
        Extract usage examples from tool documentation.

        Args:
            documentation: Tool documentation text

        Returns:
            List of extracted ToolExample objects
        """
        if not documentation:
            return []

        examples = []

        # Pattern 1: Explicit examples with headers
        example_patterns = [
            r"example\s*\d*[:\-]\s*(.+?)(?=\n\s*example|\n\s*$|\n\n)",
            r"usage[:\-]\s*(.+?)(?=\n\s*usage|\n\s*$|\n\n)",
        ]

        for pattern in example_patterns:
            matches = re.findall(pattern, documentation, re.IGNORECASE | re.DOTALL)
            for i, match in enumerate(matches):
                example = self._parse_example_text(match.strip(), f"Example {i+1}")
                if example:
                    examples.append(example)

        # Pattern 2: Code blocks
        code_block_pattern = r"```(\w+)?\n(.*?)\n```"
        code_matches = re.findall(code_block_pattern, documentation, re.DOTALL)

        for i, (language, code) in enumerate(code_matches):
            if self._looks_like_tool_usage(code):
                # Try to find description before code block
                use_case = self._extract_use_case_before_code(documentation, code)

                examples.append(
                    ToolExample(
                        use_case=use_case or f"Code example {i+1}",
                        example_call=code.strip(),
                        language=language or "python",
                        effectiveness_score=0.7,
                    )
                )

        # Pattern 3: Inline examples
        inline_pattern = r"(?:e\.g\.|for example|such as)[:\,\s]*([^.\n]+)"
        inline_matches = re.findall(inline_pattern, documentation, re.IGNORECASE)

        for match in inline_matches:
            if self._looks_like_tool_usage(match):
                examples.append(
                    ToolExample(
                        use_case="Inline example",
                        example_call=match.strip(),
                        effectiveness_score=0.6,
                    )
                )

        logger.debug("Extracted %d examples from documentation", len(examples))
        return examples

    def generate_synthetic_examples(self, tool_metadata: MCPToolMetadata) -> List[ToolExample]:
        """
        Generate synthetic usage examples based on tool metadata.

        Args:
            tool_metadata: Tool metadata to generate examples for

        Returns:
            List of generated ToolExample objects
        """
        try:
            if not tool_metadata or not hasattr(tool_metadata, "name"):
                logger.error("Invalid tool metadata for synthetic example generation")
                return []

            tool_name = getattr(tool_metadata, "name", "unknown")
            examples = []

            # Generate examples based on category
            try:
                category_examples = self._generate_category_examples(tool_metadata)
                if isinstance(category_examples, list):
                    examples.extend(category_examples)
                else:
                    logger.warning("Category examples is not a list for tool %s", tool_name)
            except Exception as e:
                logger.error("Error generating category examples for %s: %s", tool_name, e)

            # Generate parameter combination examples
            try:
                param_examples = self._generate_parameter_combination_examples(tool_metadata)
                if isinstance(param_examples, list):
                    examples.extend(param_examples)
                else:
                    logger.warning("Parameter examples is not a list for tool %s", tool_name)
            except Exception as e:
                logger.error("Error generating parameter combination examples for %s: %s", tool_name, e)

            # Generate edge case examples
            try:
                edge_examples = self._generate_edge_case_examples(tool_metadata)
                if isinstance(edge_examples, list):
                    examples.extend(edge_examples)
                else:
                    logger.warning("Edge case examples is not a list for tool %s", tool_name)
            except Exception as e:
                logger.error("Error generating edge case examples for %s: %s", tool_name, e)

            logger.debug("Generated %d synthetic examples for %s", len(examples), tool_name)

            return examples[:5]  # Limit to 5 examples to avoid overwhelming

        except Exception as e:
            logger.error("Critical error generating synthetic examples: %s", e)
            return []

    def validate_example_value(self, value: str, parameter: ParameterAnalysis) -> ExampleValidationResult:
        """
        Validate a generated example against parameter schema.

        Args:
            value: Example value to validate
            parameter: Parameter to validate against

        Returns:
            ExampleValidationResult with validation status
        """
        # String validation
        if parameter.type == "string":
            if "minLength" in parameter.constraints:
                min_len = parameter.constraints["minLength"]
                if len(value) < min_len:
                    return ExampleValidationResult(
                        is_valid=False,
                        error_message=f"Value too short, minimum length is {min_len}",
                        suggested_fix=value + "x" * (min_len - len(value)),
                    )

            if "maxLength" in parameter.constraints:
                max_len = parameter.constraints["maxLength"]
                if len(value) > max_len:
                    return ExampleValidationResult(
                        is_valid=False,
                        error_message=f"Value too long, maximum length is {max_len}",
                        suggested_fix=value[:max_len],
                    )

            if "pattern" in parameter.constraints:
                pattern = parameter.constraints["pattern"]
                if not re.match(pattern, value):
                    return ExampleValidationResult(
                        is_valid=False,
                        error_message=f"Value does not match required pattern: {pattern}",
                    )

            if "enum" in parameter.constraints:
                enum_values = parameter.constraints["enum"]
                if value not in enum_values:
                    return ExampleValidationResult(
                        is_valid=False,
                        error_message=f"Value must be one of: {enum_values}",
                        suggested_fix=str(enum_values[0]),
                    )

        # Number validation
        elif parameter.type in ["number", "integer"]:
            try:
                num_value = float(value) if parameter.type == "number" else int(value)

                if "minimum" in parameter.constraints:
                    minimum = parameter.constraints["minimum"]
                    if num_value < minimum:
                        return ExampleValidationResult(
                            is_valid=False, error_message=f"Value below minimum: {minimum}", suggested_fix=str(minimum)
                        )

                if "maximum" in parameter.constraints:
                    maximum = parameter.constraints["maximum"]
                    if num_value > maximum:
                        return ExampleValidationResult(
                            is_valid=False, error_message=f"Value above maximum: {maximum}", suggested_fix=str(maximum)
                        )

            except ValueError:
                return ExampleValidationResult(
                    is_valid=False,
                    error_message=f"Invalid {parameter.type} value: {value}",
                    suggested_fix="0" if parameter.type == "integer" else "0.0",
                )

        # Boolean validation
        elif parameter.type == "boolean":
            if value.lower() not in ["true", "false", "1", "0"]:
                return ExampleValidationResult(
                    is_valid=False, error_message="Boolean value must be true or false", suggested_fix="true"
                )

        return ExampleValidationResult(is_valid=True)

    def _generate_category_examples(self, tool_metadata: MCPToolMetadata) -> List[ToolExample]:
        """Generate examples based on tool category."""
        examples = []
        category = tool_metadata.category

        templates = self.CATEGORY_TEMPLATES.get(category, {})

        for template_name, template in templates.items():
            # Fill template with parameter examples
            filled_template = self._fill_example_template(template, tool_metadata)
            if filled_template:
                examples.append(
                    ToolExample(
                        use_case=template_name.replace("_", " ").title(),
                        example_call=filled_template,
                        effectiveness_score=0.8,
                    )
                )

        return examples

    def _generate_parameter_combination_examples(self, tool_metadata: MCPToolMetadata) -> List[ToolExample]:
        """Generate examples with different parameter combinations."""
        examples = []

        if not tool_metadata.parameters:
            return examples

        # Example 1: Required parameters only
        required_params = [p for p in tool_metadata.parameters if p.required]
        if required_params:
            param_strs = []
            for param in required_params:
                example_val = self._get_parameter_example_value(param)
                param_strs.append(f'{param.name}="{example_val}"')

            examples.append(
                ToolExample(
                    use_case="Basic usage with required parameters",
                    example_call=f"{tool_metadata.name.lower()}({', '.join(param_strs)})",
                    effectiveness_score=0.9,
                )
            )

        # Example 2: All parameters
        if len(tool_metadata.parameters) > 1:
            all_param_strs = []
            for param in tool_metadata.parameters[:4]:  # Limit to 4 for readability
                example_val = self._get_parameter_example_value(param)
                all_param_strs.append(f'{param.name}="{example_val}"')

            examples.append(
                ToolExample(
                    use_case="Complete usage with all parameters",
                    example_call=f"{tool_metadata.name.lower()}({', '.join(all_param_strs)})",
                    effectiveness_score=0.7,
                )
            )

        # Example 3: Common optional parameters
        optional_params = [p for p in tool_metadata.parameters if not p.required]
        if required_params and optional_params:
            # Combine required + first optional
            mixed_params = required_params[:2] + optional_params[:1]
            mixed_strs = []
            for param in mixed_params:
                example_val = self._get_parameter_example_value(param)
                mixed_strs.append(f'{param.name}="{example_val}"')

            examples.append(
                ToolExample(
                    use_case="Usage with optional parameters",
                    example_call=f"{tool_metadata.name.lower()}({', '.join(mixed_strs)})",
                    effectiveness_score=0.8,
                )
            )

        return examples

    def _generate_edge_case_examples(self, tool_metadata: MCPToolMetadata) -> List[ToolExample]:
        """Generate examples for edge cases and special scenarios."""
        examples = []

        # Example with minimal parameters
        if tool_metadata.parameters:
            min_required = min(
                [p for p in tool_metadata.parameters if p.required], default=None, key=lambda x: len(x.description)
            )
            if min_required:
                example_val = self._get_parameter_example_value(min_required)
                examples.append(
                    ToolExample(
                        use_case="Minimal usage example",
                        example_call=f'{tool_metadata.name.lower()}({min_required.name}="{example_val}")',
                        effectiveness_score=0.6,
                    )
                )

        # Example with default values
        default_params = [p for p in tool_metadata.parameters if p.default_value is not None]
        if default_params:
            examples.append(
                ToolExample(
                    use_case="Using default values",
                    example_call=(
                        f"# {default_params[0].name} defaults to {default_params[0].default_value}\n"
                        f"{tool_metadata.name.lower()}(...)"
                    ),
                    effectiveness_score=0.5,
                )
            )

        return examples

    def _parse_example_text(self, text: str, default_use_case: str) -> Optional[ToolExample]:
        """Parse example text and create ToolExample object."""
        text = text.strip()
        if not text or len(text) < 5:
            return None

        # Try to separate use case from code
        lines = text.split("\n")
        use_case = default_use_case
        example_call = text

        # If multiple lines, first might be description
        if len(lines) > 1:
            first_line = lines[0].strip()
            if not self._looks_like_code(first_line) and len(first_line) > 10:
                use_case = first_line
                example_call = "\n".join(lines[1:]).strip()

        return ToolExample(
            use_case=use_case,
            example_call=example_call,
            effectiveness_score=0.7,
        )

    def _looks_like_tool_usage(self, text: str) -> bool:
        """Check if text looks like tool usage code."""
        # Look for common patterns
        patterns = [
            r"\w+\s*\(",  # function call
            r"\w+\.\w+",  # method call
            r'=\s*["\']',  # parameter assignment
            r"--\w+",  # command line flags
        ]

        return any(re.search(pattern, text) for pattern in patterns)

    def _looks_like_code(self, text: str) -> bool:
        """Check if text looks like code rather than descriptive text."""
        code_indicators = ["(", ")", "=", '"', "'", "{", "}", "[", "]", ";"]
        return any(indicator in text for indicator in code_indicators)

    def _extract_use_case_before_code(self, documentation: str, code: str) -> Optional[str]:
        """Extract use case description that appears before a code block."""
        # Find the code in documentation
        code_index = documentation.find(code)
        if code_index == -1:
            return None

        # Look at text before the code
        before_code = documentation[:code_index]
        lines = before_code.split("\n")

        # Look for descriptive lines near the code
        for line in reversed(lines[-5:]):  # Check last 5 lines before code
            line = line.strip()
            if line and not line.startswith("```") and len(line) > 10:
                # Clean up common prefixes
                line = re.sub(r"^(example\s*\d*[:\-]?\s*)", "", line, flags=re.IGNORECASE)
                return line

        return None

    def _fill_example_template(self, template: str, tool_metadata: MCPToolMetadata) -> Optional[str]:
        """Fill example template with actual parameter values."""
        # Find placeholders in template
        placeholders = re.findall(r"\{(\w+)\}", template)

        filled_template = template
        for placeholder in placeholders:
            # Find matching parameter
            matching_param = None
            for param in tool_metadata.parameters:
                if placeholder.lower() in param.name.lower() or param.name.lower() in placeholder.lower():
                    matching_param = param
                    break

            if matching_param:
                example_val = self._get_parameter_example_value(matching_param)
                filled_template = filled_template.replace(f"{{{placeholder}}}", example_val)
            else:
                # Use placeholder-based default
                default_val = self._get_placeholder_default(placeholder)
                filled_template = filled_template.replace(f"{{{placeholder}}}", default_val)

        return filled_template if "{" not in filled_template else None

    def _get_parameter_example_value(self, parameter: ParameterAnalysis) -> str:
        """Get an appropriate example value for a parameter."""
        # Use existing examples if available
        if parameter.examples:
            return str(parameter.examples[0])

        # Generate based on parameter name and type
        name_lower = parameter.name.lower()

        if parameter.type == "string":
            if "path" in name_lower or "file" in name_lower:
                return "/path/to/file.txt"
            elif "url" in name_lower:
                return "https://example.com"
            elif "command" in name_lower:
                return "ls -la"
            elif "pattern" in name_lower:
                return "search_pattern"
            elif "query" in name_lower:
                return "search query"
            else:
                return "example_value"

        elif parameter.type in ["number", "integer"]:
            if "timeout" in parameter.description.lower():
                return "5000"
            elif "limit" in name_lower:
                return "10"
            elif "port" in name_lower:
                return "8080"
            else:
                return "100"

        elif parameter.type == "boolean":
            return "true"

        elif parameter.type == "array":
            return '["item1", "item2"]'

        else:
            return "value"

    def _get_placeholder_default(self, placeholder: str) -> str:
        """Get default value for a template placeholder."""
        placeholder_defaults = {
            "file_path": "/path/to/file.txt",
            "url": "https://example.com",
            "command": "ls -la",
            "pattern": "search_pattern",
            "query": "search query",
            "timeout": "5000",
            "old_value": "old_text",
            "new_value": "new_text",
            "location": "current_directory",
            "criteria": "filter_criteria",
            "script": "script.py",
            "environment": "development",
            "task": "system_task",
            "endpoint": "/api/endpoint",
            "objective": "task_objective",
            "items": "task_items",
            "action": "agent_action",
        }

        return placeholder_defaults.get(placeholder.lower(), "example_value")
