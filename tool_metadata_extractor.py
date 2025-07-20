#!/usr/bin/env python3
"""
Tool Metadata Extractor

This module provides utilities for extracting comprehensive metadata from
tool definitions, including parameter analysis and category identification.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MCPToolMetadata:
    """Comprehensive metadata extracted from a tool definition."""

    name: str
    description: str
    category: str
    parameters: List["ParameterMetadata"] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "parameters": [p.to_dict() for p in self.parameters],
            "examples": self.examples,
            "capabilities": self.capabilities,
            "constraints": self.constraints,
        }


@dataclass
class ParameterMetadata:
    """Metadata for a tool parameter."""

    name: str
    type: str
    required: bool
    description: str
    default_value: Optional[Any] = None
    constraints: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "type": self.type,
            "required": self.required,
            "description": self.description,
            "default_value": self.default_value,
            "constraints": self.constraints,
            "examples": self.examples,
        }


class ToolMetadataExtractor:
    """
    Extracts comprehensive metadata from tool definitions.

    Analyzes tool definitions to extract names, descriptions, parameters,
    categories, and other metadata useful for semantic search and discovery.
    """

    # Category mapping based on tool names and descriptions
    CATEGORY_PATTERNS = {
        "file_ops": [
            r"\b(read|write|edit|file|directory|path|ls|list)\b",
            r"\b(create|delete|modify|save|load)\b",
            r"\bfilesystem\b",
        ],
        "execution": [r"\b(bash|shell|command|execute|run|script)\b", r"\b(terminal|console|process)\b"],
        "search": [r"\b(search|find|grep|glob|pattern|match)\b", r"\b(query|lookup|filter)\b"],
        "development": [r"\b(task|todo|plan|agent|development|coding)\b", r"\b(project|workspace|session)\b"],
        "web": [r"\b(web|http|url|fetch|download|internet)\b", r"\b(browser|online|website)\b"],
        "notebook": [r"\b(notebook|jupyter|ipynb|cell)\b", r"\b(python|interactive)\b"],
        "workflow": [r"\b(workflow|plan|mode|exit|transition)\b", r"\b(process|step|stage)\b"],
    }

    def __init__(self):
        """Initialize the metadata extractor."""
        logger.info("Initialized Tool Metadata Extractor")

    def extract_tool_metadata(self, tool_def: Dict[str, Any]) -> MCPToolMetadata:
        """
        Extract comprehensive metadata from a tool definition.

        Args:
            tool_def: Tool definition dictionary

        Returns:
            MCPToolMetadata object with extracted information
        """
        name = tool_def.get("name", "Unknown Tool")
        description = tool_def.get("description", "")

        logger.debug("Extracting metadata for tool: %s", name)

        # Identify category
        category = self._identify_category(name, description)

        # Extract parameters
        parameters = self._extract_parameters(tool_def.get("parameters", {}))

        # Extract capabilities from description
        capabilities = self._extract_capabilities(description)

        # Extract constraints from description
        constraints = self._extract_constraints(description)

        # Generate examples (placeholder for now)
        examples = self._generate_examples(name, description, parameters)

        metadata = MCPToolMetadata(
            name=name,
            description=description,
            category=category,
            parameters=parameters,
            examples=examples,
            capabilities=capabilities,
            constraints=constraints,
        )

        logger.debug("Extracted metadata for %s: category=%s, %d parameters", name, category, len(parameters))

        return metadata

    def _identify_category(self, name: str, description: str) -> str:
        """
        Identify the category of a tool based on its name and description.

        Args:
            name: Tool name
            description: Tool description

        Returns:
            Identified category string
        """
        text_to_analyze = f"{name} {description}".lower()

        # Count matches for each category
        category_scores = {}
        for category, patterns in self.CATEGORY_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_to_analyze, re.IGNORECASE))
                score += matches
            category_scores[category] = score

        # Return category with highest score
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            if category_scores[best_category] > 0:
                return best_category

        # Default category
        return "utility"

    def _extract_parameters(self, params_def: Dict[str, Any]) -> List[ParameterMetadata]:
        """
        Extract parameter metadata from parameter definitions.

        Args:
            params_def: Parameter definitions dictionary

        Returns:
            List of ParameterMetadata objects
        """
        parameters = []

        # Handle different parameter definition formats
        if isinstance(params_def, list):
            # List format (from our system tools catalog)
            for param_def in params_def:
                param = self._extract_single_parameter(param_def)
                parameters.append(param)
        elif isinstance(params_def, dict):
            # Dictionary format
            for param_name, param_info in params_def.items():
                if isinstance(param_info, dict):
                    param_def = {"name": param_name, **param_info}
                else:
                    param_def = {"name": param_name, "type": "string", "required": False}
                param = self._extract_single_parameter(param_def)
                parameters.append(param)

        return parameters

    def _extract_single_parameter(self, param_def: Dict[str, Any]) -> ParameterMetadata:
        """
        Extract metadata for a single parameter.

        Args:
            param_def: Single parameter definition

        Returns:
            ParameterMetadata object
        """
        name = param_def.get("name", "unknown")
        param_type = param_def.get("type", "string")
        required = param_def.get("required", False)
        description = param_def.get("description", "")
        default_value = param_def.get("default_value")

        # Extract constraints from description
        constraints = self._extract_parameter_constraints(description, param_type)

        # Generate examples based on parameter type and description
        examples = self._generate_parameter_examples(name, param_type, description)

        return ParameterMetadata(
            name=name,
            type=param_type,
            required=required,
            description=description,
            default_value=default_value,
            constraints=constraints,
            examples=examples,
        )

    def _extract_capabilities(self, description: str) -> List[str]:
        """
        Extract capabilities from tool description.

        Args:
            description: Tool description text

        Returns:
            List of identified capabilities
        """
        capabilities = []

        # Capability patterns
        capability_patterns = [
            (r"\b(timeout|time limit)\b", "timeout_support"),
            (r"\b(batch|multiple|bulk)\b", "batch_processing"),
            (r"\b(regex|pattern|wildcard)\b", "pattern_matching"),
            (r"\b(recursive|deep|nested)\b", "recursive_operation"),
            (r"\b(async|asynchronous|concurrent)\b", "async_support"),
            (r"\b(cache|caching|cached)\b", "caching"),
            (r"\b(stream|streaming)\b", "streaming"),
            (r"\b(filter|filtering)\b", "filtering"),
            (r"\b(sort|sorting|order)\b", "sorting"),
            (r"\b(compress|compression|zip)\b", "compression"),
        ]

        description_lower = description.lower()
        for pattern, capability in capability_patterns:
            if re.search(pattern, description_lower):
                capabilities.append(capability)

        return capabilities

    def _extract_constraints(self, description: str) -> List[str]:
        """
        Extract constraints from tool description.

        Args:
            description: Tool description text

        Returns:
            List of identified constraints
        """
        constraints = []

        # Constraint patterns
        constraint_patterns = [
            (r"must be (absolute|full) path", "requires_absolute_path"),
            (r"timeout.*?(\d+)\s*(?:ms|milliseconds|seconds|minutes)", "has_timeout_limit"),
            (r"maximum.*?(\d+)", "has_size_limit"),
            (r"only.*?supported", "limited_support"),
            (r"requires.*?permission", "requires_permissions"),
            (r"not.*?available.*?on.*?apple.*?silicon", "apple_silicon_incompatible"),
        ]

        description_lower = description.lower()
        for pattern, constraint in constraint_patterns:
            if re.search(pattern, description_lower):
                constraints.append(constraint)

        return constraints

    def _extract_parameter_constraints(self, description: str, param_type: str) -> List[str]:
        """
        Extract constraints specific to a parameter.

        Args:
            description: Parameter description
            param_type: Parameter type

        Returns:
            List of parameter constraints
        """
        constraints = []

        description_lower = description.lower()

        # Type-specific constraints
        if param_type == "string":
            if "path" in description_lower:
                constraints.append("must_be_valid_path")
            if "url" in description_lower:
                constraints.append("must_be_valid_url")
            if "regex" in description_lower or "pattern" in description_lower:
                constraints.append("must_be_valid_regex")

        elif param_type == "number" or param_type == "integer":
            # Extract numeric constraints
            if re.search(r"positive", description_lower):
                constraints.append("must_be_positive")
            if re.search(r"between.*?(\d+).*?and.*?(\d+)", description_lower):
                constraints.append("has_range_constraint")

        elif param_type == "array":
            if re.search(r"non-empty", description_lower):
                constraints.append("must_not_be_empty")

        return constraints

    def _generate_parameter_examples(self, name: str, param_type: str, description: str) -> List[str]:
        """
        Generate example values for a parameter.

        Args:
            name: Parameter name
            param_type: Parameter type
            description: Parameter description

        Returns:
            List of example values
        """
        examples = []

        name_lower = name.lower()
        description_lower = description.lower()

        if param_type == "string":
            if "path" in name_lower or "path" in description_lower:
                examples.extend(["/Users/user/file.txt", "/home/user/document.py", "./src/main.js"])
            elif "url" in name_lower or "url" in description_lower:
                examples.extend(["https://example.com", "https://api.github.com/users"])
            elif "command" in name_lower:
                examples.extend(["ls -la", "python script.py", "npm install"])
            elif "query" in name_lower:
                examples.extend(["search term", "function definition", "class implementation"])
            else:
                examples.extend(["example_value", "sample_text"])

        elif param_type == "number" or param_type == "integer":
            if "timeout" in description_lower:
                examples.extend(["5000", "10000", "30000"])
            elif "limit" in description_lower:
                examples.extend(["10", "50", "100"])
            else:
                examples.extend(["1", "5", "10"])

        elif param_type == "boolean":
            examples.extend(["true", "false"])

        elif param_type == "array":
            if "pattern" in description_lower:
                examples.extend(['["*.py", "*.js"]', '["src/**/*"]'])
            else:
                examples.extend(['["item1", "item2"]', '["value"]'])

        return examples[:3]  # Limit to 3 examples

    def _generate_examples(self, name: str, description: str, parameters: List[ParameterMetadata]) -> List[str]:
        """
        Generate usage examples for a tool.

        Args:
            name: Tool name
            description: Tool description
            parameters: List of tool parameters

        Returns:
            List of usage examples
        """
        examples = []

        # Generate basic usage examples based on tool type
        name_lower = name.lower()

        if "read" in name_lower:
            examples.extend(["Read a configuration file", "View source code file", "Check log file contents"])
        elif "write" in name_lower:
            examples.extend(["Create a new script file", "Save configuration settings", "Generate output file"])
        elif "search" in name_lower or "find" in name_lower:
            examples.extend(["Find function definitions", "Search for error patterns", "Locate configuration values"])
        elif "bash" in name_lower or "command" in name_lower:
            examples.extend(["Install dependencies", "Run build scripts", "Check system status"])

        return examples[:3]  # Limit to 3 examples
