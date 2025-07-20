#!/usr/bin/env python3
"""
MCP Metadata Extractor

This module provides sophisticated metadata extraction capabilities that can parse
tool definitions, docstrings, and schemas to create rich, searchable metadata for 
MCP tools. This enables intelligent tool discovery based on functionality, 
parameters, and usage patterns.
"""

from typing import Any, Dict, List, Optional

from docstring_parser import DocstringParser
from example_generator import ExampleGenerator
from logging_config import get_logger
from mcp_metadata_types import (
    ComplexityAnalysis,
    DocumentationAnalysis,
    MCPToolMetadata,
    ParameterAnalysis,
    ToolExample,
    UsagePattern,
)
from schema_analyzer import SchemaAnalyzer
from usage_pattern_detector import UsagePatternDetector

logger = get_logger(__name__)


class MCPMetadataExtractor:
    """Extract rich metadata from MCP tool definitions"""

    def __init__(self, schema_analyzer: SchemaAnalyzer, docstring_parser: DocstringParser):
        """Initialize the metadata extractor with required analyzers."""
        self.schema_analyzer = schema_analyzer
        self.docstring_parser = docstring_parser
        self.pattern_detector = UsagePatternDetector()
        self.example_generator = ExampleGenerator()

        logger.info("Initialized MCP Metadata Extractor")

    def extract_from_tool_definition(self, tool_def: Dict[str, Any]) -> MCPToolMetadata:
        """
        Extract comprehensive metadata from tool definition.

        Args:
            tool_def: Tool definition dictionary containing name, description, and parameters

        Returns:
            MCPToolMetadata object with extracted comprehensive metadata
        """
        name = tool_def.get("name", "Unknown Tool")
        description = tool_def.get("description", "")

        logger.debug("Extracting comprehensive metadata for tool: %s", name)

        # Parse tool documentation
        doc_analysis = self.parse_tool_documentation(description)

        # Analyze parameter schema - handle both old and new formats
        parameters_schema = tool_def.get("parameters", {})
        if isinstance(parameters_schema, list):
            # Old format - list of parameter definitions
            parameter_analyses = self.analyze_parameter_schema(parameters_schema)
        else:
            # New format - JSON schema object
            parameter_analyses = self.analyze_parameter_schema(parameters_schema)

        # Create base metadata object
        metadata = MCPToolMetadata(
            name=name,
            description=doc_analysis.description or description,
            category=self._infer_category(name, description),
            parameters=parameter_analyses,
            examples=doc_analysis.examples,
            documentation_analysis=doc_analysis,
        )

        # Infer usage patterns
        usage_patterns = self.infer_usage_patterns(metadata)
        metadata.usage_patterns = usage_patterns

        # Generate complexity analysis
        complexity_analysis = self.pattern_detector.analyze_parameter_complexity(parameter_analyses)
        metadata.complexity_analysis = complexity_analysis

        # Generate additional examples if needed
        if len(metadata.examples) < 3:  # Ensure at least 3 examples
            synthetic_examples = self.example_generator.generate_synthetic_examples(metadata)
            metadata.examples.extend(synthetic_examples)

        logger.debug(
            "Extracted metadata for %s: %d parameters, %d patterns, %d examples",
            name,
            len(parameter_analyses),
            len(usage_patterns),
            len(metadata.examples),
        )

        return metadata

    def analyze_parameter_schema(self, schema: Dict[str, Any]) -> List[ParameterAnalysis]:
        """
        Deep analysis of parameter schemas.

        Args:
            schema: JSON schema dictionary for parameters

        Returns:
            List of ParameterAnalysis objects with detailed parameter metadata
        """
        if not schema:
            return []

        return self.schema_analyzer.analyze_schema(schema)

    def parse_tool_documentation(self, docstring: str) -> DocumentationAnalysis:
        """
        Extract structured information from tool documentation.

        Args:
            docstring: Tool documentation string

        Returns:
            DocumentationAnalysis object with parsed documentation
        """
        if not docstring or not docstring.strip():
            return DocumentationAnalysis(description="")

        return self.docstring_parser.parse_structured_docstring(docstring)

    def infer_usage_patterns(self, tool_metadata: MCPToolMetadata) -> List[UsagePattern]:
        """
        Infer common usage patterns from tool structure.

        Args:
            tool_metadata: Tool metadata to analyze

        Returns:
            List of identified usage patterns
        """
        return self.pattern_detector.identify_common_patterns(tool_metadata)

    def _infer_category(self, name: str, description: str) -> str:
        """
        Infer tool category from name and description.

        Args:
            name: Tool name
            description: Tool description

        Returns:
            Inferred category string
        """
        # Category patterns (from original tool_metadata_extractor.py)
        category_patterns = {
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

        import re

        text_to_analyze = f"{name} {description}".lower()

        # Count matches for each category
        category_scores = {}
        for category, patterns in category_patterns.items():
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
