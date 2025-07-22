#!/usr/bin/env python3
"""
Tests for MCP Tool Metadata Extraction System

This test suite validates the sophisticated metadata extraction capabilities
for MCP tools, including schema analysis, docstring parsing, usage pattern
recognition, and example generation.
"""

from unittest.mock import Mock, patch

import pytest

from turboprop.docstring_parser import DocstringParser
from turboprop.example_generator import ExampleGenerator

# Import the modules we're testing (these will be created)
from turboprop.mcp_metadata_extractor import MCPMetadataExtractor
from turboprop.mcp_metadata_types import (
    ComplexityAnalysis,
    DocumentationAnalysis,
    MCPToolMetadata,
    ParameterAnalysis,
    ToolExample,
)
from turboprop.schema_analyzer import SchemaAnalyzer
from turboprop.usage_pattern_detector import UsagePatternDetector


class TestSchemaAnalyzer:
    """Test cases for JSON schema analysis."""

    def test_analyze_simple_schema(self):
        """Test analysis of a simple parameter schema."""
        schema = {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file",
                },
                "timeout": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 600000,
                    "description": "Timeout in milliseconds",
                },
            },
            "required": ["file_path"],
        }

        analyzer = SchemaAnalyzer()
        analysis = analyzer.analyze_schema(schema)

        assert len(analysis) == 2
        file_param = next(p for p in analysis if p.name == "file_path")
        assert file_param.type == "string"
        assert file_param.required is True
        assert file_param.description == "The absolute path to the file"

        timeout_param = next(p for p in analysis if p.name == "timeout")
        assert timeout_param.type == "number"
        assert timeout_param.required is False
        assert timeout_param.constraints["minimum"] == 0
        assert timeout_param.constraints["maximum"] == 600000

    def test_analyze_nested_schema(self):
        """Test analysis of nested object schema."""
        schema = {
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_string": {"type": "string", "description": "Text to replace"},
                            "new_string": {"type": "string", "description": "Replacement text"},
                        },
                        "required": ["old_string", "new_string"],
                    },
                    "minItems": 1,
                },
            },
            "required": ["edits"],
        }

        analyzer = SchemaAnalyzer()
        analysis = analyzer.analyze_schema(schema)

        edits_param = analysis[0]
        assert edits_param.name == "edits"
        assert edits_param.type == "array"
        assert edits_param.constraints["minItems"] == 1
        assert edits_param.complexity_score > 0.5  # Should be complex due to nested structure

    def test_extract_parameter_constraints(self):
        """Test extraction of various parameter constraints."""
        analyzer = SchemaAnalyzer()

        # String constraints
        string_constraints = analyzer.extract_constraints(
            {
                "type": "string",
                "pattern": r"^[a-zA-Z]+$",
                "minLength": 1,
                "maxLength": 100,
                "enum": ["option1", "option2"],
            }
        )
        assert "pattern" in string_constraints
        assert "minLength" in string_constraints
        assert "enum" in string_constraints

        # Number constraints
        number_constraints = analyzer.extract_constraints(
            {
                "type": "number",
                "minimum": 0,
                "maximum": 100,
                "exclusiveMinimum": True,
            }
        )
        assert number_constraints["minimum"] == 0
        assert number_constraints["maximum"] == 100
        assert number_constraints["exclusiveMinimum"] is True

    def test_calculate_complexity_score(self):
        """Test complexity score calculation."""
        analyzer = SchemaAnalyzer()

        # Simple parameter
        simple_param = ParameterAnalysis(
            name="file_path",
            type="string",
            required=True,
            description="A file path",
            constraints={},
            default_value=None,
            examples=[],
            complexity_score=0.0,
        )
        score = analyzer.calculate_complexity_score(simple_param)
        assert 0.0 <= score <= 1.0
        assert score < 0.3  # Should be low complexity

        # Complex parameter with many constraints
        complex_param = ParameterAnalysis(
            name="edits",
            type="array",
            required=True,
            description="Complex nested array parameter",
            constraints={"minItems": 1, "items": {"type": "object"}},
            default_value=None,
            examples=[],
            complexity_score=0.0,
        )
        complex_score = analyzer.calculate_complexity_score(complex_param)
        assert complex_score > score  # Should be higher than simple parameter


class TestDocstringParser:
    """Test cases for docstring parsing and analysis."""

    def test_parse_google_style_docstring(self):
        """Test parsing Google-style docstrings."""
        docstring = """
        Executes a bash command in a persistent shell session.

        This tool provides secure command execution with timeout support
        and proper error handling for development workflows.

        Args:
            command (str): The command to execute
            timeout (int, optional): Timeout in milliseconds. Defaults to 120000.
            description (str, optional): A description of what the command does.

        Returns:
            CommandResult: The result of command execution with stdout/stderr

        Example:
            Execute a simple command:
            ```python
            result = bash_tool(command="ls -la", timeout=5000)
            ```

        Note:
            Commands are executed in a persistent shell session.
            Use absolute paths for reliability.

        Warning:
            Avoid running untrusted commands as they execute with full permissions.
        """

        parser = DocstringParser()
        analysis = parser.parse_structured_docstring(docstring)

        assert analysis.description is not None
        assert "secure command execution" in analysis.description
        assert len(analysis.parameters) == 3

        command_param = next(p for p in analysis.parameters if p["name"] == "command")
        assert command_param["type"] == "str"
        assert command_param["required"] is True

        timeout_param = next(p for p in analysis.parameters if p["name"] == "timeout")
        assert timeout_param["required"] is False
        assert timeout_param["default"] == "120000"

        assert len(analysis.examples) > 0
        assert "ls -la" in analysis.examples[0].example_call

        assert any("persistent shell session" in note for note in analysis.notes)
        assert any("untrusted commands" in warning for warning in analysis.warnings)

    def test_parse_sphinx_style_docstring(self):
        """Test parsing Sphinx-style docstrings."""
        docstring = """
        Read a file from the local filesystem.

        :param file_path: The absolute path to the file to read
        :type file_path: str
        :param limit: The number of lines to read
        :type limit: int, optional
        :returns: The file contents
        :rtype: str
        :raises FileNotFoundError: When the file does not exist

        .. example::
           Read a Python file:

           >>> content = read_file("/path/to/script.py")
        """

        parser = DocstringParser()
        analysis = parser.parse_structured_docstring(docstring)

        assert analysis.description == "Read a file from the local filesystem."
        assert len(analysis.parameters) == 2

        file_param = analysis.parameters[0]
        assert file_param["name"] == "file_path"
        assert file_param["type"] == "str"

    def test_extract_code_examples(self):
        """Test extraction of code examples from docstrings."""
        docstring = """
        Tool for searching files.

        Example usage:
        ```bash
        grep "pattern" file.txt
        ```

        Python example:
        ```python
        result = grep_tool(pattern="class.*:", path="src/")
        if result.matches:
            print(f"Found {len(result.matches)} matches")
        ```
        """

        parser = DocstringParser()
        examples = parser.extract_examples(docstring)

        assert len(examples) >= 2
        # Check that we extracted examples with the right content
        all_code = " ".join(ex.code for ex in examples)
        assert "grep" in all_code
        assert "grep_tool" in all_code

    def test_identify_best_practices(self):
        """Test identification of best practices from docstrings."""
        docstring = """
        File editing tool.

        Best practices:
        - Always backup files before editing
        - Use absolute paths for reliability
        - Test changes in development environment first

        Performance tip: For large files, consider using streaming operations.

        Warning: This tool will overwrite existing files.
        """

        parser = DocstringParser()
        practices = parser.identify_best_practices(docstring)

        practices_text = " ".join(practices).lower()
        assert "backup" in practices_text
        assert "streaming" in practices_text
        assert len(practices) > 0  # At least some practices were found


class TestUsagePatternDetector:
    """Test cases for usage pattern detection and analysis."""

    def test_analyze_parameter_complexity(self):
        """Test analysis of parameter complexity."""
        parameters = [
            ParameterAnalysis(
                name="file_path",
                type="string",
                required=True,
                description="File path",
                constraints={},
                default_value=None,
                examples=[],
                complexity_score=0.1,
            ),
            ParameterAnalysis(
                name="edits",
                type="array",
                required=True,
                description="Complex edit operations",
                constraints={"minItems": 1, "items": {"type": "object"}},
                default_value=None,
                examples=[],
                complexity_score=0.8,
            ),
            ParameterAnalysis(
                name="timeout",
                type="number",
                required=False,
                description="Timeout value",
                constraints={"minimum": 0},
                default_value=5000,
                examples=[],
                complexity_score=0.2,
            ),
        ]

        detector = UsagePatternDetector()
        analysis = detector.analyze_parameter_complexity(parameters)

        assert analysis.total_parameters == 3
        assert analysis.required_parameters == 2
        assert analysis.optional_parameters == 1
        assert analysis.complex_parameters == 1  # The edits parameter
        assert 0.0 < analysis.overall_complexity < 1.0

    def test_identify_common_patterns(self):
        """Test identification of common usage patterns."""
        tool_metadata = MCPToolMetadata(
            name="Edit",
            description="Performs exact string replacements in files",
            category="file_ops",
            parameters=[
                ParameterAnalysis(
                    name="file_path",
                    type="string",
                    required=True,
                    description="The absolute path to the file to modify",
                    constraints={},
                    default_value=None,
                    examples=[],
                    complexity_score=0.1,
                ),
                ParameterAnalysis(
                    name="old_string",
                    type="string",
                    required=True,
                    description="The text to replace",
                    constraints={},
                    default_value=None,
                    examples=[],
                    complexity_score=0.1,
                ),
                ParameterAnalysis(
                    name="new_string",
                    type="string",
                    required=True,
                    description="The text to replace it with",
                    constraints={},
                    default_value=None,
                    examples=[],
                    complexity_score=0.1,
                ),
            ],
            examples=[],
            usage_patterns=[],
            complexity_analysis=None,
            documentation_analysis=None,
        )

        detector = UsagePatternDetector()
        patterns = detector.identify_common_patterns(tool_metadata)

        # Should identify file editing pattern
        edit_pattern = next((p for p in patterns if "file" in p.pattern_name.lower()), None)
        assert edit_pattern is not None
        assert edit_pattern.complexity_level in ["basic", "intermediate", "advanced"]
        assert 0.0 <= edit_pattern.success_probability <= 1.0

    def test_generate_complexity_score(self):
        """Test overall tool complexity score generation."""
        detector = UsagePatternDetector()

        # Simple tool (Read)
        simple_tool = MCPToolMetadata(
            name="Read",
            description="Read file contents",
            category="file_ops",
            parameters=[
                ParameterAnalysis(
                    name="file_path",
                    type="string",
                    required=True,
                    description="Path to file",
                    constraints={},
                    default_value=None,
                    examples=[],
                    complexity_score=0.1,
                )
            ],
            examples=[],
            usage_patterns=[],
            complexity_analysis=None,
            documentation_analysis=None,
        )
        simple_score = detector.generate_complexity_score(simple_tool)
        assert 0.0 <= simple_score <= 0.4

        # Complex tool (MultiEdit)
        complex_tool = MCPToolMetadata(
            name="MultiEdit",
            description="Makes multiple edits to a single file in one operation",
            category="file_ops",
            parameters=[
                ParameterAnalysis(
                    name="file_path",
                    type="string",
                    required=True,
                    description="Path to file",
                    constraints={},
                    default_value=None,
                    examples=[],
                    complexity_score=0.2,
                ),
                ParameterAnalysis(
                    name="edits",
                    type="array",
                    required=True,
                    description="Array of edit operations",
                    constraints={"minItems": 1, "items": {"type": "object"}},
                    default_value=None,
                    examples=[],
                    complexity_score=0.9,
                ),
            ],
            examples=[],
            usage_patterns=[],
            complexity_analysis=None,
            documentation_analysis=None,
        )
        complex_score = detector.generate_complexity_score(complex_tool)
        assert complex_score > simple_score
        assert 0.2 <= complex_score <= 1.0  # Adjusted for more realistic scoring


class TestExampleGenerator:
    """Test cases for example extraction and generation."""

    def test_extract_examples_from_documentation(self):
        """Test extraction of examples from tool documentation."""
        documentation = r"""
        Search tool for finding patterns in files.

        Example 1: Find all Python functions
        ```python
        grep_tool(pattern=r"def \w+\(", path="src/", glob="*.py")
        ```

        Example 2: Search in specific file
        ```bash
        grep "import" main.py
        ```
        """

        generator = ExampleGenerator()
        examples = generator.extract_examples_from_documentation(documentation)

        assert len(examples) >= 2
        python_example = next((e for e in examples if "def " in e.example_call), None)
        assert python_example is not None
        assert python_example.use_case == "Find all Python functions"

    def test_generate_synthetic_examples(self):
        """Test generation of synthetic usage examples."""
        tool_metadata = MCPToolMetadata(
            name="Bash",
            description="Execute bash commands with timeout support",
            category="execution",
            parameters=[
                ParameterAnalysis(
                    name="command",
                    type="string",
                    required=True,
                    description="The command to execute",
                    constraints={},
                    default_value=None,
                    examples=["ls -la", "python script.py"],
                    complexity_score=0.2,
                ),
                ParameterAnalysis(
                    name="timeout",
                    type="number",
                    required=False,
                    description="Timeout in milliseconds",
                    constraints={"minimum": 0, "maximum": 600000},
                    default_value=120000,
                    examples=["5000", "30000"],
                    complexity_score=0.3,
                ),
            ],
            examples=[],
            usage_patterns=[],
            complexity_analysis=None,
            documentation_analysis=None,
        )

        generator = ExampleGenerator()
        examples = generator.generate_synthetic_examples(tool_metadata)

        assert len(examples) > 0
        # Should generate examples for common command types
        common_commands = ["ls", "python", "npm", "git"]
        found_common = any(any(cmd in ex.example_call for cmd in common_commands) for ex in examples)
        assert found_common

    def test_validate_generated_examples(self):
        """Test validation of generated examples against schema."""
        parameter = ParameterAnalysis(
            name="timeout",
            type="number",
            required=False,
            description="Timeout in milliseconds",
            constraints={"minimum": 0, "maximum": 600000},
            default_value=None,
            examples=[],
            complexity_score=0.2,
        )

        generator = ExampleGenerator()

        # Valid example
        valid_result = generator.validate_example_value("5000", parameter)
        assert valid_result.is_valid is True

        # Invalid example (exceeds maximum)
        invalid_result = generator.validate_example_value("700000", parameter)
        assert invalid_result.is_valid is False
        assert "maximum" in invalid_result.error_message


class TestMCPMetadataExtractor:
    """Test cases for the main metadata extractor."""

    def test_extract_comprehensive_metadata(self):
        """Test comprehensive metadata extraction from tool definition."""
        tool_def = {
            "name": "Read",
            "description": """
            Reads a file from the local filesystem with line offset and limit support.

            Args:
                file_path (str): The absolute path to the file to read
                limit (int, optional): The number of lines to read
                offset (int, optional): The line number to start reading from

            Returns:
                str: The file contents

            Example:
                Read first 100 lines of a log file:
                ```python
                content = read_file("/var/log/app.log", limit=100)
                ```
            """,
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "The absolute path to the file to read"},
                    "limit": {"type": "number", "description": "The number of lines to read"},
                    "offset": {"type": "number", "description": "The line number to start reading from"},
                },
                "required": ["file_path"],
            },
        }

        # Mock the dependencies with proper return values
        schema_analyzer = Mock(spec=SchemaAnalyzer)
        schema_analyzer.analyze_schema.return_value = [
            ParameterAnalysis(
                name="file_path",
                type="string",
                required=True,
                description="The absolute path to the file to read",
                constraints={},
                default_value=None,
                examples=[],
                complexity_score=0.1,
            ),
            ParameterAnalysis(
                name="limit",
                type="number",
                required=False,
                description="The number of lines to read",
                constraints={},
                default_value=None,
                examples=[],
                complexity_score=0.2,
            ),
            ParameterAnalysis(
                name="offset",
                type="number",
                required=False,
                description="The line number to start reading from",
                constraints={},
                default_value=None,
                examples=[],
                complexity_score=0.2,
            ),
        ]

        docstring_parser = Mock(spec=DocstringParser)
        docstring_parser.parse_structured_docstring.return_value = DocumentationAnalysis(
            description="Reads a file from the local filesystem with line offset and limit support.",
            parameters=[
                {
                    "name": "file_path",
                    "type": "str",
                    "required": True,
                    "description": "The absolute path to the file to read",
                },
                {"name": "limit", "type": "int", "required": False, "description": "The number of lines to read"},
                {
                    "name": "offset",
                    "type": "int",
                    "required": False,
                    "description": "The line number to start reading from",
                },
            ],
            examples=[ToolExample(use_case="Read log file", example_call='read_file("/var/log/app.log", limit=100)')],
        )

        extractor = MCPMetadataExtractor(schema_analyzer, docstring_parser)
        metadata = extractor.extract_from_tool_definition(tool_def)

        assert metadata.name == "Read"
        assert "filesystem" in metadata.description
        assert len(metadata.parameters) == 3

        # Verify all components were called
        schema_analyzer.analyze_schema.assert_called_once()
        docstring_parser.parse_structured_docstring.assert_called_once()

    @patch("turboprop.mcp_metadata_extractor.UsagePatternDetector")
    @patch("turboprop.mcp_metadata_extractor.ExampleGenerator")
    def test_integration_with_pattern_detector(self, mock_example_gen, mock_pattern_detector):
        """Test integration with usage pattern detector and example generator."""
        # Setup mocks
        mock_detector = Mock()
        mock_generator = Mock()
        mock_pattern_detector.return_value = mock_detector
        mock_example_gen.return_value = mock_generator

        # Mock the detector methods
        mock_detector.identify_common_patterns.return_value = []
        mock_detector.analyze_parameter_complexity.return_value = ComplexityAnalysis(
            total_parameters=0,
            required_parameters=0,
            optional_parameters=0,
            complex_parameters=0,
            overall_complexity=0.0,
        )

        # Mock the generator methods
        mock_generator.generate_synthetic_examples.return_value = []

        schema_analyzer = Mock(spec=SchemaAnalyzer)
        schema_analyzer.analyze_schema.return_value = []

        docstring_parser = Mock(spec=DocstringParser)
        docstring_parser.parse_structured_docstring.return_value = DocumentationAnalysis(
            description="Test", examples=[]
        )

        extractor = MCPMetadataExtractor(schema_analyzer, docstring_parser)

        tool_def = {"name": "TestTool", "description": "Test", "parameters": {}}
        _ = extractor.extract_from_tool_definition(tool_def)

        # Verify pattern detection and example generation were called
        mock_detector.identify_common_patterns.assert_called_once()
        mock_generator.generate_synthetic_examples.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
