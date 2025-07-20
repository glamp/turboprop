#!/usr/bin/env python3
"""
Docstring Parser

This module provides parsing and analysis of structured docstrings in various
formats (Google, Sphinx, NumPy) to extract parameter descriptions, usage notes,
examples, and best practices from tool documentation.
"""

import re
from typing import Any, Dict, List, Optional

from logging_config import get_logger
from mcp_metadata_types import CodeExample, DocumentationAnalysis, ToolExample

# Create alias for backward compatibility
DocstringAnalysis = DocumentationAnalysis

logger = get_logger(__name__)


class DocstringParser:
    """Parse and analyze tool documentation in various formats."""

    def __init__(self):
        """Initialize the docstring parser."""
        logger.info("Initialized Docstring Parser")

    def parse_structured_docstring(self, docstring: str) -> DocstringAnalysis:
        """
        Parse Google/Sphinx/NumPy style docstrings.

        Args:
            docstring: The docstring text to parse

        Returns:
            DocstringAnalysis object with parsed sections
        """
        if not docstring or not docstring.strip():
            return DocstringAnalysis(description="")

        # Clean up docstring
        docstring = self._clean_docstring(docstring)

        logger.debug("Parsing structured docstring (%d chars)", len(docstring))

        # Try different parsing strategies
        if self._is_google_style(docstring):
            return self._parse_google_style(docstring)
        elif self._is_sphinx_style(docstring):
            return self._parse_sphinx_style(docstring)
        elif self._is_numpy_style(docstring):
            return self._parse_numpy_style(docstring)
        else:
            # Fall back to basic parsing
            return self._parse_basic_docstring(docstring)

    def extract_examples(self, docstring: str) -> List[CodeExample]:
        """
        Extract code examples from documentation.

        Args:
            docstring: Documentation text containing code examples

        Returns:
            List of CodeExample objects
        """
        examples = []

        # Find code blocks with triple backticks
        code_block_pattern = r"```(\w+)?\n(.*?)\n```"
        matches = re.findall(code_block_pattern, docstring, re.DOTALL)

        for language, code in matches:
            examples.append(CodeExample(language=language or "text", code=code.strip(), description=""))

        # Find inline code with single backticks
        inline_code_pattern = r"`([^`]+)`"
        inline_matches = re.findall(inline_code_pattern, docstring)

        for code in inline_matches:
            if len(code.strip()) > 10:  # Only longer code snippets
                examples.append(CodeExample(language="text", code=code, description="Inline code example"))

        return examples

    def identify_best_practices(self, docstring: str) -> List[str]:
        """
        Extract best practices and recommendations.

        Args:
            docstring: Documentation text

        Returns:
            List of best practice strings
        """
        practices = []

        # Patterns for identifying best practices
        practice_patterns = [
            r"best practice[s]?[:\-\s]*([^\n\.]+)",
            r"tip[:\-\s]*([^\n\.]+)",
            r"recommendation[:\-\s]*([^\n\.]+)",
            r"should[:\s]+([^\n\.]+)",
            r"always[:\s]+([^\n\.]+)",
            r"never[:\s]+([^\n\.]+)",
            r"avoid[:\s]+([^\n\.]+)",
            r"performance[:\s]+([^\n\.]+)",
        ]

        docstring_lower = docstring.lower()

        for pattern in practice_patterns:
            matches = re.findall(pattern, docstring_lower, re.IGNORECASE)
            for match in matches:
                if len(match.strip()) > 5:  # Filter out very short matches
                    practices.append(match.strip().capitalize())

        # Look for bullet point practices
        bullet_pattern = r"^\s*[\-\*]\s*([^\n]+)"
        lines = docstring.split("\n")

        for line in lines:
            if any(keyword in line.lower() for keyword in ["practice", "tip", "should", "avoid"]):
                match = re.search(bullet_pattern, line)
                if match:
                    practices.append(match.group(1).strip())

        return practices[:5]  # Limit to 5 practices

    def _clean_docstring(self, docstring: str) -> str:
        """Clean up docstring formatting."""
        # Remove leading/trailing whitespace
        docstring = docstring.strip()

        # Remove common indentation
        lines = docstring.split("\n")
        if len(lines) > 1:
            # Find minimum indentation (excluding empty lines)
            min_indent = float("inf")
            for line in lines[1:]:  # Skip first line
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    min_indent = min(min_indent, indent)

            if min_indent != float("inf") and min_indent > 0:
                # Remove common indentation
                cleaned_lines = [lines[0]]  # Keep first line as-is
                for line in lines[1:]:
                    if line.strip():
                        cleaned_lines.append(line[min_indent:])
                    else:
                        cleaned_lines.append("")
                docstring = "\n".join(cleaned_lines)

        return docstring

    def _is_google_style(self, docstring: str) -> bool:
        """Check if docstring follows Google style."""
        google_sections = [
            "Args:",
            "Arguments:",
            "Returns:",
            "Return:",
            "Raises:",
            "Example:",
            "Examples:",
            "Note:",
            "Warning:",
        ]
        return any(section in docstring for section in google_sections)

    def _is_sphinx_style(self, docstring: str) -> bool:
        """Check if docstring follows Sphinx style."""
        sphinx_patterns = [":param", ":type", ":returns:", ":rtype:", ":raises:"]
        return any(pattern in docstring for pattern in sphinx_patterns)

    def _is_numpy_style(self, docstring: str) -> bool:
        """Check if docstring follows NumPy style."""
        numpy_sections = ["Parameters\n--------", "Returns\n-------", "Examples\n--------"]
        return any(section in docstring for section in numpy_sections)

    def _parse_google_style(self, docstring: str) -> DocstringAnalysis:
        """Parse Google-style docstring."""
        sections = self._split_google_sections(docstring)

        description = sections.get("description", "")
        parameters = self._parse_google_parameters(sections.get("Args", ""))
        examples = self._parse_google_examples(sections.get("Examples", sections.get("Example", "")))
        notes = self._extract_notes(sections.get("Note", ""))
        warnings = self._extract_warnings(sections.get("Warning", ""))
        return_info = self._parse_google_returns(sections.get("Returns", sections.get("Return", "")))

        return DocstringAnalysis(
            description=description,
            parameters=parameters,
            examples=examples,
            notes=notes,
            warnings=warnings,
            return_info=return_info,
        )

    def _parse_sphinx_style(self, docstring: str) -> DocstringAnalysis:
        """Parse Sphinx-style docstring."""
        # Extract main description (everything before first :param)
        description_match = re.search(r"^(.*?)(?=:param|:returns?|:rtype|$)", docstring, re.DOTALL)
        description = description_match.group(1).strip() if description_match else ""

        # Parse parameters
        parameters = []
        param_pattern = r":param\s+([^:]+):\s*([^\n]+)"
        param_matches = re.findall(param_pattern, docstring)

        for param_name, param_desc in param_matches:
            # Look for corresponding type information
            type_pattern = rf":type\s+{re.escape(param_name)}:\s*([^\n]+)"
            type_match = re.search(type_pattern, docstring)
            param_type = type_match.group(1).strip() if type_match else "unknown"

            # Determine if required (basic heuristic)
            required = "optional" not in param_desc.lower()

            parameters.append(
                {
                    "name": param_name.strip(),
                    "type": param_type,
                    "description": param_desc.strip(),
                    "required": required,
                    "default": self._extract_default_value(param_desc),
                }
            )

        # Parse return information
        return_info = None
        return_match = re.search(r":returns?:\s*([^\n]+)", docstring)
        rtype_match = re.search(r":rtype:\s*([^\n]+)", docstring)

        if return_match or rtype_match:
            return_info = {
                "description": return_match.group(1).strip() if return_match else "",
                "type": rtype_match.group(1).strip() if rtype_match else "unknown",
            }

        # Extract examples
        examples = []
        example_pattern = r"\.\.\s+example::(.*?)(?=\n\S|\Z)"
        example_matches = re.findall(example_pattern, docstring, re.DOTALL | re.IGNORECASE)

        for example_text in example_matches:
            examples.extend(self._parse_example_text(example_text))

        return DocstringAnalysis(
            description=description,
            parameters=parameters,
            examples=examples,
            return_info=return_info,
        )

    def _parse_numpy_style(self, docstring: str) -> DocstringAnalysis:
        """Parse NumPy-style docstring."""
        # Split into sections based on underline patterns
        sections = {}
        current_section = "description"
        current_content = []

        lines = docstring.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Check if next line is underline
            if i + 1 < len(lines) and re.match(r"^-{3,}$", lines[i + 1].strip()):
                # Save current section
                if current_content:
                    sections[current_section] = "\n".join(current_content).strip()
                    current_content = []

                # Start new section
                current_section = line.lower()
                i += 2  # Skip the underline
                continue

            current_content.append(lines[i])
            i += 1

        # Save last section
        if current_content:
            sections[current_section] = "\n".join(current_content).strip()

        # Parse sections
        description = sections.get("description", "").strip()
        parameters = self._parse_numpy_parameters(sections.get("parameters", ""))
        examples = self._parse_numpy_examples(sections.get("examples", ""))

        return DocstringAnalysis(
            description=description,
            parameters=parameters,
            examples=examples,
        )

    def _parse_basic_docstring(self, docstring: str) -> DocstringAnalysis:
        """Parse basic unstructured docstring."""
        # Just extract description and try to find examples
        examples = []

        # Look for example patterns in text
        example_patterns = [
            r"example[:\-\s]*(.*?)(?=\n\n|\Z)",
            r"usage[:\-\s]*(.*?)(?=\n\n|\Z)",
        ]

        for pattern in example_patterns:
            matches = re.findall(pattern, docstring, re.IGNORECASE | re.DOTALL)
            for match in matches:
                examples.extend(self._parse_example_text(match))

        return DocstringAnalysis(
            description=docstring.strip(),
            examples=examples,
        )

    def _split_google_sections(self, docstring: str) -> Dict[str, str]:
        """Split Google-style docstring into sections."""
        sections = {}
        lines = docstring.split("\n")
        current_section = "description"
        current_content = []

        for line in lines:
            # Check for section headers
            if re.match(r"^\s*(Args|Arguments|Returns?|Raises?|Examples?|Notes?|Warnings?):\s*$", line.strip()):
                # Save current section
                if current_content:
                    sections[current_section] = "\n".join(current_content).strip()
                    current_content = []

                # Start new section
                current_section = line.strip().rstrip(":")
                continue

            current_content.append(line)

        # Save last section
        if current_content:
            sections[current_section] = "\n".join(current_content).strip()

        return sections

    def _parse_google_parameters(self, args_text: str) -> List[Dict[str, Any]]:
        """Parse Google-style Args section."""
        parameters = []
        if not args_text:
            return parameters

        # Pattern: parameter_name (type, optional): description
        param_pattern = r"^\s*(\w+)\s*\(([^)]+)\):\s*(.+)"

        for line in args_text.split("\n"):
            line = line.strip()
            if not line:
                continue

            match = re.match(param_pattern, line)
            if match:
                param_name = match.group(1)
                type_info = match.group(2)
                description = match.group(3)

                # Parse type and required status
                required = "optional" not in type_info.lower()
                param_type = re.sub(r",?\s*optional", "", type_info, flags=re.IGNORECASE).strip()

                parameters.append(
                    {
                        "name": param_name,
                        "type": param_type,
                        "description": description,
                        "required": required,
                        "default": self._extract_default_value(description),
                    }
                )

        return parameters

    def _parse_google_examples(self, examples_text: str) -> List[ToolExample]:
        """Parse Google-style Examples section."""
        if not examples_text:
            return []

        return self._parse_example_text(examples_text)

    def _parse_google_returns(self, returns_text: str) -> Optional[Dict[str, Any]]:
        """Parse Google-style Returns section."""
        if not returns_text:
            return None

        # Pattern: type: description
        match = re.match(r"^\s*([^:]+):\s*(.+)", returns_text.strip())
        if match:
            return {
                "type": match.group(1).strip(),
                "description": match.group(2).strip(),
            }
        else:
            return {
                "type": "unknown",
                "description": returns_text.strip(),
            }

    def _parse_numpy_parameters(self, params_text: str) -> List[Dict[str, Any]]:
        """Parse NumPy-style Parameters section."""
        parameters = []
        if not params_text:
            return parameters

        # Pattern: parameter_name : type
        #          Description text
        current_param = None

        for line in params_text.split("\n"):
            # Check for parameter definition line
            if ":" in line and not line.startswith(" "):
                # Save previous parameter
                if current_param:
                    parameters.append(current_param)

                # Parse new parameter
                name_type = line.split(":", 1)
                param_name = name_type[0].strip()
                param_type = name_type[1].strip() if len(name_type) > 1 else "unknown"

                current_param = {
                    "name": param_name,
                    "type": param_type,
                    "description": "",
                    "required": "optional" not in param_type.lower(),
                    "default": None,
                }

            # Continuation of description
            elif current_param and line.strip():
                if current_param["description"]:
                    current_param["description"] += " " + line.strip()
                else:
                    current_param["description"] = line.strip()

        # Save last parameter
        if current_param:
            parameters.append(current_param)

        return parameters

    def _parse_numpy_examples(self, examples_text: str) -> List[ToolExample]:
        """Parse NumPy-style Examples section."""
        if not examples_text:
            return []

        return self._parse_example_text(examples_text)

    def _parse_example_text(self, text: str) -> List[ToolExample]:
        """Parse example text and extract tool examples."""
        examples = []

        # Look for code blocks
        code_examples = self.extract_examples(text)

        for i, code_ex in enumerate(code_examples):
            # Try to extract use case from surrounding text
            use_case = f"Example {i+1}"

            # Look for descriptive text before the code
            lines_before_code = text.split(code_ex.code)[0].split("\n")
            for line in reversed(lines_before_code):
                line = line.strip()
                if line and not line.startswith("```") and len(line) > 10:
                    use_case = line
                    break

            examples.append(
                ToolExample(
                    use_case=use_case,
                    example_call=code_ex.code,
                    language=code_ex.language,
                )
            )

        return examples

    def _extract_notes(self, notes_text: str) -> List[str]:
        """Extract individual notes from notes section."""
        if not notes_text:
            return []

        notes = []
        for line in notes_text.split("\n"):
            line = line.strip()
            if line and len(line) > 10:
                notes.append(line)

        return notes

    def _extract_warnings(self, warnings_text: str) -> List[str]:
        """Extract individual warnings from warnings section."""
        if not warnings_text:
            return []

        warnings = []
        for line in warnings_text.split("\n"):
            line = line.strip()
            if line and len(line) > 10:
                warnings.append(line)

        return warnings

    def _extract_default_value(self, description: str) -> Optional[str]:
        """Extract default value from parameter description."""
        # Common patterns for default values
        default_patterns = [
            r"defaults?\s+to\s+([^.\s]+)",
            r"default[:\s]*([^.\s]+)",
            r"\(default[:\s]*([^)]+)\)",
        ]

        for pattern in default_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None
