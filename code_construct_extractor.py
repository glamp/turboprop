#!/usr/bin/env python3
"""
code_construct_extractor.py: AST-based code construct extraction for semantic indexing.

This module provides intelligent extraction of programming constructs (functions, classes,
variables, imports) from source code files using language-specific parsing techniques.

Classes:
- CodeConstruct: Data class representing an extracted programming construct
- PythonConstructExtractor: Python AST-based extraction
- JavaScriptConstructExtractor: JavaScript pattern-based extraction
- CodeConstructExtractor: Main orchestrator for language-aware extraction
"""

import ast
import hashlib
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Union

from language_detection import LanguageDetector

logger = logging.getLogger(__name__)

# Constants for function/method length estimation
DEFAULT_FUNCTION_LENGTH_ESTIMATE = 10
FALLBACK_LINE_ESTIMATE = 10


@dataclass
class CodeConstruct:
    """
    Represents an extracted code construct with metadata.

    This class encapsulates all information about a programming construct
    (function, class, variable, import) extracted from source code.
    """
    construct_type: str  # 'function', 'class', 'method', 'variable', 'import'
    name: str
    start_line: int
    end_line: int
    signature: str
    docstring: Optional[str] = None
    parent_construct_id: Optional[str] = None

    def compute_construct_id(self, file_id: str) -> str:
        """
        Compute a unique identifier for this construct.

        Args:
            file_id: The ID of the file containing this construct

        Returns:
            SHA-256 hash string uniquely identifying this construct
        """
        # Create a unique identifier based on file_id + signature + location
        identifier_text = f"{file_id}:{self.signature}:{self.start_line}:{self.end_line}"
        return hashlib.sha256(identifier_text.encode()).hexdigest()

    def get_embedding_text(self) -> str:
        """
        Get the text representation for embedding generation.

        Returns:
            String combining signature and docstring for embedding
        """
        parts = [self.signature]
        if self.docstring:
            parts.append(self.docstring)
        return "\n".join(parts)


class PythonConstructExtractor:
    """
    Python-specific construct extractor using AST parsing.

    Extracts functions, classes, methods, variables, and imports with their
    metadata including signatures, docstrings, and line numbers.
    """

    def extract_constructs(self, content: str, file_path: str) -> List[CodeConstruct]:
        """
        Extract Python constructs using AST parsing.

        Args:
            content: Python source code content
            file_path: Path to the source file

        Returns:
            List of CodeConstruct objects
        """
        try:
            tree = ast.parse(content)
            lines = content.split('\n')
            constructs = []
            processed_classes = set()
            processed_functions = set()

            # Extract module-level constructs only (not nested ones)
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    construct = self._extract_function(node, lines)
                    if construct:
                        constructs.append(construct)
                        processed_functions.add(id(node))
                elif isinstance(node, ast.ClassDef):
                    class_construct = self._extract_class(node, lines)
                    if class_construct:
                        constructs.append(class_construct)
                        processed_classes.add(id(node))
                        # Extract class methods
                        methods = self._extract_class_methods(node, lines, class_construct.compute_construct_id(""))
                        constructs.extend(methods)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_constructs = self._extract_import(node, lines)
                    constructs.extend(import_constructs)
                elif isinstance(node, ast.Assign):
                    variable_constructs = self._extract_variable_assignment(node, lines)
                    constructs.extend(variable_constructs)

            return constructs

        except SyntaxError as syntax_error:
            logger.warning("Python syntax error in %s: %s", file_path, syntax_error)
            return []
        except Exception as error:
            # Broad catch is necessary as AST parsing can fail in various ways beyond SyntaxError
            logger.error("Error extracting Python constructs from %s: %s", file_path, error)
            return []

    def _extract_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
                          lines: List[str]) -> Optional[CodeConstruct]:
        """Extract a function definition."""
        try:
            # Build function signature
            signature_parts = []
            if isinstance(node, ast.AsyncFunctionDef):
                signature_parts.append("async")

            signature_parts.append("def")
            signature_parts.append(node.name)

            # Extract parameters
            args = []
            for arg in node.args.args:
                arg_str = arg.arg
                if arg.annotation:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                args.append(arg_str)

            # Add default parameters
            defaults_offset = len(node.args.args) - len(node.args.defaults)
            for i, default in enumerate(node.args.defaults):
                args[defaults_offset + i] += f" = {ast.unparse(default)}"

            signature = f"{' '.join(signature_parts)}({', '.join(args)})"

            # Add return type annotation if present
            if node.returns:
                signature += f" -> {ast.unparse(node.returns)}"

            signature += ":"

            # Extract docstring
            docstring = ast.get_docstring(node)

            return CodeConstruct(
                construct_type="function",
                name=node.name,
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                signature=signature,
                docstring=docstring
            )

        except Exception as error:
            logger.debug("Error extracting function %s: %s", getattr(node, 'name', 'unknown'), error)
            return None

    def _extract_class(self, node: ast.ClassDef, lines: List[str]) -> Optional[CodeConstruct]:
        """Extract a class definition."""
        try:
            # Build class signature
            signature = f"class {node.name}"

            # Add base classes if present
            if node.bases:
                base_names = [ast.unparse(base) for base in node.bases]
                signature += f"({', '.join(base_names)})"

            signature += ":"

            # Extract docstring
            docstring = ast.get_docstring(node)

            return CodeConstruct(
                construct_type="class",
                name=node.name,
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                signature=signature,
                docstring=docstring
            )

        except Exception as error:
            logger.debug("Error extracting class %s: %s", getattr(node, 'name', 'unknown'), error)
            return None

    def _extract_class_methods(self, class_node: ast.ClassDef, lines: List[str], parent_id: str) -> List[CodeConstruct]:
        """Extract methods from a class."""
        methods = []

        for node in class_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_construct = self._extract_function(node, lines)
                if method_construct:
                    method_construct.construct_type = "method"
                    method_construct.parent_construct_id = parent_id
                    methods.append(method_construct)

        return methods

    def _extract_import(self, node: Union[ast.Import, ast.ImportFrom], lines: List[str]) -> List[CodeConstruct]:
        """Extract import statements."""
        constructs = []

        try:
            if isinstance(node, ast.Import):
                # For simple imports, create one construct per import statement
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    signature = f"import {alias.name}"
                    if alias.asname:
                        signature += f" as {alias.asname}"

                    constructs.append(CodeConstruct(
                        construct_type="import",
                        name=name,
                        start_line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        signature=signature
                    ))

            elif isinstance(node, ast.ImportFrom):
                # For from imports, group multiple imports into one construct per statement
                module = node.module or ""
                imported_names = []

                for alias in node.names:
                    if alias.asname:
                        imported_names.append(f"{alias.name} as {alias.asname}")
                    else:
                        imported_names.append(alias.name)

                # Create a single construct for the entire from statement
                name_for_import = f"{module}.{imported_names[0]}" if module else imported_names[0]
                signature = f"from {module} import {', '.join(imported_names)}"

                constructs.append(CodeConstruct(
                    construct_type="import",
                    name=name_for_import,
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    signature=signature
                ))

        except Exception as error:
            logger.debug("Error extracting import: %s", error)

        return constructs

    def _extract_variable_assignment(self, node: ast.Assign, lines: List[str]) -> List[CodeConstruct]:
        """Extract module-level variable assignments."""
        constructs = []

        try:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    # Simple variable assignment
                    value_str = ast.unparse(node.value)
                    signature = f"{target.id} = {value_str}"

                    constructs.append(CodeConstruct(
                        construct_type="variable",
                        name=target.id,
                        start_line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        signature=signature
                    ))

        except Exception as error:
            logger.debug("Error extracting variable assignment: %s", error)

        return constructs

    def _is_module_level(self, node: ast.AST, tree: ast.Module) -> bool:
        """Check if a node is at module level (not inside a function or class)."""
        # This is a simplified check - in a full implementation, we'd track the AST context
        return True  # For now, assume all assignments we process are module-level


class JavaScriptConstructExtractor:
    """
    JavaScript/TypeScript construct extractor using pattern matching.

    Extracts functions, classes, and other constructs using regex patterns
    since JavaScript AST parsing is more complex than Python.
    """

    def __init__(self):
        """Initialize JavaScript pattern matchers."""
        self.function_patterns = [
            re.compile(r'function\s+(\w+)\s*\([^)]*\)\s*\{', re.MULTILINE),
            re.compile(r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*\{', re.MULTILINE),
            re.compile(r'const\s+(\w+)\s*=\s*(\w+)\s*=>\s*[^{;]+;?', re.MULTILINE),  # Single expression arrow functions
        ]

        self.class_pattern = re.compile(r'class\s+(\w+)(?:\s+extends\s+\w+)?\s*\{', re.MULTILINE)

        self.method_pattern = re.compile(r'^\s*(\w+)\s*\([^)]*\)\s*\{', re.MULTILINE)

    def extract_constructs(self, content: str, file_path: str) -> List[CodeConstruct]:
        """
        Extract JavaScript constructs using pattern matching.

        Args:
            content: JavaScript source code content
            file_path: Path to the source file

        Returns:
            List of CodeConstruct objects
        """
        constructs = []
        lines = content.split('\n')

        # Extract imports/requires
        constructs.extend(self._extract_imports_js(content, lines))

        # Extract functions
        constructs.extend(self._extract_functions(content, lines))

        # Extract classes and their methods
        class_constructs = self._extract_classes(content, lines)
        constructs.extend(class_constructs)

        # Extract methods from classes
        for class_construct in class_constructs:
            methods = self._extract_class_methods_js(content, class_construct, lines)
            constructs.extend(methods)

        return constructs

    def _extract_functions(self, content: str, lines: List[str]) -> List[CodeConstruct]:
        """Extract function declarations and expressions."""
        constructs = []

        for pattern in self.function_patterns:
            for match in pattern.finditer(content):
                function_name = match.group(1)
                start_line = content[:match.start()].count('\n') + 1

                # Extract signature from matched text
                signature = self._extract_function_signature(match.group(0))

                # Try to find the end of the function
                end_line = self._find_function_end(content, match.start(), lines)

                constructs.append(CodeConstruct(
                    construct_type="function",
                    name=function_name,
                    start_line=start_line,
                    end_line=end_line,
                    signature=signature
                ))

        return constructs

    def _extract_classes(self, content: str, lines: List[str]) -> List[CodeConstruct]:
        """Extract class definitions."""
        constructs = []

        for match in self.class_pattern.finditer(content):
            class_name = match.group(1)
            start_line = content[:match.start()].count('\n') + 1
            signature = match.group(0)[:-1].strip()  # Remove the opening brace and strip whitespace

            # Try to find the end of the class
            end_line = self._find_brace_end(content, match.end() - 1, lines)

            constructs.append(CodeConstruct(
                construct_type="class",
                name=class_name,
                start_line=start_line,
                end_line=end_line,
                signature=signature
            ))

        return constructs

    def _extract_class_methods_js(
            self,
            content: str,
            class_construct: CodeConstruct,
            lines: List[str]) -> List[CodeConstruct]:
        """Extract methods from a JavaScript class."""
        methods = []

        # Find the class content between its start and end lines
        class_start_line = class_construct.start_line - 1  # Convert to 0-based
        class_end_line = class_construct.end_line - 1

        class_content = '\n'.join(lines[class_start_line:class_end_line + 1])

        # Look for method patterns within the class
        method_patterns = [
            re.compile(r'^\s*(\w+)\s*\([^)]*\)\s*\{', re.MULTILINE),  # Regular methods
            re.compile(r'^\s*async\s+(\w+)\s*\([^)]*\)\s*\{', re.MULTILINE),  # Async methods
            re.compile(r'^\s*static\s+(\w+)\s*\([^)]*\)\s*\{', re.MULTILINE),  # Static methods
            re.compile(r'^\s*(constructor)\s*\([^)]*\)\s*\{', re.MULTILINE),  # Constructor
        ]

        for pattern in method_patterns:
            for match in pattern.finditer(class_content):
                method_name = match.group(1)
                # Calculate the actual line number in the full content
                method_line_in_class = class_content[:match.start()].count('\n')
                method_start_line = class_start_line + method_line_in_class + 1

                # Extract method signature
                signature = match.group(0).rstrip('{').strip()

                # Try to find the end of the method
                method_end_line = self._find_method_end_js(class_content, match.start(), method_start_line)

                methods.append(CodeConstruct(
                    construct_type="method",
                    name=method_name,
                    start_line=method_start_line,
                    end_line=method_end_line,
                    signature=signature,
                    parent_construct_id=class_construct.compute_construct_id("")
                ))

        return methods

    def _find_method_end_js(self, class_content: str, method_start_pos: int, method_start_line: int) -> int:
        """Find the end line of a JavaScript method."""
        # Simple heuristic: look for the matching brace
        brace_count = 0
        lines = class_content[method_start_pos:].split('\n')

        for i, line in enumerate(lines):
            brace_count += line.count('{') - line.count('}')
            if brace_count == 0 and i > 0:  # Found matching closing brace
                return method_start_line + i

        # Fallback: assume method is about 5-10 lines
        return method_start_line + min(DEFAULT_FUNCTION_LENGTH_ESTIMATE, len(lines))

    def _extract_function_signature(self, matched_text: str) -> str:
        """Extract clean function signature from matched text."""
        # Remove the opening brace and clean up whitespace
        return matched_text.rstrip('{').strip()

    def _find_function_end(self, content: str, start_pos: int, lines: List[str]) -> int:
        """Find the end line of a function using brace matching."""
        return self._find_brace_end(content, start_pos, lines)

    def _find_brace_end(self, content: str, start_pos: int, lines: List[str]) -> int:
        """Find the matching closing brace."""
        brace_count = 0
        in_string = False
        string_char = None
        i = start_pos

        while i < len(content):
            char = content[i]

            # Handle string literals
            if char in ['"', "'", '`'] and (i == 0 or content[i - 1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None

            # Count braces outside of strings
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return content[:i].count('\n') + 1

            i += 1

        # If we didn't find the end, return a reasonable estimate
        return content[:start_pos].count('\n') + FALLBACK_LINE_ESTIMATE

    def _extract_imports_js(self, content: str, lines: List[str]) -> List[CodeConstruct]:
        """Extract JavaScript import/require statements."""
        constructs = []
        
        # Patterns for different types of imports
        patterns = [
            # const express = require('express')
            re.compile(r'const\s+(\w+)\s*=\s*require\(["\']([^"\']+)["\']\)', re.MULTILINE),
            # const { User } = require('./models/User')
            re.compile(r'const\s*\{\s*([^}]+)\s*\}\s*=\s*require\(["\']([^"\']+)["\']\)', re.MULTILINE),
            # import React from 'react'
            re.compile(r'import\s+(\w+)\s+from\s+["\']([^"\']+)["\']', re.MULTILINE),
            # import { useState } from 'react'
            re.compile(r'import\s*\{\s*([^}]+)\s*\}\s+from\s+["\']([^"\']+)["\']', re.MULTILINE),
            # import * as React from 'react'
            re.compile(r'import\s*\*\s+as\s+(\w+)\s+from\s+["\']([^"\']+)["\']', re.MULTILINE),
        ]
        
        for pattern in patterns:
            for match in pattern.finditer(content):
                start_line = content[:match.start()].count('\n') + 1
                imported_name = match.group(1).strip()
                module_path = match.group(2)
                
                # Create signature from the matched text
                signature = match.group(0)
                
                constructs.append(CodeConstruct(
                    construct_type="import",
                    name=imported_name,
                    start_line=start_line,
                    end_line=start_line,
                    signature=signature
                ))
        
        return constructs


class CodeConstructExtractor:
    """
    Main orchestrator for intelligent code construct extraction.

    Detects file language and delegates to appropriate specialized extractor
    for optimal context-aware construct extraction.
    """

    def __init__(self, language_detector: Optional[LanguageDetector] = None):
        """Initialize the construct extractor with language detection."""
        self.language_detector = language_detector or LanguageDetector()
        self.python_extractor = PythonConstructExtractor()
        self.js_extractor = JavaScriptConstructExtractor()

    def extract_constructs(self, content: str, file_path: str) -> List[CodeConstruct]:
        """
        Extract constructs from content based on file language.

        Args:
            content: Source code content
            file_path: Path to the source file for language detection

        Returns:
            List of CodeConstruct objects
        """
        try:
            # Detect language
            detection_result = self.language_detector.detect_language(file_path, content)

            # Choose appropriate extractor
            if detection_result.language == "Python":
                return self.python_extractor.extract_constructs(content, file_path)
            elif detection_result.language in ["JavaScript", "TypeScript"]:
                return self.js_extractor.extract_constructs(content, file_path)
            else:
                # For unsupported languages, return empty list
                logger.debug("No construct extractor available for language: %s", detection_result.language)
                return []

        except Exception as error:
            logger.error("Error extracting constructs from %s: %s", file_path, error)
            return []

    def extract_from_content(self, content: str, language: str) -> List[CodeConstruct]:
        """
        Extract constructs from content given an explicit language.
        
        Args:
            content: Source code content
            language: Programming language (e.g., "python", "javascript", "typescript")
            
        Returns:
            List of CodeConstruct objects
        """
        try:
            language_lower = language.lower()
            
            # Choose appropriate extractor based on language
            if language_lower == "python":
                return self.python_extractor.extract_constructs(content, f"<content>.py")
            elif language_lower in ["javascript", "typescript", "js", "ts"]:
                return self.js_extractor.extract_constructs(content, f"<content>.{language_lower}")
            else:
                # For unsupported languages, return empty list
                logger.debug("No construct extractor available for language: %s", language)
                return []
                
        except Exception as error:
            logger.error("Error extracting constructs from content (language: %s): %s", language, error)
            return []


# Standalone utility function
def extract_constructs_from_content(content: str, language: str) -> List[CodeConstruct]:
    """
    Extract programming constructs from code content with specified language.
    
    Args:
        content: Source code content
        language: Programming language (e.g., "python", "javascript", "typescript")
        
    Returns:
        List of CodeConstruct objects
    """
    extractor = CodeConstructExtractor()
    return extractor.extract_from_content(content, language)
