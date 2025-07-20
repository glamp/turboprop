#!/usr/bin/env python3
"""
Tests for code construct extraction functionality.

This test suite verifies the extraction of programming constructs (functions, classes,
variables) from source code files using AST parsing and pattern matching.
"""

import tempfile
from pathlib import Path

from code_construct_extractor import (
    CodeConstruct,
    CodeConstructExtractor,
    JavaScriptConstructExtractor,
    PythonConstructExtractor,
)
from database_manager import DatabaseManager
from language_detection import LanguageDetector


class TestCodeConstruct:
    """Test the CodeConstruct data class."""

    def test_create_code_construct(self):
        """Test creating a CodeConstruct instance."""
        construct = CodeConstruct(
            construct_type="function",
            name="test_func",
            start_line=10,
            end_line=15,
            signature="def test_func(arg1: str, arg2: int) -> bool:",
            docstring="Test function documentation",
        )

        assert construct.construct_type == "function"
        assert construct.name == "test_func"
        assert construct.start_line == 10
        assert construct.end_line == 15
        assert construct.signature == "def test_func(arg1: str, arg2: int) -> bool:"
        assert construct.docstring == "Test function documentation"
        assert construct.parent_construct_id is None

    def test_compute_construct_id(self):
        """Test compute_construct_id generates consistent hashes."""
        construct1 = CodeConstruct(
            construct_type="function",
            name="test_func",
            start_line=10,
            end_line=15,
            signature="def test_func():",
        )

        construct2 = CodeConstruct(
            construct_type="function",
            name="test_func",
            start_line=10,
            end_line=15,
            signature="def test_func():",
        )

        # Same constructs should have same IDs
        assert construct1.compute_construct_id("file123") == construct2.compute_construct_id("file123")

        # Different signatures should have different IDs
        construct3 = CodeConstruct(
            construct_type="function",
            name="test_func",
            start_line=10,
            end_line=15,
            signature="def test_func(arg: str):",
        )

        assert construct1.compute_construct_id("file123") != construct3.compute_construct_id("file123")


class TestPythonConstructExtractor:
    """Test Python-specific construct extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = PythonConstructExtractor()

    def test_extract_simple_function(self):
        """Test extracting a simple function definition."""
        code = '''
def hello_world():
    """Simple greeting function."""
    return "Hello, World!"
'''
        constructs = self.extractor.extract_constructs(code, "test.py")

        assert len(constructs) == 1
        construct = constructs[0]
        assert construct.construct_type == "function"
        assert construct.name == "hello_world"
        assert construct.signature == "def hello_world():"
        assert construct.docstring == "Simple greeting function."
        assert construct.start_line == 2
        assert construct.end_line == 4

    def test_extract_function_with_parameters(self):
        """Test extracting function with parameters and type hints."""
        code = '''
def calculate_sum(a: int, b: int = 0) -> int:
    """Calculate the sum of two integers.

    Args:
        a: First integer
        b: Second integer (default: 0)

    Returns:
        Sum of a and b
    """
    return a + b
'''
        constructs = self.extractor.extract_constructs(code, "test.py")

        assert len(constructs) == 1
        construct = constructs[0]
        assert construct.construct_type == "function"
        assert construct.name == "calculate_sum"
        assert construct.signature == "def calculate_sum(a: int, b: int = 0) -> int:"
        assert "Calculate the sum of two integers." in construct.docstring
        assert construct.start_line == 2

    def test_extract_class_definition(self):
        """Test extracting a class definition."""
        code = '''
class Calculator:
    """A simple calculator class."""

    def __init__(self, initial_value: int = 0):
        """Initialize the calculator."""
        self.value = initial_value

    def add(self, number: int) -> int:
        """Add a number to the current value."""
        self.value += number
        return self.value
'''
        constructs = self.extractor.extract_constructs(code, "test.py")

        # Should extract class and its methods
        assert len(constructs) == 3

        # Check class construct
        class_construct = next(c for c in constructs if c.construct_type == "class")
        assert class_construct.name == "Calculator"
        assert class_construct.signature == "class Calculator:"
        assert class_construct.docstring == "A simple calculator class."
        assert class_construct.start_line == 2

        # Check method constructs
        method_constructs = [c for c in constructs if c.construct_type == "method"]
        assert len(method_constructs) == 2

        init_method = next(c for c in method_constructs if c.name == "__init__")
        assert init_method.signature == "def __init__(self, initial_value: int = 0):"
        assert init_method.parent_construct_id is not None

    def test_extract_class_with_inheritance(self):
        """Test extracting class with inheritance."""
        code = '''
class ScientificCalculator(Calculator):
    """Advanced calculator with scientific functions."""

    def power(self, exponent: int) -> int:
        """Raise value to the power of exponent."""
        self.value = self.value ** exponent
        return self.value
'''
        constructs = self.extractor.extract_constructs(code, "test.py")

        class_construct = next(c for c in constructs if c.construct_type == "class")
        assert class_construct.name == "ScientificCalculator"
        assert class_construct.signature == "class ScientificCalculator(Calculator):"

    def test_extract_global_variables(self):
        """Test extracting global variable assignments."""
        code = """
# Configuration constants
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30.0
API_BASE_URL = "https://api.example.com"

def some_function():
    pass
"""
        constructs = self.extractor.extract_constructs(code, "test.py")

        variable_constructs = [c for c in constructs if c.construct_type == "variable"]
        assert len(variable_constructs) == 3

        max_retries = next(c for c in variable_constructs if c.name == "MAX_RETRIES")
        assert max_retries.signature == "MAX_RETRIES = 3"

    def test_extract_imports(self):
        """Test extracting import statements."""
        code = """
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

def some_function():
    pass
"""
        constructs = self.extractor.extract_constructs(code, "test.py")

        import_constructs = [c for c in constructs if c.construct_type == "import"]
        assert len(import_constructs) == 4

        # Check specific imports
        import_names = {c.name for c in import_constructs}
        assert "os" in import_names
        assert "pathlib.Path" in import_names
        assert "typing.List" in import_names

    def test_extract_async_function(self):
        """Test extracting async function definitions."""
        code = '''
async def fetch_data(url: str) -> dict:
    """Fetch data from URL asynchronously."""
    # Implementation here
    return {}
'''
        constructs = self.extractor.extract_constructs(code, "test.py")

        assert len(constructs) == 1
        construct = constructs[0]
        assert construct.construct_type == "function"
        assert construct.name == "fetch_data"
        assert construct.signature == "async def fetch_data(url: str) -> dict:"

    def test_extract_empty_file(self):
        """Test extraction from empty file."""
        code = ""
        constructs = self.extractor.extract_constructs(code, "test.py")
        assert len(constructs) == 0

    def test_extract_syntax_error_handling(self):
        """Test handling of files with syntax errors."""
        code = """
def broken_function(
    # Missing closing parenthesis and colon
    pass
"""
        # Should not raise exception, just return empty list
        constructs = self.extractor.extract_constructs(code, "test.py")
        assert len(constructs) == 0


class TestJavaScriptConstructExtractor:
    """Test JavaScript/TypeScript construct extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = JavaScriptConstructExtractor()

    def test_extract_function_declaration(self):
        """Test extracting function declarations."""
        code = """
function calculateSum(a, b) {
    // Calculate the sum of two numbers
    return a + b;
}
"""
        constructs = self.extractor.extract_constructs(code, "test.js")

        assert len(constructs) == 1
        construct = constructs[0]
        assert construct.construct_type == "function"
        assert construct.name == "calculateSum"
        assert construct.signature == "function calculateSum(a, b)"

    def test_extract_arrow_function(self):
        """Test extracting arrow function expressions."""
        code = """
const multiply = (x, y) => {
    return x * y;
};

const square = x => x * x;
"""
        constructs = self.extractor.extract_constructs(code, "test.js")

        assert len(constructs) == 2

        multiply_func = next(c for c in constructs if c.name == "multiply")
        assert multiply_func.construct_type == "function"
        assert multiply_func.signature == "const multiply = (x, y) =>"

    def test_extract_class_definition(self):
        """Test extracting ES6 class definitions."""
        code = """
class Calculator {
    constructor(initialValue = 0) {
        this.value = initialValue;
    }

    add(number) {
        this.value += number;
        return this.value;
    }

    static isCalculator(obj) {
        return obj instanceof Calculator;
    }
}
"""
        constructs = self.extractor.extract_constructs(code, "test.js")

        # Should extract class and methods
        class_construct = next(c for c in constructs if c.construct_type == "class")
        assert class_construct.name == "Calculator"
        assert class_construct.signature == "class Calculator"

        method_constructs = [c for c in constructs if c.construct_type == "method"]
        assert len(method_constructs) >= 2  # constructor and add method


class TestCodeConstructExtractor:
    """Test the main CodeConstructExtractor orchestrator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.language_detector = LanguageDetector()
        self.extractor = CodeConstructExtractor(self.language_detector)

    def test_extract_python_file(self):
        """Test extraction from Python file."""
        code = '''
def hello():
    """Say hello."""
    print("Hello!")

class Greeter:
    """A greeting class."""

    def greet(self, name):
        return f"Hello, {name}!"
'''
        constructs = self.extractor.extract_constructs(code, "test.py")

        assert len(constructs) > 0
        # Should contain both function and class constructs
        construct_types = {c.construct_type for c in constructs}
        assert "function" in construct_types
        assert "class" in construct_types

    def test_extract_javascript_file(self):
        """Test extraction from JavaScript file."""
        code = """
function greet(name) {
    return `Hello, ${name}!`;
}

class Person {
    constructor(name) {
        this.name = name;
    }
}
"""
        constructs = self.extractor.extract_constructs(code, "test.js")

        assert len(constructs) > 0
        construct_types = {c.construct_type for c in constructs}
        assert "function" in construct_types

    def test_extract_unsupported_language(self):
        """Test extraction from unsupported language falls back gracefully."""
        code = """
#include <iostream>
int main() {
    std::cout << "Hello World" << std::endl;
    return 0;
}
"""
        # Should not crash, but may return empty list
        constructs = self.extractor.extract_constructs(code, "test.cpp")
        assert isinstance(constructs, list)


class TestConstructExtractionIntegration:
    """Integration tests for construct extraction with database."""

    def test_database_construct_storage(self):
        """Test storing constructs in the database."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.duckdb"
            db_manager = DatabaseManager(db_path)

            # Create the code_constructs table
            db_manager.create_constructs_table()

            # Create a sample construct
            construct = CodeConstruct(
                construct_type="function",
                name="test_func",
                start_line=1,
                end_line=5,
                signature="def test_func():",
                docstring="Test function",
            )

            file_id = "test_file_123"
            construct_id = construct.compute_construct_id(file_id)

            # Store the construct
            db_manager.store_construct(construct, file_id, construct_id, [0.1] * 384)

            # Verify it was stored
            with db_manager.get_connection() as conn:
                result = conn.execute("SELECT * FROM code_constructs WHERE id = ?", (construct_id,)).fetchone()

                assert result is not None
                assert result[2] == "function"  # construct_type
                assert result[3] == "test_func"  # name

            db_manager.cleanup()
