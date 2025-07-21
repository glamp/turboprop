#!/usr/bin/env python3
"""
Test suite for intelligent snippet extraction functionality.

Tests for the snippet_extractor module including:
- Language-aware parsing (Python, JavaScript, etc.)
- Intelligent boundary detection
- Multi-snippet support
- Line number integration
- Fallback extraction for unsupported languages
"""

from unittest.mock import Mock

import pytest
from turboprop.language_detection import LanguageDetectionResult, LanguageDetector
from turboprop.search_result_types import CodeSnippet
from turboprop.snippet_extractor import (
    ExtractedSnippet,
    GenericSnippetExtractor,
    JavaScriptSnippetExtractor,
    PythonSnippetExtractor,
    SnippetExtractor,
)


class TestExtractedSnippet:
    """Test the ExtractedSnippet data class."""

    def test_extracted_snippet_creation(self):
        """Test basic creation of ExtractedSnippet."""
        snippet = ExtractedSnippet(
            text="def hello():\n    return 'world'",
            start_line=10,
            end_line=11,
            relevance_score=0.9,
            snippet_type="function",
        )

        assert snippet.text == "def hello():\n    return 'world'"
        assert snippet.start_line == 10
        assert snippet.end_line == 11
        assert snippet.relevance_score == 0.9
        assert snippet.snippet_type == "function"

    def test_extracted_snippet_to_code_snippet(self):
        """Test conversion to CodeSnippet."""
        extracted = ExtractedSnippet(
            text="class Example:\n    pass", start_line=5, end_line=6, relevance_score=0.8, snippet_type="class"
        )

        code_snippet = extracted.to_code_snippet()

        assert isinstance(code_snippet, CodeSnippet)
        assert code_snippet.text == "class Example:\n    pass"
        assert code_snippet.start_line == 5
        assert code_snippet.end_line == 6


class TestPythonSnippetExtractor:
    """Test Python-specific snippet extraction."""

    @pytest.fixture
    def python_extractor(self):
        """Create a Python snippet extractor."""
        return PythonSnippetExtractor()

    def test_extract_complete_function(self, python_extractor):
        """Test extraction of complete function with docstring."""
        content = '''import os
import sys

def calculate_sum(a, b):
    """Calculate the sum of two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum of a and b
    """
    result = a + b
    return result

def another_function():
    pass
'''

        snippets = python_extractor.extract_snippets(content, "calculate sum")

        assert len(snippets) == 1
        snippet = snippets[0]
        assert snippet.snippet_type == "function"
        assert "def calculate_sum(a, b):" in snippet.text
        assert '"""Calculate the sum of two numbers.' in snippet.text
        assert "return result" in snippet.text
        assert snippet.start_line == 4
        assert snippet.end_line == 15  # Function actually ends at line 15
        assert snippet.relevance_score > 0.5

    def test_extract_class_with_methods(self, python_extractor):
        """Test extraction of class with relevant methods."""
        content = '''class Calculator:
    """A simple calculator class."""

    def __init__(self):
        self.history = []

    def add(self, a, b):
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def subtract(self, a, b):
        """Subtract two numbers."""
        return a - b

class OtherClass:
    pass
'''

        snippets = python_extractor.extract_snippets(content, "add numbers")

        assert len(snippets) >= 1
        # Should extract the entire Calculator class or at least the add method
        found_add_method = any("def add(self, a, b):" in s.text for s in snippets)
        assert found_add_method

    def test_extract_with_imports(self, python_extractor):
        """Test that relevant imports are included."""
        content = '''import json
import requests
from typing import List, Dict

def process_data(data: List[Dict]) -> str:
    """Process data using json."""
    return json.dumps(data)

def unrelated_function():
    print("hello")
'''

        snippets = python_extractor.extract_snippets(content, "process data json")

        assert len(snippets) >= 1
        # Should include the import for json
        snippet_text = snippets[0].text
        assert "import json" in snippet_text or "from typing import" in snippet_text
        assert "def process_data" in snippet_text

    def test_fallback_on_syntax_error(self, python_extractor):
        """Test fallback extraction when Python code has syntax errors."""
        content = """def broken_function(
    # Missing closing parenthesis
    return "this is broken"

def working_function():
    return "this works"
"""

        snippets = python_extractor.extract_snippets(content, "working function")

        # Should still extract something, even with syntax errors
        assert len(snippets) >= 1


class TestJavaScriptSnippetExtractor:
    """Test JavaScript-specific snippet extraction."""

    @pytest.fixture
    def js_extractor(self):
        """Create a JavaScript snippet extractor."""
        return JavaScriptSnippetExtractor()

    def test_extract_function_declaration(self, js_extractor):
        """Test extraction of function declaration."""
        content = """const express = require('express');
const app = express();

function calculateTotal(items) {
    let total = 0;
    for (const item of items) {
        total += item.price;
    }
    return total;
}

function anotherFunction() {
    console.log('hello');
}

module.exports = { calculateTotal };
"""

        snippets = js_extractor.extract_snippets(content, "calculate total")

        assert len(snippets) >= 1
        snippet = snippets[0]
        assert "function calculateTotal(items)" in snippet.text
        assert "return total;" in snippet.text
        assert snippet.snippet_type == "function"

    def test_extract_arrow_function(self, js_extractor):
        """Test extraction of arrow function."""
        content = """import { useState } from 'react';

const MyComponent = () => {
    const [count, setCount] = useState(0);

    const incrementCounter = () => {
        setCount(count + 1);
    };

    return <div>{count}</div>;
};

export default MyComponent;
"""

        snippets = js_extractor.extract_snippets(content, "increment counter")

        assert len(snippets) >= 1
        # Should find the incrementCounter arrow function or the entire component
        found_increment = any("incrementCounter" in s.text for s in snippets)
        assert found_increment

    def test_extract_class_with_methods(self, js_extractor):
        """Test extraction of class with methods."""
        content = """class DataProcessor {
    constructor(options) {
        this.options = options;
    }

    processItems(items) {
        return items.map(item => this.processItem(item));
    }

    processItem(item) {
        return { ...item, processed: true };
    }
}

module.exports = DataProcessor;
"""

        snippets = js_extractor.extract_snippets(content, "process items")

        assert len(snippets) >= 1
        # Should extract the class or at least the processItems method
        found_process_items = any("processItems" in s.text for s in snippets)
        assert found_process_items


class TestGenericSnippetExtractor:
    """Test generic fallback snippet extraction."""

    @pytest.fixture
    def generic_extractor(self):
        """Create a generic snippet extractor."""
        return GenericSnippetExtractor()

    def test_extract_by_lines(self, generic_extractor):
        """Test line-based extraction for unsupported languages."""
        content = """package main

import "fmt"

func calculateSum(a int, b int) int {
    result := a + b
    return result
}

func main() {
    sum := calculateSum(5, 3)
    fmt.Println(sum)
}
"""

        snippets = generic_extractor.extract_snippets(content, "calculate sum")

        assert len(snippets) >= 1
        snippet = snippets[0]
        assert "calculateSum" in snippet.text
        assert snippet.start_line > 0
        assert snippet.end_line >= snippet.start_line

    def test_intelligent_boundary_detection(self, generic_extractor):
        """Test intelligent boundary detection using braces."""
        content = """public class Calculator {
    private int value;

    public int add(int a, int b) {
        return a + b;
    }

    public int multiply(int a, int b) {
        return a * b;
    }
}
"""

        snippets = generic_extractor.extract_snippets(content, "add method")

        assert len(snippets) >= 1
        # Should extract the complete add method
        snippet = snippets[0]
        assert "public int add(int a, int b)" in snippet.text
        assert snippet.text.count("{") == snippet.text.count("}")  # Balanced braces


class TestSnippetExtractor:
    """Test the main SnippetExtractor orchestrator."""

    @pytest.fixture
    def language_detector(self):
        """Mock language detector."""
        detector = Mock(spec=LanguageDetector)
        return detector

    @pytest.fixture
    def snippet_extractor(self, language_detector):
        """Create a SnippetExtractor with mocked language detector."""
        return SnippetExtractor(language_detector=language_detector)

    def test_extract_python_snippets(self, snippet_extractor, language_detector):
        """Test extraction for Python files."""
        language_detector.detect_language.return_value = LanguageDetectionResult(
            language="Python", file_type=".py", confidence=1.0, category="source"
        )

        content = '''def hello_world():
    """Print hello world."""
    print("Hello, World!")
    return "success"
'''

        snippets = snippet_extractor.extract_snippets(content=content, file_path="test.py", query="hello world")

        assert len(snippets) >= 1
        assert snippets[0].snippet_type in ["function", "generic"]
        assert "hello_world" in snippets[0].text

    def test_extract_javascript_snippets(self, snippet_extractor, language_detector):
        """Test extraction for JavaScript files."""
        language_detector.detect_language.return_value = LanguageDetectionResult(
            language="JavaScript", file_type=".js", confidence=1.0, category="source"
        )

        content = """function greetUser(name) {
    return `Hello, ${name}!`;
}

const anotherFunction = () => {
    console.log("test");
};
"""

        snippets = snippet_extractor.extract_snippets(content=content, file_path="test.js", query="greet user")

        assert len(snippets) >= 1
        assert "greetUser" in snippets[0].text

    def test_extract_unknown_language_fallback(self, snippet_extractor, language_detector):
        """Test fallback extraction for unknown languages."""
        language_detector.detect_language.return_value = LanguageDetectionResult(
            language="Unknown", file_type=".xyz", confidence=0.0, category="unknown"
        )

        content = """some random content
with multiple lines
that contains search terms
and more content
"""

        snippets = snippet_extractor.extract_snippets(content=content, file_path="test.xyz", query="search terms")

        assert len(snippets) >= 1
        assert "search terms" in snippets[0].text

    def test_multi_snippet_extraction(self, snippet_extractor, language_detector):
        """Test extraction of multiple relevant snippets."""
        language_detector.detect_language.return_value = LanguageDetectionResult(
            language="Python", file_type=".py", confidence=1.0, category="source"
        )

        content = '''def search_function():
    """This function performs searching."""
    return "search result"

class SearchEngine:
    """A search engine class."""

    def __init__(self):
        self.index = {}

    def search_items(self, query):
        """Search for items."""
        return self.index.get(query, [])

def unrelated_function():
    return "nothing"
'''

        snippets = snippet_extractor.extract_snippets(content=content, file_path="test.py", query="search")

        # Should extract multiple relevant snippets
        assert len(snippets) >= 2

        # Should be ranked by relevance
        assert snippets[0].relevance_score >= snippets[1].relevance_score

        # Should contain search-related content
        search_content_found = any("search" in s.text.lower() for s in snippets)
        assert search_content_found

    def test_snippet_size_limits(self, snippet_extractor, language_detector):
        """Test that snippet extraction respects size limits."""
        language_detector.detect_language.return_value = LanguageDetectionResult(
            language="Python", file_type=".py", confidence=1.0, category="source"
        )

        # Create a very long function
        long_content = (
            '''def very_long_function():
    """A very long function for testing."""
'''
            + '    print("line")\n' * 100
        )

        snippets = snippet_extractor.extract_snippets(
            content=long_content,
            file_path="test.py",
            query="very long function",
            max_snippet_length=500,  # Limit snippet size
        )

        assert len(snippets) >= 1
        # Should respect the size limit
        assert len(snippets[0].text) <= 500 + 50  # Allow some buffer for truncation markers


class TestIntegrationScenarios:
    """Test integration scenarios with real-world code patterns."""

    def test_extract_from_real_python_file(self):
        """Test extraction from a realistic Python file."""
        # This simulates the kind of code found in the actual codebase
        content = '''#!/usr/bin/env python3
"""
search_operations.py: Search functionality for code search.
"""

import logging
from typing import List, Optional
from turboprop.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

def search_index_enhanced(db_manager: DatabaseManager, query: str, k: int) -> List:
    """Enhanced search with structured results.

    Args:
        db_manager: Database manager instance
        query: Search query
        k: Number of results

    Returns:
        List of search results
    """
    try:
        # Search logic here
        results = []
        return results
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []

class SearchManager:
    """Manages search operations."""

    def __init__(self, db_manager):
        self.db_manager = db_manager

    def perform_search(self, query):
        """Perform a search operation."""
        return search_index_enhanced(self.db_manager, query, 5)
'''

        extractor = SnippetExtractor()
        snippets = extractor.extract_snippets(
            content=content, file_path="search_operations.py", query="enhanced search"
        )

        assert len(snippets) >= 1

        # Should extract the search_index_enhanced function
        enhanced_search_found = any("search_index_enhanced" in s.text and "def " in s.text for s in snippets)
        assert enhanced_search_found

        # Should include proper line numbers
        for snippet in snippets:
            assert snippet.start_line > 0
            assert snippet.end_line >= snippet.start_line
