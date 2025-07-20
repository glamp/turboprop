#!/usr/bin/env python3
"""
snippet_extractor.py: Intelligent code snippet extraction with language-aware parsing.

This module provides context-aware code snippet extraction that shows complete logical units 
(functions, classes) instead of arbitrary character truncations, enabling better AI understanding.

Classes:
- ExtractedSnippet: Data class for extracted code snippets with metadata
- PythonSnippetExtractor: Python AST-based extraction
- JavaScriptSnippetExtractor: JavaScript pattern-based extraction  
- GenericSnippetExtractor: Fallback extraction for unsupported languages
- SnippetExtractor: Main orchestrator for language-aware extraction
"""

import ast
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

from search_result_types import CodeSnippet
from language_detection import LanguageDetector, LanguageDetectionResult
from config import config

logger = logging.getLogger(__name__)


@dataclass
class ExtractedSnippet:
    """
    Represents an extracted code snippet with metadata.
    
    This class extends the basic snippet concept with relevance scoring
    and snippet type classification for better ranking and presentation.
    """
    text: str
    start_line: int
    end_line: int
    relevance_score: float
    snippet_type: str = "generic"  # function, class, method, generic
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    
    def to_code_snippet(self) -> CodeSnippet:
        """Convert to CodeSnippet for compatibility with existing data structures."""
        return CodeSnippet(
            text=self.text,
            start_line=self.start_line,
            end_line=self.end_line,
            context_before=self.context_before,
            context_after=self.context_after
        )


class PythonSnippetExtractor:
    """
    Python-specific snippet extractor using AST parsing.
    
    Extracts complete functions, classes, and methods with their docstrings
    and relevant context like imports.
    """
    
    def __init__(self):
        """Initialize the Python snippet extractor."""
        self.import_patterns = [
            re.compile(r'^(import\s+\w+(?:\.\w+)*)', re.MULTILINE),
            re.compile(r'^(from\s+\w+(?:\.\w+)*\s+import\s+[^#\n]+)', re.MULTILINE)
        ]
    
    def extract_snippets(
        self, 
        content: str, 
        query: str, 
        max_snippets: int = 3,
        max_snippet_length: int = 2000
    ) -> List[ExtractedSnippet]:
        """
        Extract Python snippets using AST parsing.
        
        Args:
            content: Python source code
            query: Search query for relevance ranking
            max_snippets: Maximum number of snippets to return
            max_snippet_length: Maximum length per snippet
            
        Returns:
            List of ExtractedSnippet objects ranked by relevance
        """
        try:
            # Store max length for use in helper methods
            self._current_max_length = max_snippet_length
            
            tree = ast.parse(content)
            lines = content.split('\n')
            
            candidates = []
            
            # Extract all function and class definitions
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    snippet = self._extract_function(node, lines, content, query)
                    if snippet:
                        candidates.append(snippet)
                elif isinstance(node, ast.ClassDef):
                    snippet = self._extract_class(node, lines, content, query)
                    if snippet:
                        candidates.append(snippet)
            
            # If no AST-based matches, try module-level extraction
            if not candidates:
                module_snippet = self._extract_module_level(content, query, max_snippet_length)
                if module_snippet:
                    candidates.append(module_snippet)
            
            # Sort by relevance and return top matches
            candidates.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Filter out low-relevance matches if we have high-relevance ones
            if candidates:
                top_score = candidates[0].relevance_score
                # If we have a very high scoring match, only return highly relevant ones
                if top_score > 0.8:
                    candidates = [c for c in candidates if c.relevance_score > 0.7]
                
            return candidates[:max_snippets]
            
        except SyntaxError as e:
            logger.warning(f"Python syntax error in snippet extraction: {e}")
            # Fall back to generic extraction
            return self._fallback_extraction(content, query, max_snippet_length)
        except Exception as e:
            logger.error(f"Error in Python snippet extraction: {e}")
            return self._fallback_extraction(content, query, max_snippet_length)
    
    def _extract_function(
        self, 
        node: ast.FunctionDef, 
        lines: List[str], 
        content: str, 
        query: str
    ) -> Optional[ExtractedSnippet]:
        """Extract a complete function definition."""
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        
        # Get function text including docstring
        function_lines = lines[start_line - 1:end_line]
        function_text = '\n'.join(function_lines)
        
        # Add relevant imports if they exist
        imports = self._find_relevant_imports(content, function_text)
        if imports:
            function_text = '\n'.join(imports) + '\n\n' + function_text
        
        # Calculate relevance based on query match
        relevance_score = self._calculate_relevance(
            function_text, query, node.name, "function"
        )
        
        # Use the method's max_snippet_length parameter instead of config
        current_max_length = getattr(self, '_current_max_length', config.search.SNIPPET_CONTENT_MAX_LENGTH)
        if len(function_text) > current_max_length:
            function_text = function_text[:current_max_length] + "..."
        
        return ExtractedSnippet(
            text=function_text,
            start_line=start_line,
            end_line=end_line,
            relevance_score=relevance_score,
            snippet_type="function"
        )
    
    def _extract_class(
        self, 
        node: ast.ClassDef, 
        lines: List[str], 
        content: str, 
        query: str
    ) -> Optional[ExtractedSnippet]:
        """Extract a complete class definition or relevant methods."""
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        
        class_lines = lines[start_line - 1:end_line]
        class_text = '\n'.join(class_lines)
        
        # Add relevant imports
        imports = self._find_relevant_imports(content, class_text)
        if imports:
            class_text = '\n'.join(imports) + '\n\n' + class_text
        
        # Calculate relevance
        relevance_score = self._calculate_relevance(
            class_text, query, node.name, "class"
        )
        
        # Use the method's max_snippet_length parameter instead of config
        current_max_length = getattr(self, '_current_max_length', config.search.SNIPPET_CONTENT_MAX_LENGTH)
        
        # If class is too long, try to extract relevant methods only
        if len(class_text) > current_max_length:
            relevant_methods = self._extract_relevant_methods(node, lines, query)
            if relevant_methods:
                return relevant_methods
        
        if len(class_text) > current_max_length:
            class_text = class_text[:current_max_length] + "..."
        
        return ExtractedSnippet(
            text=class_text,
            start_line=start_line,
            end_line=end_line,
            relevance_score=relevance_score,
            snippet_type="class"
        )
    
    def _extract_relevant_methods(
        self, 
        class_node: ast.ClassDef, 
        lines: List[str], 
        query: str
    ) -> Optional[ExtractedSnippet]:
        """Extract the most relevant methods from a class."""
        best_method = None
        best_score = 0.0
        
        for node in class_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_start = node.lineno
                method_end = node.end_lineno or method_start
                method_lines = lines[method_start - 1:method_end]
                method_text = '\n'.join(method_lines)
                
                score = self._calculate_relevance(
                    method_text, query, node.name, "method"
                )
                
                if score > best_score:
                    best_score = score
                    best_method = ExtractedSnippet(
                        text=method_text,
                        start_line=method_start,
                        end_line=method_end,
                        relevance_score=score,
                        snippet_type="method"
                    )
        
        return best_method
    
    def _extract_module_level(
        self, 
        content: str, 
        query: str, 
        max_length: int
    ) -> Optional[ExtractedSnippet]:
        """Extract module-level code when no specific functions/classes match."""
        lines = content.split('\n')
        
        # Find lines containing query terms
        matching_lines = []
        for i, line in enumerate(lines):
            if any(term.lower() in line.lower() for term in query.split()):
                matching_lines.append(i)
        
        if not matching_lines:
            return None
        
        # Extract a reasonable context around matching lines
        start_line = max(0, min(matching_lines) - 2)
        end_line = min(len(lines), max(matching_lines) + 5)
        
        snippet_text = '\n'.join(lines[start_line:end_line])
        
        if len(snippet_text) > max_length:
            snippet_text = snippet_text[:max_length] + "..."
        
        return ExtractedSnippet(
            text=snippet_text,
            start_line=start_line + 1,
            end_line=end_line,
            relevance_score=0.5,  # Medium relevance for module-level matches
            snippet_type="module"
        )
    
    def _find_relevant_imports(self, content: str, extracted_text: str) -> List[str]:
        """Find imports that are relevant to the extracted code."""
        imports = []
        
        for pattern in self.import_patterns:
            matches = pattern.findall(content)
            for import_stmt in matches:
                # Check if any imported names are used in the extracted text
                if self._is_import_relevant(import_stmt, extracted_text):
                    imports.append(import_stmt)
        
        return imports[:3]  # Limit to 3 most relevant imports
    
    def _is_import_relevant(self, import_stmt: str, code_text: str) -> bool:
        """Check if an import statement is relevant to the code."""
        # Extract module/function names from import statement
        if import_stmt.startswith('from'):
            # from module import name1, name2
            match = re.search(r'import\s+(.+)', import_stmt)
            if match:
                imported_names = [name.strip() for name in match.group(1).split(',')]
                return any(name in code_text for name in imported_names)
        else:
            # import module
            match = re.search(r'import\s+(\w+)', import_stmt)
            if match:
                module_name = match.group(1)
                return module_name in code_text
        
        return False
    
    def _calculate_relevance(
        self, 
        text: str, 
        query: str, 
        name: str, 
        snippet_type: str
    ) -> float:
        """Calculate relevance score for a snippet."""
        score = 0.0
        query_terms = query.lower().split()
        text_lower = text.lower()
        name_lower = name.lower()
        
        # Exact name match gets highest score
        if any(term in name_lower for term in query_terms):
            score += 1.0
        
        # Partial name match
        elif any(term in name_lower for term in query_terms if len(term) > 2):
            score += 0.8
        
        # Content match
        content_matches = sum(1 for term in query_terms if term in text_lower)
        score += (content_matches / len(query_terms)) * 0.6
        
        # Boost for function/class types
        if snippet_type in ["function", "class"]:
            score += 0.2
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _fallback_extraction(
        self, 
        content: str, 
        query: str, 
        max_length: int
    ) -> List[ExtractedSnippet]:
        """Fallback to simple extraction when AST parsing fails."""
        lines = content.split('\n')
        query_terms = query.lower().split()
        
        # Find lines with query matches
        matching_lines = []
        for i, line in enumerate(lines):
            if any(term in line.lower() for term in query_terms):
                matching_lines.append(i)
        
        if not matching_lines:
            # Return first part of content if no matches
            text = content[:max_length]
            if len(content) > max_length:
                text += "..."
            
            return [ExtractedSnippet(
                text=text,
                start_line=1,
                end_line=min(len(lines), text.count('\n') + 1),
                relevance_score=0.1,
                snippet_type="generic"
            )]
        
        # Extract context around matches
        snippets = []
        for match_line in matching_lines[:2]:  # Limit to 2 matches
            start = max(0, match_line - 2)
            end = min(len(lines), match_line + 5)
            
            snippet_text = '\n'.join(lines[start:end])
            if len(snippet_text) > max_length:
                snippet_text = snippet_text[:max_length] + "..."
            
            snippets.append(ExtractedSnippet(
                text=snippet_text,
                start_line=start + 1,
                end_line=end,
                relevance_score=0.4,
                snippet_type="generic"
            ))
        
        return snippets


class JavaScriptSnippetExtractor:
    """
    JavaScript/TypeScript snippet extractor using pattern matching.
    
    Extracts functions, classes, and arrow functions using regex patterns
    since JavaScript AST parsing is more complex than Python.
    """
    
    def __init__(self):
        """Initialize JavaScript pattern matchers."""
        # We'll use a different approach - find function starts and then find matching braces
        self.function_start_patterns = [
            re.compile(r'function\s+(\w+)\s*\([^)]*\)\s*\{'),
            re.compile(r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*\{'),
            re.compile(r'(\w+)\s*\([^)]*\)\s*\{'),  # Method definitions
        ]
        
        self.import_patterns = [
            re.compile(r'^(import\s+[^;]+;?)', re.MULTILINE),
            re.compile(r'^(const\s+\w+\s*=\s*require\([^)]+\);?)', re.MULTILINE),
        ]
    
    def extract_snippets(
        self, 
        content: str, 
        query: str, 
        max_snippets: int = 3,
        max_snippet_length: int = 2000
    ) -> List[ExtractedSnippet]:
        """
        Extract JavaScript snippets using pattern matching with proper brace balancing.
        
        Args:
            content: JavaScript/TypeScript source code
            query: Search query for relevance ranking
            max_snippets: Maximum number of snippets to return
            max_snippet_length: Maximum length per snippet
            
        Returns:
            List of ExtractedSnippet objects ranked by relevance
        """
        lines = content.split('\n')
        candidates = []
        
        # Extract functions using brace balancing
        for pattern in self.function_start_patterns:
            for match in pattern.finditer(content):
                snippet = self._extract_complete_function(
                    match, content, lines, query, max_snippet_length
                )
                if snippet and not self._is_duplicate(snippet, candidates):
                    candidates.append(snippet)
        
        # Extract classes with similar approach
        class_pattern = re.compile(r'class\s+(\w+)(?:\s+extends\s+\w+)?\s*\{')
        for match in class_pattern.finditer(content):
            snippet = self._extract_complete_class(
                match, content, lines, query, max_snippet_length
            )
            if snippet and not self._is_duplicate(snippet, candidates):
                candidates.append(snippet)
        
        # If no matches, try fallback extraction
        if not candidates:
            fallback = self._fallback_extraction(content, query, max_snippet_length)
            candidates.extend(fallback)
        
        # Sort by relevance and return top matches
        candidates.sort(key=lambda x: x.relevance_score, reverse=True)
        return candidates[:max_snippets]
    
    def _extract_complete_function(
        self,
        match: re.Match,
        content: str,
        lines: List[str],
        query: str,
        max_length: int
    ) -> Optional[ExtractedSnippet]:
        """Extract a complete function using brace balancing."""
        function_name = match.group(1)
        start_pos = match.start()
        
        # Find the opening brace position
        brace_pos = content.find('{', match.start())
        if brace_pos == -1:
            return None
        
        # Find the matching closing brace
        end_pos = self._find_matching_brace(content, brace_pos)
        if end_pos == -1:
            return None
        
        # Extract the complete function
        function_text = content[start_pos:end_pos + 1]
        
        # Calculate line numbers
        start_line = content[:start_pos].count('\n') + 1
        end_line = content[:end_pos + 1].count('\n') + 1
        
        # Add relevant imports
        imports = self._find_relevant_imports(content, function_text)
        if imports:
            function_text = '\n'.join(imports) + '\n\n' + function_text
        
        # Calculate relevance
        relevance_score = self._calculate_relevance(function_text, query, function_name, "function")
        
        if len(function_text) > max_length:
            function_text = function_text[:max_length] + "..."
        
        return ExtractedSnippet(
            text=function_text,
            start_line=start_line,
            end_line=end_line,
            relevance_score=relevance_score,
            snippet_type="function"
        )
    
    def _extract_complete_class(
        self,
        match: re.Match,
        content: str,
        lines: List[str],
        query: str,
        max_length: int
    ) -> Optional[ExtractedSnippet]:
        """Extract a complete class using brace balancing."""
        class_name = match.group(1)
        start_pos = match.start()
        
        # Find the opening brace
        brace_pos = content.find('{', match.start())
        if brace_pos == -1:
            return None
        
        # Find the matching closing brace
        end_pos = self._find_matching_brace(content, brace_pos)
        if end_pos == -1:
            return None
        
        # Extract the complete class
        class_text = content[start_pos:end_pos + 1]
        
        # Calculate line numbers
        start_line = content[:start_pos].count('\n') + 1
        end_line = content[:end_pos + 1].count('\n') + 1
        
        # Add relevant imports
        imports = self._find_relevant_imports(content, class_text)
        if imports:
            class_text = '\n'.join(imports) + '\n\n' + class_text
        
        # Calculate relevance
        relevance_score = self._calculate_relevance(class_text, query, class_name, "class")
        
        if len(class_text) > max_length:
            class_text = class_text[:max_length] + "..."
        
        return ExtractedSnippet(
            text=class_text,
            start_line=start_line,
            end_line=end_line,
            relevance_score=relevance_score,
            snippet_type="class"
        )
    
    def _find_matching_brace(self, content: str, start_pos: int) -> int:
        """Find the matching closing brace for an opening brace."""
        brace_count = 0
        in_string = False
        in_single_quote = False
        in_regex = False
        i = start_pos
        
        while i < len(content):
            char = content[i]
            prev_char = content[i - 1] if i > 0 else ''
            
            # Handle string literals
            if char == '"' and prev_char != '\\' and not in_single_quote:
                in_string = not in_string
            elif char == "'" and prev_char != '\\' and not in_string:
                in_single_quote = not in_single_quote
            
            # Skip processing inside strings
            if in_string or in_single_quote:
                i += 1
                continue
                
            # Handle regex literals (basic detection)
            if char == '/' and prev_char != '\\' and not in_regex:
                # Simple regex detection - this could be improved
                if i + 1 < len(content) and content[i + 1] != '*' and content[i + 1] != '/':
                    in_regex = True
            elif char == '/' and in_regex and prev_char != '\\':
                in_regex = False
            
            if in_regex:
                i += 1
                continue
            
            # Count braces
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return i
            
            i += 1
        
        return -1  # No matching brace found
    
    def _create_snippet_from_match(
        self, 
        match: re.Match, 
        lines: List[str], 
        content: str, 
        query: str, 
        snippet_type: str, 
        max_length: int
    ) -> Optional[ExtractedSnippet]:
        """Create a snippet from a regex match."""
        match_text = match.group(1)
        
        # Find line numbers
        start_pos = match.start(1)
        end_pos = match.end(1)
        
        start_line = content[:start_pos].count('\n') + 1
        end_line = content[:end_pos].count('\n') + 1
        
        # Add relevant imports
        imports = self._find_relevant_imports(content, match_text)
        if imports:
            match_text = '\n'.join(imports) + '\n\n' + match_text
        
        # Calculate relevance
        # Extract name from the match (simplified)
        name = self._extract_name_from_match(match_text, snippet_type)
        relevance_score = self._calculate_relevance(match_text, query, name, snippet_type)
        
        if len(match_text) > max_length:
            match_text = match_text[:max_length] + "..."
        
        return ExtractedSnippet(
            text=match_text,
            start_line=start_line,
            end_line=end_line,
            relevance_score=relevance_score,
            snippet_type=snippet_type
        )
    
    def _extract_name_from_match(self, text: str, snippet_type: str) -> str:
        """Extract the name/identifier from matched text."""
        if snippet_type == "function":
            # Try function name
            match = re.search(r'function\s+(\w+)', text)
            if match:
                return match.group(1)
            # Try const name = 
            match = re.search(r'const\s+(\w+)\s*=', text)
            if match:
                return match.group(1)
        elif snippet_type == "class":
            match = re.search(r'class\s+(\w+)', text)
            if match:
                return match.group(1)
        
        return "anonymous"
    
    def _find_relevant_imports(self, content: str, extracted_text: str) -> List[str]:
        """Find imports relevant to extracted JavaScript code."""
        imports = []
        
        for pattern in self.import_patterns:
            matches = pattern.findall(content)
            for import_stmt in matches:
                if self._is_js_import_relevant(import_stmt, extracted_text):
                    imports.append(import_stmt)
        
        return imports[:2]  # Limit to 2 imports
    
    def _is_js_import_relevant(self, import_stmt: str, code_text: str) -> bool:
        """Check if JavaScript import is relevant."""
        # Extract imported names/modules
        if 'from' in import_stmt:
            # import { name1, name2 } from 'module'
            match = re.search(r'import\s*\{([^}]+)\}', import_stmt)
            if match:
                names = [name.strip() for name in match.group(1).split(',')]
                return any(name in code_text for name in names)
        elif 'require' in import_stmt:
            # const module = require('module')
            match = re.search(r'const\s+(\w+)', import_stmt)
            if match:
                return match.group(1) in code_text
        
        return False
    
    def _is_duplicate(self, snippet: ExtractedSnippet, existing: List[ExtractedSnippet]) -> bool:
        """Check if snippet is duplicate or very similar to existing ones."""
        for existing_snippet in existing:
            # Check if line ranges overlap significantly
            if (snippet.start_line <= existing_snippet.end_line and 
                snippet.end_line >= existing_snippet.start_line):
                overlap = min(snippet.end_line, existing_snippet.end_line) - max(snippet.start_line, existing_snippet.start_line)
                total = max(snippet.end_line, existing_snippet.end_line) - min(snippet.start_line, existing_snippet.start_line)
                if total > 0 and overlap / total > 0.5:  # 50% overlap threshold
                    return True
        return False
    
    def _calculate_relevance(self, text: str, query: str, name: str, snippet_type: str) -> float:
        """Calculate relevance score for JavaScript snippet."""
        score = 0.0
        query_terms = query.lower().split()
        text_lower = text.lower()
        name_lower = name.lower()
        
        # Name match scoring
        if any(term in name_lower for term in query_terms):
            score += 1.0
        elif any(term in name_lower for term in query_terms if len(term) > 2):
            score += 0.7
        
        # Content match scoring
        content_matches = sum(1 for term in query_terms if term in text_lower)
        score += (content_matches / len(query_terms)) * 0.5
        
        # Type bonus
        if snippet_type in ["function", "class"]:
            score += 0.2
        
        return min(score, 1.0)
    
    def _fallback_extraction(
        self, 
        content: str, 
        query: str, 
        max_length: int
    ) -> List[ExtractedSnippet]:
        """Fallback extraction for JavaScript."""
        lines = content.split('\n')
        query_terms = query.lower().split()
        
        matching_lines = []
        for i, line in enumerate(lines):
            if any(term in line.lower() for term in query_terms):
                matching_lines.append(i)
        
        if not matching_lines:
            # Return beginning of content
            text = content[:max_length]
            if len(content) > max_length:
                text += "..."
            
            return [ExtractedSnippet(
                text=text,
                start_line=1,
                end_line=min(len(lines), text.count('\n') + 1),
                relevance_score=0.1,
                snippet_type="generic"
            )]
        
        # Extract around first match
        match_line = matching_lines[0]
        start = max(0, match_line - 3)
        end = min(len(lines), match_line + 8)
        
        snippet_text = '\n'.join(lines[start:end])
        if len(snippet_text) > max_length:
            snippet_text = snippet_text[:max_length] + "..."
        
        return [ExtractedSnippet(
            text=snippet_text,
            start_line=start + 1,
            end_line=end,
            relevance_score=0.4,
            snippet_type="generic"
        )]


class GenericSnippetExtractor:
    """
    Generic snippet extractor for unsupported languages.
    
    Uses intelligent line-based extraction with boundary detection
    for languages that don't have specific AST support.
    """
    
    def extract_snippets(
        self, 
        content: str, 
        query: str, 
        max_snippets: int = 2,
        max_snippet_length: int = 1000
    ) -> List[ExtractedSnippet]:
        """
        Extract snippets using intelligent line-based analysis.
        
        Args:
            content: Source code content
            query: Search query for relevance ranking
            max_snippets: Maximum number of snippets
            max_snippet_length: Maximum length per snippet
            
        Returns:
            List of extracted snippets
        """
        lines = content.split('\n')
        query_terms = [term.lower() for term in query.split() if len(term) > 2]
        
        if not query_terms:
            return self._extract_beginning(content, max_snippet_length)
        
        # Find lines that match query terms
        matching_lines = []
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(term in line_lower for term in query_terms):
                matching_lines.append(i)
        
        if not matching_lines:
            return self._extract_beginning(content, max_snippet_length)
        
        snippets = []
        
        # Extract snippets around matches
        for match_line in matching_lines[:max_snippets]:
            snippet = self._extract_around_line(
                lines, match_line, query, max_snippet_length
            )
            if snippet and not self._overlaps_existing(snippet, snippets):
                snippets.append(snippet)
        
        return snippets
    
    def _extract_around_line(
        self, 
        lines: List[str], 
        center_line: int, 
        query: str, 
        max_length: int
    ) -> ExtractedSnippet:
        """Extract content around a specific line using intelligent boundaries."""
        # Try to find logical boundaries (functions, classes, etc.)
        start_line, end_line = self._find_boundaries(lines, center_line)
        
        # Extract the snippet
        snippet_lines = lines[start_line:end_line + 1]
        snippet_text = '\n'.join(snippet_lines)
        
        if len(snippet_text) > max_length:
            snippet_text = snippet_text[:max_length] + "..."
        
        # Calculate basic relevance
        relevance_score = self._calculate_basic_relevance(snippet_text, query)
        
        return ExtractedSnippet(
            text=snippet_text,
            start_line=start_line + 1,  # Convert to 1-based
            end_line=end_line + 1,
            relevance_score=relevance_score,
            snippet_type="generic"
        )
    
    def _find_boundaries(self, lines: List[str], center_line: int) -> Tuple[int, int]:
        """Find intelligent boundaries around a line."""
        # Look for function/class patterns
        start_line = self._find_start_boundary(lines, center_line)
        end_line = self._find_end_boundary(lines, center_line, start_line)
        
        return start_line, end_line
    
    def _find_start_boundary(self, lines: List[str], center_line: int) -> int:
        """Find the start boundary for extraction."""
        # Look backwards for function/class definitions or major separators
        start_patterns = [
            re.compile(r'^\s*(def|function|class)\s+\w+', re.IGNORECASE),
            re.compile(r'^\s*(public|private|protected)?\s*(class|func|fn)\s+\w+', re.IGNORECASE),
            re.compile(r'^\w+.*\{$'),  # Something starting a block
            re.compile(r'^/\*\*|\*/'),  # Documentation blocks
        ]
        
        start = max(0, center_line - 10)  # Look back up to 10 lines
        
        for i in range(center_line, start - 1, -1):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check for function/class patterns
            for pattern in start_patterns:
                if pattern.match(line):
                    return i
        
        return start
    
    def _find_end_boundary(self, lines: List[str], center_line: int, start_line: int) -> int:
        """Find the end boundary for extraction."""
        max_end = min(len(lines) - 1, center_line + 15)  # Look ahead up to 15 lines
        
        # If we detected a block start, look for balanced braces
        start_text = lines[start_line] if start_line < len(lines) else ""
        if '{' in start_text:
            return self._find_brace_end(lines, start_line, max_end)
        
        # Otherwise, look for natural breaks
        for i in range(center_line + 1, max_end + 1):
            if i >= len(lines):
                break
                
            line = lines[i].strip()
            
            # Stop at empty lines followed by new definitions
            if not line and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if (re.match(r'^\s*(def|function|class)', next_line, re.IGNORECASE) or
                    re.match(r'^\w+.*\{$', next_line)):
                    return i - 1
        
        return max_end
    
    def _find_brace_end(self, lines: List[str], start_line: int, max_line: int) -> int:
        """Find the end of a brace-delimited block."""
        brace_count = 0
        in_string = False
        string_char = None
        
        for i in range(start_line, min(max_line + 1, len(lines))):
            line = lines[i]
            
            j = 0
            while j < len(line):
                char = line[j]
                
                # Handle string literals
                if char in ['"', "'"] and (j == 0 or line[j-1] != '\\'):
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
                            return i  # Found the matching closing brace
                
                j += 1
        
        return max_line
    
    def _calculate_basic_relevance(self, text: str, query: str) -> float:
        """Calculate basic relevance score."""
        query_terms = query.lower().split()
        text_lower = text.lower()
        
        matches = sum(1 for term in query_terms if term in text_lower)
        return min(matches / len(query_terms), 1.0) if query_terms else 0.1
    
    def _overlaps_existing(
        self, 
        snippet: ExtractedSnippet, 
        existing: List[ExtractedSnippet]
    ) -> bool:
        """Check if snippet significantly overlaps with existing ones."""
        for existing_snippet in existing:
            if (snippet.start_line <= existing_snippet.end_line and 
                snippet.end_line >= existing_snippet.start_line):
                # Calculate overlap percentage
                overlap_start = max(snippet.start_line, existing_snippet.start_line)
                overlap_end = min(snippet.end_line, existing_snippet.end_line)
                overlap_size = overlap_end - overlap_start
                
                snippet_size = snippet.end_line - snippet.start_line
                if snippet_size > 0 and overlap_size / snippet_size > 0.6:
                    return True
                    
        return False
    
    def _extract_beginning(self, content: str, max_length: int) -> List[ExtractedSnippet]:
        """Extract from the beginning when no matches found."""
        lines = content.split('\n')
        text = content[:max_length]
        if len(content) > max_length:
            text += "..."
        
        return [ExtractedSnippet(
            text=text,
            start_line=1,
            end_line=min(len(lines), text.count('\n') + 1),
            relevance_score=0.1,
            snippet_type="generic"
        )]


class SnippetExtractor:
    """
    Main orchestrator for intelligent snippet extraction.
    
    Detects file language and delegates to appropriate specialized extractor
    for optimal context-aware code snippet extraction.
    """
    
    def __init__(self, language_detector: Optional[LanguageDetector] = None):
        """Initialize the snippet extractor with language detection."""
        self.language_detector = language_detector or LanguageDetector()
        self.python_extractor = PythonSnippetExtractor()
        self.js_extractor = JavaScriptSnippetExtractor()
        self.generic_extractor = GenericSnippetExtractor()
    
    def extract_snippets(
        self, 
        content: str, 
        file_path: str, 
        query: str, 
        max_snippets: int = 3,
        max_snippet_length: Optional[int] = None
    ) -> List[ExtractedSnippet]:
        """
        Extract intelligent snippets from content based on language and query.
        
        Args:
            content: Source code content
            file_path: Path to the source file for language detection
            query: Search query for relevance ranking
            max_snippets: Maximum number of snippets to return
            max_snippet_length: Maximum length per snippet (uses config default if None)
            
        Returns:
            List of ExtractedSnippet objects ranked by relevance
        """
        if max_snippet_length is None:
            max_snippet_length = config.search.SNIPPET_CONTENT_MAX_LENGTH
        
        try:
            # Detect language
            detection_result = self.language_detector.detect_language(file_path, content)
            
            # Choose appropriate extractor based on language
            if detection_result.language == "Python":
                snippets = self.python_extractor.extract_snippets(
                    content, query, max_snippets, max_snippet_length
                )
            elif detection_result.language in ["JavaScript", "TypeScript"]:
                snippets = self.js_extractor.extract_snippets(
                    content, query, max_snippets, max_snippet_length
                )
            else:
                snippets = self.generic_extractor.extract_snippets(
                    content, query, max_snippets, max_snippet_length
                )
            
            # Ensure we always return at least one snippet
            if not snippets:
                snippets = self._create_fallback_snippet(content, file_path, max_snippet_length)
            
            return snippets
            
        except Exception as e:
            logger.error(f"Error in snippet extraction for {file_path}: {e}")
            return self._create_fallback_snippet(content, file_path, max_snippet_length)
    
    def _create_fallback_snippet(
        self, 
        content: str, 
        file_path: str, 
        max_length: int
    ) -> List[ExtractedSnippet]:
        """Create a basic fallback snippet when extraction fails."""
        lines = content.split('\n')
        text = content[:max_length]
        if len(content) > max_length:
            text += "..."
        
        return [ExtractedSnippet(
            text=text,
            start_line=1,
            end_line=min(len(lines), text.count('\n') + 1),
            relevance_score=0.1,
            snippet_type="fallback"
        )]