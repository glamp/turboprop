"""Language detection module for identifying programming languages and file types."""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class LanguageDetectionResult:
    """Result of language detection analysis."""
    language: str
    file_type: str
    confidence: float
    category: str = "unknown"


class LanguageDetector:
    """
    Detects programming languages and categorizes files based on extensions and content.

    Provides intelligent language detection using file extensions as primary indicators
    and content analysis as fallback for ambiguous cases.
    """

    def __init__(self):
        """Initialize the language detector with extension mappings."""
        self._last_confidence = 0.0
        self._extension_to_language = {
            # Python
            '.py': 'Python',
            '.pyi': 'Python',
            '.pyx': 'Python',
            '.pyw': 'Python',

            # JavaScript/TypeScript
            '.js': 'JavaScript',
            '.jsx': 'JavaScript',
            '.ts': 'TypeScript',
            '.tsx': 'TypeScript',
            '.mjs': 'JavaScript',
            '.cjs': 'JavaScript',

            # Web technologies
            '.html': 'HTML',
            '.htm': 'HTML',
            '.css': 'CSS',
            '.scss': 'SCSS',
            '.sass': 'Sass',
            '.less': 'Less',

            # Java/JVM languages
            '.java': 'Java',
            '.kt': 'Kotlin',
            '.kts': 'Kotlin',
            '.scala': 'Scala',
            '.groovy': 'Groovy',
            '.clj': 'Clojure',

            # C/C++
            '.c': 'C',
            '.h': 'C',
            '.cpp': 'C++',
            '.cxx': 'C++',
            '.cc': 'C++',
            '.hpp': 'C++',
            '.hxx': 'C++',

            # Other compiled languages
            '.rs': 'Rust',
            '.go': 'Go',
            '.cs': 'C#',
            '.fs': 'F#',
            '.swift': 'Swift',

            # Scripting languages
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.pl': 'Perl',
            '.pm': 'Perl',
            '.sh': 'Shell',
            '.bash': 'Shell',
            '.zsh': 'Shell',
            '.fish': 'Shell',
            '.ps1': 'PowerShell',

            # Configuration files
            '.json': 'JSON',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.toml': 'TOML',
            '.ini': 'INI',
            '.cfg': 'INI',
            '.conf': 'Config',
            '.xml': 'XML',

            # Documentation
            '.md': 'Markdown',
            '.markdown': 'Markdown',
            '.rst': 'reStructuredText',
            '.txt': 'Text',
            '.tex': 'LaTeX',

            # Data formats
            '.csv': 'CSV',
            '.sql': 'SQL',
            '.graphql': 'GraphQL',
            '.gql': 'GraphQL',

            # Binary/media files
            '.png': 'Binary',
            '.jpg': 'Binary',
            '.jpeg': 'Binary',
            '.gif': 'Binary',
            '.pdf': 'Binary',
            '.zip': 'Binary',
            '.tar': 'Binary',
            '.gz': 'Binary',
        }

        # Special files without extensions
        self._special_files = {
            'Dockerfile': 'Dockerfile',
            'Makefile': 'Makefile',
            'Gemfile': 'Ruby',
            'requirements.txt': 'Requirements',
            'package.json': 'JSON',
            'tsconfig.json': 'JSON',
            '.gitignore': 'Config',
            '.env': 'Config',
        }

        # Content-based detection patterns
        self._content_patterns = {
            'Python': [
                re.compile(r'#!/usr/bin/env python'),
                re.compile(r'#!/usr/bin/python'),
                re.compile(r'from \w+ import'),
                re.compile(r'import \w+(\.\w+)*'),
                re.compile(r'import \w+ as \w+'),
                re.compile(r'def \w+\([^)]*\):'),
                re.compile(r'class \w+(\([^)]*\))?:'),
                re.compile(r'async def \w+\('),
                re.compile(r'if __name__ == "__main__"'),
                re.compile(r'elif\s'),
                re.compile(r'self\.\w+'),
                re.compile(r'\w+\.\w+\(\)'),  # Method calls like pd.DataFrame()
            ],
            'JavaScript': [
                re.compile(r'#!/usr/bin/env node'),
                re.compile(r'#!/usr/bin/node'),
                re.compile(r'console\.log\('),
                re.compile(r'function \w+\([^)]*\)\s*\{'),
                re.compile(r'var \w+\s*='),
                re.compile(r'const \w+\s*='),
                re.compile(r'let \w+\s*='),
                re.compile(r'require\(["\'][^"\']+["\']\)'),
                re.compile(r'module\.exports\s*='),
                re.compile(r'\([^)]*\)\s*=>\s*\{'),
                re.compile(r'class \w+\s*\{'),
                re.compile(r'\.\w+\([^)]*\)\s*\{'),  # Method calls
            ],
            'TypeScript': [
                re.compile(r'interface \w+\s*\{'),
                re.compile(r'type \w+\s*='),
                re.compile(r'enum \w+\s*\{'),
                re.compile(r'import.*from\s+["\'][^"\']+["\']'),
                re.compile(r'export\s+(class|function|interface)'),
            ],
            'Java': [
                re.compile(r'package \w+(\.\w+)*;'),
                re.compile(r'import \w+(\.\w+)*\.\*?;'),
                re.compile(r'public\s+class\s+\w+'),
                re.compile(r'public\s+static\s+void\s+main'),
                re.compile(r'System\.out\.print'),
                re.compile(r'@\w+'),  # Annotations
            ],
            'Go': [
                re.compile(r'package\s+\w+'),
                re.compile(r'import\s*\('),
                re.compile(r'import\s+"[^"]*"'),
                re.compile(r'func\s+\w+\('),
                re.compile(r'func\s+main\s*\(\s*\)'),  # More flexible main function
                re.compile(r'type\s+\w+\s+struct'),
                re.compile(r'fmt\.Print'),
            ],
            'Rust': [
                re.compile(r'use\s+\w+::'),
                re.compile(r'fn\s+\w+\('),
                re.compile(r'fn\s+main\(\)'),
                re.compile(r'struct\s+\w+\s*\{'),
                re.compile(r'impl\s+\w+'),
                re.compile(r'#\[derive\('),
                re.compile(r'let\s+\w+\s*='),
                re.compile(r'match\s+\w+\s*\{'),
            ],
            'C': [
                re.compile(r'#include\s*<[^>]+>'),
                re.compile(r'#include\s*"[^"]*"'),
                re.compile(r'int\s+main\s*\('),
                re.compile(r'printf\s*\('),
            ],
            'C++': [
                re.compile(r'#include\s*<[^>]+>'),
                re.compile(r'using\s+namespace\s+std'),
                re.compile(r'std::'),
                re.compile(r'cout\s*<<'),
                re.compile(r'class\s+\w+\s*\{'),
            ],
            'C#': [
                re.compile(r'using\s+System'),
                re.compile(r'namespace\s+\w+'),
                re.compile(r'public\s+class\s+\w+'),
                re.compile(r'Console\.WriteLine'),
                re.compile(r'static\s+void\s+Main'),
            ],
            'HTML': [
                re.compile(r'<!DOCTYPE\s+html>', re.IGNORECASE),
                re.compile(r'<html[^>]*>'),
                re.compile(r'<head[^>]*>'),
                re.compile(r'<body[^>]*>'),
                re.compile(r'<div[^>]*>'),
                re.compile(r'<script[^>]*>'),
                re.compile(r'<style[^>]*>'),
            ],
            'CSS': [
                re.compile(r'\w+\s*\{[^}]*\}'),
                re.compile(r'[.#]\w+\s*\{'),
                re.compile(r'background-color\s*:'),
                re.compile(r'font-family\s*:'),
                re.compile(r'margin\s*:\s*\d+'),
            ],
            'Shell': [
                re.compile(r'#!/bin/bash'),
                re.compile(r'#!/bin/sh'),
                re.compile(r'#!/usr/bin/env bash'),
            ]
        }

        # File categories
        self._language_to_category = {
            # Source code languages
            'Python': 'source', 'JavaScript': 'source', 'TypeScript': 'source',
            'Java': 'source', 'C': 'source', 'C++': 'source', 'Rust': 'source',
            'Go': 'source', 'Ruby': 'source', 'PHP': 'source', 'Swift': 'source',
            'Kotlin': 'source', 'Scala': 'source', 'C#': 'source', 'F#': 'source',
            'Clojure': 'source', 'Perl': 'source', 'HTML': 'source', 'CSS': 'source',
            'SCSS': 'source', 'Sass': 'source', 'Less': 'source', 'SQL': 'source',
            'GraphQL': 'source', 'Shell': 'source', 'PowerShell': 'source',

            # Configuration files
            'JSON': 'configuration', 'YAML': 'configuration', 'TOML': 'configuration',
            'INI': 'configuration', 'Config': 'configuration', 'XML': 'configuration',
            'Requirements': 'configuration',

            # Documentation
            'Markdown': 'documentation', 'reStructuredText': 'documentation',
            'Text': 'documentation', 'LaTeX': 'documentation',

            # Build/deployment
            'Dockerfile': 'build', 'Makefile': 'build',

            # Data
            'CSV': 'data',

            # Binary
            'Binary': 'binary',
        }

    def detect_language(self, file_path: str, content: str) -> LanguageDetectionResult:
        """
        Detect the programming language and file type for a given file.

        Args:
            file_path: Path to the file
            content: Content of the file

        Returns:
            LanguageDetectionResult with detected language, file type, and confidence
        """
        path = Path(file_path)
        file_name = path.name
        file_type = path.suffix.lower()

        # Check if content appears to be binary
        if self._is_binary_content(content):
            language = self._extension_to_language.get(file_type, 'Binary')
            binary_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.pdf', '.zip']
            if language == 'Binary' or file_type in binary_extensions:
                self._last_confidence = 1.0
                return LanguageDetectionResult(
                    language='Binary',
                    file_type=file_type,
                    confidence=1.0,
                    category='binary'
                )

        # Try extension-based detection first
        if file_type and file_type in self._extension_to_language:
            language = self._extension_to_language[file_type]
            category = self._language_to_category.get(language, 'unknown')
            self._last_confidence = 1.0
            return LanguageDetectionResult(
                language=language,
                file_type=file_type,
                confidence=1.0,
                category=category
            )

        # Check special files without extensions
        if file_name in self._special_files:
            language = self._special_files[file_name]
            category = self._language_to_category.get(language, 'unknown')
            self._last_confidence = 1.0
            return LanguageDetectionResult(
                language=language,
                file_type=file_type,
                confidence=1.0,
                category=category
            )

        # Fall back to content-based detection
        detected_language = self._detect_by_content(content)
        if detected_language:
            category = self._language_to_category.get(detected_language, 'unknown')
            self._last_confidence = 0.8
            return LanguageDetectionResult(
                language=detected_language,
                file_type=file_type,
                confidence=0.8,
                category=category
            )

        # Default case
        self._last_confidence = 0.0
        return LanguageDetectionResult(
            language='Unknown',
            file_type=file_type,
            confidence=0.0,
            category='unknown'
        )

    def _is_binary_content(self, content: str) -> bool:
        """
        Check if content appears to be binary based on null bytes and control characters.

        Args:
            content: File content to check

        Returns:
            True if content appears to be binary
        """
        if not content:
            return False

        # Check for null bytes (common in binary files)
        if '\x00' in content:
            return True

        # Check for high ratio of control characters
        control_chars = sum(1 for c in content if ord(c) < 32 and c not in '\t\n\r')
        if len(content) > 0 and control_chars / len(content) > 0.3:
            return True

        return False

    def _detect_by_content(self, content: str) -> Optional[str]:
        """
        Detect language based on content patterns with scoring.

        Args:
            content: File content to analyze

        Returns:
            Detected language name or None if no match
        """
        if not content.strip():
            return None

        # Score each language based on pattern matches
        language_scores = {}
        
        for language, patterns in self._content_patterns.items():
            score = 0
            matches = 0
            
            for pattern in patterns:
                if pattern.search(content):
                    matches += 1
                    # Give higher weight to more specific patterns
                    pattern_weight = 1
                    if len(pattern.pattern) > 20:  # More specific patterns
                        pattern_weight = 2
                    score += pattern_weight
            
            # Only consider languages that have at least one match
            if matches > 0:
                language_scores[language] = score
        
        # Return language with highest score
        if language_scores:
            best_language = max(language_scores, key=language_scores.get)
            # Update confidence based on score
            best_score = language_scores[best_language]
            if best_score >= 3:
                self._last_confidence = 0.95
            elif best_score >= 2:
                self._last_confidence = 0.9
            else:
                self._last_confidence = 0.8
                
            # Require at least a score of 2 for confident detection  
            if best_score >= 2:
                return best_language
            # For single pattern matches, check for discriminating patterns
            elif best_score == 1:
                # Use more strict criteria for single matches
                return self._validate_single_match(content, best_language)

        return None
        
    def _validate_single_match(self, content: str, language: str) -> Optional[str]:
        """Validate a single pattern match with additional checks."""
        # For languages that might be confused, do additional validation
        if language == 'JavaScript':
            # Check for JavaScript-specific features
            if any(pattern in content for pattern in ['require(', 'module.exports', '=>', 'const ', 'let ']):
                self._last_confidence = 0.85  # Slightly higher for validation
                return language
        elif language == 'Java':
            # Check for Java-specific features
            if any(pattern in content for pattern in ['public class', 'System.out', 'package ', '@']):
                self._last_confidence = 0.85
                return language
        elif language == 'Python':
            # Check for Python-specific features  
            if any(pattern in content for pattern in ['def ', 'class ', ':', '__']):
                self._last_confidence = 0.85
                return language
        elif language == 'Go':
            # Check for Go-specific features
            if any(pattern in content for pattern in ['package ', 'func ', 'import (', 'fmt.']):
                self._last_confidence = 0.85
                return language
        elif language == 'Rust':
            # Check for Rust-specific features
            if any(pattern in content for pattern in ['fn ', 'use ', 'let ', '::', 'match ']):
                self._last_confidence = 0.85
                return language
                
        return None

    def get_supported_languages(self) -> List[str]:
        """Get list of all supported languages."""
        languages = set(self._extension_to_language.values())
        languages.update(self._special_files.values())
        return sorted(list(languages))

    def get_supported_extensions(self) -> List[str]:
        """Get list of all supported file extensions."""
        return sorted(list(self._extension_to_language.keys()))
    
    def detect_from_content(self, content: str) -> str:
        """
        Detect language based solely on content analysis.
        
        Args:
            content: Code content to analyze
            
        Returns:
            Detected language name (lowercase)
        """
        detected_language = self._detect_by_content(content)
        if detected_language:
            return detected_language.lower()
        
        self._last_confidence = 0.0
        return "unknown"
    
    def get_confidence(self) -> float:
        """
        Get confidence score of the last detection operation.
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        return self._last_confidence


# Standalone utility functions
def detect_language_from_content(content: str) -> str:
    """
    Detect programming language from code content only.
    
    Args:
        content: Code content to analyze
        
    Returns:
        Detected language name (lowercase)
    """
    detector = LanguageDetector()
    return detector.detect_from_content(content)


def detect_language_from_extension(filename: str) -> str:
    """
    Detect programming language from file extension only.
    
    Args:
        filename: Filename or path to analyze
        
    Returns:
        Detected language name (lowercase)
    """
    path = Path(filename)
    extension = path.suffix.lower()
    
    # Create detector and get extension mapping
    detector = LanguageDetector()
    language = detector._extension_to_language.get(extension)
    
    if language:
        return language.lower()
    
    # Check special files without extensions
    file_name = path.name
    special_language = detector._special_files.get(file_name)
    if special_language:
        return special_language.lower()
    
    return "unknown"
