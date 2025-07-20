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
                re.compile(r'import \w+'),
                re.compile(r'from \w+ import'),
                re.compile(r'def \w+\('),
                re.compile(r'class \w+:'),
            ],
            'JavaScript': [
                re.compile(r'#!/usr/bin/env node'),
                re.compile(r'#!/usr/bin/node'),
                re.compile(r'console\.log\('),
                re.compile(r'function \w+\('),
                re.compile(r'var \w+ ='),
                re.compile(r'const \w+ ='),
                re.compile(r'let \w+ ='),
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
            return LanguageDetectionResult(
                language=detected_language,
                file_type=file_type,
                confidence=0.8,
                category=category
            )

        # Default case
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
        Detect language based on content patterns.

        Args:
            content: File content to analyze

        Returns:
            Detected language name or None if no match
        """
        if not content.strip():
            return None

        # Check content patterns for each language
        for language, patterns in self._content_patterns.items():
            for pattern in patterns:
                if pattern.search(content):
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
