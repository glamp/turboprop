#!/usr/bin/env python3
"""
search_result_types.py: Enhanced data structures for search results.

This module defines structured data classes that replace simple tuples in search results,
providing rich metadata and support for complex AI reasoning while maintaining backward
compatibility with existing tools.

Classes:
- CodeSnippet: Represents a code fragment with line numbers and context
- CodeSearchResult: Comprehensive search result with metadata
- SearchMetadata: Overall search execution information
"""

from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .config import config
from .ide_integration import SyntaxHighlightingHint, get_ide_navigation_urls, get_mcp_navigation_actions

# Constants
PERCENTAGE_MULTIPLIER = 100.0


def _ensure_float(value: Union[float, Decimal, None]) -> float:
    """
    Safely convert numeric values to float, handling both float and Decimal types.

    Validates that similarity scores are within the expected range (0.0 to 1.0) and
    handles type conversion between float and Decimal types as returned by DuckDB.

    Args:
        value: Numeric value (float or Decimal), or None

    Returns:
        float: The value as a float

    Raises:
        ValueError: If value is None, negative, or outside the range [0.0, 1.0]
        TypeError: If value is not a supported numeric type

    Examples:
        >>> _ensure_float(0.85)  # float input
        0.85
        >>> from decimal import Decimal
        >>> _ensure_float(Decimal('0.92'))  # Decimal input from DuckDB
        0.92
    """
    if value is None:
        raise ValueError("Similarity score cannot be None")

    if not isinstance(value, (float, Decimal)):
        raise TypeError(f"Unsupported type for similarity score: {type(value)}. Expected float or Decimal.")

    # Convert to float
    if isinstance(value, Decimal):
        float_value = float(value)
    else:
        float_value = value

    # Validate bounds for similarity scores
    if float_value < 0.0 or float_value > 1.0:
        raise ValueError(f"Similarity score must be between 0.0 and 1.0, got: {float_value}")

    return float_value


@dataclass
class CodeSnippet:
    """
    Represents a code fragment with line numbers and contextual information.

    This class encapsulates a piece of code with its location information,
    enabling precise navigation and better understanding of the code context.
    """

    text: str
    start_line: int
    end_line: int
    context_before: Optional[str] = None
    context_after: Optional[str] = None

    def __str__(self) -> str:
        """
        String representation showing the code with line information.

        Returns:
            Human-readable string with line numbers and code text
        """
        if self.start_line == self.end_line:
            line_info = f"Line {self.start_line}"
        else:
            line_info = f"Lines {self.start_line}-{self.end_line}"

        max_length = config.search.SNIPPET_DISPLAY_MAX_LENGTH
        truncated_text = self.text[:max_length]
        ellipsis = "..." if len(self.text) > max_length else ""
        return f"{line_info}: {truncated_text}{ellipsis}"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the CodeSnippet
        """
        return asdict(self)


@dataclass
class CodeSearchResult:
    """
    Comprehensive search result with rich metadata and backward compatibility.

    This class represents a single search result with detailed information about
    the matched file, including the code snippet(s), similarity score, file metadata,
    confidence assessment, and explainable match reasons. Supports both single and
    multiple snippets per file.
    """

    file_path: str
    snippet: CodeSnippet
    similarity_score: Union[float, Decimal]
    file_metadata: Optional[Dict[str, Any]] = None
    confidence_level: Optional[str] = None
    # Additional snippets from the same file (for multi-snippet support)
    additional_snippets: List[CodeSnippet] = field(default_factory=list)
    # Repository context information
    repository_context: Optional[Dict[str, Any]] = None
    # Advanced ranking and explainability fields
    match_reasons: List[str] = field(default_factory=list)
    ranking_score: Optional[float] = None
    ranking_factors: Optional[Dict[str, float]] = None
    # IDE navigation and integration fields
    ide_navigation_urls: Optional[List[Dict[str, Any]]] = None
    syntax_highlighting_hints: Optional[List[SyntaxHighlightingHint]] = None
    mcp_navigation_actions: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize default values after dataclass construction."""
        # Convert similarity_score to float for consistency
        self.similarity_score = _ensure_float(self.similarity_score)

        if self.file_metadata is None:
            self.file_metadata = {}

        if self.confidence_level is None:
            # Auto-assign confidence based on configurable similarity score thresholds
            if self.similarity_score >= config.search.HIGH_CONFIDENCE_THRESHOLD:
                self.confidence_level = "high"
            elif self.similarity_score >= config.search.MEDIUM_CONFIDENCE_THRESHOLD:
                self.confidence_level = "medium"
            else:
                self.confidence_level = "low"

    @property
    def all_snippets(self) -> List[CodeSnippet]:
        """
        Get all snippets (primary + additional) for this search result.

        Returns:
            List of all CodeSnippet objects for this file
        """
        return [self.snippet] + self.additional_snippets

    def add_snippet(self, snippet: CodeSnippet) -> None:
        """
        Add an additional snippet to this search result.

        Args:
            snippet: CodeSnippet to add
        """
        self.additional_snippets.append(snippet)

    def generate_ide_navigation(self) -> None:
        """
        Generate IDE navigation URLs and actions for this search result.

        This method populates the ide_navigation_urls and mcp_navigation_actions
        fields with data suitable for IDE integration.
        """
        primary_line = self.snippet.start_line

        # Generate IDE navigation URLs
        nav_urls = get_ide_navigation_urls(self.file_path, primary_line)
        self.ide_navigation_urls = [
            {"ide": url.display_name, "url": url.url, "available": url.is_available, "ide_type": url.ide_type.value}
            for url in nav_urls
        ]

        # Generate MCP navigation actions
        self.mcp_navigation_actions = get_mcp_navigation_actions(self.file_path, primary_line)

    def generate_syntax_hints(self, file_content: Optional[str] = None) -> None:
        """
        Generate syntax highlighting hints for this search result.

        Args:
            file_content: Optional file content to analyze. If not provided,
                         will attempt to read the file from disk.
        """
        if file_content is None:
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    file_content = f.read()
            except (IOError, UnicodeDecodeError):
                # If we can't read the file, skip syntax highlighting
                return

        from ide_integration import ide_integration

        self.syntax_highlighting_hints = ide_integration.generate_syntax_hints(
            self.file_path, file_content, self.snippet.start_line
        )

    @classmethod
    def from_multi_snippets(
        cls,
        file_path: str,
        snippets: List[CodeSnippet],
        similarity_score: Union[float, Decimal],
        file_metadata: Optional[Dict[str, Any]] = None,
        repository_context: Optional[Dict[str, Any]] = None,
    ) -> "CodeSearchResult":
        """
        Create a CodeSearchResult from multiple snippets.

        Args:
            file_path: Path to the file
            snippets: List of CodeSnippet objects (first becomes primary)
            similarity_score: Similarity score for the result
            file_metadata: Optional file metadata
            repository_context: Optional repository context

        Returns:
            CodeSearchResult with primary and additional snippets
        """
        if not snippets:
            raise ValueError("At least one snippet is required")

        primary_snippet = snippets[0]
        additional_snippets = snippets[1:] if len(snippets) > 1 else []

        return cls(
            file_path=file_path,
            snippet=primary_snippet,
            similarity_score=similarity_score,
            file_metadata=file_metadata,
            additional_snippets=additional_snippets,
            repository_context=repository_context,
        )

    @property
    def similarity_percentage(self) -> float:
        """
        Get similarity score as a percentage.

        Returns:
            Similarity score as percentage (0-100)
        """
        return _ensure_float(self.similarity_score) * PERCENTAGE_MULTIPLIER

    def get_relative_path(self, base_path: str) -> str:
        """
        Get file path relative to a base directory.

        Args:
            base_path: Base directory path to make relative to

        Returns:
            Relative path if file_path is under base_path, otherwise original path
        """
        try:
            file_path_obj = Path(self.file_path)
            base_path_obj = Path(base_path)

            if str(file_path_obj).startswith(str(base_path_obj)):
                return str(file_path_obj.relative_to(base_path_obj))
            else:
                return self.file_path
        except ValueError:
            # ValueError typically occurs when paths are on different drives or malformed
            # Fall back to original path for cross-drive path calculations
            return self.file_path
        except OSError:
            # OSError occurs with invalid path characters or permission issues
            # Fall back to original path for filesystem access issues
            return self.file_path

    @classmethod
    def from_tuple(
        cls, legacy_tuple: Tuple[str, str, float], repository_context: Optional[Dict[str, Any]] = None
    ) -> "CodeSearchResult":
        """
        Create CodeSearchResult from legacy tuple format.

        This method provides backward compatibility with the existing tuple format
        used throughout the codebase: (file_path, snippet_text, distance_score).

        Args:
            legacy_tuple: Tuple in format (file_path, snippet_text, distance_score)
            repository_context: Optional repository context to include

        Returns:
            CodeSearchResult instance created from tuple data
        """
        file_path, snippet_text, distance_score = legacy_tuple

        # Convert distance to similarity (distance = 1 - similarity)
        similarity_score = 1.0 - distance_score

        # Create basic code snippet (assuming single line for legacy compatibility)
        snippet = CodeSnippet(text=snippet_text, start_line=1, end_line=1)

        return cls(
            file_path=file_path,
            snippet=snippet,
            similarity_score=similarity_score,
            repository_context=repository_context,
        )

    def to_tuple(self) -> Tuple[str, str, float]:
        """
        Convert to legacy tuple format for backward compatibility.

        Returns:
            Tuple in format (file_path, snippet_text, distance_score)
        """
        # Convert similarity to distance (distance = 1 - similarity)
        distance_score = 1.0 - self.similarity_score

        return (self.file_path, self.snippet.text, distance_score)

    def __str__(self) -> str:
        """
        String representation for backward compatibility.

        This returns a format that can be used in contexts expecting the
        old tuple-based string representations.

        Returns:
            String representation compatible with legacy formats
        """
        distance = 1.0 - self.similarity_score
        return f"({self.file_path!r}, {self.snippet.text!r}, {distance:.3f})"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation suitable for JSON/API responses
        """
        result = {
            "file_path": self.file_path,
            "snippet": self.snippet.to_dict(),
            "similarity_score": self.similarity_score,
            "similarity_percentage": self.similarity_percentage,
            "file_metadata": self.file_metadata,
            "confidence_level": self.confidence_level,
            "repository_context": self.repository_context,
            "match_reasons": self.match_reasons,
            "ranking_score": self.ranking_score,
            "ranking_factors": self.ranking_factors,
            "ide_navigation_urls": self.ide_navigation_urls,
            "mcp_navigation_actions": self.mcp_navigation_actions,
        }

        # Convert syntax highlighting hints to dict format
        if self.syntax_highlighting_hints:
            result["syntax_highlighting_hints"] = [
                {
                    "language": hint.language,
                    "token_type": hint.token_type,
                    "start_line": hint.start_line,
                    "end_line": hint.end_line,
                    "start_column": hint.start_column,
                    "end_column": hint.end_column,
                }
                for hint in self.syntax_highlighting_hints
            ]

        # Include additional snippets if present
        if self.additional_snippets:
            result["additional_snippets"] = [s.to_dict() for s in self.additional_snippets]
            result["total_snippets"] = len(self.all_snippets)

        return result


@dataclass
class SearchMetadata:
    """
    Overall search execution information and statistics.

    This class captures metadata about a search operation, including timing,
    result statistics, and search parameters used.
    """

    query: str
    total_results: int
    execution_time: Optional[float] = None
    confidence_distribution: Optional[Dict[str, int]] = None
    search_parameters: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        """
        String representation of search metadata.

        Returns:
            Human-readable summary of search execution
        """
        time_info = f" in {self.execution_time:.3f}s" if self.execution_time else ""
        return f"Search '{self.query}': {self.total_results} results{time_info}"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the SearchMetadata
        """
        return asdict(self)


# Utility functions for working with search results


def convert_legacy_results(legacy_results: list) -> Tuple[list, SearchMetadata]:
    """
    Convert a list of legacy tuple results to structured format.

    Args:
        legacy_results: List of tuples in format (file_path, snippet, distance)

    Returns:
        Tuple of (CodeSearchResult list, SearchMetadata)
    """
    structured_results = []

    for legacy_tuple in legacy_results:
        result = CodeSearchResult.from_tuple(legacy_tuple)
        structured_results.append(result)

    # Create metadata
    confidence_counts = {"high": 0, "medium": 0, "low": 0}
    for result in structured_results:
        if result.confidence_level is not None:
            confidence_counts[result.confidence_level] += 1

    metadata = SearchMetadata(
        query="legacy_conversion", total_results=len(structured_results), confidence_distribution=confidence_counts
    )

    return structured_results, metadata


def results_to_legacy_format(results: list) -> list:
    """
    Convert structured results back to legacy tuple format.

    Args:
        results: List of CodeSearchResult instances

    Returns:
        List of tuples in legacy format (file_path, snippet, distance)
    """
    return [result.to_tuple() for result in results]
