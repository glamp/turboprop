#!/usr/bin/env python3
"""
construct_search.py: Construct-level semantic search for programming constructs.

This module provides search functionality specifically for code constructs (functions, classes,
methods, variables, imports) extracted and indexed in the code_constructs table. It enables
more precise and granular search results compared to file-level search.

Classes:
- ConstructSearchResult: Data class representing a construct search result
- ConstructSearchOperations: Main class containing all construct search functionality

Functions:
- search_constructs: Search constructs by semantic similarity
- search_constructs_by_type: Search constructs filtered by type
- get_related_constructs: Find constructs related to a given construct
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import config
from database_manager import DatabaseManager
from embedding_helper import EmbeddingGenerator
from error_handling_utils import handle_search_errors, handle_statistics_errors
from exceptions import DatabaseError, SearchError
from search_result_types import CodeSearchResult, CodeSnippet

logger = logging.getLogger(__name__)

# Constants for construct search
CONSTRUCTS_TABLE_NAME = "code_constructs"
CONSTRUCT_EMBEDDING_DIMENSIONS = 384


@dataclass
class ConstructIdentity:
    """Core construct identification information."""

    construct_id: str
    construct_type: str  # 'function', 'class', 'method', 'variable', 'import'
    name: str
    signature: str


@dataclass
class ConstructLocation:
    """Construct location and position information."""

    file_path: str
    start_line: int
    end_line: int
    file_id: Optional[str] = None


@dataclass
class ConstructMetadata:
    """Additional construct metadata."""

    docstring: Optional[str] = None
    parent_construct_id: Optional[str] = None


@dataclass
class SearchResultMetrics:
    """Search-related metrics and confidence."""

    similarity_score: float
    confidence_level: Optional[str] = None

    def __post_init__(self):
        """Initialize confidence level based on similarity score."""
        if self.confidence_level is None:
            if self.similarity_score >= config.search.HIGH_CONFIDENCE_THRESHOLD:
                self.confidence_level = "high"
            elif self.similarity_score >= config.search.MEDIUM_CONFIDENCE_THRESHOLD:
                self.confidence_level = "medium"
            else:
                self.confidence_level = "low"


@dataclass
class ConstructSearchResult:
    """
    Represents a search result for a code construct with detailed metadata.

    This class uses composition to organize construct information into focused components.
    """

    identity: ConstructIdentity
    location: ConstructLocation
    metadata: ConstructMetadata
    metrics: SearchResultMetrics

    @property
    def construct_id(self) -> str:
        return self.identity.construct_id

    @property
    def construct_type(self) -> str:
        return self.identity.construct_type

    @property
    def name(self) -> str:
        return self.identity.name

    @property
    def signature(self) -> str:
        return self.identity.signature

    @property
    def file_path(self) -> str:
        return self.location.file_path

    @property
    def start_line(self) -> int:
        return self.location.start_line

    @property
    def end_line(self) -> int:
        return self.location.end_line

    @property
    def file_id(self) -> Optional[str]:
        return self.location.file_id

    @property
    def docstring(self) -> Optional[str]:
        return self.metadata.docstring

    @property
    def parent_construct_id(self) -> Optional[str]:
        return self.metadata.parent_construct_id

    @property
    def similarity_score(self) -> float:
        return self.metrics.similarity_score

    @property
    def confidence_level(self) -> Optional[str]:
        return self.metrics.confidence_level

    def to_code_search_result(self) -> CodeSearchResult:
        """
        Convert to a CodeSearchResult for compatibility with existing interfaces.

        Returns:
            CodeSearchResult object with construct information as snippet
        """
        # Create snippet from construct signature and docstring
        snippet_text = self.signature
        if self.docstring:
            snippet_text += f"\n\n{self.docstring}"

        snippet = CodeSnippet(text=snippet_text, start_line=self.start_line, end_line=self.end_line)

        file_metadata = {
            "construct_id": self.construct_id,
            "construct_type": self.construct_type,
            "construct_name": self.name,
            "parent_construct_id": self.parent_construct_id,
            "language": self._detect_language_from_path(self.file_path),
            "filename": Path(self.file_path).name,
            "directory": str(Path(self.file_path).parent),
        }

        return CodeSearchResult(
            file_path=self.file_path,
            snippet=snippet,
            similarity_score=self.similarity_score,
            file_metadata=file_metadata,
            confidence_level=self.confidence_level,
        )

    def _detect_language_from_path(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        ext = Path(file_path).suffix.lower()
        return config.file_processing.EXTENSION_TO_LANGUAGE_MAP.get(ext, "unknown")

    @classmethod
    def create(
        cls,
        construct_id: str,
        file_path: str,
        construct_type: str,
        name: str,
        signature: str,
        start_line: int,
        end_line: int,
        similarity_score: float,
        docstring: Optional[str] = None,
        parent_construct_id: Optional[str] = None,
        file_id: Optional[str] = None,
    ) -> "ConstructSearchResult":
        """Factory method to create ConstructSearchResult with the old interface for compatibility."""
        return cls(
            identity=ConstructIdentity(
                construct_id=construct_id, construct_type=construct_type, name=name, signature=signature
            ),
            location=ConstructLocation(file_path=file_path, start_line=start_line, end_line=end_line, file_id=file_id),
            metadata=ConstructMetadata(docstring=docstring, parent_construct_id=parent_construct_id),
            metrics=SearchResultMetrics(similarity_score=similarity_score),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "construct_id": self.construct_id,
            "file_path": self.file_path,
            "construct_type": self.construct_type,
            "name": self.name,
            "signature": self.signature,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "similarity_score": self.similarity_score,
            "docstring": self.docstring,
            "parent_construct_id": self.parent_construct_id,
            "file_id": self.file_id,
            "confidence_level": self.confidence_level,
        }


class ConstructSearchOperations:
    """
    Main class for construct-level search operations.

    This class provides methods for searching code constructs with various
    filtering and ranking options, as well as finding relationships between
    constructs.
    """

    def __init__(self, db_manager: DatabaseManager, embedder: EmbeddingGenerator):
        """Initialize with database manager and embedding generator."""
        self.db_manager = db_manager
        self.embedder = embedder

    @handle_search_errors("construct search")
    def search_constructs(
        self, query: str, k: int = 10, construct_types: Optional[List[str]] = None, min_similarity: float = 0.1
    ) -> List[ConstructSearchResult]:
        """
        Search code constructs using semantic similarity.

        Args:
            query: Natural language search query
            k: Maximum number of results to return
            construct_types: Filter by construct types (e.g., ['function', 'class'])
            min_similarity: Minimum similarity score threshold

        Returns:
            List of ConstructSearchResult objects ranked by similarity
        """
        # Generate query embedding
        query_embedding = self.embedder.generate_embeddings([query])[0]

        # Build SQL query with optional type filtering
        base_sql = f"""
            SELECT
                cc.id as construct_id,
                cf.path as file_path,
                cc.construct_type,
                cc.name,
                cc.signature,
                cc.start_line,
                cc.end_line,
                cc.docstring,
                cc.parent_construct_id,
                cc.file_id,
                list_dot_product(cc.embedding, ?) as similarity_score
            FROM {CONSTRUCTS_TABLE_NAME} cc
            JOIN code_files cf ON cc.file_id = cf.id
            WHERE cc.embedding IS NOT NULL
        """

        params = [query_embedding]

        # Add type filtering if specified
        if construct_types:
            placeholders = ",".join(["?"] * len(construct_types))
            base_sql += f" AND cc.construct_type IN ({placeholders})"
            params.extend(construct_types)

        # Add similarity threshold and ordering
        base_sql += """
            AND list_dot_product(cc.embedding, ?) >= ?
            ORDER BY similarity_score DESC
            LIMIT ?
        """
        params.extend([query_embedding, min_similarity, k])

        # Execute query
        with self.db_manager.get_connection() as conn:
            results = conn.execute(base_sql, params).fetchall()

        # Convert to ConstructSearchResult objects
        construct_results = []
        for row in results:
            construct_results.append(
                ConstructSearchResult.create(
                    construct_id=row[0],
                    file_path=row[1],
                    construct_type=row[2],
                    name=row[3],
                    signature=row[4],
                    start_line=row[5],
                    end_line=row[6],
                    docstring=row[7],
                    parent_construct_id=row[8],
                    file_id=row[9],
                    similarity_score=row[10],
                )
            )

        logger.info("Found %d construct matches for query: %s", len(construct_results), query)
        return construct_results

    def search_functions(self, query: str, k: int = 10) -> List[ConstructSearchResult]:
        """
        Search specifically for functions and methods.

        Args:
            query: Natural language search query
            k: Maximum number of results to return

        Returns:
            List of function/method ConstructSearchResult objects
        """
        return self.search_constructs(query=query, k=k, construct_types=["function", "method"])

    def search_classes(self, query: str, k: int = 10) -> List[ConstructSearchResult]:
        """
        Search specifically for classes.

        Args:
            query: Natural language search query
            k: Maximum number of results to return

        Returns:
            List of class ConstructSearchResult objects
        """
        return self.search_constructs(query=query, k=k, construct_types=["class"])

    def search_imports(self, query: str, k: int = 10) -> List[ConstructSearchResult]:
        """
        Search specifically for import statements.

        Args:
            query: Natural language search query
            k: Maximum number of results to return

        Returns:
            List of import ConstructSearchResult objects
        """
        return self.search_constructs(query=query, k=k, construct_types=["import"])

    def get_related_constructs(self, construct_id: str, k: int = 5) -> List[ConstructSearchResult]:
        """
        Find constructs related to a given construct (same file, parent/child relationships).

        Args:
            construct_id: ID of the construct to find related constructs for
            k: Maximum number of related constructs to return

        Returns:
            List of related ConstructSearchResult objects
        """
        try:
            # First get the construct details
            with self.db_manager.get_connection() as conn:
                construct_info = conn.execute(
                    """
                    SELECT file_id, parent_construct_id, construct_type, name
                    FROM code_constructs
                    WHERE id = ?
                """,
                    (construct_id,),
                ).fetchone()

                if not construct_info:
                    logger.warning("Construct not found: %s", construct_id)
                    return []

                file_id, parent_construct_id = construct_info[:2]

                # Find related constructs from same file, parent/child relationships
                related_sql = """
                    SELECT
                        cc.id as construct_id,
                        cf.path as file_path,
                        cc.construct_type,
                        cc.name,
                        cc.signature,
                        cc.start_line,
                        cc.end_line,
                        cc.docstring,
                        cc.parent_construct_id,
                        cc.file_id,
                        0.8 as similarity_score  -- High similarity for related constructs
                    FROM code_constructs cc
                    JOIN code_files cf ON cc.file_id = cf.id
                    WHERE cc.id != ? AND (
                        cc.file_id = ? OR                      -- Same file
                        cc.parent_construct_id = ? OR          -- Child constructs
                        cc.id = ?                              -- Parent construct
                    )
                    ORDER BY
                        CASE
                            WHEN cc.parent_construct_id = ? THEN 1  -- Child constructs first
                            WHEN cc.id = ? THEN 2                   -- Parent construct second
                            ELSE 3                                   -- Same file constructs last
                        END,
                        cc.start_line
                    LIMIT ?
                """

                results = conn.execute(
                    related_sql,
                    (
                        construct_id,  # Exclude the construct itself
                        file_id,  # Same file
                        construct_id,  # Child constructs (parent_construct_id = construct_id)
                        parent_construct_id,  # Parent construct
                        construct_id,  # For ordering child constructs first
                        parent_construct_id,  # For ordering parent construct second
                        k,
                    ),
                ).fetchall()

                # Convert to ConstructSearchResult objects
                related_constructs = []
                for row in results:
                    related_constructs.append(
                        ConstructSearchResult.create(
                            construct_id=row[0],
                            file_path=row[1],
                            construct_type=row[2],
                            name=row[3],
                            signature=row[4],
                            start_line=row[5],
                            end_line=row[6],
                            docstring=row[7],
                            parent_construct_id=row[8],
                            file_id=row[9],
                            similarity_score=row[10],
                        )
                    )

                logger.info("Found %d related constructs for %s", len(related_constructs), construct_id)
                return related_constructs

        except DatabaseError as error:
            logger.error("Error finding related constructs for %s: %s", construct_id, error)
            return []
        except Exception as error:
            # Fallback for unexpected errors
            logger.error("Unexpected error finding related constructs for %s: %s", construct_id, error)
            raise SearchError(f"Related construct search failed for {construct_id}: {error}") from error

    @handle_statistics_errors("construct statistics")
    def get_construct_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about indexed constructs.

        Returns:
            Dictionary with construct statistics
        """
        with self.db_manager.get_connection() as conn:
            # Get overall counts
            total_constructs = conn.execute("SELECT COUNT(*) FROM code_constructs").fetchone()[0]

            # Get counts by type
            type_counts = conn.execute(
                """
                SELECT construct_type, COUNT(*) as count
                FROM code_constructs
                GROUP BY construct_type
                ORDER BY count DESC
            """
            ).fetchall()

            # Get embedding coverage
            embedded_constructs = conn.execute(
                """
                SELECT COUNT(*) FROM code_constructs
                WHERE embedding IS NOT NULL
            """
            ).fetchone()[0]

            return {
                "total_constructs": total_constructs,
                "embedded_constructs": embedded_constructs,
                "embedding_coverage": embedded_constructs / total_constructs if total_constructs > 0 else 0,
                "construct_types": dict(type_counts),
            }


def format_construct_search_results(
    results: List[ConstructSearchResult], query: str, show_signatures: bool = True, show_docstrings: bool = True
) -> str:
    """
    Format construct search results for display.

    Args:
        results: List of ConstructSearchResult objects
        query: Original search query
        show_signatures: Whether to include construct signatures
        show_docstrings: Whether to include docstrings when available

    Returns:
        Formatted string representation of the results
    """
    if not results:
        return f"No construct matches found for query: '{query}'"

    formatted_lines = [f"🔍 Found {len(results)} construct matches for: '{query}'\n"]

    for i, result in enumerate(results, 1):
        # Result header with construct info
        confidence_emoji = {"high": "🎯", "medium": "✅", "low": "⚠️"}.get(result.confidence_level, "❓")

        formatted_lines.append(
            f"{confidence_emoji} [{i}] {result.construct_type.upper()}: {result.name} "
            f"(similarity: {result.similarity_score:.3f})"
        )

        # File location
        formatted_lines.append(f"   📁 {result.file_path}:{result.start_line}-{result.end_line}")

        # Construct signature
        if show_signatures and result.signature:
            formatted_lines.append(f"   💻 {result.signature}")

        # Docstring if available
        if show_docstrings and result.docstring:
            # Truncate long docstrings
            docstring = result.docstring[:200] + "..." if len(result.docstring) > 200 else result.docstring
            formatted_lines.append(f"   📝 {docstring}")

        # Parent relationship
        if result.parent_construct_id:
            formatted_lines.append(f"   👆 Child of: {result.parent_construct_id}")

        formatted_lines.append("")  # Empty line between results

    return "\n".join(formatted_lines)
