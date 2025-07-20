#!/usr/bin/env python3
"""
search_utils.py: Shared utility functions for search operations.

This module contains utility functions that are shared between search_operations.py
and hybrid_search.py to avoid circular import dependencies.
"""

import logging
import math
import sys
from pathlib import Path
from typing import List

from config import config
from database_manager import DatabaseManager
from embedding_helper import EmbeddingGenerator
from search_result_types import CodeSearchResult, CodeSnippet
from snippet_extractor import SnippetExtractor

# Constants
TABLE_NAME = "code_files"
DIMENSIONS = 384

logger = logging.getLogger(__name__)


def detect_file_language(file_path: str) -> str:
    """
    Detect programming language from file extension using centralized config mapping.

    Args:
        file_path: Path to the file

    Returns:
        Programming language name or 'unknown'
    """
    ext = Path(file_path).suffix.lower()
    return config.file_processing.EXTENSION_TO_LANGUAGE_MAP.get(ext, "unknown")


def create_enhanced_snippet(content: str, file_path: str, query: str = "") -> CodeSnippet:
    """
    Create an enhanced code snippet with intelligent extraction using language-aware parsing.

    Args:
        content: File content
        file_path: Path to the file
        query: Search query for relevance-based extraction

    Returns:
        CodeSnippet with enhanced information from intelligent extraction
    """
    # Use the new intelligent snippet extractor
    extractor = SnippetExtractor()

    try:
        # Extract intelligent snippets
        extracted_snippets = extractor.extract_snippets(
            content=content,
            file_path=file_path,
            query=query,
            max_snippets=1,  # For compatibility, return only the best snippet
            max_snippet_length=config.search.SNIPPET_CONTENT_MAX_LENGTH,
        )

        if extracted_snippets:
            # Convert the best ExtractedSnippet to CodeSnippet
            return extracted_snippets[0].to_code_snippet()

    except Exception as e:
        logger.warning(f"Intelligent snippet extraction failed for {file_path}: {e}")
        # Fall back to simple extraction

    # Fallback to simple truncation if intelligent extraction fails
    lines = content.split("\n")

    if len(content) <= config.search.SNIPPET_CONTENT_MAX_LENGTH:
        # Short content - include all lines
        return CodeSnippet(text=content, start_line=1, end_line=len(lines))
    else:
        # Truncated content - use first configured characters
        snippet_text = content[: config.search.SNIPPET_CONTENT_MAX_LENGTH] + "..."

        # Calculate how many lines the snippet covers
        snippet_lines = snippet_text.count("\n") + 1

        return CodeSnippet(text=snippet_text, start_line=1, end_line=min(snippet_lines, len(lines)))


def create_multi_snippets(content: str, file_path: str, query: str, max_snippets: int = 3) -> List[CodeSnippet]:
    """
    Create multiple intelligent code snippets from content.

    Args:
        content: File content
        file_path: Path to the file
        query: Search query for relevance-based extraction
        max_snippets: Maximum number of snippets to return

    Returns:
        List of CodeSnippet objects ranked by relevance
    """
    extractor = SnippetExtractor()

    try:
        # Extract multiple intelligent snippets
        extracted_snippets = extractor.extract_snippets(
            content=content,
            file_path=file_path,
            query=query,
            max_snippets=max_snippets,
            max_snippet_length=config.search.SNIPPET_CONTENT_MAX_LENGTH,
        )

        # Convert ExtractedSnippet objects to CodeSnippet objects
        return [snippet.to_code_snippet() for snippet in extracted_snippets]

    except Exception as e:
        logger.warning(f"Multi-snippet extraction failed for {file_path}: {e}")
        # Fall back to single snippet
        return [create_enhanced_snippet(content, file_path, query)]


def extract_file_metadata(file_path: str, content: str) -> dict:
    """
    Extract metadata about a file.

    Args:
        file_path: Path to the file
        content: File content

    Returns:
        Dictionary with file metadata
    """
    path_obj = Path(file_path)

    return {
        "language": detect_file_language(file_path),
        "extension": path_obj.suffix,
        "size": len(content.encode("utf-8")),  # Size in bytes
        "type": "source" if path_obj.suffix in {".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs"} else "other",
        "filename": path_obj.name,
        "directory": str(path_obj.parent),
    }


def search_index_enhanced(
    db_manager: DatabaseManager, embedder: EmbeddingGenerator, query: str, k: int
) -> List[CodeSearchResult]:
    """
    Enhanced search that returns structured CodeSearchResult objects.

    This function:
    1. Generates an embedding for the search query
    2. Uses DuckDB's array operations to compute cosine similarity
    3. Returns structured search results with rich metadata

    Args:
        db_manager: DatabaseManager instance
        embedder: EmbeddingGenerator instance for generating query embeddings
        query: Search query string
        k: Number of top results to return

    Returns:
        List of CodeSearchResult objects with enhanced metadata
    """
    try:
        # Generate embedding for the search query
        query_embedding = embedder.encode(query)

        # Prepare the query embedding and pre-calculate its norm for optimization
        query_emb_list = query_embedding.tolist()
        query_norm = math.sqrt(sum(x * x for x in query_emb_list))

        # Optimized SQL query for semantic similarity search using cosine similarity
        #
        # Mathematical background:
        # Cosine similarity = dot_product(A, B) / (||A|| × ||B||)
        # where ||A|| is the L2 norm (magnitude) of vector A
        #
        # Optimization: Pre-calculate query norm to avoid redundant SQL calculations
        # The query calculates:
        # 1. list_dot_product(embedding, query) - dot product of stored embedding and query
        # 2. sqrt(list_dot_product(embedding, embedding)) - L2 norm of stored embedding
        # 3. Uses pre-calculated query_norm instead of sqrt(list_dot_product(query, query))
        # 4. distance = 1 - cosine_similarity (lower distance = higher similarity)
        #
        # DuckDB-specific functions used:
        # - list_dot_product(): Computes dot product of two arrays/lists
        # - ?::DOUBLE[384]: Parameter placeholder cast to 384-dimensional double array
        # - sqrt(): Standard square root function for embedding norm calculation
        sql_query = f"""
        SELECT
            path,
            content,
            1 - (list_dot_product(embedding, ?::DOUBLE[{DIMENSIONS}]) /
                 (sqrt(list_dot_product(embedding, embedding)) * ?)) as distance
        FROM {TABLE_NAME}
        WHERE embedding IS NOT NULL
        ORDER BY distance ASC
        LIMIT ?
        """

        # Execute the search query with optimized parameters
        # (query embedding once + pre-calculated norm)
        results = db_manager.execute_with_retry(sql_query, (query_emb_list, query_norm, k))

        # Format results as CodeSearchResult objects
        structured_results = []
        for path, content, distance in results:
            # Convert distance to similarity score
            similarity_score = 1.0 - distance

            # Create enhanced snippet with query context
            snippet = create_enhanced_snippet(content, path, query)

            # Extract file metadata
            file_metadata = extract_file_metadata(path, content)

            # Create CodeSearchResult
            search_result = CodeSearchResult(
                file_path=path, snippet=snippet, similarity_score=similarity_score, file_metadata=file_metadata
            )

            structured_results.append(search_result)

        return structured_results

    except Exception as e:
        logger.error(
            f"Enhanced search failed for query '{query}' with k={k}. "
            f"Database connection: {db_manager is not None}. "
            f"Embedder: {embedder is not None}. "
            f"Error: {e}",
            exc_info=True,
        )
        print(f"❌ Search failed for query '{query}': {e}", file=sys.stderr)
        return []
