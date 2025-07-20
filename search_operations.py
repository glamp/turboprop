#!/usr/bin/env python3
"""
search_operations.py: Search functionality for the Turboprop code search system.

This module contains functions for searching the code index:
- Semantic search using embeddings
- Result formatting and presentation
- Search result ranking and filtering
"""

import logging
import sys
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

from database_manager import DatabaseManager
from embedding_helper import EmbeddingGenerator
from search_result_types import CodeSnippet, CodeSearchResult, SearchMetadata

# Constants
TABLE_NAME = "code_files"
DIMENSIONS = 384

# Setup logging
logger = logging.getLogger(__name__)


def _detect_file_language(file_path: str) -> str:
    """
    Detect programming language from file extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Programming language name or 'unknown'
    """
    extension_map = {
        '.py': 'python',
        '.js': 'javascript', 
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.jsx': 'javascript',
        '.java': 'java',
        '.c': 'c',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.h': 'c',
        '.hpp': 'cpp',
        '.cs': 'csharp',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.m': 'objective-c',
        '.sh': 'bash',
        '.bash': 'bash',
        '.html': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.xml': 'xml',
        '.md': 'markdown',
        '.rst': 'restructuredtext',
        '.txt': 'text'
    }
    
    ext = Path(file_path).suffix.lower()
    return extension_map.get(ext, 'unknown')


def _create_enhanced_snippet(content: str, file_path: str) -> CodeSnippet:
    """
    Create an enhanced code snippet with intelligent extraction.
    
    Args:
        content: File content
        file_path: Path to the file
        
    Returns:
        CodeSnippet with enhanced information
    """
    # For now, use the simple truncation approach with line number calculation
    # TODO: In future iterations, add intelligent function/class boundary detection
    
    lines = content.split('\n')
    
    if len(content) <= 200:
        # Short content - include all lines
        return CodeSnippet(
            text=content,
            start_line=1,
            end_line=len(lines)
        )
    else:
        # Truncated content - use first 200 characters
        snippet_text = content[:200] + "..."
        
        # Calculate how many lines the snippet covers
        snippet_lines = snippet_text.count('\n') + 1
        
        return CodeSnippet(
            text=snippet_text,
            start_line=1,
            end_line=min(snippet_lines, len(lines))
        )


def _extract_file_metadata(file_path: str, content: str) -> dict:
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
        'language': _detect_file_language(file_path),
        'extension': path_obj.suffix,
        'size': len(content.encode('utf-8')),  # Size in bytes
        'type': 'source' if path_obj.suffix in {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs'} else 'other',
        'filename': path_obj.name,
        'directory': str(path_obj.parent)
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

        # Prepare the query embedding as a list for DuckDB
        query_emb_list = query_embedding.tolist()

        # Use DuckDB's array operations for cosine similarity
        # Cosine similarity = dot_product / (norm_a * norm_b)
        # Since embeddings are normalized, we can use dot product directly
        sql_query = f"""
        SELECT
            path,
            content,
            1 - (list_dot_product(embedding, ?::DOUBLE[{DIMENSIONS}]) /
                 (sqrt(list_dot_product(embedding, embedding)) *
                  sqrt(list_dot_product(
                      ?::DOUBLE[{DIMENSIONS}], ?::DOUBLE[{DIMENSIONS}]
                  )))) as distance
        FROM {TABLE_NAME}
        WHERE embedding IS NOT NULL
        ORDER BY distance ASC
        LIMIT ?
        """

        # Execute the search query
        results = db_manager.execute_with_retry(sql_query, (query_emb_list, query_emb_list, query_emb_list, k))

        # Format results as CodeSearchResult objects
        structured_results = []
        for path, content, distance in results:
            # Convert distance to similarity score
            similarity_score = 1.0 - distance
            
            # Create enhanced snippet
            snippet = _create_enhanced_snippet(content, path)
            
            # Extract file metadata
            file_metadata = _extract_file_metadata(path, content)
            
            # Create CodeSearchResult
            search_result = CodeSearchResult(
                file_path=path,
                snippet=snippet,
                similarity_score=similarity_score,
                file_metadata=file_metadata
            )
            
            structured_results.append(search_result)

        return structured_results

    except Exception as e:
        logger.error(f"Error during enhanced search: {e}")
        print(f"❌ Search failed: {e}", file=sys.stderr)
        return []


def search_index(
    db_manager: DatabaseManager, embedder: EmbeddingGenerator, query: str, k: int
) -> List[Tuple[str, str, float]]:
    """
    Search the code index for files semantically similar to the given query.
    
    LEGACY COMPATIBILITY VERSION: This function maintains the original tuple-based
    return format for backward compatibility. New code should use search_index_enhanced().

    This function:
    1. Generates an embedding for the search query
    2. Uses DuckDB's array operations to compute cosine similarity
    3. Returns the top k most similar files with their similarity scores

    Args:
        db_manager: DatabaseManager instance
        embedder: EmbeddingGenerator instance for generating query embeddings
        query: Search query string
        k: Number of top results to return

    Returns:
        List of tuples containing (file_path, content_snippet, distance_score)
    """
    # Use the enhanced search and convert to legacy format
    enhanced_results = search_index_enhanced(db_manager, embedder, query, k)
    
    # Convert to legacy tuple format
    legacy_results = []
    for result in enhanced_results:
        legacy_tuple = result.to_tuple()
        legacy_results.append(legacy_tuple)
    
    return legacy_results


def format_enhanced_search_results(results: List[CodeSearchResult], query: str, repo_path: Optional[str] = None) -> str:
    """
    Format enhanced search results for display with rich metadata.

    Args:
        results: List of CodeSearchResult objects
        query: Original search query
        repo_path: Optional repository path for relative path display

    Returns:
        Formatted string representation of enhanced search results
    """
    if not results:
        return f"No results found for query: '{query}'"

    formatted_lines = []
    formatted_lines.append(f"🔍 Found {len(results)} results for: '{query}'")
    formatted_lines.append("=" * 50)

    for i, result in enumerate(results, 1):
        # Format path (use relative path if repo_path provided)
        display_path = result.get_relative_path(repo_path) if repo_path else result.file_path
        
        # Display enriched information
        formatted_lines.append(f"{i}. {display_path}")
        formatted_lines.append(f"   Similarity: {result.similarity_percentage:.1f}% ({result.confidence_level} confidence)")
        
        # Show file metadata if available
        if result.file_metadata:
            metadata = result.file_metadata
            lang = metadata.get('language', 'unknown')
            size = metadata.get('size', 0)
            if size > 0:
                size_str = f"{size} bytes" if size < 1024 else f"{size/1024:.1f}KB"
                formatted_lines.append(f"   Type: {lang} ({size_str})")
            else:
                formatted_lines.append(f"   Type: {lang}")
        
        # Show snippet with line information
        snippet = result.snippet
        if snippet.start_line == snippet.end_line:
            line_info = f"Line {snippet.start_line}"
        else:
            line_info = f"Lines {snippet.start_line}-{snippet.end_line}"
        
        formatted_lines.append(f"   {line_info}: {snippet.text.strip()}")
        formatted_lines.append("")

    return "\n".join(formatted_lines)


def format_search_results(results: List[Tuple[str, str, float]], query: str, repo_path: Optional[str] = None) -> str:
    """
    Format search results for display.

    Args:
        results: List of search results from search_index
        query: Original search query
        repo_path: Optional repository path for relative path display

    Returns:
        Formatted string representation of search results
    """
    if not results:
        return f"No results found for query: '{query}'"

    formatted_lines = []
    formatted_lines.append(f"🔍 Found {len(results)} results for: '{query}'")
    formatted_lines.append("=" * 50)

    for i, (path, snippet, distance) in enumerate(results, 1):
        # Convert distance to similarity percentage
        similarity_pct = (1 - distance) * 100

        # Format path (use relative path if repo_path provided)
        display_path = path
        if repo_path and path.startswith(repo_path):
            display_path = path[len(repo_path) :].lstrip("/")

        formatted_lines.append(f"{i}. {display_path}")
        formatted_lines.append(f"   Similarity: {similarity_pct:.1f}%")
        formatted_lines.append(f"   Preview: {snippet.strip()}")
        formatted_lines.append("")

    return "\n".join(formatted_lines)


def get_file_content(db_manager: DatabaseManager, file_path: str) -> Optional[str]:
    """
    Get the full content of a file from the database.

    Args:
        db_manager: DatabaseManager instance
        file_path: Path to the file

    Returns:
        File content if found, None otherwise
    """
    try:
        result = db_manager.execute_with_retry(f"SELECT content FROM {TABLE_NAME} WHERE path = ?", (file_path,))
        return result[0][0] if result else None
    except Exception as e:
        logger.error(f"Error getting file content for {file_path}: {e}")
        return None


def search_by_file_extension(db_manager: DatabaseManager, extension: str, limit: int = 10) -> List[Tuple[str, str]]:
    """
    Search for files by file extension.

    Args:
        db_manager: DatabaseManager instance
        extension: File extension to search for (e.g., '.py', '.js')
        limit: Maximum number of results to return

    Returns:
        List of tuples containing (file_path, content_snippet)
    """
    try:
        # Ensure extension starts with a dot
        if not extension.startswith("."):
            extension = "." + extension

        sql_query = f"""
        SELECT path, content
        FROM {TABLE_NAME}
        WHERE path LIKE '%{extension}'
        ORDER BY path
        LIMIT ?
        """

        results = db_manager.execute_with_retry(sql_query, (limit,))

        # Format results with snippets
        formatted_results = []
        for path, content in results:
            snippet = content[:200] + "..." if len(content) > 200 else content
            formatted_results.append((path, snippet))

        return formatted_results

    except Exception as e:
        logger.error(f"Error searching by extension {extension}: {e}")
        return []


def search_by_filename(db_manager: DatabaseManager, filename_pattern: str, limit: int = 10) -> List[Tuple[str, str]]:
    """
    Search for files by filename pattern.

    Args:
        db_manager: DatabaseManager instance
        filename_pattern: Pattern to search for in filenames
        limit: Maximum number of results to return

    Returns:
        List of tuples containing (file_path, content_snippet)
    """
    try:
        sql_query = f"""
        SELECT path, content
        FROM {TABLE_NAME}
        WHERE path LIKE '%{filename_pattern}%'
        ORDER BY path
        LIMIT ?
        """

        results = db_manager.execute_with_retry(sql_query, (limit,))

        # Format results with snippets
        formatted_results = []
        for path, content in results:
            snippet = content[:200] + "..." if len(content) > 200 else content
            formatted_results.append((path, snippet))

        return formatted_results

    except Exception as e:
        logger.error(f"Error searching by filename pattern {filename_pattern}: {e}")
        return []


def get_search_statistics(db_manager: DatabaseManager) -> dict:
    """
    Get statistics about the search index.

    Args:
        db_manager: DatabaseManager instance

    Returns:
        Dictionary with search statistics
    """
    try:
        stats = {}

        # Total files indexed
        result = db_manager.execute_with_retry(f"SELECT COUNT(*) FROM {TABLE_NAME}")
        stats["total_files"] = result[0][0] if result else 0

        # Files with embeddings
        result = db_manager.execute_with_retry(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE embedding IS NOT NULL")
        stats["files_with_embeddings"] = result[0][0] if result else 0

        # File types
        result = db_manager.execute_with_retry(
            f"""
            SELECT
                CASE
                    WHEN path LIKE '%.py' THEN 'Python'
                    WHEN path LIKE '%.js' THEN 'JavaScript'
                    WHEN path LIKE '%.ts' THEN 'TypeScript'
                    WHEN path LIKE '%.java' THEN 'Java'
                    WHEN path LIKE '%.cpp' OR path LIKE '%.c' THEN 'C/C++'
                    WHEN path LIKE '%.go' THEN 'Go'
                    WHEN path LIKE '%.rs' THEN 'Rust'
                    WHEN path LIKE '%.md' THEN 'Markdown'
                    WHEN path LIKE '%.json' THEN 'JSON'
                    WHEN path LIKE '%.yml' OR path LIKE '%.yaml' THEN 'YAML'
                    ELSE 'Other'
                END as file_type,
                COUNT(*) as count
            FROM {TABLE_NAME}
            GROUP BY file_type
            ORDER BY count DESC
            """
        )
        stats["file_types"] = {row[0]: row[1] for row in result} if result else {}

        return stats

    except Exception as e:
        logger.error(f"Error getting search statistics: {e}")
        return {}


def find_similar_files_enhanced(
    db_manager: DatabaseManager,
    embedder: EmbeddingGenerator,
    reference_file: str,
    k: int = 5,
) -> List[CodeSearchResult]:
    """
    Find files similar to a reference file using enhanced structured results.

    Args:
        db_manager: DatabaseManager instance
        embedder: EmbeddingGenerator instance
        reference_file: Path to the reference file
        k: Number of similar files to return

    Returns:
        List of CodeSearchResult objects containing enhanced similarity information
    """
    try:
        # Get the embedding of the reference file
        result = db_manager.execute_with_retry(f"SELECT embedding FROM {TABLE_NAME} WHERE path = ?", (reference_file,))

        if not result:
            logger.warning(f"Reference file not found in index: {reference_file}")
            return []

        reference_embedding = result[0][0]

        # Find similar files
        sql_query = f"""
        SELECT
            path,
            content,
            1 - (list_dot_product(embedding, ?::DOUBLE[{DIMENSIONS}]) /
                 (sqrt(list_dot_product(embedding, embedding)) *
                  sqrt(list_dot_product(
                      ?::DOUBLE[{DIMENSIONS}], ?::DOUBLE[{DIMENSIONS}]
                  )))) as distance
        FROM {TABLE_NAME}
        WHERE embedding IS NOT NULL AND path != ?
        ORDER BY distance ASC
        LIMIT ?
        """

        results = db_manager.execute_with_retry(
            sql_query,
            (
                reference_embedding,
                reference_embedding,
                reference_embedding,
                reference_file,
                k,
            ),
        )

        # Format results as CodeSearchResult objects
        structured_results = []
        for path, content, distance in results:
            # Convert distance to similarity score
            similarity_score = 1.0 - distance
            
            # Create enhanced snippet
            snippet = _create_enhanced_snippet(content, path)
            
            # Extract file metadata
            file_metadata = _extract_file_metadata(path, content)
            
            # Create CodeSearchResult
            search_result = CodeSearchResult(
                file_path=path,
                snippet=snippet,
                similarity_score=similarity_score,
                file_metadata=file_metadata
            )
            
            structured_results.append(search_result)

        return structured_results

    except Exception as e:
        logger.error(f"Error finding similar files to {reference_file}: {e}")
        return []


def find_similar_files(
    db_manager: DatabaseManager,
    embedder: EmbeddingGenerator,
    reference_file: str,
    k: int = 5,
) -> List[Tuple[str, str, float]]:
    """
    Find files similar to a reference file.
    
    LEGACY COMPATIBILITY VERSION: This function maintains the original tuple-based
    return format for backward compatibility. New code should use find_similar_files_enhanced().

    Args:
        db_manager: DatabaseManager instance
        embedder: EmbeddingGenerator instance
        reference_file: Path to the reference file
        k: Number of similar files to return

    Returns:
        List of tuples containing (file_path, content_snippet, distance_score)
    """
    # Use the enhanced version and convert to legacy format
    enhanced_results = find_similar_files_enhanced(db_manager, embedder, reference_file, k)
    
    # Convert to legacy tuple format
    legacy_results = []
    for result in enhanced_results:
        legacy_tuple = result.to_tuple()
        legacy_results.append(legacy_tuple)
    
    return legacy_results
