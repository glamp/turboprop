#!/usr/bin/env python3
"""
search_operations.py: Search functionality for the Turboprop code search system.

This module contains functions for searching the code index:
- Semantic search using embeddings
- Result formatting and presentation
- Search result ranking and filtering
"""

import sys
from typing import List, Tuple, Optional
import logging

from database_manager import DatabaseManager
from embedding_helper import EmbeddingGenerator

# Constants
TABLE_NAME = "code_files"
DIMENSIONS = 384

# Setup logging
logger = logging.getLogger(__name__)


def search_index(
    db_manager: DatabaseManager, 
    embedder: EmbeddingGenerator, 
    query: str, 
    k: int
) -> List[Tuple[str, str, float]]:
    """
    Search the code index for files semantically similar to the given query.

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
        List of tuples containing (file_path, content_snippet, similarity_score)
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
                  sqrt(list_dot_product(?::DOUBLE[{DIMENSIONS}], ?::DOUBLE[{DIMENSIONS}])))) as distance
        FROM {TABLE_NAME}
        WHERE embedding IS NOT NULL
        ORDER BY distance ASC
        LIMIT ?
        """
        
        # Execute the search query
        results = db_manager.execute_with_retry(
            sql_query, 
            (query_emb_list, query_emb_list, query_emb_list, k)
        )
        
        # Format results
        formatted_results = []
        for path, content, distance in results:
            # Create a snippet from the content (first 200 characters)
            snippet = content[:200] + "..." if len(content) > 200 else content
            formatted_results.append((path, snippet, distance))
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error during search: {e}")
        print(f"❌ Search failed: {e}", file=sys.stderr)
        return []


def format_search_results(
    results: List[Tuple[str, str, float]], 
    query: str, 
    repo_path: Optional[str] = None
) -> str:
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
            display_path = path[len(repo_path):].lstrip('/')
        
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
        result = db_manager.execute_with_retry(
            f"SELECT content FROM {TABLE_NAME} WHERE path = ?",
            (file_path,)
        )
        return result[0][0] if result else None
    except Exception as e:
        logger.error(f"Error getting file content for {file_path}: {e}")
        return None


def search_by_file_extension(
    db_manager: DatabaseManager, 
    extension: str, 
    limit: int = 10
) -> List[Tuple[str, str]]:
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
        if not extension.startswith('.'):
            extension = '.' + extension
            
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


def search_by_filename(
    db_manager: DatabaseManager, 
    filename_pattern: str, 
    limit: int = 10
) -> List[Tuple[str, str]]:
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
        result = db_manager.execute_with_retry(
            f"SELECT COUNT(*) FROM {TABLE_NAME}"
        )
        stats['total_files'] = result[0][0] if result else 0
        
        # Files with embeddings
        result = db_manager.execute_with_retry(
            f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE embedding IS NOT NULL"
        )
        stats['files_with_embeddings'] = result[0][0] if result else 0
        
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
        stats['file_types'] = {row[0]: row[1] for row in result} if result else {}
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting search statistics: {e}")
        return {}


def find_similar_files(
    db_manager: DatabaseManager, 
    embedder: EmbeddingGenerator, 
    reference_file: str, 
    k: int = 5
) -> List[Tuple[str, str, float]]:
    """
    Find files similar to a reference file.
    
    Args:
        db_manager: DatabaseManager instance
        embedder: EmbeddingGenerator instance
        reference_file: Path to the reference file
        k: Number of similar files to return
        
    Returns:
        List of tuples containing (file_path, content_snippet, similarity_score)
    """
    try:
        # Get the embedding of the reference file
        result = db_manager.execute_with_retry(
            f"SELECT embedding FROM {TABLE_NAME} WHERE path = ?",
            (reference_file,)
        )
        
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
                  sqrt(list_dot_product(?::DOUBLE[{DIMENSIONS}], ?::DOUBLE[{DIMENSIONS}])))) as distance
        FROM {TABLE_NAME}
        WHERE embedding IS NOT NULL AND path != ?
        ORDER BY distance ASC
        LIMIT ?
        """
        
        results = db_manager.execute_with_retry(
            sql_query, 
            (reference_embedding, reference_embedding, reference_embedding, reference_file, k)
        )
        
        # Format results
        formatted_results = []
        for path, content, distance in results:
            snippet = content[:200] + "..." if len(content) > 200 else content
            formatted_results.append((path, snippet, distance))
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error finding similar files to {reference_file}: {e}")
        return []