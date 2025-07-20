#!/usr/bin/env python3
"""
search_operations.py: Search functionality for the Turboprop code search system.

This module contains functions for searching the code index:
- Semantic search using embeddings
- Result formatting and presentation
- Search result ranking and filtering
"""

import logging
import math
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

from database_manager import DatabaseManager
from embedding_helper import EmbeddingGenerator
from search_result_types import CodeSnippet, CodeSearchResult
from snippet_extractor import SnippetExtractor
from config import config
from mcp_response_types import (
    SearchResponse, QueryAnalysis, ResultCluster,
    create_search_response_from_results
)
import response_config
from construct_search import ConstructSearchOperations, ConstructSearchResult
from result_ranking import rank_search_results, RankingWeights, generate_match_explanations

# Constants
TABLE_NAME = "code_files"
DIMENSIONS = 384

# Setup logging
logger = logging.getLogger(__name__)


def _detect_file_language(file_path: str) -> str:
    """
    Detect programming language from file extension using centralized config mapping.

    Args:
        file_path: Path to the file

    Returns:
        Programming language name or 'unknown'
    """
    ext = Path(file_path).suffix.lower()
    return config.file_processing.EXTENSION_TO_LANGUAGE_MAP.get(ext, 'unknown')


def _create_enhanced_snippet(content: str, file_path: str, query: str = "") -> CodeSnippet:
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
            max_snippet_length=config.search.SNIPPET_CONTENT_MAX_LENGTH
        )

        if extracted_snippets:
            # Convert the best ExtractedSnippet to CodeSnippet
            return extracted_snippets[0].to_code_snippet()

    except Exception as e:
        logger.warning(f"Intelligent snippet extraction failed for {file_path}: {e}")
        # Fall back to simple extraction

    # Fallback to simple truncation if intelligent extraction fails
    lines = content.split('\n')

    if len(content) <= config.search.SNIPPET_CONTENT_MAX_LENGTH:
        # Short content - include all lines
        return CodeSnippet(
            text=content,
            start_line=1,
            end_line=len(lines)
        )
    else:
        # Truncated content - use first configured characters
        snippet_text = content[:config.search.SNIPPET_CONTENT_MAX_LENGTH] + "..."

        # Calculate how many lines the snippet covers
        snippet_lines = snippet_text.count('\n') + 1

        return CodeSnippet(
            text=snippet_text,
            start_line=1,
            end_line=min(snippet_lines, len(lines))
        )


def _create_multi_snippets(
    content: str, file_path: str, query: str, max_snippets: int = 3
) -> List[CodeSnippet]:
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
            max_snippet_length=config.search.SNIPPET_CONTENT_MAX_LENGTH
        )

        # Convert ExtractedSnippet objects to CodeSnippet objects
        return [snippet.to_code_snippet() for snippet in extracted_snippets]

    except Exception as e:
        logger.warning(f"Multi-snippet extraction failed for {file_path}: {e}")
        # Fall back to single snippet
        return [_create_enhanced_snippet(content, file_path, query)]


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
        'type': 'source' if path_obj.suffix in {
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs'
        } else 'other',
        'filename': path_obj.name,
        'directory': str(path_obj.parent)
    }


def search_index_with_advanced_ranking(
    db_manager: DatabaseManager,
    embedder: EmbeddingGenerator,
    query: str,
    k: int,
    repo_path: Optional[str] = None,
    ranking_weights: Optional[RankingWeights] = None,
    enable_advanced_ranking: bool = True
) -> List[CodeSearchResult]:
    """
    Enhanced search with advanced multi-factor ranking and explainable results.

    This function provides the most sophisticated search experience by combining
    semantic similarity with advanced ranking algorithms, match explanations,
    and confidence scoring.

    Args:
        db_manager: DatabaseManager instance
        embedder: EmbeddingGenerator instance for generating query embeddings
        query: Search query string
        k: Number of top results to return
        repo_path: Optional repository path for relative path display and Git info
        ranking_weights: Optional custom ranking weights
        enable_advanced_ranking: Whether to apply advanced ranking (default: True)

    Returns:
        List of CodeSearchResult objects with advanced ranking and explanations
    """
    try:
        # Get initial search results using the existing enhanced search
        initial_results = search_index_enhanced(db_manager, embedder, query, k * 2)  # Get more candidates

        if not initial_results or not enable_advanced_ranking:
            return initial_results[:k]  # Return original results if ranking disabled

        # Apply advanced ranking
        ranked_results = rank_search_results(
            results=initial_results,
            query=query,
            repo_path=repo_path,
            git_info=None,  # TODO: Add Git integration for file recency
            ranking_weights=ranking_weights
        )

        # Generate match explanations and add to results
        explanations = generate_match_explanations(ranked_results, query)
        for result in ranked_results:
            if result.file_path in explanations:
                result.match_reasons = explanations[result.file_path]

        # Return top k results after advanced ranking
        return ranked_results[:k]

    except Exception as e:
        logger.error(
            f"Advanced ranking search failed for query '{query}' with k={k}. "
            f"Database connection: {db_manager is not None}. "
            f"Embedder: {embedder is not None}. "
            f"Error: {e}",
            exc_info=True
        )
        # Fallback to basic enhanced search
        logger.info("Falling back to basic enhanced search")
        return search_index_enhanced(db_manager, embedder, query, k)


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
            snippet = _create_enhanced_snippet(content, path, query)

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
        logger.error(
            f"Enhanced search failed for query '{query}' with k={k}. "
            f"Database connection: {db_manager is not None}. "
            f"Embedder: {embedder is not None}. "
            f"Error: {e}",
            exc_info=True
        )
        print(f"❌ Search failed for query '{query}': {e}", file=sys.stderr)
        return []


def search_index(
    db_manager: DatabaseManager, embedder: EmbeddingGenerator, query: str, k: int
) -> List[Tuple[str, str, float]]:
    """
    Search the code index for files semantically similar to the given query.

    LEGACY COMPATIBILITY VERSION: This function maintains the original tuple-based
    return format for backward compatibility. New code should use search_index_enhanced().

    Args:
        db_manager: DatabaseManager instance
        embedder: EmbeddingGenerator instance for generating query embeddings
        query: Search query string
        k: Number of top results to return

    Returns:
        List of tuples containing (file_path, content_snippet, distance_score)
    """
    try:
        # Generate embedding for the query
        query_emb = embedder.generate(query)
        query_emb_list = query_emb.tolist()

        # Construct SQL query for semantic search with cosine similarity
        sql_query = f"""
        SELECT path, content, (1 - array_dot_product(?, embedding) /
                              (list_norm(?) * list_norm(embedding))) AS distance
        FROM {TABLE_NAME}
        WHERE embedding IS NOT NULL
        ORDER BY distance ASC
        LIMIT ?
        """

        # Execute the search query - direct tuple format, no object creation
        results = db_manager.execute_with_retry(sql_query, (query_emb_list, query_emb_list, k))

        # Format results as simple tuples with basic snippets
        legacy_results = []
        for path, content, distance in results:
            # Create simple snippet (no object overhead)
            snippet = content[:config.search.SNIPPET_CONTENT_MAX_LENGTH] + \
                "..." if len(content) > config.search.SNIPPET_CONTENT_MAX_LENGTH else content
            legacy_results.append((path, snippet, distance))

        return legacy_results

    except Exception as e:
        logger.error(
            f"Legacy search failed for query '{query}' with k={k}. "
            f"Database connection: {db_manager is not None}. "
            f"Embedder: {embedder is not None}. "
            f"Error: {e}",
            exc_info=True
        )
        print(f"❌ Search failed for query '{query}': {e}", file=sys.stderr)
        return []


def search_with_comprehensive_response(
    db_manager: DatabaseManager,
    embedder: EmbeddingGenerator,
    query: str,
    k: int = 10,
    include_clusters: bool = True,
    include_suggestions: bool = True,
    include_query_analysis: bool = True,
    enable_advanced_ranking: bool = True,
    repo_path: Optional[str] = None,
    ranking_weights: Optional[RankingWeights] = None
) -> SearchResponse:
    """
    Perform enhanced search and return comprehensive structured response.

    This function combines semantic search with advanced ranking, rich metadata, clustering,
    and suggestions to provide Claude with comprehensive search information.

    Args:
        db_manager: DatabaseManager instance
        embedder: EmbeddingGenerator instance
        query: Search query string
        k: Number of results to return
        include_clusters: Whether to include result clustering
        include_suggestions: Whether to include query suggestions
        include_query_analysis: Whether to analyze the query
        enable_advanced_ranking: Whether to use advanced multi-factor ranking
        repo_path: Optional repository path for enhanced ranking context
        ranking_weights: Optional custom ranking weights

    Returns:
        SearchResponse with comprehensive metadata and suggestions
    """
    start_time = time.time()

    try:
        # Perform the search with optional advanced ranking
        if enable_advanced_ranking:
            results = search_index_with_advanced_ranking(
                db_manager, embedder, query, k, repo_path, ranking_weights
            )
        else:
            results = search_index_enhanced(db_manager, embedder, query, k)
        execution_time = time.time() - start_time

        # Create the base response
        response = create_search_response_from_results(
            query=query,
            results=results,
            execution_time=execution_time,
            add_clusters=include_clusters,
            add_suggestions=include_suggestions
        )

        # Add query analysis if requested
        if include_query_analysis:
            response.query_analysis = _analyze_search_query(query, results)

        # Add performance notes
        if execution_time > 1.0:
            response.performance_notes.append(
                f"Search took {execution_time:.2f}s - consider indexing optimization"
            )
        elif execution_time < 0.1:
            response.performance_notes.append(
                f"Fast search in {execution_time:.3f}s - index is well optimized"
            )

        return response

    except Exception as e:
        logger.error(f"Comprehensive search failed: {e}", exc_info=True)

        # Return error response
        return SearchResponse(
            query=query,
            results=[],
            total_results=0,
            execution_time=time.time() - start_time,
            performance_notes=[f"Search failed: {str(e)}"]
        )


def _analyze_search_query(query: str, results: List[CodeSearchResult]) -> QueryAnalysis:
    """
    Analyze a search query and provide insights and suggestions.

    Args:
        query: The search query string
        results: The search results obtained

    Returns:
        QueryAnalysis with insights and suggestions
    """
    analysis = QueryAnalysis(original_query=query)

    # Analyze query complexity
    query_words = query.lower().split()
    if len(query_words) == 1:
        analysis.query_complexity = "low"
        analysis.search_hints.append("Single-word queries may be too broad - try adding context")
    elif len(query_words) <= 3:
        analysis.query_complexity = "medium"
    else:
        analysis.query_complexity = "high"
        analysis.search_hints.append("Complex queries are good for specific searches")

    # Estimate result quality and suggest refinements
    if not results:
        analysis.suggested_refinements.extend([
            f"Try broader terms like: '{_suggest_broader_terms(query)}'",
            "Check spelling and try synonyms",
            "Consider programming language context"
        ])
        analysis.search_hints.append("No results found - try different search terms")
    elif len(results) < 3:
        analysis.suggested_refinements.extend([
            f"Related terms: '{_suggest_related_terms(query)}'",
            "Try including file type or framework context"
        ])
        analysis.search_hints.append("Few results - consider broader or related terms")
    elif len(results) >= 8:
        analysis.suggested_refinements.extend([
            f"More specific: '{query} function'",
            f"With context: '{query} implementation'",
            "Add language context like 'python' or 'javascript'"
        ])
        analysis.search_hints.append("Many results - add more specific terms to narrow down")

    # Analyze result confidence
    high_confidence = sum(1 for r in results if r.confidence_level == "high")
    if high_confidence == 0:
        analysis.search_hints.append("Low confidence results - try more specific terms")
    elif high_confidence == len(results):
        analysis.search_hints.append("High confidence results - query matches well")

    # Add technical hints based on query content
    if any(term in query.lower() for term in ['function', 'method', 'def']):
        analysis.search_hints.append("Searching for functions - results show function definitions")
    elif any(term in query.lower() for term in ['class', 'interface', 'struct']):
        analysis.search_hints.append("Searching for types - results show class/interface definitions")
    elif any(term in query.lower() for term in ['error', 'exception', 'handling']):
        analysis.search_hints.append("Searching for error handling - check try-catch blocks")

    analysis.estimated_result_count = len(results)
    return analysis


def _suggest_broader_terms(query: str) -> str:
    """Generate broader search terms from a specific query."""
    # Simple word replacement for common technical terms
    broader_terms = {
        'authentication': 'auth login user',
        'database': 'db data storage',
        'configuration': 'config settings',
        'implementation': 'code function',
        'optimization': 'performance speed',
        'validation': 'validate check',
        'serialization': 'serialize json',
        'initialization': 'init setup start'
    }

    query_lower = query.lower()
    for specific, broader in broader_terms.items():
        if specific in query_lower:
            return broader

    # If no specific replacement, just use the original
    return query


def _suggest_related_terms(query: str) -> str:
    """Generate related search terms from a query."""
    # Simple related term suggestions
    related_terms = {
        'login': 'authentication signin user',
        'auth': 'login token session',
        'database': 'sql query connection',
        'config': 'settings environment variables',
        'error': 'exception handling try catch',
        'test': 'unittest pytest testing',
        'api': 'endpoint route http',
        'json': 'parse serialize data',
        'file': 'read write stream',
        'cache': 'memory storage redis'
    }

    query_lower = query.lower()
    for term, related in related_terms.items():
        if term in query_lower:
            return related

    return f"{query} related"


def cluster_results_by_language(results: List[CodeSearchResult]) -> List[ResultCluster]:
    """
    Cluster search results by programming language.

    Args:
        results: List of search results to cluster

    Returns:
        List of ResultCluster objects grouped by language
    """
    language_groups = {}

    for result in results:
        lang = 'unknown'
        if result.file_metadata and 'language' in result.file_metadata:
            lang = result.file_metadata['language']

        if lang not in language_groups:
            language_groups[lang] = []
        language_groups[lang].append(result)

    clusters = []
    for language, lang_results in language_groups.items():
        if len(lang_results) > 1:  # Only cluster multiple results
            avg_score = sum(r.similarity_score for r in lang_results) / len(lang_results)

            cluster = ResultCluster(
                cluster_name=f"{language} Files" if language != 'unknown' else 'Other Files',
                cluster_type="language",
                results=lang_results,
                cluster_score=avg_score,
                cluster_description=f"Search results from {language} source files"
            )
            clusters.append(cluster)

    # Sort clusters by score
    clusters.sort(key=lambda c: c.cluster_score, reverse=True)
    return clusters


def cluster_results_by_directory(results: List[CodeSearchResult]) -> List[ResultCluster]:
    """
    Cluster search results by directory structure.

    Args:
        results: List of search results to cluster

    Returns:
        List of ResultCluster objects grouped by directory
    """
    directory_groups = {}

    for result in results:
        directory = str(Path(result.file_path).parent)

        if directory not in directory_groups:
            directory_groups[directory] = []
        directory_groups[directory].append(result)

    clusters = []
    for directory, dir_results in directory_groups.items():
        if len(dir_results) > 1:  # Only cluster multiple results
            avg_score = sum(r.similarity_score for r in dir_results) / len(dir_results)
            dir_name = Path(directory).name or "root"

            cluster = ResultCluster(
                cluster_name=f"{dir_name}/ directory",
                cluster_type="directory",
                results=dir_results,
                cluster_score=avg_score,
                cluster_description=f"Search results from {directory}"
            )
            clusters.append(cluster)

    # Sort clusters by score
    clusters.sort(key=lambda c: c.cluster_score, reverse=True)
    return clusters


def cluster_results_by_confidence(results: List[CodeSearchResult]) -> List[ResultCluster]:
    """
    Cluster search results by confidence level.

    Args:
        results: List of search results to cluster

    Returns:
        List of ResultCluster objects grouped by confidence
    """
    confidence_groups = {'high': [], 'medium': [], 'low': []}

    for result in results:
        confidence = result.confidence_level or 'low'
        confidence_groups[confidence].append(result)

    clusters = []
    for confidence_level, conf_results in confidence_groups.items():
        if conf_results:  # Include all confidence groups that have results
            avg_score = sum(r.similarity_score for r in conf_results) / len(conf_results)

            cluster = ResultCluster(
                cluster_name=f"{confidence_level.title()} Confidence",
                cluster_type="confidence",
                results=conf_results,
                cluster_score=avg_score,
                cluster_description=f"Results with {confidence_level} confidence level"
            )
            clusters.append(cluster)

    # Sort by confidence level (high -> medium -> low)
    confidence_order = {'high': 3, 'medium': 2, 'low': 1}
    clusters.sort(key=lambda c: confidence_order.get(
        c.cluster_name.split()[0].lower(), 0
    ), reverse=True)

    return clusters


def generate_cross_references(results: List[CodeSearchResult]) -> List[str]:
    """
    Generate cross-references between related code constructs.

    Args:
        results: List of search results to analyze

    Returns:
        List of cross-reference descriptions
    """
    cross_refs = []

    # Group results by directory to find related files
    directory_groups = {}
    for result in results:
        directory = str(Path(result.file_path).parent)
        if directory not in directory_groups:
            directory_groups[directory] = []
        directory_groups[directory].append(result)

    # Find directories with multiple matches
    for directory, dir_results in directory_groups.items():
        if len(dir_results) > 1:
            filenames = [Path(r.file_path).name for r in dir_results]
            max_files = response_config.MAX_FILENAMES_IN_CROSS_REF
            files_list = ", ".join(filenames[:max_files])
            ellipsis = "..." if len(filenames) > max_files else ""
            cross_refs.append(
                f"Related files in {Path(directory).name}/: {files_list}{ellipsis}"
            )

    # Find files with similar names
    file_stems = {}
    for result in results:
        stem = Path(result.file_path).stem.lower()
        if stem not in file_stems:
            file_stems[stem] = []
        file_stems[stem].append(result)

    for stem, stem_results in file_stems.items():
        if len(stem_results) > 1:
            extensions = [Path(r.file_path).suffix for r in stem_results]
            cross_refs.append(
                f"Related '{stem}' files: {', '.join(extensions)}"
            )

    return cross_refs[:response_config.MAX_CROSS_REFERENCES]  # Limit to top cross-references


def format_advanced_search_results(
    results: List[CodeSearchResult], query: str, repo_path: Optional[str] = None
) -> str:
    """
    Format advanced search results with match explanations and ranking information.

    Args:
        results: List of CodeSearchResult objects with advanced ranking
        query: Original search query
        repo_path: Optional repository path for relative path display

    Returns:
        Formatted string representation of advanced search results
    """
    if not results:
        return f"No results found for query: '{query}'"

    formatted_lines = []
    formatted_lines.append(f"🔍 Found {len(results)} results for: '{query}' (with advanced ranking)")
    formatted_lines.append("=" * 60)

    for i, result in enumerate(results, 1):
        # Format path (use relative path if repo_path provided)
        display_path = result.get_relative_path(repo_path) if repo_path else result.file_path

        # Display enhanced information with ranking score
        ranking_score = result.ranking_score or result.similarity_score
        formatted_lines.append(f"{i}. {display_path}")
        formatted_lines.append(
            f"   📊 Ranking: {ranking_score:.3f} | Similarity: {result.similarity_percentage:.1f}% "
            f"({result.confidence_level} confidence)")

        # Show ranking factors if available
        if result.ranking_factors:
            factors = result.ranking_factors
            formatted_lines.append(
                f"   🎯 Factors: similarity={factors.get('embedding_similarity', 0):.2f}, "
                f"file_type={factors.get('file_type_relevance', 0):.2f}, "
                f"construct={factors.get('construct_type_matching', 0):.2f}"
            )

        # Show file metadata if available
        if result.file_metadata:
            metadata = result.file_metadata
            lang = metadata.get('language', 'unknown')
            size = metadata.get('size', 0)
            if size > 0:
                size_str = f"{size} bytes" if size < 1024 else f"{size/1024:.1f}KB"
                formatted_lines.append(f"   📄 Type: {lang} ({size_str})")
            else:
                formatted_lines.append(f"   📄 Type: {lang}")

        # Show primary snippet with line information
        snippet = result.snippet
        if snippet.start_line == snippet.end_line:
            line_info = f"Line {snippet.start_line}"
        else:
            line_info = f"Lines {snippet.start_line}-{snippet.end_line}"

        formatted_lines.append(f"   💻 {line_info}: {snippet.text.strip()}")

        # Show match reasons if available
        if result.match_reasons:
            formatted_lines.append("   🎯 Why this matches:")
            for reason in result.match_reasons[:3]:  # Show top 3 reasons
                formatted_lines.append(f"      • {reason}")

        # Show additional snippets if present
        if result.additional_snippets:
            formatted_lines.append(f"   ➕ Additional snippets ({len(result.additional_snippets)}):")
            for i, additional_snippet in enumerate(result.additional_snippets[:2], 1):  # Show first 2
                if additional_snippet.start_line == additional_snippet.end_line:
                    add_line_info = f"Line {additional_snippet.start_line}"
                else:
                    add_line_info = (
                        f"Lines {additional_snippet.start_line}-{additional_snippet.end_line}"
                    )

                # Truncate additional snippets for display
                add_text = additional_snippet.text.strip()
                if len(add_text) > 100:
                    add_text = add_text[:100] + "..."

                formatted_lines.append(f"     {i}. {add_line_info}: {add_text}")

        formatted_lines.append("")

    return "\n".join(formatted_lines)


def format_enhanced_search_results(
    results: List[CodeSearchResult], query: str, repo_path: Optional[str] = None
) -> str:
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
        formatted_lines.append(
            f"   Similarity: {result.similarity_percentage:.1f}% "
            f"({result.confidence_level} confidence)")

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

        # Show primary snippet with line information
        snippet = result.snippet
        if snippet.start_line == snippet.end_line:
            line_info = f"Line {snippet.start_line}"
        else:
            line_info = f"Lines {snippet.start_line}-{snippet.end_line}"

        formatted_lines.append(f"   {line_info}: {snippet.text.strip()}")

        # Show additional snippets if present
        if result.additional_snippets:
            formatted_lines.append(f"   Additional snippets ({len(result.additional_snippets)}):")
            for i, additional_snippet in enumerate(result.additional_snippets, 1):
                if additional_snippet.start_line == additional_snippet.end_line:
                    add_line_info = f"Line {additional_snippet.start_line}"
                else:
                    add_line_info = (
                        f"Lines {additional_snippet.start_line}-{additional_snippet.end_line}"
                    )

                # Truncate additional snippets for display
                add_text = additional_snippet.text.strip()
                if len(add_text) > 100:
                    add_text = add_text[:100] + "..."

                formatted_lines.append(f"     {i}. {add_line_info}: {add_text}")

        formatted_lines.append("")

    return "\n".join(formatted_lines)


def format_search_results(
    results: List[Tuple[str, str, float]], query: str, repo_path: Optional[str] = None
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
        result = db_manager.execute_with_retry(
            f"SELECT content FROM {TABLE_NAME} WHERE path = ?", (file_path,))
        return result[0][0] if result else None
    except Exception as e:
        logger.error(f"Error getting file content for {file_path}: {e}")
        return None


def search_by_file_extension(
    db_manager: DatabaseManager, extension: str, limit: int = 10
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
        if not extension.startswith("."):
            extension = "." + extension

        sql_query = f"""
        SELECT path, content
        FROM {TABLE_NAME}
        WHERE path LIKE ?
        ORDER BY path
        LIMIT ?
        """

        # Use parameterized query for the extension pattern
        extension_pattern = f"%{extension}"
        results = db_manager.execute_with_retry(sql_query, (extension_pattern, limit))

        # Format results with snippets
        formatted_results = []
        for path, content in results:
            snippet = content[:config.search.SNIPPET_CONTENT_MAX_LENGTH] + \
                "..." if len(content) > config.search.SNIPPET_CONTENT_MAX_LENGTH else content
            formatted_results.append((path, snippet))

        return formatted_results

    except Exception as e:
        logger.error(f"Error searching by extension {extension}: {e}")
        return []


def search_by_filename(
    db_manager: DatabaseManager, filename_pattern: str, limit: int = 10
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
            snippet = content[:config.search.SNIPPET_CONTENT_MAX_LENGTH] + \
                "..." if len(content) > config.search.SNIPPET_CONTENT_MAX_LENGTH else content
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
        result = db_manager.execute_with_retry(
            f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE embedding IS NOT NULL")
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
        result = db_manager.execute_with_retry(
            f"SELECT embedding FROM {TABLE_NAME} WHERE path = ?", (reference_file,))

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

            # Create enhanced snippet with query context
            snippet = _create_enhanced_snippet(content, path, "")

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


# Hybrid Search Functions - Combining File-level and Construct-level Search

def search_hybrid(
    db_manager: DatabaseManager,
    embedder: EmbeddingGenerator,
    query: str,
    k: int = 10,
    construct_weight: float = 0.7,
    file_weight: float = 0.3,
    construct_types: Optional[List[str]] = None
) -> List[CodeSearchResult]:
    """
    Perform hybrid search combining file-level and construct-level results.

    This function searches both files and constructs, then intelligently merges
    and ranks the results based on configurable weights. Construct matches are
    typically weighted higher for precision.

    Args:
        db_manager: DatabaseManager instance
        embedder: EmbeddingGenerator instance
        query: Natural language search query
        k: Maximum number of results to return
        construct_weight: Weight for construct-level matches (0.0 to 1.0)
        file_weight: Weight for file-level matches (0.0 to 1.0)
        construct_types: Optional filter for construct types

    Returns:
        List of CodeSearchResult objects with merged and ranked results
    """
    try:
        # Initialize construct search operations
        construct_ops = ConstructSearchOperations(db_manager, embedder)

        # Search constructs with higher k to get more candidates
        construct_k = min(k * 2, 20)  # Search more constructs for better selection
        construct_results = construct_ops.search_constructs(
            query=query,
            k=construct_k,
            construct_types=construct_types
        )

        # Search files
        file_results = search_index_enhanced(db_manager, embedder, query, k)

        # Convert construct results to CodeSearchResult for merging
        construct_code_results = []
        for construct_result in construct_results:
            code_result = construct_result.to_code_search_result()
            # Apply construct weight to similarity score
            code_result.similarity_score *= construct_weight
            construct_code_results.append(code_result)

        # Apply file weight to file results
        for file_result in file_results:
            file_result.similarity_score *= file_weight

        # Merge results and deduplicate by file path
        merged_results = []
        seen_files = set()

        # Prioritize construct results (they're usually more precise)
        for result in construct_code_results:
            file_key = result.file_path
            if file_key not in seen_files:
                merged_results.append(result)
                seen_files.add(file_key)
            elif construct_weight > file_weight:
                # Replace file result with construct result if construct weight is higher
                for i, existing_result in enumerate(merged_results):
                    if existing_result.file_path == file_key:
                        # Keep the higher scoring result
                        if result.similarity_score > existing_result.similarity_score:
                            merged_results[i] = result
                        break

        # Add unique file results
        for result in file_results:
            file_key = result.file_path
            if file_key not in seen_files:
                merged_results.append(result)
                seen_files.add(file_key)

        # Sort by weighted similarity score and limit results
        merged_results.sort(key=lambda r: r.similarity_score, reverse=True)
        final_results = merged_results[:k]

        # Enhance results with construct context where applicable
        enhanced_results = []
        for result in final_results:
            enhanced_result = _add_construct_context(result, construct_results)
            enhanced_results.append(enhanced_result)

        logger.info(f"Hybrid search found {len(enhanced_results)} merged results for query: {query}")
        return enhanced_results

    except Exception as e:
        logger.error(f"Error in hybrid search for query '{query}': {e}")
        return []


def search_with_construct_focus(
    db_manager: DatabaseManager,
    embedder: EmbeddingGenerator,
    query: str,
    k: int = 10,
    construct_types: Optional[List[str]] = None
) -> List[CodeSearchResult]:
    """
    Search with primary focus on constructs, falling back to file search if needed.

    This function prioritizes construct-level results but includes file-level
    results to ensure comprehensive coverage.

    Args:
        db_manager: DatabaseManager instance
        embedder: EmbeddingGenerator instance
        query: Natural language search query
        k: Maximum number of results to return
        construct_types: Optional filter for construct types

    Returns:
        List of CodeSearchResult objects prioritizing construct matches
    """
    return search_hybrid(
        db_manager=db_manager,
        embedder=embedder,
        query=query,
        k=k,
        construct_weight=0.8,
        file_weight=0.2,
        construct_types=construct_types
    )


def _add_construct_context(
    result: CodeSearchResult,
    construct_results: List[ConstructSearchResult]
) -> CodeSearchResult:
    """
    Add construct context to a CodeSearchResult when applicable.

    Args:
        result: CodeSearchResult to enhance
        construct_results: List of construct results for context

    Returns:
        Enhanced CodeSearchResult with construct context
    """
    # Find constructs from the same file
    file_constructs = [
        c for c in construct_results
        if c.file_path == result.file_path
    ]

    if not file_constructs:
        return result

    # Add construct information to file metadata
    if not result.file_metadata:
        result.file_metadata = {}

    # Add construct summary to metadata
    construct_summary = {
        'related_constructs': len(file_constructs),
        'construct_types': list(set(c.construct_type for c in file_constructs)),
        'top_constructs': [
            {
                'name': c.name,
                'type': c.construct_type,
                'line': c.start_line,
                'signature': c.signature[:100] + "..." if len(c.signature) > 100 else c.signature
            }
            for c in sorted(file_constructs, key=lambda x: x.similarity_score, reverse=True)[:3]
        ]
    }

    result.file_metadata['construct_context'] = construct_summary
    return result


def format_hybrid_search_results(
    results: List[CodeSearchResult],
    query: str,
    show_construct_context: bool = True
) -> str:
    """
    Format hybrid search results with construct context.

    Args:
        results: List of CodeSearchResult objects
        query: Original search query
        show_construct_context: Whether to show construct context information

    Returns:
        Formatted string representation of hybrid search results
    """
    if not results:
        return f"No hybrid search results found for query: '{query}'"

    formatted_lines = [f"🔍 Found {len(results)} hybrid results for: '{query}'\n"]

    for i, result in enumerate(results, 1):
        # Result header with confidence and type information
        confidence_emoji = {
            'high': '🎯',
            'medium': '✅',
            'low': '⚠️'
        }.get(result.confidence_level, '❓')

        formatted_lines.append(
            f"{confidence_emoji} [{i}] {result.file_path} "
            f"(similarity: {result.similarity_score:.3f})"
        )

        # Show main snippet
        snippet = result.snippet
        if snippet.start_line == snippet.end_line:
            line_info = f"Line {snippet.start_line}"
        else:
            line_info = f"Lines {snippet.start_line}-{snippet.end_line}"

        formatted_lines.append(f"   📄 {line_info}: {snippet.text.strip()[:150]}...")

        # Show construct context if available and requested
        if (show_construct_context and result.file_metadata and
            'construct_context' in result.file_metadata):

            context = result.file_metadata['construct_context']
            construct_count = context['related_constructs']
            construct_types = ', '.join(context['construct_types'])

            formatted_lines.append(f"   🔧 {construct_count} constructs: {construct_types}")

            # Show top constructs
            for construct in context['top_constructs']:
                formatted_lines.append(
                    f"      • {construct['type']}: {construct['name']} "
                    f"(line {construct['line']})"
                )

        formatted_lines.append("")  # Empty line between results

    return "\n".join(formatted_lines)


def search_constructs_with_file_context(
    db_manager: DatabaseManager,
    embedder: EmbeddingGenerator,
    query: str,
    k: int = 10,
    construct_types: Optional[List[str]] = None
) -> Tuple[List[ConstructSearchResult], List[CodeSearchResult]]:
    """
    Search constructs and provide related file context.

    This function performs construct search and then finds the containing
    files to provide full context for each match.

    Args:
        db_manager: DatabaseManager instance
        embedder: EmbeddingGenerator instance
        query: Natural language search query
        k: Maximum number of construct results to return
        construct_types: Optional filter for construct types

    Returns:
        Tuple of (construct_results, file_context_results)
    """
    try:
        # Initialize construct search operations
        construct_ops = ConstructSearchOperations(db_manager, embedder)

        # Search constructs
        construct_results = construct_ops.search_constructs(
            query=query,
            k=k,
            construct_types=construct_types
        )

        if not construct_results:
            return [], []

        # Get unique file paths from construct results
        unique_file_paths = list(set(c.file_path for c in construct_results))

        # Get file context for each unique file
        file_context_results = []
        for file_path in unique_file_paths:
            try:
                # Get file content and create CodeSearchResult
                content = get_file_content(db_manager, file_path)
                if content:
                    # Create enhanced snippet focusing on the file's most relevant constructs
                    file_constructs = [c for c in construct_results if c.file_path == file_path]
                    best_construct = max(file_constructs, key=lambda c: c.similarity_score)

                    snippet = CodeSnippet(
                        text=content[best_construct.start_line:best_construct.end_line],
                        start_line=best_construct.start_line,
                        end_line=best_construct.end_line
                    )

                    file_metadata = {
                        'language': _detect_file_language(file_path),
                        'filename': Path(file_path).name,
                        'directory': str(Path(file_path).parent),
                        'construct_matches': len(file_constructs)
                    }

                    file_result = CodeSearchResult(
                        file_path=file_path,
                        snippet=snippet,
                        similarity_score=best_construct.similarity_score,
                        file_metadata=file_metadata,
                        confidence_level=best_construct.confidence_level
                    )

                    file_context_results.append(file_result)

            except Exception as e:
                logger.warning(f"Error getting file context for {file_path}: {e}")
                continue

        logger.info(f"Found {len(construct_results)} construct matches with {len(file_context_results)} file contexts")
        return construct_results, file_context_results

    except Exception as e:
        logger.error(f"Error in construct search with file context for query '{query}': {e}")
        return [], []
