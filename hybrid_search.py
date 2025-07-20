#!/usr/bin/env python3
"""
hybrid_search.py: Hybrid search implementation combining semantic and exact text matching.

This module provides:
- Reciprocal Rank Fusion (RRF) for combining semantic and text results
- Weighted score combination with configurable weights
- Adaptive weighting based on query characteristics
- Search mode selection (semantic-only, text-only, hybrid)
- Query preprocessing and routing
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

from database_manager import DatabaseManager
from embedding_helper import EmbeddingGenerator
from search_result_types import CodeSearchResult
from search_utils import (
    create_enhanced_snippet,
    extract_file_metadata,
    search_index_enhanced
)

logger = logging.getLogger(__name__)


class SearchMode(Enum):
    """Search mode options for hybrid search."""
    SEMANTIC_ONLY = "semantic"
    TEXT_ONLY = "text"
    HYBRID = "hybrid"
    AUTO = "auto"  # Automatically determine best mode based on query


@dataclass
class FusionWeights:
    """Configuration for fusion algorithm weights."""
    semantic_weight: float = 0.6
    text_weight: float = 0.4
    rrf_k: int = 60  # RRF parameter
    boost_exact_matches: bool = True
    exact_match_boost: float = 1.5


@dataclass
class QueryCharacteristics:
    """Analysis of query characteristics to guide search strategy."""
    has_quoted_phrases: bool = False
    has_boolean_operators: bool = False
    has_regex_patterns: bool = False
    has_wildcards: bool = False
    is_technical_term: bool = False
    is_natural_language: bool = False
    word_count: int = 0
    estimated_intent: str = "general"  # general, exact, semantic, technical


@dataclass
class HybridSearchResult:
    """Enhanced result container for hybrid search."""
    code_result: CodeSearchResult
    semantic_score: float = 0.0
    text_score: float = 0.0
    fusion_score: float = 0.0
    semantic_rank: int = -1
    text_rank: int = -1
    fusion_method: str = "unknown"
    match_type: str = "hybrid"  # semantic, text, hybrid


class QueryAnalyzer:
    """Analyzes search queries to determine optimal search strategy."""

    # Technical terms that suggest exact matching might be preferred
    TECHNICAL_INDICATORS = {
        'function', 'class', 'method', 'variable', 'import', 'export',
        'const', 'let', 'var', 'def', 'async', 'await', 'return',
        'if', 'else', 'for', 'while', 'try', 'catch', 'throw',
        'interface', 'type', 'struct', 'enum', 'union'
    }

    # Natural language indicators that suggest semantic search
    SEMANTIC_INDICATORS = {
        'how', 'what', 'why', 'where', 'when', 'which', 'who',
        'find', 'search', 'look', 'get', 'create', 'build', 'make',
        'implement', 'handle', 'manage', 'process', 'parse', 'convert'
    }

    def analyze_query(self, query: str) -> QueryCharacteristics:
        """
        Analyze a search query to determine its characteristics.

        Args:
            query: Search query string

        Returns:
            QueryCharacteristics with analysis results
        """
        query_lower = query.lower().strip()

        # Handle quoted phrases properly for word counting
        # Replace quoted phrases with placeholders to count them as single units
        import re
        quoted_phrases = re.findall(r'"[^"]*"', query_lower)
        temp_query = query_lower

        # Replace each quoted phrase with a placeholder
        for i, phrase in enumerate(quoted_phrases):
            temp_query = temp_query.replace(phrase, f'__QUOTED_PHRASE_{i}__', 1)

        # Now split and count words/units
        words = temp_query.split()

        characteristics = QueryCharacteristics(
            word_count=len(words)
        )

        # Check for quoted phrases
        characteristics.has_quoted_phrases = bool(re.search(r'"[^"]*"', query))

        # Check for Boolean operators
        boolean_ops = ['AND', 'OR', 'NOT', '&', '|', '-']
        characteristics.has_boolean_operators = any(
            op in query.upper() for op in boolean_ops
        )

        # Check for regex patterns (basic detection)
        regex_indicators = [r'\[', r'\]', r'\{', r'\}', r'\^', r'\$', r'\\']
        characteristics.has_regex_patterns = any(
            indicator in query for indicator in regex_indicators
        )

        # Check for wildcards
        characteristics.has_wildcards = '*' in query or '?' in query

        # Check for technical terms
        technical_matches = sum(1 for word in words if word in self.TECHNICAL_INDICATORS)
        characteristics.is_technical_term = technical_matches >= 1

        # Check for natural language patterns
        semantic_matches = sum(1 for word in words if word in self.SEMANTIC_INDICATORS)
        characteristics.is_natural_language = semantic_matches >= 1

        # Estimate intent
        if characteristics.has_quoted_phrases or characteristics.has_boolean_operators:
            characteristics.estimated_intent = "exact"
        elif characteristics.has_regex_patterns:
            characteristics.estimated_intent = "technical"
        elif characteristics.is_technical_term and len(words) <= 3:
            characteristics.estimated_intent = "technical"
        elif characteristics.is_natural_language or len(words) >= 4:
            characteristics.estimated_intent = "semantic"
        else:
            characteristics.estimated_intent = "general"

        return characteristics


class HybridSearchEngine:
    """Main hybrid search engine combining semantic and text search."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        embedder: EmbeddingGenerator,
        default_weights: Optional[FusionWeights] = None
    ):
        """
        Initialize the hybrid search engine.

        Args:
            db_manager: Database manager instance
            embedder: Embedding generator for semantic search
            default_weights: Default fusion weights configuration
        """
        self.db_manager = db_manager
        self.embedder = embedder
        self.query_analyzer = QueryAnalyzer()
        self.default_weights = default_weights or FusionWeights()

        # Ensure FTS index exists
        try:
            self.db_manager.create_fts_index()
        except Exception as e:
            logger.warning("Failed to create FTS index: %s", e)

    def search(
        self,
        query: str,
        k: int = 10,
        mode: SearchMode = SearchMode.AUTO,
        fusion_weights: Optional[FusionWeights] = None,
        enable_query_expansion: bool = True
    ) -> List[HybridSearchResult]:
        """
        Perform hybrid search with automatic mode selection and result fusion.

        Args:
            query: Search query string
            k: Number of results to return
            mode: Search mode (auto, hybrid, semantic, text)
            fusion_weights: Custom fusion weights
            enable_query_expansion: Whether to expand semantic queries

        Returns:
            List of HybridSearchResult objects
        """
        start_time = time.time()

        try:
            # Analyze query characteristics
            query_chars = self.query_analyzer.analyze_query(query)
            logger.debug("Query analysis for '%s': %s", query, query_chars)

            # Determine search mode
            if mode == SearchMode.AUTO:
                mode = self._determine_search_mode(query_chars)
                logger.debug("Auto-determined search mode: %s", mode.value)

            # Get fusion weights
            weights = fusion_weights or self._adapt_weights(query_chars)

            # Execute search based on mode
            if mode == SearchMode.SEMANTIC_ONLY:
                results = self._search_semantic_only(query, k, enable_query_expansion)
            elif mode == SearchMode.TEXT_ONLY:
                results = self._search_text_only(query, k)
            else:  # HYBRID mode
                results = self._search_hybrid(query, k, weights, enable_query_expansion)

            execution_time = time.time() - start_time
            logger.info(
                "Hybrid search for '%s' returned %d results in %.3fs (mode: %s)",
                query, len(results), execution_time, mode.value
            )

            return results

        except Exception as e:
            logger.error("Hybrid search failed for query '%s': %s", query, e)
            return []

    def _determine_search_mode(self, characteristics: QueryCharacteristics) -> SearchMode:
        """
        Automatically determine the best search mode based on query characteristics.

        Args:
            characteristics: Analyzed query characteristics

        Returns:
            Recommended search mode
        """
        # Prefer exact/text search for specific patterns
        if (characteristics.has_quoted_phrases or
            characteristics.has_boolean_operators or
            characteristics.has_regex_patterns):
            return SearchMode.TEXT_ONLY

        # Prefer semantic search for natural language queries
        if (characteristics.is_natural_language and
            characteristics.word_count >= 3 and
            not characteristics.is_technical_term):
            return SearchMode.SEMANTIC_ONLY

        # Prefer text search for short technical queries
        if (characteristics.is_technical_term and
            characteristics.word_count <= 2):
            return SearchMode.TEXT_ONLY

        # Default to hybrid for balanced results
        return SearchMode.HYBRID

    def _adapt_weights(self, characteristics: QueryCharacteristics) -> FusionWeights:
        """
        Adapt fusion weights based on query characteristics.

        Args:
            characteristics: Analyzed query characteristics

        Returns:
            Adapted fusion weights
        """
        weights = FusionWeights(
            semantic_weight=self.default_weights.semantic_weight,
            text_weight=self.default_weights.text_weight,
            rrf_k=self.default_weights.rrf_k,
            boost_exact_matches=self.default_weights.boost_exact_matches,
            exact_match_boost=self.default_weights.exact_match_boost
        )

        # Boost text search for exact queries
        if characteristics.has_quoted_phrases or characteristics.has_boolean_operators:
            weights.text_weight = 0.8
            weights.semantic_weight = 0.2
            weights.exact_match_boost = 2.0

        # Boost semantic search for natural language
        elif characteristics.is_natural_language and characteristics.word_count >= 4:
            weights.semantic_weight = 0.8
            weights.text_weight = 0.2

        # Balanced weights for technical terms
        elif characteristics.is_technical_term:
            weights.semantic_weight = 0.5
            weights.text_weight = 0.5
            weights.exact_match_boost = 1.8

        return weights

    def _search_semantic_only(
        self,
        query: str,
        k: int,
        enable_expansion: bool
    ) -> List[HybridSearchResult]:
        """Execute semantic-only search."""
        try:
            # Expand query if enabled
            expanded_query = query
            if enable_expansion:
                expanded_query = self._expand_query(query)
                logger.debug("Expanded query from '%s' to '%s'", query, expanded_query)

            # Perform semantic search
            semantic_results = search_index_enhanced(
                self.db_manager, self.embedder, expanded_query, k
            )

            # Convert to HybridSearchResult
            hybrid_results = []
            for i, result in enumerate(semantic_results):
                hybrid_result = HybridSearchResult(
                    code_result=result,
                    semantic_score=result.similarity_score,
                    text_score=0.0,
                    fusion_score=result.similarity_score,
                    semantic_rank=i + 1,
                    text_rank=-1,
                    fusion_method="semantic_only",
                    match_type="semantic"
                )
                hybrid_results.append(hybrid_result)

            return hybrid_results

        except Exception as e:
            logger.error("Semantic-only search failed: %s", e)
            return []

    def _convert_to_code_search_result(self, path: str, content: str, score: float, query: str) -> CodeSearchResult:
        """Convert raw search result to CodeSearchResult object."""
        snippet = create_enhanced_snippet(content, path, query)
        file_metadata = extract_file_metadata(path, content)

        return CodeSearchResult(
            file_path=path,
            snippet=snippet,
            similarity_score=score,
            file_metadata=file_metadata
        )

    def _search_text_only(self, query: str, k: int) -> List[HybridSearchResult]:
        """Execute text-only search using full-text search."""
        try:
            # Perform full-text search
            text_results = self.db_manager.search_full_text(
                query, limit=k, enable_fuzzy=True
            )

            # Convert to HybridSearchResult
            hybrid_results = []
            for i, (file_id, path, content, relevance_score) in enumerate(text_results):
                # Create CodeSearchResult using helper method
                code_result = self._convert_to_code_search_result(path, content, relevance_score, query)

                hybrid_result = HybridSearchResult(
                    code_result=code_result,
                    semantic_score=0.0,
                    text_score=relevance_score,
                    fusion_score=relevance_score,
                    semantic_rank=-1,
                    text_rank=i + 1,
                    fusion_method="text_only",
                    match_type="text"
                )
                hybrid_results.append(hybrid_result)

            return hybrid_results

        except Exception as e:
            logger.error("Text-only search failed: %s", e)
            return []

    def _search_hybrid(
        self,
        query: str,
        k: int,
        weights: FusionWeights,
        enable_expansion: bool
    ) -> List[HybridSearchResult]:
        """Execute hybrid search with result fusion."""
        try:
            # Get more candidates for better fusion
            candidate_k = min(k * 3, 50)

            # Perform semantic search
            expanded_query = query
            if enable_expansion:
                expanded_query = self._expand_query(query)

            semantic_results = search_index_enhanced(
                self.db_manager, self.embedder, expanded_query, candidate_k
            )

            # Perform text search
            text_results = self.db_manager.search_full_text(
                query, limit=candidate_k, enable_fuzzy=True
            )

            # Convert text results to CodeSearchResult format
            text_code_results = []
            for file_id, path, content, relevance_score in text_results:
                code_result = self._convert_to_code_search_result(path, content, relevance_score, query)
                text_code_results.append(code_result)

            # Fuse results using RRF and weighted scoring
            fused_results = self._fuse_results(
                semantic_results, text_code_results, weights, query
            )

            # Return top k results
            return fused_results[:k]

        except Exception as e:
            logger.error("Hybrid search failed: %s", e)
            return []

    def _fuse_results(
        self,
        semantic_results: List[CodeSearchResult],
        text_results: List[CodeSearchResult],
        weights: FusionWeights,
        query: str
    ) -> List[HybridSearchResult]:
        """
        Fuse semantic and text search results using RRF and weighted scoring.

        Args:
            semantic_results: Results from semantic search
            text_results: Results from text search
            weights: Fusion weights configuration
            query: Original search query

        Returns:
            List of fused HybridSearchResult objects
        """
        # Create lookup tables for ranking
        semantic_lookup = {result.file_path: (i, result) for i, result in enumerate(semantic_results)}
        text_lookup = {result.file_path: (i, result) for i, result in enumerate(text_results)}

        # Get all unique file paths
        all_paths = set(semantic_lookup.keys()) | set(text_lookup.keys())

        fused_results = []

        for path in all_paths:
            semantic_rank = -1
            text_rank = -1
            semantic_score = 0.0
            text_score = 0.0
            base_result = None

            # Get semantic result if available
            if path in semantic_lookup:
                semantic_rank, semantic_result = semantic_lookup[path]
                semantic_score = semantic_result.similarity_score
                base_result = semantic_result

            # Get text result if available
            if path in text_lookup:
                text_rank, text_result = text_lookup[path]
                text_score = text_result.similarity_score
                if base_result is None:
                    base_result = text_result

            if base_result is None:
                continue

            # Calculate RRF score
            rrf_score = 0.0
            if semantic_rank >= 0:
                rrf_score += 1.0 / (weights.rrf_k + semantic_rank + 1)
            if text_rank >= 0:
                rrf_score += 1.0 / (weights.rrf_k + text_rank + 1)

            # Calculate weighted score
            weighted_score = (
                semantic_score * weights.semantic_weight +
                text_score * weights.text_weight
            )

            # Boost exact matches if enabled
            fusion_score = weighted_score
            if weights.boost_exact_matches and self._is_exact_match(base_result, query):
                fusion_score *= weights.exact_match_boost

            # Combine RRF and weighted scoring
            final_score = 0.7 * fusion_score + 0.3 * rrf_score

            # Determine match type
            match_type = "hybrid"
            if semantic_rank >= 0 and text_rank < 0:
                match_type = "semantic"
            elif text_rank >= 0 and semantic_rank < 0:
                match_type = "text"

            hybrid_result = HybridSearchResult(
                code_result=base_result,
                semantic_score=semantic_score,
                text_score=text_score,
                fusion_score=final_score,
                semantic_rank=semantic_rank + 1 if semantic_rank >= 0 else -1,
                text_rank=text_rank + 1 if text_rank >= 0 else -1,
                fusion_method="rrf_weighted",
                match_type=match_type
            )
            fused_results.append(hybrid_result)

        # Sort by fusion score
        fused_results.sort(key=lambda r: r.fusion_score, reverse=True)

        return fused_results

    def _is_exact_match(self, result: CodeSearchResult, query: str) -> bool:
        """Check if result contains exact match for query terms."""
        query_lower = query.lower().strip('"')
        content_lower = result.snippet.text.lower()
        path_lower = result.file_path.lower()

        # Check for exact phrase match
        if query_lower in content_lower or query_lower in path_lower:
            return True

        # Check for all words present
        query_words = query_lower.split()
        if len(query_words) > 1:
            content_words = content_lower.split()
            path_words = path_lower.split('/')[-1].split('.')  # filename
            all_words = content_words + path_words

            matches = sum(1 for word in query_words if word in all_words)
            return matches == len(query_words)

        return False

    def _expand_query(self, query: str) -> str:
        """
        Expand query with related terms for better semantic search.

        This is a simple implementation - in production this could use
        word embeddings, synonyms, or domain-specific expansions.

        Args:
            query: Original query

        Returns:
            Expanded query
        """
        # Simple expansion mappings for common programming concepts
        expansions = {
            'auth': 'authentication login signin user',
            'db': 'database data storage',
            'api': 'endpoint route http rest',
            'test': 'unittest testing spec',
            'config': 'configuration settings environment',
            'error': 'exception handling try catch',
            'parse': 'parsing parser decode',
            'json': 'javascript object notation data',
            'http': 'request response web api',
            'async': 'asynchronous await promise'
        }

        query_lower = query.lower()
        for term, expansion in expansions.items():
            if term in query_lower and len(query.split()) <= 3:
                return f"{query} {expansion}"

        return query


class HybridSearchFormatter:
    """Formats hybrid search results for display."""

    @staticmethod
    def format_results(
        results: List[HybridSearchResult],
        query: str,
        show_scores: bool = True,
        show_match_type: bool = True,
        repo_path: Optional[str] = None
    ) -> str:
        """
        Format hybrid search results for display.

        Args:
            results: List of hybrid search results
            query: Original search query
            show_scores: Whether to show scoring details
            show_match_type: Whether to show match type info
            repo_path: Repository path for relative path display

        Returns:
            Formatted string representation
        """
        if not results:
            return f"No hybrid search results found for query: '{query}'"

        lines = []
        lines.append(f"🔍 Found {len(results)} hybrid search results for: '{query}'")
        lines.append("=" * 60)

        for i, result in enumerate(results, 1):
            # Format path
            display_path = result.code_result.file_path
            if repo_path and display_path.startswith(repo_path):
                display_path = display_path[len(repo_path):].lstrip('/')

            # Result header with match type indicator
            match_emoji = {
                'semantic': '🧠',
                'text': '📝',
                'hybrid': '🔀'
            }.get(result.match_type, '❓')

            lines.append(f"{match_emoji} [{i}] {display_path}")

            # Show scoring information if requested
            if show_scores:
                lines.append(
                    f"   📊 Fusion: {result.fusion_score:.3f} "
                    f"(semantic: {result.semantic_score:.3f}, "
                    f"text: {result.text_score:.3f})"
                )

            # Show match type and ranking if requested
            if show_match_type:
                rank_info = []
                if result.semantic_rank > 0:
                    rank_info.append(f"semantic #{result.semantic_rank}")
                if result.text_rank > 0:
                    rank_info.append(f"text #{result.text_rank}")

                if rank_info:
                    lines.append(f"   🎯 Ranks: {', '.join(rank_info)} | Method: {result.fusion_method}")

            # Show snippet
            snippet = result.code_result.snippet
            if snippet.start_line == snippet.end_line:
                line_info = f"Line {snippet.start_line}"
            else:
                line_info = f"Lines {snippet.start_line}-{snippet.end_line}"

            lines.append(f"   💻 {line_info}: {snippet.text.strip()[:200]}...")
            lines.append("")

        return "\n".join(lines)


# Convenience functions for easy integration

def search_hybrid(
    db_manager: DatabaseManager,
    embedder: EmbeddingGenerator,
    query: str,
    k: int = 10,
    mode: Union[str, SearchMode] = SearchMode.AUTO,
    **kwargs
) -> List[CodeSearchResult]:
    """
    Convenience function for hybrid search that returns CodeSearchResult objects.

    Args:
        db_manager: Database manager instance
        embedder: Embedding generator instance
        query: Search query
        k: Number of results to return
        mode: Search mode (auto, hybrid, semantic, text)
        **kwargs: Additional arguments passed to HybridSearchEngine

    Returns:
        List of CodeSearchResult objects
    """
    # Convert string mode to enum if needed
    if isinstance(mode, str):
        mode = SearchMode(mode)

    engine = HybridSearchEngine(db_manager, embedder, **kwargs)
    hybrid_results = engine.search(query, k, mode)

    # Extract CodeSearchResult objects
    return [hr.code_result for hr in hybrid_results]


def search_hybrid_with_details(
    db_manager: DatabaseManager,
    embedder: EmbeddingGenerator,
    query: str,
    k: int = 10,
    mode: Union[str, SearchMode] = SearchMode.AUTO,
    **kwargs
) -> List[HybridSearchResult]:
    """
    Convenience function for hybrid search that returns detailed HybridSearchResult objects.

    Args:
        db_manager: Database manager instance
        embedder: Embedding generator instance
        query: Search query
        k: Number of results to return
        mode: Search mode
        **kwargs: Additional arguments passed to HybridSearchEngine

    Returns:
        List of HybridSearchResult objects with detailed scoring
    """
    # Convert string mode to enum if needed
    if isinstance(mode, str):
        mode = SearchMode(mode)

    engine = HybridSearchEngine(db_manager, embedder, **kwargs)
    return engine.search(query, k, mode)
