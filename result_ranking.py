#!/usr/bin/env python3
"""
result_ranking.py: Advanced result ranking and confidence scoring for Turboprop.

This module implements sophisticated ranking algorithms that go beyond simple embedding similarity
to provide more relevant search results. It includes multi-factor ranking, explainable search results,
and advanced confidence scoring.

Classes:
- RankingWeights: Configuration for ranking factor weights
- ResultRanker: Main ranking engine

Functions:
- rank_search_results: Main entry point for ranking results
- generate_match_explanations: Generate human-readable match explanations
- calculate_advanced_confidence: Multi-factor confidence calculation
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from search_result_types import CodeSearchResult
from ranking_scorers import FileTypeScorer, ConstructTypeScorer, RecencyScorer, FileSizeScorer
from ranking_utils import MatchReasonGenerator, ConfidenceScorer, ResultDeduplicator, RankingContext
from ranking_exceptions import InvalidRankingWeightsError, RankingError, InvalidSearchResultError, RankingContextError

logger = logging.getLogger(__name__)


@dataclass
class RankingWeights:
    """Configuration for ranking factor weights."""
    embedding_similarity: float = 0.4
    file_type_relevance: float = 0.2
    construct_type_matching: float = 0.2
    file_recency: float = 0.1
    file_size_optimization: float = 0.1

    def __post_init__(self):
        """
        Validate that weights are valid and sum to approximately 1.0.

        Raises:
            InvalidRankingWeightsError: If weights are invalid
        """
        # Validate individual weights
        for field_name, value in [
            ('embedding_similarity', self.embedding_similarity),
            ('file_type_relevance', self.file_type_relevance),
            ('construct_type_matching', self.construct_type_matching),
            ('file_recency', self.file_recency),
            ('file_size_optimization', self.file_size_optimization)
        ]:
            if not isinstance(value, (int, float)):
                raise InvalidRankingWeightsError(f"{field_name} must be a number, got {type(value)}")
            if not (0.0 <= value <= 1.0):
                raise InvalidRankingWeightsError(f"{field_name} must be between 0.0 and 1.0, got {value}")

        # Validate total weight sum
        total = (
            self.embedding_similarity
            + self.file_type_relevance
            + self.construct_type_matching
            + self.file_recency
            + self.file_size_optimization
        )

        if not (0.95 <= total <= 1.05):  # Allow small floating point errors
            if abs(total - 1.0) > 0.1:  # Large deviation - error
                raise InvalidRankingWeightsError(f"Ranking weights must sum to approximately 1.0, got {total:.3f}")
            else:  # Small deviation - warning
                logger.warning(f"Ranking weights sum to {total:.3f}, not 1.0")


class ResultRanker:
    """Main ranking engine that coordinates all ranking components."""

    def __init__(self, weights: Optional[RankingWeights] = None):
        """
        Initialize the result ranker.

        Args:
            weights: Optional custom ranking weights
        """
        self.weights = weights or RankingWeights()

    def rank_results(
        self,
        results: List[CodeSearchResult],
        context: RankingContext
    ) -> List[CodeSearchResult]:
        """
        Rank search results using multi-factor algorithm.

        Args:
            results: List of search results to rank
            context: Ranking context with query information

        Returns:
            Ranked list of results with enhanced metadata

        Raises:
            RankingContextError: If context is invalid
            InvalidSearchResultError: If results are malformed
            RankingError: If ranking fails
        """
        try:
            # Validate inputs
            if results is None:
                raise InvalidSearchResultError("Results list cannot be None")

            if not isinstance(results, list):
                raise InvalidSearchResultError(f"Results must be a list, got {type(results)}")

            if not results:
                return results

            if not context:
                raise RankingContextError("Ranking context cannot be None")

            if not context.query or not isinstance(context.query, str):
                raise RankingContextError("Context must have a valid query string")

            logger.info(f"Ranking {len(results)} results with multi-factor algorithm")

            # Add context information to make it available for scoring
            enhanced_context = RankingContext(
                query=context.query,
                query_keywords=context.query_keywords or self._extract_keywords(context.query),
                query_type=context.query_type or ConstructTypeScorer.detect_query_type(context.query),
                repo_path=context.repo_path,
                git_info=context.git_info,
                all_results=results
            )

            # Calculate composite scores and enhance results
            enhanced_results = []
            for result in results:
                enhanced_result = self._enhance_result(result, enhanced_context)
                enhanced_results.append(enhanced_result)

            # Apply deduplication and diversity filtering
            deduplicated = ResultDeduplicator.deduplicate_results(enhanced_results)
            diverse_results = ResultDeduplicator.ensure_diversity(deduplicated)

            # Sort by composite score
            diverse_results.sort(key=lambda r: float(r.file_metadata.get(
                'composite_score', 0) if r.file_metadata else 0), reverse=True)

            logger.info(f"Ranked results: {len(results)} → {len(diverse_results)} after deduplication/diversity")
            return diverse_results

        except (RankingContextError, InvalidSearchResultError):
            raise  # Re-raise specific errors
        except Exception as e:
            logger.error(f"Error ranking {len(results) if results else 0} results: {e}")
            raise RankingError(f"Failed to rank search results: {e}") from e

    def _enhance_result(self, result: CodeSearchResult, context: RankingContext) -> CodeSearchResult:
        """
        Enhance a single result with ranking metadata.

        Args:
            result: Search result to enhance
            context: Ranking context

        Returns:
            Enhanced result with ranking metadata
        """
        # Calculate individual ranking factors
        embedding_score = result.similarity_score
        file_type_score = FileTypeScorer.score_file_type(result.file_path, context.query)
        construct_score = ConstructTypeScorer.score_construct_match(result, context.query_type)
        recency_score = RecencyScorer.score_recency(result.file_path, context.git_info)
        size_score = FileSizeScorer.score_file_size(result)

        # Calculate composite score
        composite_score = (
            embedding_score * self.weights.embedding_similarity
            + file_type_score * self.weights.file_type_relevance
            + construct_score * self.weights.construct_type_matching
            + recency_score * self.weights.file_recency
            + size_score * self.weights.file_size_optimization
        )

        # Generate match reasons
        match_reasons = MatchReasonGenerator.generate_reasons(result, context)

        # Calculate advanced confidence
        advanced_confidence = ConfidenceScorer.calculate_advanced_confidence(
            result, context, match_reasons
        )

        # Enhance result metadata
        if not result.file_metadata:
            result.file_metadata = {}

        result.file_metadata.update({
            'composite_score': composite_score,
            'ranking_factors': {
                'embedding_similarity': embedding_score,
                'file_type_relevance': file_type_score,
                'construct_type_matching': construct_score,
                'file_recency': recency_score,
                'file_size_optimization': size_score
            },
            'match_reasons': [
                {
                    'category': reason.category,
                    'description': reason.description,
                    'confidence': reason.confidence,
                    'details': reason.details
                }
                for reason in match_reasons
            ],
            'advanced_confidence': advanced_confidence
        })

        # Update the result's confidence level with advanced scoring
        result.confidence_level = advanced_confidence

        return result

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query."""
        # Simple keyword extraction - could be enhanced with NLP
        words = re.findall(r'\b\w+\b', query.lower())
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords


# Main API functions

def rank_search_results(
    results: List[CodeSearchResult],
    query: str,
    repo_path: Optional[str] = None,
    git_info: Optional[Dict] = None,
    ranking_weights: Optional[RankingWeights] = None
) -> List[CodeSearchResult]:
    """
    Main entry point for ranking search results.

    Args:
        results: List of search results to rank
        query: Original search query
        repo_path: Optional repository path
        git_info: Optional Git information
        ranking_weights: Optional custom ranking weights

    Returns:
        Ranked and enhanced list of search results

    Raises:
        RankingError: If ranking fails
        InvalidRankingWeightsError: If custom weights are invalid
        RankingContextError: If query is invalid
    """
    try:
        if not results:
            return results

        if not query or not isinstance(query, str):
            raise RankingContextError("Query must be a non-empty string")

        context = RankingContext(
            query=query,
            query_keywords=[],  # Will be extracted by ranker
            repo_path=repo_path,
            git_info=git_info
        )

        ranker = ResultRanker(ranking_weights)
        return ranker.rank_results(results, context)

    except (RankingError, InvalidRankingWeightsError, RankingContextError):
        raise  # Re-raise specific errors
    except Exception as e:
        logger.error(f"Unexpected error in rank_search_results: {e}")
        raise RankingError(f"Failed to rank search results: {e}") from e


def generate_match_explanations(
    results: List[CodeSearchResult],
    query: str
) -> Dict[str, List[str]]:
    """
    Generate human-readable explanations for search matches.

    Args:
        results: List of search results
        query: Original search query

    Returns:
        Dictionary mapping file paths to match explanations
    """
    context = RankingContext(query=query, query_keywords=[])
    explanations = {}

    for result in results:
        if result.file_metadata and 'match_reasons' in result.file_metadata:
            reasons = result.file_metadata['match_reasons']
            explanations[result.file_path] = [reason['description'] for reason in reasons]
        else:
            # Generate reasons on-the-fly if not already present
            reasons = MatchReasonGenerator.generate_reasons(result, context)
            explanations[result.file_path] = [reason.description for reason in reasons]

    return explanations


def calculate_advanced_confidence(
    result: CodeSearchResult,
    query: str,
    all_results: Optional[List[CodeSearchResult]] = None
) -> str:
    """
    Calculate advanced confidence for a single result.

    Args:
        result: Search result to assess
        query: Original search query
        all_results: Optional list of all results for cross-validation

    Returns:
        Confidence level: 'high', 'medium', or 'low'
    """
    context = RankingContext(
        query=query,
        query_keywords=[],
        all_results=all_results
    )

    # Generate match reasons for confidence calculation
    reasons = MatchReasonGenerator.generate_reasons(result, context)

    return ConfidenceScorer.calculate_advanced_confidence(result, context, reasons)
