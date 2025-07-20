#!/usr/bin/env python3
"""
ranking_utils.py: Utility classes for result ranking and analysis.

This module contains utility classes that support the ranking system with
match explanation generation, confidence assessment, and result deduplication.

Classes:
- MatchReasonGenerator: Generates explanations for why results match queries
- ConfidenceScorer: Advanced confidence assessment using multiple signals
- ResultDeduplicator: Handles result clustering and deduplication
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

from search_result_types import CodeSearchResult
from ranking_scorers import ConstructTypeScorer
from ranking_exceptions import MatchReasonGenerationError, ResultDeduplicationError, InvalidSearchResultError, RankingContextError
from ranking_config import get_ranking_config, FileTypeConstants, QueryTypeConstants

logger = logging.getLogger(__name__)


@dataclass
class MatchReason:
    """Represents a reason why a search result matches the query."""
    category: str  # 'name_match', 'content_match', 'context_match', 'structure_match'
    description: str
    confidence: float  # 0.0 to 1.0
    details: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        """Human-readable match reason."""
        return self.description


@dataclass
class RankingContext:
    """Context information used for ranking decisions."""
    query: str
    query_keywords: List[str]
    query_type: Optional[str] = None  # 'function', 'class', 'variable', etc.
    repo_path: Optional[str] = None
    git_info: Optional[Dict[str, Any]] = None
    all_results: Optional[List[CodeSearchResult]] = None


class MatchReasonGenerator:
    """Generates human-readable explanations for why results match queries."""

    @classmethod
    def generate_reasons(cls, result: CodeSearchResult, context: RankingContext) -> List[MatchReason]:
        """
        Generate match reasons for a search result.

        Args:
            result: Search result to analyze
            context: Ranking context with query information

        Returns:
            List of MatchReason objects explaining the match

        Raises:
            MatchReasonGenerationError: If reasons cannot be generated
            InvalidSearchResultError: If result is malformed
            RankingContextError: If context is invalid
        """
        try:
            # Validate inputs
            if not result:
                raise InvalidSearchResultError("Search result cannot be None")

            if not context:
                raise RankingContextError("Ranking context cannot be None")

            if not hasattr(result, 'file_path') or not result.file_path:
                raise InvalidSearchResultError("Search result must have a valid file_path")

            if not hasattr(result, 'snippet') or not result.snippet:
                raise InvalidSearchResultError("Search result must have a valid snippet")

            if not context.query or not isinstance(context.query, str):
                raise RankingContextError("Context must have a valid query string")

            reasons = []
            query_lower = context.query.lower()
            keywords = context.query_keywords or []

            # If no keywords provided, extract them from the query
            if not keywords:
                config = get_ranking_config()
                words = re.findall(r'\b\w+\b', query_lower)
                keywords = [word for word in words
                            if word not in QueryTypeConstants.STOP_WORDS
                            and len(word) > config.match_reasons.min_keyword_length]

            # Analyze filename and path matching
            path_obj = Path(result.file_path)
            filename_lower = path_obj.name.lower()

            # Check for filename matches
            for keyword in keywords:
                if keyword in filename_lower:
                    reasons.append(MatchReason(
                        category='name_match',
                        description=f"Filename contains '{keyword}'",
                        confidence=0.8,
                        details={'matched_keyword': keyword, 'filename': path_obj.name}
                    ))

            # Check for directory structure matches
            path_parts = [part.lower() for part in path_obj.parts]
            for keyword in keywords:
                if any(keyword in part for part in path_parts):
                    reasons.append(MatchReason(
                        category='structure_match',
                        description=f"File path contains '{keyword}' directory",
                        confidence=0.6,
                        details={'matched_keyword': keyword}
                    ))

            # Analyze snippet content
            snippet_text = result.snippet.text.lower()

            # Function/class name matching
            if any(pattern in snippet_text for pattern in ['def ', 'class ', 'function ']):
                for keyword in keywords:
                    if re.search(rf'\b{re.escape(keyword)}\b', snippet_text):
                        reasons.append(MatchReason(
                            category='content_match',
                            description=f"Code contains '{keyword}' identifier",
                            confidence=0.9,
                            details={'matched_keyword': keyword}
                        ))

                # Comment and docstring analysis
                if any(pattern in snippet_text for pattern in QueryTypeConstants.COMMENT_PATTERNS):
                    for keyword in keywords:
                        if keyword in snippet_text:
                            reasons.append(MatchReason(
                                category='content_match',
                                description=f"Documentation mentions '{keyword}'",
                                confidence=0.7,
                                details={'matched_keyword': keyword}
                            ))

            # Import and library analysis
            if 'import' in snippet_text or 'from' in snippet_text:
                for keyword in keywords:
                    if keyword in snippet_text:
                        reasons.append(MatchReason(
                            category='context_match',
                            description=f"Imports library related to '{keyword}'",
                            confidence=0.6,
                            details={'matched_keyword': keyword}
                        ))

            # File type relevance
            extension = path_obj.suffix.lower()
            if extension in FileTypeConstants.SOURCE_CODE_TYPES:
                reasons.append(MatchReason(
                    category='structure_match',
                    description=f"Source code file ({extension})",
                    confidence=0.5,
                    details={'file_type': extension}
                ))

            # Construct type matching
            query_type = ConstructTypeScorer.detect_query_type(context.query)
            if query_type:
                construct_score = ConstructTypeScorer.score_construct_match(result, query_type)
                if construct_score > 0.7:
                    reasons.append(MatchReason(
                        category='structure_match',
                        description=f"Contains {query_type} construct matching query",
                        confidence=construct_score,
                        details={'construct_type': query_type}
                    ))

            # High similarity score
            if result.similarity_score > 0.8:
                reasons.append(MatchReason(
                    category='content_match',
                    description=f"High semantic similarity ({result.similarity_percentage:.1f}%)",
                    confidence=result.similarity_score,
                    details={'similarity_score': result.similarity_score}
                ))

            # Sort reasons by confidence (highest first)
            reasons.sort(key=lambda r: r.confidence, reverse=True)

            # Return top reasons (avoid overwhelming output)
            return reasons[:5]

        except (InvalidSearchResultError, RankingContextError):
            raise  # Re-raise specific errors
        except Exception as e:
            logger.error(f"Error generating match reasons for {getattr(result, 'file_path', 'unknown')}: {e}")
            raise MatchReasonGenerationError(f"Failed to generate match reasons: {e}") from e


class ConfidenceScorer:
    """Advanced confidence scoring using multiple signals."""

    @classmethod
    def calculate_advanced_confidence(
        cls,
        result: CodeSearchResult,
        context: RankingContext,
        match_reasons: List[MatchReason]
    ) -> str:
        """
        Calculate advanced confidence level using multiple factors.

        Args:
            result: Search result to assess
            context: Ranking context
            match_reasons: List of match reasons

        Returns:
            Confidence level: 'high', 'medium', or 'low'
        """
        config = get_ranking_config()
        confidence_weights = config.confidence_weights
        confidence_score = 0.0

        # Base similarity score
        confidence_score += result.similarity_score * confidence_weights.similarity_weight

        # Match reason quality
        if match_reasons:
            avg_reason_confidence = sum(r.confidence for r in match_reasons) / len(match_reasons)
            confidence_score += avg_reason_confidence * confidence_weights.match_reasons_weight

        # Query-result type alignment
        query_type = ConstructTypeScorer.detect_query_type(context.query)
        type_score = ConstructTypeScorer.score_construct_match(result, query_type)
        confidence_score += type_score * confidence_weights.type_alignment_weight

        # Cross-validation with similar results
        if context.all_results:
            # Check if this result is consistent with other high-scoring results
            similar_results = [
                r for r in context.all_results
                if abs(r.similarity_score - result.similarity_score) < config.similarity.cross_validation_similarity
            ]
            consistency_score = min(len(similar_results) / 3, 1.0)  # More consistent = higher confidence
            confidence_score += consistency_score * confidence_weights.cross_validation_weight

        # Convert to confidence levels using configurable thresholds
        confidence_thresholds = config.confidence
        if confidence_score >= confidence_thresholds.high_confidence:
            return 'high'
        elif confidence_score >= confidence_thresholds.medium_confidence:
            return 'medium'
        else:
            return 'low'


class ResultDeduplicator:
    """Handles result clustering and deduplication."""

    @classmethod
    def deduplicate_results(
        cls, results: List[CodeSearchResult], similarity_threshold: Optional[float] = None
    ) -> List[CodeSearchResult]:
        """
        Remove near-duplicate results based on content similarity.

        Args:
            results: List of search results
            similarity_threshold: Threshold for considering results as duplicates

        Returns:
            Deduplicated list of results
        """
        try:
            if len(results) <= 1:
                return results

            # Use configurable threshold if not provided
            if similarity_threshold is None:
                config = get_ranking_config()
                similarity_threshold = config.similarity.deduplication_threshold

            deduplicated = []
            seen_signatures: Set[str] = set()

            for result in results:
                # Create a signature for the result
                signature = cls._create_result_signature(result)

                # Check for near-duplicates
                is_duplicate = False
                for seen_sig in seen_signatures:
                    if cls._signatures_similar(signature, seen_sig, similarity_threshold):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    deduplicated.append(result)
                    seen_signatures.add(signature)

            logger.info(f"Deduplicated {len(results)} results to {len(deduplicated)}")
            return deduplicated

        except Exception as e:
            logger.error(f"Error deduplicating results: {e}")
            raise ResultDeduplicationError(f"Failed to deduplicate results: {e}") from e

    @classmethod
    def _create_result_signature(cls, result: CodeSearchResult) -> str:
        """Create a signature for result comparison."""
        # Use filename + first few lines of content with configurable length
        config = get_ranking_config()
        preview_length = config.deduplication.signature_preview_length

        filename = Path(result.file_path).name
        content_preview = result.snippet.text[:preview_length].strip()
        return f"{filename}:{content_preview}"

    @classmethod
    def _signatures_similar(cls, sig1: str, sig2: str, threshold: float) -> bool:
        """Check if two signatures are similar enough to be considered duplicates."""
        # Simple similarity check - could be enhanced with edit distance
        common_chars = sum(1 for c1, c2 in zip(sig1, sig2) if c1 == c2)
        max_len = max(len(sig1), len(sig2))
        if max_len == 0:
            return True
        similarity = common_chars / max_len
        return similarity >= threshold

    @classmethod
    def ensure_diversity(
            cls,
            results: List[CodeSearchResult],
            max_per_directory: Optional[int] = None) -> List[CodeSearchResult]:
        """
        Ensure result diversity by limiting results per directory.

        Args:
            results: List of search results
            max_per_directory: Maximum results per directory

        Returns:
            Filtered results with better diversity
        """
        # Get config and use configurable max per directory if not provided
        config = get_ranking_config()
        if max_per_directory is None:
            max_per_directory = config.deduplication.max_results_per_directory

        directory_counts: Dict[str, int] = defaultdict(int)
        diverse_results = []

        for result in results:
            directory = str(Path(result.file_path).parent)

            if directory_counts[directory] < max_per_directory:
                diverse_results.append(result)
                directory_counts[directory] += 1
            else:
                # Only add if it has significantly higher score than existing results from this dir
                existing_scores = [
                    r.similarity_score for r in diverse_results
                    if str(Path(r.file_path).parent) == directory
                ]
                if result.similarity_score > max(existing_scores) + config.deduplication.score_improvement_threshold:
                    # Replace lowest scoring result from this directory
                    min_idx = min(
                        (i for i, r in enumerate(diverse_results)
                         if str(Path(r.file_path).parent) == directory),
                        key=lambda i: diverse_results[i].similarity_score
                    )
                    diverse_results[min_idx] = result

        return diverse_results
