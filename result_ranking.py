#!/usr/bin/env python3
"""
result_ranking.py: Advanced result ranking and confidence scoring for Turboprop.

This module implements sophisticated ranking algorithms that go beyond simple embedding similarity
to provide more relevant search results. It includes multi-factor ranking, explainable search results,
and advanced confidence scoring.

Classes:
- RankingWeights: Configuration for ranking factor weights
- MatchReason: Represents why a result matches the query
- RankingContext: Context information for ranking decisions
- ResultRanker: Main ranking engine
- ConfidenceScorer: Advanced confidence assessment
- MatchReasonGenerator: Generates explanations for search matches
- ResultDeduplicator: Handles result clustering and deduplication

Functions:
- rank_search_results: Main entry point for ranking results
- generate_match_explanations: Generate human-readable match explanations
- calculate_advanced_confidence: Multi-factor confidence calculation
"""

import logging
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from collections import defaultdict, Counter

from search_result_types import CodeSearchResult, CodeSnippet
from construct_search import ConstructSearchResult
from config import config

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
        """Validate that weights sum to 1.0."""
        total = (
            self.embedding_similarity + 
            self.file_type_relevance + 
            self.construct_type_matching + 
            self.file_recency + 
            self.file_size_optimization
        )
        if not (0.95 <= total <= 1.05):  # Allow small floating point errors
            logger.warning(f"Ranking weights sum to {total:.3f}, not 1.0")


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


class FileTypeScorer:
    """Scores results based on file type relevance."""
    
    # Source code files get highest relevance for code queries
    SOURCE_CODE_TYPES = {
        '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp',
        '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala'
    }
    
    # Test files are relevant but lower priority
    TEST_FILE_PATTERNS = ['test_', '_test', '.test.', 'spec_', '_spec', '.spec.']
    
    # Documentation files have medium relevance
    DOC_FILE_TYPES = {'.md', '.rst', '.txt', '.doc', '.html'}
    
    # Configuration files have lower relevance for most code queries
    CONFIG_FILE_TYPES = {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf'}

    @classmethod
    def score_file_type(cls, file_path: str, query: str) -> float:
        """
        Score file type relevance (0.0 to 1.0).
        
        Args:
            file_path: Path to the file
            query: Search query
            
        Returns:
            Relevance score based on file type
        """
        path_obj = Path(file_path)
        extension = path_obj.suffix.lower()
        filename = path_obj.name.lower()
        
        # Check if this is a test file
        is_test_file = any(pattern in filename for pattern in cls.TEST_FILE_PATTERNS)
        
        # Base score by file type
        if extension in cls.SOURCE_CODE_TYPES:
            base_score = 1.0
            # Slightly reduce score for test files unless query is about testing
            if is_test_file and 'test' not in query.lower():
                base_score = 0.8
        elif extension in cls.DOC_FILE_TYPES:
            # Higher score for docs if query seems documentation-related
            doc_keywords = ['readme', 'documentation', 'guide', 'tutorial', 'example']
            if any(keyword in query.lower() for keyword in doc_keywords):
                base_score = 0.9
            else:
                base_score = 0.6
        elif extension in cls.CONFIG_FILE_TYPES:
            # Higher score for config files if query is configuration-related
            config_keywords = ['config', 'settings', 'environment', 'setup', 'deploy']
            if any(keyword in query.lower() for keyword in config_keywords):
                base_score = 0.8
            else:
                base_score = 0.4
        else:
            base_score = 0.5  # Unknown file type gets neutral score
        
        return base_score


class ConstructTypeScorer:
    """Scores results based on construct type matching."""
    
    QUERY_TYPE_KEYWORDS = {
        'function': ['function', 'func', 'def ', 'method', 'procedure'],  # Added space after 'def' to avoid matching "definition"
        'class': ['class', 'type', 'struct', 'interface', 'object'],
        'variable': ['variable', 'var', 'constant', 'const', 'field', 'property'],
        'import': ['import', 'include', 'require', 'using', 'from'],
        'exception': ['error', 'exception', 'throw', 'catch', 'try'],
        'module': ['module', 'package', 'namespace', 'library']
    }
    
    @classmethod
    def detect_query_type(cls, query: str) -> Optional[str]:
        """
        Detect the likely construct type from query keywords.
        
        Args:
            query: Search query string
            
        Returns:
            Detected construct type or None
        """
        query_lower = query.lower()
        for construct_type, keywords in cls.QUERY_TYPE_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                return construct_type
        return None
    
    @classmethod
    def score_construct_match(cls, result: CodeSearchResult, query_type: Optional[str]) -> float:
        """
        Score construct type matching (0.0 to 1.0).
        
        Args:
            result: Search result to score
            query_type: Detected query type
            
        Returns:
            Score based on construct type matching
        """
        if not query_type:
            return 0.5  # Neutral score when query type is unknown
        
        # Check if result has construct information
        if hasattr(result, 'file_metadata') and result.file_metadata:
            construct_context = result.file_metadata.get('construct_context')
            if construct_context:
                construct_types = construct_context.get('construct_types', [])
                if query_type in construct_types:
                    return 1.0
                # Partial match for related types
                if query_type == 'function' and 'method' in construct_types:
                    return 0.8
                if query_type == 'class' and any(t in construct_types for t in ['interface', 'struct']):
                    return 0.8
        
        # Fallback: analyze snippet content for construct type indicators
        snippet_text = result.snippet.text.lower()
        
        if query_type == 'function':
            if any(indicator in snippet_text for indicator in ['def ', 'function ', 'func ']):
                return 0.9
        elif query_type == 'class':
            if any(indicator in snippet_text for indicator in ['class ', 'struct ', 'interface ']):
                return 0.9
        elif query_type == 'variable':
            if any(indicator in snippet_text for indicator in ['var ', 'const ', '= ']):
                return 0.7
        elif query_type == 'import':
            if any(indicator in snippet_text for indicator in ['import ', 'from ', 'require(']):
                return 0.9
        
        return 0.3  # Low score for mismatched types


class RecencyScorer:
    """Scores results based on file recency using Git information."""
    
    @classmethod
    def get_file_modification_time(cls, file_path: str, git_info: Optional[Dict] = None) -> Optional[datetime]:
        """
        Get file modification time, preferring Git info over filesystem.
        
        Args:
            file_path: Path to the file
            git_info: Optional Git information
            
        Returns:
            Modification datetime or None
        """
        try:
            # Try to get Git commit time first (more accurate for version-controlled files)
            if git_info and 'file_modifications' in git_info:
                git_time = git_info['file_modifications'].get(file_path)
                if git_time:
                    return datetime.fromisoformat(git_time)
            
            # Fallback to filesystem modification time
            stat = os.stat(file_path)
            return datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        except (OSError, ValueError) as e:
            logger.debug(f"Could not get modification time for {file_path}: {e}")
            return None
    
    @classmethod
    def score_recency(cls, file_path: str, git_info: Optional[Dict] = None) -> float:
        """
        Score file recency (0.0 to 1.0).
        
        Args:
            file_path: Path to the file
            git_info: Optional Git information
            
        Returns:
            Recency score (higher for more recently modified files)
        """
        mod_time = cls.get_file_modification_time(file_path, git_info)
        if not mod_time:
            return 0.5  # Neutral score if we can't determine modification time
        
        now = datetime.now(tz=timezone.utc)
        age_days = (now - mod_time).total_seconds() / (24 * 3600)
        
        # Score based on age with exponential decay
        # Recent files (< 7 days) get high scores
        # Files older than 365 days get low scores
        if age_days <= 7:
            return 1.0
        elif age_days <= 30:
            return 0.9
        elif age_days <= 90:
            return 0.7
        elif age_days <= 180:
            return 0.5
        elif age_days <= 365:
            return 0.3
        else:
            return 0.1


class FileSizeScorer:
    """Scores results based on file size optimization."""
    
    OPTIMAL_SIZE_RANGE = (1000, 10000)  # 1KB to 10KB is often optimal for code files
    
    @classmethod
    def score_file_size(cls, result: CodeSearchResult) -> float:
        """
        Score file size optimization (0.0 to 1.0).
        
        Args:
            result: Search result to score
            
        Returns:
            Size optimization score
        """
        if not result.file_metadata or 'size' not in result.file_metadata:
            return 0.5  # Neutral score if size is unknown
        
        size_bytes = result.file_metadata['size']
        
        # Very small files might be incomplete or not useful
        if size_bytes < 100:
            return 0.2
        
        # Optimal size range gets highest score
        if cls.OPTIMAL_SIZE_RANGE[0] <= size_bytes <= cls.OPTIMAL_SIZE_RANGE[1]:
            return 1.0
        
        # Score based on distance from optimal range
        if size_bytes < cls.OPTIMAL_SIZE_RANGE[0]:
            # Small files
            ratio = size_bytes / cls.OPTIMAL_SIZE_RANGE[0]
            return 0.4 + 0.6 * ratio
        else:
            # Large files - diminishing returns with size
            if size_bytes <= 50000:  # Up to 50KB
                return 0.8
            elif size_bytes <= 100000:  # Up to 100KB
                return 0.6
            elif size_bytes <= 500000:  # Up to 500KB
                return 0.4
            else:
                return 0.2  # Very large files


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
        """
        reasons = []
        query_lower = context.query.lower()
        keywords = context.query_keywords or []
        
        # If no keywords provided, extract them from the query
        if not keywords:
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            words = re.findall(r'\b\w+\b', query_lower)
            keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
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
        if any(pattern in snippet_text for pattern in ['"""', "'''", '//', '#']):
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
        if extension in FileTypeScorer.SOURCE_CODE_TYPES:
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
        confidence_score = 0.0
        
        # Base similarity score (weight: 40%)
        confidence_score += result.similarity_score * 0.4
        
        # Match reason quality (weight: 30%)
        if match_reasons:
            avg_reason_confidence = sum(r.confidence for r in match_reasons) / len(match_reasons)
            confidence_score += avg_reason_confidence * 0.3
        
        # Query-result type alignment (weight: 20%)
        query_type = ConstructTypeScorer.detect_query_type(context.query)
        type_score = ConstructTypeScorer.score_construct_match(result, query_type)
        confidence_score += type_score * 0.2
        
        # Cross-validation with similar results (weight: 10%)
        if context.all_results:
            # Check if this result is consistent with other high-scoring results
            similar_results = [r for r in context.all_results if abs(r.similarity_score - result.similarity_score) < 0.1]
            consistency_score = min(len(similar_results) / 3, 1.0)  # More consistent = higher confidence
            confidence_score += consistency_score * 0.1
        
        # Convert to confidence levels
        if confidence_score >= 0.8:
            return 'high'
        elif confidence_score >= 0.6:
            return 'medium'
        else:
            return 'low'


class ResultDeduplicator:
    """Handles result clustering and deduplication."""
    
    @classmethod
    def deduplicate_results(cls, results: List[CodeSearchResult], similarity_threshold: float = 0.9) -> List[CodeSearchResult]:
        """
        Remove near-duplicate results based on content similarity.
        
        Args:
            results: List of search results
            similarity_threshold: Threshold for considering results as duplicates
            
        Returns:
            Deduplicated list of results
        """
        if len(results) <= 1:
            return results
        
        deduplicated = []
        seen_signatures = set()
        
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
    
    @classmethod
    def _create_result_signature(cls, result: CodeSearchResult) -> str:
        """Create a signature for result comparison."""
        # Use filename + first few lines of content
        filename = Path(result.file_path).name
        content_preview = result.snippet.text[:100].strip()
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
    def ensure_diversity(cls, results: List[CodeSearchResult], max_per_directory: int = 3) -> List[CodeSearchResult]:
        """
        Ensure result diversity by limiting results per directory.
        
        Args:
            results: List of search results
            max_per_directory: Maximum results per directory
            
        Returns:
            Filtered results with better diversity
        """
        directory_counts = defaultdict(int)
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
                if result.similarity_score > max(existing_scores) + 0.1:
                    # Replace lowest scoring result from this directory
                    min_idx = min(
                        (i for i, r in enumerate(diverse_results) 
                         if str(Path(r.file_path).parent) == directory),
                        key=lambda i: diverse_results[i].similarity_score
                    )
                    diverse_results[min_idx] = result
        
        return diverse_results


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
        """
        if not results:
            return results
        
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
        diverse_results.sort(key=lambda r: r.file_metadata.get('composite_score', 0), reverse=True)
        
        logger.info(f"Ranked results: {len(results)} → {len(diverse_results)} after deduplication/diversity")
        return diverse_results
    
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
            embedding_score * self.weights.embedding_similarity +
            file_type_score * self.weights.file_type_relevance +
            construct_score * self.weights.construct_type_matching +
            recency_score * self.weights.file_recency +
            size_score * self.weights.file_size_optimization
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
    """
    if not results:
        return results
    
    context = RankingContext(
        query=query,
        query_keywords=None,  # Will be extracted by ranker
        repo_path=repo_path,
        git_info=git_info
    )
    
    ranker = ResultRanker(ranking_weights)
    return ranker.rank_results(results, context)


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
    context = RankingContext(query=query, query_keywords=None)
    explanations = {}
    
    for result in results:
        if 'match_reasons' in result.file_metadata:
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
        query_keywords=None,
        all_results=all_results
    )
    
    # Generate match reasons for confidence calculation
    reasons = MatchReasonGenerator.generate_reasons(result, context)
    
    return ConfidenceScorer.calculate_advanced_confidence(result, context, reasons)