#!/usr/bin/env python3
"""
ranking_config.py: Configuration constants and thresholds for the ranking system.

This module centralizes all configurable values, thresholds, and magic numbers
used throughout the ranking system to make them easily adjustable.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional


@dataclass
class ConfidenceThresholds:
    """Thresholds for confidence level determination."""
    high_confidence: float = 0.8
    medium_confidence: float = 0.6
    low_confidence: float = 0.0


@dataclass
class SimilarityThresholds:
    """Thresholds for similarity-based decisions."""
    high_similarity: float = 0.8
    construct_match_threshold: float = 0.7
    deduplication_threshold: float = 0.9
    cross_validation_similarity: float = 0.1


@dataclass
class RecencyThresholds:
    """Thresholds for file recency scoring (in days)."""
    very_recent: int = 7      # Files modified within 7 days get score 1.0
    recent: int = 30          # Files modified within 30 days get score 0.9
    moderately_recent: int = 90   # Files modified within 90 days get score 0.7
    somewhat_old: int = 180   # Files modified within 180 days get score 0.5
    old: int = 365           # Files modified within 365 days get score 0.3
    # Files older than 365 days get score 0.1


@dataclass
class FileSizeThresholds:
    """Thresholds for file size scoring (in bytes)."""
    min_useful_size: int = 100        # Files smaller than this get low scores
    optimal_min: int = 1000           # Start of optimal size range
    optimal_max: int = 10000          # End of optimal size range
    medium_max: int = 50000           # Up to 50KB gets score 0.8
    large_max: int = 100000           # Up to 100KB gets score 0.6
    very_large_max: int = 500000      # Up to 500KB gets score 0.4
    # Files larger than 500KB get score 0.2


@dataclass
class MatchReasonLimits:
    """Limits for match reason generation."""
    max_reasons_per_result: int = 5
    min_keyword_length: int = 2


@dataclass
class DeduplicationConfig:
    """Configuration for result deduplication."""
    signature_preview_length: int = 100
    max_results_per_directory: int = 3
    score_improvement_threshold: float = 0.1  # Minimum improvement needed to replace result


@dataclass
class ConfidenceWeights:
    """Weights for confidence score calculation."""
    similarity_weight: float = 0.4
    match_reasons_weight: float = 0.3
    type_alignment_weight: float = 0.2
    cross_validation_weight: float = 0.1


@dataclass
class RankingConstants:
    """Main configuration class containing all ranking constants."""
    confidence: ConfidenceThresholds = field(default_factory=ConfidenceThresholds)
    similarity: SimilarityThresholds = field(default_factory=SimilarityThresholds)
    recency: RecencyThresholds = field(default_factory=RecencyThresholds)
    file_size: FileSizeThresholds = field(default_factory=FileSizeThresholds)
    match_reasons: MatchReasonLimits = field(default_factory=MatchReasonLimits)
    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    confidence_weights: ConfidenceWeights = field(default_factory=ConfidenceWeights)



# File type constants
class FileTypeConstants:
    """Constants for file type classification."""

    SOURCE_CODE_TYPES: Set[str] = {
        '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp',
        '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala'
    }

    TEST_FILE_PATTERNS: List[str] = ['test_', '_test', '.test.', 'spec_', '_spec', '.spec.']

    DOC_FILE_TYPES: Set[str] = {'.md', '.rst', '.txt', '.doc', '.html'}

    CONFIG_FILE_TYPES: Set[str] = {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf'}

    # Keywords for detecting query intent
    DOC_QUERY_KEYWORDS: List[str] = ['readme', 'documentation', 'guide', 'tutorial', 'example']

    CONFIG_QUERY_KEYWORDS: List[str] = ['config', 'settings', 'environment', 'setup', 'deploy']


# Query type constants
class QueryTypeConstants:
    """Constants for query type detection."""

    TYPE_KEYWORDS: Dict[str, List[str]] = {
        'function': ['function', 'func', 'def ', 'method', 'procedure'],
        'class': ['class', 'type', 'struct', 'interface', 'object'],
        'variable': ['variable', 'var', 'constant', 'const', 'field', 'property'],
        'import': ['import', 'include', 'require', 'using', 'from'],
        'exception': ['error', 'exception', 'throw', 'catch', 'try'],
        'module': ['module', 'package', 'namespace', 'library']
    }

    FUNCTION_INDICATORS: List[str] = ['def ', 'function ', 'func ']
    CLASS_INDICATORS: List[str] = ['class ', 'struct ', 'interface ']
    VARIABLE_INDICATORS: List[str] = ['var ', 'const ', '= ']
    IMPORT_INDICATORS: List[str] = ['import ', 'from ', 'require(']

    # Comment patterns for content analysis
    COMMENT_PATTERNS: List[str] = ['"""', "'''", '//', '#']

    # Stop words for keyword extraction
    STOP_WORDS: Set[str] = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into',
        'through', 'during', 'before', 'after', 'above', 'below',
        'under', 'over', 'between'
    }


# Default configuration instance
DEFAULT_CONFIG = RankingConstants()


def get_ranking_config() -> RankingConstants:
    """Get the current ranking configuration."""
    return DEFAULT_CONFIG


def update_ranking_config(config: RankingConstants) -> None:
    """Update the global ranking configuration."""
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = config
